import copy
import math

import numpy as np
import matplotlib.pyplot as plt
from enviorment.make_env import CPTEnv
from functools import partial
from algorithm.utils.logger import Logger
import itertools
from scipy.special import softmax
plt.ion()
def main():
    config = {
        'iWorld': 1,
        'lr': 0.05,
        'batch_size': 32,
        'gamma': 0.99,
        'buffer_limit': 50000,
        'log_interval': 5,
        'max_episodes': 1000,
        'max_epsilon': 0.9,
        'min_epsilon': 0.1,
        'test_episodes': 5,
        'warm_up_steps': 10,
        'no_penalty_steps': 1000,
        'update_iter': 25,
        'monitor': False,
    }
    Qlearning(**config)


# def Qlearning(iWorld =1, max_episodes=20000, discount_factor = 0.9, alpha = 0.005,n_warmup = 1000):
def Qlearning(iWorld, lr, gamma, batch_size, buffer_limit, log_interval, max_episodes,
         max_epsilon, min_epsilon, test_episodes, warm_up_steps, no_penalty_steps,update_iter,monitor):
    TYPE = 'Baseline'
    ALG = 'QJoint'
    Logger.file_name = f'Fig_{ALG}_{TYPE}'
    Logger.update_save_directory(f'results/{ALG}/{TYPE}_W{iWorld}/')
    Logger.update_plt_title(f'W{iWorld} {ALG} Training Results')
    Logger.make_directory()

    alpha = lr
    discount_factor = gamma

    env = CPTEnv(iWorld)
    env.disable_prey =True

    Q = Qfun(env.observation_space, env.action_space)


    fix_state = lambda sk: [list(np.array(s) - 1) for s in sk]
    warmup_epsilons = max_epsilon * np.ones(warm_up_steps)
    improve_epsilons = np.linspace(max_epsilon, min_epsilon, max_episodes - warm_up_steps)
    epi_epsilons = np.hstack([warmup_epsilons, improve_epsilons])

    # max_rationality = 10
    # min_rationality = 1
    # warmup_rationality = min_rationality * np.ones(warm_up_steps)
    # improve_rationality = np.linspace(min_rationality, max_rationality, max_episodes - warm_up_steps)
    # epi_rationality = np.hstack([warmup_rationality, improve_rationality])

    verbose = True
    _ns = Q.ns
    _na = Q.num_actions


    for ith_episode in range(max_episodes):
        # Update exploration parameter
        eps = epi_epsilons[ith_episode]
        # lamb = epi_rationality[ith_episode]
        # Q.set_rationalities(lamb,lamb)
        Q.set_rationalities(100, 100)
        Q.epsilon = eps

        # trackers
        score = np.zeros(2)

        # reset environment
        done = False
        env.seed(ith_episode)
        env.reset()
        if ith_episode > no_penalty_steps: env.enable_penalty = True
        else: env.enable_penalty = False
        state = env.reset()
        state = fix_state(state)

        eligibility = {
            'trace': np.zeros([20,2],dtype=object),
            'lambda': 0.95
        }
        while not done:
            # decide and take action
            actionR,Qsa0_R,actionH,Qsa0_H = Q.sample_actions(state[0])
            joint_action = [actionR,actionH]
            next_state, reward, done, info = env.step(joint_action)
            next_state = fix_state(next_state)
            SAi = list(state[0]) + joint_action
            SAi_flipped = [state[0][i] for i in [1,0,2]] + [joint_action[i] for i in [1,0]]


            eligibility['trace'][env.step_count-1, 0] = 1
            eligibility['trace'][env.step_count-1, 1] = copy.copy(SAi)


            # Get best next action quality
            next_actionR,Qmax1_R,next_actionH,Qmax1_H = Q.sample_best_actions(next_state[0])

            #print(f'R[{state}=>{next_state[0:2]}]',end='')

            for itrace in range(env.step_count):
                et = eligibility['trace'][itrace, 0]
                SAi = eligibility['trace'][itrace, 1]

                # Robot Q update ###################
                k = 0
                TD_err = (reward[k] + gamma * Qmax1_R - Qsa0_R)
                Q.tbl[SAi + [k]] = Q.tbl[SAi + [k]] + (alpha) * TD_err * et

                # Human Q update ###################
                k = 1
                TD_err = (reward[k] + gamma * Qmax1_H - Qsa0_H)
                Q.tbl[SAi + [k]] =  Q.tbl[SAi + [k]] + (alpha) * TD_err * et
                # k = 1
                # TD_err = (reward[k] + gamma * Qmax1_H - Qsa0_H)
                # Q.tbl[SAi_flipped + [int(not(k))]] = Q.tbl[SAi + [int(not(k))]] + (alpha) * TD_err * et

            Q.tbl[SAi, 1] = Q.tbl[SAi, 0]
            Q.tbl[SAi_flipped,1] = Q.tbl[SAi_flipped,0]

            # Close iteration
            state = np.copy(next_state)
            score += np.array(reward)

            eligibility['trace'][:, 0] *=  eligibility['lambda']*gamma


        Logger.log_episode(np.mean(score), env.step_count, env.is_caught)

        # REPORT
        if verbose:
            report = f'\r'
            report += f'\t| epI={ith_episode} '
            report += f'\tlam={[round(Q.lambR, 2),round(Q.lambH, 2)]}'
            report += f'\tepsilon={round(Q.epsilon, 2)}'
            report += f'\tP(Success) = {round(Logger.Epi_Psuccess[-1], 2)}% '
            report += f'\tR = {np.round(Logger.Epi_Reward[-1], 2)} '
            report += f'\tQrange = {[round(np.min(Q.tbl), 1), round(np.max(Q.tbl), 1)]}'
            print(f'{report}',end='')

            if env.is_caught:
                print('')

        if ith_episode % log_interval == 0:
            Logger.draw()
    Logger.save()



class Qfun(object):
    def __init__(self,observation_space,action_space):

        """
        ROBOT IS ROW PLAYER
        """
        self.num_actions = action_space[0].n
        self.num_observe = observation_space[0].shape
        self.num_agents = len(observation_space)
        _ns = 5
        _naR = 5
        _naH = 5
        _nk = 2
        self.tbl = np.zeros([_ns,_ns,_ns,_ns,_ns,_ns,_naR,_naH,_nk])

        self.epsilon = 0.8

        self.selection_policy = 'epsilon_greedy'

        self.lambH = 5
        self.lambR = 5
        self.lambH_hat = self.lambH
        self.lambR_hat = self.lambR
        self.ns = _ns

    def set_rationalities(self,lambR,lambH):
        self.lambH = lambR
        self.lambR = lambH
        self.lambH_hat = self.lambH
        self.lambR_hat = self.lambR

    def get_Qs(self,state):
        x0, y0, x1, y1, x2, y2 = state
        qs = self.tbl[x0, y0, x1, y1, x2, y2]  # 5 x 5 bimatrix game
        return qs

    def sample_actions(self, state):
        qs = self.get_Qs(state)
        iR, iH = 0, 1

        #################################################
        # select action according to selection policy
        if np.random.rand() <= self.epsilon:
            actionR = np.random.choice(np.arange(5))
            actionH = np.random.choice(np.arange(5))
        else:
            # Assume sophistocation = 1
            #################################################
            # start with uniform ego pda
            mprobR_hat = (np.ones(qs.shape[iR]) / qs.shape[iR]).reshape([1, -1])
            mprobH_hat = (np.ones(qs.shape[iH]) / qs.shape[iH]).reshape([-1, 1])

            # reduce to partner action value vector
            qH_hat = np.dot(mprobR_hat, qs[:, :, iR])
            qR_hat = np.dot(qs[:, :, iH], mprobH_hat)

            # calc prob of partner transition give value vector
            mprobH_hat = softmax(self.lambR_hat * qR_hat)  # np.exp(lamb*qR_hat)/np.sum(np.exp(qR_hat))
            mprobR_hat = softmax(self.lambH_hat * qH_hat)  # np.exp(lamb*qH_hat)/np.sum(np.exp(qH_hat))

            #################################################
            # Reduce game to expected EGO action-value vector
            Exp_qH = np.dot(mprobR_hat, qs[:, :, iR])
            Exp_qR = np.dot(qs[:, :, iH], mprobH_hat)

            # actionR = np.argmax(Exp_qR)
            # actionH = np.argmax(Exp_qH)
            actionR = np.random.choice(np.arange(5), p=softmax(self.lambR * Exp_qR).flatten())  # np.argmax(Exp_qR)
            actionH = np.random.choice(np.arange(5), p=softmax(self.lambH * Exp_qH).flatten())  # np.argmax(Exp_qH)
            # actionR = np.random.choice(np.arange(5), p=(np.exp(self.lambR*Exp_qR)/np.exp(self.lambR*Exp_qR)).flatten())
            # actionH = np.random.choice(np.arange(5), p=(np.exp(self.lambH*Exp_qH)/np.exp(self.lambH*Exp_qH)).flatten())

        # get quality of that action
        qR = qs[actionR, actionH, iR]
        qH = qs[actionR, actionH, iH]

        return actionR, qR, actionH, qH

    def sample_best_actions(self,state):
        qs = self.get_Qs(state)
        iR,iH = 0,1

        # Assume sophistocation = 1
        #################################################
        # start with uniform ego pda
        mprobR_hat = (np.ones(qs.shape[iR]) / qs.shape[iR]).reshape([1, -1])
        mprobH_hat = (np.ones(qs.shape[iH]) / qs.shape[iH]).reshape([-1, 1])

        # reduce to partner action value vector
        qH_hat = np.dot(mprobR_hat, qs[:, :, iR])
        qR_hat = np.dot(qs[:, :, iH], mprobH_hat)

        # calc prob of partner transition give value vector
        mprobH_hat = softmax(self.lambR_hat * qR_hat)  # np.exp(lamb*qR_hat)/np.sum(np.exp(qR_hat))
        mprobR_hat = softmax(self.lambH_hat * qH_hat)  # np.exp(lamb*qH_hat)/np.sum(np.exp(qH_hat))

        #################################################
        # Reduce game to expected EGO action-value vector
        Exp_qH = np.dot(mprobR_hat, qs[:, :, iR])
        Exp_qR = np.dot(qs[:, :, iH], mprobH_hat)

        #################################################
        # select best action
        actionR = np.argmax(Exp_qR)
        actionH = np.argmax(Exp_qH)

        # get quality of that action
        qbestR = qs[actionR, actionH, iR]
        qbestH = qs[actionR, actionH, iH]

        return actionR,qbestR,actionH,qbestH


if __name__ == "__main__":

    main()
    # Qlearning()