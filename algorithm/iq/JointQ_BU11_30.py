import copy
import math

import numpy as np
import matplotlib.pyplot as plt
from enviorment.make_env import CPTEnv
from functools import partial
from algorithm.utils.logger import Logger
import itertools
plt.ion()
def main():
    # env = CPTEnv()
    # Qk = Qfun(env.observation_space, env.action_space)
    # obs = env.reset()
    # Qk.Qk[0][:, :, :, :, :, :, 0, 0] = 100
    # pd_ego = Qk.sample_pdA(obs)
    # pd_partner = pd_ego[::-1]
    # # print(f'obs = {obs}')
    # # print(f'pd_ego = {pd_ego}')
    # # print(f'pd_partner = {pd_partner}')
    # print(Qk.sample_action(obs, 0.01, pd_partner=pd_partner))
    # pass

    config = {
        'iWorld': 1,
        'lr': 0.1,
        'batch_size': 32,
        'gamma': 0.99,
        'buffer_limit': 50000,
        'log_interval': 100,
        'max_episodes': 20000,
        # 'max_epsilon': 0.8,
        # 'min_epsilon': 0.1,
        'max_epsilon': 0.1,
        'min_epsilon': 0.8,
        'test_episodes': 5,
        'warm_up_steps': 1000,
        'no_penalty_steps': 20000,
        'update_iter': 10,
        'monitor': False,
    }
    Qlearning(**config)


# def Qlearning(iWorld =1, max_episodes=20000, discount_factor = 0.9, alpha = 0.005,n_warmup = 1000):
def Qlearning(iWorld, lr, gamma, batch_size, buffer_limit, log_interval, max_episodes,
         max_epsilon, min_epsilon, test_episodes, warm_up_steps, no_penalty_steps,update_iter,monitor):

    TYPE = 'Baseline'
    Logger.file_name = f'Fig_JointQ_{TYPE}'
    Logger.fname_notes = f'W{iWorld}'
    Logger.update_fname()
    Logger.make_directory()

    alpha = lr
    discount_factor = gamma

    env = CPTEnv(iWorld)
    env.disable_prey =True

    Qk = Qfun(env.observation_space, env.action_space)
    sample_best = 0


    warmup_epsilons = max_epsilon * np.ones(warm_up_steps)
    improve_epsilons = np.linspace(max_epsilon, min_epsilon, max_episodes - warm_up_steps)
    epi_epsilons = np.hstack([warmup_epsilons, improve_epsilons])



    min_rationality,max_rationality = 0.01,5

    #epi_eps = np.hstack([0.9 * np.ones(n_warmup), np.linspace(0.9, epsilon, num_episodes - n_warmup)])
    plt_ever_ith_episode = 200
    verbose = True
    # Initialize stat tracker

    stats = {}
    stats['epi_reward'] = np.empty([0,2])
    stats['epi_length'] = []
    stats['epi_caught'] = []
    stats['epi_psucc'] = []
    stats['filt_window'] = 99

    #
    #
    # fig, axs = plt.subplots(3, 1)
    # stats['line_reward'], = axs[0].plot(np.zeros(2), lw=0.5)
    # stats['line_mreward'], = axs[0].plot(np.zeros(2))
    # stats['line_len'], = axs[1].plot(np.zeros(2), lw=0.5)
    # stats['line_mlen'], = axs[1].plot(np.zeros(2))
    # stats['line_psucc'], = axs[2].plot(np.zeros(2), lw=0.5)
    # stats['line_mpsucc'], = axs[2].plot(np.zeros(2))
    # axs[0].set_title('JointQ Training Results')
    # axs[0].set_ylabel('Epi Reward')
    # axs[1].set_ylabel('Epi Length')
    # axs[1].set_xlabel('Episode')
    # axs[2].set_ylabel('P(Catch)')
    # axs[2].set_xlabel('Episode')

    rationality = None
    eps = None
    for ith_episode in range(max_episodes):
        done = False
        env.seed(ith_episode)
        env.reset()
        # env.render()
        eps = epi_epsilons[ith_episode]

        _ns = Qk.ns
        _na = Qk.num_actions

        eligibility = {
            'lambda': 0.95,
            'trace': np.zeros([_ns,_ns,_ns,_ns,_ns,_ns,_na,_na])
                       }

        # rationality = max(min_rationality, (max_rationality - min_rationality) * (ith_episode / (0.4 * (num_episodes-n_warmup))))
        # if ith_episode<n_warmup: rationality= min_rationality
        # Qk.update_policy([rationality,rationality])

        stats['epi_reward'] = np.vstack( [stats['epi_reward'],np.zeros([1,2])])
        stats['epi_length'].append(0)
        stats['epi_caught'].append(0)
        stats['epi_psucc'].append(0)

        if ith_episode > no_penalty_steps: env.enable_penalty = True
        else: env.enable_penalty = False

        ski = env.reset()
        while not done:
            aki = Qk.sample_action(ski,epsilon=eps)
            skj, rkj, done, info = env.step(aki)
            # if ith_episode<=no_pen_episodes:
            #     rkj = [max(0,r) for r in rkj]
            #


            Qnew_k = []
            Qk_siai = Qk.at(ski,action=aki)
            Qk_sj = Qk.at(skj)
            pdkj_partner = Qk.sample_pdA(skj,reverse=True)
            akj = Qk.sample_action(ski, sample_best,pd_partner=pdkj_partner)

            eligibility['trace'][ski[0]+aki] += 1

            ek = eligibility['trace']

            for agent_i,reward in enumerate(rkj):
                Q_si = Qk_siai[agent_i]
                Qmax_sj = Qk_sj[agent_i][int(akj[0]),int(akj[1])]
                TD_err = (reward + discount_factor * Qmax_sj)  - Q_si

                # for all states

                Qk.Qk[agent_i] = Qk.Qk[agent_i] + (lr)*TD_err*(ek)


                # TD_target = reward + discount_factor * Qmax_sj #RQ[sj][ajR]
                # TD_err = TD_target - Q_si #RQ[si][aiR]
                # Qnew_k.append(Q_si + alpha * TD_err)


            eligibility['trace'][skj] *= eligibility['lambda'] * discount_factor

            # Qk.update_at(ski,aki,Qnew_k)
            ski = copy.deepcopy(skj)

        # Collect episode data
            stats['epi_reward'][ith_episode] += np.array(reward)
            stats['epi_length'][ith_episode] = env._step_count
            #print(f'{env._step_count} | {done}')
        stats['epi_caught'][ith_episode] = 1 if env._step_count < env._max_steps else 0
        stats['epi_psucc'][ith_episode] = np.mean(stats['epi_caught'][-7:]) if ith_episode>7 else  np.mean(stats['epi_caught'])

        score = stats['epi_reward'][-1]
        Logger.log_episode(np.mean(score), env.step_count, env.is_caught)

        # REPORT
        if verbose:
            report = f'\r'
            report += f'\t| epI={ith_episode} '
            if Qk.DM == 'epsilon_greedy': report += f'\t eps={round(eps, 2)}'
            else: report += f'\tlam={round(rationality, 2)}'
            report += f'\tP(Success) = {round(stats["epi_psucc"][ith_episode], 2)}% '
            report += f'\tR = {np.round(stats["epi_reward"][ith_episode], 2)} '

            report += f'\tQrangeR = {[round(np.min(Qk.Qk[0]), 1), round(np.max(Qk.Qk[0]), 1)]}'
            report += f'\tQrangeH = {[round(np.min(Qk.Qk[1]), 1), round(np.max(Qk.Qk[1]), 1)]}'
            print(f'{report}')

        if ith_episode%plt_ever_ith_episode==0:
            Logger.draw()
    #         x = np.arange(len(stats['epi_reward']))
    #         yreward = np.mean(stats['epi_reward'],axis = 1)
    #         stats['line_reward'].set_xdata(x), stats['line_reward'].set_ydata(yreward)
    #         stats['line_mreward'].set_xdata(x), stats['line_mreward'].set_ydata(move_ave(yreward))
    #         stats['line_reward'].axes.set_ylim([min(yreward) - 0.1 * abs(min(yreward)), 1.1 * max(yreward)])
    #         stats['line_reward'].axes.set_xlim([0, max(x)])
    #
    #         ylen = stats['epi_length']
    #         stats['line_len'].set_xdata(x), stats['line_len'].set_ydata(ylen)
    #         stats['line_mlen'].set_xdata(x), stats['line_mlen'].set_ydata(move_ave(ylen))
    #         stats['line_len'].axes.set_xlim([0, max(x)])
    #         stats['line_len'].axes.set_ylim( [min(ylen) - 0.1 * abs(min(ylen)), 1.1 * max(ylen)])
    #
    #         ysucc = stats['epi_psucc']
    #         stats['line_psucc'].set_xdata(x), stats['line_psucc'].set_ydata(ysucc)
    #         stats['line_mpsucc'].set_xdata(x), stats['line_mpsucc'].set_ydata(move_ave(ysucc))
    #         stats['line_psucc'].axes.set_xlim([0, max(x)])
    #         stats['line_psucc'].axes.set_ylim( [-0.1, max(0.6,1.1 * max(ysucc))])
    #         fig.canvas.flush_events()
    #         fig.canvas.draw()
    # plt.savefig('JointQ_Fig')
class Qfun(object):
    def __init__(self,observation_space,action_space):
        self.num_actions = action_space[0].n
        self.num_observe = observation_space[0].shape
        self.num_agents = len(observation_space)
        self.Qk = []

        self.DM = 'epsilon_greedy'
        # self.DM = 'boltzmann'

        self.Boltzmann = lambda qA, lam: np.exp(lam*qA)/np.sum(np.exp(lam*qA))
        self.policy_k = [partial(self.Boltzmann, lam=1),partial(self.Boltzmann, lam=1)]
        _ns = 7
        _naR = 5
        _naH = 5
        for agent_i in range(self.num_agents):
            self.Qk.append(np.zeros([_ns,_ns,_ns,_ns,_ns,_ns,_naR,_naH]))
        self.ns = _ns

        # d_scale = 0.1
        # max_dist = math.dist([1,1],[5,5])
        # _SA = [_ns,_ns,_ns,_ns,_ns,_ns,_naR,_naH]
        # #_Q = np.zeros([_ns,_ns,_ns,_ns,_ns,_ns,_naR,_naH])
        # SA = [np.arange(sa) for sa in _SA]
        # SA_prod = list(itertools.product(*SA))
        # for i,state_action in enumerate(SA_prod):
        #     print(f'\r prog = {round(100*i/len(SA_prod),1)}',end='')
        #     s1 = np.array(state_action[0:2])
        #     s2 = np.array(state_action[2:4])
        #     sprey = np.array(state_action[4:6])
        #     a1 = np.array(state_action[6])
        #     a2 = np.array(state_action[7])
        #
        #     d2prey1 = max_dist-math.dist(self._sim_move(s1, a1),sprey)
        #     d2prey2 = max_dist-math.dist(self._sim_move(s2, a2), sprey)
        #
        #     self.Qk[0][np.array(state_action)] = d_scale*d2prey1
        #     self.Qk[1][np.array(state_action)] = d_scale*d2prey2

    def _sim_move(self,curr_pos,move):
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:
            next_pos = curr_pos
            # next_pos = curr_pos
        else:
            raise Exception('Action Not found!')
        return next_pos




    def sample_pdA(self,obs_k,pd_partner=None,reverse=False):
        if pd_partner is None: pd_partner = np.ones([self.num_agents, self.num_actions]) / self.num_actions
        pd_ego = np.zeros([2,5])
        Qks = self.at(obs_k)
        for agent_i in range(self.num_agents):
            Qs = Qks[agent_i]
            policy = self.policy_k[agent_i]
            pd_notk = pd_partner[agent_i]
            if agent_i == 1: Qs = Qs.T
            qA = np.sum(pd_notk * Qs, axis=0)
            pd_ego[agent_i] = policy(qA)
        if reverse: pd_ego = pd_ego[::-1]
        return pd_ego

    def update_policy(self,rationality_k):
        self.policy_k = [partial(self.Boltzmann, lam=rationality_k[0]),
                         partial(self.Boltzmann, lam=rationality_k[1])]

    def sample_action(self,obs_k,epsilon,pd_partner=None):
        if pd_partner is None: pd_partner = np.ones([self.num_agents, self.num_actions]) / self.num_actions
        action_k = np.zeros(2)
        Qks = self.at(obs_k)
        for agent_i in range(self.num_agents):
            Qs = Qks[agent_i] if agent_i == 0 else Qks[agent_i].T
            pd_notk = pd_partner[agent_i]

            if self.DM == 'epsilon_greedy':
                # # Epsilon Greedy
                if np.random.rand() <= epsilon:
                    action_k[agent_i] = np.random.choice(np.arange(self.num_actions))
                else:
                    qA = np.sum(pd_notk * Qs, axis=0)
                    action_k[agent_i] = np.random.choice(np.array(np.where(qA==np.max(qA))).flatten())
            else:
                # Boltzmann
                policy = self.policy_k[agent_i]
                qA = np.sum(pd_notk * Qs, axis=0)
                action_k[agent_i] = np.random.choice(np.arange(int(self.num_actions)),p=policy(qA))


        return [int(action_k[agent_i]) for agent_i in range(self.num_agents)]


    def update_at(self,obs_k,action,Qnew_k):
        for agent_i in range(self.num_agents):
            obs = obs_k[agent_i]
            self.Qk[agent_i][obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], action[0], action[1]] = Qnew_k[agent_i]


    #############################
    def at(self,obs_k,action=None):
        Qres = []
        for agent_i in range(self.num_agents):
            obs = obs_k[agent_i]
            Q = self.Qk[agent_i]
            qres = Q[obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], :, :]
            if action is not None:
                qres = qres[int(action[0]),int(action[1])]
            Qres.append(qres)
        return Qres
def move_ave(data,window=100):
    new_data = np.empty(np.size(data))
    for i in range(len(data)):
        ub = int(min(len(data), i + window/2))
        lb = int(max(0, i - window/2))
        new_data[i] = np.mean(data[lb:ub])
    return new_data

if __name__ == "__main__":
    main()
    # Qlearning()