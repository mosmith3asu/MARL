import collections
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ma_gym.wrappers import Monitor
import itertools
from enviorment.make_env import CPTEnv
from algorithm.utils.logger import Logger
plt.ion()



class ReplayBuffer:
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            # done_mask_lst.append((np.ones(len(done)) - done).tolist())
            done_mask_lst.append((np.ones(1) - done).tolist())

        return torch.tensor(s_lst, dtype=torch.float), \
               torch.tensor(a_lst, dtype=torch.float), \
               torch.tensor(r_lst, dtype=torch.float), \
               torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst, dtype=torch.float)

    def size(self):
        return len(self.buffer)


class QNet(nn.Module):
    def __init__(self, observation_space, action_space):
        super(QNet, self).__init__()
        self.num_agents = len(observation_space)
        self.n_joint_actions = 25
        self.n_solo_actions = 5
        self.rationality = 1.0
        self.w_explore = 0.0

        self.joint2solo = np.array(list(itertools.product(*[np.arange(5), np.arange(5)])), dtype=int)
        self.solo2joint = np.zeros([5, 5], dtype=int)
        for aJ, joint_action in enumerate(self.joint2solo):
            aR, aH = joint_action
            self.solo2joint[aR, aH] = aJ
        ijoint = np.zeros([2, 5, 25],dtype=np.float32)
        for k in [0, 1]:
            for ak in range(5):
                idxs = np.array(np.where(self.joint2solo[:, k] == ak)).flatten()
                ijoint[k, ak, idxs] = 1
        self.ijoint = torch.from_numpy(ijoint)


        n_obs =observation_space[0].shape[0]

        for agent_i in range(self.num_agents):
            setattr(self, 'agent_{}'.format(agent_i), nn.Sequential(nn.Linear(n_obs, 128),
                                                                    nn.ReLU(),
                                                                    nn.Linear(128, 64),
                                                                    nn.ReLU(),
                                                                    nn.Linear(64, self.n_joint_actions)))
                                                                    #nn.Linear(64, action_space[agent_i].n)))

    def forward(self, obs):
        q_values = [torch.empty(obs.shape[0], )] * self.num_agents
        for agent_i in range(self.num_agents):
            q_values[agent_i] = getattr(self, 'agent_{}'.format(agent_i))(obs[:, agent_i, :]).unsqueeze(1)
        return torch.cat(q_values, dim=1)



    def add_exploration(self,q,w_explore):
        q = q.flatten()
        sp = w_explore # noise size w.r.t. the range of qualities (decrease => less exploration)
        qi_range = float(torch.max(q) - torch.min(q))  # range of q values

        # Undirected Exploration
        psi = np.random.exponential(size= self.n_joint_actions)  # exponential dist. scaled to match range of q-values
        sf = (sp * qi_range / np.max(psi)) # scaling factor
        eta =  torch.from_numpy(sf * psi)


        # from scipy.special import softmax
        # q = np.linspace(-1,1,25)
        # sp = w_explore  # noise size w.r.t. the range of qualities (decrease => less exploration)
        # qi_range = float(np.max(q) - np.min(q))  # range of q values
        # # Undirected Exploration
        # psi = np.random.exponential(size=self.n_joint_actions)  # exponential dist. scaled to match range of q-values
        # sf = (sp * qi_range / np.max(psi))  # scaling factor
        # eta = sf * psi
        #
        # print(qi_range)
        #
        #
        # plt.ioff()
        # plt.close()
        # plt.figure()
        # plt.plot(softmax(q))
        # plt.plot(softmax(eta))
        # plt.plot(softmax(q+eta))
        #
        # plt.show()

        # return cumulative Q
        return (q + eta).flatten()
    def sample_action(self, obs, w_explore,rationality=1,epsilon=0.0):

        self.rationality = rationality
        self.w_explore = w_explore
        self.epsilon = epsilon
        QsA = self.forward(obs)  # 1 x n_agents x n_actions

        if np.random.rand() <= epsilon: # explore
        #aJ = np.random.choice(np.arange(25))
            #action = torch.Tensor([[aJ, aJ]])
            #action[0] = torch.randint(0, QsA.shape[2], action[0].shape).float()
            aJ = torch.randint(0, QsA.shape[2],[1,1])
            action = aJ * torch.ones((QsA.shape[0], QsA.shape[1],))

        else:
            qAH = QsA[0, 1].detach() # qAR = self.add_exploration(QsA[0, 0].detach(),w_explore=w_explore).reshape([5,5]) # 1 x n_actions
            qAR = QsA[0, 0].detach() # qAH = self.add_exploration(QsA[0, 1].detach(),w_explore=w_explore).reshape([5,5]) # 1 x n_actions
            """ JOINT ACTION SELECTION-----------------------------
            # # Perform ToM Inference on partner probability
            # phhatA_H = torch.sum(self.ijoint[1] * 1 / self.n_solo_actions, dim=0)
            # phhatA_R = torch.sum(self.ijoint[0] * 1 / self.n_solo_actions, dim=0)
            # qhat_H = qAH * phhatA_R #torch.sum(qAH * phhat_H, dim=0)
            # qhat_R = qAR * phhatA_H #torch.sum(qAR * phhat_R, dim=0)
            #
            # phatA_H = torch.special.softmax(rationality*qhat_H,dim=0)
            # phatA_R = torch.special.softmax(rationality*qhat_R,dim=0)
            # qH = qAH * phatA_R
            # qR = qAR * phatA_H
            #
            # # Choose Ego's Desired Joint Action
            # # aH_joint = torch.argmax(qH)
            # # aR_joint = torch.argmax(qR)
            # pA_H = torch.special.softmax(rationality * qH, dim=0).detach().numpy()
            # pA_R = torch.special.softmax(rationality * qR, dim=0).detach().numpy()
            # aH_joint = np.random.choice(np.arange(self.n_joint_actions),p=pA_H)
            # aR_joint = np.random.choice(np.arange(self.n_joint_actions),p=pA_R)
            #
            # # Convert to actualized joint action
            # aH = self.joint2solo[aH_joint, 1]
            # aR = self.joint2solo[aR_joint, 0]
            # aJ = self.solo2joint[aR,aH]
            # action = aJ * torch.ones((QsA.shape[0], QsA.shape[1],))
            # p_uniform = 1 / self.n_solo_actions
            # phhatA_H = torch.sum(self.ijoint[1] * p_uniform, dim=1)
            # phhatA_R = torch.sum(self.ijoint[0] * p_uniform, dim=1)
            """

            # Level 0 sophistication
            phhatA_H = torch.ones([ 1, self.n_solo_actions], dtype=qAH.dtype) / self.n_solo_actions
            phhatA_R = torch.ones([ 1, self.n_solo_actions], dtype=qAR.dtype) / self.n_solo_actions
            phhatA_H = torch.matmul(phhatA_H,self.ijoint[1] )
            phhatA_R = torch.matmul(phhatA_R,self.ijoint[0] )

            # Level 1 sophistication
            qhat_H = torch.matmul(self.ijoint[1],(qAH * phhatA_R).T).flatten()
            qhat_R = torch.matmul(self.ijoint[0],(qAR * phhatA_H).T).flatten()
            phatA_H = torch.special.softmax(rationality * qhat_H, dim=0)
            phatA_R = torch.special.softmax(rationality * qhat_R, dim=0)
            phatA_H = torch.matmul(phatA_H, self.ijoint[1])
            phatA_R = torch.matmul(phatA_R, self.ijoint[0])

            # Select Ego Action
            qH = torch.matmul(self.ijoint[1], (qAH * phatA_R).T).flatten()
            qR = torch.matmul(self.ijoint[0], (qAR * phatA_H).T).flatten()
            aH = torch.argmax(qH)
            aR = torch.argmax(qR)
            aJ = self.solo2joint[aR, aH]
            action = aJ * torch.ones((QsA.shape[0], QsA.shape[1],))

        return action
    def sample_action2(self, obs, w_explore,rationality=5):
        global episode_i
        self.rationality = rationality
        self.w_explore = w_explore
        try: lambdaR,lambdaH = rationality
        except: lambdaR,lambdaH = rationality,rationality

        QsA = self.forward(obs) # 1 x n_agents x n_actions
        # qAR = QsA[0, 0].reshape([5, 5])
        # qAH = QsA[0, 1].reshape([5, 5])
        qAR = self.add_exploration(QsA[0, 0].detach(),w_explore=w_explore).reshape([5,5]) # 1 x n_actions
        qAH = self.add_exploration(QsA[0, 1].detach(),w_explore=w_explore).reshape([5,5]) # 1 x n_actions

        # Assume uniform ego transition estimated by partner
        phhatA_H = torch.ones([1,self.n_solo_actions],dtype=qAH.dtype)/self.n_solo_actions
        phhatA_R = torch.ones([self.n_solo_actions,1],dtype=qAR.dtype)/self.n_solo_actions

        # Estimated partner transition
        phatA_H = torch.special.softmax(lambdaH * torch.matmul(phhatA_R.T, qAH), dim=1)
        phatA_R = torch.special.softmax(lambdaR * torch.matmul(qAR, phhatA_H.T), dim=0)

        # Get probability of ego transition then joint transition
        pA_H = torch.special.softmax(lambdaH * torch.matmul(phatA_R.T, qAH), dim=1)
        pA_R = torch.special.softmax(lambdaR * torch.matmul(qAR, phatA_H.T), dim=0)

        print(f'\r', end='')
        # print(f'[{episode_i}]{len(list(np.where(pA_J > 0.01)[0]))}', end='')
        # print(f'\t qR:{round(float(torch.max(qAR.detach()) - torch.min(qAR.detach())),2)}', end='')
        # print(f'\t qH:{round(float(torch.max(qAH.detach()) - torch.min(qAH.detach())),2)}', end='')
        print(f'\t qR:{ round(float(torch.min(qAR.detach())), 2), round(float(torch.max(qAR.detach())), 2)}', end='')
        print(f'\t{list(np.round(pA_R.flatten().detach().numpy(), 2))}', end='')
        print(f'\t qH:{ round(float(torch.min(qAH.detach())), 2), round(float(torch.max(qAH.detach())), 2)}', end='')
        print(f'\t{list(np.round(pA_H.flatten().detach().numpy(), 2))}', end='')
        # print(f'\t{list(np.round(pA_J.detach().numpy(), 2))}', end='')

        # Choose actions
        aH = np.random.choice(np.arange(5), p= pA_H.flatten().detach().numpy())
        aR = np.random.choice(np.arange(5), p= pA_R.flatten().detach().numpy())
        aJ = (self.n_solo_actions-1) * aR + aH # joint action
        #pA_J = torch.prod(torch.cartesian_prod(pA_R.flatten(), pA_H.flatten()), dim=1)
        # aJ = np.random.choice(np.arange(self.n_joint_actions), p=pA_J.detach().numpy())
        action = torch.Tensor([[aJ, aJ]])

        return action


def train(q, q_target, memory, optimizer, gamma, batch_size, update_iter=10):
    for _ in range(update_iter):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(2, a.unsqueeze(-1).long()).squeeze(-1)
        max_q_prime = q_target(s_prime).max(dim=2)[0]
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(env, num_episodes, q):
    old_w_explore = q.w_explore
    old_epsilon = q.epsilon
    score = np.zeros(env.n_agents)
    for episode_i in range(num_episodes):
        state = env.reset()
        done = False# [False for _ in range(env.n_agents)]
        while not done:
            action = q.sample_action(torch.Tensor(state).unsqueeze(0),w_explore=0.0,epsilon=0.0)[0].data.cpu().numpy().tolist()
            next_state, reward, done, info = env.step(action,is_joint=True)
            score += np.array(reward)
            state = next_state
    q.w_explore = old_w_explore
    q.epsilon = old_epsilon
    return np.mean(score / num_episodes)


def run(iWorld, lr, gamma, batch_size, buffer_limit, log_interval, max_episodes,test_episodes,
        full_exploit_steps,full_exploit,full_explore_steps, full_explore,
        max_explore, min_explore, no_penalty_steps, warm_up_steps,
       update_iter):
    global episode_i

    TYPE = 'Baseline'
    ALG = 'Joint'
    Logger.file_name = f'Fig_IDQN_{ALG}_{TYPE}'
    Logger.update_save_directory(Logger.project_root + f'results\\IDQN_W{iWorld}\\')
    Logger.update_plt_title(f'W{iWorld} IDQN Training Results')
    Logger.make_directory()

    env = CPTEnv(iWorld)
    test_env = CPTEnv(iWorld)
    memory = ReplayBuffer(buffer_limit)

    q = QNet(env.observation_space, env.action_space)
    q_target = QNet(env.observation_space, env.action_space)
    q_target.load_state_dict(q.state_dict())
    optimizer = optim.Adam(q.parameters(), lr=lr)

    # SCHEDULE EXPLORATION POLICY
    iexplore  = full_explore * np.ones(full_explore_steps)
    iexploit = full_exploit * np.ones(full_exploit_steps)
    iimprove = np.linspace(max_explore, min_explore, max_episodes - full_explore_steps - full_exploit_steps)
    epi_explore = np.hstack([iexplore, iimprove, iexploit])

    max_epsilon = 0.5
    min_epsilon = 0.0
    warmup_epsilons = max_epsilon * np.ones(warm_up_steps)
    improve_epsilons = np.linspace(max_epsilon, min_epsilon, max_episodes - warm_up_steps)
    epi_epsilons = np.hstack([warmup_epsilons, improve_epsilons])

    train_scores = np.zeros([log_interval,2])
    for episode_i in range(max_episodes):

        # Get episode parameters
        w_explore = epi_explore[episode_i]
        epsilon = epi_epsilons[episode_i]


        cumR = np.zeros(2)
        done = False

        # Intialize episode
        state = env.reset()
        env.enable_penalty = True if episode_i > no_penalty_steps else False
        test_env.enable_penalty = True if episode_i > no_penalty_steps else False


        # Run episode until terminal
        while not done:
            action = q.sample_action(torch.Tensor(state).unsqueeze(0), w_explore=w_explore,epsilon=epsilon)[0]
            action = action.data.cpu().numpy().tolist()
            next_state, reward, done, info = env.step(action,is_joint=True)
            memory.put((state, action, (np.array(reward)).tolist(), next_state, np.array(done, dtype=int).tolist()))
            train_scores[episode_i % log_interval,:] += np.array(reward)
            cumR += np.array(reward)
            state = next_state

        # Close episode
        Logger.log_episode(np.mean(cumR), env.step_count, env.is_caught)

        if memory.size() > warm_up_steps:
            train(q, q_target, memory, optimizer, gamma, batch_size, update_iter)


        # REPORTING ##################
        time2report = (episode_i % log_interval == 0 and episode_i != 0)
        if time2report:
            q_target.load_state_dict(q.state_dict())
            test_score = test(test_env, test_episodes, q)
            print("\r#{:<20} episodes avg train score : {:<15} test score: {:<10} n_buffer : {:<10}  eps : {:<10} lambda : {:<10} w_explore : {:<10}"
                  .format(f'[{episode_i}/{max_episodes} epi]',
                          np.round(np.mean(train_scores),1), round(test_score,1),
                          round(memory.size()), round(q.epsilon,2), round(q.rationality,2), round(q.w_explore,2)),
                  end='')
            Logger.draw()
            print('\t [closed]')
            print(f'\t| [{episode_i}] Caught ({round(100*Logger.Epi_Psuccess[-1],1)}%) @ ',end='')
            train_scores = np.zeros([log_interval, 2])

        if env.is_caught:
            # print(f'\t| [{episode_i}] Caught @ {state[0][-2:]}',end='')
            print(f'{state[0][-2:]} ',end='')


    Logger.save()
    torch.save(q, Logger.save_dir + f'QModel_{ALG}_{TYPE}.torch')

    env.close()
    Logger.close()
    test_env.close()
    print(f'\n\n')


if __name__ == "__main__":

    config = {
        'iWorld': 1,
        'lr': 0.001,#'lr': 0.0005,
        'gamma': 0.9,

        'max_episodes': 5000,
        'no_penalty_steps': 30000,
        'warm_up_steps': 500,
        'full_explore_steps': 20, 'full_explore': 100.0,
        'full_exploit_steps': 1000, 'full_exploit': 0.0,
        'max_explore': 10.0, 'min_explore': 0.0,
        'batch_size': 32,
        'buffer_limit': 50000,
        'log_interval': 100,
        'test_episodes': 7,
        'update_iter': 10,#'update_iter': 10,
    }


    run(**config)

