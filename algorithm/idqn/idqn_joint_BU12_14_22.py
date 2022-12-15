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
        self.sample_method = 'NR'
        self.settings = None

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
        eta =  np.array(sf * psi,dtype=np.float32)
        eta = torch.from_numpy(eta)

        # return cumulative Q
        return (q + eta).flatten()


    def load_settings(self,Qnet_settings):
        Qnet_settings

    def sample_NR_action(self, obs, w_explore,rationality=1.0,epsilon=0.0,choose_best=False):
        """
        Sample noisy-rational action
        - softmax sampled on ego quality [1x5]
        - new quality conditined on partner pA
        """


        QsA = self.forward(obs)  # 1 x n_agents x n_actions
        scale_final_rationality = 1

        if np.random.rand() <= epsilon:
            aJ = torch.randint(0, QsA.shape[2], [1, 1])
        else:
            # qAR = QsA[0, 0].detach()  # qAR = self.add_exploration(QsA[0, 0].detach(),w_explore=w_explore) # 1 x n_actions
            # qAH = QsA[0, 1].detach()  # qAH = self.add_exploration(QsA[0, 1].detach(),w_explore=w_explore) # 1 x n_actions
            #"""
            # Level 0 sophistication
            _qAH = self.add_exploration(QsA[0, 1].detach(), w_explore=w_explore)
            _qAR = self.add_exploration(QsA[0, 0].detach(), w_explore=w_explore)
            phhatA_H = torch.ones([1, self.n_solo_actions], dtype=_qAH.dtype) / self.n_solo_actions
            phhatA_R = torch.ones([1, self.n_solo_actions], dtype=_qAR.dtype) / self.n_solo_actions
            phhatA_H = torch.matmul(phhatA_H, self.ijoint[1])
            phhatA_R = torch.matmul(phhatA_R, self.ijoint[0])

            # Level 1 sophistication
            qhat_H = torch.matmul(self.ijoint[1], (_qAH * phhatA_R).T).flatten()
            qhat_R = torch.matmul(self.ijoint[0], (_qAR * phhatA_H).T).flatten()
            phatA_H = torch.matmul(torch.special.softmax(rationality * qhat_H, dim=0),self.ijoint[1])
            phatA_R = torch.matmul(torch.special.softmax(rationality * qhat_R, dim=0),self.ijoint[0])

            # Select Ego Action
            q_H = torch.matmul(self.ijoint[1], (_qAH * phatA_R).T).flatten()
            q_R = torch.matmul(self.ijoint[0], (_qAR * phatA_H).T).flatten()
            pA_H = torch.special.softmax(rationality * q_H, dim=0)
            pA_R = torch.special.softmax(rationality * q_R, dim=0)
            # aH = np.random.choice(np.arange(self.n_solo_actions), p=pA_H.detach().numpy())
            # aR = np.random.choice(np.arange(self.n_solo_actions), p=pA_R.detach().numpy())
            aH = torch.argmax(pA_H).detach().int()
            aR = torch.argmax(pA_R).detach().int()
            # Get joint action
            aJ = self.solo2joint[aR, aH] # joint action
            """
            # Level 0 sophistication
            joint_qAH = self.add_exploration(QsA[0, 1].detach(), w_explore=w_explore)
            joint_qAR = self.add_exploration(QsA[0, 0].detach(), w_explore=w_explore)
            ego_phhatA_H = torch.ones([1, self.n_solo_actions], dtype=joint_qAH.dtype) / self.n_solo_actions
            ego_phhatA_R = torch.ones([1, self.n_solo_actions], dtype=joint_qAR.dtype) / self.n_solo_actions
            joint_phhatA_H = torch.matmul(ego_phhatA_H, self.ijoint[1])
            joint_phhatA_R = torch.matmul(ego_phhatA_R, self.ijoint[0])

            # Level 1 sophistication
            joint_qhat_H = (joint_qAH * joint_phhatA_R)
            joint_qhat_R = (joint_qAR * joint_phhatA_H)
            ego_qhatA_H = torch.matmul(self.ijoint[1], joint_qhat_H.T).flatten()
            ego_qhatA_R = torch.matmul(self.ijoint[0], joint_qhat_R.T).flatten()
            ego_phatA_H = torch.special.softmax(rationality * ego_qhatA_H , dim=0)
            ego_phatA_R = torch.special.softmax(rationality * ego_qhatA_R , dim=0)
            joint_phatA_H = torch.matmul(ego_phatA_H, self.ijoint[1])
            joint_phatA_R = torch.matmul(ego_phatA_R, self.ijoint[0])

            # Level 2 sophistication
            joint_q_H = (joint_qAH * joint_phatA_R)
            joint_q_R = (joint_qAR * joint_phatA_H)
            ego_q_H = torch.matmul(self.ijoint[1],joint_q_H.T).flatten()
            ego_q_R = torch.matmul(self.ijoint[0],joint_q_R.T).flatten()
            ego_pA_H = torch.special.softmax(rationality * ego_q_H, dim=0)
            ego_pA_R = torch.special.softmax(rationality * ego_q_R, dim=0)
            joint_phatA_H = torch.matmul(ego_pA_H, self.ijoint[1])
            joint_phatA_R = torch.matmul(ego_pA_R, self.ijoint[0])

            # Select Ego Action
            # joint_q_H = (joint_qAH * joint_phatA_R)
            # joint_q_R = (joint_qAR * joint_phatA_H)
            # ego_q_H = torch.matmul(self.ijoint[1], joint_q_H.T).flatten()
            # ego_q_R = torch.matmul(self.ijoint[0], joint_q_R.T).flatten()
            # ego_pA_H = torch.special.softmax(rationality * ego_q_H, dim=0)
            # ego_pA_R = torch.special.softmax(rationality * ego_q_R, dim=0)
            # aH = torch.argmax(ego_pA_H).detach().int()
            # aR = torch.argmax(ego_pA_R).detach().int()


            ego_pA_H = ego_pA_H.detach().numpy()
            ego_pA_R = ego_pA_R.detach().numpy()
            aH = np.random.choice(np.arange(self.n_solo_actions), p= ego_pA_H)
            aR = np.random.choice(np.arange(self.n_solo_actions), p= ego_pA_R)
            # Get joint action
            aJ = self.solo2joint[aR, aH]  # joint action


            # Report ----------------------
            ego_q_H = ego_q_H.detach().numpy()
            ego_q_R = ego_q_R.detach().numpy()
            dqH = round(np.max(ego_q_H) - np.min(ego_q_H), 2)
            dqR = round(np.max(ego_q_R) - np.min(ego_q_R), 2)
            print(f'\r   R{np.array([dqR]).round(2)}:{ego_pA_R.round(2)} '
                  f'\t\t H{np.array([dqH]).round(2)}:{ego_pA_H.round(2)}',end='')

            """

            #aJ = torch.argmax(QsA[0, 0].detach() + QsA[0, 1].detach())
        action = aJ * torch.ones((QsA.shape[0], QsA.shape[1],))
        return action

    def sample_SW_action(self, obs, w_explore,rationality=1.0,epsilon=0.0):
        """Sample social welfare action"""
        QsA = self.forward(obs)
        # qAR = self.add_exploration(QsA[0, 0].detach(), w_explore=w_explore)  # 1 x n_actions
        # qAH = self.add_exploration(QsA[0, 1].detach(), w_explore=w_explore)  # 1 x n_actions
        # aJ = torch.argmax(qAR + qAH)
        if np.random.rand() <= epsilon:
            aJ = torch.randint(0, QsA.shape[2], [1, 1])
        else:
            aJ = torch.argmax( QsA[0, 0].detach()+QsA[0, 1].detach())
        action = aJ * torch.ones((QsA.shape[0], QsA.shape[1],))
        return action

    def sample_action(self, obs, w_explore,rationality=1.0,epsilon=0.0,choose_best=False):
        if rationality is None: rationality = self.rationality
        if w_explore is None: w_explore = self.w_explore
        if epsilon is None: epsilon = self.epsilon


        self.rationality = rationality
        self.w_explore = w_explore
        self.epsilon = epsilon
        if self.sample_method == 'NR':
            action = self.sample_NR_action( obs, w_explore, rationality=rationality,epsilon=epsilon)
        elif self.sample_method == 'SW':
            action = self.sample_SW_action(obs, w_explore, rationality=rationality, epsilon=epsilon)
        else: raise Exception(f'unknown sample method [{self.sample_method}]')
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
    old_rationality = q.rationality

    length =np.zeros(env.n_agents)
    psucc = np.zeros(env.n_agents)
    score = np.zeros(env.n_agents)
    for episode_i in range(num_episodes):
        state = env.reset()
        done = False# [False for _ in range(env.n_agents)]
        while not done:
            action = q.sample_action(torch.Tensor(state).unsqueeze(0),w_explore=0.0,epsilon=0.0,rationality=None)
            action = action[0].data.cpu().numpy().tolist()
            next_state, reward, done, info = env.step(action,is_joint=True)
            score += np.array(reward)
            state = next_state

        if env.is_caught: psucc +=1
        length += env.step_count


    q.w_explore = old_w_explore
    q.epsilon = old_epsilon
    q.rationality = old_rationality

    final_score = np.mean(score / num_episodes)
    final_length =np.mean(length / num_episodes)
    final_psucc = np.mean(psucc / num_episodes)
    return final_score,final_length,final_psucc



def run(run_config):
    # Unpack configuration -------------------------------
    iWorld = run_config['iWorld']
    lr = run_config['lr']
    gamma = run_config['gamma']
    lamb = run_config['lambda']
    start_sp, end_sp = run_config['sp']
    start_epsilon, end_epsilon = run_config['epsilon']
    action_sampler = run_config['action_sampler']

    max_episodes = run_config['max_episodes']
    no_penalty_steps = run_config['no_penalty_steps']
    warm_up_steps = run_config['warm_up_steps']
    full_exploit_steps = run_config['full_exploit_steps']
    full_explore_steps = run_config['full_explore_steps']

    batch_size = run_config['batch_size']
    buffer_limit = run_config['buffer_limit']
    log_interval = run_config['log_interval']
    update_iter = run_config['update_iter']

    SCALE_REWARDS = run_config['r_scale']
    test_episodes = run_config['test_episodes']
    # full_exploit = run_config['full_exploit']
    # full_explore = run_config['full_explore']

    # Set up logger -------------------------------
    TYPE = 'Baseline'
    ALG = 'Joint'
    Logger.file_name = f'Fig_IDQN_{ALG}_{TYPE}'
    Logger.update_save_directory(Logger.project_root + f'results\\IDQN_W{iWorld}\\')
    Logger.update_plt_title(f'W{iWorld} IDQN Training Results')
    Logger.make_directory()


    # Create environment -------------------------------
    env = CPTEnv(iWorld)
    test_env = CPTEnv(iWorld)
    memory = ReplayBuffer(buffer_limit)

    # Create Q functions -------------------------------
    q = QNet(env.observation_space, env.action_space)
    q.settings = run_config
    q.action_sampler = action_sampler

    q_target = QNet(env.observation_space, env.action_space)
    q_target.load_state_dict(q.state_dict())
    optimizer = optim.Adam(q.parameters(), lr=lr)

    # SCHEDULE EXPLORATION POLICY -------------------------------
    iexplore  = start_sp * np.ones(full_explore_steps)
    iexploit = end_sp * np.ones(full_exploit_steps)
    iimprove = np.linspace(start_sp, end_sp, max_episodes - full_explore_steps - full_exploit_steps)
    epi_explore = np.hstack([iexplore, iimprove, iexploit])

    warmup_epsilons = start_epsilon * np.ones(full_explore_steps)
    perform_epsilons = end_epsilon * np.ones(full_exploit_steps)
    improve_epsilons = np.linspace(start_epsilon, end_epsilon,  max_episodes - full_explore_steps - full_exploit_steps)
    epi_epsilons = np.hstack([warmup_epsilons, improve_epsilons,perform_epsilons])

    if isinstance(lamb,list):
        start_rationality = lamb[0]
        stop_rationality = lamb[1]
        ioptimal = start_rationality * np.ones(full_explore_steps)
        irealworld = stop_rationality * np.ones(full_exploit_steps)
        itransition = np.linspace(start_rationality, stop_rationality, max_episodes - full_explore_steps - full_exploit_steps)
        # slope = 1/1.5
        # iinv = np.power(1/(np.linspace(1,10,max_episodes - full_explore_steps - full_exploit_steps)-0.1)-0.1,slope)
        # itransition = (max_rationality-min_rationality)*iinv + min_rationality
        epi_rationality = np.hstack([ioptimal, itransition, irealworld])
    else:
        print(f'Static lambda = {lamb}')
        epi_rationality = lamb * np.ones(max_episodes)



    train_scores = np.zeros([log_interval,2])
    for episode_i in range(max_episodes):
        # Get episode parameters
        w_explore = epi_explore[episode_i]
        epsilon = epi_epsilons[episode_i]
        rationality = epi_rationality[episode_i]
        cumR = np.zeros(2)
        done = False

        # Intialize episode
        state = env.reset()
        env.enable_penalty = True if episode_i > no_penalty_steps else False
        test_env.enable_penalty = True if episode_i > no_penalty_steps else False

        # Run episode until terminal
        while not done:
            action = q.sample_action(torch.Tensor(state).unsqueeze(0), w_explore=w_explore,epsilon=epsilon,rationality=rationality)
            action = action[0].data.cpu().numpy().tolist()
            next_state, reward, done, info = env.step(action,is_joint=True)
            reward = [r*SCALE_REWARDS for r in reward]

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
            #test_score,test_length,test_psucc = test(test_env, test_episodes, q)
            #Logger.log_episode(test_score,test_length,test_psucc)
            print("\r#{:<20}".format(f'[{episode_i}/{max_episodes} epi]'),end= '')
            print("train score: {:<10} ".format(np.round(np.mean(train_scores), 1)), end='')
            #print("test score: {:<20}".format(f'[r:{np.round(test_score, 1)},l:{np.round(test_length, 1)},ps:{np.round(test_psucc, 1)}]'), end='')
            print("n_buffer : {:<10}  ".format(memory.size()), end='')
            print("eps : {:<10} ".format(round(q.epsilon,2)), end='')
            print("lambda : {:<10}".format( round(q.rationality,2)), end='')
            print("w_explore : {:<10}".format(round(q.w_explore,2)), end='')
            print("sampler : {:<10}".format(q.sample_method), end='')
            #print("Qrange : {:<10}".format(f'[{torch.min(q)},{torch.max(q)}]'), end='')
            Logger.draw()
            print('\t [closed]')
            train_scores = np.zeros([log_interval, 2])

    Logger.save()
    torch.save(q, Logger.save_dir + f'QModel_{ALG}_{TYPE}.torch')

    env.close()
    # Logger.close()
    test_env.close()
    print(f'\n\n')


if __name__ == "__main__":
    max_episodes = int(40e3)
    ################# working ####################################
    # config = {
    #     'iWorld': 1,
    #     'lr': 0.0001,#'lr': 0.0005,
    #     'gamma': 0.99,
    #     'lambda': [5,5],
    #     'sp': [1.5,0.05],
    #     'epsilon': [0.0, 0.0],#[0.6, 0.05],
    #     'r_scale': 1,
    #     'max_episodes': max_episodes,
    #     'no_penalty_steps': 5000,# int((1/4) * max_episodes),
    #     'warm_up_steps': 1000,
    #     'full_explore_steps': 1000, #'full_explore': 100.0,
    #     'full_exploit_steps': 3000, #'full_exploit': 0.1,
    #     'batch_size': 32, 'buffer_limit': 50000,
    #     'log_interval': 100, 'test_episodes': 5,  'update_iter': 20,
    # }
    ##############################################################
    config = {
        'iWorld': 1,
        'lr': 0.001,  # 'lr': 0.0005,
        'gamma': 0.90,
        'lambda': [10, 10],
        'sp': [2, 0.05],
        'epsilon': [0.0, 0.0],  # [0.6, 0.05],
        'r_scale': 1,
        'action_sampler': 'SW',
        'max_episodes': max_episodes,
        'no_penalty_steps': 000,  # int((1/4) * max_episodes),
        'warm_up_steps': 2000,
        'full_explore_steps': 2000,  # 'full_explore': 100.0,
        'full_exploit_steps': 3000,  # 'full_exploit': 0.1,
        'batch_size': 32, 'buffer_limit': 50000,
        'log_interval': 100, 'test_episodes': 5, 'update_iter': 10,
    }
    # config = {
    #     'iWorld': 1,
    #     'lr': 0.00005,  # 'lr': 0.0005,
    #     'gamma': 0.95,
    #     'lambda': [2, 2],
    #     'sp': [2, 0.05],
    #     'max_episodes': 20000,
    #     'no_penalty_steps': 10000,
    #     'warm_up_steps': 1000,
    #     'full_explore_steps': 1000,  # 'full_explore': 100.0,
    #     'full_exploit_steps': 1000,  # 'full_exploit': 0.1,
    #     'batch_size': 32, 'buffer_limit': 50000,
    #     'log_interval': 100,
    #     'test_episodes': 5,
    #     'update_iter': 10,  # 'update_iter': 10,
    # }


    BATCH_WORLDS = [1,2,3,4,5,6,7]
    #LR = [None, 0.00005,0.00005,0.00005,0.00005,0.00005,0.00005,0.00005]
    for iworld in BATCH_WORLDS:
        config['iWorld'] = iworld
        run(config)

    # run(config)

