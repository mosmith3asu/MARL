import collections
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ma_gym.wrappers import Monitor

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
        self.w_undir_explore = 0.5 # w_undir_explore
        self.w_dir_explore = 10

        n_obs =observation_space[0].shape[0]
        obs_shape = [5 for _ in range(n_obs)]

        self.action_frequency = np.zeros(obs_shape+[self.n_joint_actions])

        for agent_i in range(self.num_agents):
            #n_obs = observation_space[agent_i].shape[0]

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



    def add_exploration(self,q):

        sp = self.w_undir_explore # noise size w.r.t. the range of qualities (decrease => less exploration)
        qi_range = np.max(q, axis=1) - np.min(q, axis=1)  # range of q values


        # Undirected Exploration
        psi = np.random.exponential(size=[1, self.n_joint_actions])  # exponential dist. scaled to match range of q-values
        sf = (sp * qi_range / np.max(psi))
        iuniform = np.array(np.where(qi_range > 1e3)).flatten()
        sf[iuniform] = 1
        eta = sf * psi

        # Directed exploration
        #   The directed term gives a bonus to the actions that have been rarely elected
        theta = self.w_dir_explore  # positive factor used to weight the directed exploration
        n_SA = self.action_frequency  # the number of time steps in which action U has been elected
        rho = (qi_range*theta) / np.exp(n_SA)

        # Choose action
        ai = np.argmax(self.q + eta + rho, axis=self.ax_action)
        self.action_frequency[:, ai] = (1) * phi_i  # update selected action freq
        return ai, phi_i

    def sample_action(self, obs, rationality):
        try: lambdaR,lambdaH = rationality
        except: lambdaR,lambdaH = rationality,rationality


        QsA = self.forward(obs) # 1 x n_agents x n_actions
        qAR = QsA[0,0].reshape([5,5]) # 1 x n_actions
        qAH = QsA[0,1].reshape([5,5]) # 1 x n_actions

        # Assume uniform partner transition
        phhatA_H = torch.ones([1,self.n_solo_actions])/self.n_solo_actions
        phhatA_R = torch.ones([self.n_solo_actions,1])/self.n_solo_actions

        # Estimated partner transition
        phatA_H = torch.special.softmax(lambdaH * torch.sum(qAH * phhatA_R, dim=0), dim=0)
        phatA_R = torch.special.softmax(lambdaR * torch.sum(qAR * phhatA_H, dim=1), dim=0)

        # Get probability of ego transition
        pA_H = torch.special.softmax(lambdaH * torch.sum(qAH * phatA_R, dim=0), dim=0)
        pA_R = torch.special.softmax(lambdaR * torch.sum(qAR * phatA_H, dim=1), dim=0)

        # Choose actions
        aH = np.random.choice(np.arange(5),p=np.array((pA_H / torch.sum(pA_H)).data))
        aR = np.random.choice(np.arange(5),p=np.array((pA_R / torch.sum(pA_R)).data))
        aJ = 5 * aR + aH # joint action
        #action = torch.Tensor([[aR,aH]])
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
    score = np.zeros(env.n_agents)
    for episode_i in range(num_episodes):
        state = env.reset()
        done = False# [False for _ in range(env.n_agents)]
        while not done:
            action = q.sample_action(torch.Tensor(state).unsqueeze(0), rationality=100)[0].data.cpu().numpy().tolist()
            next_state, reward, done, info = env.step(action,is_joint=True)
            score += np.array(reward)
            state = next_state

    return sum(score / num_episodes)


def run(iWorld, lr, gamma, batch_size, buffer_limit, log_interval, max_episodes,
         max_epsilon, min_epsilon, test_episodes, warm_up_steps, no_penalty_steps,update_iter, monitor=False):



    TYPE = 'Baseline'
    ALG = 'Joint'
    Logger.file_name = f'Fig_IDQN_{ALG}_{TYPE}'
    Logger.update_save_directory(Logger.project_root + f'results\\IDQN_W{iWorld}\\')
    Logger.update_plt_title(f'W{iWorld} IDQN Training Results')
    Logger.make_directory()

    env = CPTEnv(iWorld)
    test_env = CPTEnv(iWorld)
    if monitor:
        test_env = Monitor(test_env, directory=Logger.save_dir+'recordings/{}'.format('custom'),force=True,
                           video_callable=lambda episode_id: episode_id % 50 == 0)
    memory = ReplayBuffer(buffer_limit)

    q = QNet(env.observation_space, env.action_space)
    q_target = QNet(env.observation_space, env.action_space)
    q_target.load_state_dict(q.state_dict())
    optimizer = optim.Adam(q.parameters(), lr=lr)

    max_rationality = 5
    min_rationality = 0.1

    warmup_rats = min_rationality * np.ones(warm_up_steps)
    improve_rats = np.linspace(min_rationality, max_rationality, max_episodes - warm_up_steps)
    epi_rats = np.hstack([warmup_rats, improve_rats])


    train_scores = np.zeros([log_interval,2])

    for episode_i in range(max_episodes):
        # Get episode parameters
        rationality = epi_rats[episode_i]
        # epsilon = epi_epsilons[episode_i]
        cumR = np.zeros(2)
        # train_score = np.zeros(env.n_agents)


        # Intialize episode
        state = env.reset()
        done = False

        while not done:
            action = q.sample_action(torch.Tensor(state).unsqueeze(0), rationality)[0].data.cpu().numpy().tolist()
            next_state, reward, done, info = env.step(action,is_joint=True)
            memory.put((state, action, (np.array(reward)).tolist(), next_state, np.array(done, dtype=int).tolist()))
            train_scores[episode_i % log_interval,:] += np.array(reward)
            cumR += np.array(reward)
            state = next_state


        Logger.log_episode(np.mean(cumR), env.step_count, env.is_caught)
        # score = np.zeros(env.n_agents)
        env.enable_penalty = True if episode_i > no_penalty_steps else False

        if memory.size() > warm_up_steps:
            train(q, q_target, memory, optimizer, gamma, batch_size, update_iter)

        time2report = (episode_i % log_interval == 0 and episode_i != 0)
        if time2report:
            q_target.load_state_dict(q.state_dict())
            test_score = test(test_env, test_episodes, q)
            print("\r#{:<20} episodes avg train score : {:<15} test score: {:<10} n_buffer : {:<10}  lambda : {:<10}"
                  .format(f'[{episode_i}/{max_episodes} epi]',
                          np.round(np.mean(train_scores),1), round(test_score,1),
                          round(memory.size()), round(rationality,2)), end='')
            Logger.draw()
            print('\t [closed]')
            print(f'\t| [{episode_i}] Caught ({round(100*Logger.Epi_Psuccess[-1],1)}%) @ ',end='')

            train_scores = np.zeros([log_interval, 2])


        if env.is_caught:
            # print(f'\t| [{episode_i}] Caught @ {state[0][-2:]}',end='')
            print(f'{state[0][-2:]} ',end='')

    env.close()
    Logger.save()
    test_env.close()
    # torch.save(q, f'results/recordings/idqn/QModel.torch')
    torch.save(q, Logger.save_dir+f'QModel_{ALG}_{TYPE}.torch')
    print(f'\n\n')


if __name__ == "__main__":

    config = {
              'iWorld': 1,
              'lr': 0.0005,
              'batch_size': 32,
              'gamma': 0.95,
              'buffer_limit': 50000,
              'log_interval': 100,
              'max_episodes': 20000,
              'max_epsilon': 0.8,
              'min_epsilon': 0.1,
              'test_episodes': 5,
              'warm_up_steps': 1000,
              'no_penalty_steps': 20000,
              'update_iter': 10,
              'monitor': False,
    }


    run(**config)

