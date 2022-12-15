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
        for agent_i in range(self.num_agents):
            n_obs = observation_space[agent_i].shape[0]

            setattr(self, 'agent_{}'.format(agent_i), nn.Sequential(nn.Linear(n_obs, 128),
                                                                    nn.ReLU(),
                                                                    nn.Linear(128, 64),
                                                                    nn.ReLU(),
                                                                    nn.Linear(64, action_space[agent_i].n)))

    def forward(self, obs):
        q_values = [torch.empty(obs.shape[0], )] * self.num_agents
        for agent_i in range(self.num_agents):
            q_values[agent_i] = getattr(self, 'agent_{}'.format(agent_i))(obs[:, agent_i, :]).unsqueeze(1)

        return torch.cat(q_values, dim=1)

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        mask = (torch.rand((out.shape[0],)) <= epsilon)
        action = torch.empty((out.shape[0], out.shape[1],))
        action[mask] = torch.randint(0, out.shape[2], action[mask].shape).float()
        action[~mask] = out[~mask].argmax(dim=2).float() #<==== Q Value =======
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
            action = q.sample_action(torch.Tensor(state).unsqueeze(0), epsilon=0)[0].data.cpu().numpy().tolist()
            next_state, reward, done, info = env.step(action)
            score += np.array(reward)
            state = next_state

    return sum(score / num_episodes)


def run(iWorld, lr, gamma, batch_size, buffer_limit, log_interval, max_episodes,
         max_epsilon, min_epsilon, test_episodes, warm_up_steps, no_penalty_steps,update_iter, monitor=False):

    TYPE = 'Baseline'
    Logger.file_name = f'Fig_IDQN_{TYPE}'
    Logger.fname_notes = f'W{iWorld}'
    Logger.update_fname()
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

    warmup_epsilons = max_epsilon * np.ones(warm_up_steps)
    improve_epsilons = np.linspace(max_epsilon,min_epsilon,max_episodes - warm_up_steps)
    epi_epsilons = np.hstack([warmup_epsilons,improve_epsilons])

    score = np.zeros(env.n_agents)
    for episode_i in range(max_episodes):
        epsilon = epi_epsilons[episode_i]
        state = env.reset()
        done = False
        while not done:
            action = q.sample_action(torch.Tensor(state).unsqueeze(0), epsilon)[0].data.cpu().numpy().tolist()
            next_state, reward, done, info = env.step(action)
            memory.put((state, action, (np.array(reward)).tolist(), next_state, np.array(done, dtype=int).tolist()))
            score += np.array(reward)
            state = next_state


        Logger.log_episode(np.mean(score), env.step_count, env.is_caught)
        score = np.zeros(env.n_agents)
        if episode_i > no_penalty_steps: env.enable_penalty = True
        else: env.enable_penalty = False
        if memory.size() > warm_up_steps: train(q, q_target, memory, optimizer, gamma, batch_size, update_iter)

        time2report = (episode_i % log_interval == 0 and episode_i != 0)
        if time2report:
            q_target.load_state_dict(q.state_dict())
            test_score = test(test_env, test_episodes, q)
            print("\r#{:<20} episodes avg train score : {:<20} test score: {:<10} n_buffer : {:<10}  eps : {:<10}"
                  .format(f'[{episode_i}/{max_episodes} epi]',
                          round(sum(score),1), round(test_score,1),
                          round(memory.size()), round(epsilon,2)), end='')
            Logger.draw()
            print('\t [closed]', end='')

    env.close()
    Logger.save()
    test_env.close()
    # torch.save(q, f'results/recordings/idqn/QModel.torch')
    torch.save(q, Logger.save_dir+f'QModel_{TYPE}.torch')
    print(f'\n\n')


#
# if __name__ == '__main__':
#     kwargs = {
#               'lr': 0.0005,
#               'batch_size': 32,
#               'gamma': 0.99,
#               'buffer_limit': 50000,
#               'log_interval': 20,
#               'max_episodes': 30000,
#               'max_epsilon': 0.9,
#               'min_epsilon': 0.1,
#               'test_episodes': 5,
#               'warm_up_steps': 1000,
#               'no_penalty_steps': 1000,
#               'update_iter': 10,
#               'monitor': True}
#
#
#     run(**kwargs)