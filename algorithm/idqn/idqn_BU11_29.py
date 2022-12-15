import collections
import random
import matplotlib.pyplot as plt
from algorithm.utils.logger import Logger
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ma_gym.wrappers import Monitor
from enviorment.make_env import CPTEnv
from scipy import signal,stats
import pickle
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


def run(lr, gamma, batch_size, buffer_limit, log_interval, max_episodes,
         max_epsilon, min_epsilon, test_episodes, warm_up_steps, update_iter, monitor=False):
    # env = gym.make(env_name)
    # test_env = gym.make(env_name)
    start_penalty_step = 2000
    update_nepi = 100
    stats = {}
    stats['Epi Reward'] = []
    stats['Epi Length'] = []

    stats['Epi'] = 0
    stats['n_resample'] = 5
    stats['max_memory_size'] = 600


    fig, axs = plt.subplots(3, 1)
    stats['line_reward'], = axs[0].plot(np.zeros(2),lw=0.5)
    stats['line_mreward'], = axs[0].plot(np.zeros(2))
    stats['line_len'], = axs[1].plot(np.zeros(2),lw=0.5)
    stats['line_mlen'], = axs[1].plot(np.zeros(2))
    stats['line_psucc'], = axs[2].plot(np.zeros(2), lw=0.5)
    stats['line_mpsucc'], = axs[2].plot(np.zeros(2))
    axs[0].set_title('IDQN Training Results')
    axs[0].set_ylabel('Epi Reward')
    axs[1].set_ylabel('Epi Length')
    axs[1].set_xlabel('Episode')

    env = CPTEnv(1)
    test_env = CPTEnv(1)
    if monitor:
        test_env = Monitor(test_env, directory='recordings/idqn/{}'.format('custom'),force=True,
                           video_callable=lambda episode_id: episode_id % 50 == 0)
    memory = ReplayBuffer(buffer_limit)

    q = QNet(env.observation_space, env.action_space)
    q_target = QNet(env.observation_space, env.action_space)
    q_target.load_state_dict(q.state_dict())
    optimizer = optim.Adam(q.parameters(), lr=lr)



    score = np.zeros(env.n_agents)
    for episode_i in range(max_episodes):
        epsilon = max(min_epsilon, max_epsilon - (max_epsilon - min_epsilon) * (episode_i / (0.4 * max_episodes)))
        state = env.reset()
        done = False
        while not done:
            # env.action_space.sample()
            action = q.sample_action(torch.Tensor(state).unsqueeze(0), epsilon)[0].data.cpu().numpy().tolist()
            next_state, reward, done, info = env.step(action)
            memory.put((state, action, (np.array(reward)).tolist(), next_state, np.array(done, dtype=int).tolist()))
            score += np.array(reward)
            state = next_state
        # stats['Epi Reward'].append(sum(score))
        stats['Epi Reward'].append(np.mean(score))
        stats['Epi Length'].append(env._step_count)
        stats['Epi'] = episode_i

        if len(stats['Epi Reward'])>stats['max_memory_size']:
            n_resample = stats['n_resample']
            stats['Epi Reward'] = list(move_ave(stats['Epi Reward'],window=max(n_resample,5))[::n_resample])
            stats['Epi Length'] = list(move_ave(stats['Epi Length'],window=max(n_resample,5))[::n_resample])

        score = np.zeros(env.n_agents)
        if episode_i > start_penalty_step: env.enable_peanlty = True
        else: env.enable_peanlty = False

        if memory.size() > warm_up_steps:
            train(q, q_target, memory, optimizer, gamma, batch_size, update_iter)

        if episode_i % log_interval == 0 and episode_i != 0:
            q_target.load_state_dict(q.state_dict())
            test_score = test(test_env, test_episodes, q)
            print("#{:<10}/{} episodes , avg train score : {:.1f}, test score: {:.1f} n_buffer : {}, eps : {:.1f}"
                  .format(episode_i, max_episodes, sum(score), test_score, memory.size(), epsilon))



        if episode_i % update_nepi == 0:
            x = np.linspace(0,stats['Epi'],len(stats['Epi Reward']))
            stats['line_reward'].set_xdata(x), stats['line_reward'].set_ydata(stats['Epi Reward'])
            stats['line_len'].set_xdata(x), stats['line_len'].set_ydata(stats['Epi Length'])
            stats['line_mreward'].set_xdata(x), stats['line_mreward'].set_ydata(move_ave(stats['Epi Reward']))
            stats['line_mlen'].set_xdata(x), stats['line_mlen'].set_ydata(move_ave(stats['Epi Length']))

            stats['line_reward'].axes.set_xlim([0,max(x)])
            stats['line_reward'].axes.set_ylim([min(stats['Epi Reward'])-0.1*abs(min(stats['Epi Reward'])), 1.1*max(stats['Epi Reward'])])
            stats['line_len'].axes.set_xlim([0, max(x)])
            stats['line_len'].axes.set_ylim([min(stats['Epi Length'])-0.1*abs(min(stats['Epi Length'])), 1.1*max(stats['Epi Length'])])

            fig.canvas.flush_events()
            fig.canvas.draw()


    env.close()
    test_env.close()
    plt.savefig('recordings/idqn/training_figure')
    torch.save(q, f'recordings/idqn/QModel.torch')
    # fig,axs = plt.subplots(2,1)
    # axs[0].plot(stats['Epi Reward'])
    # axs[0].set_title('Epi Reward')
    # axs[1].plot(stats['Epi Len'])
    # axs[1].set_title('Epi Length')
    # plt.show()
def move_ave(data,window=101):
    if window%2==0: window+=1
    if 5*len(data) > window:
        buff0 = np.mean(data[int(window/4):]) * np.ones(window)
        buff1 = np.mean(data[:-int(window/4)]) * np.ones(window)
        tmp_data = np.hstack([buff0, data, buff1])
        filt = stats.norm.pdf(np.arange(window), loc=window / 2, scale=window / 5)
        new_data = signal.fftconvolve(tmp_data, filt / np.sum(filt), mode='full')
        ndiff = np.abs(len(data) - len(new_data))
        new_data = new_data[int(ndiff/2):-int(ndiff/2)]
    else: new_data = data
    return new_data

if __name__ == '__main__':

    kwargs = {#'env_name': 'ma_gym:Switch2-v1',
              'lr': 0.0005,
              'batch_size': 32,
              'gamma': 0.99,
              'buffer_limit': 50000,
              'log_interval': 20,
              'max_episodes': 30000,
              'max_epsilon': 0.9,
              'min_epsilon': 0.1,
              'test_episodes': 5,
              'warm_up_steps': 1000,
              'update_iter': 10,
              'monitor': True}
    # if USE_WANDB:
    #     import wandb
    #
    #     wandb.init(project='minimal-marl', config={'algo': 'idqn', **kwargs}, monitor_gym=True)

    run(**kwargs)