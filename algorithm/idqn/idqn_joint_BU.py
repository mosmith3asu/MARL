import collections
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ma_gym.wrappers import Monitor
from scipy.special import softmax
# from enviorment.make_env import CPTEnv
from enviorment.make_joint_env  import Joint_CPTEnv
from algorithm.utils.logger import Logger
from functools import partial

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


class AgentDMs(object):
    def __init__(self):
        self.action_samplers = [None,None]
        self.sophistications = [1,1]

    def set_Optimal_Sampler(self,k):
        def Optimal(q):
            pda = np.zeros(np.shape(q))
            pda[np.argmax(q)] = 1
            return pda
        self.action_samplers[k] = Optimal

    def set_Boltzmann_Sampler(self,k,rationality):
        def Boltzmann(q,theta):
            pda = np.nan_to_num(softmax(q*theta))
            pda = np.ones(len(pda))/len(pda) if np.sum(pda) == 0 else pda/np.sum(pda)
            return pda
            # a_choice = np.random.choice(np.array(len(pda)),p=pda)
            # return a_choice
        self.action_samplers[k] = partial(Boltzmann,theta=rationality)


class QNet(nn.Module):
    def __init__(self, observation_space, action_space):
        super(QNet, self).__init__()
        self.num_agents = len(observation_space)
        _nA = action_space[0].n
        self.pda_UNIFORM = np.ones(5)/5

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
        # out = self.forward(obs)
        # mask = (torch.rand((out.shape[0],)) <= epsilon)
        # action = torch.empty((out.shape[0], out.shape[1],))
        # action[mask] = torch.randint(0, out.shape[2], action[mask].shape).float()
        # action[~mask] = out[~mask].argmax(dim=2).float() #<==== Q Value =======
        global DMk
        out = self.forward(obs)
        action = torch.empty((out.shape[0], out.shape[1],)).int()
        qKsA = np.array(out.data)[0]

        QK_solo = np.array([self.get_controllable_Qs(k, qKsA, DMk.action_samplers, ToM=DMk.sophistications[k])
                  for k in range(self.num_agents)])

        rand_num =np.random.rand()
        if  rand_num<= epsilon: # explore
            _actions = np.random.randint(0,5,size=self.num_agents)
            action[0,:] = torch.Tensor(_actions)
        else: # exploit
            _pdaK = [DMk.action_samplers[k](QK_solo[k]) for k in range(self.num_agents)]
            _actions = [np.random.choice(np.arange(5),p=_pdaK[k]) for k in range(self.num_agents)]
            action[0, :] = torch.Tensor(np.array(_actions))

        return action


    def get_controllable_Qs(self, k, qKsA, action_samplers, ToM=0):
        global ia_solo2jointlist
        def getQ(k,QKsA, pda_notk=None):
            Qs_cntrl = np.empty(5)
            pd_anotk = np.array(pda_notk).reshape([1, -1])
            for ak in range(ia_solo2jointlist.shape[1]):
                asolo_idxs = ia_solo2jointlist[k, ak]
                Qs_cntrl[ak] = np.sum(QKsA[k, asolo_idxs] * pd_anotk)
            return Qs_cntrl

        notk = int(not k)
        pda_UNIFORM = self.pda_UNIFORM
        if ToM == 0:
            q_k = getQ(k=k,QKsA=qKsA, pda_notk=pda_UNIFORM)
        elif ToM == 1:
            q_notk = getQ(k=notk,QKsA=qKsA, pda_notk=pda_UNIFORM)
            q_k = getQ(k=k, QKsA=qKsA, pda_notk=action_samplers[notk](q_notk))
        elif ToM == 2:
            q_k = getQ(k=k, QKsA=qKsA, pda_notk=pda_UNIFORM)
            q_notk = getQ(k=notk, QKsA=qKsA,pda_notk=action_samplers[k](q_k))
            q_k = getQ(k=k, QKsA=qKsA, pda_notk=action_samplers[notk](q_notk))
        return np.array(q_k)






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
            ego_action = q.sample_action(torch.Tensor(state).unsqueeze(0), epsilon=0)[0].data.cpu().numpy().tolist()
            joint_action = np.where(np.all(env.JointAction_a2iagent == np.array([ego_action]),axis=1))[0][0]
            joint_action = [joint_action,joint_action]
            next_state, reward, done, info = env.step(joint_action)
            score += np.array(reward)
            state = next_state

    return sum(score / num_episodes)


def run(iWorld, lr, gamma, batch_size, buffer_limit, log_interval, max_episodes,
         max_epsilon, min_epsilon, test_episodes, warm_up_steps, no_penalty_steps,update_iter, monitor=False):
    global DMk, ia_solo2jointlist
    DMk = AgentDMs()
    DMk.set_Boltzmann_Sampler(k = 0, rationality = 5)
    DMk.set_Boltzmann_Sampler(k = 1, rationality = 5)

    TYPE = 'Joint_Baseline'
    Logger.file_name = f'Fig_IDQN_{TYPE}'
    Logger.update_save_directory(f'results/IDQN_W{iWorld}/')
    Logger.update_plt_title(f'W{iWorld} IDQN Training Results')
    Logger.make_directory()

    # env = CPTEnv(iWorld)
    # env = CPTEnv(iWorld)
    # test_env = CPTEnv(iWorld)
    env = Joint_CPTEnv(iWorld)
    ia_solo2jointlist = env.JointAction_asolo2jointlist
    test_env = Joint_CPTEnv(iWorld)
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
            ego_action = q.sample_action(torch.Tensor(state).unsqueeze(0), epsilon)[0].data.cpu().numpy().tolist()
            joint_action = np.where(np.all(env.JointAction_a2iagent == np.array([ego_action]), axis=1))[0][0]
            joint_action = [joint_action, joint_action]
            next_state, reward, done, info = env.step(joint_action)

            memory.put((state, joint_action, (np.array(reward)).tolist(), next_state, np.array(done, dtype=int).tolist()))
            score += np.array(reward)

            # print(f'[{reward}] {ego_action} => {next_state[0]}')
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


if __name__ == "__main__":

    config = {
              'iWorld': 2,
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


    BATCH_WORLDS = [1,2,3,4,5,6,7]
    # BATCH_WORLDS = [ 4]
    for iworld in BATCH_WORLDS:
        config['iWorld'] = iworld
        run(**config)