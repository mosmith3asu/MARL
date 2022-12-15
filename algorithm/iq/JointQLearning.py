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


class JointQ(nn.Module):
    def __init__(self, observation_space, action_space):
        # super(QNet, self).__init__()
        self.num_agents = len(observation_space)
        _nA = action_space[0].n
        self.pda_UNIFORM = np.ones(5)/5

        _nk = 2
        _ns = 5
        _na_joint = 25
        self.tbl =np.array( [np.zeros([_ns,_ns,_ns,_ns,_ns,_ns,_na_joint]) for k in range(2)])

        # for agent_i in range(self.num_agents):
        #     n_obs = observation_space[agent_i].shape[0]
        #
        #     setattr(self, 'agent_{}'.format(agent_i), nn.Sequential(nn.Linear(n_obs, 128),
        #                                                             nn.ReLU(),
        #                                                             nn.Linear(128, 64),
        #                                                             nn.ReLU(),
        #                                                             nn.Linear(64, action_space[agent_i].n)))

    # def forward(self, obs):
    #     q_values = [torch.empty(obs.shape[0], )] * self.num_agents
    #     for agent_i in range(self.num_agents):
    #         q_values[agent_i] = getattr(self, 'agent_{}'.format(agent_i))(obs[:, agent_i, :]).unsqueeze(1)
    #     return torch.cat(q_values, dim=1)
    def eval_Qcntrl(selfk,k, QKsA, pda_notk=None):
        Qs_cntrl = np.empty(5)
        pd_anotk = np.array(pda_notk).reshape([1, -1])
        for ak in range(ia_solo2jointlist.shape[1]):
            asolo_idxs = ia_solo2jointlist[k, ak]
            Qs_cntrl[ak] = np.sum(QKsA[k, asolo_idxs] * pd_anotk)
        return Qs_cntrl

    def sample_inference(self, state, DMk):
        global ia_solo2jointlist
        qKsA = self.get_QKs(state)
        ToMK = DMk.sophistications
        action_samplers = DMk.action_samplers

        pda_UNIFORM = self.pda_UNIFORM
        ego_inference = np.zeros([2,5])

        for k in range(2):
            notk = int(not k)
            if ToMK[k] == 0:
                ego_inference[k, :] =pda_UNIFORM
            elif ToMK[k] == 1:
                q_notk = self.eval_Qcntrl(k=notk, QKsA=qKsA, pda_notk=pda_UNIFORM)
                pda_notk = action_samplers[notk](q_notk)
                ego_inference[k,:] =pda_notk
            elif ToMK[k]:
                q_k = self.eval_Qcntrl(k=k, QKsA=qKsA, pda_notk=pda_UNIFORM)
                q_notk = self.eval_Qcntrl(k=notk, QKsA=qKsA, pda_notk=action_samplers[k](q_k))
                pda_notk = action_samplers[notk](q_notk)
                ego_inference[k,:] = pda_notk

        joint_inference = np.zeros([2,25])
        for k in range(2):
            notk = int(not k)
            for a_notk,pa_notk in enumerate(ego_inference[k]):
                anotk_joint_idxs = ia_solo2jointlist[k, a_notk]
                joint_inference[k,anotk_joint_idxs] = pa_notk

        return ego_inference,joint_inference


    def get_QKs(self,state):
        qKsA = np.zeros([2, 25])
        for k in range(2):
            x0, y0, x1, y1, x2, y2 = list(np.array(state[k]) - 1)
            _q = self.tbl[k, x0, y0, x1, y1, x2, y2]
            qKsA[k, :] = _q
        return qKsA
    def sample_action(self, state, epsilon,pda_partner,DMk):
        # qKsA = np.array([self.tbl[[k] + obs] for k in range(2)])
        qKsA = self.get_QKs(state)
        QK_solo = np.zeros([2,5])
        for k in range(2):
            q_k = self.eval_Qcntrl(k=k, QKsA=qKsA, pda_notk=pda_partner[k])
            QK_solo[k,:] = q_k

        # EXPLORE/EXPLOIT
        ego_action =np.zeros([1,2],dtype='int8')
        rand_num =np.random.rand()
        if  rand_num<= epsilon: # explore
            _actions = np.random.randint(0,5,size=self.num_agents)
            ego_action[0,:] = torch.Tensor(_actions)
        else: # exploit
            _pdaK = [DMk.action_samplers[k](QK_solo[k]) for k in range(self.num_agents)]
            _actions = [np.random.choice(np.arange(5),p=_pdaK[k]) for k in range(self.num_agents)]
            ego_action[0, :] = torch.Tensor(np.array(_actions))

        # # Get joint conditional prob
        # pda_joint_cond = np.zeros([2, 25])

        return ego_action


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


    TYPE = 'Baseline'
    Logger.file_name = f'Fig_IDQN_{TYPE}'
    Logger.fname_notes = f'W{iWorld}'
    Logger.update_fname()
    Logger.make_directory()

    # env = CPTEnv(iWorld)
    # env = CPTEnv(iWorld)
    # test_env = CPTEnv(iWorld)
    env = Joint_CPTEnv(iWorld)
    ia_solo2jointlist = env.JointAction_asolo2jointlist
    test_env = Joint_CPTEnv(iWorld)

    q = JointQ(env.observation_space, env.action_space)

    warmup_epsilons = max_epsilon * np.ones(warm_up_steps)
    improve_epsilons = np.linspace(max_epsilon,min_epsilon,max_episodes - warm_up_steps)
    epi_epsilons = np.hstack([warmup_epsilons,improve_epsilons])

    score = np.zeros(env.n_agents)
    for episode_i in range(max_episodes):
        epsilon = epi_epsilons[episode_i]
        state = env.reset()
        done = False
        while not done:
            ego_inference, joint_inference = q.sample_inference(state, DMk)
            ego_actions = q.sample_action(state, epsilon,ego_inference,DMk)
            joint_action = np.where(np.all(env.JointAction_a2iagent == np.array(ego_actions), axis=1))[0][0]
            joint_action = [joint_action, joint_action]


            pd_conditional = np.zeros([2,25])
            for k in range(2):
                ajoint_ego = env.JointAction_asolo2jointlist[k, ego_actions[0][k]]
                pd_conditional[k,ajoint_ego] = 1
                pd_conditional[k,:] *= joint_inference[k]


            next_state, reward, done, info = env.step(joint_action)


            next_ego_inference, next_joint_inference = q.sample_inference(next_state, DMk)
            next_ego_actions = q.sample_action(next_state, 0, next_ego_inference, DMk)
            next_joint_action = np.where(np.all(env.JointAction_a2iagent == np.array(next_ego_actions), axis=1))[0][0]
            next_joint_action = [next_joint_action, next_joint_action]

            Qmax_sj = [q.get_QKs(next_state)]

            TD_target = reward + gamma * Qmax_sj  # RQ[sj][ajR]
            q.tbl = (1-lr)*q.tbl + (lr)*(reward +)

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
              'iWorld': 1,
              'lr': 0.0005,
              'batch_size': 32,
              'gamma': 0.95,
              'buffer_limit': 50000,
              'log_interval': 100,
              'max_episodes': 10000,
              'max_epsilon': 0.8,
              'min_epsilon': 0.1,
              'test_episodes': 5,
              'warm_up_steps': 1000,
              'no_penalty_steps': 10000,
              'update_iter': 10,
              'monitor': False,
    }

    run(**config)