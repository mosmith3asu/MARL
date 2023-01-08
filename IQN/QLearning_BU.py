############################################################
## Packages ################################################
import logging
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import itertools
from itertools import count
from torch.utils.data.sampler import WeightedRandomSampler
from utilities.make_env import PursuitEvastionGame
from collections import namedtuple, deque
Transition = namedtuple('Transition',  ('state', 'action', 'next_state', 'reward','done'))


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)






#################################################################
## DQN algorithm ################################################
class Qfunction(object):


    @classmethod
    def construct_path(_, iWorld, policy_type, algorithm):
        fname = f'{algorithm}_{policy_type}.torch'
        project_dir = os.getcwd().split('MARL')[0] + 'MARL\\'
        file_path = project_dir + f'results\\IDQN_W{iWorld}\\{fname}'
        return file_path

    @classmethod
    def load(_, iWorld, policy_type, algorithm, verbose=True):

        try:
            file_path = Qfunction.construct_path(iWorld, policy_type, algorithm)
            q = torch.load(file_path)
        except:
            logging.warning('Inverted policy and algorithm name')
            file_path = Qfunction.construct_path(iWorld, algorithm, policy_type)
            q = torch.load(file_path)
        file_size = os.path.getsize(file_path)
        if verbose: print(
            f'\nLOADED EXISTING QNet \n\t| Path: {file_path} \n\t| Size: {file_size} \n\t| Device: {q.tensor_type["device"]}')
        return q

    @classmethod
    def save(_, module, iWorld, algorithm, policy_type, axes = None,verbose=True):
        if axes is not None: module.axes = axes #copy.deepcopy(axes)
        file_path = Qfunction.construct_path(iWorld, policy_type, algorithm)
        torch.save(module, file_path)
        file_size = os.path.getsize(file_path)
        if verbose: print(f'\nQNet SAVED \n\t| Path: {file_path} \n\t| Size: {round(file_size / 1000)}MB')

    @classmethod
    def preview(_, iWorld, policy_type, algorithm):
        q = Qfunction.load(iWorld, policy_type, algorithm)
        assert q.axes is not None, 'tried previewing DQN figure that does not exist'
        plt.ioff()
        plt.show()

        # if verbose: print(
        #     f'\nLOADED EXISTING QNet \n\t| Path: {file_path} \n\t| Size: {file_size} \n\t| Device: {q.tensor_type["device"]}')
        # return q

    def __init__(self,device,dtype,sophistocation,run_config=None,rationality=1):
        #super(DQN, self).__init__()
        self.n_act = 25
        self.n_ego_actions = 5
        self.n_obs = 6
        self.n_agents = 2

        self.pos_offset = 1

        self.walls = torch.tensor([[1,1],[1,3],[3,1],[3,3]])



        self.rationality = rationality
        #self.rationality = 2.0; logging.warning(f'IDQN>> Rationality set to {self.rationality}')
        self.sophistocation = sophistocation
        self.tensor_type = {'device':device,'dtype':dtype}
        self.run_config = run_config

        self.axes = None


        joint2solo = np.array(list(itertools.product(*[np.arange(5), np.arange(5)])), dtype=int)
        solo2joint = np.zeros([5, 5], dtype=int)
        for aJ, joint_action in enumerate(joint2solo):
            aR, aH = joint_action
            solo2joint[aR, aH] = aJ
        ijoint = np.zeros([2, 5, 25], dtype=np.float32)
        for k in range(self.n_agents):
            for ak in range(5):
                idxs = np.array(np.where(joint2solo[:, k] == ak)).flatten()
                ijoint[k, ak, idxs] = 1


        self.ijoint = torch.as_tensor(ijoint,**self.tensor_type)
        self.solo2joint = torch.as_tensor(solo2joint,**self.tensor_type)
        self.joint2solo = torch.as_tensor(joint2solo, **self.tensor_type)

        self.env = None

        nxy = 5
        # self.tbl = {}
        # self.tbl[0] = torch.zeros([nxy, nxy, nxy, nxy, nxy, nxy, self.n_act])
        # self.tbl[1] = torch.zeros([nxy,nxy,nxy,nxy,nxy,nxy,self.n_act])
        self.tbl = torch.zeros([self.n_agents,nxy,nxy,nxy,nxy,nxy,nxy,self.n_act])

        self.ENABLE_DIR_EXPLORE = False
        self.DIR_EXPLORATION = 10
        self.state_visitation = torch.zeros([nxy, nxy, nxy, nxy, nxy, nxy])
        self.action2idx = {'down': 0, 'left': 1, 'up': 2, 'right': 3, 'wait': 4}
        self.idx2action = {v: k for k, v in self.action2idx.items()}
        self.explicit_actions = {'down': torch.tensor([1, 0], **self.tensor_type),
                                 'left': torch.tensor([0, -1], **self.tensor_type),
                                 'up': torch.tensor([-1, 0], **self.tensor_type),
                                 'right': torch.tensor([0, -1], **self.tensor_type),
                                 'wait': torch.tensor([0, 0], **self.tensor_type)}
        self.max_dist =  torch.dist(torch.tensor([0.,0.]), torch.tensor([4.,4.]))

        if self.ENABLE_DIR_EXPLORE: logging.warning('Directed Exploration enabled')
        if self.rationality!=1: logging.warning(f'Rationality ={self.rationality}')

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    # def forward(self, obs,action=None):
    #     obs = obs.reshape([-1, 6]) - self.pos_offset
    #     n_batch = obs.shape[0]
    #     k_idxs = [[tuple(k * torch.ones(n_batch, dtype=torch.int)) for k in range(self.n_agents)]]
    #     s_idxs = list(zip(*obs.int()))
    #     qAk = torch.transpose(self.tbl[k_idxs + s_idxs], dim0=0, dim1=1)
    #     if action is not None:
    #         action = action.reshape([-1, 1])
    #         b_idxs = [torch.zeros(n_batch).long()]
    #         qAk = qAk[b_idxs +k_idxs +list(zip(*action))].reshape([n_batch,self.n_agents,1])
    #     return qAk
    def forward(self, obs, action=None):
        obs = obs.reshape([-1, 6]) - self.pos_offset
        n_batch = obs.shape[0]
        qSA = torch.zeros([n_batch,2,self.n_act]) if action is None else torch.zeros([n_batch,2,1])

        for k in range(self.n_agents):
            for ibatch in range(n_batch):
                x0, y0, x1, y1, x2, y2 = obs[ibatch].long()
                a = np.arange(self.n_act) if action is None else action.reshape([-1, 1])[ibatch].long()
                qSA[ibatch, k] = self.tbl[k, x0, y0, x1, y1, x2, y2, a]

        return qSA



    def __call__(self, *args, **kwargs):
        return self.forward( *args, **kwargs)


    def QRE(self,qAk,get_pd = True,get_q=False):
        sophistocation = self.sophistocation
        n_agents = self.n_agents
        n_joint_act = self.n_act
        n_ego_act = 5
        rationality = self.rationality
        n_batch = qAk.shape[0]

        pdAjointk = torch.ones([n_batch, n_agents, n_joint_act], **self.tensor_type) / n_joint_act
        qAego = torch.empty([n_batch, n_agents, n_ego_act], **self.tensor_type)
        pdAegok = torch.empty([n_batch, n_agents, n_ego_act], **self.tensor_type)
        for isoph in range(sophistocation):
            new_pdAjointk = torch.zeros([n_batch, n_agents, n_joint_act])

            for k in range(n_agents):
                ijoint_batch = self.ijoint[k, :, :].T.repeat([n_batch, 1, 1])
                qAJ_conditioned = qAk[:, k, :] * pdAjointk[:, int(not k), :]
                qAego[:, k, :] = torch.bmm(qAJ_conditioned.unsqueeze(1), ijoint_batch).squeeze()
                pdAegok[:, k, :] = torch.special.softmax(rationality * qAego[:, k, :], dim=-1)
                new_pdAjointk[:, k, :] = torch.bmm(pdAegok[:, k, :].unsqueeze(1), torch.transpose(ijoint_batch, dim0=1, dim1=2)).squeeze()
            pdAjointk = new_pdAjointk.clone().detach()
        if get_pd and get_q: return pdAegok,qAego
        elif get_pd:  return pdAegok#.detach().long()  # pdAjointk
        elif get_q: return qAego
        else: raise Exception('Unknown QRE parameter')


    def ego2joint_action(self,aR,aH):
        return self.solo2joint[aR.long(), aH.long()].clone().detach()


    def sample_action(self,obs,epsilon,best=False,agent=2):
        """
        Sophistocation 0: n/a
        Sophistocation 1: I assume you move with uniform probability
        Sophistocation 2: I assume that (you assume I move with uniform probability)
        Sophistocation 3: I assume that [you assume that (I assume you move with uniform probability)]
        :param obs:
        :return:
        """
        # obs = obs - self.pos_offset
        #
        #

        if best: epsilon=0

        iR,iH,iBoth = 0,1,2
        assert agent in [0,1,2], 'unknown agne sampling'

        n_batch = obs.shape[0]
        ak = torch.empty(n_batch)
        pAnotk = torch.empty([n_batch,self.n_ego_actions])
        aJ = torch.zeros([n_batch, 1], device=self.tensor_type['device'], dtype=torch.int64)

        if torch.rand(1) < epsilon:
            if self.ENABLE_DIR_EXPLORE:
                q_undir = torch.rand([1,25])
                q_dir = self.directed_exploration(obs- self.pos_offset)
                pdAJ = torch.softmax(q_dir + q_undir, dim=-1)
                aJ = torch.tensor([list(WeightedRandomSampler(pdAJ.squeeze(), 1, replacement=True))])
            else:
                aJ = torch.randint(0, self.n_act, [n_batch, 1], device = self.tensor_type['device'], dtype=torch.int64)
        else:
            qegok = self.forward(obs)
            # is_zeros = all(torch.all(torch.all(qegok==0,dim=2),dim=0).numpy())
            # if not is_zeros: logging.warning(f'FOUND Q-VALUE qego = {qegok.numpy()}')
            pdAegok,qegok = self.QRE(qegok,get_pd=True,get_q=True)

            for ibatch in range(n_batch):
                if best:
                    aR,aH = [torch.argmax(pdAegok[ibatch, k, :]) for k in range(2)]
                else:
                    aR = list(WeightedRandomSampler(pdAegok[ibatch, 0, :], 1, replacement=True))[0]
                    aH = list(WeightedRandomSampler(pdAegok[ibatch, 1, :], 1, replacement=True))[0]
                aJ[ibatch] = self.solo2joint[aR,aH]
                if agent == iR :
                    ak[ibatch] = aR
                    pAnotk[ibatch,:] = pdAegok[ibatch, int(not(iR)), :]
                elif agent == iH:
                    ak[ibatch] = aH
                    pAnotk[ibatch,:] = pdAegok[ibatch,  int(not(iH)), :]

        if agent == iBoth:
            if best:
                # Qk_max = torch.max(qegok,dim=-1, keepdim=True)[0]
                Qk_exp  = torch.sum(qegok*pdAegok,dim=2,keepdim=True)
                return aJ, Qk_exp
            else: return aJ
        else:  return ak,pAnotk

    def update(self, transitions, ALPHA, GAMMA, LAMBDA):
        # Unpack memory in batches
        # transitions = replay_memory.get(dump=True)
        done_mask = torch.cat(transitions.done)
        state_batch = torch.cat(list(map(lambda s: s.reshape([-1, 6]), transitions.state)), dim=0)
        action_batch = torch.cat(transitions.action).reshape([-1, 1])
        next_state_batch = torch.cat(list(map(lambda s: s.reshape([-1, 6]), transitions.next_state)), dim=0)
        reward_batch = torch.cat(transitions.reward).reshape([-1, 2])
        n_batch = len(transitions.state)

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #reward_batch = torch.pow(reward_batch, 2)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # !!!!!! YOU HAVE TO ITERATE THROUGH REWARDS IF GIVING MORE THAN ONE REWARD AT END OF GAME !!!!
        et_discount = torch.pow(GAMMA * LAMBDA, torch.arange(n_batch).flip(0))
        et_reward_batch = torch.zeros(reward_batch.shape)
        for ibatch in range(n_batch):
            et_reward_batch[ibatch, :] = et_discount[ibatch] * torch.sum(reward_batch[ibatch:, :], dim=0)

        # Compute indicies
        # b_idxs = [torch.zeros(n_batch).long()]
        # k_idxs = [[tuple(k * torch.ones(n_batch, dtype=torch.int)) for k in range(self.n_agents)]]
        # s_idxs = list(zip(*state_batch.int()))
        # a_idxs = list(zip(*action_batch))

        # Compute Q(s_t, a) - the model computes Q(s_t)
        # qSA = self.forward(state_batch, action=action_batch)
        # qSA = torch.zeros([n_batch,self.n_agents])
        # qSA_prime = torch.zeros([n_batch,self.n_agents])
        aJ, qSA_prime_batch = self.sample_action(next_state_batch, epsilon=0, best=True)  #
        qSA_prime_batch[done_mask] = 0

        for k in range(2):
            for ibatch in range(n_batch):
                x0,y0,x1,y1,x2,y2 = state_batch[ibatch].long() - self.pos_offset
                a = action_batch[ibatch]
                # x0p,y0p,x1p,y1p,x2p,y2p  = next_state_batch[ibatch]
                qSA = self.tbl[k,x0,y0,x1,y1,x2,y2,a].squeeze()
                qSA_prime = qSA_prime_batch[ibatch,k].squeeze()
                et = et_reward_batch[ibatch,k].squeeze()
                TD_err = (et + GAMMA * qSA_prime - qSA)
                qSA_new = qSA + (ALPHA * TD_err)
                self.tbl[k,x0,y0,x1,y1,x2,y2,a] = qSA_new


    # def update(self, transitions, ALPHA, GAMMA, LAMBDA):
    #     # Unpack memory in batches
    #     # transitions = replay_memory.get(dump=True)
    #     done_mask = torch.cat(transitions.done)
    #     state_batch = torch.cat(list(map(lambda s: s.reshape([-1, 6]), transitions.state)), dim=0) - self.pos_offset
    #     action_batch = torch.cat(transitions.action).reshape([-1, 1])
    #     next_state_batch = torch.cat(list(map(lambda s: s.reshape([-1, 6]), transitions.next_state)), dim=0)
    #     reward_batch = torch.cat(transitions.reward).reshape([-1, 2])
    #     n_batch = len(transitions.state)
    #
    #     # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #     reward_batch = torch.pow(reward_batch,2) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #     # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #
    #
    #     # !!!!!! YOU HAVE TO ITERATE THROUGH REWARDS IF GIVING MORE THAN ONE REWARD AT END OF GAME !!!!
    #     et_discount = torch.pow(GAMMA * LAMBDA, torch.arange(n_batch).flip(0))
    #     et_reward_batch = torch.zeros(reward_batch.shape)
    #     for ibatch in range(n_batch):
    #         et_reward_batch[ibatch, :] = et_discount[ibatch] * torch.sum(reward_batch[ibatch:, :], dim=0)
    #
    #     # Compute indicies
    #     # b_idxs = [torch.zeros(n_batch).long()]
    #     k_idxs = [[tuple(k * torch.ones(n_batch, dtype=torch.int)) for k in range(self.n_agents)]]
    #     s_idxs = list(zip(*state_batch.int()))
    #     a_idxs = list(zip(*action_batch))
    #
    #     # Compute Q(s_t, a) - the model computes Q(s_t)
    #     # qSA = self.forward(state_batch, action=action_batch)
    #     qSA = self.tbl[k_idxs + s_idxs + a_idxs].T.unsqueeze(-1)
    #
    #     #  Compute V(s_{t+1}) for all next states. (next_state_values)
    #     aJ, qSA_prime = self.sample_action(next_state_batch, epsilon=0, best=True)  #
    #     qSA_prime[done_mask] = 0
    #
    #     # if torch.any(reward_batch > 0):
    #     #     print(f'reward found')
    #
    #     # Compute the expected Q values
    #     TD_err = (et_reward_batch.unsqueeze(-1) + GAMMA*qSA_prime - qSA)
    #     # TD_err = (reward_batch.unsqueeze(-1) + GAMMA * qSA_prime - qSA)
    #     # TD_err = 100*torch.ones(TD_err.shape)
    #     # self.tbl[0] += 1000
    #     self.tbl[k_idxs + s_idxs + a_idxs] = qSA.squeeze().T + (ALPHA * TD_err).squeeze().T


    def simulate(self,env,epsilon):
        observations = []
        with torch.no_grad():
            state = env.reset()  # Initialize the environment and get it's state
            self.state_visitation[list(state.int().numpy().flatten()-self.pos_offset)] +=1
            for t in itertools.count():
                action = self.sample_action(state, epsilon)
                next_state, reward, done, _ = env.step(action.squeeze())
                observations.append([state, action, next_state, reward,torch.tensor([done])])
                self.state_visitation[list(next_state.int().numpy().flatten() - self.pos_offset)] += 1
                if done: break
                state = next_state.clone().detach()
        return observations

    def test_policy(self,env, num_episodes):
        with torch.no_grad():
            length = 0
            psucc = 0
            score = np.zeros(env.n_agents)
            for episode_i in range(num_episodes):
                state = env.reset()
                for t in count():
                    action = self.sample_action(state, epsilon=0)
                    next_state, reward, done, _ = env.step(action.squeeze())
                    score += reward.detach().flatten().cpu().numpy()
                    state = next_state.clone()
                    if done: break
                if env.check_caught(env.current_positions): psucc += 1
                length += env.step_count

        final_score = list(score / num_episodes)
        final_length = length / num_episodes
        final_psucc = psucc / num_episodes

        return final_score, final_length, final_psucc


    def check_obs(self,obs):
        in_bnds = (torch.all(obs >= 0) and torch.all(obs <= 4))
        in_wall = any([torch.any(torch.all(self.walls==pk,dim=1)) for pk in obs.reshape([3, 2])])
        is_adm = (in_bnds and not in_wall)
        return is_adm
    def directed_exploration(self,obs):
        q_inadmissable = -1e3
        obs = obs.reshape([-1, 6])
        n_batch = obs.shape[0]
        pos = obs.clone().detach().squeeze()
        qEXP = torch.zeros([1,self.n_act])
        #k_idxs = [[tuple(k * torch.ones(n_batch, dtype=torch.int)) for k in range(self.n_agents)]]

        for aJ in range(self.n_act):
            aR,aH = self.joint2solo[aJ]
            next_pos = pos.clone().detach()
            next_pos[0:2] = self.move(aR, pos[0:2])
            next_pos[2:4] = self.move(aH, pos[2:4])
            is_adm = self.check_obs(next_pos)

            # in_bnds = (torch.all(next_pos >= 0) and torch.all(next_pos <= 4))
            # in_wall = any([torch.any(torch.all(self.walls==pk,dim=1)) for pk in next_pos.reshape([3, 2])])
            # is_adm = (in_bnds and not in_wall)
            # print(f'{aJ}:adm: {is_adm} bnd: {in_bnds} wall: {in_wall} pos {next_pos}')

            if is_adm:
                next_pos = next_pos.reshape([-1, 6])
                s_idxs = list(zip(*next_pos.int()))
                discount = 1/torch.exp(self.state_visitation[s_idxs])
                dist2prey = torch.zeros(2)
                next_pos = next_pos.reshape([3, 2])
                dist2prey[0] = torch.dist(next_pos[0], next_pos[2])
                dist2prey[1] = torch.dist(next_pos[1], next_pos[2])
                q_dist = torch.pow((self.max_dist  - dist2prey.mean()) / self.max_dist ,1)
                qEXP[0,aJ] = (discount * self.DIR_EXPLORATION) * q_dist
                # print(f'{aJ}:adm: {is_adm} dist{list(dist2prey.numpy())} q{(discount * self.DIR_EXPLORATION) * q_dist} pos {next_pos}')
            else: qEXP[0, aJ] = q_inadmissable

        return qEXP
    def move(self, ego_action, curr_pos):
        if isinstance(ego_action, torch.Tensor): ego_action = int(ego_action)
        assert ego_action in range(self.n_ego_actions), 'Unknown ego action in env.move()'
        action_name = self.idx2action[int(ego_action)]
        next_pos = curr_pos + self.explicit_actions[action_name]
        return next_pos



class ReplayMemory(object):

    def __init__(self, capacity=20):
        self.n_agents = 2
        self.capacity = capacity
        self.memory = deque([],maxlen=self.n_agents*capacity)

    def push(self, *args):
        """Save a transition"""
        # self.memory.append(Transition(*[list(arg.numpy()) for arg in args]))
        self.memory.append(Transition(*args))

    def get(self,dump=False):
        transitions = list(self.memory)
        batch = Transition(*zip(*transitions))
        if dump:  self.memory = deque([],maxlen=self.capacity)
        return batch

    def __len__(self):
        return len(self.memory)

# def main():
#     device, dtype, sophistocation = 'cpu', torch.float32, 5
#     Q = Qfunction(device, dtype, sophistocation)
#     n_agents = 2
#     n_act = 25
#     nxy = 5
#     n_mem = 2
#     tbl = torch.zeros([n_agents, nxy, nxy, nxy, nxy, nxy, nxy, n_act])
#     aJ = 1
#     xy = 1
#
#     batch1 = [1, 1, 1, 1, 1, 1, 1, 1]
#     batch2 = [1, 1, 1, 1, 1, 1, 1, 2]
#     tbl[0, 1, 1, 1, 1, 1, 1, 1] = 2
#     tbl[1, 1, 1, 1, 1, 1, 1, 1] = 10
#     rewards = torch.tensor([1, 1])
#     action = torch.tensor([aJ])
#     state = torch.tensor([xy, xy, xy, xy, xy, xy],dtype=torch.int)
#     replay_memory = ReplayMemory()
#     next_state = 2 * torch.ones(state.shape)
#     done = torch.tensor([False])
#     for i in range(n_mem):
#         replay_memory.push(state, action, next_state, rewards, done)
#     replay_memory.push(state, action, next_state, rewards, torch.tensor([True]))
#
#
#     Q.train(memory.get(dump=True),ALPHA=0.1, GAMMA=0.99,LAMBDA=0.7)
#     # transitions = replay_memory.get()
#     # n_batch = len(transitions.state)
#     # k_idxs = [[tuple(k * torch.ones(n_batch, dtype=torch.int)) for k in range(n_agents)]]
#     # s_idxs = list(zip(*transitions.state))
#     #qAk = torch.transpose(tbl[k_idxs + s_idxs], dim0=0, dim1=1)
#
#     #qSA = Q(state_batch, action=action_batch)  # ,state_action_values
#
#
#
#
#
# if __name__ == "__main__":
#     main()