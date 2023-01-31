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
from collections import namedtuple, deque
from utilities.config_manager import CFG
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
        plt.close(q.axes)
        if'n_egoA' not in q.__dict__.keys():
            q.n_egoA = q.n_ego_actions
        if 'n_jointA' not in q.__dict__.keys():
            q.n_jointA = q.n_act

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

    def __call__(self, *args, **kwargs):
        return self.Qindex( *args, **kwargs)


    def __init__(self,device,dtype,sophistocation,run_config=None,rationality=1):
        #super(DQN, self).__init__()

        # this_CFG = run_config if run_config is not None else copy.deepcopy(CFG)
        this_CFG = run_config
        self.n_jointA = this_CFG.n_jointA
        self.n_egoA = this_CFG.n_egoA
        self.n_obs = this_CFG.n_obs
        self.n_agents = this_CFG.n_agents
        self.rationality = this_CFG.rationality
        self.ToM = this_CFG.ToM
        self.tensor_type = {'device': this_CFG.device, 'dtype': this_CFG.dtype}
        self.run_config = this_CFG
        self.env = None
        nxy = this_CFG.grid_sz
        self.nxy = nxy
        # Env definition
        # nxy = CFG.nxy
        # self.n_jointA = 25
        # self.n_egoA = 5
        # self.n_obs = 6
        # self.n_agents = 2
        # self.axes = None
        # self.env = None
        # self.nxy = nxy

        self.rationality = rationality
        self.sophistocation = sophistocation
        self.tensor_type = {'device': device, 'dtype': dtype}
        self.run_config = run_config

        self.ijoint, self.solo2joint, self.joint2solo, \
        self.action2idx, self.idx2action, self.explicit_actions \
            = self.init_conversion_mats()


        # Learning Config
        self.reward_power = CFG.reward_power
        self.DIR_EXPLORATION = CFG.DIR_EXPLORATION
        self.ENABLE_DIR_EXPLORE = CFG.ENABLE_DIR_EXPLORE
        self.ENABLE_INIT_Q = CFG.ENABLE_INIT_Q
        self.q_inadmissable = -1e4
        self.tbl = torch.zeros([self.n_agents, nxy, nxy, nxy, nxy, nxy, nxy, self.n_jointA])
        self.state_visitation = torch.zeros([nxy, nxy, nxy, nxy, nxy, nxy])
        self.max_dist =  torch.dist(torch.tensor([0.,0.]), torch.tensor([4.,4.]))
        self.walls, self.pos_offset = self.init_walls()
        if self.ENABLE_INIT_Q: self.init_bad_Q()

        # self.tbl[1] =1
        # s=torch.tensor([[1, 1, 1, 1, 1, 1],[2, 2, 2, 2, 2, 2],[2, 2, 2, 2, 2, 2]])
        # a = torch.tensor([0,1,2])
        # k = torch.tensor([0, 1, 1])
        # if self.ENABLE_DIR_EXPLORE: logging.warning('Directed Exploration enabled')
        # if self.rationality!=1: logging.warning(f'Rationality ={self.rationality}')


        self.check_defaults()
    def init_walls(self):
        nxy =  self.nxy

        walls = torch.tensor([[2, 2], [2, 4], [4, 2], [4, 4]])

        if nxy == 7:
            pos_offset = 0
            new_walls = []
            for r in range(nxy):
                for c in [0, nxy - 1]:
                    new_walls.append([r, c])
            for c in range(nxy):
                for r in [0, nxy - 1]:
                    new_walls.append([r, c])
            walls = torch.cat([walls, torch.tensor(new_walls)], dim=0)
        elif nxy == 5:
            pos_offset = 1
            walls -= pos_offset
        return walls,pos_offset
    def init_conversion_mats(self):
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
        ijoint = torch.as_tensor(ijoint, **self.tensor_type)
        solo2joint = torch.as_tensor(solo2joint, **self.tensor_type)
        joint2solo = torch.as_tensor(joint2solo, **self.tensor_type)

        action2idx = {'down': 0, 'left': 1, 'up': 2, 'right': 3, 'wait': 4}
        idx2action = {v: k for k, v in action2idx.items()}
        explicit_actions = {'down': torch.tensor([1, 0], **self.tensor_type),
                                 'left': torch.tensor([0, -1], **self.tensor_type),
                                 'up': torch.tensor([-1, 0], **self.tensor_type),
                                 'right': torch.tensor([0, -1], **self.tensor_type),
                                 'wait': torch.tensor([0, 0], **self.tensor_type)}
        return ijoint,solo2joint,joint2solo, action2idx, idx2action, explicit_actions
    def check_defaults(self):
        checks = {}
        checks['reward_power'] = 1
        checks['nxy'] = 5
        checks['ENABLE_INIT_Q'] = False
        checks['ENABLE_DIR_EXPLORE'] = False

        for key in checks.keys():
            val = checks[key]
            if self.__dict__[key] != val:
                logging.warning(f'[QFunction] - Non-Default Param {key}={val} should be {self.__dict__[key]}')
    def init_bad_Q(self):
        logging.warning('Initializing Q with walls')
        for wall_state in self.walls:
            for action in self.explicit_actions.values():
                new_state = wall_state.detach() - action
                x, y = new_state.long()
                if (x < self.nxy and y<self.nxy) and (x >= 0 and y >= 0):
                    self.tbl[:, x, y, :, :, :, :, :] = self.q_inadmissable
                    self.tbl[:, :, :, x, y, :, :, :] = self.q_inadmissable
                    self.tbl[:, :, :, :, :, x, y, :] = self.q_inadmissable


    def Qindex(self,s,*args,a=None,get_idx=False,**kwargs): #k=None
        """
        :param args: (state,action)_ibatch
        :param kwargs: named state or action batches
        :return:  tbl_slice \in [n_batch,n_agent,{state,action}]
        """

        if len(args) == 1: a = args[0]
        elif 'a' in kwargs.keys(): a = kwargs['a']
        elif 'action' in kwargs.keys(): a = kwargs['action']

        if s is None:   x0, y0, x1, y1, x2, y2 = [slice(None) for _  in range(self.n_obs)]
        else:           x0, y0, x1, y1, x2, y2 = list(zip(*s.reshape([-1, self.n_obs]).to(int)))
        if a is None:   a,n_a = slice(None),self.n_jointA
        else:           a,n_a =  a.flatten(),1#a.reshape([-1, 1])
        # K = tuple([ [[0,1]] for _ in range(s.shape[0])])
        # # n_a = self.n_jointA if a is None else 1
        # index = [K, x0, y0, x1, y1, x2, y2, a]
        index = [slice(None), x0, y0, x1, y1, x2, y2, a]
        tbl_slice = self.tbl[index]  # tbl_slice = self.tbl[:, x0, y0, x1, y1, x2, y2, a]
        tbl_slice = torch.transpose(tbl_slice,dim0=0,dim1=1)
        tbl_slice = tbl_slice.reshape([tbl_slice.shape[0],self.n_agents,n_a])
        if get_idx: return tbl_slice,index
        else: return tbl_slice



    def QRE(self,qAk,get_pd = True,get_q=False):
        sophistocation = self.sophistocation
        n_agents = self.n_agents
        n_joint_act = self.n_jointA
        n_egaA = 5
        rationality = self.rationality
        n_batch = qAk.shape[0]

        pdAjointk = torch.ones([n_batch, n_agents, n_joint_act], **self.tensor_type) / n_joint_act
        qAego = torch.empty([n_batch, n_agents, n_egaA], **self.tensor_type)
        pdAegok = torch.empty([n_batch, n_agents, n_egaA], **self.tensor_type)
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


    def sample_sim_action(self,obs,agent):
        """
        Sophistocation 0: n/a
        Sophistocation 1: I assume you move with uniform probability
        Sophistocation 2: I assume that (you assume I move with uniform probability)
        Sophistocation 3: I assume that [you assume that (I assume you move with uniform probability)]
        :param obs:
        :return:
        """

        iR,iH = 0,1
        assert agent in [0,1], 'unknown agne sampling'
        n_batch = obs.shape[0]
        ak_batch = torch.empty(n_batch)
        pAnotk = torch.empty([n_batch,self.n_egoA])
        qegok = self(obs)
        pdAegok,qegok = self.QRE(qegok,get_pd=True,get_q=True)

        for ibatch in range(n_batch):
            if agent == iR: ak = torch.argmax(pdAegok[ibatch, iR, :])
            else: ak = list(WeightedRandomSampler(pdAegok[ibatch, iH, :], 1, replacement=True))[0]
            ak_batch[ibatch] = ak
            pAnotk[ibatch, :] = pdAegok[ibatch, int(not (agent)), :]
        return ak_batch, pAnotk

    def sample_best_action(self,obs,agent=2):
        """
        pAnotk: probability of partner (-k) actions given k's MM of -k
        Qk_exp: expected quality of k controllable action
        """
        iR, iH, iBoth = 0, 1, 2
        if agent == 0: kslice = 0
        elif agent == 1: kslice = 1
        elif agent == 2: kslice = slice(0,2)
        else: raise Exception('unknown agent sampling')

        n_batch = obs.shape[0]
        ak = torch.empty([n_batch, self.n_agents], dtype=torch.int64)
        aJ = torch.zeros([n_batch, 1], device=self.tensor_type['device'], dtype=torch.int64)
        Qk_exp = torch.zeros([n_batch, self.n_agents, 1], **self.tensor_type)
        pAnotk = torch.empty([n_batch, self.n_agents, self.n_egoA], **self.tensor_type)

        with torch.no_grad():  # <=== CAUSED MEMORY ERROR WITHOUT ===
            qAjointk = self(obs)  # agent k's quality over joint actions
            pdAegok, qegok = self.QRE(qAjointk, get_pd=True, get_q=True)

            for ibatch in range(n_batch):
                # Each agen chooses the best controllable action => aJ = a_k X a_-k
                # for themselves conditioned on partner action
                aR = torch.argmax(pdAegok[ibatch, iR, :])
                aH = torch.argmax(pdAegok[ibatch, iH, :])
                aJ[ibatch] = self.solo2joint[aR, aH] # lookup table

                # Store batch stats
                ak[ibatch,iR] = aR
                pAnotk[ibatch,iR,:] = pdAegok[ibatch, int(not (iR)), :]
                Qk_exp[ibatch, iR ] = torch.sum(qegok[ibatch, iR] * pdAegok[ibatch, iR, :])

                ak[ibatch,iH] = aH
                pAnotk[ibatch, iH,:] = pdAegok[ibatch, int(not (iH)), :]
                Qk_exp[ibatch, iH ] = torch.sum(qegok[ibatch, iH] * pdAegok[ibatch, iH, :])

                # Assume perfect coordination (Pareto) (!! UNTESTED !!!)
                # aJ = torch.argmax(torch.mean(qAjointk[ibatch,:,:],dim=1))
                # aR,aH = self.joint2solo[aJ] # lookup table

        return aJ[:,kslice], Qk_exp[:,kslice]

    def sample_action(self, obs, epsilon, agent=2):
        """
        (ToM) Sophistocation 0: n/a
        (ToM) Sophistocation 1: I assume you move with uniform probability
        (ToM) Sophistocation 2: I assume that (you assume I move with uniform probability)
        (ToM) Sophistocation 3: I assume that [you assume that (I assume you move with uniform probability)]
        :param obs:
        :return:
        """
        iR, iH, iBoth = 0, 1, 2
        if agent == 0: kslice = 0
        elif agent == 1: kslice = 1
        elif agent == 2: kslice = slice(0, 2)
        else: raise Exception('unknown agent sampling')

        n_batch = obs.shape[0]
        ak = torch.empty([n_batch, self.n_agents], dtype=torch.int64)
        aJ = torch.zeros([n_batch, 1], device=self.tensor_type['device'], dtype=torch.int64)
        Qk_exp = torch.zeros([n_batch, self.n_agents], **self.tensor_type)
        pAnotk = torch.empty([n_batch, self.n_agents, self.n_egoA], **self.tensor_type)

        if torch.rand(1) < epsilon:
            aJ = torch.randint(0, self.n_jointA, [n_batch, 1], device=self.tensor_type['device'], dtype=torch.int64)
        else:
            with torch.no_grad():  # <=== CAUSED MEMORY ERROR WITHOUT ===
                qAjointk = self(obs)  # agent k's quality over joint actions
                pdAegok, qegok = self.QRE(qAjointk, get_pd=True, get_q=True)

                for ibatch in range(n_batch):
                    # Noisy rational sample both agent actions
                    aR = list(WeightedRandomSampler(pdAegok[ibatch, 0, :], 1, replacement=True))[0]
                    aH = list(WeightedRandomSampler(pdAegok[ibatch, 1, :], 1, replacement=True))[0]
                    aJ[ibatch] = self.solo2joint[aR, aH]

                    # Store batch stats
                    ak[ibatch, iR] = aR
                    pAnotk[ibatch, iR, :] = pdAegok[ibatch, int(not (iR)), :]
                    Qk_exp[ibatch, iR] = torch.sum(qegok[ibatch, iR] * pdAegok[ibatch, iR, :])

                    ak[ibatch, iH] = aH
                    pAnotk[ibatch, iH, :] = pdAegok[ibatch, int(not (iH)), :]
                    Qk_exp[ibatch, iH] = torch.sum(qegok[ibatch, iH] * pdAegok[ibatch, iH, :])

        if agent == iBoth: return aJ
        else:  return ak[:, kslice], pAnotk[:, kslice]


    def update(self, transitions, ALPHA, GAMMA, LAMBDA,FORGET_FACTOR=1):
        # Unpack memory in batches
        # transitions = replay_memory.get(dump=True)
        done_mask = torch.cat(transitions.done)
        state_batch = torch.cat(list(map(lambda s: s.reshape([-1, 6]), transitions.state)), dim=0)
        action_batch = torch.cat(transitions.action).reshape([-1, 1])
        next_state_batch = torch.cat(list(map(lambda s: s.reshape([-1, 6]), transitions.next_state)), dim=0)
        reward_batch = torch.cat(transitions.reward).reshape([-1, 2])
        n_batch = len(transitions.state)

        sign = np.sign(reward_batch) if self.reward_power %2 ==0 else 1
        reward_batch = sign * torch.pow(reward_batch, self.reward_power)

        # !!!!!! YOU HAVE TO ITERATE THROUGH REWARDS IF GIVING MORE THAN ONE REWARD AT END OF GAME !!!!
        et_discount = torch.pow(GAMMA * LAMBDA, torch.arange(n_batch).flip(0))
        et_reward_batch = torch.zeros(reward_batch.shape)
        for ibatch in range(n_batch):
            et_reward_batch[ibatch, :] = et_discount[ibatch] * torch.sum(reward_batch[ibatch:, :], dim=0)


        # for k in range(2):
        #     for ibatch in range(n_batch):
        #         x0,y0,x1,y1,x2,y2 = state_batch[ibatch].long() - self.pos_offset
        #         a = action_batch[ibatch]
        #         qSA = self.tbl[k,x0,y0,x1,y1,x2,y2,a].squeeze()
        #         qSA_prime = qSA_prime_batch[ibatch,k].squeeze()
        #         et = et_reward_batch[ibatch,k].squeeze()
        #
        #         TD_err = (et + GAMMA * qSA_prime - qSA)
        #         qSA_new = (FORGET_FACTOR) * qSA + (ALPHA * TD_err)
        #         self.tbl[k,x0,y0,x1,y1,x2,y2,a] = qSA_new

        qSA, index = self.Qindex(state_batch, action_batch, get_idx=True)
        # _, qSA_prime = self.sample_action(next_state_batch, epsilon=0, best=True)  #
        _, qSA_prime = self.sample_best_action(next_state_batch)  #

        qSA_prime[done_mask] = 0


        TD_err = (et_reward_batch.unsqueeze(-1) + GAMMA * qSA_prime - qSA)
        qSA_new = (FORGET_FACTOR) * qSA + (ALPHA * TD_err)

        # self.tbl[index] = qSA_new.squeeze().T
        self.tbl[index] = torch.transpose(qSA_new, dim0=0, dim1=1).reshape(self.tbl[index].shape)



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
        qEXP = torch.zeros([1,self.n_jointA])
        #k_idxs = [[tuple(k * torch.ones(n_batch, dtype=torch.int)) for k in range(self.n_agents)]]

        for aJ in range(self.n_jointA):
            aR,aH = self.joint2solo[aJ]
            next_pos = pos.clone().detach()
            next_pos[0:2] = self.move(aR, pos[0:2])
            next_pos[2:4] = self.move(aH, pos[2:4])
            is_adm = self.check_obs(next_pos)

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
        assert ego_action in range(self.n_egoA), 'Unknown ego action in env.move()'
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

