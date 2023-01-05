############################################################
## Packages ################################################
import logging
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import itertools
from torch.utils.data.sampler import WeightedRandomSampler
from utilities.make_env import PursuitEvastionGame

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


#################################################################
## DQN algorithm ################################################
class DQN(nn.Module):

    @classmethod
    def construct_path(_, iWorld, policy_type, algorithm):
        fname = f'{algorithm}_{policy_type}.torch'
        project_dir = os.getcwd().split('MARL')[0] + 'MARL\\'
        file_path = project_dir + f'results\\IDQN_W{iWorld}\\{fname}'
        return file_path

    @classmethod
    def load(_, iWorld, policy_type, algorithm, verbose=True):

        try:
            file_path = DQN.construct_path(iWorld, policy_type, algorithm)
            q = torch.load(file_path)
        except:
            logging.warning('Inverted policy and algorithm name')
            file_path = DQN.construct_path(iWorld, algorithm, policy_type)
            q = torch.load(file_path)
        file_size = os.path.getsize(file_path)
        if verbose: print(
            f'\nLOADED EXISTING QNet \n\t| Path: {file_path} \n\t| Size: {file_size} \n\t| Device: {q.tensor_type["device"]}')
        return q

    @classmethod
    def save(_, module, iWorld, algorithm, policy_type, axes = None,verbose=True):
        if axes is not None: module.axes = axes #copy.deepcopy(axes)
        file_path = DQN.construct_path(iWorld, policy_type, algorithm)
        torch.save(module, file_path)
        file_size = os.path.getsize(file_path)
        if verbose: print(f'\nQNet SAVED \n\t| Path: {file_path} \n\t| Size: {round(file_size / 1000)}MB')

    @classmethod
    def preview(_, iWorld, policy_type, algorithm):
        q = DQN.load(iWorld, policy_type, algorithm)
        assert q.axes is not None, 'tried previewing DQN figure that does not exist'
        plt.ioff()
        plt.show()

        # if verbose: print(
        #     f'\nLOADED EXISTING QNet \n\t| Path: {file_path} \n\t| Size: {file_size} \n\t| Device: {q.tensor_type["device"]}')
        # return q

    def __init__(self,device,dtype,sophistocation,run_config=None,rationality=5):
        super(DQN, self).__init__()
        self.n_act = 25
        self.n_ego_actions = 5
        self.n_obs = 6
        self.n_agents = 2
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
        self.nnk = {}
        for k in range(self.n_agents):
            setattr(self, 'agent_{}'.format(k),
                    nn.Sequential(nn.Linear(self.n_obs, 128), nn.ReLU(),
                                  nn.Linear(128, 128), nn.ReLU(),
                                  nn.Linear(128, self.n_act))
                    )

            for ak in range(5):
                idxs = np.array(np.where(joint2solo[:, k] == ak)).flatten()
                ijoint[k, ak, idxs] = 1


        self.ijoint = torch.as_tensor(ijoint,**self.tensor_type)
        self.solo2joint = torch.as_tensor(solo2joint,**self.tensor_type)
        self.joint2solo = torch.as_tensor(joint2solo, **self.tensor_type)

        self.env = None

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, obs):
        q_values = [getattr(self, 'agent_{}'.format(k))(obs).unsqueeze(1) for k in range(self.n_agents)]
        return torch.cat(q_values, dim=1)

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


    #################################################################################
    # WORKING #######################################################################
    # """ !!!!!!!!!!!!!!!!!!!!!!!! DO NOT TOUCH !!!!!!!!!!!!!!!!!!!!!! """
    # def QRE(self,qAk):
    #     sophistocation = self.sophistocation
    #     n_agents = self.n_agents
    #     n_joint_act = self.n_act
    #     n_ego_act = 5
    #     rationality = self.rationality
    #     n_batch = qAk.shape[0]
    #
    #     Ktracker = [1, 0] if sophistocation % 2 == 0 else [0, 1]
    #     pdAjointk = torch.ones([n_batch, n_agents, n_joint_act], **self.tensor_type) / n_joint_act
    #
    #     qAego = torch.empty([n_batch, n_agents, n_ego_act], **self.tensor_type)
    #     pdAegok = torch.empty([n_batch, n_agents, n_ego_act], **self.tensor_type)
    #     for isoph in range(sophistocation):
    #         new_pdAjointk = torch.zeros([n_batch, n_agents, n_joint_act])
    #         for k in range(n_agents):
    #             ijoint_batch = self.ijoint[Ktracker[k], :, :].T.repeat([n_batch, 1, 1])
    #             notk = int(not k)
    #
    #             qAJ_conditioned = qAk[:, k, :] * pdAjointk[:, notk, :]
    #             qAego[:, k, :] = torch.bmm(qAJ_conditioned.unsqueeze(1), ijoint_batch).squeeze()
    #             pdAegok[:, k, :] = torch.special.softmax(rationality * qAego[:, k, :], dim=-1)
    #             new_pdAjointk[:, k, :] = torch.bmm(pdAegok[:, k, :].unsqueeze(1), torch.transpose(ijoint_batch, dim0=1, dim1=2)).squeeze()
    #         pdAjointk = new_pdAjointk.clone().detach()
    #         Ktracker.reverse()
    #         pdAjointk = pdAjointk[:, Ktracker, :]
    #     return pdAegok#.detach().long()  # pdAjointk
    ####################################################################################



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
        if best: epsilon=0

        iR,iH,iBoth = 0,1,2
        assert agent in [0,1,2], 'unknown agne sampling'

        n_batch = obs.shape[0]
        ak = torch.empty(n_batch)
        pAnotk = torch.empty([n_batch,self.n_ego_actions])
        aJ = torch.zeros([n_batch, 1], device=self.tensor_type['device'], dtype=torch.int64)

        if torch.rand(1) < epsilon:
            aJ = torch.randint(0, self.n_act, [n_batch, 1], device = self.tensor_type['device'], dtype=torch.int64)
        else:
            with torch.no_grad(): # <=== CAUSED MEMORY ERROR WITHOUT ===
                pdAegok,qegok = self.QRE(self.forward(obs),get_pd=True,get_q=True)

                for ibatch in range(n_batch):
                    if best:
                        aR = torch.argmax(pdAegok[ibatch, 0, :])
                        aH = torch.argmax(pdAegok[ibatch, 1, :])
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

    def simulate(self,env,epsilon):
        # if self.env == None:
        #     self.env=env
        #     # iWorld = self.run_config['iWorld']
        #     # self.env = PursuitEvastionGame(iWorld, **self.tensor_type)

        observations = []
        with torch.no_grad():
            state = env.reset()  # Initialize the environment and get it's state
            for t in itertools.count():
                action = self.sample_action(state, epsilon)
                next_state, reward, done, _ = env.step(action.squeeze())
                if done: observations.append([state, action, None, reward]); break
                else: observations.append([state, action, next_state, reward])
                state = next_state.clone().detach()

        return observations

