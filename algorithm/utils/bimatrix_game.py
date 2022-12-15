import numpy as np
# import matplotlib.pyplot as plt



class JointQ_Handler():
    def __init__(self,state0,, DMk,ToMk=(1,1)):
    # def __init__(self, Vk, state_observations):

        self.ACTION_MEANING = {0: "DOWN", 1: "LEFT", 2: "UP",  3: "RIGHT",  4: "NOOP",}

        self.n_agents = 2
        self.n_actions = len(self.ACTION_MEANING.keys())
        self.n_joint_actions = self.n_actions*self.n_actions
        self.grid_sz = 5
        self.n_observations = 6
        self.pda_UNIFORM = np.ones(self.n_actions)/self.n_actions
        self.DMk = DMk

        _sz = [self.n_agents]
        _sz += [self.grid_sz for _ in range(self.n_observations)]
        _sz +=  [self.n_joint_actions]
        # _sz += [self.n_actions for _ in range(self.n_agents)]
        self.Qk = np.zeros(_sz)

        self.discount_factor = 0.8
        self.learning_rate = 0.005

        # qk = [self.get_controllable_Qs(k=k,state=state0,DMk=DMk,ToM = ToMk[k]) for k in range(self.n_agents)]


    def updateQ(self,k,state0,joint_action0,state1,reward1_k):
        alpha = self.learning_rate
        gamma = self.discount_factor
        Q_s0 = self.Qk[k][state0][joint_action0]
        a1_best = self.sample_action(self.Qk[k][state1],best=True)
        Qmax_s1 = self.Qk[k][state1][a1_best]

        TD_target = reward1_k[k] + gamma * Qmax_s1  # RQ[sj][ajR]
        TD_err = TD_target - Q_s0  # RQ[si][aiR]
        qnew = Q_s0 + alpha * TD_err

        self.Qk[k][state0][joint_action0] = qnew


    def get_controllable_Qs(self,k,state,DMk,ToM=0):
        notk = int(not k)
        if ToM==0:
            q_k = self.get_TomQs(k=k, state=state, pda_notk=self.pda_UNIFORM)
        elif ToM==1:
            q_notk = self.get_TomQs(k=notk, state=state, pda_notk=self.pda_UNIFORM)
            q_k = self.get_TomQs(k=k, state=state, pda_notk=DMk[notk](q_notk))
        elif ToM == 2:
            q_k = self.get_TomQs(k=k, state=state, pda_notk=self.pda_UNIFORM)
            q_notk = self.get_TomQs(k=notk, state=state, pda_notk=DMk[k](q_k))
            q_k = self.get_TomQs(k=k, state=state, pda_notk=DMk[notk](q_notk))
        return q_k

    def get_TomQs(self,k,state,pda_notk=None):
        Qs =  self.Qk[k][state]
        if k==0:
            pd_anotk = np.array(pda_notk).reshape([1, -1])
            Qs_cntrl= [np.sum(Qs[ak, :] * pd_anotk) for ak in range(self.n_actions)]
        else:
            pd_anotk = np.array(pda_notk).reshape([-1, 1])
            Qs_cntrl = [np.sum(Qs[:,ak] * pd_anotk) for ak in range(self.n_actions)]
        return Qs_cntrl
    def joint_transition(self,state,ak0,ak1):
        new_state = np.copy(np.array(state))
        new_state[0, :] += self.action_idx2move(ak0)
        new_state[1, :] += self.action_idx2move(ak1)
        return new_state.flatten()
    def action_idx2move(self,a):
        if a == 0:   move = [1,0] # down
        elif a == 1: move = [0, - 1] # left
        elif a == 2: move = [ -1, 0] # up
        elif a == 3: move = [0, 1]  # right
        elif a == 4: move = [0,0] # no-op
        else: raise Exception('invalid move index')
        return np.array(move)
#
#
# class BiMatrixGame():
#     def __init__(self,state0,Qk, DMk,ToMk=(1,1)):
#     # def __init__(self, Vk, state_observations):
#         self.ACTION_MEANING = {0: "DOWN", 1: "LEFT", 2: "UP",  3: "RIGHT",  4: "NOOP",}
#         self.Qk = np.copy(Qk)
#         self.n_agents = len(Qk)
#         self.n_actions = len(self.ACTION_MEANING.keys())
#         self._shape = [self.n_agents]+[self.n_actions for _ in range(self.n_agents)]
#         self.grid_sz = 5
#         self.pda_UNIFORM = np.ones(self.n_actions)/self.n_actions
#         self.DMk = DMk
#
#         self.G = np.empty(self._shape)
#         #Vk = Q[k,x0,y0,x1,y1,x2,y2,am0,am1_hat] * p_am1_hat
#
#
#         qk = [self.get_ToMQ(k=k,state=state0,DMk=DMk,ToM = ToMk[k]) for k in range(self.n_agents)]
#
#
#
#         k = 0
#         ak0 = 0
#         ak1 = 0
#         # x0,y0,x1,y1,x2,y2 = self.joint_transition(state0,ak0,ak1)
#         # self.G[k,ak0,ak1] = Vk[k,x0,y0,x1,y1,x2,y2]
#
#         # self.G[k, ak0, ak1]
#
#     def get_ToMQs(self,k,state,DMk,ToM=0):
#         notk = int(not k)
#         if ToM==0:
#             q_k = self.get_controllableQ(k=k, state=state, pda_notk=self.pda_UNIFORM)
#         elif ToM==1:
#             q_notk = self.get_controllableQ(k=notk, state=state, pda_notk=self.pda_UNIFORM)
#             q_k = self.get_controllableQ(k=k, state=state, pda_notk=DMk[notk](q_notk))
#         elif ToM == 2:
#             q_k = self.get_controllableQ(k=k, state=state, pda_notk=self.pda_UNIFORM)
#             q_notk = self.get_controllableQ(k=notk, state=state, pda_notk=DMk[k](q_k))
#             q_k = self.get_controllableQ(k=k, state=state, pda_notk=DMk[notk](q_notk))
#         return q_k
#
#     def get_controllableQ(self,k,state,pda_notk=None):
#         Qs =  self.Qk[k][state]
#         if k==0:
#             pd_anotk = np.array(pda_notk).reshape([1, -1])
#             Qs_cntrl= [np.sum(Qs[ak, :] * pd_anotk) for ak in range(self.n_actions)]
#         else:
#             pd_anotk = np.array(pda_notk).reshape([-1, 1])
#             Qs_cntrl = [np.sum(Qs[:,ak] * pd_anotk) for ak in range(self.n_actions)]
#         return Qs_cntrl
#     def joint_transition(self,state,ak0,ak1):
#         new_state = np.copy(np.array(state))
#         new_state[0, :] += self.action_idx2move(ak0)
#         new_state[1, :] += self.action_idx2move(ak1)
#         return new_state.flatten()
#     def action_idx2move(self,a):
#         if a == 0:   move = [1,0] # down
#         elif a == 1: move = [0, - 1] # left
#         elif a == 2: move = [ -1, 0] # up
#         elif a == 3: move = [0, 1]  # right
#         elif a == 4: move = [0,0] # no-op
#         else: raise Exception('invalid move index')
#         return np.array(move)



if __name__ == "__main__":
    main()
