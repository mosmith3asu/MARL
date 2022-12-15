import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
from enviorment.make_worlds import WorldDefs
import itertools

class MarkovDecisionProcess(object):
    def __init__(self,iWorld):

        self.n_agents = 2
        self.p_pen = 0.5
        self.r_pen = -3
        self.r_catch = 20
        self.n_actions = 5
        self.n_joint_actions =  self.n_actions* self.n_actions
        self.world = WorldDefs.world[iWorld]

        self.States =  self.init_states()
        self.Actions,self.JointActions = self.init_actions()
        self.Rewards = self.init_rewards()



    def init_states(self):
        nx, ny, ipen = 5, 5, 2
        return np.zeros([nx, ny, ipen, nx, ny, ipen, nx, ny], dtype=int)
    def init_actions(self):
        iact0, iact1 = 5, 5
        Actions = np.zeros([iact0, iact1, self.n_agents],dtype=int)
        JointActions = np.zeros(self.n_joint_actions, dtype=object)
        ijoint = 0
        for ia0 in range(self.n_actions):
            for ia1 in range(self.n_actions):
                Actions[ia0, ia1 ] = ijoint
                JointActions[ijoint] = np.array([ia0, ia1])
                ijoint += 1
        return Actions,JointActions

    def init_rewards(self):
        print(f'\t| Initializing rewards...')
        xy = np.arange(5)
        agent0,agent1 = 0,1
        in_pen = 1
        i = 0

        #penalties =np.array( self.world.penalty_states)
        walls = np.array(self.world.walls)
        States = itertools.product(*[xy for _ in range(6)])
        Rewards = np.zeros(list(self.States.shape) + [self.n_agents])
        for x0, y0, x1, y1, x2, y2 in States:
            loc0 = np.array(x0, x0)
            loc1 = np.array(x1, y1)
            loc2 = np.array(x2, y2)

            # If caught get catch reward
            d1 = np.sum(np.abs(loc0-loc2))
            d2 = np.sum(np.abs(loc1-loc2))
            if d1 <=1 and d2 <=1: Rewards[x0, x0,:, x1, y1,:, x2, y2,:] = self.r_catch

            # Add arbitrary penalties to state
            Rewards[x0, x0, in_pen, x1, y1, :, x2, y2, agent0] += self.p_pen * self.r_pen
            Rewards[x0, x0, :, x1, y1, in_pen, x2, y2, agent1] += self.p_pen * self.r_pen
            # in_pen0 = int(any(np.all(np.array(loc0) == penalties, axis=1)))
            # in_pen1 = int(any(np.all(np.array(loc1) == penalties, axis=1)))
            # if in_pen0: Rewards[x0, x0, in_pen0, x1, y1,:, x2, y2, 0] += self.p_pen * self.r_pen
            # if in_pen1: Rewards[x0, x0, :, x1, y1, in_pen1, x2, y2, 1] += self.p_pen * self.r_pen

            # Make wall rewards NaN
            in_wall0 = any(np.all(np.array(loc0) == walls, axis=1))
            in_wall1 = any(np.all(np.array(loc1) == walls, axis=1))
            in_wall2 = any(np.all(np.array(loc2) == walls, axis=1))
            if any([in_wall0,in_wall1,in_wall2]):
                Rewards[x0, x0, :, x1, y1, :, x2, y2, :] = np.nan

            print(f'\r\t| [{i}] state', end='')
            i += 1

        return Rewards




class Qfunction(object):
    def __init__(self,iWorld):
        self.MDP =  MarkovDecisionProcess(iWorld)
        nact0, nact1 = 5, 5
        Qshape = list(self.MDP.States.shape()) + [nact0, nact1]
        self.tbl = np.zeros(Qshape)





def main():

    iWorld = 1
    Q = Qfunction(iWorld)
    print(f'')

if __name__ == "__main__":
    main()
