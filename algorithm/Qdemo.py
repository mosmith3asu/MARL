import math
import time

from learning.MDP.MDP_settings import MDP_file_path
from learning.Qlearning.Qlearning_Cent_utils import *
from learning.MDP.MDP_tools import MDP_DataClass,contains_border,Struct,is_caught

from decision_models.CPT_prelec import CPT
# from learning.learning_utils.enviorment_utils import Enviorment
from functools import partial
import itertools
import numpy as np
import matplotlib.pyplot as plt
from decision_models.CPT_prelec import CPT
import random
from math import dist
from tqdm import tqdm
try: import cPickle as pickle
except: import pickle

Await = np.array([0, 0])

if __name__ == "__main__":
    plt.ion()
    fig = plt.figure()
    ax_reward = fig.add_subplot(211)
    x = np.linspace(0, 10 * np.pi, 100)
    y = np.sin(x)
    rewards_line, = ax_reward.plot(x, y, c='r', linewidth=0.2)
    rewards_filt_line, = ax_reward.plot(x, y, c='b', linewidth=0.5)
    ax_reward.set_ylabel('Rewards')
    ax_reward.set_xlabel('Episode')

    ax_psucc = fig.add_subplot(212)
    psucc_line, = ax_psucc.plot(x, y, c='r', linewidth=0.2)
    psucc_filt_line, = ax_psucc.plot(x, y, c='b', linewidth=0.5)
    ax_psucc.set_ylabel('P(success | nG=7)')
    ax_psucc.set_xlabel('Episode')
    ax_psucc.set_ylim([0, 60])
    FILTER_TRACKER = False

    # ax_catch_freq = fig.add_subplot(313)


def update_state_plot(stats):
    # Rewards Plot ---------------
    scale = 1.1
    xdata = np.arange(np.size(stats.epi_reward))
    ydata = stats.epi_reward
    rewards_line.set_xdata(xdata)
    rewards_line.set_ydata(ydata)

    yfilt = move_ave_filter(ydata)
    rewards_filt_line.set_xdata(xdata)
    rewards_filt_line.set_ydata(yfilt)

    ax_reward.set_xlim([min(xdata),max(xdata)])
    # ax_reward.set_ylim([-scale * max(np.abs(ydata)), scale * max(np.abs(ydata))])
    ax_reward.set_ylim([min(ydata), scale * max(np.abs(ydata))])

    # Prob of Catch Plot ---------------
    ydata = stats.epi_psucc
    psucc_line.set_xdata(xdata)
    psucc_line.set_ydata(ydata)

    yfilt = move_ave_filter(ydata)
    psucc_filt_line.set_xdata(xdata)
    psucc_filt_line.set_ydata(yfilt)
    ax_psucc.set_xlim([min(xdata),max(xdata)])
    ax_psucc.set_ylim([0, max(scale * max(ydata),20)])


    fig.canvas.draw()
    fig.canvas.flush_events()
def batch_CPTparams(nVars=5, BOUNDS01=(0.2, 0.8), BOUNDS0inf=(0.2, 5)):
    H_CPT_params = {}
    H_CPT_params['b'] = [0]  # reference
    H_CPT_params['gamma'] = np.linspace(BOUNDS01[0], BOUNDS01[1], nVars)  # utility gain
    H_CPT_params['lam'] = np.linspace(BOUNDS0inf[0], BOUNDS0inf[1], nVars)  # relative weighting of gains/losses
    H_CPT_params['alpha'] = np.linspace(BOUNDS0inf[0], BOUNDS0inf[1], nVars)  # prelec parameter
    H_CPT_params['delta'] = np.linspace(BOUNDS01[0], BOUNDS01[1], nVars)  # convexity gain?
    H_CPT_params['theta'] = [10]  # rationality
    H_CPT_params['k'] = [1]  # ToM
    names = [key for key in H_CPT_params]
    a = [H_CPT_params[key] for key in H_CPT_params]
    CPT_Batch = []
    for params in list(itertools.product(*a)):
        this_batch = {}
        for iname, name in enumerate(names):
            this_batch[name] = params[iname]
        CPT_Batch.append(this_batch)
    return CPT_Batch


class Enviorment():
    def __init__(self,**kwargs):
        """iworld,MDP
        Transition probabilities are only for partner (not evader) since the action is only subject to uncertainty for partner action
        Evader transition probability is evaluated during the TD update when evaluating future actions since its action is not yet observed

        """
        self.filename = 'Dec_Env.pkl'
        self.done = False
        self.was_caught = False
        self.round = 0
        self.nMoves = 20
        self.reinit_params = {}
        self.reinit_params['done'] = False
        self.reinit_params['was_caught'] = False
        self.reinit_params['round'] = 0

        if 'load' in kwargs:
            if kwargs['load']==True: self.load()
            else: raise(f'Invalid enviorment attempted to laod kwargs={kwargs}')
        else:
            iworld = kwargs['iworld']
            MDP = kwargs['MDP']
            SAVE = kwargs['save']
            self.evader_rationality = kwargs['evader_rationality'] if 'evader_rationality' in kwargs else 10
            self.new(iworld,MDP)
            # for key in new_params: self.__dict__[key] = new_params[key]

            if SAVE: self.save()
    def init_penalties(self,iworld, R_player, P_pen, R_pen):
        player = 0
        world = WORLDS[iworld]['array']
        pen_states = np.array(np.where(world == WORLDS['pen_val'])).T
        Pen_layer = np.zeros(np.shape(R_player))
        for pen_state in pen_states:
            i_pen_states = np.where(np.all(self.S[:, player, :] == pen_state, axis=1))
            Pen_layer[i_pen_states] = P_pen * R_pen
        R_player += Pen_layer
        return R_player
    #################################################################
    # PLAY FUNCTIONS ################################################
    def step(self,si,ai0,pd_human):
        ai1 = np.randome.choice(np.arange(self.nA),p = pd_human)
        statej = self.S[si] + np.array([self.A[ai0],self.A[ai1],Await])
        sj = np.where(np.all(self.S == statej, axis=(1, 2)))[0]
        reward = self.R[sj]
        return reward,sj
    def reset(self):
        # Reset env params
        for key in self.reinit_params:
            self.__dict__[key] = self.reinit_params[key]
        # Find new start state
        bad_start = True
        while bad_start:
            si = random.choice(np.arange(self.nS))
            state = np.copy(self.S[si])
            d0 = dist(state[0], state[2])
            d1 = dist(state[1], state[2])
            if d0 <= 2.5 and d1 <= 2.5: bad_start = True  # at least 1 player is more than 2 tiles away
            else:  bad_start = False
        return si
    def is_done(self,si):
        if is_caught(self.S[si]):
            self.done = True
            self.was_caught = True
        elif self.round > self.nMoves: self.done = True
        return self.done
    #################################################################
    # DATA MANAGMENT ################################################
    def load(self):
        print(f'Loading ({self.filename})...', end='')
        # self.__dict__.update(np.load(self.filename,allow_pickle=True))
        with open(self.filename, 'rb') as f:  tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)
        print(f'\t[DONE]')
    def save(self):
        print(f'Saving ({self.filename})...', end='')
        # np.savez_compressed(self.filename,self.__dict__)
        with open(self.filename, 'wb') as f: pickle.dump(self.__dict__, f, 2)
        print(f'\t[DONE]')
    def new(self,iworld,MDP):
        print(f'\n##########################################')
        print(  f'# Starting new Decentralized Environment #')
        print(  f'##########################################')
        self.iWorld = iworld
        self.S = MDP.joint.S
        self.A = MDP.agent.A
        self.nA = np.shape(self.A)[0]
        self.nS = np.shape(self.S)[0]
        self.R = self.init_penalties(iworld, MDP.agent.R_player, MDP.P_pen, MDP.R_pen)
        self.R_evd = MDP.joint.R_evader

        # Player Transitions =============================
        T_adm = np.zeros([self.nS, self.nA, self.nS])  # integer
        T_unf = np.zeros([self.nS, self.nA, self.nS])  # uniform
        self.T = np.zeros([self.nS, self.nA, self.nS])  # player probs
        # Get admissible trans probs -------------------
        print(f'\nCalculating admissible transitions...', end='')
        Si, Ai = np.arange(self.nS), np.arange(self.nA)
        for si, ai0 in tqdm(list(itertools.product(*[Si, Ai]))):  # player action
            for ai1 in range(self.nA):  # partner transition
                purs_action = np.array([self.A[ai0], self.A[ai1], self.A[0]])
                statej = self.S[si] + purs_action
                sj = np.where(np.all(self.S == statej, axis=(1, 2)))[0]
                T_adm[si, ai0, sj] = 1
        # Init uniform transition -----------------------
        print(f'\nCalculating uniform transitions...')
        for si, ai0 in tqdm(list(itertools.product(*[Si, Ai]))):
            Tsum = np.sum(T_adm[si, ai0, :])
            if Tsum != 0: T_unf[si, ai0, :] = T_adm[si, ai0, :] / Tsum
        # Init working transitions ----------------------
        print(f'Calculating transitions...',end='')
        self.T = np.copy(T_unf)
        print(f'\t[DONE]')
        # ================================================

        # Evader Transitions =============================
        def NR(R,rationality):
            R = np.array(R)
            eps = 0.000001
            denom =  np.sum(np.exp(rationality* R))
            pd = np.exp(rationality * R) / (eps if denom == 0 else denom)
            pd = np.nan_to_num(pd) + eps
            return pd/np.sum(pd)#(eps if np.sum(pd)==0 else np.sum(pd))
        self.pdA_evd = np.zeros([self.nS, self.nA])
        print(f'Calculating evader transitions...')
        for si in tqdm(range(self.nS)):
            Rj,Ai,Sj = [],[],[]
            for ai in range(self.nA):
                evd_action = np.array([self.A[0], self.A[0], self.A[ai]])
                statej = self.S[si] + evd_action
                sj = np.where(np.all(self.S == statej, axis=(1, 2)))[0]
                if len(sj)>=1:
                    sj = sj[0]
                    Rj.append(self.R_evd[sj,0])
                    Ai.append(ai)
                    Sj.append(sj)
            self.pdA_evd[si, Ai] = NR(Rj, self.evader_rationality)
        # ===============================================
        # update_params = {}
        # update_params['iWorld'] = self.iWorld
        # update_params['S'] = self.S
        # update_params['A'] = self.A
        # update_params['nS'] = self.nA
        # update_params['nA'] = self.nS
        # update_params['R'] = self.R
        # update_params['R_evd'] = self.R_evd
        # update_params['T_adm'] = T_adm
        # update_params['T_unf'] = T_unf
        # update_params['T'] = self.T
        # update_params['pdA_evd'] = self.pdA_evd
        # return update_params
        #for key in update_params: self.__dict__[key] = update_params[key]

def handler_human_pdA(env,si,HQ,pd_robot,humanDM,jointAi):
    percP = humanDM.prob_weight(pd_robot)  # get prob value of human actions
    Vtmp = np.zeros(env.nA)  # percieved expected value
    for aiR, aiH in jointAi:
        statej = env.S[si] + np.array([env.A[aiR], env.A[aiH], Await])
        is_bad_state = contains_border(WORLDS['empty_world']['array'], WORLDS['border_val'], statej)
        if not is_bad_state:
            sj_sim = np.array(np.where(np.all(env.S == statej, axis=(1, 2)))).flatten()[0]
            try:
                p, v = percP[aiR], humanDM.utility_weight([HQ[sj_sim, aiH]])[0]  # get perceived transforms
            except:
                pass
            Vtmp[aiH] += p * v
    aiH_adm = np.where(np.max(env.T[switch_P1P2_state(env.S, si), :, :], axis=1) > 0)[0]
    pd_human = noisy_rational(Vtmp, humanDM.theta, admissable=aiH_adm)
    aiH = np.random.choice(np.arange(env.nA), p=pd_human)
    return aiH,pd_human

def Dec_QLearning(iworld=1,num_episodes=40000, discount_factor = 0.9, alpha = 0.6, epsilon = 0.15):
    """
    Centralized Q-Learning where we consider agents being symmetrical
    """
    warmup = 2000
    epi_eps =np.hstack([0.9*np.ones(warmup), np.linspace(0.9,epsilon,num_episodes-warmup)])


    # Settings

    LOAD_ENV = True
    enable_report = True
    notify_new_batch = False
    symm_Qfun = True
    plt_ever_ith_episode = 100
    file_name = f'Qfunction_Dec_W{iworld}'
    robot_rationality = 10
    human_rationality = 10
    evader_rationality = 10 # only when saving
    robot,human,evader = 0,1,2

    # Generate CPT Batch dataset
    CPT_batch = {}
    CPT_batch['vars'] = batch_CPTparams()
    CPT_batch['n'] = len(CPT_batch['vars'])
    CPT_batch['i'] = -1 # current batch
    CPT_batch['thresh'] = 20 # number of iterations before switching batches
    CPT_batch['cnt'] = CPT_batch['thresh']  # how many iters the batch went through (auto triggered)

    # Load/Make Environment
    if LOAD_ENV:
        env = Enviorment(load=True)
    else:
        MDP = MDP_DataClass(iworld, MDP_path=f'C:\\Users\\mason\\Desktop\\Effects_of_CPT_Accuracy\\learning\\MDP\\{MDP_file_path}')
        env = Enviorment(iworld = iworld,MDP = MDP,save=True,evader_rationality=evader_rationality)
        env = Enviorment(load=True)

    # Initialize stat tracker
    stats_init = {}
    stats_init['epi_reward'] = []
    stats_init['epi_length'] = []
    stats_init['epi_caught'] = []
    stats_init['epi_psucc'] = []
    stats_init['psucc_window'] = 99
    stats = Struct(**stats_init)

    # Initialize learning variables
    RQ = np.zeros([env.nS,env.nA])
    HQ = np.zeros([env.nS,env.nA])
    # policy = partial(eps_greedy, eps=epsilon)
    # jointA = np.array(list(itertools.product(*[env.A,env.A])))
    jointAi =np.array(list(itertools.product(*[np.arange(env.nA),np.arange(env.nA)])))


    ###################################################################################################
    # Begin Q learning ################################################################################
    ###################################################################################################
    print(  '\n\n###########################')
    print(      '#### BEGIN Q-LEARNING #####')
    print(      '###########################')
    si = env.reset()
    for ith_episode in range(num_episodes):
        policy = partial(eps_greedy, eps=epi_eps[ith_episode])

        if symm_Qfun: HQ = np.copy(RQ)

        # Initialize CPT Batch
        CPT_batch['cnt'] += 1  # increment batch count
        if CPT_batch['cnt'] > CPT_batch['thresh']: # new batch
            CPT_batch['cnt'] = 0
            CPT_batch['i'] += 1
            CPT_batch['i'] = CPT_batch['i']%len(CPT_batch['vars'])
            CPT_params = CPT_batch['vars'][CPT_batch['i']]
            humanDM = CPT(**CPT_params)

            get_human_pdA =  partial(handler_human_pdA, humanDM=humanDM,jointAi=jointAi)
            if notify_new_batch: print(f'\n ===> NEW CPT Params: ',end='')
            if notify_new_batch: [print(f'\t{key} = {round(CPT_params[key],2)}',end='') for key in CPT_params]
            if notify_new_batch: print(f'')

        # Report success if caught
        if env.was_caught:
            evader_state = env.S[si][2]
            is1 = np.array(evader_state == 1, dtype='int8')
            is5 = np.array(evader_state == 5, dtype='int8')
            is_corner =  '\t <== IN CORNER ==' if np.all(is1+is5 >= 1) else ''
            print(f'\t EVADER CAUGHT @ {evader_state} {is_corner}')

        # Initialize episode stats
        stats.epi_reward.append(0)
        stats.epi_length.append(0)
        stats.epi_caught.append(int(env.was_caught))
        stats.epi_psucc.append(100*np.mean(stats.epi_caught[-stats.psucc_window:]) if np.size(stats.epi_caught)>stats.psucc_window+1 else 0)

        # Start episode play #########################################################################
        si = env.reset()
        for t in range(21):
            # HQ = np.copy(RQ)
            #########################################
            # JOINT PLAYER DECISION #################
            #########################################
            # Take Robot Action (expliot vs explore) -------------------------
            aiR_adm = np.where(np.max(env.T[si, :, :],axis=1)>0)[0] # admissable actions
            aiH_adm = np.where(np.max(env.T[switch_P1P2_state(env.S,si), :, :], axis=1) > 0)[0]  # admissable actions
            pd_a = policy(RQ[si], admissable = aiR_adm )  # greater than zero trans prob
            aiR = np.random.choice(np.arange(len(pd_a)), p=pd_a) # select eps greed action

            # Determine Human Action Distribution -------------------------
            pd_robot = np.zeros(env.nA)
            if CPT_params['k'] == 1: pd_robot = noisy_rational(HQ[si], robot_rationality, admissable=aiR_adm)  # consider R
            else: pd_robot[aiR_adm] = 1 / len(aiR_adm)  # uniform advisable action
            aiH, pd_human = get_human_pdA(env,si,HQ,pd_robot)



            # Robot action conditional on Human pd -----------------------
            # RV_tmp = np.zeros(env.nA)
            # for aiR_tmp in aiR_adm:
            #     for aiH_tmp in aiH_adm:
            #         prob = pd_human[aiH_tmp]
            #         statej = env.S[si] + np.array([env.A[aiR_tmp],env.A[aiH_tmp],Await])
            #         sj_tmp = np.array(np.where(np.all(env.S == statej, axis=(1, 2)))).flatten()[0]
            #         RV_tmp[aiR_tmp] += prob * (env.R[sj_tmp][0] +  RQ[sj_tmp,aiR_tmp])
            # pd_a = policy(RV_tmp, admissable = aiR_adm )  # greater than zero trans prob
            # aiR = np.random.choice(np.arange(len(pd_a)), p=pd_a) # select eps greed action



            #########################################
            # TAKE JOINT ACTIONS ####################
            #########################################
            # reward, sj = env.step(si, ai,pd_partner=pd_human)
            statej = env.S[si] + np.array([env.A[aiR], env.A[aiH], Await])
            sj = np.where(np.all(env.S == statej, axis=(1, 2)))[0][0]
            # rewardR = env.R[sj][0]
            # rewardH = env.R[switch_P1P2_state(env.S,sj)][0]

            rewardR = np.zeros(env.nA) # expected reward
            rewardH = np.zeros(env.nA) # expected reward
            for aiR_tmp,aiH_tmp in itertools.product(*[aiR_adm,aiH_adm]):
                statej = env.S[si] + np.array([env.A[aiR_tmp], env.A[aiH_tmp], Await])
                sj_tmp = np.array(np.where(np.all(env.S == statej, axis=(1, 2)))).flatten()[0]
                phatH = pd_human[aiH_tmp] # infeered trans prob
                phatR = pd_human[aiH_tmp] # inferred trans prob
                rewardR[aiR_tmp] += phatH * env.R[sj_tmp][0]
                rewardH[aiH_tmp] += phatR * env.R[sj_tmp][0]
            rewardR = rewardR[aiR]
            rewardH = rewardH[aiH]

            # TRACKING STATS ----------------------
            stats.epi_reward[ith_episode] += np.mean(rewardR)
            stats.epi_length[ith_episode] = t
            if enable_report:
                report = '\r'
                report += f'\t| epI={ith_episode}  t={t}  '
                report += f'\tP(Success) = {round(stats.epi_psucc[ith_episode],2)}% '
                report += f'\tR = {round(stats.epi_reward[ith_episode], 2)} '
                # report += f'\tstart={[list(stateik) for stateik in MDP.joint.S[si_start]]} '
                # state0 = env.S[si_start]
                # report += f'\tDist0 = [{round(math.dist(state0[0],state0[2]),1)},{round(math.dist(state0[1],state0[2]),1)}] '
                report += f'\tQrange = {[round(np.max(RQ),1),round(np.min(RQ),1)]}'
                try: report += f'\tave(R) = [{round(np.mean(stats.epi_reward[-stats.psucc_window-1:-1]),2)}] '
                except: pass
                print(report, end='')

            #########################################
            # UPDATE Q TABLE ########################
            #########################################
            # Calculate future action conditional on evader movement
            valR_aj = np.zeros(np.shape(RQ)[1])
            valH_aj = np.zeros(np.shape(HQ)[1])
            pd_evd = env.pdA_evd[si]
            for ai_evd in np.where(pd_evd>0)[0]:
                statej = env.S[sj] + np.array([Await, Await, env.A[ai_evd]])
                sj_sim = np.array(np.where(np.all(env.S == statej, axis=(1, 2)))).flatten()[0]
                valR_aj += pd_evd[ai_evd] * RQ[sj_sim]  # value of state conditional on evd transition prob
                if not symm_Qfun: valH_aj += pd_evd[ai_evd] * HQ[sj_sim]  # value of state conditional on evd transition prob

            # Evaluate Future (Robot) ----------------------
            ajR = np.argmax(valR_aj) # take best next expected action
            td_target = rewardR + discount_factor * RQ[sj][ajR]
            td_delta = td_target - RQ[si][aiR]
            RQ[si][aiR] += alpha * td_delta

            # Evaluate Future (Human) ----------------------
            if not symm_Qfun:
                ajH = np.argmax(valH_aj)  # take best next expected action
                td_target = rewardH + discount_factor * HQ[sj][ajH]
                td_delta = td_target - HQ[si][aiH]
                HQ[si][aiH] += alpha * td_delta

            if env.is_done(sj):  break

            ##########################################
            # EVADER TURN ############################
            ##########################################
            pd_evd = env.pdA_evd[sj]
            try:  ai_evd = np.random.choice(np.arange(len(pd_evd)), p=pd_evd/np.sum(pd_evd))
            except Exception as inst:
                raise Exception(f'EVADER ERROR \n {inst} \n pd_evd={np.shape(pd_evd)}')
            statej = env.S[sj] + np.array([Await, Await, env.A[ai_evd]])
            sj = np.where(np.all(env.S == statej, axis=(1, 2)))[0][0]

            if env.is_done(sj): break

            # Check Valid Ending State
            # is_bad_state = contains_border(WORLDS['empty_world']['array'], WORLDS['border_val'], statej)
            # if is_bad_state:  raise Exception(f"ERRR: State j is out of bounds... \n State j = {statej}")

            si = sj
            env.round += 1
        ######################
        #### END OF EPI ######
        if ith_episode % plt_ever_ith_episode==0: update_state_plot(stats)
    print(f'\n\nSaving {file_name}...')
    save_Qfun(file_name, RQ, stats)


if __name__ == "__main__":
    for i in [1,2,3,4,5,6,7]: Dec_QLearning(iworld=i)

