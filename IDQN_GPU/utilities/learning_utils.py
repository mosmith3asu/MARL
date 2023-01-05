import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import random
import torch
import torch.nn as nn
from itertools import count
from scipy.special import softmax
import torch.multiprocessing as mp
import datetime
## Replay Memory ################################################

Transition = namedtuple('Transition',  ('state', 'action', 'next_state', 'reward'))

class EpisodeTimer(object):
    def __init__(self,max_buffer=100):
        self.durations = []
        self.max_buffer = max_buffer
        self.last_sample_time = None
        self.n_del = max_buffer/10

    def sample(self):
        if self.last_sample_time is not None:
            self.durations.append(time.time() - self.last_sample_time)
            if len(self.durations) > self.max_buffer: self.durations.pop(0)
                # self.durations = self.durations[self.n_del:]
        self.last_sample_time = time.time()

    @property
    def mean_dur(self):
        if len(self.durations)>0: return np.mean(self.durations)
        else: return 0

    def remaining_time(self,i_epi,num_episodes):
        return str(datetime.timedelta(seconds=round(self.mean_dur * (num_episodes - i_epi))))


class CPT_Handler(object):
    @classmethod
    def rand(cls,assume=None,verbose=True):
        cpt = CPT_Handler(rand=True,assume=assume)
        if verbose: cpt.preview_params()
        return cpt


    def __init__(self,rand=False,assume=None):
        self.b, self.b_bounds = 0., (0, 5)
        self.lam, self.lam_bounds = 1., (1/5, 5)
        self.yg, self.yg_bounds = 1., (0.4, 1.0)
        self.yl, self.yl_bounds = 1., (0.4, 1.0)
        self.dg, self.dg_bounds = 1., (0.4, 1.0)
        self.dl, self.dl_bounds = 1., (0.4, 1.0)
        self.rationality, self.rationality_bounds = 1., (1., 1.)

        self.symm_reward_sensitivity = False
        self.symm_probability_sensitivity = False

        self.assumption_attempts = 500
        self.n_test_sensitivity = 100
        self.p_thresh_sensitivity = 0.2
        self.r_range_sensitivity = (-15, 15)
        self.r_pen = -3
        self.p_pen = 0.5

        self.n_draw_sensitivity = 100
        self.n_draw_ticks = 5

        self.paccept_sensitivity = 0
        self.attribution = 'insensitive'

        if rand: self.rand_params(assume=assume)

    def __str__(self):
        return self.flat_preview()
    def flat_preview(self):
        sigdig = 2
        # self.preview_params()
        disp = []
        for key in ['b','lam','yg','yl','dg','dl','rationality']:
            disp.append(round(self.__dict__[key], sigdig))

        dfavor = self.get_favor()
        favor = 'more gain' if dfavor >0 else 'more loss'

        return f'CPT({round(self.paccept_sensitivity,sigdig)}):{disp} => [{favor}]={dfavor}%'# G{TG} x L{TL}'

    def _get_optimal(self):
        b, lam, yg, yl, dg, dl, rationality = 0., 1., 1., 1., 1., 1., 1.
        return b, lam, yg, yl, dg, dl, rationality

    @property
    def is_optimal(self):
        check = True
        if self.b != 0.: check = False
        if self.lam != 1.: check = False
        if self.yg != 1.: check = False
        if self.yl != 1.: check = False
        if self.dg != 1.: check = False
        if self.dl != 1.: check = False
        return check

    def preview_params(self,sigdig=2):
        print(f'### [CPT Parameters] ### <==',end='')
        # print('\t|',end='')
        for key in ['b','lam','yg','yl','dg','dl','rationality']:
            print(' {:<1}: [{:<1}]'.format(key,round(self.__dict__[key],sigdig)),end='')
        print(f'\t sensitivity: [{round(self.paccept_sensitivity,sigdig)}] \t attribution: [{self.attribution}]',end='')
        print(f'')
    def transform(self, *args):
        if len(args)==2:
            r, p = args
            if self.is_optimal: return r,p
            b, lam, yg, yl, dg, dl = self.b, self.lam, self.yg, self.yl, self.dg, self.dl
        elif len(args)==8:
            r, p, b, lam, yg, yl, dg, dl = args
        else: raise Exception("UNKOWN NUMBER OF CPT TRANSFORM ARGUMENTS")

        is_cert = (p==1)
        if (r - b) >= 0:
            rhat = pow(r - b, yg)
            phat = pow(p, dg) / pow(pow(p, dg) + pow(1 - p, dg), dg)
        else:
            rhat = -lam * pow(abs(r - b), yl)
            phat = pow(p, dl) / pow(pow(p, dl) + pow(1 - p, dl), dl)

        if is_cert: phat=1
        return rhat, phat

    def plot_indifference_curve(self, ax=None):
        N = self.n_draw_sensitivity
        if ax is None: fig, ax = plt.subplots(1, 1)
        n_ticks = self.n_draw_ticks
        rmin, rmax = self.r_range_sensitivity
        r_cert = np.linspace(rmin, rmax, N)  # + r_pen/2
        r_gain = np.linspace(rmin, rmax, N)  # [0,20]

        attribution, p_accept = self._get_sensitivity(self.b, self.lam,
                                                      self.yg, self.yl,
                                                      self.dg, self.dl,
                                                      self.rationality,
                                                      return_paccept=True, N=N)

        ax.matshow(p_accept - 0.5, cmap='bwr', origin='lower')
        ax.set_title(
            f'Preference Map [{attribution}]\n (White: indifferent)(Red: prefer gamble)(Blue: prefer certainty)')
        ax.set_xlabel('$\mathbb{C}[R_{gamble} \; | \; r_{\\rho}=-3,p_{\\rho}=0.5 ]$')
        # ax.set_xlabel('$\mathbb{C}[R_{gamble}] = (1-p_{pen})R_{gain}+p_{pen}(R_{gain}-r_{pen})$')

        ax.set_xticks(np.linspace(0, 100, n_ticks))
        ax.set_yticks(np.linspace(0, 100, n_ticks))
        ax.set_xticklabels(np.round(np.linspace(r_gain[0], r_gain[-1], n_ticks), 1))

        ax.set_ylabel('$\mathbb{C}[R_{certainty}]$')
        ax.set_yticklabels(np.round(np.linspace(r_cert[0], r_cert[-1], n_ticks), 1))

    def _get_sensitivity(self, b, lam, yg, yl, dg, dl, rationality, return_paccept=False, N=None):
        iaccept = 0
        N = self.n_test_sensitivity if N is None else N
        rmin, rmax = self.r_range_sensitivity
        r_cert = np.linspace(rmin, rmax, N)  + self.r_pen/2
        r_gain = np.linspace(rmin, rmax, N)  # [0,20]
        r_loss = r_gain + self.r_pen
        p_thresh = self.p_thresh_sensitivity

        # p_accept = np.empty([N, N])
        # for r in range(N):
        #     for c in range(N):
        #         rg = r_gain[c]
        #         rl = r_loss[c]
        #         rc = r_cert[r]
        #
        #         rg_hat, pg_hat = self.transform(rg, 1 - self.p_pen, b, lam, yg, yl, dg, dl)
        #         rl_hat, pl_hat = self.transform(rl, self.p_pen, b, lam, yg, yl, dg, dl)
        #         Er_gamble = (rg_hat * pg_hat) + (rl_hat * pl_hat)
        #         Er_cert = rc - b
        #         Er_choices = np.array([Er_gamble, Er_cert])
        #
        #         pCPT = softmax(rationality * Er_choices)
        #         p_accept[r, c] = pCPT[iaccept]
        #
        # p_sum = np.mean(p_accept - 0.5)
        # if abs(p_sum) < p_thresh: attribution = 'insensitive'
        # elif p_sum >= p_thresh: attribution = 'seeking'
        # elif p_sum <= -p_thresh: attribution = 'averse'
        # else: raise Exception('Unknown CPT attribution')

        p_accept = 0
        p_sum = self.get_favor()

        if abs(p_sum) < p_thresh: attribution = 'insensitive'
        elif p_sum >= p_thresh and abs(p_sum)<2: attribution = 'seeking'
        elif p_sum <= -p_thresh and abs(p_sum)<2: attribution = 'averse'
        else:
            # raise Exception('Unknown CPT attribution')
            attribution = 'Unknown CPT attribution'
        if return_paccept: return attribution, p_accept
        else: return attribution

    def get_favor(self):
        p = 0.5

        dfavors = np.zeros(int(10))
        for r in range(dfavors.size):
            rhatG, phatG = np.array(self.transform((r + 1), p))
            rhatL, phatL = np.array(self.transform(-(r + 1), p))
            dfavors[r] = (rhatG * phatG + rhatL * phatL) / (r + 1)

        dfavor = np.nan_to_num(np.mean(dfavors)).round(1)  # round(rel_diff[0]-rel_diff[1],1)
        return dfavor


    def _sample_random_params(self, n_samples):
        b = np.random.choice(np.linspace(self.b_bounds[0], self.b_bounds[1], n_samples))
        lam_seeking = np.linspace(self.lam_bounds[0], 1, int(n_samples / 2))
        lam_averse = np.linspace(self.lam_bounds[0] + 1, self.lam_bounds[1], int(n_samples / 2))
        lam = np.random.choice(np.hstack([lam_seeking, lam_averse]))
        yg = np.random.choice(np.linspace(self.yg_bounds[0], self.yg_bounds[1], n_samples))
        yl = np.random.choice(np.linspace(self.yl_bounds[0], self.yl_bounds[1], n_samples))
        dg = np.random.choice(np.linspace(self.dg_bounds[0], self.dg_bounds[1], n_samples))
        dl = np.random.choice(np.linspace(self.dl_bounds[0], self.dl_bounds[1], n_samples))
        rationality = np.random.choice(
            np.linspace(self.rationality_bounds[0], self.rationality_bounds[1], n_samples))
        if self.symm_reward_sensitivity: yl = yg
        if self.symm_probability_sensitivity: dl = dg
        return b, lam, yg, yl, dg, dl, rationality

    def rand_params(self, assume=None, n_samples=100):
        assert assume.lower() in ['averse', 'seeking', 'insensitive','baseline', None], f'CPT parameter assumption unknown: {assume}'
        if assume is not None:
            if assume.lower() == 'baseline':
                b, lam, yg, yl, dg, dl, rationality = self._get_optimal()
                self.b, self.lam = b, lam
                self.yg, self.yl = yg, yl
                self.dg, self.dl = dg, dl
                self.rationality = rationality
            else:
                for attempt in range(self.assumption_attempts):
                    b, lam, yg, yl, dg, dl, rationality = self._sample_random_params(n_samples)
                    self.b, self.lam = b, lam
                    self.yg, self.yl = yg, yl
                    self.dg, self.dl = dg, dl
                    self.rationality = rationality

                    attribution,p_accept = self._get_sensitivity(b, lam, yg, yl, dg, dl, rationality, return_paccept=True)
                    self.attribution = attribution
                    # self.paccept_sensitivity = np.mean(p_accept - 0.5)

                    if attribution.lower() == assume.lower(): break
                    if attempt>=self.assumption_attempts-1: logging.warning(f"CPT unable to generate assumed {assume} parameters")




def soft_update(policy_net,target_net,TAU):
    # Soft update of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
    target_net.load_state_dict(target_net_state_dict)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def optimize_model(policy_net,target_net,optimizer, memory,GAMMA,BATCH_SIZE,lr_scheduler=None,update_iterations=1):
    if len(memory) < 2*BATCH_SIZE: return
    n_agents = 2
    losses = []
    for _ in range(update_iterations):
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=policy_net.tensor_type['device'], dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)#.to(policy_net.tensor_type['device'])
        action_batch = torch.cat(batch.action)#.to(policy_net.tensor_type['device'])
        reward_batch = torch.cat(batch.reward).reshape([-1,2]) #.to(policy_net.tensor_type['device'])

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        # state_action_values = policy_net(state_batch).gather(1, action_batch)
        state_action_values = policy_net(state_batch).gather(2, action_batch.unsqueeze(1).repeat([1, 2, 1]))


        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros([BATCH_SIZE,n_agents], device=policy_net.tensor_type['device'])
        with torch.no_grad():
            # aJ = target_net.sample_action(non_final_next_states,epsilon=0,best=True)
            # qA_sprime = target_net(non_final_next_states)
            # next_state_values[non_final_mask] = qA_sprime.gather(2,aJ.unsqueeze(1).repeat(1,2,1)).squeeze()

            aJ,qA_sprime = target_net.sample_action(non_final_next_states,epsilon=0,best=True)
            next_state_values[non_final_mask] = qA_sprime.squeeze()



        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(-1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()
        losses.append(loss.item())

    # mean_loss = loss.item()
    if lr_scheduler is not None:
        mean_loss = sum(losses)/len(losses)
        lr_scheduler.step(mean_loss)

# def optimize_model(policy_net,target_net,optimizer, memory,GAMMA,BATCH_SIZE):
#     if len(memory) < BATCH_SIZE: return
#     n_agents = 2
#     transitions = memory.sample(BATCH_SIZE)
#     # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
#     # detailed explanation). This converts batch-array of Transitions
#     # to Transition of batch-arrays.
#     batch = Transition(*zip(*transitions))
#
#     # Compute a mask of non-final states and concatenate the batch elements
#     # (a final state would've been the one after which simulation ended)
#     non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
#                                   device=policy_net.tensor_type['device'], dtype=torch.bool)
#     non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
#
#     #
#     # non_final_mask = torch.tensor(tuple(map(lambda _done: not _done, batch.next_state)),
#     #                               device=policy_net.tensor_type['device'], dtype=torch.bool)
#     # non_final_next_states = torch.cat([batch.next_state[si] for si in batch.next_state.numel() if not batch.done[si]])
#
#
#     state_batch = torch.cat(batch.state)#.to(policy_net.tensor_type['device'])
#     action_batch = torch.cat(batch.action)#.to(policy_net.tensor_type['device'])
#     reward_batch = torch.cat(batch.reward).reshape([-1,2]) #.to(policy_net.tensor_type['device'])
#
#     # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
#     # columns of actions taken. These are the actions which would've been taken
#     # for each batch state according to policy_net
#     # state_action_values = policy_net(state_batch).gather(1, action_batch)
#     state_action_values = policy_net(state_batch).gather(2, action_batch.unsqueeze(1).repeat([1, 2, 1]))
#
#
#     # Compute V(s_{t+1}) for all next states.
#     # Expected values of actions for non_final_next_states are computed based
#     # on the "older" target_net; selecting their best reward with max(1)[0].
#     # This is merged based on the mask, such that we'll have either the expected
#     # state value or 0 in case the state was final.
#     next_state_values = torch.zeros([BATCH_SIZE,n_agents], device=policy_net.tensor_type['device'])
#     with torch.no_grad():
#         # next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
#         aJ = target_net.sample_action(non_final_next_states,epsilon=0)
#         qA_sprime = target_net(non_final_next_states)
#         next_state_values[non_final_mask] = qA_sprime.gather(2,aJ.unsqueeze(1).repeat(1,2,1)).squeeze()
#
#
#     # Compute the expected Q values
#     expected_state_action_values = (next_state_values * GAMMA) + reward_batch
#
#     # Compute Huber loss
#     criterion = nn.SmoothL1Loss()
#     loss = criterion(state_action_values, expected_state_action_values.unsqueeze(-1))
#
#     # Optimize the model
#     optimizer.zero_grad()
#     loss.backward()
#     # In-place gradient clipping
#     torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
#     optimizer.step()

def schedule(start,end,N_start,N_end,N_total,slope=1.0):
    if start == end: return start * np.ones(N_total)
    warmup = start * np.ones(N_start)
    perform = end * np.ones(N_end)
    N_transition = N_total - N_start - N_end
    iinv = np.power(1 / (np.linspace(1, 10,N_transition) - 0.1) - 0.1, slope)
    improve = (start + end) * iinv + end
    epi_schedule = np.hstack([warmup, improve, perform])
    return epi_schedule



def test(env, num_episodes, policy_net):
    with torch.no_grad():
        length  = 0
        psucc   = 0
        score   = np.zeros(env.n_agents)
        for episode_i in range(num_episodes):
            state = env.reset()
            for t in count():
                action = policy_net.sample_action(state, epsilon=0)
                next_state, reward, done, _ = env.step(action.squeeze())
                score += reward.detach().flatten().cpu().numpy()
                state = next_state.clone()
                if done: break
            if env.check_caught(env.current_positions): psucc +=1
            length += env.step_count

    final_score     = list(score/ num_episodes)
    final_length    = length/num_episodes
    final_psucc     = psucc/ num_episodes

    return final_score,final_length,final_psucc



def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


def spawn_env_worker(env, policy_net, epsilon,que, num_iterations = 4):
    # processes = []
    # manager = mp.Manager()
    # que = manager.Queue()
    # num_proc = 4
    # plt.ioff()
    # for iproc in range(num_proc):
    #     p = mp.Process(target=spawn_env_worker,args=(copy.deepcopy(env), policy_net, epsilon,que,))
    #     p.start()
    #     processes.append(p) #spawn_env_worker(iWorld, policy_net, epsilon)
    #
    # for p in processes:
    #     p.join()
    # que.put(None)
    # plt.ion()
    #
    # for _ in count():
    #     sample = que.get()
    #     if sample is None:  break
    #     else:
    #         # print(data.shape)
    #         state = sample[0:6]
    #         action = sample[6]
    #         next_state = None if torch.all(sample[7:13]==0) else sample[7:13]
    #         reward = sample[13:14]
    #         memory.push(state, action, next_state, reward)


    # print(f'start_proc')
    # for i in range(num_iterations):
    #     que.put(i*torch.ones(1, 10))
    # mp.Event().wait()
    #
    nVars = 15
    history = torch.empty([0,nVars])#.share_memory_()
    observations = torch.empty([0, nVars])

    # history = []
    for iter in range(num_iterations):
        # iter_history = []
        state = env.reset()  # Initialize the environment and get it's state
        done = False
        for t in count():
            action = policy_net.sample_action(state, epsilon)
            next_state, reward, done, _ = env.step(action.squeeze())
            # obs = [state, action, None if done else next_state, reward]
            obs = torch.cat([state.flatten(), action.flatten(), torch.zeros(6).flatten() if done else next_state.flatten(), reward.flatten()])
            observations = torch.cat([observations, obs.reshape([1, nVars])], dim=0)

            # obs.share_memory_()
            # que = torch.cat([que,  obs.reshape([1, nVars])], dim=0)
            # que.put(obs.s)

            # history = torch.cat([history,obs.reshape([1,nVars])],dim=1)


            if done:                break

    #return history
    # print(observations.shape)
    # history = torch.cat([history, observations], dim=0)
    que.put(observations)

class ExecutionTimer(object):
    def __init__(self,max_samples=10_000,sigdig = 3,enable=True):
        # self.profiles = []
        self.profiles = {}
        self.max_samples = max_samples
        self.sigdig = sigdig
        self.tstarts = {}
        self.main_iter = 0
        self.ENABLE = enable

    def __call__(self, *args, **kwargs):
        if not self.ENABLE: return None
        return self.add_profile(*args, **kwargs)

    def preview_percent(self):
        if not self.ENABLE: return None
        if len(self.profiles['main']['dur']) > 0:
            mean_execution_time = np.mean(self.profiles['main']['dur'])
            print(f'Execution Times [Percent]: ############')
            for name in self.profiles.keys():
                profile_durs = np.array(self.profiles[name]['dur'])
                disp = '\t| {:<10}: {:<5}%'.format(name, (np.mean(profile_durs) / mean_execution_time).round(self.sigdig))
                print(disp)
            # print(f'\t| {name}: {(np.mean(ave_durs)/mean_execution_time).round(self.sigdig)}%')
    # def preview_percent(self):
    #     if not self.ENABLE: return None
    #     mean_execution_time =  np.mean(self.profiles['main']['dur'])
    #     # if len(self.profiles['main']['dur']) > 0:
    #     print(f'Execution Times [Percent]: ############')
    #     for name in self.profiles.keys():
    #         profile_durs = np.array(self.profiles[name]['dur'])
    #         profile_imains = np.array(self.profiles[name]['main_iter'])
    #         ave_durs = []
    #         for imain in  self.profiles['main']['main_iter']:
    #            idurs = profile_durs[np.where(profile_imains == imain)]
    #            if len(idurs) > 0: ave_durs.append(np.mean(idurs))
    #
    #         disp = '\t| {:<10}: {:<5}%'.format(name,(np.mean(ave_durs)/mean_execution_time).round(self.sigdig))
    #         print(disp)
    #         #print(f'\t| {name}: {(np.mean(ave_durs)/mean_execution_time).round(self.sigdig)}%')

    def preview_all(self):
        if not self.ENABLE: return None
        print(f'Execution Times: ############')
        for name in self.profiles.keys():
            print(f'\t| {name}: {np.mean(self.profiles[name]).round(self.sigdig)}')

    def mean_duration(self,name):
        if not self.ENABLE: return None
        return np.mean(self.profiles[name])


    def add_profile(self,name,status):
        if not self.ENABLE: return None
        assert status == 'start' or status == 'stop', f'incorrect input arg [{status}]'
        if status == 'start':
            self.tstarts[name] = time.time()
            if name not in self.profiles.keys():
                self.profiles[name] = {'dur': [], 'main_iter': []}
        elif status == 'stop':
            self.profiles[name]['dur'].append(time.time() - self.tstarts[name])
            self.profiles[name]['main_iter'].append(self.main_iter)

    def main_profile(self,status):
        if not self.ENABLE: return None

        assert status == 'start' or status == 'stop', f'incorrect input arg [{status}]'
        name = 'main'
        if status == 'start':
            self.tstarts[name] = time.time()
            if name not in self.profiles.keys():
                self.profiles[name] = {'dur':[],'main_iter':[]}
        elif status == 'stop':
            self.profiles[name]['dur'].append(time.time() - self.tstarts[name])
            self.profiles[name]['main_iter'].append(self.main_iter)
            self.main_iter += 1

# """
# import logging
# import time
# import numpy as np
# import matplotlib.pyplot as plt
# from collections import namedtuple, deque
# import random
# import torch
# import torch.nn as nn
# from itertools import count
#
# ## Replay Memory ################################################
#
# Transition = namedtuple('Transition',  ('state', 'action', 'next_state', 'reward'))
#
#
#
# class CPT_Handler(object):
#     @classmethod
#     def rand(cls,assume=None,verbose=True):
#         cpt = CPT_Handler(rand=True,assume=assume)
#         if verbose: cpt.preview_params()
#         return cpt
#
#
#     def __init__(self,rand=False,assume=None):
#         self.b, self.b_bounds = 0., (0, 8.5)
#         self.lam, self.lam_bounds = 1., (1/5, 5)
#         self.yg, self.yg_bounds = 1., (0.4, 1.0)
#         self.yl, self.yl_bounds = 1., (0.4, 1.0)
#         self.dg, self.dg_bounds = 1., (0.4, 1.0)
#         self.dl, self.dl_bounds = 1., (0.4, 1.0)
#         self.rationality, self.rationality_bounds = 1., (1., 1.)
#
#         self.symm_reward_sensitivity = False
#         self.symm_probability_sensitivity = False
#
#         self.assumption_attempts = 500
#         self.n_test_sensitivity = 100
#         self.p_thresh_sensitivity = 0.2
#         self.r_range_sensitivity = (-15, 15)
#         self.r_pen = -3
#         self.p_pen = 0.5
#
#         self.n_draw_sensitivity = 100
#         self.n_draw_ticks = 5
#
#         self.paccept_sensitivity = 0
#         self.attribution = 'insensitive'
#
#         if rand: self.rand_params(assume=assume)
#
#     def __str__(self):
#         return self.flat_preview()
#     def flat_preview(self):
#         sigdig = 2
#         # self.preview_params()
#         disp = []
#         for key in ['b','lam','yg','yl','dg','dl','rationality']:
#             disp.append(round(self.__dict__[key], sigdig))
#
#         dfavor = self.get_favor()
#         favor = 'more gain' if dfavor >0 else 'more loss'
#
#         return f'CPT({round(self.paccept_sensitivity,sigdig)}):{disp} => [{favor}]={dfavor}%'# G{TG} x L{TL}'
#
#     def _get_optimal(self):
#         b, lam, yg, yl, dg, dl, rationality = 0., 1., 1., 1., 1., 1., 1.
#         return b, lam, yg, yl, dg, dl, rationality
#
#     @property
#     def is_optimal(self):
#         check = True
#         if self.b != 0.: check = False
#         if self.lam != 1.: check = False
#         if self.yg != 1.: check = False
#         if self.yl != 1.: check = False
#         if self.dg != 1.: check = False
#         if self.dl != 1.: check = False
#         return check
#
#     def preview_params(self,sigdig=2):
#         print(f'### [CPT Parameters] ### <==',end='')
#         # print('\t|',end='')
#         for key in ['b','lam','yg','yl','dg','dl','rationality']:
#             print(' {:<1}: [{:<1}]'.format(key,round(self.__dict__[key],sigdig)),end='')
#         print(f'\t sensitivity: [{round(self.paccept_sensitivity,sigdig)}] \t attribution: [{self.attribution}]',end='')
#         print(f'')
#     def transform(self, *args):
#         if len(args)==2:
#             r, p = args
#             if self.is_optimal: return r,p
#             b, lam, yg, yl, dg, dl = self.b, self.lam, self.yg, self.yl, self.dg, self.dl
#         elif len(args)==8:
#             r, p, b, lam, yg, yl, dg, dl = args
#         else: raise Exception("UNKOWN NUMBER OF CPT TRANSFORM ARGUMENTS")
#
#         is_cert = (p==1)
#         if (r - b) >= 0:
#             rhat = pow(r - b, yg)
#             phat = pow(p, dg) / pow(pow(p, dg) + pow(1 - p, dg), dg)
#         else:
#             rhat = -lam * pow(abs(r - b), yl)
#             phat = pow(p, dl) / pow(pow(p, dl) + pow(1 - p, dl), dl)
#
#         if is_cert: phat=1
#         return rhat, phat
#
#     def plot_indifference_curve(self, ax=None):
#         N = self.n_draw_sensitivity
#         if ax is None: fig, ax = plt.subplots(1, 1)
#         n_ticks = self.n_draw_ticks
#         rmin, rmax = self.r_range_sensitivity
#         r_cert = np.linspace(rmin, rmax, N)  # + r_pen/2
#         r_gain = np.linspace(rmin, rmax, N)  # [0,20]
#
#         attribution, p_accept = self._get_sensitivity(self.b, self.lam,
#                                                       self.yg, self.yl,
#                                                       self.dg, self.dl,
#                                                       self.rationality,
#                                                       return_paccept=True, N=N)
#
#         ax.matshow(p_accept - 0.5, cmap='bwr', origin='lower')
#         ax.set_title(
#             f'Preference Map [{attribution}]\n (White: indifferent)(Red: prefer gamble)(Blue: prefer certainty)')
#         ax.set_xlabel('$\mathbb{C}[R_{gamble} \; | \; r_{\\rho}=-3,p_{\\rho}=0.5 ]$')
#         # ax.set_xlabel('$\mathbb{C}[R_{gamble}] = (1-p_{pen})R_{gain}+p_{pen}(R_{gain}-r_{pen})$')
#
#         ax.set_xticks(np.linspace(0, 100, n_ticks))
#         ax.set_yticks(np.linspace(0, 100, n_ticks))
#         ax.set_xticklabels(np.round(np.linspace(r_gain[0], r_gain[-1], n_ticks), 1))
#
#         ax.set_ylabel('$\mathbb{C}[R_{certainty}]$')
#         ax.set_yticklabels(np.round(np.linspace(r_cert[0], r_cert[-1], n_ticks), 1))
#
#     def _get_sensitivity(self, b, lam, yg, yl, dg, dl, rationality, return_paccept=False, N=None):
#         iaccept = 0
#         N = self.n_test_sensitivity if N is None else N
#         rmin, rmax = self.r_range_sensitivity
#         r_cert = np.linspace(rmin, rmax, N)  + self.r_pen/2
#         r_gain = np.linspace(rmin, rmax, N)  # [0,20]
#         r_loss = r_gain + self.r_pen
#         p_thresh = self.p_thresh_sensitivity
#
#         # p_accept = np.empty([N, N])
#         # for r in range(N):
#         #     for c in range(N):
#         #         rg = r_gain[c]
#         #         rl = r_loss[c]
#         #         rc = r_cert[r]
#         #
#         #         rg_hat, pg_hat = self.transform(rg, 1 - self.p_pen, b, lam, yg, yl, dg, dl)
#         #         rl_hat, pl_hat = self.transform(rl, self.p_pen, b, lam, yg, yl, dg, dl)
#         #         Er_gamble = (rg_hat * pg_hat) + (rl_hat * pl_hat)
#         #         Er_cert = rc - b
#         #         Er_choices = np.array([Er_gamble, Er_cert])
#         #
#         #         pCPT = softmax(rationality * Er_choices)
#         #         p_accept[r, c] = pCPT[iaccept]
#         #
#         # p_sum = np.mean(p_accept - 0.5)
#         # if abs(p_sum) < p_thresh: attribution = 'insensitive'
#         # elif p_sum >= p_thresh: attribution = 'seeking'
#         # elif p_sum <= -p_thresh: attribution = 'averse'
#         # else: raise Exception('Unknown CPT attribution')
#
#         p_accept = 0
#         p_sum = self.get_favor()
#
#         if abs(p_sum) < p_thresh: attribution = 'insensitive'
#         elif p_sum >= p_thresh: attribution = 'seeking'
#         elif p_sum <= -p_thresh: attribution = 'averse'
#         else: raise Exception('Unknown CPT attribution')
#
#         if return_paccept: return attribution, p_accept
#         else: return attribution
#
#     def get_favor(self):
#         p = 0.5
#
#         dfavors = np.zeros(int(10))
#         for r in range(dfavors.size):
#             rhatG, phatG = np.array(self.transform((r + 1), p))
#             rhatL, phatL = np.array(self.transform(-(r + 1), p))
#             dfavors[r] = (rhatG * phatG + rhatL * phatL) / (r + 1)
#
#         dfavor = np.nan_to_num(np.mean(dfavors)).round(1)  # round(rel_diff[0]-rel_diff[1],1)
#         return dfavor
#
#
#     def _sample_random_params(self, n_samples):
#         b = np.random.choice(np.linspace(self.b_bounds[0], self.b_bounds[1], n_samples))
#         lam_seeking = np.linspace(self.lam_bounds[0], 1, int(n_samples / 2))
#         lam_averse = np.linspace(self.lam_bounds[0] + 1, self.lam_bounds[1], int(n_samples / 2))
#         lam = np.random.choice(np.hstack([lam_seeking, lam_averse]))
#         yg = np.random.choice(np.linspace(self.yg_bounds[0], self.yg_bounds[1], n_samples))
#         yl = np.random.choice(np.linspace(self.yl_bounds[0], self.yl_bounds[1], n_samples))
#         dg = np.random.choice(np.linspace(self.dg_bounds[0], self.dg_bounds[1], n_samples))
#         dl = np.random.choice(np.linspace(self.dl_bounds[0], self.dl_bounds[1], n_samples))
#         rationality = np.random.choice(
#             np.linspace(self.rationality_bounds[0], self.rationality_bounds[1], n_samples))
#         if self.symm_reward_sensitivity: yl = yg
#         if self.symm_probability_sensitivity: dl = dg
#         return b, lam, yg, yl, dg, dl, rationality
#
#     def rand_params(self, assume=None, n_samples=100):
#         assert assume.lower() in ['averse', 'seeking', 'insensitive','baseline', None], f'CPT parameter assumption unknown: {assume}'
#         if assume is not None:
#             if assume.lower() == 'baseline':
#                 b, lam, yg, yl, dg, dl, rationality = self._get_optimal()
#                 self.b, self.lam = b, lam
#                 self.yg, self.yl = yg, yl
#                 self.dg, self.dl = dg, dl
#                 self.rationality = rationality
#             else:
#                 for attempt in range(self.assumption_attempts):
#                     b, lam, yg, yl, dg, dl, rationality = self._sample_random_params(n_samples)
#                     self.b, self.lam = b, lam
#                     self.yg, self.yl = yg, yl
#                     self.dg, self.dl = dg, dl
#                     self.rationality = rationality
#
#                     attribution,p_accept = self._get_sensitivity(b, lam, yg, yl, dg, dl, rationality, return_paccept=True)
#                     self.attribution = attribution
#                     # self.paccept_sensitivity = np.mean(p_accept - 0.5)
#
#                     if attribution.lower() == assume.lower(): break
#                     if attempt>=self.assumption_attempts-1: logging.warning(f"CPT unable to generate assumed {assume} parameters")
#
#
#
#
# def soft_update(policy_net,target_net,TAU):
#     # Soft update of the target network's weights
#     # θ′ ← τ θ + (1 −τ )θ′
#     target_net_state_dict = target_net.state_dict()
#     policy_net_state_dict = policy_net.state_dict()
#     for key in policy_net_state_dict:
#         target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
#     target_net.load_state_dict(target_net_state_dict)
#
#
# class ReplayMemory(object):
#
#     def __init__(self, capacity):
#         self.memory = deque([],maxlen=capacity)
#
#     def push(self, *args):
#         """Save a transition"""
#         self.memory.append(Transition(*args))
#
#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)
#
#     def __len__(self):
#         return len(self.memory)
#
#
#
# # def optimize_model(policy_net,target_net,optimizer, memory,GAMMA,BATCH_SIZE,lr_scheduler=None,update_iterations=1):
# #     if len(memory) < BATCH_SIZE: return
# #     n_agents = 2
# #     transitions = memory.sample(BATCH_SIZE)
# #     # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
# #     # detailed explanation). This converts batch-array of Transitions
# #     # to Transition of batch-arrays.
# #     batch = Transition(*zip(*transitions))
# #
# #     # Compute a mask of non-final states and concatenate the batch elements
# #     # (a final state would've been the one after which simulation ended)
# #     non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
# #                                   device=policy_net.tensor_type['device'], dtype=torch.bool)
# #     non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
# #
# #     state_batch = torch.cat(batch.state)#.to(policy_net.tensor_type['device'])
# #     action_batch = torch.cat(batch.action)#.to(policy_net.tensor_type['device'])
# #     reward_batch = torch.cat(batch.reward).reshape([-1,2]) #.to(policy_net.tensor_type['device'])
# #
# #     # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
# #     # columns of actions taken. These are the actions which would've been taken
# #     # for each batch state according to policy_net
# #     # state_action_values = policy_net(state_batch).gather(1, action_batch)
# #     state_action_values = policy_net(state_batch).gather(2, action_batch.unsqueeze(1).repeat([1, 2, 1]))
# #
# #
# #     # Compute V(s_{t+1}) for all next states.
# #     # Expected values of actions for non_final_next_states are computed based
# #     # on the "older" target_net; selecting their best reward with max(1)[0].
# #     # This is merged based on the mask, such that we'll have either the expected
# #     # state value or 0 in case the state was final.
# #     next_state_values = torch.zeros([BATCH_SIZE,n_agents], device=policy_net.tensor_type['device'])
# #     with torch.no_grad():
# #         # next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
# #         aJ = target_net.sample_action(non_final_next_states,epsilon=0)
# #         qA_sprime = target_net(non_final_next_states)
# #         next_state_values[non_final_mask] = qA_sprime.gather(2,aJ.unsqueeze(1).repeat(1,2,1)).squeeze()
# #
# #
# #     # Compute the expected Q values
# #     expected_state_action_values = (next_state_values * GAMMA) + reward_batch
# #
# #     # Compute Huber loss
# #     criterion = nn.SmoothL1Loss()
# #     loss = criterion(state_action_values, expected_state_action_values.unsqueeze(-1))
# #
# #     # Optimize the model
# #     optimizer.zero_grad()
# #     loss.backward()
# #     # In-place gradient clipping
# #     torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
# #     optimizer.step()
#
#
#
#
# def optimize_model(policy_net,target_net,optimizer, memory,GAMMA,BATCH_SIZE,lr_scheduler=None,update_iterations=1):
#     if len(memory) < 2*BATCH_SIZE: return
#     n_agents = 2
#     losses = []
#     for _ in range(update_iterations):
#         transitions = memory.sample(BATCH_SIZE)
#         # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
#         # detailed explanation). This converts batch-array of Transitions
#         # to Transition of batch-arrays.
#         batch = Transition(*zip(*transitions))
#
#         # Compute a mask of non-final states and concatenate the batch elements
#         # (a final state would've been the one after which simulation ended)
#         non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
#                                       device=policy_net.tensor_type['device'], dtype=torch.bool)
#         non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
#
#         state_batch = torch.cat(batch.state)#.to(policy_net.tensor_type['device'])
#         action_batch = torch.cat(batch.action)#.to(policy_net.tensor_type['device'])
#         reward_batch = torch.cat(batch.reward).reshape([-1,2]) #.to(policy_net.tensor_type['device'])
#
#         # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
#         # columns of actions taken. These are the actions which would've been taken
#         # for each batch state according to policy_net
#         # state_action_values = policy_net(state_batch).gather(1, action_batch)
#         state_action_values = policy_net(state_batch).gather(2, action_batch.unsqueeze(1).repeat([1, 2, 1]))
#
#
#         # Compute V(s_{t+1}) for all next states.
#         # Expected values of actions for non_final_next_states are computed based
#         # on the "older" target_net; selecting their best reward with max(1)[0].
#         # This is merged based on the mask, such that we'll have either the expected
#         # state value or 0 in case the state was final.
#         next_state_values = torch.zeros([BATCH_SIZE,n_agents], device=policy_net.tensor_type['device'])
#         with torch.no_grad():
#             # next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
#             aJ = target_net.sample_action(non_final_next_states,epsilon=0)
#             qA_sprime = target_net(non_final_next_states)
#             next_state_values[non_final_mask] = qA_sprime.gather(2,aJ.unsqueeze(1).repeat(1,2,1)).squeeze()
#
#
#         # Compute the expected Q values
#         expected_state_action_values = (next_state_values * GAMMA) + reward_batch
#
#         # Compute Huber loss
#         criterion = nn.SmoothL1Loss()
#         loss = criterion(state_action_values, expected_state_action_values.unsqueeze(-1))
#
#         # Optimize the model
#         optimizer.zero_grad()
#         loss.backward()
#         # In-place gradient clipping
#         torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
#         optimizer.step()
#         losses.append(loss.item())
#
#     # mean_loss = loss.item()
#     if lr_scheduler is not None:
#         mean_loss = sum(losses)/len(losses)
#         lr_scheduler.step(mean_loss)
#
# def schedule(start,end,N_start,N_end,N_total,slope=1.0):
#     if start == end: return start * np.ones(N_total)
#     warmup = start * np.ones(N_start)
#     perform = end * np.ones(N_end)
#     N_transition = N_total - N_start - N_end
#     iinv = np.power(1 / (np.linspace(1, 10,N_transition) - 0.1) - 0.1, slope)
#     improve = (start + end) * iinv + end
#     epi_schedule = np.hstack([warmup, improve, perform])
#     return epi_schedule
#
#
#
# def test(env, num_episodes, policy_net):
#     with torch.no_grad():
#         length  = 0
#         psucc   = 0
#         score   = np.zeros(env.n_agents)
#         for episode_i in range(num_episodes):
#             state = env.reset()
#             for t in count():
#                 action = policy_net.sample_action(state, epsilon=0)
#                 next_state, reward, done, _ = env.step(action.squeeze())
#                 score += reward.detach().flatten().cpu().numpy()
#                 state = next_state.clone()
#                 if done: break
#             if env.check_caught(env.current_positions): psucc +=1
#             length += env.step_count
#
#     final_score     = list(score/ num_episodes)
#     final_length    = length/num_episodes
#     final_psucc     = psucc/ num_episodes
#
#     return final_score,final_length,final_psucc
#
#
#
# def sizeof_fmt(num, suffix='B'):
#     ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
#     for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
#         if abs(num) < 1024.0:
#             return "%3.1f %s%s" % (num, unit, suffix)
#         num /= 1024.0
#     return "%.1f %s%s" % (num, 'Yi', suffix)
#
#
# def spawn_env_worker(env, policy_net, epsilon,que, num_iterations = 4):
#     # processes = []
#     # manager = mp.Manager()
#     # que = manager.Queue()
#     # num_proc = 4
#     # plt.ioff()
#     # for iproc in range(num_proc):
#     #     p = mp.Process(target=spawn_env_worker,args=(copy.deepcopy(env), policy_net, epsilon,que,))
#     #     p.start()
#     #     processes.append(p) #spawn_env_worker(iWorld, policy_net, epsilon)
#     #
#     # for p in processes:
#     #     p.join()
#     # que.put(None)
#     # plt.ion()
#     #
#     # for _ in count():
#     #     sample = que.get()
#     #     if sample is None:  break
#     #     else:
#     #         # print(data.shape)
#     #         state = sample[0:6]
#     #         action = sample[6]
#     #         next_state = None if torch.all(sample[7:13]==0) else sample[7:13]
#     #         reward = sample[13:14]
#     #         memory.push(state, action, next_state, reward)
#
#
#     # print(f'start_proc')
#     # for i in range(num_iterations):
#     #     que.put(i*torch.ones(1, 10))
#     # mp.Event().wait()
#     #
#     nVars = 15
#     history = torch.empty([0,nVars])#.share_memory_()
#     observations = torch.empty([0, nVars])
#
#     # history = []
#     for iter in range(num_iterations):
#         # iter_history = []
#         state = env.reset()  # Initialize the environment and get it's state
#         done = False
#         for t in count():
#             action = policy_net.sample_action(state, epsilon)
#             next_state, reward, done, _ = env.step(action.squeeze())
#             # obs = [state, action, None if done else next_state, reward]
#             obs = torch.cat([state.flatten(), action.flatten(), torch.zeros(6).flatten() if done else next_state.flatten(), reward.flatten()])
#             observations = torch.cat([observations, obs.reshape([1, nVars])], dim=0)
#
#             # obs.share_memory_()
#             # que = torch.cat([que,  obs.reshape([1, nVars])], dim=0)
#             # que.put(obs.s)
#
#             # history = torch.cat([history,obs.reshape([1,nVars])],dim=1)
#
#
#             if done:                break
#
#     #return history
#     # print(observations.shape)
#     # history = torch.cat([history, observations], dim=0)
#     que.put(observations)
#
# class ExecutionTimer(object):
#     def __init__(self,max_samples=10_000,sigdig = 3,enable=True):
#         # self.profiles = []
#         self.profiles = {}
#         self.max_samples = max_samples
#         self.sigdig = sigdig
#         self.tstarts = {}
#         self.main_iter = 0
#         self.ENABLE = enable
#
#     def __call__(self, *args, **kwargs):
#         if not self.ENABLE: return None
#         return self.add_profile(*args, **kwargs)
#
#     def preview_percent(self):
#         if not self.ENABLE: return None
#         if len(self.profiles['main']['dur']) > 0:
#             mean_execution_time = np.mean(self.profiles['main']['dur'])
#             print(f'Execution Times [Percent]: ############')
#             for name in self.profiles.keys():
#                 profile_durs = np.array(self.profiles[name]['dur'])
#                 disp = '\t| {:<10}: {:<5}%'.format(name, (np.mean(profile_durs) / mean_execution_time).round(self.sigdig))
#                 print(disp)
#             # print(f'\t| {name}: {(np.mean(ave_durs)/mean_execution_time).round(self.sigdig)}%')
#     # def preview_percent(self):
#     #     if not self.ENABLE: return None
#     #     mean_execution_time =  np.mean(self.profiles['main']['dur'])
#     #     # if len(self.profiles['main']['dur']) > 0:
#     #     print(f'Execution Times [Percent]: ############')
#     #     for name in self.profiles.keys():
#     #         profile_durs = np.array(self.profiles[name]['dur'])
#     #         profile_imains = np.array(self.profiles[name]['main_iter'])
#     #         ave_durs = []
#     #         for imain in  self.profiles['main']['main_iter']:
#     #            idurs = profile_durs[np.where(profile_imains == imain)]
#     #            if len(idurs) > 0: ave_durs.append(np.mean(idurs))
#     #
#     #         disp = '\t| {:<10}: {:<5}%'.format(name,(np.mean(ave_durs)/mean_execution_time).round(self.sigdig))
#     #         print(disp)
#     #         #print(f'\t| {name}: {(np.mean(ave_durs)/mean_execution_time).round(self.sigdig)}%')
#
#     def preview_all(self):
#         if not self.ENABLE: return None
#         print(f'Execution Times: ############')
#         for name in self.profiles.keys():
#             print(f'\t| {name}: {np.mean(self.profiles[name]).round(self.sigdig)}')
#
#     def mean_duration(self,name):
#         if not self.ENABLE: return None
#         return np.mean(self.profiles[name])
#
#
#     def add_profile(self,name,status):
#         if not self.ENABLE: return None
#         assert status == 'start' or status == 'stop', f'incorrect input arg [{status}]'
#         if status == 'start':
#             self.tstarts[name] = time.time()
#             if name not in self.profiles.keys():
#                 self.profiles[name] = {'dur': [], 'main_iter': []}
#         elif status == 'stop':
#             self.profiles[name]['dur'].append(time.time() - self.tstarts[name])
#             self.profiles[name]['main_iter'].append(self.main_iter)
#
#     def main_profile(self,status):
#         if not self.ENABLE: return None
#
#         assert status == 'start' or status == 'stop', f'incorrect input arg [{status}]'
#         name = 'main'
#         if status == 'start':
#             self.tstarts[name] = time.time()
#             if name not in self.profiles.keys():
#                 self.profiles[name] = {'dur':[],'main_iter':[]}
#         elif status == 'stop':
#             self.profiles[name]['dur'].append(time.time() - self.tstarts[name])
#             self.profiles[name]['main_iter'].append(self.main_iter)
#             self.main_iter += 1
#
#
# """