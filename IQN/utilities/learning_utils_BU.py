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

        return f'CPT({round(self.paccept_sensitivity,sigdig)}):{disp} => [{favor}]={dfavor}x'# G{TG} x L{TL}'

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
        preference = p_accept - 0.5 # >0 indicates prefer gamble

        cbar_loc = 'right'
        cbar_pad = 0.01
        cbar_tickrot = 90
        cbar_va = 'center'
        cbar_ha = 'left' if cbar_tickrot == 0 else 'center'
        tick_labels = ['\n0%\n(reject)', '\n50%\n(indiff)', '\n100%\n(accept)']
        x_label = '$\mathbb{C}[R_{gamble} \; | \; p_{\\rho}=0.5 ]$' # r_{\\rho}
        y_label = '$\mathbb{C}[R_{certainty}]$'
        param_string =  'AUIC:{:>3}'.format(np.round(np.mean(preference),1)) + \
                        '\n$\mathcal{P}_{CPT}$:'+\
                        '\n -$b$:{:>5}'.format(round(self.b,1)) + \
                        '\n -$\lambda$:{:>5}'.format(round(self.lam,1)) + \
                        '\n -$\gamma^+$:{:>5}'.format(round(self.yg,1)) + \
                        '\n -$\gamma^-$:{:>5}'.format(round(self.yl,1)) + \
                        '\n -$\delta^+$:{:>5}'.format(round(self.dg,1)) + \
                        '\n -$\delta^-$:{:>5}'.format(round(self.dl,1))


        im = ax.matshow(preference, cmap='bwr', origin='lower')
        cbar = plt.colorbar(im,ax=ax, ticks=[np.min(preference), 0, np.max(preference)],location=cbar_loc,pad =cbar_pad)

        # cbar.ax.set_yticklabels( ['\n0%\n(reject)','\n50%\n(indifferent)', '\n100%\n(accept)'],rotation=90,va='center',ha='center')  # vertically oriented colorbar
        cbar.ax.set_yticklabels(tick_labels, rotation=cbar_tickrot,
                                va=cbar_va, ha=cbar_ha)  # vertically oriented colorbar
        # ax.legend()
        # ax.set_title(f'Preference for Accepting Gambles')

        # ' ' +
        ax.set_title( f'Risk-{attribution}')

        ax.annotate(param_string, xy=(0.05, 0.95),
                    xycoords='axes fraction',
                    size=8, ha='left', va='center',
                    bbox=dict(boxstyle='round', fc='w'))

        ax.set_xlabel(x_label)

        ax.set_xticks(np.linspace(0, 100, n_ticks))
        ax.set_yticks(np.linspace(0, 100, n_ticks))
        ax.set_xticklabels(np.round(np.linspace(r_gain[0], r_gain[-1], n_ticks), 1))

        ax.set_ylabel(y_label)
        ax.set_yticklabels(np.round(np.linspace(r_cert[0], r_cert[-1], n_ticks), 1))

    def _get_sensitivity(self, b, lam, yg, yl, dg, dl, rationality, return_paccept=False, N=None):
        iaccept = 0
        N = self.n_test_sensitivity if N is None else N
        rmin, rmax = self.r_range_sensitivity
        r_cert = np.linspace(rmin, rmax, N)  + self.r_pen/2
        r_gain = np.linspace(rmin, rmax, N)  # [0,20]
        r_loss = r_gain + self.r_pen
        p_thresh = self.p_thresh_sensitivity

        p_accept = np.empty([N, N])
        for r in range(N):
            for c in range(N):
                rg = r_gain[c]
                rl = r_loss[c]
                rc = r_cert[r]

                rg_hat, pg_hat = self.transform(rg, 1 - self.p_pen, b, lam, yg, yl, dg, dl)
                rl_hat, pl_hat = self.transform(rl, self.p_pen, b, lam, yg, yl, dg, dl)
                Er_gamble = (rg_hat * pg_hat) + (rl_hat * pl_hat)
                Er_cert = rc - b
                Er_choices = np.array([Er_gamble, Er_cert])

                pCPT = softmax(rationality * Er_choices)
                p_accept[r, c] = pCPT[iaccept]

        p_sum = np.mean(p_accept - 0.5)
        if abs(p_sum) < p_thresh: attribution = 'insensitive'
        elif p_sum >= p_thresh: attribution = 'seeking'
        elif p_sum <= -p_thresh: attribution = 'averse'
        else: raise Exception('Unknown CPT attribution')

        # p_accept = 0
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
        if assume is not None:
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


class ParamScheduler(object):
    def __init__(self,CFG):
        self.schedules = {}
        EPS_START = CFG.eps_schedule.START
        EPS_END = CFG.eps_schedule.END
        EPS_DECAY = CFG.eps_schedule.DECAY
        EPS_EXPLORE = CFG.eps_schedule.EXPLORE
        episodes = np.hstack([np.zeros(EPS_EXPLORE), np.arange(CFG.num_episodes - EPS_EXPLORE)])


        self.schedules['epsilon'] = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * episodes / EPS_DECAY)


        PEN_SCALE_START = CFG.pen_scale_schedule.START
        PEN_SCALE_END = CFG.pen_scale_schedule.END
        PEN_SCALE_DECAY = CFG.pen_scale_schedule.DECAY
        PEN_SCALE_EXPLORE = CFG.pen_scale_schedule.EXPLORE
        episodes = np.hstack([np.zeros(PEN_SCALE_EXPLORE), np.arange(CFG.num_episodes - PEN_SCALE_EXPLORE)])
        self.schedules['penalty'] = PEN_SCALE_END + (PEN_SCALE_START - PEN_SCALE_END) * np.exp(-1. * episodes / PEN_SCALE_DECAY)

        RCATCH_SCALE_START = CFG.rcatch_scale_schedule.START
        RCATCH_SCALE_END = CFG.rcatch_scale_schedule.END
        RCATCH_SCALE_DECAY = CFG.rcatch_scale_schedule.DECAY
        RCATCH_SCALE_EXPLORE = CFG.rcatch_scale_schedule.EXPLORE
        episodes = np.hstack([np.zeros(RCATCH_SCALE_EXPLORE), np.arange(CFG.num_episodes - RCATCH_SCALE_EXPLORE)])
        self.schedules['rcatch'] = RCATCH_SCALE_END + (RCATCH_SCALE_START - RCATCH_SCALE_END) * np.exp(
            -1. * episodes / RCATCH_SCALE_DECAY)
    def get_epi_param(self,iepi,name):
        return self.schedules[name][iepi]
    def __call__(self, *args, **kwargs):
        return self.get_epi_param(*args, **kwargs)


# class ParamScheduler(object):
#     def __init__(self,CFG):
#         self.schedules = {}
#         EPS_START = CFG.eps_schedule['START']
#         EPS_END = CFG.eps_schedule['END']
#         EPS_DECAY = CFG.eps_schedule['DECAY']
#         EPS_EXPLORE = CFG.eps_schedule['EXPLORE']
#         episodes = np.hstack([np.zeros(EPS_EXPLORE), np.arange(CFG.num_episodes - EPS_EXPLORE)])
#
#
#         self.schedules['epsilon'] = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * episodes / EPS_DECAY)
#
#
#         PEN_SCALE_START = CFG.pen_scale_schedule['START']
#         PEN_SCALE_END = CFG.pen_scale_schedule['END']
#         PEN_SCALE_DECAY = CFG.pen_scale_schedule['DECAY']
#         PEN_SCALE_EXPLORE = CFG.pen_scale_schedule['EXPLORE']
#         episodes = np.hstack([np.zeros(PEN_SCALE_EXPLORE), np.arange(CFG.num_episodes - PEN_SCALE_EXPLORE)])
#         self.schedules['penalty'] = PEN_SCALE_END + (PEN_SCALE_START - PEN_SCALE_END) * np.exp(-1. * episodes / PEN_SCALE_DECAY)
#
#         RCATCH_SCALE_START = CFG.rcatch_scale_schedule['START']
#         RCATCH_SCALE_END = CFG.rcatch_scale_schedule['END']
#         RCATCH_SCALE_DECAY = CFG.rcatch_scale_schedule['DECAY']
#         RCATCH_SCALE_EXPLORE = CFG.rcatch_scale_schedule['EXPLORE']
#         episodes = np.hstack([np.zeros(RCATCH_SCALE_EXPLORE), np.arange(CFG.num_episodes - RCATCH_SCALE_EXPLORE)])
#         self.schedules['rcatch'] = RCATCH_SCALE_END + (RCATCH_SCALE_START - RCATCH_SCALE_END) * np.exp(
#             -1. * episodes / RCATCH_SCALE_DECAY)
#     def get_epi_param(self,iepi,name):
#         return self.schedules[name][iepi]
#     def __call__(self, *args, **kwargs):
#         return self.get_epi_param(*args, **kwargs)
#


def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


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
