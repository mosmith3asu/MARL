import math
from datetime import datetime
import matplotlib.pyplot as plt

import numpy as np
from scipy import signal, stats
from math import ceil
import warnings

from matplotlib.widgets import Button

# import mplcyberpunk
# plt.style.use("cyberpunk")
# plt.rcParams['axes.facecolor'] = (0,0,0.1)
# plt.rcParams['figure.facecolor'] = (0,0,0.01)

import os




class RL_Logger(object):
    fig = None
    axs = None
    lw_small = 0.25
    lw_med = 1
    agent_colors = ['r', 'b']
    epi_length_color = 'g'
    epi_psucc_color = 'm'
    ls_raw = '--'
    ls_filt = '-'
    enable_legend = True


    reward_raw_config0  = {'lw': lw_small,   'c': agent_colors[0], 'ls': ls_raw,'label':['raw: $r_{R}$','raw: $r_{H}$'][0]}
    reward_filt_config0 = {'lw': lw_med,     'c': agent_colors[0], 'ls': ls_filt,'label':['filtered: $r_{R}$','filtered: $r_{H}$'][0]}
    reward_raw_config1  = {'lw': lw_small,   'c': agent_colors[1], 'ls': ls_raw, 'label': ['raw: $r_{R}$', 'raw: $r_{H}$'][1]}
    reward_filt_config1 = {'lw': lw_med,     'c': agent_colors[1], 'ls': ls_filt, 'label': ['filtered: $r_{R}$', 'filtered: $r_{H}$'][1]}


    length_raw_config = {'lw': lw_small, 'c': epi_length_color, 'ls': ls_raw, 'label': 'raw: epi len'}
    length_filt_config = {'lw': lw_med, 'c': epi_length_color, 'ls': ls_filt, 'label': 'filtered: epi len'}
    psucc_raw_config = {'lw': lw_small, 'c': epi_psucc_color, 'ls': ls_raw, 'label': 'raw: P(Success)'}
    psucc_filt_config = {'lw': lw_med, 'c': epi_psucc_color, 'ls': ls_filt, 'label': 'filtered: P(Success)'}


    # line_reward = None
    # line_mreward = None
    raw_reward_lines = [None,None]
    filt_reward_lines = [None,None]

    line_len = None
    line_mlen = None
    line_psucc = None
    line_mpsucc = None

    ax_button = None
    close_button = None


    def __init__(self,fname_notes=''):
        plt.ion()
        if RL_Logger.fig is None:
            self.new_plot()
        self.Epi_Reward = []
        self.Epi_Length = []
        self.Epi_Psuccess = []
        # self.Epi_Reward = np.zeros([0,2],dtype=np)
        # self.Epi_Length = np.zeros([0,1])
        # self.Epi_Psuccess = np.zeros([0,1])

        self.max_memory_size = 4000
        self.max_memory_resample = 3
        self.keep_n_early_memory = 1800
        self.is_resampled = False
        self.current_episode = 0
        self.psuccess_window = 7
        self.filter_window = 100
        self.auto_draw = False
        self.itick = 0
        self.refresh_rate = 1

        self.end_button_state = 0

        self._psuccess_buffer = []

        self.nepi_since_last_draw = 0
        self.draw_every_n_episodes = 0

        self.xdata = np.arange(2)
        self.timestamp = datetime.now().strftime("--%b%d--h%H-m%M")
        self.fname_notes = fname_notes
        # self.save_dir = f'results/IDQN_{self.fname_notes}_{self.timestamp}/recordings/idqn/'
        # self.save_dir = f'results/IDQN_{self.fname_notes}_{self.timestamp}/'

        self.project_root = os.getcwd().split('MARL')[0]+'MARL\\'
        self.save_dir = self.project_root + f'results\\IDQN_{self.fname_notes}\\'
        self.file_name = 'Fig_IDQN'


    def end_button_callback(self,event):
        self.end_button_state = 1

    def new_plot(self):
        # plt.close(RL_Logger.fig)
        dummy_data = np.zeros([3,1])
        # plt.clf()

        # if RL_Logger.fig is not None:
            # plt.clf()
            # plt.close(RL_Logger.fig)
        RL_Logger.fig, RL_Logger.axs = plt.subplots(3, 1,constrained_layout=True)
        RL_Logger.fig.set_size_inches(11, 8.5)

        # RL_Logger.raw_reward_lines = RL_Logger.axs[0].plot(np.tile(dummy_data,[1,2]),**RL_Logger.reward_raw_config)
        # RL_Logger.filt_reward_lines = RL_Logger.axs[0].plot(np.tile(dummy_data,[1,2]),**RL_Logger.reward_filt_config)
        RL_Logger.raw_reward_lines[0] = RL_Logger.axs[0].plot(dummy_data, **RL_Logger.reward_raw_config0)[0]
        RL_Logger.filt_reward_lines[0] = RL_Logger.axs[0].plot(dummy_data, **RL_Logger.reward_filt_config0)[0]
        RL_Logger.raw_reward_lines[1] = RL_Logger.axs[0].plot(dummy_data, **RL_Logger.reward_raw_config1)[0]
        RL_Logger.filt_reward_lines[1] = RL_Logger.axs[0].plot(dummy_data, **RL_Logger.reward_filt_config1)[0]

        RL_Logger.line_len, = RL_Logger.axs[1].plot(np.zeros(2), **RL_Logger.length_raw_config)
        RL_Logger.line_mlen, = RL_Logger.axs[1].plot(np.zeros(2), **RL_Logger.length_filt_config)
        RL_Logger.line_psucc, = RL_Logger.axs[2].plot(np.zeros(2), **RL_Logger.psucc_raw_config)
        RL_Logger.line_mpsucc, = RL_Logger.axs[2].plot(np.zeros(2),**RL_Logger.psucc_filt_config)

        if RL_Logger.enable_legend:
            for i in range(len(RL_Logger.axs)): RL_Logger.axs[i].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

        RL_Logger.axs[0].set_title('IDQN Training Results')
        RL_Logger.axs[0].set_ylabel('Epi Reward')
        RL_Logger.axs[1].set_ylabel('Epi Length')
        RL_Logger.axs[1].set_ylim([-0.1, 20.1])
        RL_Logger.axs[2].set_ylabel('P(Success)')
        RL_Logger.axs[-1].set_xlabel('Episode')


        w = 0.15
        h = 0.075
        RL_Logger.ax_button = RL_Logger.fig.add_axes([1-1.1*w, 0.75*h, w, h])
        RL_Logger.close_button = Button(RL_Logger.ax_button, 'end training')
        RL_Logger.close_button.on_clicked(self.end_button_callback)


    def update_save_directory(self,dir): self.save_dir = dir #f'results/IDQN_{fname_notes}/'
    def update_plt_title(self,title): self.axs[0].set_title(title)


    def make_directory(self):
        print(f'Initializing Data Storage')
        try: os.mkdir(self.save_dir),print(f'\t| Making root results directory [{self.save_dir}]...')
        except: print(f'\t| Root results directory already exists [{self.save_dir}]...')

        # subdir = self.save_dir + 'recordings/'
        # try: os.mkdir(subdir),print(f'\t| Making sub directory [{subdir}]...')
        # except: print(f'\t| Sub directory already exists [{subdir}]...')

        # subdir = self.save_dir + 'recordings/idqn'
        # try: os.mkdir(subdir), print(f'\t| Making sub directory [{subdir}]...')
        # except:  print(f'\t| Sub directory already exists [{subdir}]...')
    def draw(self,verbose=False):

        if self.auto_draw:
            if self.nepi_since_last_draw >= self.draw_every_n_episodes:
                if verbose: print(f'[Plotting...]')
                self.update_plt_data()
                self.fig.canvas.flush_events()
                self.fig.canvas.draw()
                self.nepi_since_last_draw = 0
            else: self.nepi_since_last_draw += 1
        else:
            if verbose: print(f'[Plotting...]', end='')
            self.update_plt_data()
            self.fig.canvas.flush_events()
            self.fig.canvas.draw()
            self.nepi_since_last_draw = 0



    def update_data(self,line, xdata, ydata,yscale=None):
        line.axes.relim()
        line.axes.autoscale_view()
        if yscale is not None:
            line.axes.set_ylim(yscale)
        line.set_xdata(xdata), line.set_ydata(ydata)

    def update_plt_data(self):
        warnings.filterwarnings("ignore")
        x = self.xdata

        rewardsK = np.array(self.Epi_Reward)
        for il in range(len(self.raw_reward_lines)):
            self.update_data(self.raw_reward_lines[il],x, rewardsK[:,il])
            self.update_data(self.filt_reward_lines[il], x, self.filter(rewardsK[:,il]))
        self.update_data(self.line_len, x, self.Epi_Length,yscale=[-0.1,20.1])
        self.update_data(self.line_mlen, x, self.filter(self.Epi_Length),yscale=[-0.1,20.1])
        self.update_data(self.line_psucc, x, self.Epi_Psuccess,yscale=[-0.1,1.1])
        self.update_data(self.line_mpsucc, x, self.filter( self.Epi_Psuccess),yscale=[-0.1,1.1])

        warnings.filterwarnings("default")


    def log_episode(self,agent_reward,episode_length,was_success,buffered=True,episode=None):
        self.check_resample()
        self.Epi_Reward.append(agent_reward)
        self.Epi_Length.append(episode_length)


        # update probability of success
        if buffered:
            self._psuccess_buffer.append(int(was_success))
            if len(self._psuccess_buffer) > self.psuccess_window: self._psuccess_buffer.pop(0)
            psuccess = np.mean(self._psuccess_buffer)
            self.Epi_Psuccess.append(psuccess)
        else:
            self.Epi_Psuccess.append(was_success)

        if episode is None:  self.current_episode += 1
        else: self.current_episode = episode
        # self.xdata = np.linspace(0, self.current_episode, len(self.Epi_Reward))

        if self.is_resampled:
            n_keep =self.keep_n_early_memory
            xdata_keep = np.arange(n_keep)
            xdata_resampled = np.linspace(n_keep, self.current_episode, len(self.Epi_Reward)-n_keep)
            self.xdata = np.hstack([xdata_keep,xdata_resampled])
        else:
            self.xdata = np.linspace(0, self.current_episode, len(self.Epi_Reward))

        if self.auto_draw: self.draw()


    def check_resample(self):
        if len(self.Epi_Reward) > self.max_memory_size:
            n_keep = self.keep_n_early_memory
            n_resample = self.max_memory_resample
            _Epi_Reward = list(self.filter(self.Epi_Reward, window=max(n_resample, 3)))
            _Epi_Length = list(self.filter(self.Epi_Length, window=max(n_resample, 3)))
            _Epi_Psuccess = list(self.filter(self.Epi_Psuccess, window=max(n_resample, 3)))

            self.Epi_Reward     = _Epi_Reward[:n_keep] + _Epi_Reward[n_keep::n_resample]
            self.Epi_Length     = _Epi_Length[:n_keep] + _Epi_Length[n_keep::n_resample]
            self.Epi_Psuccess   = _Epi_Psuccess[:n_keep] + _Epi_Psuccess[n_keep::n_resample]
            self.is_resampled = True
            # print(f'[Resampling logger...]')

    def filter(self,data,window=None):
        if window is None: window = self.filter_window
        if window % 2 == 0: window += 1
        if len(data) > window:
            buff0 = np.mean(data[ceil(window / 4):]) * np.ones(window)
            buff1 = np.mean(data[:-ceil(window / 4)]) * np.ones(window)
            tmp_data = np.hstack([buff0, data, buff1])
            filt = stats.norm.pdf(np.arange(window), loc=window / 2, scale=window / 5)
            new_data = signal.fftconvolve(tmp_data, filt / np.sum(filt), mode='full')
            ndiff = np.abs(len(data) - len(new_data))
            new_data = new_data[int(ndiff / 2):-int(ndiff / 2)]
        else: new_data = data
        # filt = signal.gaussian(window, std=3)
        # filt = filt/np.sum(filt)
        # new_data = signal.fftconvolve(data, filt, mode='same')
        return new_data

    def flush(self):
        RL_Logger.fig.canvas.flush_events()

    def tick(self):
        if self.itick % self.refresh_rate==0:
            self.fig.canvas.flush_events()
            self.fig.canvas.draw()
        self.itick += 1
    def save(self):
        # path = self.save_dir + self.file_name +self.fname_notes+ self.timestamp
        path = self.save_dir + self.file_name
        plt.savefig( path)
        print(f'Saved logger figure in [{path}]')
        # print("date and time:", self.save_dir + self.file_name + self.timestamp)

    def close(self):
        self.end_button_state = 0
        self.Epi_Reward = []
        self.Epi_Length = []
        self.Epi_Psuccess = []
        self._psuccess_buffer = []
        self.timestamp = datetime.now().strftime("--%b%d--h%H-m%M")
        self.project_root = os.getcwd().split('MARL')[0] + 'MARL\\'
        self.save_dir = self.project_root + f'results\\IDQN_{self.fname_notes}\\'
        self.file_name = 'Fig_IDQN'

        # self.new_plot()
        dummy_y = np.zeros([3, 1])
        dummy_x = np.arange(3).reshape(dummy_y.shape)

        for il in range(len(self.raw_reward_lines)):
            self.update_data(self.raw_reward_lines[il], dummy_x, dummy_y)
            self.update_data(self.filt_reward_lines[il], dummy_x, dummy_y)
        self.update_data(self.line_len, dummy_x, dummy_y, yscale=[-0.1, 20.1])
        self.update_data(self.line_mlen, dummy_x, dummy_y, yscale=[-0.1, 20.1])
        self.update_data(self.line_psucc, dummy_x, dummy_y, yscale=[-0.1, 1.1])
        self.update_data(self.line_mpsucc, dummy_x, dummy_y, yscale=[-0.1, 1.1])


        # plt.ioff()
        # plt.show()

    def blocking_preview(self):
        plt.ioff()
        plt.show()



if __name__ == "__main__":
    import time

    Logger = RL_Logger()
    for trial in range(3):
        for epi in range(15):
            agent_rewards = [math.sin(epi),math.cos(epi)]
            length = epi % 20
            psucc = epi
            Logger.log_episode(agent_rewards,length,psucc)
            Logger.draw()
            time.sleep(0.1)
        Logger.close()
