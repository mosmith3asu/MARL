
from datetime import datetime
import os
import matplotlib.pyplot as plt
import mplcyberpunk
import numpy as np
from scipy import signal, stats
from math import ceil
import warnings


plt.ion()
plt.style.use("cyberpunk")
plt.rcParams['axes.facecolor'] = (0,0,0.1)
plt.rcParams['figure.facecolor'] = (0,0,0.01)

import os

print()


class RL_Logger(object):
    fig = None
    axs = None
    LW = 0.25
    line_reward = None
    line_mreward = None
    line_len = None
    line_mlen = None
    line_psucc = None
    line_mpsucc = None
    axs = None

    def __init__(self,fname_notes=''):
        self.new_plot()
        self.Epi_Reward = []
        self.Epi_Length = []
        self.Epi_Psuccess = []
        self.max_memory_size = 6000
        self.max_memory_resample = 3
        self.keep_n_early_memory = 1800
        self.is_resampled = False
        self.current_episode = 0
        self.psuccess_window = 7
        self.filter_window = 100
        self.auto_draw = False

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


    def new_plot(self):

        if RL_Logger.fig is not None: plt.close(RL_Logger.fig)
        RL_Logger.fig, RL_Logger.axs = plt.subplots(3, 1)
        RL_Logger.fig.set_size_inches(11, 8.5)
        RL_Logger.line_reward, = RL_Logger.axs[0].plot(np.zeros(2), lw=RL_Logger.LW)
        RL_Logger.line_mreward, = RL_Logger.axs[0].plot(np.zeros(2))
        RL_Logger.line_len, = RL_Logger.axs[1].plot(np.zeros(2), lw=RL_Logger.LW)
        RL_Logger.line_mlen, = RL_Logger.axs[1].plot(np.zeros(2))
        RL_Logger.line_psucc, = RL_Logger.axs[2].plot(np.zeros(2), lw=RL_Logger.LW)
        RL_Logger.line_mpsucc, = RL_Logger.axs[2].plot(np.zeros(2))
        RL_Logger.axs[0].set_title('IDQN Training Results')
        RL_Logger.axs[0].set_ylabel('Epi Reward')
        RL_Logger.axs[1].set_ylabel('Epi Length')
        RL_Logger.axs[1].set_ylim([-0.1, 20.1])
        RL_Logger.axs[2].set_ylabel('P(Success)')
        RL_Logger.axs[-1].set_xlabel('Episode')

    def update_save_directory(self,dir): self.save_dir = dir #f'results/IDQN_{fname_notes}/'
    def update_plt_title(self,title): self.axs[0].set_title(title)


    def make_directory(self):
        print(f'Initializing Data Storage')
        try: os.mkdir(self.save_dir),print(f'\t| Making root results directory [{self.save_dir}]...')
        except: print(f'\t| Root results directory already exists [{self.save_dir}]...')

        subdir = self.save_dir + 'recordings/'
        try: os.mkdir(subdir),print(f'\t| Making sub directory [{subdir}]...')
        except: print(f'\t| Sub directory already exists [{subdir}]...')

        # subdir = self.save_dir + 'recordings/idqn'
        # try: os.mkdir(subdir), print(f'\t| Making sub directory [{subdir}]...')
        # except:  print(f'\t| Sub directory already exists [{subdir}]...')
    def draw(self,):

        if self.auto_draw:
            if self.nepi_since_last_draw >= self.draw_every_n_episodes:
                print(f'[Plotting...]')
                self.update_plt_data()
                self.fig.canvas.flush_events()
                self.fig.canvas.draw()
                self.nepi_since_last_draw = 0
            else: self.nepi_since_last_draw += 1
        else:
            print(f'[Plotting...]', end='')
            self.update_plt_data()
            self.fig.canvas.flush_events()
            self.fig.canvas.draw()
            self.nepi_since_last_draw = 0

    def update_plt_data(self):
        warnings.filterwarnings("ignore")
        x = self.xdata

        y,line = self.Epi_Reward,self.line_reward
        line.axes.set_xlim([0, max(x)])
        line.axes.set_ylim([min(y) - 0.1 * abs(min(y)), 1.1 * max(y)])
        line.set_xdata(x), line.set_ydata(y)
        y, line = self.filter(self.Epi_Reward), self.line_mreward
        line.set_xdata(x), line.set_ydata(y)

        y, line = self.Epi_Length,  self.line_len
        line.axes.set_xlim([0, max(x)])
        # line.axes.set_ylim([min(y) - 0.1 * abs(min(y)), 1.1 * max(y)])
        line.set_xdata(x), line.set_ydata(y)
        y, line = self.filter(self.Epi_Length), self.line_mlen
        line.set_xdata(x), line.set_ydata(y)

        y, line = self.Epi_Psuccess, self.line_psucc
        line.axes.set_xlim([0, max(x)])
        line.axes.set_ylim([0, 1])
        line.set_xdata(x), line.set_ydata(y)
        y, line = self.filter(self.Epi_Psuccess), self.line_mpsucc
        line.set_xdata(x), line.set_ydata(y)

        warnings.filterwarnings("default")


    def log_episode(self,cum_reward,episode_length,was_success,buffered=True,episode=None):
        self.check_resample()
        self.Epi_Reward.append(cum_reward)
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

    def tick(self):
        self.fig.canvas.flush_events()
        self.fig.canvas.draw()
    def save(self):
        # path = self.save_dir + self.file_name +self.fname_notes+ self.timestamp
        path = self.save_dir + self.file_name
        plt.savefig( path)
        print(f'Saved logger figure in [{path}]')
        # print("date and time:", self.save_dir + self.file_name + self.timestamp)

    def close(self):
        plt.ioff()
        plt.show()




Logger = RL_Logger()
# if __name__ == "__main__":
#
#     for epi in range(100):
#         logger.episode_update(epi,epi*2,epi*3)
#         logger.draw()
#         time.sleep(0.5)

"""

import time

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal,stats
import time
plt.ion()
logger_fig, logger_axs = plt.subplots(3, 1)
line_reward, = logger_axs[0].plot(np.zeros(2), lw=0.5)
line_mreward, = logger_axs[0].plot(np.zeros(2))
line_len, = logger_axs[1].plot(np.zeros(2), lw=0.5)
line_mlen, = logger_axs[1].plot(np.zeros(2))
line_psucc, = logger_axs[2].plot(np.zeros(2), lw=0.5)
line_mpsucc, = logger_axs[2].plot(np.zeros(2))
logger_axs[0].set_title('IDQN Training Results')
logger_axs[0].set_ylabel('Epi Reward')
logger_axs[1].set_ylabel('Epi Length')
logger_axs[2].set_ylabel('P(Success)')
logger_axs[1].set_xlabel('Episode')

class RL_Logger(object):
    def __init__(self):
        self.Epi_Reward = []
        self.Epi_Length = []
        self.Epi_Psuccess = []
        self.max_memory_size = 6000
        self.max_memory_resample = 3
        self.current_episode = 0
        self.xdata = np.arange(2)

    def draw(self):

        logger_fig.canvas.flush_events()
        logger_fig.canvas.draw()

    def update_plt_data(self):
        x = self.xdata

        y,line = self.Epi_Reward,self.line_reward
        line.axes.set_xlim([0, max(x)])
        line.axes.set_ylim([min(y) - 0.1 * abs(min(y)), 1.1 * max(y)])
        line.set_xdata(x), line.set_ydata(y)
        y, line = self.Epi_Reward, self.line_mreward
        line.set_xdata(x), line.set_ydata(y)

        y, line = self.Epi_Length,  self.line_len
        line.axes.set_xlim([0, max(x)])
        line.axes.set_ylim([min(y) - 0.1 * abs(min(y)), 1.1 * max(y)])
        line.set_xdata(x), line.set_ydata(y)
        y, line = self.Epi_Length, self.line_mlen
        line.set_xdata(x), line.set_ydata(y)

        y, line = self.Epi_Psuccess, self.line_psucc
        line.axes.set_xlim([0, max(x)])
        line.axes.set_ylim([min(y) - 0.1 * abs(min(y)), 1.1 * max(y)])
        line.set_xdata(x), line.set_ydata(y)
        y, line = self.Epi_Psuccess, self.line_mpsucc
        line.set_xdata(x), line.set_ydata(y)





    # def iter_update(self,reward,episode_length,p_success):
    #     self.Epi_Reward += reward
    #     self.Epi_Length = episode_length
    #     self.Epi_Psuccess = p_success
    def episode_update(self,cum_reward,episode_length,p_succes):
        self.Epi_Reward.append(cum_reward)
        self.Epi_Length.append(episode_length)
        self.Epi_Psuccess.append(p_succes)
        self.current_episode += 1
        self.xdata = np.linspace(0, self.current_episode, len(self.Epi_Reward))


    def check_resample(self):
        if len(self.Epi_Reward) > self.max_memory_size:
            n_resample = self.max_memory_resample
            self.Epi_Reward = list(self.filter(self.Epi_Reward, window=max(n_resample, 5))[::n_resample])
            self.Epi_Length = list(self.filter(self.Epi_Length, window=max(n_resample, 5))[::n_resample])

    def filter(self,data,window):
        if window % 2 == 0: window += 1
        if 5 * len(data) > window:
            buff0 = np.mean(data[int(window / 4):]) * np.ones(window)
            buff1 = np.mean(data[:-int(window / 4)]) * np.ones(window)
            tmp_data = np.hstack([buff0, data, buff1])
            filt = stats.norm.pdf(np.arange(window), loc=window / 2, scale=window / 5)
            new_data = signal.fftconvolve(tmp_data, filt / np.sum(filt), mode='full')
            ndiff = np.abs(len(data) - len(new_data))
            new_data = new_data[int(ndiff / 2):-int(ndiff / 2)]
        else:
            new_data = data
        return new_data




if __name__ == "__main__":
    logger = RL_Logger()
    for epi in range(100):
        print(epi)
        logger.episode_update(epi,epi*2,epi*3)
        logger.draw()
        time.sleep(0.5)

"""

#
# from datetime import datetime
# import os
# import matplotlib.pyplot as plt
# import mplcyberpunk
# import numpy as np
# from scipy import signal, stats
# from math import ceil
# plt.ion()
# plt.style.use("cyberpunk")
# plt.rcParams['axes.facecolor'] = (0,0,0.1)
# plt.rcParams['figure.facecolor'] = (0,0,0.01)
#
# import os
#
# print()
#
#
# class RL_Logger(object):
#     fig, axs = plt.subplots(3, 1)
#     fig.set_size_inches(11, 8.5)
#     LW = 0.25
#     line_reward, = axs[0].plot(np.zeros(2), lw=LW)
#     line_mreward, = axs[0].plot(np.zeros(2))
#     line_len, = axs[1].plot(np.zeros(2), lw=LW)
#     line_mlen, = axs[1].plot(np.zeros(2))
#     line_psucc, = axs[2].plot(np.zeros(2), lw=LW)
#     line_mpsucc, = axs[2].plot(np.zeros(2))
#     axs[0].set_title('IDQN Training Results')
#     axs[0].set_ylabel('Epi Reward')
#     axs[1].set_ylabel('Epi Length')
#     axs[1].set_ylim([-0.1,20.1])
#     axs[2].set_ylabel('P(Success)')
#     axs[-1].set_xlabel('Episode')
#
#     def __init__(self,fname_notes=''):
#
#
#         self.Epi_Reward = []
#         self.Epi_Length = []
#         self.Epi_Psuccess = []
#         self.max_memory_size = 6000
#         self.max_memory_resample = 3
#         self.keep_n_early_memory = 1800
#         self.is_resampled = False
#         self.current_episode = 0
#         self.psuccess_window = 7
#         self.filter_window = 100
#         self.auto_draw = False
#
#         self._psuccess_buffer = []
#
#         self.nepi_since_last_draw = 0
#         self.draw_every_n_episodes = 0
#
#         self.xdata = np.arange(2)
#         self.timestamp = datetime.now().strftime("--%b%d--h%H-m%M")
#         self.fname_notes = fname_notes
#         # self.save_dir = f'results/IDQN_{self.fname_notes}_{self.timestamp}/recordings/idqn/'
#         # self.save_dir = f'results/IDQN_{self.fname_notes}_{self.timestamp}/'
#
#         self.project_root = os.getcwd().split('MARL')[0]+'MARL\\'
#         self.save_dir = self.project_root + f'results\\IDQN_{self.fname_notes}\\'
#         self.file_name = 'Fig_IDQN'
#
#     def update_save_directory(self,dir): self.save_dir = dir #f'results/IDQN_{fname_notes}/'
#     def update_plt_title(self,title): self.axs[0].set_title(title)
#
#
#     def make_directory(self):
#         print(f'Initializing Data Storage')
#         try: os.mkdir(self.save_dir),print(f'\t| Making root results directory [{self.save_dir}]...')
#         except: print(f'\t| Root results directory already exists [{self.save_dir}]...')
#
#         subdir = self.save_dir + 'recordings/'
#         try: os.mkdir(subdir),print(f'\t| Making sub directory [{subdir}]...')
#         except: print(f'\t| Sub directory already exists [{subdir}]...')
#
#         # subdir = self.save_dir + 'recordings/idqn'
#         # try: os.mkdir(subdir), print(f'\t| Making sub directory [{subdir}]...')
#         # except:  print(f'\t| Sub directory already exists [{subdir}]...')
#     def draw(self,):
#
#         if self.auto_draw:
#             if self.nepi_since_last_draw >= self.draw_every_n_episodes:
#                 print(f'[Plotting...]')
#                 self.update_plt_data()
#                 self.fig.canvas.flush_events()
#                 self.fig.canvas.draw()
#                 self.nepi_since_last_draw = 0
#             else: self.nepi_since_last_draw += 1
#         else:
#             print(f'[Plotting...]', end='')
#             self.update_plt_data()
#             self.fig.canvas.flush_events()
#             self.fig.canvas.draw()
#             self.nepi_since_last_draw = 0
#
#     def update_plt_data(self):
#         x = self.xdata
#
#         y,line = self.Epi_Reward,self.line_reward
#         line.axes.set_xlim([0, max(x)])
#         line.axes.set_ylim([min(y) - 0.1 * abs(min(y)), 1.1 * max(y)])
#         line.set_xdata(x), line.set_ydata(y)
#         y, line = self.filter(self.Epi_Reward), self.line_mreward
#         line.set_xdata(x), line.set_ydata(y)
#
#         y, line = self.Epi_Length,  self.line_len
#         line.axes.set_xlim([0, max(x)])
#         # line.axes.set_ylim([min(y) - 0.1 * abs(min(y)), 1.1 * max(y)])
#         line.set_xdata(x), line.set_ydata(y)
#         y, line = self.filter(self.Epi_Length), self.line_mlen
#         line.set_xdata(x), line.set_ydata(y)
#
#         y, line = self.Epi_Psuccess, self.line_psucc
#         line.axes.set_xlim([0, max(x)])
#         line.axes.set_ylim([0, 1])
#         line.set_xdata(x), line.set_ydata(y)
#         y, line = self.filter(self.Epi_Psuccess), self.line_mpsucc
#         line.set_xdata(x), line.set_ydata(y)
#
#
#
#     def log_episode(self,cum_reward,episode_length,was_success,buffered=True,episode=None):
#         self.check_resample()
#         self.Epi_Reward.append(cum_reward)
#         self.Epi_Length.append(episode_length)
#
#         # update probability of success
#         if buffered:
#             self._psuccess_buffer.append(int(was_success))
#             if len(self._psuccess_buffer) > self.psuccess_window: self._psuccess_buffer.pop(0)
#             psuccess = np.mean(self._psuccess_buffer)
#             self.Epi_Psuccess.append(psuccess)
#         else:
#             self.Epi_Psuccess.append(was_success)
#
#         if episode is None:  self.current_episode += 1
#         else: self.current_episode = episode
#         # self.xdata = np.linspace(0, self.current_episode, len(self.Epi_Reward))
#
#         if self.is_resampled:
#             n_keep =self.keep_n_early_memory
#             xdata_keep = np.arange(n_keep)
#             xdata_resampled = np.linspace(n_keep, self.current_episode, len(self.Epi_Reward)-n_keep)
#             self.xdata = np.hstack([xdata_keep,xdata_resampled])
#         else:
#             self.xdata = np.linspace(0, self.current_episode, len(self.Epi_Reward))
#
#         if self.auto_draw: self.draw()
#
#
#     def check_resample(self):
#         if len(self.Epi_Reward) > self.max_memory_size:
#             n_keep = self.keep_n_early_memory
#             n_resample = self.max_memory_resample
#             _Epi_Reward = list(self.filter(self.Epi_Reward, window=max(n_resample, 3)))
#             _Epi_Length = list(self.filter(self.Epi_Length, window=max(n_resample, 3)))
#             _Epi_Psuccess = list(self.filter(self.Epi_Psuccess, window=max(n_resample, 3)))
#
#             self.Epi_Reward     = _Epi_Reward[:n_keep] + _Epi_Reward[n_keep::n_resample]
#             self.Epi_Length     = _Epi_Length[:n_keep] + _Epi_Length[n_keep::n_resample]
#             self.Epi_Psuccess   = _Epi_Psuccess[:n_keep] + _Epi_Psuccess[n_keep::n_resample]
#             self.is_resampled = True
#             # print(f'[Resampling logger...]')
#
#     def filter(self,data,window=None):
#         if window is None: window = self.filter_window
#         if window % 2 == 0: window += 1
#         if len(data) > window:
#             buff0 = np.mean(data[ceil(window / 4):]) * np.ones(window)
#             buff1 = np.mean(data[:-ceil(window / 4)]) * np.ones(window)
#             tmp_data = np.hstack([buff0, data, buff1])
#             filt = stats.norm.pdf(np.arange(window), loc=window / 2, scale=window / 5)
#             new_data = signal.fftconvolve(tmp_data, filt / np.sum(filt), mode='full')
#             ndiff = np.abs(len(data) - len(new_data))
#             new_data = new_data[int(ndiff / 2):-int(ndiff / 2)]
#         else: new_data = data
#         # filt = signal.gaussian(window, std=3)
#         # filt = filt/np.sum(filt)
#         # new_data = signal.fftconvolve(data, filt, mode='same')
#         return new_data
#
#
#     def save(self):
#         # path = self.save_dir + self.file_name +self.fname_notes+ self.timestamp
#         path = self.save_dir + self.file_name
#         plt.savefig( path)
#         print(f'Saved logger figure in [{path}]')
#         # print("date and time:", self.save_dir + self.file_name + self.timestamp)
#
#     def close(self):
#         plt.ioff()
#         plt.show()
#
#
#
#
# Logger = RL_Logger()
# # if __name__ == "__main__":
# #
# #     for epi in range(100):
# #         logger.episode_update(epi,epi*2,epi*3)
# #         logger.draw()
# #         time.sleep(0.5)
#
# """
#
# import time
#
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import signal,stats
# import time
# plt.ion()
# logger_fig, logger_axs = plt.subplots(3, 1)
# line_reward, = logger_axs[0].plot(np.zeros(2), lw=0.5)
# line_mreward, = logger_axs[0].plot(np.zeros(2))
# line_len, = logger_axs[1].plot(np.zeros(2), lw=0.5)
# line_mlen, = logger_axs[1].plot(np.zeros(2))
# line_psucc, = logger_axs[2].plot(np.zeros(2), lw=0.5)
# line_mpsucc, = logger_axs[2].plot(np.zeros(2))
# logger_axs[0].set_title('IDQN Training Results')
# logger_axs[0].set_ylabel('Epi Reward')
# logger_axs[1].set_ylabel('Epi Length')
# logger_axs[2].set_ylabel('P(Success)')
# logger_axs[1].set_xlabel('Episode')
#
# class RL_Logger(object):
#     def __init__(self):
#         self.Epi_Reward = []
#         self.Epi_Length = []
#         self.Epi_Psuccess = []
#         self.max_memory_size = 6000
#         self.max_memory_resample = 3
#         self.current_episode = 0
#         self.xdata = np.arange(2)
#
#     def draw(self):
#
#         logger_fig.canvas.flush_events()
#         logger_fig.canvas.draw()
#
#     def update_plt_data(self):
#         x = self.xdata
#
#         y,line = self.Epi_Reward,self.line_reward
#         line.axes.set_xlim([0, max(x)])
#         line.axes.set_ylim([min(y) - 0.1 * abs(min(y)), 1.1 * max(y)])
#         line.set_xdata(x), line.set_ydata(y)
#         y, line = self.Epi_Reward, self.line_mreward
#         line.set_xdata(x), line.set_ydata(y)
#
#         y, line = self.Epi_Length,  self.line_len
#         line.axes.set_xlim([0, max(x)])
#         line.axes.set_ylim([min(y) - 0.1 * abs(min(y)), 1.1 * max(y)])
#         line.set_xdata(x), line.set_ydata(y)
#         y, line = self.Epi_Length, self.line_mlen
#         line.set_xdata(x), line.set_ydata(y)
#
#         y, line = self.Epi_Psuccess, self.line_psucc
#         line.axes.set_xlim([0, max(x)])
#         line.axes.set_ylim([min(y) - 0.1 * abs(min(y)), 1.1 * max(y)])
#         line.set_xdata(x), line.set_ydata(y)
#         y, line = self.Epi_Psuccess, self.line_mpsucc
#         line.set_xdata(x), line.set_ydata(y)
#
#
#
#
#
#     # def iter_update(self,reward,episode_length,p_success):
#     #     self.Epi_Reward += reward
#     #     self.Epi_Length = episode_length
#     #     self.Epi_Psuccess = p_success
#     def episode_update(self,cum_reward,episode_length,p_succes):
#         self.Epi_Reward.append(cum_reward)
#         self.Epi_Length.append(episode_length)
#         self.Epi_Psuccess.append(p_succes)
#         self.current_episode += 1
#         self.xdata = np.linspace(0, self.current_episode, len(self.Epi_Reward))
#
#
#     def check_resample(self):
#         if len(self.Epi_Reward) > self.max_memory_size:
#             n_resample = self.max_memory_resample
#             self.Epi_Reward = list(self.filter(self.Epi_Reward, window=max(n_resample, 5))[::n_resample])
#             self.Epi_Length = list(self.filter(self.Epi_Length, window=max(n_resample, 5))[::n_resample])
#
#     def filter(self,data,window):
#         if window % 2 == 0: window += 1
#         if 5 * len(data) > window:
#             buff0 = np.mean(data[int(window / 4):]) * np.ones(window)
#             buff1 = np.mean(data[:-int(window / 4)]) * np.ones(window)
#             tmp_data = np.hstack([buff0, data, buff1])
#             filt = stats.norm.pdf(np.arange(window), loc=window / 2, scale=window / 5)
#             new_data = signal.fftconvolve(tmp_data, filt / np.sum(filt), mode='full')
#             ndiff = np.abs(len(data) - len(new_data))
#             new_data = new_data[int(ndiff / 2):-int(ndiff / 2)]
#         else:
#             new_data = data
#         return new_data
#
#
#
#
# if __name__ == "__main__":
#     logger = RL_Logger()
#     for epi in range(100):
#         print(epi)
#         logger.episode_update(epi,epi*2,epi*3)
#         logger.draw()
#         time.sleep(0.5)
#
# """