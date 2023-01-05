import numpy as np
import matplotlib.pyplot as plt
from IDQN_GPU.utilities.learning_utils import schedule

EPS_START = 1.0
EPS_END = 0.05

num_episodes = 20_000
EPS_DECAY0 = int(num_episodes/20)
EPS_DECAY1 = int(num_episodes/5)
EPS_EXPLORE = int(num_episodes/10)

PEN_SCALE_DECAY = 1000
PEN_SCALE_START = 0
PEN_SCALE_END = 1
PEN_SCALE_EXPLORE = 1000

EPS_SLOPE = 1/3
episodes = np.arange(num_episodes)

_episodes = np.hstack([np.zeros(EPS_EXPLORE), np.arange(num_episodes - EPS_EXPLORE)])
decay_epsilon0 = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * _episodes / EPS_DECAY0)
decay_epsilon1 = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * _episodes / EPS_DECAY1)

_episodes = np.hstack([np.zeros(PEN_SCALE_EXPLORE), np.arange(num_episodes - PEN_SCALE_EXPLORE)])
pen_scale = PEN_SCALE_END + (PEN_SCALE_START - PEN_SCALE_END) * np.exp(-1. * _episodes / PEN_SCALE_DECAY)

# decay_epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * episodes / EPS_DECAY)
schedule_epsilons = schedule(EPS_START, EPS_END, 0, 0, num_episodes, slope=EPS_SLOPE)

plt.plot(schedule_epsilons,label='schedule')
plt.plot(decay_epsilon0,label='decay0')
plt.plot(decay_epsilon1,label='decay1')
plt.plot(pen_scale,label='pen_scale')
plt.legend()
plt.show()


# from algorithm.idqn.idqn import run
# if __name__ == "__main__":
#
#     config = {
#               'iWorld': 2,
#               'lr': 0.0005,
#               'batch_size': 32,
#               'gamma': 0.95,
#               'buffer_limit': 50000,
#               'log_interval': 100,
#               'max_episodes': 20000,
#               'max_epsilon': 0.8,
#               'min_epsilon': 0.1,
#               'test_episodes': 5,
#               'warm_up_steps': 1000,
#               'no_penalty_steps': 3000,
#               'update_iter': 10,
#               'monitor': False,
#     }
#
#
#     BATCH_WORLDS = [1,2,3,4,5,6,7]
#     # BATCH_WORLDS = [ 4]
#     for iworld in BATCH_WORLDS:
#         config['iWorld'] = iworld
#         run(**config)
#
#
#
#
#     """
#     default_config = {  # 'env_name': 'ma_gym:Switch2-v1',
#         'iWorld': 1,
#         'lr': 0.0005,
#         'batch_size': 32,
#         'gamma': 0.99,
#         'buffer_limit': 50000,
#         'log_interval': 20,
#         'max_episodes': 30000,
#         'max_epsilon': 0.9,
#         'min_epsilon': 0.1,
#         'test_episodes': 5,
#         'warm_up_steps': 1000,
#         'no_penalty_steps': 2000,
#         'update_iter': 10,
#         'monitor': True,
#     }
#     """