# import numpy as np
# import matplotlib.pyplot as plt
from IDQN import DQN
from utilities.make_env import PursuitEvastionGame
from utilities.learning_utils import optimize_model,test,ReplayMemory, soft_update
from utilities.learning_utils import CPT_Handler
from utilities.training_logger import RL_Logger

def main():
    is_continued = False
    is_loaded = False
    algorithm_name = 'IDQNDELETE'
    algorithm_name += '_extended' if is_continued else ''
    #
    # plt.ioff()
    DQN.preview(5, 'Baseline', algorithm_name)


def subfun():
    pass


if __name__ == "__main__":
    main()
