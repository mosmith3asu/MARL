############################################################
## Packages ################################################
import sys; sys.path.append("../examples")
import logging
import random
import numpy as np
import torch
from QLearning import Qfunction,ReplayMemory
from utilities.make_env import PursuitEvastionGame
from utilities.learning_utils import CPT_Handler, ParamScheduler,EpisodeTimer
from utilities.training_logger import RL_Logger
from utilities.config_manager import CFG


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.set_grad_enabled(False)

def main():
    W = CFG.WORLDS
    P = CFG.policy_type
    algorithm_name = CFG.algorithm_name
    is_continued = CFG.is_continued


    for iworld in (W if isinstance(W,list) else [W]):
        for policy in (P if isinstance(P,list) else [P]):
            # is_loaded = True if policy in ['Averse', 'Seeking'] else False
            is_loaded = False
            run_IDQN(iworld, algorithm_name=algorithm_name, policy_type=policy, is_loaded=is_loaded, continued=is_continued)



#################################################################
## Training algorithm ###########################################
def run_IDQN(iWorld,algorithm_name,policy_type,is_loaded, continued):

    # IMPORT TRAINING PARAMETERS -------------------------------
    # CFG = DefaultConfig(iWorld,algorithm_name=algorithm_name,policy_type=policy_type,is_loaded=is_loaded,continued=continued)

    # CREATE LEARNING OBJECTS -------------------------------
    # torch.set_default_dtype(torch.__dict__[CFG.dtype])
    torch.set_default_dtype(CFG.dtype)



    if is_loaded or continued:
        if continued: _name = algorithm_name.replace('_extended',"")
        else: _name = algorithm_name
        Q = Qfunction.load(iWorld,policy_type = 'Baseline' ,algorithm = _name)
    else:
        Q = Qfunction(CFG.device, CFG.dtype, sophistocation=CFG.ToM, run_config=CFG,rationality=CFG.rationality)

    memory = ReplayMemory()

    # Create environment -------------------------
    env = PursuitEvastionGame(iWorld,CFG.device,CFG.dtype)
    test_env = PursuitEvastionGame(iWorld, CFG.device, CFG.dtype)

    # Set up logger -------------------------------
    Logger = RL_Logger()
    Logger.file_name = f'Fig_{algorithm_name}_{policy_type}' # + '_extended' if continued else ''
    Logger.update_save_directory(Logger.project_root + f'results\\IDQN_W{iWorld}\\')
    Logger.update_plt_title(f'[W{iWorld}-{policy_type}] {algorithm_name.replace("_", " ")} Training Results')
    Logger.make_directory(); Logger.filter_window = 10; Logger.refresh_rate = 10
    epi_timer = EpisodeTimer()

    # Handle Parameters ------------------------------
    epi_params = ParamScheduler(CFG)
    print(CFG)

    #############################################################################
    # BEGIN EPISODES ############################################################
    for i_episode in range(CFG.num_episodes):
        # SET EPISODE PARAMETERS -----------------------------------------------
        Logger.flush()
        epi_timer.sample()
        epsilon = epi_params(i_episode,'epsilon')
        env.scale_penalty = epi_params(i_episode,'penalty')
        env.scale_rcatch = epi_params(i_episode,'rcatch')
        env.enable_rand_init = (i_episode < CFG.rand_init_episodes)

        if Logger.end_button_state==1: logging.warning(f'EXITING FROM INTERFACE\n\n'); break
        if i_episode % CFG.CPT_interval == 0:  env.CPT = CPT_Handler.rand(assume=policy_type, verbose=False)

        # PLAY GAME -------------------------------------------------------------
        observations = Q.simulate(env, epsilon=epsilon)
        for state, action, next_state, reward,done in observations:
            memory.push(state, action, next_state, reward,done)

        # Perform one step of the optimization (on the policy network) ----------
        Q.update(memory.get(dump=True),CFG.LR[policy_type], CFG.GAMMA,CFG.ET_DECAY,CFG.FF[policy_type])

        # Report ----------------------------------------------------------------
        if i_episode % CFG.report_interval == 0:
            test_score, test_length, test_psucc = Q.test_policy(test_env, CFG.test_episodes)
            Logger.log_episode(test_score, test_length, test_psucc, buffered=False, episode=i_episode)
            Logger.draw()

            disp = ''
            disp += f'[{epi_timer.remaining_time(i_episode,CFG.num_episodes)}] '
            disp += f'[W{iWorld} {policy_type}] '
            # disp += f'STATE[{env.current_positions.detach().numpy().flatten()}] '
            disp += '{:<20}'.format(f'Epi[{i_episode}/{CFG.num_episodes}]')
            disp += '{:<20}'.format(f'Q[{round(Q.tbl.min().item(), 1)},{round(Q.tbl.max().item(), 1)}]')
            disp += "stats: [s/epi:{:<4} ".format(np.round(epi_timer.mean_dur, 2)) + f'eps:{round(epsilon,2)} ] \t'
            disp += '{:<30}'.format(f'scale[rcatch:{round(env.scale_rcatch,1)} pen:{round(env.scale_penalty,1)}]')
            disp += "test score: {:<35}".format(  f'[r:{np.round(test_score, 1)} l:{np.round(test_length, 1)} ps:{np.round(test_psucc, 1)}]\t')
            disp += f'{env.CPT}'
            print(disp)
        elif i_episode % CFG.test_interval == 0:
            test_score, test_length, test_psucc = Q.test_policy(test_env, CFG.test_episodes)
            Logger.log_episode(test_score, test_length, test_psucc, buffered=False, episode=i_episode)

        torch.clear_autocast_cache()

    print('Complete')
    # DQN.save(Q,iWorld,policy_type ,algorithm_name  + '_extended' if continued else '',axes=Logger.axs)
    # DQN.save(Q,iWorld,policy_type ,algorithm_name,axes=Logger.axs )#
    # Logger.blocking_preview()
    Qfunction.save(Q,iWorld,policy_type = policy_type ,algorithm = algorithm_name,axes=Logger.fig)#
    Logger.save()
    Logger.close()




if __name__ == "__main__":
    main()

