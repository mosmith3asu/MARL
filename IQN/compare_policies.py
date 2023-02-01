import copy
import logging

import torch
from os import listdir
from QLearning import Qfunction
from utilities.make_env import PursuitEvastionGame
from itertools import count
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import random
from utilities.config_manager import CFG

import math
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.float_format', '{:20,.2f}'.format)
pd.set_option('display.max_colwidth', None)
iR, iH = 0, 1

# policy_name2sym = {}
# policy_name2sym['Baseline'] = 'π_0'
# policy_name2sym['Averse'] = 'π_A'
# policy_name2sym['Seeking'] = 'π_S'
policy_name2sym = {}
policy_name2sym['Baseline'] = '0'
policy_name2sym['Averse'] = 'A'
policy_name2sym['Seeking'] = 'S'
# corr_assum_marker_key = '* '
corr_assum_marker_key = ' '
corr_assum_marker_plot = '✓'

legend_sz = 13

def main():
    global i_correct
    global i_incorrect
    global group_face_colors
    global i_first_set
    global i_last_set
    global test_cases
    save_dir = "../results/policy_comparisons/"
    save_indv_fname = "Fig_AssumptionComparison.png"
    save_summary_fname = "Fig_AssumptionComparison_Summary.png"

    WORLDS = [1, 2,3,4,5,6,7]  #
    test_episodes = 1000
    c0 = 0.35 # inital color value
    dpi = 1 / plt.rcParams['figure.dpi']  # pixel in inches

    nRows, nCols = 1, 1
    figW, figH = nCols*1500 * dpi, min(1000 * dpi, nRows * 400 * dpi)
    summary_fig, summary_ax = plt.subplots(1,1,constrained_layout=True,figsize=(figW,figH))

    i_first_set = [0, 1]  # i_first_set = [0]
    i_last_set = [-2,-1]
    group_face_colors,test_cases = [], []
    n=0
    test_cases.append(['Baseline','Baseline']); n+=1
    test_cases.append(['Averse', 'Baseline']);  n+=1
    # test_cases.append(['Seeking', 'Baseline']); n+=1
    group_face_colors += [(inc * (1-c0)/n, inc * (1-c0)/n, min([1, c0 + inc * (1-c0)/n]), 1.0) for inc in range(n)]
    # group_face_colors += [(inc * (1-c0)/n, min([1, c0 + inc * (1-c0)/n]), inc * (1-c0)/n, 1.0) for inc in range(n)]

    n=0
    test_cases.append(['Averse', 'Averse']);    n += 1
    test_cases.append(['Baseline', 'Averse']);  n+=1
    # test_cases.append(['Seeking', 'Averse']);   n += 1
    group_face_colors += [(inc * (1-c0)/n, min([1, c0 + inc * (1-c0)/n]), inc * (1-c0)/n, 1.0) for inc in range(n)]

    n = 0
    # test_cases.append(['Baseline', 'Seeking']); n+=1
    # test_cases.append(['Seeking', 'Seeking']);  n += 1
    # test_cases.append(['Averse', 'Seeking']); n += 1
    # group_face_colors += [(inc * (1-c0)/n, inc * (1-c0)/n, min([1, c0 + inc * (1-c0)/n]), 1.0) for inc in range(n)]



    i_correct = [(case[0]==case[1])for case in test_cases]
    i_correct = list(np.where(np.array(i_correct)==True)[0])
    i_incorrect = list(np.where(np.array(i_correct) == False)[0])


    iplt = 0
    r,c= 0,-1
    all_data = None
    disp_df_master = None
    for iWorld in WORLDS:
        plot_df = None
        for icase,case in enumerate(test_cases):
            policy_type = copy.deepcopy(case)

            print(f'W{iWorld}: {case} \t {policy_type}')
            policyR = Qfunction.load(iWorld, policy_type = policy_type[iR], algorithm=CFG.algorithm_name, verbose=False)
            policyH = Qfunction.load(iWorld, policy_type = policy_type[iH], algorithm=CFG.algorithm_name, verbose=False)
            plt.close(policyR.axes)
            plt.close(policyH.axes)

            disp_df,this_plot_df,data_df = test_policies(iWorld,policy_type,policyR,policyH,num_episodes=test_episodes)

            if plot_df is None:  plot_df = this_plot_df
            else: plot_df = pd.concat([plot_df, this_plot_df])

            if disp_df_master is None: disp_df_master = disp_df
            else: disp_df_master = pd.concat([disp_df_master,disp_df])

            if all_data is None: all_data = data_df
            else: all_data = pd.concat([all_data, data_df.copy()])

        fname = save_indv_fname.split('.png')[0] + f'_W{iWorld}.png'
        comp_fig, comp_ax = plt.subplots(1,1,constrained_layout=True,figsize=(figW,figH))
        plot_evaluation(comp_ax,iWorld,plot_df)
        comp_fig.savefig(save_dir + fname)
        plt.close(comp_fig)
        iplt +=1

    print(disp_df_master)
    print('\n\n\n\n\n')
    print(all_data)

    # comp_ax = comp_axs[-1,c]#subfigs[-1, iComp].subplots(1, 1)
    # get_summary(comp_ax, all_data)

    # comp_fig.savefig(save_dir + save_indv_fname)
    get_summary(summary_ax, all_data)
    summary_fig.savefig(save_dir + save_summary_fname); print(f'Saved: {save_dir + save_summary_fname}')
    plt.show()

def get_summary(ax,df,has_legend=True):
    global group_face_colors
    global test_cases

    df_summary = None
    df.set_index(['World', 'Case'], inplace=True)
    idxs = np.array([list(idx) for idx in df.index.values])
    conditions = idxs[0:len(test_cases), 1]

    for cond in conditions:
        mean_cond = df.xs(cond, level=1, drop_level=False).mean()
        mean_cond = pd.DataFrame(mean_cond).T
        mean_cond.insert(0, 'Case', [cond],True)
        if df_summary is None: df_summary = mean_cond
        else: df_summary = pd.concat([df_summary, mean_cond])

    print(f'\n######## SUMMARY ########')
    print(df_summary)


    df_summary.set_index('Case', inplace=True)
    df_summary.T.plot(ax=ax, kind="bar",color=group_face_colors)
    for tick in ax.get_xticklabels(): tick.set_rotation(0)
    ax.set_ylabel("Performance")  # ax.set_xlabel("Metric")
    ax.set_title(f"Mean Simulation Results")
    if has_legend:
        # ax.legend(title='Conditions: ($\hat{\pi}_{H}$ x $\pi_{H}$)', bbox_to_anchor=(1.01, 1), loc='upper left',
        #           borderaxespad=0)
        legend = ax.legend(title='Conditions: ($\mathcal{C}^{\;\hat{\pi}_{H}}_{\;\pi_{H}}$)', prop={'size': legend_sz},
                  bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
        plt.setp(legend.get_title(), fontsize=legend_sz)
    set_true_style(ax)
    ax.set_ylim([0, 25.1])


def move_axes(ax, fig, subplot_spec=111):
    """Move an Axes object from a figure to a new pyplot managed Figure in
    the specified subplot."""

    # get a reference to the old figure context so we can release it
    old_fig = ax.figure

    # remove the Axes from it's original Figure context
    ax.remove()

    # set the pointer from the Axes to the new figure
    ax.figure = fig

    # add the Axes to the registry of axes for the figure
    fig.axes.append(ax)
    # twice, I don't know why...
    fig.add_axes(ax)

    # then to actually show the Axes in the new figure we have to make
    # a subplot with the positions etc for the Axes to go, so make a
    # subplot which will have a dummy Axes
    dummy_ax = fig.add_subplot(subplot_spec)

    # then copy the relevant data from the dummy to the ax
    ax.set_position(dummy_ax.get_position())

    # then remove the dummy
    dummy_ax.remove()

    # close the figure the original axis was bound to
    plt.close(old_fig)

def test_policies(iWorld,policy_type,policyR,policyH,num_episodes,sigdig=2):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    pscale = 10
    # Check and get parameters ###############
    device = 'cpu'
    dtype = torch.float32#torch.__dict__[settingsR.dtype]

    # policy_types = f"({policy_name2sym[policy_type[iR]]} x {policy_name2sym[policy_type[iH]]})"
    policy_types =  "$\mathcal{C}^{\; \hat{\pi}_{" + f"{policy_name2sym[policy_type[iR]]}" \
                    + "}}_{\; \pi_{" + f"{policy_name2sym[policy_type[iH]]}" +"}}$"
    # "$\mathcal{C}^{\hat{\pi}_{H}}_{\pi_{H}}$"
    if policy_type[iR] == policy_type[iH]: policy_types=corr_assum_marker_key +policy_types
    else: policy_types = '  ' + policy_types
    env = PursuitEvastionGame(iWorld, device, dtype)

    # Define data trackers ############
    length = np.zeros([num_episodes,1])
    psucc = np.zeros([num_episodes,1])
    scores = np.zeros([num_episodes,env.n_agents])
    phat_anotk = np.zeros([num_episodes,env.n_agents])
    in_pens = np.zeros([num_episodes, env.n_agents])
    catch_freq = np.zeros([7,7],dtype=int)

    # policyR.rationality = 10
    # policyH.rationality = 10
    # logging.warning('MODIFIED RATIONALITY')

    # Test in episodes ##############
    for episode_i in range(num_episodes):
        state = env.reset()
        env.scale_penalty = 1.0
        env.scale_rcatch = 1.0
        R_phat_aH = []
        H_phat_aR = []
        for t in count():

            aR, phatAH = policyR.sample_sim_action(state, agent=iR)
            aH, phatAR = policyH.sample_sim_action(state, agent=iH)
            # aR,phatAH = policyR.sample_action(state, epsilon=0, agent=iR)
            # aH,phatAR = policyH.sample_action(state, epsilon=0, agent=iH)

            action = policyR.ego2joint_action(aR, aH)
            next_state, reward, done, _ = env.step(action.squeeze())

            scores[episode_i] += reward.detach().flatten().cpu().numpy()
            in_pens[episode_i, iR] += env.check_is_penalty(env.current_positions[iR])
            in_pens[episode_i, iH] += env.check_is_penalty(env.current_positions[iH])
            R_phat_aH.append(phatAH.detach().numpy().flatten()[int(aH.item())])
            H_phat_aR.append(phatAR.detach().numpy().flatten()[int(aR.item())])
            state = next_state.clone()
            if done: break

        phat_anotk[episode_i, iR] = np.mean(R_phat_aH)
        phat_anotk[episode_i, iH] = np.mean(H_phat_aR)
        if env.check_caught(env.current_positions):
            psucc[episode_i] = 1
            prey_pos = env.current_positions[-1,:].detach().numpy().astype(int)
            catch_freq[prey_pos[0],prey_pos[1]] += 1
        length[episode_i] = env.step_count


    team_score,sig_team_score = np.mean(scores).round(sigdig).round(sigdig),np.std(np.mean(scores,axis=1)).round(sigdig)
    final_score,sig_score    = np.mean(scores,axis=0).round(sigdig), np.std(scores,axis=0).round(sigdig)
    final_in_pen, sig_in_pen = np.mean(in_pens, axis=0).round(sigdig), np.std(in_pens, axis=0).round(sigdig)
    final_phata, sig_phata   = np.mean(phat_anotk, axis=0).round(sigdig), np.std(phat_anotk, axis=0).round(sigdig)

    final_length,sig_length  = np.mean(length).round(sigdig), np.std(length).round(sigdig)
    final_psucc,sig_psucc   = np.mean(psucc).round(sigdig), np.std(psucc).round(sigdig)

    # _ptypes = f"(${policy_name2sym[policy_type[iR]]}$ x ${policy_name2sym[policy_type[iH]]}$)"
    _ptypes = "$\mathcal{C}^{\; \hat{\pi}_{" + f"{policy_name2sym[policy_type[iR]]}" \
                   + "}}_{\; \pi_{" + f"{policy_name2sym[policy_type[iH]]}" + "}}$"
    if policy_type[iR] == policy_type[iH]: _ptypes = corr_assum_marker_key +_ptypes
    else: _ptypes = '  ' +_ptypes


    plot_dict = {}
    plot_dict["World"] = [f'{iWorld}']
    plot_dict["Case"] = [_ptypes]
    plot_dict["R's Cum Reward \n $\Sigma_{t} (R_{(R,t)}$)"] = [float(final_score[iR])]
    plot_dict["H's Cum Reward \n $\Sigma_{t} (R_{(H,t)})$"] = [float(final_score[iH])]
    plot_dict['Ave Cum Reward \n $\Sigma_{t} (\\bar{R}_{t}$)'] = [float(team_score)]
    plot_dict['Episode Length \n $|T|$'] = [float(final_length)]
    plot_dict["#R in Penalty \n$|s_{(R,t)} \; \in \; S_{\\rho}|$"] = [float(final_in_pen[iR])]
    plot_dict["#H in Penalty \n$|s_{(H,t)} \; \in \; S_{\\rho}|$"] = [float(final_in_pen[iH])]

    plot_dict['' if pscale ==1 else f'(x{pscale}) ' + 'Prob of Catching \n $P(catch)$'] = [pscale*float(final_psucc)]
    plot_dict['' if pscale ==1 else f'(x{pscale}) ' + "R's MM(H) \n$\hat{p}(a_{(H,t)} | \hat{\pi}_{H})$"] = [pscale*float(final_phata[iR])]
    plot_dict['' if pscale ==1 else f'(x{pscale}) ' + "H's MM(R) \n$\hat{p}(a_{(R,t)} | \hat{\pi}_{R})$"] = [pscale*float(final_phata[iH])]
    # plot_dict['terminal state'] = [np.unravel_index(np.argmax(catch_freq), catch_freq.shape)]
    plot_df = pd.DataFrame.from_dict(plot_dict)


    disp_dict = {}
    disp_dict["Case"] = [policy_types]
    disp_dict["World"] = [f'{iWorld}']
    disp_dict['reward_R'] = f'{final_score[iR]} ± {sig_score[iR]}'
    disp_dict['reward_H'] = f'{final_score[iH]} ± {sig_score[iH]}'
    disp_dict['reward_both'] = f'{team_score} ± {sig_team_score}'
    disp_dict['Episode Length'] =f'{final_length} ± {sig_length}'
    disp_dict['P(catch)'] = f'{final_psucc}'
    disp_dict['terminal state'] = [np.unravel_index(np.argmax(catch_freq), catch_freq.shape)]
    disp_df = pd.DataFrame.from_dict(disp_dict)
    disp_df.set_index(['World','Case'], inplace=True)

    data_df = plot_df.copy()
    return disp_df,plot_df,data_df

def plot_evaluation(ax,iWorld,df,has_legend=True):
    # group_face_colors = ['r','g','y','b','m']

    global group_face_colors
    group_edge_colors = ['k', 'k', 'w', 'w', 'w']
    # Remove worlds from df
    new_indexs = []
    for index in df['Case']:
        new_indexs.append(index.replace(f'[W{iWorld}]', ""))
    df['Case'] = new_indexs
    df.set_index('Case', inplace=True)
    df = df.drop(columns = 'World')


    # Plot bar plot
    if has_legend:
        df.T.plot(ax = ax,kind="bar",color=group_face_colors)#,,edgecolor=group_edge_colors)color=group_face_colors,
        # ax.legend(title='Conditions: ($\hat{\pi}_{H}$ x $\pi_{H}$)',
        # bbox_to_anchor=(1.01, 1), loc='upper left',  borderaxespad=0)
        legend = ax.legend(title='Conditions: ($\mathcal{C}^{\;\hat{\pi}_{H}}_{\;\pi_{H}}$)',prop={'size': legend_sz},
                  bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
        plt.setp(legend.get_title(), fontsize=legend_sz)
    else:
        df.T.plot(ax = ax,kind="bar",color=group_face_colors,legend=None)#,,edgecolor=group_edge_colors)color=group_face_colors,



    for tick in ax.get_xticklabels(): tick.set_rotation(0)
    ax.set_ylabel("Performance")  # ax.set_xlabel("Metric")
    ax.set_title(f"World {iWorld} - Simulated Conditions")



    ax.set_ylim([0,25.1])
    set_true_style(ax)

def set_true_style(ax):
    global i_correct
    global i_incorrect
    global i_first_set
    global i_last_set

    w_pad = 0.25
    for icontainer in range(len(ax.containers[0])):
        # Move baseline
        for idata in i_first_set:
            rect = ax.containers[idata][icontainer]
            ax.containers[idata][icontainer].set(x=rect.get_x() - w_pad * rect.get_width())

        # Move last set
        for idata in i_last_set:
            rect = ax.containers[idata][icontainer]
            ax.containers[idata][icontainer].set(x=rect.get_x() + w_pad * rect.get_width())

        for idata in i_correct:
            rect = ax.containers[idata][icontainer]
            ax.containers[idata][icontainer].set(edgecolor='k', linewidth=1, linestyle='-')
            # ax.containers[idata][icontainer].set(x=rect.get_x() - 0.5 * rect.get_width())
            # ax.containers[icorrect][icontainer].set(hatch='/')
            # ax.containers[icorrect][icontainer].set(capstyle='round')
            corners = rect.get_corners()
            x = np.mean([corners[0][0], corners[1][0]])
            y = np.mean([corners[2][1], corners[3][1]])
            ax.text(x, y, corr_assum_marker_plot, fontsize=10, ha='center')
            # ax.text(x, y, ['BL', 'A', 'S'][i], fontsize=10, ha='center',va='bottom')
            # i+=1


if __name__ == "__main__":
    main()
