import collections
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ma_gym.wrappers import Monitor
import itertools
from enviorment.make_env import CPTEnv
from algorithm.utils.logger import Logger
plt.ion()



class ReplayBuffer:
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            # done_mask_lst.append((np.ones(len(done)) - done).tolist())
            done_mask_lst.append((np.ones(1) - done).tolist())

        return torch.tensor(s_lst, dtype=torch.float), \
               torch.tensor(a_lst, dtype=torch.float), \
               torch.tensor(r_lst, dtype=torch.float), \
               torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst, dtype=torch.float)

    def size(self):
        return len(self.buffer)


class QNet(nn.Module):
    epsilon = 0.0
    rationality = 1.0
    w_explore = 0.0

    sample_method = 'NR'
    def __init__(self, observation_space, action_space):
        super(QNet, self).__init__()
        self.num_agents = len(observation_space)
        self.n_joint_actions = 25
        self.n_solo_actions = 5
        # self.rationality = 1.0
        # self.w_explore = 0.0
        # self.sample_method = 'NR'
        self.settings = None

        self.joint2solo = np.array(list(itertools.product(*[np.arange(5), np.arange(5)])), dtype=int)
        self.solo2joint = np.zeros([5, 5], dtype=int)
        for aJ, joint_action in enumerate(self.joint2solo):
            aR, aH = joint_action
            self.solo2joint[aR, aH] = aJ
        ijoint = np.zeros([2, 5, 25],dtype=np.float32)
        for k in [0, 1]:
            for ak in range(5):
                idxs = np.array(np.where(self.joint2solo[:, k] == ak)).flatten()
                ijoint[k, ak, idxs] = 1
        self.ijoint = torch.from_numpy(ijoint)


        n_obs =observation_space[0].shape[0]

        for agent_i in range(self.num_agents):
            setattr(self, 'agent_{}'.format(agent_i), nn.Sequential(nn.Linear(n_obs, 128),
                                                                    nn.ReLU(),
                                                                    nn.Linear(128, 64),
                                                                    nn.ReLU(),
                                                                    nn.Linear(64, self.n_joint_actions)))
                                                                    #nn.Linear(64, action_space[agent_i].n)))

    def forward(self, obs):
        q_values = [torch.empty(obs.shape[0], )] * self.num_agents
        for agent_i in range(self.num_agents):
            q_values[agent_i] = getattr(self, 'agent_{}'.format(agent_i))(obs[:, agent_i, :]).unsqueeze(1)
        return torch.cat(q_values, dim=1)

    def add_exploration(self,q,w_explore,directed = True):
        q = q.flatten()
        sp = w_explore # noise size w.r.t. the range of qualities (decrease => less exploration)
        qi_range = float(torch.max(q) - torch.min(q))  # range of q values

        # Undirected Exploration
        psi = np.random.exponential(size= self.n_joint_actions)  # exponential dist. scaled to match range of q-values
        sf = (sp * qi_range / np.max(psi)) # scaling factor
        eta =  np.array(sf * psi,dtype=np.float32)
        eta = torch.from_numpy(eta)

        # if directed:
        #     # Directed exploration
        #     #   The directed term gives a bonus to the actions that have been rarely elected
        #     theta = self.w_dir_explore  # positive factor used to weight the directed exploration
        #     n_SA = self.action_frequency  # the number of time steps in which action U has been elected
        #     rho = (qi_range * theta) / np.exp(n_SA)

        # return cumulative Q
        return (q + eta).flatten()

    def sample_NR_action(self, obs, w_explore,rationality,epsilon):
        """
        Sample noisy-rational action
        - softmax sampled on ego quality [1x5]
        - new quality conditined on partner pA
        """


        QsA = self.forward(obs)  # 1 x n_agents x n_actions
        scale_final_rationality = 1

        if np.random.rand() <= epsilon:
            aJ = torch.randint(0, QsA.shape[2], [1, 1])
        else:
            # qAR = QsA[0, 0].detach()  # qAR = self.add_exploration(QsA[0, 0].detach(),w_explore=w_explore) # 1 x n_actions
            # qAH = QsA[0, 1].detach()  # qAH = self.add_exploration(QsA[0, 1].detach(),w_explore=w_explore) # 1 x n_actions
            #"""
            # Level 0 sophistication
            _qAH = self.add_exploration(QsA[0, 1].detach(), w_explore=w_explore)
            _qAR = self.add_exploration(QsA[0, 0].detach(), w_explore=w_explore)
            phhatA_H = torch.ones([1, self.n_solo_actions], dtype=_qAH.dtype) / self.n_solo_actions
            phhatA_R = torch.ones([1, self.n_solo_actions], dtype=_qAR.dtype) / self.n_solo_actions
            phhatA_H = torch.matmul(phhatA_H, self.ijoint[1])
            phhatA_R = torch.matmul(phhatA_R, self.ijoint[0])

            # Level 1 sophistication
            qhat_H = torch.matmul(self.ijoint[1], (_qAH * phhatA_R).T).flatten()
            qhat_R = torch.matmul(self.ijoint[0], (_qAR * phhatA_H).T).flatten()
            phatA_H = torch.matmul(torch.special.softmax(rationality * qhat_H, dim=0),self.ijoint[1])
            phatA_R = torch.matmul(torch.special.softmax(rationality * qhat_R, dim=0),self.ijoint[0])

            # Select Ego Action
            q_H = torch.matmul(self.ijoint[1], (_qAH * phatA_R).T).flatten()
            q_R = torch.matmul(self.ijoint[0], (_qAR * phatA_H).T).flatten()
            pA_H = torch.special.softmax(rationality * q_H, dim=0)
            pA_R = torch.special.softmax(rationality * q_R, dim=0)
            if self.sample_method == 'NR_max':
                aH = torch.argmax(pA_H).detach().int()
                aR = torch.argmax(pA_R).detach().int()
            elif self.sample_method == 'NR_max':
                aH = np.random.choice(np.arange(self.n_solo_actions), p=pA_H.detach().numpy())
                aR = np.random.choice(np.arange(self.n_solo_actions), p=pA_R.detach().numpy())
            else:  raise Exception("unknown NR sampliong method")


            self.sample_action_report(qA=[q_R,q_H],pA=[pA_R,pA_H])
            # Get joint action
            aJ = self.solo2joint[aR, aH] # joint action
            """
            # Level 0 sophistication
            joint_qAH = self.add_exploration(QsA[0, 1].detach(), w_explore=w_explore)
            joint_qAR = self.add_exploration(QsA[0, 0].detach(), w_explore=w_explore)
            ego_phhatA_H = torch.ones([1, self.n_solo_actions], dtype=joint_qAH.dtype) / self.n_solo_actions
            ego_phhatA_R = torch.ones([1, self.n_solo_actions], dtype=joint_qAR.dtype) / self.n_solo_actions
            joint_phhatA_H = torch.matmul(ego_phhatA_H, self.ijoint[1])
            joint_phhatA_R = torch.matmul(ego_phhatA_R, self.ijoint[0])

            # Level 1 sophistication
            joint_qhat_H = (joint_qAH * joint_phhatA_R)
            joint_qhat_R = (joint_qAR * joint_phhatA_H)
            ego_qhatA_H = torch.matmul(self.ijoint[1], joint_qhat_H.T).flatten()
            ego_qhatA_R = torch.matmul(self.ijoint[0], joint_qhat_R.T).flatten()
            ego_phatA_H = torch.special.softmax(rationality * ego_qhatA_H , dim=0)
            ego_phatA_R = torch.special.softmax(rationality * ego_qhatA_R , dim=0)
            joint_phatA_H = torch.matmul(ego_phatA_H, self.ijoint[1])
            joint_phatA_R = torch.matmul(ego_phatA_R, self.ijoint[0])

            # Level 2 sophistication
            joint_q_H = (joint_qAH * joint_phatA_R)
            joint_q_R = (joint_qAR * joint_phatA_H)
            ego_q_H = torch.matmul(self.ijoint[1],joint_q_H.T).flatten()
            ego_q_R = torch.matmul(self.ijoint[0],joint_q_R.T).flatten()
            ego_pA_H = torch.special.softmax(rationality * ego_q_H, dim=0)
            ego_pA_R = torch.special.softmax(rationality * ego_q_R, dim=0)
            joint_phatA_H = torch.matmul(ego_pA_H, self.ijoint[1])
            joint_phatA_R = torch.matmul(ego_pA_R, self.ijoint[0])

            # Select Ego Action
            # joint_q_H = (joint_qAH * joint_phatA_R)
            # joint_q_R = (joint_qAR * joint_phatA_H)
            # ego_q_H = torch.matmul(self.ijoint[1], joint_q_H.T).flatten()
            # ego_q_R = torch.matmul(self.ijoint[0], joint_q_R.T).flatten()
            # ego_pA_H = torch.special.softmax(rationality * ego_q_H, dim=0)
            # ego_pA_R = torch.special.softmax(rationality * ego_q_R, dim=0)
            # aH = torch.argmax(ego_pA_H).detach().int()
            # aR = torch.argmax(ego_pA_R).detach().int()


            ego_pA_H = ego_pA_H.detach().numpy()
            ego_pA_R = ego_pA_R.detach().numpy()
            aH = np.random.choice(np.arange(self.n_solo_actions), p= ego_pA_H)
            aR = np.random.choice(np.arange(self.n_solo_actions), p= ego_pA_R)
            # Get joint action
            aJ = self.solo2joint[aR, aH]  # joint action

            """

            #aJ = torch.argmax(QsA[0, 0].detach() + QsA[0, 1].detach())
        action = aJ * torch.ones((QsA.shape[0], QsA.shape[1],))
        return action
    def sample_action_report(self,qA,pA,report_range=True):
        # Report ----------------------
        ego_pA_R, ego_pA_H = pA
        ego_q_R, ego_q_H = qA

        if isinstance(ego_pA_R,torch.Tensor): ego_pA_R = ego_pA_R.detach().numpy().flatten()
        if isinstance(ego_pA_H, torch.Tensor): ego_pA_H = ego_pA_H.detach().numpy().flatten()
        if isinstance(ego_q_R, torch.Tensor): ego_q_R = ego_q_R.detach().numpy().flatten()
        if isinstance(ego_q_H, torch.Tensor): ego_q_H = ego_q_H.detach().numpy().flatten()
        minmix_qH = [min(ego_q_H), max(ego_q_H)]
        minmix_qR = [min(ego_q_R), max(ego_q_R)]
        if report_range:
            print('\r\t| {:<50} {:<50}'.format(f' R{tuple(np.round(minmix_qR, 1))}: {ego_pA_R.round(2)}',
                                               f' H{tuple(np.round(minmix_qH, 1))}: {ego_pA_H.round(2)}'), end='')
        else:
            dqH = float(minmix_qH[1]-minmix_qH[0])
            dqR = float(minmix_qR[1]-minmix_qR[0])
            print('\r\t| {:<50} {:<50}'.format(f' R({round(dqR,2)}): {ego_pA_R.round(2)}',f'H({round(dqH, 2)}): {ego_pA_H.round(2)}'), end='')

        # print(f'\r   R({round(dqR,2)}):{ego_pA_R.round(2)}'
        #       f'\t\t H({round(dqH,2)}):{ego_pA_H.round(2)}' , end='')

        # print(f'\r   R{np.array([dqR]).round(2)}:{ego_pA_R.round(2)} '
        #       f'\t\t H{np.array([dqH]).round(2)}:{ego_pA_H.round(2)}', end='')
    def sample_SW_action(self, obs, w_explore,rationality,epsilon):
        """Sample social welfare action"""
        QsA = self.forward(obs)
        # qAR = self.add_exploration(QsA[0, 0].detach(), w_explore=w_explore)  # 1 x n_actions
        # qAH = self.add_exploration(QsA[0, 1].detach(), w_explore=w_explore)  # 1 x n_actions
        # aJ = torch.argmax(qAR + qAH)
        if np.random.rand() <= epsilon: aJ = torch.randint(0, QsA.shape[2], [1, 1])
        else:  aJ = torch.argmax( QsA[0, 0]+QsA[0, 1])
        action = aJ * torch.ones((QsA.shape[0], QsA.shape[1],))
        return action

    def sample_action(self, obs, w_explore=None,rationality=None,epsilon=None,choose_best=False):
        # Load Defaults
        if rationality is None: rationality = self.rationality
        if w_explore is None: w_explore = self.w_explore
        if epsilon is None: epsilon = self.epsilon

        assert w_explore == 0 or epsilon==0, f"sample_action has both exploration policies [eps:{epsilon} sp:{w_explore}]"

        if self.sample_method in ['NR','NR_max']:
            action = self.sample_NR_action( obs, w_explore = w_explore, rationality=rationality,epsilon=epsilon)
        elif self.sample_method == 'SW':
            action = self.sample_SW_action(obs,  w_explore = w_explore, rationality=rationality, epsilon=epsilon)
        else: raise Exception(f'unknown sample method [{self.sample_method}]')
        return action


def train(q, q_target, memory, optimizer, gamma, batch_size, update_iter=10):
    for _ in range(update_iter):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(2, a.unsqueeze(-1).long()).squeeze(-1)
        max_q_prime = q_target(s_prime).max(dim=2)[0]
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
def test(env, num_episodes, q):
    old_w_explore = q.w_explore
    old_epsilon = q.epsilon
    old_rationality = q.rationality

    length =np.zeros(env.n_agents)
    psucc = np.zeros(env.n_agents)
    score = np.zeros(env.n_agents)
    for episode_i in range(num_episodes):
        state = env.reset()
        done = False# [False for _ in range(env.n_agents)]
        while not done:
            action = q.sample_action(torch.Tensor(state).unsqueeze(0),w_explore=0.0,epsilon=0.0,rationality=None)
            action = action[0].data.cpu().numpy().tolist()
            next_state, reward, done, info = env.step(action,is_joint=True)
            score += np.array(reward)
            state = next_state

        if env.is_caught: psucc +=1
        length += env.step_count


    q.w_explore = old_w_explore
    q.epsilon = old_epsilon
    q.rationality = old_rationality

    final_score = np.mean(score / num_episodes)
    final_length =np.mean(length / num_episodes)
    final_psucc = np.mean(psucc / num_episodes)
    return final_score,final_length,final_psucc

def schedule(start,end,N,slope=1.0):
    iinv = np.power(1 / (np.linspace(1, 10, N) - 0.1) - 0.1, slope)
    return (start + end) * iinv + end
    #return np.linspace(start, end, N)

def get_CPT_assumption(assume,res=100):
    CPT_params = {'reference': 0, 'lamb': 1, 'gamma_gain': 1, 'gamma_loss': 1,
                  'delta_gain': 1, 'delta_loss': 1, 'agentk': 1}

    if assume == 'averse':
        lamb = np.random.choice(np.linspace(1.5, 3, res))
        r_discount = np.random.choice(np.linspace(0.01,0.4,res))
        p_discount =  np.random.choice(np.linspace(0.01,0.4,res))
        CPT_params['reference'] = 0
        CPT_params['lamb']       = lamb
        CPT_params['gamma_gain'] = r_discount
        CPT_params['gamma_loss'] = 1-r_discount
        CPT_params['delta_gain'] = p_discount
        CPT_params['delta_loss'] = 1-p_discount
    elif assume == 'seeking':
        lamb = np.random.choice(np.linspace(0, 0.9, res))
        r_discount = np.random.choice(np.linspace(0.6,0.99, res))
        p_discount = np.random.choice(np.linspace(0.6,0.99, res))
        CPT_params['reference'] = 0
        CPT_params['lamb'] = lamb
        CPT_params['gamma_gain'] = r_discount
        CPT_params['gamma_loss'] = 1 - r_discount
        CPT_params['delta_gain'] = p_discount
        CPT_params['delta_loss'] = 1 - p_discount
    elif assume == 'optimal':
        CPT_params['reference'] = 0
        CPT_params['lamb']       = 1
        CPT_params['gamma_gain'] = 1
        CPT_params['gamma_loss'] = 1
        CPT_params['delta_gain'] = 1
        CPT_params['delta_loss'] = 1
    else: raise Exception('Unknown CPT assumption')

    assert CPT_params['lamb'] > 0 , 'lamb bounds error'
    assert CPT_params['gamma_gain'] > 0 and CPT_params['gamma_gain'] <= 1,'gamma_gain bounds error'
    assert CPT_params['gamma_loss'] > 0 and CPT_params['gamma_loss'] <= 1, 'gamma_loss bounds error'
    assert CPT_params['delta_gain'] > 0 and CPT_params['delta_gain'] <= 1, 'delta_gain bounds error'
    assert CPT_params['delta_loss'] > 0 and CPT_params['delta_loss'] <= 1, 'delta_loss bounds error'

    disp_dict = {}
    for key in CPT_params:
        disp_dict[key] = round(float(CPT_params[key]),2)
    print(f'\r|\t New CPT Parameters{disp_dict}')
    return CPT_params
def print_dict(this_dict,title='Dictionary'):
    print(title)
    for key in this_dict.keys():
        print('\t| {:<20}: {:<20}'.format(key,f'{this_dict[key]}'))
    print(f'\n\n')
def run(run_config):

    # Unpack configuration -------------------------------
    print_dict(run_config, title='Run Config')

    iWorld = run_config['iWorld']
    lr = run_config['lr']
    TYPE = run_config['type']
    gamma = run_config['gamma']
    start_lamb,end_lamb = run_config['lambda']
    start_sp, end_sp = run_config['sp']
    start_epsilon, end_epsilon = run_config['epsilon']
    action_sampler = run_config['action_sampler']

    max_episodes = run_config['max_episodes']
    no_penalty_steps = run_config['no_penalty_steps']
    warm_up_steps = run_config['warm_up_steps']
    full_exploit_steps = run_config['full_exploit_steps']
    full_explore_steps = run_config['full_explore_steps']

    batch_size = run_config['batch_size']
    buffer_limit = run_config['buffer_limit']
    log_interval = run_config['log_interval']
    update_iter = run_config['update_iter']
    seed = run_config['seed']

    SCALE_REWARDS = run_config['r_scale']
    test_episodes = run_config['test_episodes']
    # full_exploit = run_config['full_exploit']
    # full_explore = run_config['full_explore']

    # Set up logger -------------------------------
    # plt.close()
    sample_CPT_iter = 200
    CPT_assumption = TYPE#'averse'

    #TYPE = CPT_assumption #'Baseline'
    ALG = 'Joint'
    Logger.file_name = f'Fig_IDQN_{ALG}_{TYPE}'
    Logger.update_save_directory(Logger.project_root + f'results\\IDQN_W{iWorld}\\')
    Logger.update_plt_title(f'W{iWorld} {TYPE} IDQN Training Results')
    Logger.make_directory()
    Logger.filter_window = 10


    # Create environment -------------------------------
    if SCALE_REWARDS is not None:
        f_scaleR = lambda _rewards: [np.sign(r) * abs(pow(r, SCALE_REWARDS))for r in _rewards]
    else: f_scaleR = None

    env = CPTEnv(iWorld)
    test_env = CPTEnv(iWorld)

    env.f_scaleR = f_scaleR
    #test_env.f_scaleR = f_scaleR
    memory = ReplayBuffer(buffer_limit)

    # Create Q functions -------------------------------
    if seed is not None: torch.manual_seed(seed)

    if TYPE in  ['averse','seeking']:
        q = torch.load(Logger.save_dir + f'QModel_{ALG}_Baseline.torch')
    else:
        q = QNet(env.observation_space, env.action_space)
        q.settings = run_config

    q_target = QNet(env.observation_space, env.action_space)
    q_target.load_state_dict(q.state_dict())
    optimizer = optim.Adam(q.parameters(), lr=lr)

    # SCHEDULE EXPLORATION POLICY -------------------------------
    improve_slope = 1 / 1.5
    n_improve_epi = (max_episodes - full_explore_steps - full_exploit_steps)

    warmup_sp  = start_sp * np.ones(full_explore_steps)
    perform_sp = end_sp * np.ones(full_exploit_steps)
    improve_sp = schedule(start_sp,end_sp,n_improve_epi,slope=improve_slope)
    epi_sp = np.hstack([warmup_sp, improve_sp, perform_sp])

    warmup_epsilons = start_epsilon * np.ones(full_explore_steps)
    perform_epsilons = end_epsilon * np.ones(full_exploit_steps)
    improve_epsilons = schedule(start_epsilon,end_epsilon,n_improve_epi,slope=improve_slope)
    epi_epsilons = np.hstack([warmup_epsilons, improve_epsilons,perform_epsilons])

    warmup_lamb  = start_lamb * np.ones(full_explore_steps)
    perform_lamb = end_lamb * np.ones(full_exploit_steps)
    improve_lamb = schedule(start_lamb, end_lamb, n_improve_epi, slope=improve_slope)
    epi_rationality = np.hstack([warmup_lamb, perform_lamb, improve_lamb])


    train_scores = np.zeros([log_interval,2])
    for episode_i in range(max_episodes):

        # Get episode parameters
        w_explore = epi_sp[episode_i]
        epsilon = epi_epsilons[episode_i]
        rationality = epi_rationality[episode_i]

        q.sample_method = action_sampler
        q.w_explore = w_explore
        q.epsilon = epsilon
        q.rationality = rationality

        q_target.sample_method = action_sampler
        q_target.w_explore = w_explore
        q_target.epsilon = epsilon
        q_target.rationality = rationality

        cumR = np.zeros(2)
        done = False

        # Intialize episode
        state = env.reset()
        env.enable_penalty = True if episode_i > no_penalty_steps else False
        test_env.enable_penalty = True if episode_i > no_penalty_steps else False
        if TYPE in ['averse', 'seeking']:
            if episode_i % sample_CPT_iter == 0:
                CPT_params = get_CPT_assumption(CPT_assumption)
                env.set_CPT(**CPT_params)




        Logger.tick()

        # Run episode until terminal
        while not done:
            action = q.sample_action(torch.Tensor(state).unsqueeze(0))
            action = action[0].data.cpu().numpy().tolist()
            next_state, reward, done, info = env.step(action,is_joint=True)

            memory.put((state, action, (np.array(reward)).tolist(), next_state, np.array(done, dtype=int).tolist()))
            train_scores[episode_i % log_interval,:] += np.array(reward)
            cumR += np.array(reward)
            state = next_state

        # Close episode
        #Logger.log_episode(np.mean(cumR), env.step_count, env.is_caught)
        if memory.size() > warm_up_steps:
            train(q, q_target, memory, optimizer, gamma, batch_size, update_iter)


        # REPORTING ##################
        time2report = (episode_i % log_interval == 0 and episode_i != 0)
        if time2report:
            q_target.load_state_dict(q.state_dict())
            test_score,test_length,test_psucc = test(test_env, test_episodes, q)
            Logger.log_episode(test_score,test_length,test_psucc,buffered=False,episode=episode_i)
            print("\r", end='')
            print(f"{TYPE}  ", end='')
            print("#{:<20}".format(f'[{episode_i}/{max_episodes} epi]'),end= '')
            print("train score: {:<8} ".format(np.round(np.mean(train_scores), 1)), end='')
            print("test score: {:<25}".format(f'[r:{np.round(test_score, 1)} l:{np.round(test_length, 1)} ps:{np.round(test_psucc, 1)}]'), end='')
            print("n_buffer : {:<10}  ".format(memory.size()), end='')
            print("eps : {:<6} ".format(round(q.epsilon,2)), end='')
            print("lam : {:<6}".format( round(q.rationality,2)), end='')
            print("sp : {:<6}".format(round(q.w_explore,2)), end='')
            print("sampler : {:<10}".format(q.sample_method), end='')
            #print("Qrange : {:<10}".format(f'[{torch.min(q)},{torch.max(q)}]'), end='')
            Logger.draw()
            print('\t [closed]')
            train_scores = np.zeros([log_interval, 2])

            assert q.sample_method == q_target.sample_method, f"q-q_target sample_method mismatched {q.sample_method,q_target.sample_method}"
            assert q.w_explore == q_target.w_explore, f"q-q_target w_explore mismatched {q.w_explore,q_target.w_explore}"
            assert q.epsilon == q_target.epsilon, f"q-q_target epsilon mismatched {q.epsilon,q_target.epsilon}"
            assert q.rationality == q_target.rationality, f"q-q_target rationality mismatched {q.rationality,q_target.rationality}"


    Logger.save()
    torch.save(q, Logger.save_dir + f'QModel_{ALG}_{TYPE}.torch')

    env.close()
    # Logger.close()
    test_env.close()
    print(f'\n\n')


if __name__ == "__main__":
    max_episodes = int(20e3)
    """################# working ################################
    config = {
        'iWorld': 1,
        'lr': 0.0001,  # increase if learns no policy decrease if unstable rewards
        # 'lr': 0.00005,  # 'lr': 0.0005,
        'gamma': 0.90,
        'lambda': [10, 10],
        'sp': [5, 0.05],
        'epsilon': [0.0, 0.0],  # [0.6, 0.05],
        'r_scale': None,
        'action_sampler': 'NR_max',
        'max_episodes': max_episodes,
        'no_penalty_steps': 0,  # int((1/4) * max_episodes),
        'warm_up_steps': 2000,
        'full_explore_steps': 1000,  # 'full_explore': 100.0,
        'full_exploit_steps': 3000,  # 'full_exploit': 0.1,
        'batch_size': 32,
        'buffer_limit': 50000,  # increase if learns bad polciy after learning good one
        'log_interval': 100,
        'test_episodes': 5,
        'update_iter': 10,
        'seed': None
    }
    ##############################################################"""
    config = {
        'iWorld':    1,
        'r_scale':   1, # None
        'lr':        0.00005, # increase if learns no policy decrease if unstable rewards
        'gamma':     0.90,
        'type': 'Baseline',
        'lambda':    [10, 10],
        'sp':        [4.0, 0.2], 'epsilon':   [0.0, 0.00],
        # 'sp':        [0.0, 0.00], 'epsilon':   [0.9, 0.05],
        'action_sampler': 'NR_max',
        'max_episodes': max_episodes,
        'no_penalty_steps': 0,
        'warm_up_steps': 1000,
        'full_explore_steps': 0,
        'full_exploit_steps': 3000,
        'batch_size': 64,
        'buffer_limit': 100_000, # increase if learns bad policy after learning good one
        'log_interval': 100,
        'test_episodes': 5,
        'update_iter': 10,
        'seed': 0
    }
    # config = {
    #     'iWorld': 1,
    #     'lr': 0.00005,  # 'lr': 0.0005,
    #     'gamma': 0.95,
    #     'lambda': [2, 2],
    #     'sp': [2, 0.05],
    #     'max_episodes': 20000,
    #     'no_penalty_steps': 10000,
    #     'warm_up_steps': 1000,
    #     'full_explore_steps': 1000,  # 'full_explore': 100.0,
    #     'full_exploit_steps': 1000,  # 'full_exploit': 0.1,
    #     'batch_size': 32, 'buffer_limit': 50000,
    #     'log_interval': 100,
    #     'test_episodes': 5,
    #     'update_iter': 10,  # 'update_iter': 10,
    # }


    BATCH_WORLDS = [3,4,5,6,7]
    #LR = [None, 0.00005,0.00005,0.00005,0.00005,0.00005,0.00005,0.00005]
    for iworld in BATCH_WORLDS:
        config['iWorld'] = iworld
        run(config)

    # run(config)
