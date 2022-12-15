import collections
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR,CyclicLR
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
    sp = 0.0
    theta = 0.0

    sample_method = 'NR'
    def __init__(self, observation_space, action_space):
        super(QNet, self).__init__()
        self.num_agents = len(observation_space)
        self.n_joint_actions = 25
        self.n_solo_actions = 5
        # self.rationality = 1.0
        # self.sp = 0.0
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
        obs_shape = [7 for _ in range(n_obs)]
        self.action_frequency = np.zeros(obs_shape + [self.n_joint_actions],dtype=int)
        for agent_i in range(self.num_agents):
            setattr(self, 'agent_{}'.format(agent_i), nn.Sequential(nn.Linear(n_obs, 128),
                                                                    nn.ReLU(),
                                                                    nn.Linear(128, 64),
                                                                    nn.ReLU(),
                                                                    nn.Linear(64, self.n_joint_actions)))
                                                                    #nn.Linear(64, action_space[agent_i].n)))
        # model = Pipe(model, chunks=8) # https://pytorch.org/docs/stable/pipeline.html

    def forward(self, obs):
        q_values = [torch.empty(obs.shape[0], )] * self.num_agents
        for agent_i in range(self.num_agents):
            q_values[agent_i] = getattr(self, 'agent_{}'.format(agent_i))(obs[:, agent_i, :]).unsqueeze(1)
        return torch.cat(q_values, dim=1)

    def add_exploration(self,q,sp,theta,obs):
        global is_training
        #q = q.flatten()
        if is_training:
            pass
        sp = sp # noise size w.r.t. the range of qualities (decrease => less exploration)
        qi_range = (torch.max(q,dim=1).values - torch.min(q,dim=1).values).numpy()  # range of q values
        n_batch = qi_range.shape[0]
        # Undirected Exploration
        if sp > 0:
            psi = np.random.exponential(size= [n_batch,self.n_joint_actions])  # exponential dist. scaled to match range of q-values
            sf = (sp * qi_range / np.max(psi)) # scaling factor
            eta =  np.array(sf * psi,dtype=np.float32)
            eta = torch.from_numpy(eta)
        else: eta = 0
        # Directed exploration
        #   The directed term gives a bonus to the actions that have been rarely elected
        #   theta is a positive factor used to weight the directed exploration
        # if theta > 0:
        #     x0, y0, x1, y1, x2, y2 = obs[0, 0].detach().numpy().astype(int) if isinstance(obs, torch.Tensor) else obs[0]
        #     n_SA = self.action_frequency[x0, y0, x1, y1, x2, y2, :].flatten()  # the number of time steps in which action U has been elected
        #     n_SA[np.where(n_SA > 100)] = 100
        #     rho = (theta / np.exp(n_SA)).astype(np.float32)
        #     rho = torch.from_numpy(rho)
        # else: rho = 0
        rho = 0

        # return cumulative Q
        return (q + eta + rho).flatten()

    def sample_NR_action(self, obs, sp, theta, rationality,epsilon):
        global is_training

        """
        Sample noisy-rational action
        - softmax sampled on ego quality [1x5]
        - new quality conditined on partner pA
        """
        QsA = self.forward(obs)  # n_batch x n_agents x n_actions

        n_batch,n_agent,n_act = QsA.shape
        all_batch,all_agent,all_act =np.arange(n_batch),np.arange(n_agent),np.arange(n_act)

        if np.random.rand() <= epsilon:
            aJ = torch.randint(0, n_act, [n_batch, 1])
        else:
            # Level 0 sophistication
            #_qAH = self.add_exploration(QsA[0, 1].detach(), sp=sp, theta=theta, obs=obs)
            #_qAR = self.add_exploration(QsA[0, 0].detach(), sp=sp, theta=theta, obs=obs)
            _qAH = QsA[all_batch, 1].detach()
            _qAR = QsA[all_batch, 0].detach()
            # _qAH = self.add_exploration(QsA[all_batch, 1].detach(), sp=sp, theta=theta, obs=obs)
            # _qAR = self.add_exploration(QsA[all_batch, 0].detach(), sp=sp, theta=theta, obs=obs)

            phhatA_H = torch.ones([n_batch, self.n_solo_actions], dtype=_qAH.dtype) / self.n_solo_actions
            phhatA_R = torch.ones([n_batch, self.n_solo_actions], dtype=_qAR.dtype) / self.n_solo_actions
            phhatA_H = torch.matmul(phhatA_H, self.ijoint[1])
            phhatA_R = torch.matmul(phhatA_R, self.ijoint[0])

            # Level 1 sophistication
            # qhat_H = torch.matmul(self.ijoint[1], (_qAH * phhatA_R).T).flatten()
            # qhat_R = torch.matmul(self.ijoint[0], (_qAR * phhatA_H).T).flatten()
            qhat_H = torch.matmul(_qAH * phhatA_R, self.ijoint[1].T)
            qhat_R = torch.matmul(_qAR * phhatA_H, self.ijoint[0].T)
            phatA_H = torch.matmul(torch.special.softmax(rationality * qhat_H, dim=1) ,self.ijoint[1])
            phatA_R = torch.matmul(torch.special.softmax(rationality * qhat_R, dim=1) ,self.ijoint[0])

            # Select Ego Action
            # q_H = torch.matmul(self.ijoint[1], (_qAH * joint_phatA_R).T).flatten()
            # q_R = torch.matmul(self.ijoint[0], (_qAR * joint_phatA_H).T).flatten()
            q_H = torch.matmul(_qAH * phatA_R, self.ijoint[1].T)
            q_R = torch.matmul(_qAR * phatA_H, self.ijoint[0].T)
            pA_H = torch.special.softmax(rationality * q_H, dim=1)
            pA_R = torch.special.softmax(rationality * q_R, dim=1)
            if self.sample_method == 'NR_max':
                aH = torch.argmax(pA_H,dim=1).int()
                aR = torch.argmax(pA_R,dim=1).int()
            elif self.sample_method == 'NR':
                aH = np.random.choice(np.arange(self.n_solo_actions), p=pA_H.detach().numpy())
                aR = np.random.choice(np.arange(self.n_solo_actions), p=pA_R.detach().numpy())
            else:  raise Exception("unknown NR sampliong method")

            #self.sample_action_report(qA=[q_R,q_H],pA=[pA_R,pA_H])
            # Get joint action
            aJ = torch.tensor([self.solo2joint[aR[i], aH[i]] for i in range(n_batch)]).reshape([n_batch,1])
        action = torch.cat((aJ,aJ),1)
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

    def sample_SW_action(self, obs, sp, theta, rationality,epsilon):
        """Sample social welfare action"""
        QsA = self.forward(obs)
        if np.random.rand() <= epsilon:
            aJ = torch.randint(0, QsA.shape[2], [1, 1])
        else:
            _qAH = self.add_exploration(QsA[0, 1].detach(), sp=sp,theta=theta, obs=obs)
            _qAR = self.add_exploration(QsA[0, 0].detach(), sp=sp,theta=theta, obs=obs)
            aJ = torch.argmax( _qAR+_qAH)
        action = aJ * torch.ones((QsA.shape[0], QsA.shape[1],))
        return action

    def sample_action(self, obs, sp=None,theta=None,rationality=None,epsilon=None,best=False):
        # Load Defaults
        if rationality is None: rationality = self.rationality
        if sp is None: sp = self.sp
        if epsilon is None: epsilon = self.epsilon
        if theta is None: theta = self.theta
        if best: sp,theta,epsilon = 0,0,0

        assert sp == 0 or epsilon==0, f"sample_action has both exploration policies [eps:{epsilon} sp:{sp}]"

        if self.sample_method in ['NR','NR_max']:
            action = self.sample_NR_action(obs, sp = sp, theta=theta, rationality=rationality,epsilon=epsilon)
        elif self.sample_method == 'SW':
            action = self.sample_SW_action(obs,  sp = sp, theta=theta, rationality=rationality, epsilon=epsilon)
        else: raise Exception(f'unknown sample method [{self.sample_method}]')

        if theta > 0:
            x0, y0, x1, y1, x2, y2 = obs[0,0].detach().numpy().astype(int) if isinstance(obs, torch.Tensor) else obs[0]
            aJ = int(action[0, 0].detach()) if isinstance(action,torch.Tensor) else action[0]
            self.action_frequency[x0,y0,x1,y1,x2,y2,aJ] += 1
        return action



def train(q, q_target, memory, optimizer, gamma, batch_size, update_iter=10,lr_scheduler=None):
    global is_training
    is_training = True
    for _ in range(update_iter):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(2, a.unsqueeze(-1).long()).squeeze(-1)
        # Quantal Response Equilibrium ########################
        qs_prime = q_target(s_prime)
        a_prime = q_target.sample_action(torch.Tensor(s_prime), best=True)
        max_q_prime = torch.empty([qs_prime.shape[0], qs_prime.shape[1]])
        for k in [0,1]: max_q_prime[:,k] = qs_prime[np.arange(batch_size),k,a_prime[:,k].numpy()]


        # # Quantal Response Equilibrium ########################
        # qs_prime = q_target(s_prime)
        # max_q_prime = torch.empty([qs_prime.shape[0], qs_prime.shape[1]])
        # for i,si_prime in enumerate(s_prime):
        #     a_prime = q_target.sample_action(torch.Tensor(si_prime).unsqueeze(0), best=True).flatten()[0]
        #     max_q_prime[i, :] = qs_prime[i, :, a_prime.long()]

        # DEFAULT ####################
        # max_q_prime = q_target(s_prime).max(dim=2)[0] # 32 x 2
        # DEFAULT ####################

        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if lr_scheduler is not None:
        lr_scheduler.step()

def test(env, num_episodes, q):
    old_sp = q.sp
    old_theta = q.theta
    old_epsilon = q.epsilon
    old_rationality = q.rationality

    length =np.zeros(env.n_agents)
    psucc = np.zeros(env.n_agents)
    score = np.zeros(env.n_agents)
    for episode_i in range(num_episodes):
        state = env.reset()
        done = False# [False for _ in range(env.n_agents)]
        while not done:
            action = q.sample_action(torch.Tensor(state).unsqueeze(0), best=True)
            action = action[0].data.cpu().numpy().tolist()
            next_state, reward, done, info = env.step(action,is_joint=True)
            score += np.array(reward)
            state = next_state

        if env.is_caught: psucc +=1
        length += env.step_count

    q.sp = old_sp
    q.theta = old_theta
    q.epsilon = old_epsilon
    q.rationality = old_rationality

    final_score = np.mean(score / num_episodes)
    final_length =np.mean(length / num_episodes)
    final_psucc = np.mean(psucc / num_episodes)
    return final_score,final_length,final_psucc

def schedule(start,end,N,slope=1.0):
    if start == end: return start*np.ones(N)
    iinv = np.power(1 / (np.linspace(1, 10, N) - 0.1) - 0.1, slope)
    return (start + end) * iinv + end
    #return np.linspace(start, end, N)

def print_dict(this_dict,title='Dictionary'):
    print(title)
    for key in this_dict.keys():
        print('\t| {:<20}: {:<20}'.format(key,f'{this_dict[key]}'))
    print(f'\n\n')
def run(run_config):
    global is_training
    is_training = False
    # Unpack configuration -------------------------------
    print_dict(run_config, title='Run Config')
    iWorld = run_config['iWorld']
    lr = run_config['lr']
    gamma = run_config['gamma']
    start_lamb,end_lamb = run_config['lambda']
    start_theta,end_theta = run_config['theta']
    start_sp, end_sp = run_config['sp']
    start_epsilon, end_epsilon = run_config['epsilon']
    action_sampler = run_config['action_sampler']
    optimizer_name = run_config['optimizer']
    schedule_lr_enable = run_config['schedule_lr']
    schedule_slope = run_config['schedule_slope']
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

    # Set up logger -------------------------------
    # plt.close()
    TYPE = 'Baseline'
    ALG = 'Joint'
    Logger.file_name = f'Fig_IDQN_{ALG}_{TYPE}'
    Logger.update_save_directory(Logger.project_root + f'results\\IDQN_W{iWorld}\\')
    Logger.update_plt_title(f'W{iWorld} IDQN Training Results')
    Logger.make_directory()
    Logger.filter_window = 10

    # Create environment -------------------------------
    if SCALE_REWARDS is not None:
        f_scaleR = lambda _rewards: [np.sign(r) * abs(pow(r, SCALE_REWARDS))for r in _rewards]
    else: f_scaleR = None

    env = CPTEnv(iWorld)
    test_env = CPTEnv(iWorld)

    env.f_scaleR = f_scaleR
    memory = ReplayBuffer(buffer_limit)

    # Create Q functions -------------------------------
    if seed is not None: torch.manual_seed(seed)
    q = QNet(env.observation_space, env.action_space)
    q.settings = run_config

    q_target = QNet(env.observation_space, env.action_space)
    q_target.load_state_dict(q.state_dict())
    if optimizer_name=='Adam': optimizer = optim.Adam(q.parameters(), lr=lr)
    elif optimizer_name=='SparseAdam': optimizer = optim.SparseAdam(q.parameters(), lr=lr)
    elif optimizer_name == 'SGD': optimizer = optim.SGD(q.parameters(),lr=lr)
    else: raise Exception(f'Unknown optimizer name {optimizer_name}')
    if schedule_lr_enable:
        #lr_scheduler = ExponentialLR(optimizer, gamma=0.9)
        lr_scheduler = CyclicLR(optimizer, base_lr=lr, max_lr=lr*10)
    else: lr_scheduler = None

    # SCHEDULE EXPLORATION POLICY -------------------------------
    #improve_slope = 1 / 1.5
    improve_slope = schedule_slope
    n_improve_epi = (max_episodes - full_explore_steps - full_exploit_steps)

    warmup_sp  = start_sp * np.ones(full_explore_steps)
    perform_sp = end_sp * np.ones(full_exploit_steps)
    improve_sp = schedule(start_sp,end_sp,n_improve_epi,slope=improve_slope)
    epi_sp = np.hstack([warmup_sp, improve_sp, perform_sp])

    warmup_theta = start_theta * np.ones(full_explore_steps)
    perform_theta = end_theta * np.ones(full_exploit_steps)
    improve_theta = schedule(start_theta, end_theta, n_improve_epi, slope=improve_slope)
    epi_theta = np.hstack([warmup_theta, improve_theta, perform_theta])

    warmup_epsilons = start_epsilon * np.ones(full_explore_steps)
    perform_epsilons = end_epsilon * np.ones(full_exploit_steps)
    improve_epsilons = schedule(start_epsilon,end_epsilon,n_improve_epi,slope=improve_slope)
    epi_epsilons = np.hstack([warmup_epsilons, improve_epsilons,perform_epsilons])

    warmup_lamb  = start_lamb * np.ones(full_explore_steps)
    perform_lamb = end_lamb * np.ones(full_exploit_steps)
    improve_lamb = schedule(start_lamb, end_lamb, n_improve_epi, slope=improve_slope)
    epi_rationality = np.hstack([warmup_lamb, perform_lamb, improve_lamb])

    warmup_ended = 0
    train_scores = np.zeros([log_interval,2])
    for episode_i in range(max_episodes):
        if warmup_ended==1: _,warmup_ended = print(f'\n Begin Q-updates...'),-1
        if episode_i == no_penalty_steps:  print(f'\n Penalties added...')
        if episode_i == full_explore_steps:  print(f'\n Ending full explore added...')

        # Get episode parameters
        sp = epi_sp[episode_i]
        epsilon = epi_epsilons[episode_i]
        rationality = epi_rationality[episode_i]
        theta = epi_theta[episode_i]

        q.sample_method = action_sampler
        q.sp = sp
        q.epsilon = epsilon
        q.rationality = rationality
        q.theta = theta

        q_target.sample_method = action_sampler
        q_target.sp = sp
        q_target.epsilon = epsilon
        q_target.rationality = rationality
        q_target.theta = theta

        cumR = np.zeros(2)
        done = False

        # Intialize episode
        state = env.reset()
        env.enable_penalty = True if episode_i > no_penalty_steps else False
        test_env.enable_penalty = True if episode_i > no_penalty_steps else False
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

            if warmup_ended == 0:
                if memory.size() == warm_up_steps: warmup_ended = 1

        # Close episode
        #Logger.log_episode(np.mean(cumR), env.step_count, env.is_caught)
        if memory.size() > warm_up_steps:
            train(q, q_target, memory, optimizer, gamma, batch_size, update_iter=update_iter,lr_scheduler=lr_scheduler)


        # REPORTING ##################
        time2report = (episode_i % log_interval == 0 and episode_i != 0)
        if time2report:
            q_target.load_state_dict(q.state_dict())
            test_score,test_length,test_psucc = test(test_env, test_episodes, q)
            Logger.log_episode(test_score,test_length,test_psucc,buffered=False,episode=episode_i)
            print("\r#{:<20}".format(f'[{episode_i}/{max_episodes} epi]'),end= '')
            print(f'||', end='')
            print("train score: {:<8} ".format(np.round(np.mean(train_scores), 1)), end='')
            print("test score: {:<25}".format(f'[r:{np.round(test_score, 1)} l:{np.round(test_length, 1)} ps:{np.round(test_psucc, 1)}]'), end='')
            print("n_buffer : {:<10}  ".format(memory.size()), end='')
            print(f'||', end='')
            print("eps : {:<6} ".format(round(q.epsilon,2)), end='')
            print("lam : {:<6}".format( round(q.rationality,2)), end='')
            print("sp : {:<6}".format(round(q.sp,2)), end='')
            print("theta : {:<6}".format(round(q.theta, 2)), end='')
            print("sampler : {:<10}".format(q.sample_method), end='')
            print(f'||', end='')
            #print("Qrange : {:<10}".format(f'[{torch.min(q)},{torch.max(q)}]'), end='')
            Logger.draw()
            print('\t [closed]')
            train_scores = np.zeros([log_interval, 2])

            assert q.sample_method == q_target.sample_method, f"q-q_target sample_method mismatched {q.sample_method,q_target.sample_method}"
            assert q.sp == q_target.sp, f"q-q_target sp mismatched {q.sp,q_target.sp}"
            assert q.theta == q_target.theta, f"q-q_target rationality mismatched {q.theta, q_target.theta}"
            assert q.epsilon == q_target.epsilon, f"q-q_target epsilon mismatched {q.epsilon,q_target.epsilon}"
            assert q.rationality == q_target.rationality, f"q-q_target rationality mismatched {q.rationality,q_target.rationality}"


    Logger.save()
    torch.save(q, Logger.save_dir + f'QModel_{ALG}_{TYPE}.torch')

    env.close()
    # Logger.close()
    test_env.close()
    print(f'\n\n')


if __name__ == "__main__":

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
        'iWorld':   1,
        'r_scale':  None, # 2
        'lr':       0.00005, # increase if learns no policy decrease if unstable rewards
        'gamma':    0.99,
        'lambda':   [10, 10],
        'sp':       [0.0, 0.0],#[4.0, 0.01],
        'theta':    [0.0, 0.0],
        'epsilon':  [1.0, 0.1],# [1.0, 0.01],
        'schedule_slope': 1/3, # 1/1.5
        'action_sampler': 'NR_max',
        'optimizer':      'Adam',#'SGD',#'Adam',
        'schedule_lr': False,
        'max_episodes': 30_000,
        'no_penalty_steps': 0,
        'warm_up_steps': 2_000,
        'full_explore_steps': 1000,
        'full_exploit_steps': 0,
        'batch_size': 32,
        'buffer_limit': 75_000, # increase if learns bad policy after learning good one
        'log_interval': 100,
        'test_episodes': 5,
        'update_iter': 10,
        'seed': 0
    }



    BATCH_WORLDS = [1,2,3,4,5,6,7]
    #LR = [None, 0.00005,0.00005,0.00005,0.00005,0.00005,0.00005,0.00005]
    for iworld in BATCH_WORLDS:
        config['iWorld'] = iworld
        run(config)

    # run(config)

