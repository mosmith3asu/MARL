import copy
import logging
import math
import warnings

from enviorment.make_worlds import WorldDefs
import numpy as np

from gym import Env
from gym import spaces
from gym.utils import seeding

from ma_gym.envs.utils.action_space import MultiAgentActionSpace
from ma_gym.envs.utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text
from ma_gym.envs.utils.observation_space import MultiAgentObservationSpace

logger = logging.getLogger(__name__)
class CPTEnv(Env):
    """
        Predator-prey involves a grid world, in which multiple predators attempt to capture randomly moving prey.
        Agents have a 5 × 5 view and select one of five actions ∈ {Left, Right, Up, Down, Stop} at each time step.
        Prey move according to selecting a uniformly random action at each time step.
        We define the “catching” of a prey as when the prey is within the cardinal direction of at least one predator.
        Each agent’s observation includes its own coordinates, agent ID, and the coordinates of the prey relative
        to itself, if observed. The agents can separate roles even if the parameters of the neural networks are
        shared by agent ID. We test with two different grid worlds: (i) a 5 × 5 grid world with two predators and one prey,
        and (ii) a 7 × 7 grid world with four predators and two prey.
        We modify the general predator-prey, such that a positive reward is given only if multiple predators catch a prey
        simultaneously, requiring a higher degree of cooperation. The predators get a team reward of 1 if two or more
        catch a prey at the same time, but they are given negative reward −P.We experimented with three varying P vales,
        where P = 0.5, 1.0, 1.5.
        The terminating condition of this task is when all preys are caught by more than one predator.
        For every new episodes , preys are initialized into random locations. Also, preys never move by themself into
        predator's neighbourhood
        """
    metadata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self,iWorld, full_observable=True, penalty=-3.0, prey_capture_reward=20, max_steps=20, step_cost = 0.0):
        world = WorldDefs.world[iWorld]


        self.dist_cost = 0.0
        self.distance_reward = 0.0
        n_agents = 2
        self.nActions = 5

        grid_shape = world.grid_shape
        self._walls             = world.walls
        self._start_obs         = world.start_obs
        self._penalty_states    = world.penalty_states

        self.AGENT_COLOR    = world.AGENT_COLOR
        self.PREY_COLOR     = world.PREY_COLOR
        self.CELL_SIZE      = world.CELL_SIZE
        self.WALL_COLOR     = world.WALL_COLOR
        self.PEN_COLOR      = world.PEN_COLOR

        self.disable_prey = False
        assert len(grid_shape) == 2, 'expected a tuple of size 2 for grid_shape, but found {}'.format(grid_shape)
        assert grid_shape[0] > 0 and grid_shape[1] > 0, 'grid shape should be > 0'

        self.enable_penalty = True
        self._grid_shape = grid_shape
        self.n_agents = n_agents
        self._max_steps = max_steps
        self._step_count = None
        self._steps_beyond_done = None
        self._penalty = penalty
        self._p_penaly = 0.5
        self._cumulative_penalty = [0,0]
        self._step_cost = step_cost
        self._prey_capture_reward = prey_capture_reward
        self.prey_rationality = 1

        self.action_space = MultiAgentActionSpace([spaces.Discrete(5) for _ in range(self.n_agents)])
        self.agent_pos = {_: None for _ in range(self.n_agents)}
        self.prey_pos = None
        self._prey_alive = None

        self._done = False
        self.viewer = None
        self.full_observable = full_observable
        self._obs_high = grid_shape[0]*np.ones(2*(self.n_agents+1))
        self._obs_low = np.zeros(2*(self.n_agents+1))
        self.observation_space = MultiAgentObservationSpace(
                [spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)])

        self.max_dist = math.dist([0, 0], list(self._grid_shape))

        self._total_episode_reward = None
        self.seed()

    ######################
    def step(self, agents_action):
        assert (self._step_count is not None), \
            "Call reset before using step method."

        self._step_count += 1
        # rewards = [self._step_cost for _ in range(self.n_agents)]
        rewards = [0 for _ in range(self.n_agents)]

        # Agents take action #########
        for agent_i, action in enumerate(agents_action):
            if not self._done:
                self.__update_agent_pos(agent_i, action)
                # if self.is_penalty(self.agent_pos[agent_i]): self._cumulative_penalty[agent_i] += self._p_penaly*self._penaltyce

        # Get stage reward
        for agent_i in range(self.n_agents):
            if self.is_penalty(self.agent_pos[agent_i]) and self.enable_penalty:
                rewards[agent_i] += self._p_penaly*self._penalty
            if self.distance_reward >0:
                rewards[agent_i] += self._get_dist_reward(agent_i)

        # Prey Takes Action if Not Caught ###########################
        if self._prey_caught(self.prey_pos):
            self._done = True

            for agent_i in range(self.n_agents):
                rewards[agent_i] += self._prey_capture_reward
                rewards[agent_i] += -1*self._step_count
                #rewards[agent_i] += self._cumulative_penalty[agent_i]
        else: # Move Prey
            prey_move = self._prey_decide_action()
            prey_move = 4 if prey_move is None else prey_move  # default is no-op(4)
            self.__update_prey_pos(self.prey_pos, prey_move)

        # Close Step ##########################################
        if self._step_count >= self._max_steps: self._done = True # Check if terminal
        for i in range(self.n_agents): self._total_episode_reward[i] += rewards[i] # update cumulative reward

        # Check for episode overflow
        if self._done:
            if self._steps_beyond_done is None:
                self._steps_beyond_done = 0
            else:
                if self._steps_beyond_done == 0:
                    logger.warning(
                        "You are calling 'step()' even though this "
                        "environment has already returned all(done) = True. You "
                        "should always call 'reset()' once you receive "
                        "'all(done) = True' -- any further steps are undefined "
                        "behavior."
                    )
                self._steps_beyond_done += 1

        return self.get_agent_obs(), rewards, self._done, None #{'prey_alive': self._prey_alive}

    def reset(self):
        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        self.agent_pos = {}
        self.prey_pos = {}
        self.__init_obs()
        self._done = False
        self._step_count = 0
        self._steps_beyond_done = None
        return self.get_agent_obs()

    def render(self, mode='human'):
        assert (self._step_count is not None), \
            "Call reset before using render method."

        img = copy.copy(self._base_img)
        # for agent_i in range(self.n_agents):
        #     draw_circle(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_COLOR)
        #     write_cell_text(img, text=['R','H'][agent_i], pos=self.agent_pos[agent_i], cell_size=CELL_SIZE,
        #                     fill='white', margin=0.4)
        #
        # draw_circle(img, self.prey_pos, cell_size=CELL_SIZE, fill=PREY_COLOR)
        # write_cell_text(img, text='P', pos=self.prey_pos, cell_size=CELL_SIZE, fill='white', margin=0.4)
        for agent_i in range(self.n_agents):
            draw_circle(img, self.agent_pos[agent_i], cell_size=self.CELL_SIZE, fill=self.AGENT_COLOR)
            write_cell_text(img, text=['R','H'][agent_i], pos=self.agent_pos[agent_i], cell_size=self.CELL_SIZE,
                            fill='white', align='center')

        draw_circle(img, self.prey_pos, cell_size=self.CELL_SIZE, fill=self.PREY_COLOR)
        write_cell_text(img, text='P', pos=self.prey_pos, cell_size=self.CELL_SIZE, fill='white', align='center')

        img = np.asarray(img)
        if mode == 'rgb_array': return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None: self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    def seed(self, n=None):
        self.np_random, seed = seeding.np_random(n)
        return [seed]

    ######################
    @property
    def is_caught(self): return self._prey_caught(self.prey_pos)

    @property
    def step_count(self): return self._step_count

    def _prey_decide_action(self):
        lam = self.prey_rationality
        _prey_move_probs = np.zeros(self.nActions)
        for action_i in range(self.nActions):
            _move = action_i
            _pos = self.__next_pos(self.prey_pos, _move)
            if not self._prey_caught(_pos):
                agent_dist = [math.dist(self.agent_pos[agent_i],_pos) for agent_i in range(self.n_agents)]
                _prey_move_probs[action_i] = np.linalg.norm(agent_dist,ord=2)

        _prey_move_probs = np.exp(lam*_prey_move_probs)#/np.sum(lamb*_prey_move_probs)
        _prey_move_probs = _prey_move_probs/np.sum(_prey_move_probs)
        _move = self.np_random.choice(len(_prey_move_probs), 1, p=_prey_move_probs)[0]

        if  self.disable_prey: _move=4

        return _move

    def _prey_caught(self,prey_pos):
        adjacent = [math.dist(self.agent_pos[agent_i],prey_pos) for agent_i in range(self.n_agents)]
        if np.all(np.array(adjacent)<=1): return True
        else: return False
    def _get_dist_reward(self,agent_i):
        # d2prey = [math.dist(self.agent_pos[agent_i],self.prey_pos) for agent_i in range(self.n_agents)]
        # r_d2prey = [self.distance_reward * (self.max_dist-d)/self.max_dist for d in d2prey]
        d2prey = math.dist(self.agent_pos[agent_i], self.prey_pos)
        r_d2prey = self.distance_reward * (self.max_dist - d2prey) / self.max_dist
        return r_d2prey

    def get_action_meanings(self, agent_i=None):
        if agent_i is not None:
            assert agent_i <= self.n_agents
            return [ACTION_MEANING[i] for i in range(self.action_space[agent_i].n)]
        else:
            return [[ACTION_MEANING[i] for i in range(ac.n)] for ac in self.action_space]
    def action_space_sample(self):
        return [agent_action_space.sample() for agent_action_space in self.action_space]
    def __draw_base_img(self):
        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=self.CELL_SIZE, fill='white')
        for wall in self._walls:
            fill_cell(self._base_img, wall, cell_size=self.CELL_SIZE, fill=self.WALL_COLOR, margin=0.0)
        for pen in self._penalty_states:
            fill_cell(self._base_img, pen, cell_size=self.CELL_SIZE, fill=self.PEN_COLOR, margin=0.0)

    def __init_obs(self):
        _start_obs = copy.deepcopy(self._start_obs)
        for agent_i in range(self.n_agents):
            self.agent_pos[agent_i] = _start_obs[agent_i]
        self.prey_pos = _start_obs[-1]
        self.__draw_base_img()

    def get_agent_obs(self):
        _obs = []
        for agent_i in range(self.n_agents):  _obs += self.agent_pos[agent_i]
        _obs += self.prey_pos
        return [_obs for _ in range(self.n_agents)]

    def __update_agent_pos(self, agent_i, move):

        curr_pos = copy.copy(self.agent_pos[agent_i])
        next_pos = None

        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:
            next_pos = None
            # next_pos = curr_pos
        else:
            raise Exception('Action Not found!')

        if next_pos is not None: #and self._is_cell_vacant(next_pos):
            if self.is_valid(next_pos):
                self.agent_pos[agent_i] = next_pos

    def __next_pos(self, curr_pos, move):
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            next_pos = curr_pos
        return next_pos

    def __update_prey_pos(self, prey_pos, move):
        curr_pos = copy.deepcopy(prey_pos)
        if not self._done:
            next_pos = None
            if move == 0:  # down
                next_pos = [curr_pos[0] + 1, curr_pos[1]]
            elif move == 1:  # left
                next_pos = [curr_pos[0], curr_pos[1] - 1]
            elif move == 2:  # up
                next_pos = [curr_pos[0] - 1, curr_pos[1]]
            elif move == 3:  # right
                next_pos = [curr_pos[0], curr_pos[1] + 1]
            elif move == 4:  # no-op
                pass
            else:
                raise Exception('Action Not found!')


            if next_pos is not None:  # and self._is_cell_vacant(next_pos):
                if self.is_valid(next_pos):
                    self.prey_pos = next_pos
                # self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
                # self.__update_prey_view(prey_i)
            else:
                # print('prey pos not updated')
                pass
        else:
            print('prey already caught')
            #self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']

    #############################

    def is_penalty(self,pos):
        is_pen = False
        for pen_pos in self._penalty_states:
            if np.all(pen_pos == pos): is_pen = True
        return is_pen
    def is_valid(self, pos):

        is_wall = False
        for wall_pos in self._walls:
            if np.all(wall_pos==pos): is_wall = True
        return not is_wall
        #is_wall = False# (np.array(pos) in np.array(self._walls))
        # in_bounds = (0 < pos[0] < self._grid_shape[0]) and (0 < pos[1] < self._grid_shape[1]-1)
        #return (not is_wall and in_bounds)




ACTION_MEANING = {
    0: "DOWN",
    1: "LEFT",
    2: "UP",
    3: "RIGHT",
    4: "NOOP",
}

if __name__ == "__main__":
    env = CPTEnv(iWorld=1)
    env.reset()

    while True:
        env.render()
    env.render()

    print(f'nagents = {env.n_agents}')
    print(f'actions = {env.action_space}')
    print(f'obs_space = {env.observation_space}')
    print(f'action meanings = {env.get_action_meanings()}')
    print(f'action sample = {env.action_space.sample()}')
    print(f'env reset = {env.reset()}')
    print('ending...')
    obs_n, reward_n, done_n, info = env.step(env.action_space.sample())
    print(f'obs_n = {obs_n}, reward_n = {reward_n}, done_n = {done_n}, info={info}')

    ep_reward = 0
    done = False
    obs_n = env.reset()
    while not done:
        env.render()
        obs_n, reward_n, done, info = env.step(env.action_space.sample())
        ep_reward += sum(reward_n)
        # time.sleep(0.5)
    env.close()
