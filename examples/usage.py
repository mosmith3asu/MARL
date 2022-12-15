import gym
import ma_gym
# env = gym.make('PredatorPrey5x5-v0')
env = gym.make('Switch2-v1')
print(f'nagents = {env.n_agents}')
print(f'actions = {env.action_space}')
print(f'obs_space = {env.observation_space}')
print(f'action meanings = {env.get_action_meanings()}')
print(f'action sample = {env.action_space.sample()}')
print(f'env reset = {env.reset()}')
obs_n, reward_n, done_n, info = env.step(env.action_space.sample())
episode_terminate = all(done_n)
team_reward = sum(reward_n)


print('ending...')
#
#
# #### COSTOM ENV #
# import gym
#
# gym.envs.register(
#     id='MySwitch2-v0',
#     entry_point='ma_gym.envs.switch:Switch',
#     kwargs={'n_agents': 2, 'full_observable': False, 'step_cost': -0.2}
#     # It has a step cost of -0.2 now
# )
# env = gym.make('MySwitch2-v0')
#
#
# ###### Monitoring
# import gym
# from ma_gym.wrappers import Monitor
# env = gym.make('Switch2-v0')
# env = Monitor(env, directory='recordings')
# env = Monitor(env, directory='recordings',video_callable=lambda episode_id: True)
# env = Monitor(env, directory='recordings', video_callable=lambda episode_id: episode_id%10==0)