from enviorment.make_env import CPTEnv

def main(episodes = 100):
    env = CPTEnv()
    env.reset()
    env.render()
    obs_n = env.reset()
    for ep_i in range(episodes):
        done = False
        ep_reward = 0
        env.seed(ep_i)
        obs_n = env.reset()
        env.render()

        while not done:
            obs_n, reward_n, done, info = env.step(env.action_space.sample())
            ep_reward += sum(reward_n)
            env.render()
            # time.sleep(0.5)

        print('Episode #{} Reward: {}'.format(ep_i, ep_reward))
    env.close()
if __name__ == "__main__":
    main()