from algorithm.idqn.idqn import run
if __name__ == "__main__":

    config = {
              'iWorld': 2,
              'lr': 0.0005,
              'batch_size': 32,
              'gamma': 0.95,
              'buffer_limit': 50000,
              'log_interval': 100,
              'max_episodes': 20000,
              'max_epsilon': 0.8,
              'min_epsilon': 0.1,
              'test_episodes': 5,
              'warm_up_steps': 1000,
              'no_penalty_steps': 3000,
              'update_iter': 10,
              'monitor': False,
    }


    BATCH_WORLDS = [1,2,3,4,5,6,7]
    # BATCH_WORLDS = [ 4]
    for iworld in BATCH_WORLDS:
        config['iWorld'] = iworld
        run(**config)




    """
    default_config = {  # 'env_name': 'ma_gym:Switch2-v1',
        'iWorld': 1,
        'lr': 0.0005,
        'batch_size': 32,
        'gamma': 0.99,
        'buffer_limit': 50000,
        'log_interval': 20,
        'max_episodes': 30000,
        'max_epsilon': 0.9,
        'min_epsilon': 0.1,
        'test_episodes': 5,
        'warm_up_steps': 1000,
        'no_penalty_steps': 2000,
        'update_iter': 10,
        'monitor': True,
    }
    """