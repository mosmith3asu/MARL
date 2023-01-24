import numpy as np
# import matplotlib.pyplot as plt

from IQN.QLearning import Qfunction


data = {}
algorithm = 'JointQ'
for iworld in range(1,8):
    for policy_type in ['Baseline','Averse','Seeking']:
        Q = Qfunction.load(iworld,policy_type,algorithm)
        data[f'W{iworld}{policy_type}'] = Q.tbl.detach().numpy()


fname = 'Qfunctions.npz'
np.savez_compressed(fname,**data)
print(f'[{fname}] Saved...')


loaded = np.load(fname)
for key in loaded.keys():
    print(f'[{key}] Loaded: {type(loaded[key])}')