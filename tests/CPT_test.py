import matplotlib.pyplot as plt
import numpy as np
from IQN.utilities.learning_utils import CPT_Handler

def main():
    pass


def subfun():
    pass


if __name__ == "__main__":
    nRows, nCols = 2,3
    fig,axs =plt.subplots(nRows,nCols,constrained_layout = True,figsize=(7*nCols,6*nRows)) # ,
    axs = np.reshape(axs,[nRows,nCols])
    r = 0
    for r in range(nRows):
        for c, policy_type in enumerate(['Baseline','Averse','Seeking']):
            CPT = CPT_Handler.rand(assume=policy_type,p_thresh_sensitivity=0.1)

            print(f'{CPT.p_thresh_sensitivity}')
            CPT.plot_indifference_curve(ax = axs[r,c])

    plt.show()
