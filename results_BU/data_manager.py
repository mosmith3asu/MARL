# import numpy as np
# import matplotlib.pyplot as plt
import logging
import os
from os import listdir


def change_name(dir,old_fname,new_fname):
    # old_fname = "Fig_IQN_averse.png"
    # new_fname = "Fig_JointQ_averse.png"

    found_file = False
    for fname in listdir(dir):
        if fname == old_fname:
            found_file = True
            os.rename(dir + old_fname, dir + new_fname)
            break

    if not found_file: logging.warning(f'did not find file {old_fname}')
    else: print(f'Changed >> { old_fname} to {new_fname} in \t {dir}')



def main():
    algorithm_name0 = 'JointQ' # old name => new name
    algorithm_name1 = 'JointQ'

    for iWorld in [1,2,3,4,5,6,7]:
        print(f'\n World {iWorld} ##########################')
        dir = f'C:\\Users\\mason\\Desktop\\MARL\\results\\IDQN_W{iWorld}\\'


        for policy_type in ['Baseline','Averse','Seeking']:
            # changes = [[dir,f"Fig_{algorithm_name0}_{policy_type}.png",f"Fig_{algorithm_name1}_{policy_type}.png"],
            #            [dir,f"{algorithm_name0}_{policy_type}.torch",f"{algorithm_name1}_{policy_type}.torch"]]
            changes = [[dir, f"Fig_{algorithm_name0}_{policy_type}.png", f"Fig_W{iWorld}_{algorithm_name1}_{policy_type}.png"],
                       [dir, f"{algorithm_name0}_{policy_type}.torch", f"{algorithm_name1}_{policy_type}.torch"]]
            for change in changes:
                change_name(*change)


    # for iWorld in [1,2,3,4,5,6,7]:
    #     dir = f'C:\\Users\\mason\\Desktop\\MARL\\results\\IDQN_W{iWorld}\\'
    #
    #     for fname in  listdir(dir):
    #         new_fname = fname
    #
    #         if 'QModel' in fname:
    #             new_fname = new_fname.replace("QModel_","")
    #
    #
    #         if  new_fname.split("_")[0] in ['Baseline',"Seeking",'Averse']:
    #             _fname = new_fname
    #             _fname = new_fname.replace(".torch","").split("_")
    #             _fname.reverse()
    #             new_fname = "_".join(_fname) + ".torch"
    #
    #         if new_fname.split("_")[0] in ["extended"]:
    #             _fname = new_fname
    #             _fname = new_fname.replace(".torch", "").split("_")
    #             _fname = [_fname[1],_fname[2],_fname[0]]
    #             new_fname = "_".join(_fname) + ".torch"
    #
    #         if fname != new_fname:
    #             print(f'[{iWorld}] Changing {fname} to {new_fname}')
    #             os.rename(dir+fname, dir+new_fname)







    pass


def subfun():
    pass


if __name__ == "__main__":
    main()
