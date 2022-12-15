import numpy as np
# import matplotlib.pyplot as plt

class TestClass(object):
    shared_attribute = 0
    def __init__(self):
        self.solo_attribute = 0
        TestClass.shared_attribute += 1
        print('init ')
    def __setattr__(self, key, value):
        if key in ['shared_attribute']:
            TestClass.__dict__ = value
        else:
            self.__dict__[key] = value
    # @property
    # def shared_attribute_p(self):
    #     super(TestClass, self).__init__()


if __name__ == "__main__":
    # t1 = TestClass()
    # t2 = TestClass()
    #
    # TestClass.shared_attribute = None
    # t1.solo_attribute = 1
    #
    # print(f't1:{t1.shared_attribute} t2:{t2.shared_attribute}')
    # #print(f't1:{t1.shared_attibute} t2:{t2.shared_attribute}')
    # print(f't1:{t1.solo_attribute} t2:{t2.solo_attribute}')
    # start = 2
    # stop = 0.05
    #
    # slope = 1 / 1.5
    # n_improve_epi = (2000)
    # iinv = np.power(1 / (np.linspace(1, 10, n_improve_epi) - 0.1) - 0.1, slope)
    # print(([[0,-1]])

    # r_pow = 1
    # f_scaleR = lambda _rewards: [np.sign(r) * pow(r, r_pow) for r in _rewards]
    import torch
    #import pycuda.driver as cuda

    # cuda.init()
    # print(torch.cuda.current_device())
    # print(cuda.Device(0).name() )
    print(torch.cuda.is_available())
    lr = 0.01*pow(0.9999,10000)

    print(lr)