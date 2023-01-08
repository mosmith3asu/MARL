import yaml
import torch
from dataclasses import dataclass

@dataclass
class Config:
    def __init__(self,path=None,depth=1,**kwargs):
        self._depth = depth
        if path is not None:
            with open(path, 'r') as file:
                kwargs = yaml.safe_load(file)
        for key in kwargs.keys():
            val = kwargs[key]


            if key == 'dtype':
                val = torch.__dict__[val]

            self.__dict__[key] =  Config(depth=depth+1,**val) if isinstance(val,dict) else val
    def __repr__(self):
        res = ''
        if self._depth ==1: res += '\nConfiguration:'
        for key in self.__dict__:
            if key != "_depth":
                tabs = "".join(['\t' for _ in range(self._depth)])
                res+=f'\n{tabs}| {key}: {self.__dict__[key]}'

        if self._depth == 1: res += '\n'
        return res
    def __getitem__(self, key):
        return self.__dict__[key]
CFG = Config(r"C:\Users\mason\Desktop\MARL\IQN\config.yaml")
