import numpy as np
import random
class BinBandit:
    def __init__(self,k):
        self.k=k
        self.pa=np.random.uniform(0,1,k)
    def pull(self,a):
        if random.random()<=self.pa[a]:
            return 1
        else :
            return 0
    def optimal_action(self):
        return np.argmax(self.pa)