import numpy as np
class Bandit:
    def __init__(self,k):
        self.k=k
        self.q=np.random.normal(0,1,k)
        self.optimal=np.argmax(self.q)
    def pull(self,a):
        return np.random.normal(self.q[a],1)
    def optimal_action(self):
        return self.optimal
            