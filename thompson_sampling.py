import numpy as np
class ThompsonSampling:

    def __init__(self,k):
        self.k=k
        self.alpha=[1]*self.k
        self.beta=[1]*self.k
    
    def select_action(self):
        return np.argmax(np.random.beta(self.alpha,self.beta))

    def update(self,action,reward):
        if reward ==1:
            self.alpha[action]+=1
        else:
            self.beta[action]+=1
    
    def reset(self):
        self.alpha=[1]*self.k
        self.beta=[1]*self.k