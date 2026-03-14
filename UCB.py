import numpy as np
class UCB:
    def __init__(self,k,c):
        self.k=k
        self.q=[0]*self.k
        self.n=[0]*self.k
        self.t=0
        self.c=c
    def select_action(self):
        for i in range(self.k):
            if self.n[i]==0:
                return i
        return np.argmax(self.q+self.c*np.sqrt(np.log(self.t)/self.n))
    
    def update(self,action,reward):
        self.n[action]+=1
        self.t+=1
        self.q[action]=self.q[action]+(reward-self.q[action])*(1.0/self.n[action])
    
    def reset(self):
        self.t=0
        self.q=[0]*self.k
        self.n=[0]*self.k
        