import random
import numpy as np
class EpsGreedyAgent:
    def __init__(self,k,eps):
        self.k=k
        self.eps=eps
        self.q=[0.0] * k
        self.n=[0] * k
        
    def select_action(self):
        prob=random.random()
        if prob > self.eps:
            opt=np.argmax(self.q)
            return opt
        else:
            chosen=random.randint(0,self.k-1)
            return chosen
    
    def update(self,action,reward):
        self.n[action]+=1
        self.q[action]=self.q[action]+(reward-self.q[action])*(1.0/self.n[action])
        
    def reset(self):
        self.q=[0.0] * self.k
        self.n=[0] * self.k
        
        
    