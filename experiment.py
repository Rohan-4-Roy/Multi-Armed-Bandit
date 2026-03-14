from bandit_env import Bandit
from bin_bandit_env import BinBandit
from eps_greedy import EpsGreedyAgent
import numpy as np
import matplotlib.pyplot as plt
from UCB import UCB
def run_experiment(agent,runs,steps,k):
    rewards=np.zeros((runs,steps))
    optimal_actions=np.zeros((runs,steps))
    for run in range(runs):
        # bandit=Bandit(k)
        bandit = BinBandit(k)
        agent.reset()
        optimal=bandit.optimal_action()
        for step in range(steps):
            action = agent.select_action()
            reward=bandit.pull(action)
            agent.update(action,reward)
            rewards[run,step]=reward
            if action == optimal: 
                optimal_actions[run,step]=1
    avg_reward=rewards.mean(axis=0)   
    optimal_percentage=optimal_actions.mean(axis=0)  
        
    return avg_reward,optimal_percentage



# k=10
# runs=2000
# steps=1000
# epsilons=[0,0.01,0.1]
# avg_rewards=[]
# optimal_actions=[]
# for eps in epsilons:
#     agent=EpsGreedyAgent(k,eps)
#     avg_reward,opt=run_experiment(agent,runs,steps,k)
#     avg_rewards.append(avg_reward)
#     optimal_actions.append(opt)

# plt.figure()
# for i,eps in enumerate(epsilons):
#     plt.plot(avg_rewards[i],label=f"epsilon={eps}")

# plt.xlabel("Steps")
# plt.ylabel("Average Reward")
# plt.title("Average Reward vs Steps")
# plt.legend()

# plt.savefig("avg_reward_comparison.png", dpi=300)
# plt.show()
# plt.figure()

# for i, eps in enumerate(epsilons):
#     plt.plot(optimal_actions[i], label=f"epsilon = {eps}")

# plt.xlabel("Steps")
# plt.ylabel("% Optimal Action")
# plt.title("Optimal Action vs Steps")
# plt.legend()

# plt.savefig("optimal_action_comparison.png", dpi=300)
# plt.show()

            
            
# agent=UCB(10,2)
# avg_reward,optimal=run_experiment(agent,runs=2000,steps=1000,k=10)

        
# plt.plot(avg_reward)
# plt.xlabel("Steps")
# plt.ylabel("Averag Reward")
# plt.savefig("average_reward_ucb.png")
# plt.show()

# plt.plot(optimal)
# plt.xlabel("Steps")
# plt.ylabel("%Optimal Action")
# plt.savefig("optimalAction_ucb.png")
# plt.show()
                    
        