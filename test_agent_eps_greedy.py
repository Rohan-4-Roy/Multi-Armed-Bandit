from eps_greedy import EpsGreedyAgent
from bandit_env import Bandit
steps=1000
bandit=Bandit(10)
agent=EpsGreedyAgent(10,eps=0.1)
print(agent.q)
for t in range(steps):
    action=agent.select_action()
    rwd=bandit.pull(action)
    agent.update(action,rwd)
print(agent.q)