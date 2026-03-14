from bandit_env import Bandit

bandit=Bandit(10)
print("True mean:",bandit.q)
print("Optimal arm",bandit.optimal_action())

for i in range(5):
    print("Reward from arm 0",bandit.pull(0))

    