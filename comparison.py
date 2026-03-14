from eps_greedy import EpsGreedyAgent
from UCB import UCB
from experiment import run_experiment
from bandit_env import Bandit
from bin_bandit_env import BinBandit
from thompson_sampling import ThompsonSampling
import numpy as np
import matplotlib.pyplot as plt

runs = 2000
steps = 1000
k = 10

agent1 = EpsGreedyAgent(k, 0.1)
agent2 = EpsGreedyAgent(k, 0.01)
agent3 = UCB(k, c=2)

avg_r1, opt1 = run_experiment(agent1, runs, steps, k)
avg_r2, opt2 = run_experiment(agent2, runs, steps, k)
avg_r3, opt3 = run_experiment(agent3, runs, steps, k)

def run_thompson(agent, runs, steps, k):

    rewards = np.zeros((runs, steps))
    optimal_actions = np.zeros((runs, steps))

    for run in range(runs):

        bandit = BinBandit(k)
        agent.reset()

        optimal = bandit.optimal_action()

        for step in range(steps):

            action = agent.select_action()
            reward = bandit.pull(action)

            agent.update(action, reward)

            rewards[run, step] = reward

            if action == optimal:
                optimal_actions[run, step] = 1

    return rewards.mean(axis=0), optimal_actions.mean(axis=0)


agent4 = ThompsonSampling(k)

avg_r4, opt4 = run_thompson(agent4, runs, steps, k)

plt.figure()

plt.plot(avg_r1, label="ε-greedy (0.1)")
plt.plot(avg_r2, label="ε-greedy (0.01)")
plt.plot(avg_r3, label="UCB")
plt.plot(avg_r4, label="Thompson Sampling")

plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Bandit Algorithm Comparison")
plt.legend()

plt.savefig("avg_reward_all.png", dpi=300)
plt.show()

plt.figure()

plt.plot(opt1, label="ε-greedy (0.1)")
plt.plot(opt2, label="ε-greedy (0.01)")
plt.plot(opt3, label="UCB")
plt.plot(opt4, label="Thompson Sampling")

plt.xlabel("Steps")
plt.ylabel("% Optimal Action")
plt.title("Optimal Action Comparison")
plt.legend()

plt.savefig("optimal_action_all.png", dpi=300)
plt.show()