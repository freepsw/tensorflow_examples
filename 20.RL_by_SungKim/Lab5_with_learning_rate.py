import gym
from gym.envs.registration import register
import numpy as np
import random as pr

import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')
# env.render()

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])
learning_rate = .85
dis = .99 # discount factor
num_episodes = 2000

rList = [] # create lists to contain total rewards and steps per episode
for i in range(num_episodes):
    state = env.reset() # reset environment and get first new observation
    rAll = 0
    done = False

    # The Q-Table learning algorithm
    while not done:
        # Choose an action by greedily(with noise picking from Q table
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n)/(i+1))

        # Get new state and reward from environment
        new_state, reward, done, _ = env.step(action)

        # Update new state and reward from environment
        Q[state, action] = (1-learning_rate) * Q[state, action] \
            + learning_rate * (reward + dis * np.max(Q[new_state, :]))

        rAll += reward
        state = new_state

    rList.append(rAll)

print("Lab5 Success rate : ", str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)
plt.bar(range(len(rList)), rList, color = "blue")
# plt.show()