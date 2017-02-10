import gym
from gym.envs.registration import register
import numpy as np
import random as pr
import matplotlib.pyplot as plt

register(
    id = 'FrozenLake-v3',
    entry_point = 'gym.envs.toy_text:FrozenLakeEnv',
    kwargs = {'map_name' : '4x4', 'is_slippery' : False}
)
env = gym.make('FrozenLake-v3')
# env.render()

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])
dis = .99 # discount factor

num_episodes = 2000

rList = [] # create lists to contain total rewards and steps per episode
for i in range(num_episodes):
    state = env.reset() # reset environment and get first new observation
    rAll = 0
    done = False

    e = 1. / ((i // 100) + 1)

    # The Q-Table learning algorithm
    while not done:
        # Choose an action by e-greedy
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        # Get new state and reward from environment
        new_state, reward, done, _ = env.step(action)

        # Update new state and reward from environment
        Q[state, action] = reward + dis * np.max(Q[new_state, :])

        rAll += reward
        state = new_state

    rList.append(rAll)

print("E-Greedy Success rate : ", str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)
plt.bar(range(len(rList)), rList, color = "blue")
# plt.show()