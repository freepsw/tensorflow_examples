import gym
from gym.envs.registration import register
import numpy as np
import random as pr
import matplotlib.pyplot as plt

def rargmax(vector):
    """ Argmax that chooses randomly among eligible maximum indices. """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)

# if __name__ == '__main__':
#     test = [0,1,2,2]
#     for i in range(10):
#         print rargmax(test)

# register frozenlake with is_slippery False
register(
    id = 'FrozenLake-v3',
    entry_point = 'gym.envs.toy_text:FrozenLakeEnv',
    kwargs = {'map_name' : '4x4', 'is_slippery' : False}
)
env = gym.make('FrozenLake-v3')
# env.render()


Q = np.zeros([env.observation_space.n, env.action_space.n])
num_episodes = 2000

rList = []
for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False

    while not done:
        action = rargmax(Q[state, :])
        # print "action : ", action
        new_state, reward, done, _ = env.step(action)
        # print "State : newState ", state, " : ", new_state
        # print "Reward : ", reward
        print "Q[%d, %d] = %d + %d(Q[%d, :])" % (state, action, reward, np.max(Q[new_state, :]), new_state)

        Q[state, action] = reward + np.max(Q[new_state, :])

        rAll += reward
        state = new_state

    rList.append(rAll)
    print i, "st Q "
    print Q
    print np.reshape(Q, (4, 4, 4) )

print("Success rate : ", str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)
plt.bar(range(len(rList)), rList, color = 'red')
#plt.show()