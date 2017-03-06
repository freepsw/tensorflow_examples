import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make("CartPole-v0")
env.reset()
random_episodes = 0
reward_sum = 0
while random_episodes < 10:
    env.render()
    action = env.action_space.sample()
    obervation, reward, done, _ = env.step(action)
    print (obervation, reward, done)
    reward_sum += reward

    if done:
        random_episodes += 1
        print("Reward for this episode was : ", reward_sum)
        reward_sum = 0
        env.reset()