# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import random
import dqn
from collections import deque
import gym

env = gym.make('CartPole-v0')

# Constant defining our neural network
input_size = env.observation_space.shape[0] # 4개 상태
output_size = env.action_space.n # 2개 action

dis = 0.9
REPLAY_MEMORY = 50000

# 정답인 Q를 계산할 때는 targetDQN의 predict함수를 이용하고,
# 예측값인 Q_pred를 계산하고, weight를 업데이트 할 때는 mainDQN의 update함수를 이용한다.
def replay_train(mainDQN, targetDQN, train_batch):
    x_stack = np.empty(0).reshape(0, input_size)  # 한번에 모아서 학습하기 위한 자료구조
    y_stack = np.empty(0).reshape(0, output_size)

    # Get stored information from the buffer
    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state)

        if done:  # terminal?
            Q[0, action] = reward
        else:
            # get target from targetDQN
            Q[0, action] = reward + dis * np.max(targetDQN.predict(next_state))

        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state])

    # Train our network using target and predicted Q values on each episode
    return mainDQN.update(x_stack, y_stack)

def get_copy_var_ops(dest_scope_name="target", src_scope_name="main"):
    # Copy variables src_scope to dest_scope
    op_holder = []
    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name
    )
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name
    )

    for src_vars, dest_vars in zip(src_vars, dest_vars):
        op_holder.append(dest_vars.assign(src_vars.value()))

    return op_holder

def bot_play(mainDQN):
    # See our trained network in action
    s = env.reset()
    reward_sum = 0
    while True:
        env.render()
        a = np.argmax(mainDQN.predict(s))
        s, reward, done, _ = env.step(a)
        reward_sum += reward
        if done:
            print ("Total score {} ".format(reward_sum))
            break

def main():
    max_episode = 5000
    # store the previous obervations in the replay memory
    replay_buffer = deque()
    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, input_size, output_size, name="main")
        targetDQN = dqn.DQN(sess, input_size, output_size, name="target")
        tf.global_variables_initializer().run()

        # initial copy q_net -> target_net
        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        sess.run(copy_ops)

        for episode in range(max_episode):
            e = 1. /((episode / 10) + 1)
            done = False
            step_count = 0
            state = env.reset()
            while not done:
                if np.random.rand(1) < e:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(mainDQN.predict(state))

                # Get new state and reward from environment
                next_state, reward, done, _ = env.step(action)
                if done:
                    reward = -100 # big penalty

                # Save the experience to our buffer
                replay_buffer.append((state, action, reward, next_state, done))
                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()

                state = next_state
                step_count += 1
                if step_count > 10000: # good enough(막대가 넘어지지 않고 잘 유지되는 횟수)
                    break

            print ("Episode: {} steps {}".format(episode, step_count))
            if step_count > 10000:
                pass

            if episode % 10 ==1: # train every 10 episode
                # Get a random batch of experience
                for _ in range(50):
                    # Minibatch works better
                    minibatch = random.sample(replay_buffer, 10)
                    loss, _ = replay_train(mainDQN, targetDQN,  minibatch)
                print ("Loss : ", loss)

                # Copy q_net -> target_net
                sess.run(copy_ops)

        bot_play(mainDQN)

if __name__ == "__main__":
    main()

