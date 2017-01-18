# -*- coding: utf-8 -*-
import tensorflow as tf

x1_data = [1, 0, 3, 0, 5]
x2_data = [0, 2, 0, 4, 0]
y_data  = [1, 2, 3, 4, 5]

W1 = tf.Variable(tf.random_uniform([1], -1, 1))
W2 = tf.Variable(tf.random_uniform([1], -1, 1))

b = tf.Variable(tf.random_uniform([1], -1, 1))

# 정답을 계산해 보면
# W1 =1, W2=1, b=0
hypothesis = W1 * x1_data + W2 * x2_data + b

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.1)  # learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)  # goal is minimize cost

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in xrange(2001):
    sess.run(train)
    if step % 20 == 0:
        print step, sess.run(cost), sess.run(W1), sess.run(W2), sess.run(b)

# 0 5.70835 [ 1.77412081] [ 0.86150169] [ 0.48825324]
# 20 0.00407316 [ 0.96022928] [ 0.95281124] [ 0.15129729]
# 40 0.00117935 [ 0.97859919] [ 0.97460812] [ 0.08141152]
# 60 0.000341469 [ 0.98848444] [ 0.98633689] [ 0.04380676]
#
# 1980 1.42109e-14 [ 1.] [ 0.99999994] [  1.74312973e-07]
# 2000 1.42109e-14 [ 1.] [ 0.99999994] [  1.74312973e-07]