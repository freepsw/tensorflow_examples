# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
# 1) 데이터 정의
xy = np.loadtxt('04train.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))

# 2) H(X) 함수 정의 - sigmoid
h = tf.matmul(W, X)
hypothesis = tf.div(1., 1. + tf.exp(-h))

# 3) cost 함수 정의
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

# 4) cost 최소화 알고리즘 정
a = tf.Variable(0.1)  # learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)  # goal is minimize cost

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in xrange(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W)

# 5) 학습된 모델로 예측을 해 보자.
#    첫번째 변수는 b(bias)인데, 전부 1로 설정한다.
print '-----------------------------------------'
print sess.run(hypothesis, feed_dict={X: [[1], [2], [2]]}) > 0.5
print sess.run(hypothesis, feed_dict={X: [[1], [5], [5]]}) > 0.5
print sess.run(hypothesis, feed_dict={X: [[1, 1], [4, 0], [2, 10]]}) > 0.5


# 1980 0.339113 [[-6.03336668  0.48038888  1.04906142]]
# 2000 0.338924 [[-6.05253792  0.48121855  1.05242896]]
# -----------------------------------------
# [[False]]
# [[ True]]
# [[False  True]]