# -*- coding: utf-8 -*-
import tensorflow as tf

# 1) 이번에는 2차원 행렬(매트릭스)에 b를 포함하여 구성해 보자
#     b = [1,  1,  1,  1,  1]
x_data = [[1,  1,  1,  1,  1], # b를 입력
          [0., 2., 0., 4., 0.],
          [1., 0., 3., 0., 5.]]
y_data  = [1, 2, 3, 4, 5]

# 2) x1_data를 저장할 tensorflow 변수 구조도 함께 변경
#    [1,3] 매트릭스 구조로 변경 (b도 포함된 매트릭스)
W = tf.Variable(tf.random_uniform([1, 3], -1.0, 1.0))

# 3) matrix 곱셈을 위하여 tf.matmul() 함수 사용 (b를 제거)
hypothesis = tf.matmul(W, x_data)

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
        print step, sess.run(cost), sess.run(W)

# 이번엔 출력된 W가 배열로 출력되었다. ([b, x1, x2]로 b가 가장 먼저 보인다.)
# 1980 4.83169e-14 [[  1.61933144e-07   9.99999940e-01   9.99999940e-01]]
# 2000 4.83169e-14 [[  1.61933144e-07   9.99999940e-01   9.99999940e-01]]