# -*- coding: utf-8 -*-
import tensorflow as tf

# 1) 이번에는 2차원 행렬(매트릭스)로 데이터 type을 변경하자.
# (실제 데이터는 거의 이런 구조로 입력됨)
x_data = [[1., 0., 3., 0., 5.],
          [0., 2., 0., 4., 0.]]
y_data  = [1, 2, 3, 4, 5]

# 2) x1_data를 저장할 tensorflow 변수 구조도 함께 변경
#    [1,2] 매트릭스 구조로 변경
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# 3) matrix 곱셈을 위하여 tf.matmul() 함수 사용
hypothesis = tf.matmul(W, x_data) + b

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
        print step, sess.run(cost), sess.run(W), sess.run(b)

# 이번엔 출력된 W가 배열로 출력되었다. (입력된 X가 2개였기 때문에 배열로 출력됨)
# 1980 4.83169e-14 [[ 0.99999994  0.99999994]] [  1.61872777e-07]
# 2000 4.83169e-14 [[ 0.99999994  0.99999994]] [  1.61872777e-07]