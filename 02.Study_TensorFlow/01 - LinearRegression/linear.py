# -*- coding: utf-8 -*-
import tensorflow as tf

x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

# try to find values for w and b that compute y_data = W * x_data + b
# 실제 정답은 w=1, b=0이라는 것을 알 수 있다.
# 과연 학습한 모델이 정답을 찾아내는지 보자
w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# my hypothesis
hypothesis = w * x_data + b

# Simplified cost function = (예측값 - 정답)^2/전체 갯
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# minimize the cost
a = tf.Variable(0.1)  # learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a) # cost를 최소화 하기위한 알고리즘 선택
train = optimizer.minimize(cost)  # 선택한 알고리즘에 cost 함수를 전달하여,
                                  # 각 단계(learning rate)별로 cost가 최소가 될때 까지 학습

# before starting, initialize the variables
init = tf.global_variables_initializer()

# launch
sess = tf.Session()
sess.run(init)

# fit the line
for step in xrange(2001):
    sess.run(train)
    if step % 20 == 0:
        print step, sess.run(cost), sess.run(w), sess.run(b)

    # 아래와 같은 결과가 출력됨
    # 처음 10 ~ 60까지는 cost가 0.26 ~ 0.006까지 줄어들다가,
    # 1960번째 부터는 cost가 0 (예측값과 실제값의 차이가 없다)이 됨을 알수 있음
    # 이때 W = 1, b = 5.66515439e-08(거의 0)에 가까움
    # 0   0.260314[0.75645685][0.95694941]
    # 20  0.0430834[0.75892574][0.54801846]
    # 40  0.0162781[0.85181737][0.33685392]
    # 60  0.00615028[0.90891564][0.20705609]
    # ....
    # 1960 0.0 [ 1.] [  5.66515439e-08]
    # 1980 0.0 [ 1.] [  5.66515439e-08]
    # 2000 0.0 [ 1.] [  5.66515439e-08]

# learns best fit is w: [1] b: [0]
