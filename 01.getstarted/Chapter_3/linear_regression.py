# -*- coding: utf-8 -*-
import numpy as np

# 1) Test 데이터 생성 (x축으로 넓게-0.5 퍼지고, y축으로 좁게-0.1 퍼지는
#    y = a*x + b 라는 수식을 기반으로 데이터 생성함 --> 아래 학습을 통해서 이 값을 잘 예측하는지 확인해 본다 (A, B)
number_of_points = 200
x_point = []
y_point = []
a = 0.22
b = 0.78
for i in range(number_of_points):
    x = np.random.normal(0.0,0.5)
    y = a*x + b +np.random.normal(0.0,0.1)
    x_point.append([x])
    y_point.append([y])


import matplotlib.pyplot as plt

plt.plot(x_point,y_point, 'o', label='Input Data')
plt.legend()
plt.show()

# 2) Y = Ax + B 라는 선형회귀 함수에 필요한 초기값 설정
#   - A : 임의 값 할당
#   - B : 0으로 설정
#   - y : 선형 회기함수 정의
import tensorflow as tf

A = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
B = tf.Variable(tf.zeros([1]))
y = A * x_point + B

# 3) cost function과 이 cost function의 결과인 cost를 최소화하는 알고리즘 지정
# cost_function : 예측값(y_point)과 실제값(y)의 차이(절대값)의 평균 제곱근(mean square error)
cost_function = tf.reduce_mean(tf.square(y - y_point)) # 평균 제곱근 오차를 cost 함수로 사용
optimizer = tf.train.GradientDescentOptimizer(0.5) # learning rate = 0.5
train = optimizer.minimize(cost_function) # 위의 오차를 최소화하는 알고리즘을 gradient descent로 사용

model = tf.initialize_all_variables()

with tf.Session() as session:
        session.run(model)
        for step in range(0,21):
                session.run(train)
                if (step % 5) == 0:
                        plt.plot(x_point, y_point,
                                 'o',label='step = {}'.format(step))
                        plt.plot(x_point,
                                 session.run(A) * x_point + session.run(B))
                        plt.legend()
                        plt.show()
