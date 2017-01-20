# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 16:48:54 2017

@author: skiper
"""
#v출처: http://yujuwon.tistory.com/entry/TENSOR-FLOW-MNIST-인식하기 [Ju Factory]

# 1) 학습에 필요한 이미지(28 * 28)를 다운로드 한다.
# input_data.py 파일이 다른 경로에 있는 경우,
# 아래와 같은 방식으로 호출 가능
#import sys
#sys.path.append("/root/work/")
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf

# 2) tensorflow 실행에 필요한 변수 설정
#    여기서는 실제 값이 입력되지 않고, 입력할 수 있는 공간(변수)만 정의
#    x : 28 * 28의 matrix를 이미지 입력값을 벡터로 저장할 변수 x
#    W : 28 * 28(784)의 입력값을 숫자 10개중 1개로 표현할 수 있도록 10개의 행렬로 정의
#    b : 각 W에 해당하는 bias값
x = tf.placeholder(tf.float32, [None, 784]) # None는 이미지가 몇개이든 제약을 두지 않겠다는 의미.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 3) 각 입력값을 기반으로 예측값(y)를 구하는 함수 정의
#    y = Wx + b
#    이 함수를 실행하여 예측한 값과 실제값의 차이가 최소화 되도록 학습할 예정
#    여기서 softmax함수를 사용하는 이유는 y의 결과는 10개의 배열로 생성되는데,
#    이 중 가장 확률이 높은 1개의 값(index)을 추출하기 위함
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 4) 모델 학습
#    y_ : tensorflow의 실행결과를 저장할 변수
#    cross_entropy : cost를 최소화 하기 위한 함수 (http://pythonkim.tistory.com/20)
#     예측한 값과 실제 값의 차이를 최소화
#     * y  : softmax를 통해 예측한 값 (정확히는 확률) (0.1, 0.8, .....0.0)
#     * y_ : 실제 결과값 (0,1,....0) 10 배열로 구성된 값
#            (sess.run 실행시 매 batch마다 feed_dict로 입력해 줌, y_: batch_ys)
#     *
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(correct_prediction, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels}))

#print mnist.train.labels[1]
#print mnist.train.images[1] 

# 여기서 lbesl[1]은 숫자 3을 나타냄 (1이 표시된 위치가 4번째 이므로 0,1,2,3)
# label[1] = [ 0.  0.  0.  1.  0.  0.  0.  0.  0.  0.]
# images[1] = 
# [ 0.          0.          0.          0.          0.          0.          0.
#  0.          0.          0.          0.          0.          0.          0.
#  0.59215689  0.59215689  0.99215692  0.67058825  0.59215689  0.59215689
#  0.15686275  0.          0.          0.          0.          0.          0.
# 0.          0.          0.          0.          0.          0.          0.
#  0.          0.          0.          0.          0.          0.          0.
#  0.          0.          0.          0.          0.          0.          0.
#  0.        ]

# 2) image로 표현하기 위해서는 원래 2차원 행렬로 변경한다.
#    변경된 행렬을 matplot으로 렌더링
#import tensorflow as tf
#import numpy as np
#arr = np.array(mnist.train.images[1])
#arr.shape = (28,28)


#import matplotlib.pyplot as plt
#plt.imshow(arr)
#plt.show()


# 3) model 학습 및 검증


