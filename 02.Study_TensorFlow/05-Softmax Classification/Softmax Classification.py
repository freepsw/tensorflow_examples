# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

xy = np.loadtxt('05train.txt', unpack=True, dtype='float32')

x_data = np.transpose(xy[0:3])
y_data = np.transpose(xy[3:])

# 1) tf 설정
X = tf.placeholder("float", [None, 3]) # x1, x2 and x0(1, bias 데이터)
Y = tf.placeholder("float", [None, 3]) # [A, B, C] 3개의 클래스

W = tf.Variable(tf.zeros([3, 3])) # 3 : x 변수가 3개 (x0, x1, x2),
                                  # 3: 예측할 값이 3개(A, B, C)

# 2) H(x) 함수 정의
hypothesis = tf.nn.softmax(tf.matmul(X, W))

# 3) cost 함수 정의 (cross-entropy 함수)
#     * hypothesis  : softmax를 통해 예측한 값 (정확히는 확률) [0.1, 0.8, 0.0]
#     * Y           : 실제 결과값 [0, 1, 0] => B
#                     (sess.run 실행시 매 batch마다 feed_dict로 입력해 줌, Y: y_data)
learning_rate = 0.01
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), reduction_indices=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in xrange(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W)

# 4) 학습된 모델 테스트
    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7]]})
    print "a :", a, sess.run(tf.arg_max(a, 1))
    # a : [[ 0.68849677  0.26731515  0.04418808]] [0]  ==> argmax 결과(0.688의 index인 0을 출력)
    b = sess.run(hypothesis, feed_dict={X: [[1, 3, 4]]})
    print "b :", b, sess.run(tf.arg_max(b, 1))
    # b : [[ 0.24322268  0.44183081  0.3149465 ]] [1]
    c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0]]})
    print "c :", c, sess.run(tf.arg_max(c, 1))
    # c : [[ 0.02974809  0.08208466  0.8881672 ]] [2]

    all = sess.run(hypothesis, feed_dict={X: [[1, 3, 4],[1, 3, 4], [1, 1, 0]]})
    print all, sess.run(tf.arg_max(all, 1))
    # [[0.24322268  0.44183081  0.3149465]
    #  [0.24322268  0.44183081  0.3149465]
    #  [0.02974809  0.08208466  0.8881672]]
    # [1 1 2]

