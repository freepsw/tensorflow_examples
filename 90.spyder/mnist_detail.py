# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 16:48:54 2017

@author: skiper
"""
#v출처: http://yujuwon.tistory.com/entry/TENSOR-FLOW-MNIST-인식하기 [Ju Factory]


import input_data
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(2):
    result_boolean = []
    batch_x, batch_y = mnist.test.next_batch(9)
    diff_a = sess.run(tf.argmax(y,1), feed_dict={x:batch_x})
    diff_b = sess.run(tf.argmax(y_,1), feed_dict={y_:batch_y})
    print "sample output : " + str(diff_a)

    for k in range(9):
        if diff_a[k] == diff_b[k]:
            result_boolean.append("T")
        else:
            result_boolean.append("F")
    print "compare : " + str(result_boolean)

    plt.figure(i)
    coordi = [191, 192, 193, 194, 195, 196, 197, 198, 199]

    for index, image in enumerate(batch_x):
        arr = np.array(image)
        arr.shape = (28,28)
        plt.subplot(coordi[index])
        plt.imshow(arr)
        # plt.show()
print "sample input : "


