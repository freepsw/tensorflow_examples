# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import input_data

# 1) Training Set을  /tmp/data에 저장한다.

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# 2) 학습용으로 100개, 학습한 결과를 테스트할 용도로 10개씩 추출한다.
train_pixels,train_list_values = mnist.train.next_batch(100) 
test_pixels,test_list_of_values  = mnist.test.next_batch(10) 


# 3) tensorflow에서 사용할 데이터 타입으로 정의
train_pixel_tensor = tf.placeholder("float", [None, 784])
test_pixel_tensor  = tf.placeholder("float", [784])

# 4) Cost Function and distance optimization (pixel간 거리를 기반으로 비용계산)
distance = tf.reduce_sum\
           (tf.abs\
            (tf.add(train_pixel_tensor, \
                    tf.neg(test_pixel_tensor))), \
            reduction_indices=1)

# 5) 가장 작은 거리를 가지는 학습모델을 생성 (pred)
pred = tf.arg_min(distance, 0)

# 6) Testing and algorithm evaluation
accuracy = 0.
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for i in range(len(test_list_of_values)):
        # 6-1) 학습한 모델(pred)를 이용하여 테스트 데이터에 대한 index를 예측한다(nn_index)
        #      이때 사용할 데이터를 feed_dict로 입력해 준다.
        #      즉, 학습은 train_pixel_tensor로 진행하고,
        #      검증은 test_pixel_tensor로 진행하기 위해 필요한 데이터를 지정해 주는 것이다.
        nn_index = sess.run(pred,\
		                    feed_dict={train_pixel_tensor:train_pixels, test_pixel_tensor:test_pixels[i,:]})

        print "Test N� ", i,"Predicted Class: ", \
		        np.argmax(train_list_values[nn_index]),\
		        "True Class: ", np.argmax(test_list_of_values[i])

        if np.argmax(train_list_values[nn_index]) == np.argmax(test_list_of_values[i]):
            accuracy += 1./len(test_pixels)

    print "Result = ", accuracy
