# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 16:48:54 2017

@author: skiper
"""
#v출처: http://yujuwon.tistory.com/entry/TENSOR-FLOW-MNIST-인식하기 [Ju Factory]

# input_data.py 파일이 다른 경로에 있는 경우,
# 아래와 같은 방식으로 호출 가능
#import sys
#sys.path.append("/root/work/")
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print mnist.train.labels[1]
print mnist.train.images[1] 
# 여기서 lbesl[1]은 숫자 3을 나타냄 (1이 표시된 위치가 4번쨰이므로 0,1,2,3)
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
#    변경된 행렬을 matplot으로 렌더
import tensorflow as tf
import numpy as np

arr = np.array(mnist.train.images[1])
arr.shape = (28,28)


import matplotlib.pyplot as plt

plt.imshow(arr)
plt.show()
