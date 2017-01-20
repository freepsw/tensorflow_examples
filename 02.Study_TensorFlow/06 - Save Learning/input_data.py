# -*- coding: utf-8 -*-

# 이 파일은 https://github.com/tensorflow/tensorflow/blob/r0.9/tensorflow/examples/tutorials/mnist/input_data.py에서 다운로드
#"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# pycharm으로 보면 아래의 urllib, xrange가 error처럼 보일 수 있는데
# 실제 실행해 보면 정상동작한다.
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets