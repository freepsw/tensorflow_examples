#using_tensorboard.py

import tensorflow as tf

a = tf.constant(10,name="a")
b = tf.constant(90,name="b")
y = tf.Variable(a+b*2,name='y')
model = tf.initialize_all_variables()

with tf.Session() as session:
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("/Users/skiper/work/00.Source/python/env_tensorflow/tensor_log",session.graph)
    session.run(model)
    print(session.run(y))



