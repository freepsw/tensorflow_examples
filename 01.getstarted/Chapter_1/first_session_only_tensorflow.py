#first_session_only_tensorflow.py

import tensorflow as tf

x = tf.constant(1, name='x')
y = tf.Variable(x+9,name='y')

# 모든 변수를 초기화 한 후에 model을 실행하면 실제 결과값이 출력됨
model = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(model)
    print(session.run(y))
