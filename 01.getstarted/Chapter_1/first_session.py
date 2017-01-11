#first_session.py

#a simple Python code
x = 1
y = x + 9
print(y)

#....and the tensorflow translation of the previous code
import tensorflow as tf

x = tf.constant(1, name='x')
y = tf.Variable(x+9,name='y')
# 아래는 10이 출력되지 않는다. 단순히 y에 대한 계산식이 정의되었을 뿐이다
print(y)
