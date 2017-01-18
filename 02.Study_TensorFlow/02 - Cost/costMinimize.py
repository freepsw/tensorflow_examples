# -*- coding: utf-8 -*-
import tensorflow as tf

# data set
x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

# try to find values for w and b that compute y_data = W * x_data
# 위 데이터를 보면 정답이 W = 1 이라는 것을 알 수 있다.
# range is -100 ~ 100
W = tf.Variable(tf.random_uniform([1], -10.0, 10.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# my hypothesis
hypothesis = W * X

# Simplified cost functionn
# (예측값 - 실제값의 제곱의 합) /
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# minimize
descent = W - tf.mul(0.1, tf.reduce_mean(tf.mul((tf.mul(W, X) - Y), X)))
update = W.assign(descent) # cost가 감소하는 방향으로 이동한 값을 W에 업데이트 해 준다.

# before starting, initialize the variables
init = tf.global_variables_initializer()

# launch
sess = tf.Session()
sess.run(init)

# fit the line
for step in xrange(20):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W)

print sess.run(hypothesis, feed_dict={X: 5})
print sess.run(hypothesis, feed_dict={X: 2.5})


# 0 0.457145 [ 0.68701482]
# 1 0.130032 [ 0.83307457]
# 2 0.036987 [ 0.91097307]
# 3 0.0105207 [ 0.952519]
# 4 0.00299257 [ 0.97467679]
# 5 0.000851218 [ 0.9864943]
# 6 0.000242124 [ 0.99279696]
# 7 6.88715e-05 [ 0.99615836]
# 8 1.95892e-05 [ 0.99795115]
# 9 5.57242e-06 [ 0.99890727]
# 10 1.58528e-06 [ 0.99941719]
# 11 4.50858e-07 [ 0.99968916]
# 12 1.28244e-07 [ 0.99983424]
# 13 3.64522e-08 [ 0.99991161]
# 14 1.03678e-08 [ 0.99995285]
# 15 2.94652e-09 [ 0.99997485]
# 16 8.40928e-10 [ 0.99998659]
# 17 2.38742e-10 [ 0.99999285]
# 18 6.79089e-11 [ 0.99999619]
# 19 1.96536e-11 [ 0.99999797]
# [ 4.99998999]
# [ 2.49999499]