# -*- coding: utf-8 -*-
# [Lab 07]  소스코드
import tensorflow as tf
import random
from tensorflow.examples.tutorials.mnist import input_data

# 1) 데이터 및 변수 설정
learning_rate = 0.01
training_epochs = 15
batch_size = 100
display_step = 1
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# tf Graph Input
x = tf.placeholder("float", [None, 784])  # mnist data image of shape 28*28=784 (흑백이므로 color를 위한 차원은 없음)
y = tf.placeholder("float", [None, 10])  # 0-9 digits recognition => 10 classes

# set model weight
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

activation = tf.nn.softmax(tf.matmul(x, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(activation), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# 4) 학습 실행 (Launch the graph)
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        avg_cost = 0.
        # mnist.train.num_examples = 55000
        # batch_size = 100
        # total_batch = 550 ==> 1번의 training_epoch에 550개의 이미지를 학습한다
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}) / total_batch

        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
    print ("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print ("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


    # 5) Test 데이터에서 임의로 1개의 이미지를 선택하여, 정답을 예측하는지 확인해보자.
    import matplotlib.pyplot as plt
    r = random.randint(0, mnist.test.num_examples -1) # 랜덤하게 1개 선택
    # 5-1) label에는 10개의 값을 가진 vector type이 저장되어 있음 [0, 1, 2, ...., 9] 각각의 확률값이 저장
    #      실제 값은 예를들면 [0.1, 0.5, 0.1 .... 0.0] ==> "1"의 확률값이 가장 높음 (index=1)
    print "Label : ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)) # 실제 값을 출력 (정답)
    print "Predt : ", sess.run(tf.argmax(activation, 1), {x:mnist.test.images[r:r+1]}) # 모델이 예측한 값을 출력
    plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap="Greys", interpolation='nearest')
    plt.show()

# 출력 결과
# total batch :  550
# ('Epoch:', '0005', 'cost=', '0.000017785')
# [-0.04423999 -0.58194578  0.55721319 -0.09853368 -0.61277747  0.0913931
#  -0.98012233  0.00404656  0.80806178  0.12157804]
# Optimization Finished!
# ('Accuracy:', 0.96939999)

# Label :  [3] <- 정답
# Predt :  [3] <- 예측한 값
