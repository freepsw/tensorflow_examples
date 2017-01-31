# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
# 1) 관련 변수 설정
#    - char_rdic: 학습할 char에 대한 고유 index를 생성할 용도로 변수 할당
#    - char_dic : 각 char에 인덱스를 할당 (h=0, e=1, l=2, o=3)
char_rdic = list('helo')  # id(string) -> char (['h', 'e', 'l', 'o'])
char_dic = {w: i for i, w in enumerate(char_rdic)}  # ({'h': 0, 'e': 1, 'l': 2, 'o': 3})
x_data = np.array([
    [1, 0, 0, 0],  # h
    [0, 1, 0, 0],  # e
    [0, 0, 1, 0],  # l
    [0, 0, 1, 0],  # l
], dtype='f')

char_vocab_size = len(char_dic) # 4
rnn_size = char_vocab_size  # 1 hot coding (one of 4)
time_step_size = 4  # 'hell' -> predict 'ello'
batch_size = 1  # one sample

# 2) RNN model 생성
#    - rnn_cell : 최종 출력값이 rnn_size(4개)인 RNN모델을 생성함 (h, e, l, o)의 index를 예측하는 배열 크
#    - state    : 초기값 [[ 0.  0.  0.  0.]]의 배열을 생성 shape(1,4) --> 상태값을 저장.
#    - X_split  : 결국 x_data와 동일한 배열이다. (tensor객체로 변환하는 용도)
#    - output, state : input * state를 계산한 결과(output)과 변경된 상태(state)
rnn_cell = tf.nn.rnn_cell.BasicRNNCell(rnn_size)
state = tf.zeros([batch_size, rnn_cell.state_size])  # rnn_cell.state_size = 4
X_split = tf.split(0, time_step_size, x_data)
outputs, state = tf.nn.rnn(rnn_cell, X_split, state)


# 3) cost
# logits: 예측값, 2차원 배열(,4), (4,4) -> (1, 16) -> (4,4) Tensor객체
#         list of 2D Tensors of shape [batch_size x num_decoder_symbols]
# targets: 에측할 실제 값("ello"), list of 1D batch-sized int32 Tensors of the same length as logits. "
# weights: 비율 (보통 1), list of 1D batch-sized float-Tensors of the same length as logits.
logits = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])
sample = [char_dic[c] for c in "hello"]  # to index ([0, 1, 2, 2, 3])
targets = tf.reshape(sample[1:], [-1])
weights = tf.ones([time_step_size * batch_size]) # [ 1.  1.  1.  1.]

loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [weights])
cost = tf.reduce_sum(loss) / batch_size
train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)

# 4) 실행
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        sess.run(train_op)
        result = sess.run(tf.arg_max(logits, 1))
        print("%r, %r" % (result, [char_rdic[t] for t in result]))
        # print(sess.run(logits))
        # print(sess.run(state))
        print(sess.run(cost))

    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("/Users/skiper/work/00.Source/python/env_tensorflow/tensor_log", sess.graph)

# 결과 출력
# array([2, 3, 3, 3]), ['l', 'o', 'o', 'o']
# array([2, 3, 3, 3]), ['l', 'o', 'o', 'o']
# array([2, 3, 3, 3]), ['l', 'o', 'o', 'o']
# array([2, 3, 3, 3]), ['l', 'o', 'o', 'o']
# ...
# array([1, 2, 2, 3]), ['e', 'l', 'l', 'o']
# array([1, 2, 2, 3]), ['e', 'l', 'l', 'o']
# array([1, 2, 2, 3]), ['e', 'l', 'l', 'o']
# array([1, 2, 2, 3]), ['e', 'l', 'l', 'o']