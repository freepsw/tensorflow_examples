# -*- coding: utf-8 -*-
# Recurrent Neural Network
import numpy as np
import tensorflow as tf
import functions
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
],
    dtype='f')

# 학습 데이터를 index로 변환
sample = [char_dic[c] for c in "hello"]  # to index ([0, 1, 2, 2, 3])

# Configuration
char_vocab_size = len(char_dic) # 4
rnn_size = char_vocab_size  # 1 hot coding (one of 4)
time_step_size = 4  # 'hell' -> predict 'ello'
batch_size = 1  # one sample

# 2) RNN model 생성
#    - rnn_cell : 최종 출력값이 rnn_size(4개)인 RNN모델을 생성함 (h, e, l, o)의 index를 예측하는 배열 크
#    - state    : 초기값 [[ 0.  0.  0.  0.]]의 배열을 생성 shape(1,4) --> 상태값을 저장.
#    - X_split  : 결국 x_data와 동일한 배열이다. (tensor객체로 변환하는 용도)
#    - rnn cell의 크기 : 입력 벡터의 크기(여기서는 h, e, l, l 4개)에 따라서 결정
#    - output, state : input * state를 계산한 결과(output)과 변경된 상태(state)
rnn_cell = tf.nn.rnn_cell.BasicRNNCell(rnn_size)
state = tf.zeros([batch_size, rnn_cell.state_size])  # rnn_cell.state_size = 4
X_split = tf.split(0, time_step_size, x_data)
outputs, state = tf.nn.rnn(rnn_cell, X_split, state)

# outputs 출력 결과
# [
#  array([[ 0.37687305, -0.29618981,  0.16475253, -0.30309165]], dtype=float32),
#  array([[-0.13646206,  0.42270219,  0.17206028,  0.00248625]], dtype=float32),
#  array([[-0.055261  ,  0.15680611, -0.53751069,  0.5630942 ]], dtype=float32),
#  array([[-0.54226017, -0.24723642,  0.0682147 ,  0.68651956]], dtype=float32)
# ],
# state 출력 결과
#  array([[-0.54226017, -0.24723642,  0.0682147 ,  0.68651956]], dtype=float32)
# )

#functions.showOperation(tf.nn.rnn(rnn_cell, X_split, state))

# 3) cost
# logits: 예측값, 2차원 배열(,4)
#         list of 2D Tensors of shape [batch_size x num_decoder_symbols]
# targets: 에측할 실제 값("ello"), list of 1D batch-sized int32 Tensors of the same length as logits. "
# weights: 비율 (보통 1), list of 1D batch-sized float-Tensors of the same length as logits.

logits = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])
print logits
#logits = outpus
targets = tf.reshape(sample[1:], [-1])
weights = tf.ones([time_step_size * batch_size]) # [ 1.  1.  1.  1.]

# 3-1) logit에서 사용되는 2차원 배열을 생성하기 위해 필요한 변수
#      outputs는 [4, 4] shape구조로 저장된 tensor 객체이다 (RNN 결과 값)
print outputs # list
# [<tf.Tensor 'RNN/BasicRNNCell/Tanh:0' shape=(1, 4) dtype=float32>, [ 0.45873451  0.13431551  0.2665318  -0.11387707]
# <tf.Tensor 'RNN/BasicRNNCell_1/Tanh:0' shape=(1, 4) dtype=float32>,[0.21802776  -0.3968451  -0.12312219  0.39910132
# <tf.Tensor 'RNN/BasicRNNCell_2/Tanh:0' shape=(1, 4) dtype=float32>,
# <tf.Tensor 'RNN/BasicRNNCell_3/Tanh:0' shape=(1, 4) dtype=float32>]

# 3-2) output객체를 1차원 배열로 변경해 준다(concat) --> 차원만 변경함. shape(1,16) 4*4 = 16
#      reshape를 실행하기 위한 용도
# print "concat"
# functions.showOperation(tf.concat(1, outputs))
#
# print "concat1"
# functions.showOperation(tf.concat(1, outputs))
# [[ 0.45873451  0.13431551  0.2665318  -0.11387707  0.21802776 -0.3968451
#   -0.12312219  0.39910132 -0.55776322 -0.08702181  0.23161684 -0.57659125
#    0.24692856  0.79166049  0.7797873  -0.43172249]]

# 3-3) 1차원 outputs를 4개의 열을 가지는 배열로 변환한다 (rnn_size=4)
#      [1, 16] ==> 4개의 열을 가지도록 변환하면 [4,4]가 생성된다. [-1, 4]
#      여기서 -1은 4개의 열을 지정한 경우 필요한 행의 갯수를 자동으로 계산해 준다.
#      [column, row] = [행, 열]
print "reshape"
functions.showOperation(tf.reshape(tf.concat(1, outputs), [-1, rnn_size]))

# 3-4) target을 logit과 동일한 크기의 1차원 배열로 생성해 준다
#      reshape로 5개로 구성된 배열을 4개로 조정해 준다.
#      hello --> ello로 변경
print sample
functions.showOperation(tf.reshape(sample[1:], [-1])) # sample[1:] =[1, 2, 2, 3] => ello

loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [weights])
cost = tf.reduce_sum(loss) / batch_size
train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)

# 4) 실행
# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        sess.run(train_op)
        result = sess.run(tf.arg_max(logits, 1))
        print("%r, %r" % (result, [char_rdic[t] for t in result]))

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