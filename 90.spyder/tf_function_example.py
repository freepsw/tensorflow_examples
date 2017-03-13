
# -*- coding: utf-8 -*-

# In[25]:

# tensorflow util function test
import functools, operator

def getLength(t):
    temp = (dim.value for dim in t.get_shape())         # dim is Dimension class.
    return functools.reduce(operator.mul, temp)

def showConstant(t):
    sess = tf.InteractiveSession()
    print(t.eval())
    sess.close()

def showConstantDetail(t):
    sess = tf.InteractiveSession()
    print(t.eval())
    print('shape :', tf.shape(t))
    print('size  :', tf.size(t))
    print('rank  :', tf.rank(t))
    print(t.get_shape())

    sess.close()

def showVariable(v):
    sess = tf.InteractiveSession()
    v.initializer.run()
    print(v.eval())
    sess.close()

def var2Numpy(v):
    sess = tf.InteractiveSession()
    v.initializer.run()
    n = v.eval()
    sess.close()

    return n

def op2Numpy(op):
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)
    ret = sess.run(op)
    sess.close()

    return ret

def showOperation(op):
    print(op2Numpy(op))


c1 = tf.constant([1, 3, 5, 7, 9, 0, 2, 4, 6, 8])
c2 = tf.constant([1, 3, 5])
v1 = tf.constant([[1, 2, 3, 4, 5, 6], [7, 8, 9, 0, 1, 2]])
v2 = tf.constant([[1, 2, 3], [7, 8, 9]])
x_data = np.array([
    [1, 0, 0, 0],  # h
    [0, 1, 0, 0],  # e
    [0, 0, 1, 0],  # l
    [0, 0, 1, 0],  # l
],
    dtype='f')
c1 = tf.constant([1, 3, 5, 7, 9, 0, 2, 4, 6, 8])
c2 = tf.constant([1, 3, 5])
v1 = tf.constant([[1, 2, 3, 4, 5, 6], [7, 8, 9, 0, 1, 2]])
v2 = tf.constant([[1, 2, 3], [7, 8, 9]])


print('-----------slice------------')
showOperation(tf.slice(c1, [2], [3]))             # [5 7 9]
showOperation(tf.slice(v1, [0, 2], [1, 2]))       # [[3 4]]
showOperation(tf.slice(v1, [0, 2], [2, 2]))       # [[3 4] [9 0]]
showOperation(tf.slice(v1, [0, 2], [2,-1]))       # [[3 4 5 6] [9 0 1 2]]

print('-----------split------------')
showOperation(tf.split(0, 2, c1)) # [[1, 3, 5, 7, 9], [0, 2, 4, 6, 8]]
showOperation(tf.split(0, 5, c1)) # [[1, 3], [5, 7], [9, 0], [2, 4], [6, 8]]
showOperation(tf.split(0, 2, v1)) # [[[1, 2, 3, 4, 5, 6]], [[7, 8, 9, 0, 1, 2]]]
showOperation(tf.split(1, 2, v1)) # [[[1, 2, 3], [7, 8, 9]], [[4, 5, 6], [0, 1, 2]]]

print('-----------tile------------')
showOperation(tf.tile(c2, [3]))   # [1 3 5 1 3 5 1 3 5]
# [[1 2 3 1 2 3] [7 8 9 7 8 9] [1 2 3 1 2 3] [7 8 9 7 8 9]]
showOperation(tf.tile(v2, [2, 2]))

print('-----------pad------------')         # 2차원에 대해서만 동작
# [[0 0 0 0 0 0 0]
#  [0 0 1 2 3 0 0]
#  [0 0 7 8 9 0 0]
#  [0 0 0 0 0 0 0]]
showOperation(tf.pad(v2, [[1, 1], [2, 2]], 'CONSTANT'))
# [[9 8 7 8 9 8 7]
#  [3 2 1 2 3 2 1]
#  [9 8 7 8 9 8 7]
#  [3 2 1 2 3 2 1]]     # 3 2 1 2 3 2 1 2 3 2 1 처럼 반복
showOperation(tf.pad(v2, [[1, 1], [2, 2]], 'REFLECT'))
# [[2 1 1 2 3 3 2]
#  [2 1 1 2 3 3 2]
#  [8 7 7 8 9 9 8]
#  [8 7 7 8 9 9 8]]     # 3 2 1 (1 2 3) 3 2 1. 가운데와 대칭
showOperation(tf.pad(v2, [[1, 1], [2, 2]], 'SYMMETRIC'))

print('-----------concat------------')
showOperation(tf.concat(0, [c1, c2]))     # [1 3 5 7 9 0 2 4 6 8 1 3 5]
showOperation(tf.concat(1, [v1, v2]))     # [[1 2 3 4 5 6 1 2 3] [7 8 9 0 1 2 7 8 9]]
# showOperation(tf.concat(0, [v1, v2]))   # error. different column size.

c3, c4 = tf.constant([1, 3, 5]), tf.constant([[1, 3, 5], [5, 7, 9]])
v3, v4 = tf.constant([2, 4, 6]), tf.constant([[2, 4, 6], [6, 8, 0]])

print('-----------pack------------')           # 차원 증가. tf.pack([x, y]) = np.asarray([x, y])
showOperation(tf.pack([c3, v3]))      # [[1 3 5] [2 4 6]]
showOperation(tf.pack([c4, v4]))      # [[[1 3 5] [5 7 9]]  [[2 4 6] [6 8 0]]]

t1 = tf.pack([c3, v3])
t2 = tf.pack([c4, v4])

print('-----------unpack------------')         # 차원 감소
showOperation(tf.unpack(t1))          # [[1, 3, 5], [2, 4, 6]]
showOperation(tf.unpack(t2))          # [[[1, 3, 5], [5, 7, 9]],  [[2, 4, 6], [6, 8, 0]]]

print('-----------reverse------------')
showOperation(tf.reverse(c1, [True]))         # [8 6 4 2 0 9 7 5 3 1]
showOperation(tf.reverse(v1, [True, False]))  # [[7 8 9 0 1 2] [1 2 3 4 5 6]]
showOperation(tf.reverse(v1, [True, True ]))  # [[2 1 0 9 8 7] [6 5 4 3 2 1]]

print('-----------transpose------------')      # perm is useful to multi-dimension .
showOperation(tf.transpose(c3))       # [1 3 5]. not 1-D.
showOperation(tf.transpose(c4))       # [[1 5] [3 7] [5 9]]
showOperation(tf.transpose(c4, perm=[0, 1]))   # [[1 3 5] [5 7 9]]
showOperation(tf.transpose(c4, perm=[1, 0]))   # [[1 5] [3 7] [5 9]]

print('-----------gather------------')
showOperation(tf.gather(c1, [2, 5, 2, 5]))     # [5 0 5 0]
showOperation(tf.gather(v1, [0, 1]))           # [[1 2 3 4 5 6] [7 8 9 0 1 2]]
showOperation(tf.gather(v1, [[0, 0], [1, 1]])) # [[[1 2 3 4 5 6] [1 2 3 4 5 6]]  [[7 8 9 0 1 2] [7 8 9 0 1 2]]]

print('-----------one_hot------------')         # make one-hot matrix.
# [[ 1.  0.  0.]
#  [ 0.  1.  0.]
#  [ 0.  0.  1.]
#  [ 0.  1.  0.]]
showOperation(tf.one_hot([0, 1, 2, 1], 3))
# [[ 0.  0.  0.  1.]
#  [ 0.  0.  0.  0.]
#  [ 0.  1.  0.  0.]]
showOperation(tf.one_hot([3, -1, 1], 4))


# In[9]:

print sample[1:]


# In[48]:

# tensor matrix
mat1 = tf.constant([1,1,1,1])
showConstantDetail(vx)

mat4n= tf.constant([[1,1,1,1], [1,1,1,1], [1,1,1,1], [1,1,1,1]])
showConstantDetail(mat4n)

mat44 = tf.constant([[1], [1], [1], [1]])
showConstantDetail(mat44)

mat14 = tf.constant([[1,1,1,1]])
showConstantDetail(mat14)


# In[53]:

# concat examples
# https://www.tensorflow.org/versions/r0.10/api_docs/python/array_ops/slicing_and_joining#concat
mat14 = tf.constant([[2,2,1,1]])
showConstantDetail(mat14)

con1 = tf.concat(1, mat14)
showOperation(tf.concat(3, mat14))


# In[70]:

# reshape example
# https://www.tensorflow.org/api_docs/python/array_ops/shapes_and_shaping#reshape
t = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
showConstantDetail(t)

# 아래의 3개는 모두 동일한 결과를 생성함.
showConstantDetail(tf.reshape(t, [3, 4]))  # 1차원 행렬을 3*4으로 변경함.
showConstantDetail(tf.reshape(t, [3, -1])) # -1은 3으로 행을 구성한 경우 열의 갯수를 자동으로 맞춤(여기선 3)
showConstantDetail(tf.reshape(t, [-1, 4]))

showConstantDetail(tf.reshape(t, [-1, 2])) # 2을 열로 지정할 경우, -1은 자동으로 6으로 설정됨.


# In[80]:

ts1 = tf.constant([[ 0.45873451, 0.13431551, 0.2665318,   -0.11387707,  0.21802776, -0.3968451,
                    -0.12312219, 0.39910132, -0.55776322, -0.08702181,  0.23161684, -0.57659125,
                    0.24692856,  0.79166049,   0.7797873,  -0.43172249]])

showConstantDetail(ts1)

ts2 = tf.reshape(ts1, [-1, 2])
showOperation(tf.reshape(ts1, [-1, 2]))
showConstantDetail(ts2)

showOperation(tf.reshape(ts1, [4, 4]))


# In[ ]:



