# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
# 먼저 간단한 linear classifier를 구현해 보자 (http://cs231n.github.io/neural-networks-case-study/#linear)

# 1). Generating some data
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example) (300, 2)행렬
y = np.zeros(N*K, dtype='uint8') # class labels,  (300, 1) 행렬 3가지 유형의 point가 각 100개씩 존재

for j in xrange(K):
  ix = range(N*j,N*(j+1)) # j=0 -> range(0, 100)
                          # j=1 -> range(100, 200)
                          # j=3 -> range(200, 300)
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j # 정답 세팅 : ix(0~99)는 0으로 설정, ix(100~199)는 1로 설정, ix(200~299)는 2로 설정
# lets visualize the data:
# y = [0, 1, 2] 3가지 색상만 가짐
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
# print X.shape
# print X[:, :]
#plt.show() # 여기서 확인해 보면, 데이터가 -1 ~ 1 까지 적당한 분포로 생성되어 별도의 전처리가 필요없다

num_examples = X.shape[0] # 300
# some hyperparameters
step_size = 1.0 # 1e-0
reg = 0.001 # regularization strength, 1e-3

# 2) Training a Softmax Linear Classifier
# 2-1. Initialize the parameters
W = 0.01 * np.random.randn(D,K)  # (2, 3)
b = np.zeros((1,K))              # (1, 3)

for i in xrange(200):
  # 2-2. Compute the class scores (행렬 곱)
  #      - 300개 각 포인트 별로 3가지 색상중 어떤 것인지 확률(score)을 구한다.
  scores = np.dot(X, W) + b   # (300, 2) * (2, 3) = (300, 3) + (1, 3)  = (300, 3)<-- matrix broadcast로 확장되어 계산
                              # http://aikorea.org/cs231n/python-numpy-tutorial/ 브로드캐스팅
  # X1 = np.zeros((5, 2))
  # b1 = np.random.randn(1,K)
  # print "b1 : \n", b1
  # print "weight : \n", np.dot(X1, W)
  # print "bias : \n", np.dot(X1, W) + b1

  # 값을 지수적으로 증가시켜서 서로간의 차이가 크도록 unnormalized(비정규화) 한다. # get unnormalized probabilities
  exp_scores = np.exp(scores)
  # 이후에 확률적으로 표현하기 위하여 0~1로 정규화(normailze)한다 # normalize them for each example
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # (300, 3)

  # (300, 3) 행렬에서 0~99번까지는 0번째 row를 가져오고, 100~199은 1번째 row, 200~299은 2번째를 가져온다
  # 이렇게 하는 이유는 y에 저장된 데이터의 정답이 0~99는 0, 100~199는 1, 200~299는 2이기 때문이다
  # 그래서 학습을 통해서 예측한 확률과 정답이 일치하는지 확인한다.
  # 아래에서 300개 전체에 대해서 정답일 확률을 log(정답일 확률)로 계산(cross_entropy)하면,
  # 정답을 맞춘 확률이 1에 까까울 수록 log(정답일 확률)의 값은 0에 가깝게 나타난다. 즉, loss(cost)가 낮다는 의미.
  # 결국, corect_logprobs에는 300개 항목별로 loss(cost)를 저장하게 된다. (loss가 높은 것들은 나중에 back propagation을 통해서 조정됨)
  corect_logprobs = -np.log(probs[range(num_examples),y]) # (300, 3)에서 (300,1)
  # print corect_logprobs.shape, corect_logprobs
  # print probs
  # print range(num_examples)
  # print y.shape
  # print xx[99] , " : ", probs[99, 0]
  # print xx[199] , " : ", probs[199, 1]
  # print xx[299] , " : ", probs[299, 2]

  # compute the loss: average cross-entropy loss and regularization
  # - data_loss : 각 항목별 loss를 평균한 loss
  # - reg_loss : 정규화 변수(람다, λ)인 reg를 이용하여 regularization loss를 계산(공식은 웹사이트 참고.)
  # - loss : 초기 loss값은 아마도 대략 1.1정도 일 것이다. 왜냐하면, np.log(1/3)을 계산하면 그 값이 나오니까..
  #          np.log(1/3)의 의미는 정답을 맞출 확률이 1/3이라는 것(3개 중에 1개 맞추는)
  data_loss = np.sum(corect_logprobs)/num_examples
  reg_loss = 0.5*reg*np.sum(W*W)
  loss = data_loss + reg_loss
  # print loss, data_loss, reg_loss # 1.099369652 1.09936896609 6.85916365842e-07
  if i % 10 == 0:
    print "iteration %d: loss %f" % (i, loss)

  # 그럼 이제 1.1의 loss를 최대한 0에 가깝게 최적화할 수 있도록 gradient(경사도)를 계산해 보자.
  # 3). Computing the Analytic Gradient with Backpropagation
  #     - loss를 최소화 하기 위해 gradient(dscores)를 계산하여 사용할 것이다
  #     - ∂Li/∂fk = Pk -1(yi = k) -> f가 loss(Li)에 어느정도 영향을 미치는지 편미분 한 것 (back propagation)
  #     - f는 여기서 (scores = np.dot(X, W) + b)를 나타내고,
  #     - p는 여기서 (probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True))를 의미한다
  #     - 그럼 Pk -1(yi = k)의 의미는 뭘까?
  #       정답인 y의 i번째 값은 k라고 할 경우(예를 들어 y의 3번째 항목의 정답은 1, i=2, k=1)
  #       정답의 확률인 P의 k번째 값에서 1을 뺀다. P[0.2, 0.3, 0.5]일 때,
  #       정답의 확률인 0.3에서 -1을 빼면 [0.2, -0.7, 0.5]가 됨
  #       결국 ∂Li/∂fk = Pk -1(yi = k) = [0.2, -0.7, 0.5]
  #       왜 -1을 하지???, 이게 무슨 의미일까?
  #       -> Loss를 줄이기 위해서는 Loss에 minus영향을 주어야 한다.
  #       -> 그런데, 정답인 0.3은 plus영향을 주게되어 오히려 loss를 높이는 결과가 생긴다. (log(0.3)
  #          print -np.log(0.3), -np.log(0.7) # 1.203 0.356
  #       -> 결국 loss를 줄이기 위해서는 영향도를 minus로 낮추는것이다.
  #          (그런데 왜 -1인지는.. 아마도 최대값이 1이니까 정답에 가까울 수록 minus 영향을 적게 주기 위함일 듯)


##################################
# 좀 더 자세하게 번역ㅅ
##################################
  # loss는 계산했고, 이제 loss를 최소화할 것인데, 이를 gradient descent로 한다
  # 즉, 이말은 랜덤하게 초기화된 파라미터에서 시작해서,
  # 각 파라미터와 관련된 loss함수의 기울기(gradient)를 계산하고,
  # 그래서 결국, loss를 줄이기 위해서 어떻게 파라미터를 변경해야 하는지 알게된다는 의미이다
  # -> 좀 정리해서 말하면, 어떤 파라미터를 어떻게 변경해야 loss가 줄어드는지(정확도가 높아지는지) 알 수 있게 된다는 의미

  # 그럼 계산된 score(scores = np.dot(X, W) + b))가 어떻게 loss를 줄일수 있을까?
  # 이를 수식으로 표현하면 score함수가 loss에 미치는 영향, 즉 편미분 수식인 dLi/dScores가 된다
  # Li는 p(정답일 확률)을 입력값으로 계산되고, 또 p는 scores함수의 결과에 의존한다.
  # 이를 계산하기 위해서 chain rule을 사용하는 것은 재밌겠지만, 결국에는 아래와 같은 수식으로 간단하게 표현된다
  # ∂Li/∂fk = Pk -1(yi = k)

  # 예를 들어서 p가 p = [0.2, 0.3, 0.5]이고, 정답이 index 1인 값(0.3)이라 가정하자.
  # 그리고 위의 식을 대입하면, scores의 gradient(기울기)는 df = [0.2, -0.7, 0.5]가 된다.
  # gradient를 이해한다면, 위의 결과는 매우 직관적이다 (어떻게 직관적이냐?)
  # 즉 양수의 기울기는 loss를 더 증가시키게 되고,
  # 음수의 기울기는 loss를 감소시킨다는 의미이다,
  # --> 왜냐하면 df가 의미하는 것이 scores값이 loss에 얼마나 영향을 미치는지를 의미한다.(이를 기울기라 표현)
  # 양수의 값은 loss를 증가시키게 되므로,
  # 정답인 값만 음수(minus)로 변경시켜 준다. (기울기를 loss가 줄어드는 방향으로 변경한다.)
  # --> 이렇게 하는 가장 간단한 방법이 정답인 확률에 -1을 하는 것 (최대값이 1이므로, 무조건 0이하의 값이 나온다.)

 # 3-1. 각 항목별 확률이 저장된 score에 대한 기울기(gradient)를 계산한다. -> dscores에 저장
  #      (계산 결과에 따라, Loss를 줄이는 방향으로 조정)
  dscores = probs
  dscores[range(num_examples),y] -= 1 # 전체 300개 행에서 정답(y)가 있는 index만 -1을 한다. (이유는 위에서 설명)
  dscores /= num_examples  # gradient

  # 3-2. 이제 score에 대한 gradient(dscores)가 준비 되었다
  #      back propagation을 통해서 역으로 W, b의 값을 계산해 보자 (dW, db)
  #      - dW : loss를 줄일 수 있는 W의 기울기(gradient)
  #      - db : loss를 줄일 수 있는 b의 기울기
  dW = np.dot(X.T, dscores) # X(300,2)행렬을 X.T(2,300)행렬로 전치하고, 이를 dscores(300,3)와 행렬곱을 한다 -> (2,3) W 행렬과 동일
                            # [[-0.04125955 -0.05396174  0.09523223]
                            #  [ 0.08466205 -0.10061181  0.01594071]
  db = np.sum(dscores, axis=0, keepdims=True) # (1, 3)
                                              # 3가지 유형별(0,1,2)로 예측한 dscore값을 합한다
                                              # [[  3.53437887e-05   1.98766669e-05  -5.52204557e-05]]
  dW += reg*W # don't forget the regularization gradient (정규화를 위한 가중치 추가)
  # print X.T, X.shape
  # print " ------ \n"
  # print dscores, dscores.shape
  # print dW, dW.shape
  # print db, db.shape

  # 4). Performing a parameter update
  #     지금까지 모든 파라미터(W, b)가 loss에 어떤 영향을 미치는지 알수 있는 gradient를 계산하였다.
  #     이제는 negative gradient 방향으로 파라미터를 업데이트하여, loss를 줄여보자.
  # 4-1. perform a parameter update
  W += -step_size * dW
  b += -step_size * db

# 5). 정확도 측정
scores = np.dot(X, W) + b # 조정된 W,b를 이용하여 정답 예측
                          # [[ -2.95957156e-02  -3.61213609e-02   6.57170764e-02] -> 2
                          #  [ -5.24846369e-02  -4.06266724e-03   5.64075606e-02] -> 2
                          #  [ -7.45375980e-02   2.82410585e-02   4.60193135e-02] -> 2
                          #  [ -1.02765442e-01   5.80279157e-02   4.43075355e-02] -> 1
                          #  [ -1.18483668e-01   9.28507271e-02   2.50812716e-02] -> 1
                          #  [ -9.61814170e-02   1.19416065e-01  -2.37662059e-02] ->1
                          #  [ -1.62890407e-01   1.57345411e-01   4.71759800e-03] -> 1
                          #  [ -1.90216693e-01   1.88159196e-01   1.07824075e-03] -> 1
predicted_class = np.argmax(scores, axis=1) # 가장 확률이 높은 열(0, 1, 2)의 index를 가져옴.
                                            # [2 2 2 1 1 1 1 1 ...
print 'training accuracy: %.2f' % (np.mean(predicted_class == y))
# training accuracy: 0.53 --> 정확도가 너무 낮다.정확한 분류를 못함. (XOR 문제를 해결 못함)



# 6) 예측한 결과를 시각화 해보자.
#    원래 scatter plot을  [Data visualization]
# http://www.cnblogs.com/zf-blog/p/6073029.html
h=0.02
x_min , x_max = X[:,0].min() - 1, X[:,0].max() + 1
y_min , y_max = X[:,1].min() - 1, X[:,1].max() +1
xx, yy = np.meshgrid(np.arange(x_min , x_max ,h),
                     np.arange(y_min , y_max ,h))
Z = np.dot(np.c_[xx.ravel(),yy.ravel()],W) + b
Z = np.argmax(Z,axis = 1)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx,yy,Z,cmap=plt.cm.Spectral,alpha=0.8)
plt.scatter(X[:,0],X[:,1],c=y,s=40,cmap=plt.cm.Spectral)
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
# plt.show() # 화며에 출력하려면 주석 제거
#fig.savefig('spiral_linear.png')