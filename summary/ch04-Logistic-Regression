# 로지스틱 회귀(Logistic Regression) 정리

## 0. 전체 흐름 먼저 보기

로지스틱 회귀는 이름에 `회귀`가 들어가지만, 실제로는 주로 `분류(Classification)` 문제에 사용된다. 특히 결과가 두 가지 중 하나로 나뉘는 문제를 다룰 때 많이 사용한다.

예를 들어 다음과 같은 문제들이 있다.

```text
1. 시험 점수로 합격/불합격 예측
2. 메일이 스팸인지 정상 메일인지 예측
3. 어떤 사용자가 상품을 구매할지 말지 예측
4. 환자가 질병이 있는지 없는지 예측
```

이처럼 결과가 `0 또는 1`, `False 또는 True`, `불합격 또는 합격`처럼 두 가지로 나뉘는 문제를 `이진 분류(Binary Classification)`라고 한다.

로지스틱 회귀의 전체 구조는 다음과 같다.

```text
입력 데이터 x
   ↓
선형식 Wx + b 계산
   ↓
시그모이드 함수 적용
   ↓
0과 1 사이의 확률값 출력
   ↓
0.5 기준으로 0 또는 1 분류
```

---

## 1. 이진 분류(Binary Classification)

### 1.1 이진 분류란?

`이진 분류(Binary Classification)`는 정답이 두 가지 중 하나인 문제를 말한다.

예를 들어 시험 성적 데이터를 생각해보자.

| score(x) | result(y) |
|:---:|:---:|
| 45 | 불합격 |
| 50 | 불합격 |
| 55 | 불합격 |
| 60 | 합격 |
| 65 | 합격 |
| 70 | 합격 |

이 데이터를 숫자로 표현하면 보통 다음과 같이 둔다.

```text
불합격 = 0
합격 = 1
```

즉, 모델이 해야 할 일은 다음과 같다.

```text
시험 점수 x를 입력받아서
합격이면 1,
불합격이면 0을 예측하는 것
```

---

### 1.2 선형 회귀로 이진 분류를 하면 왜 애매한가?

선형 회귀의 가설식은 다음과 같다.

$$
H(x) = Wx + b
$$

여기서 `W`는 가중치 또는 기울기, `b`는 편향 또는 y절편이다.

선형 회귀는 직선을 이용해서 값을 예측한다. 그런데 이진 분류의 정답은 `0` 또는 `1`이어야 한다.

문제는 직선 함수의 출력값에는 제한이 없다는 것이다.

예를 들어 선형 회귀는 다음과 같은 값을 낼 수 있다.

```text
H(x) = -2.3
H(x) = 0.7
H(x) = 1.8
H(x) = 10.5
```

하지만 분류 문제에서는 결과가 다음처럼 나와야 한다.

```text
0 또는 1
```

따라서 단순한 직선 함수만으로는 합격/불합격처럼 두 가지 중 하나를 고르는 문제를 깔끔하게 표현하기 어렵다.

<img width="462" height="273" alt="Image" src="https://github.com/user-attachments/assets/2cb3df1b-9e6d-48da-b754-79653ea4a7c0" />

---

## 2. 시그모이드 함수(Sigmoid Function)

### 2.1 시그모이드 함수가 필요한 이유

이진 분류에서는 출력값이 `0과 1 사이`로 나오면 편하다.

왜냐하면 출력값을 확률처럼 해석할 수 있기 때문이다.

예를 들어 모델 출력이 다음과 같다고 하자.

```text
0.91
```

이 값은 다음처럼 해석할 수 있다.

```text
합격일 확률이 약 91%이다.
```

이렇게 출력값을 0과 1 사이로 만들어주는 대표적인 함수가 `시그모이드 함수(Sigmoid Function)`이다.

---

### 2.2 시그모이드 함수 식

시그모이드 함수는 다음과 같다.

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

로지스틱 회귀에서는 선형식 `Wx + b`를 시그모이드 함수에 넣는다.

$$
H(x) = \operatorname{sigmoid}(Wx + b)
$$

또는 다음처럼 쓸 수 있다.

$$
H(x) = \sigma(Wx + b) = \frac{1}{1 + e^{-(Wx+b)}}
$$

즉, 로지스틱 회귀의 가설식은 다음 흐름으로 이해하면 된다.

```text
1. 먼저 Wx + b를 계산한다.
2. 그 결과를 sigmoid 함수에 넣는다.
3. 최종 출력은 0과 1 사이의 값이 된다.
```

---

### 2.3 시그모이드 함수의 특징

시그모이드 함수의 핵심 특징은 다음과 같다.

```text
1. 입력값이 매우 커지면 출력값은 1에 가까워진다.
2. 입력값이 매우 작아지면 출력값은 0에 가까워진다.
3. 입력값이 0이면 출력값은 0.5이다.
4. 출력값은 항상 0과 1 사이이다.
```

예를 들어 다음과 같이 볼 수 있다.

```text
x가 매우 작음   → sigmoid(x) ≈ 0
x = 0        → sigmoid(x) = 0.5
x가 매우 큼    → sigmoid(x) ≈ 1
```

<img width="378" height="267" alt="Image" src="https://github.com/user-attachments/assets/caf056be-6934-4f30-8d96-485d8fb427e4" />

---

## 3. W와 b가 시그모이드 그래프에 주는 영향

로지스틱 회귀의 가설식은 다음과 같다.

$$
H(x) = \operatorname{sigmoid}(Wx + b)
$$

여기서 `W`와 `b`는 모델이 학습을 통해 찾아야 하는 값이다.

---

### 3.1 W의 역할: 그래프의 경사 조절

`W`는 그래프의 경사, 즉 얼마나 급하게 변하는지를 조절한다.

```text
W가 작을 때  → 그래프가 완만하게 변한다.
W가 클 때    → 그래프가 급격하게 변한다.
```

예를 들어 다음과 같이 생각할 수 있다.

```text
W가 작을 때:
점수가 조금 올라가도 합격 확률이 천천히 증가한다.

W가 클 때:
특정 점수를 기준으로 합격 확률이 급격하게 증가한다.
```

즉, `W`는 입력값 변화에 모델이 얼마나 민감하게 반응할지를 결정한다.

<img width="371" height="264" alt="Image" src="https://github.com/user-attachments/assets/3dfa17ee-1aa0-4170-bfac-80e558623093" />

---

### 3.2 b의 역할: 그래프의 좌우 이동

`b`는 그래프를 좌우로 이동시키는 역할을 한다.

`W`가 그래프의 급한 정도를 바꾼다면, `b`는 기준점이 어디에 위치할지를 바꾼다.

```text
b가 변하면 시그모이드 그래프가 왼쪽 또는 오른쪽으로 이동한다.
```

예를 들어 합격 기준이 60점인지, 70점인지에 따라 그래프의 위치가 달라져야 한다. 이런 기준점의 이동을 담당하는 것이 `b`라고 이해하면 된다.

<img width="375" height="258" alt="Image" src="https://github.com/user-attachments/assets/30408984-375a-4783-82d4-4725098117f7" />

---

## 4. 시그모이드 함수를 이용한 분류

시그모이드 함수의 출력값은 0과 1 사이이다.

따라서 이 값을 확률처럼 생각할 수 있다.

예를 들어 다음과 같은 출력값이 있다고 하자.

```text
H(x) = 0.83
```

그러면 다음처럼 해석할 수 있다.

```text
정답이 1일 확률이 약 83%이다.
```

하지만 최종적으로는 `0 또는 1`로 결정해야 한다. 그래서 보통 `임계값(threshold)`을 사용한다.

가장 많이 쓰는 기준은 `0.5`이다.

```text
H(x) >= 0.5  → 1로 분류
H(x) < 0.5   → 0으로 분류
```

예를 들면 다음과 같다.

| 예측값 H(x) | 최종 분류 |
|:---:|:---:|
| 0.91 | 1 |
| 0.72 | 1 |
| 0.50 | 1 |
| 0.48 | 0 |
| 0.12 | 0 |

합격/불합격 문제로 보면 다음과 같다.

```text
0.5 이상이면 합격
0.5 미만이면 불합격
```

---

## 5. 비용 함수(Cost Function)

### 5.1 비용 함수가 필요한 이유

모델이 예측한 값과 실제 정답이 얼마나 다른지 계산해야 한다.

이 차이를 숫자로 나타낸 것이 `비용(cost)` 또는 `손실(loss)`이다.

모델 학습의 목표는 다음과 같다.

```text
cost가 작아지도록 W와 b를 조정하는 것
```

---

### 5.2 선형 회귀의 비용 함수 MSE

선형 회귀에서는 보통 평균 제곱 오차를 사용했다.

$$
cost(W,b) = \frac{1}{n}\sum_{i=1}^{n}(y^{(i)} - H(x^{(i)}))^2
$$

이를 `MSE(Mean Squared Error)`라고 한다.

하지만 로지스틱 회귀에서 MSE를 그대로 사용하면 문제가 생길 수 있다.

---

### 5.3 로지스틱 회귀에서 MSE를 쓰면 생기는 문제

로지스틱 회귀의 가설식은 다음과 같다.

$$
H(x) = \operatorname{sigmoid}(Wx + b)
$$

여기에 MSE를 적용하면 비용 함수의 그래프가 복잡하고 울퉁불퉁한 형태가 될 수 있다.

이런 형태에서는 경사 하강법이 제대로 최솟값을 찾기 어렵다.

---

### 5.4 로컬 미니멈과 글로벌 미니멈

비용 함수에서 가장 낮은 지점을 찾아야 모델이 좋은 W와 b를 찾을 수 있다.

그런데 울퉁불퉁한 그래프에서는 최솟값처럼 보이는 지점이 여러 개 생길 수 있다.

```text
Local Minimum:
특정 구간 안에서는 가장 낮지만, 전체에서 가장 낮은 값은 아닌 지점

Global Minimum:
전체 함수에서 가장 낮은 지점
```

경사 하강법이 Local Minimum에 빠지면, 실제로는 더 좋은 값이 있는데도 학습이 멈춘 것처럼 될 수 있다.

<img width="282" height="262" alt="Image" src="https://github.com/user-attachments/assets/bb01d7e3-19ac-4976-a673-c10145ccabb7" />

---

## 6. 로지스틱 회귀의 비용 함수: Binary Cross Entropy

### 6.1 로그 함수를 사용하는 이유

로지스틱 회귀에서는 MSE 대신 로그 기반 비용 함수를 사용한다.

시그모이드 출력값은 0과 1 사이이므로, 실제 정답이 1인데 예측값이 0에 가까우면 큰 벌점을 줘야 한다.

반대로 실제 정답이 1인데 예측값이 1에 가까우면 작은 벌점을 줘야 한다.

---

### 6.2 실제값이 1일 때

실제값이 `y = 1`일 때는 다음 비용 함수를 사용한다.

$$
cost(H(x), y) = -\log(H(x))
$$

이 식의 의미는 다음과 같다.

```text
정답이 1인데 H(x)가 1에 가까움 → cost가 작다.
정답이 1인데 H(x)가 0에 가까움 → cost가 매우 커진다.
```

---

### 6.3 실제값이 0일 때

실제값이 `y = 0`일 때는 다음 비용 함수를 사용한다.

$$
cost(H(x), y) = -\log(1-H(x))
$$

이 식의 의미는 다음과 같다.

```text
정답이 0인데 H(x)가 0에 가까움 → cost가 작다.
정답이 0인데 H(x)가 1에 가까움 → cost가 매우 커진다.
```

<img width="266" height="224" alt="Image" src="https://github.com/user-attachments/assets/3be4c626-d4eb-47a4-af64-db9868fa8985" />

---

### 6.4 두 식을 하나로 합치기

실제값이 1일 때와 0일 때를 각각 따로 계산하면 불편하다.

그래서 두 식을 하나로 합친다.

$$
cost(H(x), y) = -[y\log(H(x)) + (1-y)\log(1-H(x))]
$$

이 식이 잘 작동하는 이유는 `y`가 0 또는 1이기 때문이다.

---

#### y = 1인 경우

$$
cost(H(x), 1) = -[1 \cdot \log(H(x)) + (1-1)\log(1-H(x))]
$$

$$
cost(H(x), 1) = -\log(H(x))
$$

`(1-y)`가 0이 되므로 두 번째 항이 사라진다.

---

#### y = 0인 경우

$$
cost(H(x), 0) = -[0 \cdot \log(H(x)) + (1-0)\log(1-H(x))]
$$

$$
cost(H(x), 0) = -\log(1-H(x))
$$

`y`가 0이 되므로 첫 번째 항이 사라진다.

---

### 6.5 전체 데이터에 대한 비용 함수

데이터가 여러 개 있을 때는 각 데이터의 cost를 구한 뒤 평균을 낸다.

$$
J(W, b) = -\frac{1}{n}\sum_{i=1}^{n}[y^{(i)}\log(H(x^{(i)})) + (1-y^{(i)})\log(1-H(x^{(i)}))]
$$

이 비용 함수를 `Binary Cross Entropy`, 줄여서 `BCE`라고 부른다.

PyTorch에서는 다음 함수로 구현할 수 있다.

```python
F.binary_cross_entropy(hypothesis, y_train)
```

---

## 7. 경사 하강법으로 W와 b 학습하기

로지스틱 회귀도 결국 W와 b를 학습해야 한다.

목표는 다음과 같다.

```text
cost가 최소가 되는 W와 b 찾기
```

경사 하강법의 업데이트 식은 다음과 같이 볼 수 있다.

$$
W := W - \alpha \frac{\partial}{\partial W} J(W,b)
$$

$$
b := b - \alpha \frac{\partial}{\partial b} J(W,b)
$$

여기서 `α`는 학습률(learning rate)이다.

```text
학습률이 너무 크면:
최솟값을 지나쳐서 학습이 불안정할 수 있다.

학습률이 너무 작으면:
학습이 너무 느릴 수 있다.
```

---

## 8. PyTorch로 로지스틱 회귀 직접 구현하기

이제 PyTorch로 로지스틱 회귀를 구현하는 흐름을 정리한다.

---

### 8.1 필요한 라이브러리 import

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
```

각 코드의 의미는 다음과 같다.

```text
1. torch
   PyTorch의 기본 기능을 사용하기 위해 import한다.

2. torch.nn
   신경망 계층을 만들 때 사용한다.

3. torch.nn.functional
   비용 함수나 활성화 함수 등을 함수 형태로 사용할 때 사용한다.

4. torch.optim
   SGD 같은 옵티마이저를 사용할 때 필요하다.

5. torch.manual_seed(1)
   랜덤 초기값을 고정해서 실행할 때마다 같은 결과가 나오게 한다.
```

---

### 8.2 훈련 데이터 만들기

```python
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)
```

여기서 `x_data`는 입력 데이터이고, 각 데이터는 두 개의 특성을 가진다.

```text
[1, 2]
[2, 3]
[3, 1]
...
```

즉, 입력 특성이 2개이다.

`y_data`는 정답 레이블이다.

```text
0, 0, 0, 1, 1, 1
```

---

### 8.3 데이터 크기 확인하기

```python
print(x_train.shape)
print(y_train.shape)
```

출력 결과는 다음과 같다.

```text
torch.Size([6, 2])
torch.Size([6, 1])
```

의미는 다음과 같다.

```text
x_train: 6개의 데이터, 각 데이터는 2개의 입력값을 가짐
          따라서 shape는 [6, 2]

y_train: 6개의 데이터, 각 데이터의 정답은 1개
          따라서 shape는 [6, 1]
```

---

### 8.4 W와 b 크기 이해하기

입력 데이터 X의 크기는 다음과 같다.

```text
X = 6 × 2
```

출력값은 다음 크기여야 한다.

```text
H(x) = 6 × 1
```

행렬 곱셈을 생각하면 다음이 되어야 한다.

```text
XW = (6 × 2)(2 × 1) = 6 × 1
```

따라서 W의 크기는 다음과 같아야 한다.

```text
W = 2 × 1
```

코드로는 다음과 같이 선언한다.

```python
W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

여기서 `requires_grad=True`는 학습 과정에서 이 값들에 대한 기울기를 계산하겠다는 뜻이다.

---

### 8.5 시그모이드 가설식 직접 구현

로지스틱 회귀의 가설식은 다음과 같다.

$$
H(x) = \operatorname{sigmoid}(XW + b)
$$

PyTorch 코드로 직접 쓰면 다음과 같다.

```python
hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(W) + b)))
```

여기서 중요한 부분은 다음과 같다.

```text
1. x_train.matmul(W)
   입력 데이터 X와 가중치 W를 행렬 곱한다.

2. + b
   편향 b를 더한다.

3. torch.exp()
   e의 거듭제곱을 계산한다.

4. 전체적으로 sigmoid 식을 직접 구현한 것이다.
```

하지만 PyTorch에는 이미 시그모이드 함수가 있다.

```python
hypothesis = torch.sigmoid(x_train.matmul(W) + b)
```

위 코드는 직접 구현한 식과 같은 의미이다.

---

### 8.6 W와 b가 0일 때 예측값

처음에 W와 b를 모두 0으로 초기화하면 다음과 같다.

```text
x_train.matmul(W) + b = 0
```

따라서 시그모이드에 들어가는 값도 0이다.

$$
\operatorname{sigmoid}(0) = 0.5
$$

그래서 모든 예측값이 0.5가 나온다.

```text
tensor([[0.5000],
        [0.5000],
        [0.5000],
        [0.5000],
        [0.5000],
        [0.5000]])
```

---

### 8.7 비용 함수 직접 계산

현재 예측값과 실제값을 비교하여 cost를 계산한다.

한 개 데이터에 대한 cost는 다음과 같이 계산할 수 있다.

```python
cost = -(y_train[0] * torch.log(hypothesis[0]) + 
            (1 - y_train[0]) * torch.log(1 - hypothesis[0]))
```

전체 데이터에 대해서는 다음처럼 계산한다.

```python
losses = -(y_train * torch.log(hypothesis) + 
           (1 - y_train) * torch.log(1 - hypothesis))

cost = losses.mean()
```

처음에 예측값이 전부 0.5이면 각 데이터의 cost는 다음과 같이 나온다.

```text
0.6931
```

전체 평균도 다음과 같다.

```text
cost = 0.6931
```

---

### 8.8 PyTorch의 binary_cross_entropy 사용

위 비용 함수는 PyTorch에서 이미 제공한다.

```python
F.binary_cross_entropy(hypothesis, y_train)
```

이 함수는 직접 계산한 BCE와 같은 결과를 낸다.

---

### 8.9 직접 구현한 전체 학습 코드

```python
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W, b], lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    hypothesis = torch.sigmoid(x_train.matmul(W) + b)

    cost = -(y_train * torch.log(hypothesis) + 
             (1 - y_train) * torch.log(1 - hypothesis)).mean()

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
```

---

### 8.10 학습 코드의 핵심 흐름

```text
1. hypothesis = torch.sigmoid(x_train.matmul(W) + b)
   현재 W와 b로 예측값을 계산한다.

2. cost = ...
   예측값과 실제값의 차이를 BCE로 계산한다.

3. optimizer.zero_grad()
   이전에 계산된 기울기를 초기화한다.

4. cost.backward()
   역전파를 통해 W와 b의 기울기를 계산한다.

5. optimizer.step()
   계산된 기울기를 이용해 W와 b를 업데이트한다.
```

---

### 8.11 학습 후 예측

학습이 끝난 뒤 예측값은 다음과 비슷하게 나온다.

```text
tensor([[2.7648e-04],
        [3.1608e-02],
        [3.8977e-02],
        [9.5622e-01],
        [9.9823e-01],
        [9.9969e-01]])
```

이를 보기 쉽게 바꾸면 다음과 같다.

```text
0.00027648
0.031608
0.038977
0.95622
0.99823
0.99969
```

0.5를 기준으로 분류하면 다음과 같다.

```python
prediction = hypothesis >= torch.FloatTensor([0.5])
print(prediction)
```

출력은 다음과 같다.

```text
tensor([[False],
        [False],
        [False],
        [ True],
        [ True],
        [ True]])
```

실제값은 다음과 같았다.

```text
[[0], [0], [0], [1], [1], [1]]
```

즉, 예측 결과가 실제값과 일치한다.

---

## 9. 인공 신경망으로 표현되는 로지스틱 회귀

로지스틱 회귀는 인공 신경망의 아주 단순한 형태로 볼 수 있다.

입력이 두 개라면 다음과 같은 구조가 된다.

```text
x1 ── w1 ┐
         ├── x1w1 + x2w2 + b ── sigmoid ── y
x2 ── w2 ┘

bias 1 ── b
```

수식으로는 다음과 같다.

$$
H(x) = \operatorname{sigmoid}(x_1w_1 + x_2w_2 + b)
$$

여기서 각 의미는 다음과 같다.

```text
x1, x2: 입력값
w1, w2: 각 입력에 곱해지는 가중치
b: 편향
sigmoid: 0과 1 사이로 변환하는 함수
y: 최종 예측값
```

<img width="235" height="177" alt="Image" src="https://github.com/user-attachments/assets/e3b72b1c-a376-4333-a940-85ac09cbcc3c" />

주의할 점도 있다.

```text
시그모이드 함수는 이진 분류의 출력층에서는 사용할 수 있지만,
깊은 인공 신경망의 은닉층에서는 예전만큼 많이 사용되지 않는다.
```

그 이유는 나중에 신경망을 배울 때 등장하는 `기울기 소실 문제`와 관련이 있다.

---

## 10. nn.Linear와 nn.Sigmoid로 로지스틱 회귀 구현하기

### 10.1 선형 회귀와 로지스틱 회귀의 구현 차이

선형 회귀의 가설식은 다음과 같았다.

$$
H(x) = Wx + b
$$

PyTorch에서는 이 부분을 `nn.Linear()`로 구현할 수 있다.

로지스틱 회귀의 가설식은 다음과 같다.

$$
H(x) = \operatorname{sigmoid}(Wx + b)
$$

따라서 PyTorch에서는 다음처럼 구현할 수 있다.

```text
nn.Linear()
   ↓
nn.Sigmoid()
```

즉, `nn.Linear()`의 결과를 `nn.Sigmoid()`에 통과시키면 로지스틱 회귀가 된다.

---

### 10.2 nn.Sequential 사용

`nn.Sequential()`은 여러 층을 순서대로 연결할 때 사용한다.

```python
model = nn.Sequential(
   nn.Linear(2, 1),
   nn.Sigmoid()
)
```

이 코드는 다음 수식을 구현한 것이다.

$$
H(x) = \operatorname{sigmoid}(Wx + b)
$$

각 부분의 의미는 다음과 같다.

```text
nn.Linear(2, 1)
입력 특성 2개를 받아 출력 1개를 만든다.
즉, x1, x2를 받아 하나의 값 Wx + b를 계산한다.

nn.Sigmoid()
Linear의 출력값을 0과 1 사이로 바꾼다.
```

---

### 10.3 처음 예측값 확인

```python
model(x_train)
```

출력 예시는 다음과 같다.

```text
tensor([[0.4020],
        [0.4147],
        [0.6556],
        [0.5948],
        [0.6788],
        [0.8061]])
```

이 값들은 0과 1 사이이므로 확률처럼 볼 수 있다.

하지만 아직 학습 전이므로 큰 의미는 없다.

왜냐하면 `nn.Linear()` 안의 W와 b가 랜덤으로 초기화되어 있기 때문이다.

---

### 10.4 nn.Sequential 모델 학습 코드

```python
optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    hypothesis = model(x_train)

    cost = F.binary_cross_entropy(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 10 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5])
        correct_prediction = prediction.float() == y_train
        accuracy = correct_prediction.sum().item() / len(correct_prediction)

        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(
            epoch, nb_epochs, cost.item(), accuracy * 100,
        ))
```

---

### 10.5 코드 흐름 설명

```text
1. optimizer = optim.SGD(model.parameters(), lr=1)
   모델 내부의 파라미터 W와 b를 SGD로 업데이트하겠다는 뜻이다.

2. hypothesis = model(x_train)
   입력 데이터를 모델에 넣어 예측값을 구한다.
   model(x_train)을 호출하면 nn.Linear와 nn.Sigmoid가 순서대로 실행된다.

3. cost = F.binary_cross_entropy(hypothesis, y_train)
   예측값과 실제값의 차이를 BCE 비용 함수로 계산한다.

4. optimizer.zero_grad()
   이전 반복에서 남아 있던 기울기를 초기화한다.

5. cost.backward()
   역전파를 수행해서 각 파라미터의 기울기를 계산한다.

6. optimizer.step()
   계산된 기울기를 이용해서 W와 b를 업데이트한다.

7. prediction = hypothesis >= 0.5
   예측 확률이 0.5 이상이면 True, 아니면 False로 분류한다.

8. correct_prediction = prediction.float() == y_train
   예측 결과와 실제 정답이 같은지 비교한다.

9. accuracy = correct_prediction.sum().item() / len(correct_prediction)
   맞힌 개수를 전체 데이터 개수로 나누어 정확도를 계산한다.
```

---

### 10.6 학습 결과

출력 예시는 다음과 같다.

```text
Epoch    0/1000 Cost: 0.539713 Accuracy 83.33%
...
Epoch 1000/1000 Cost: 0.019843 Accuracy 100.00%
```

의미는 다음과 같다.

```text
처음에는 cost가 크고 정확도가 낮을 수 있다.
학습이 진행될수록 cost가 감소한다.
최종적으로 훈련 데이터에 대해 정확도가 100%가 될 수 있다.
```

학습 후 예측값은 다음과 비슷하다.

```text
tensor([[0.0240],
        [0.1476],
        [0.2739],
        [0.7967],
        [0.9491],
        [0.9836]])
```

0.5를 기준으로 보면 다음과 같다.

```text
0.0240 → False → 0
0.1476 → False → 0
0.2739 → False → 0
0.7967 → True  → 1
0.9491 → True  → 1
0.9836 → True  → 1
```

실제값과 모두 일치한다.

---

### 10.7 학습된 W와 b 확인

```python
print(list(model.parameters()))
```

출력 예시는 다음과 같다.

```text
[Parameter containing:
tensor([[3.2534, 1.5181]], requires_grad=True), Parameter containing:
tensor([-14.4839], requires_grad=True)]
```

의미는 다음과 같다.

```text
W = [[3.2534, 1.5181]]
b = [-14.4839]
```

즉, 학습이 끝난 모델의 식은 대략 다음과 같다.

$$
H(x) = \operatorname{sigmoid}(3.2534x_1 + 1.5181x_2 - 14.4839)
$$

---

## 11. 모델을 클래스로 구현하기

### 11.1 왜 클래스로 구현하는가?

간단한 모델은 `nn.Sequential()`로 만들 수 있다.

하지만 실제 딥러닝 모델은 구조가 더 복잡해지는 경우가 많다.

그래서 PyTorch에서는 보통 클래스를 만들어 모델을 정의한다.

기본 구조는 다음과 같다.

```python
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))
```

---

### 11.2 코드 설명

#### class BinaryClassifier(nn.Module)

```python
class BinaryClassifier(nn.Module):
```

`BinaryClassifier`라는 새로운 모델 클래스를 만든다.

`nn.Module`을 상속받는 이유는 PyTorch 모델로 사용하기 위해서이다.

`nn.Module`을 상속받아야 다음과 같은 기능을 사용할 수 있다.

```text
1. 모델 파라미터 관리
2. model.parameters() 사용
3. forward 연산 자동 호출
4. 학습 모드와 평가 모드 관리
```

---

#### __init__()

```python
def __init__(self):
    super().__init__()
    self.linear = nn.Linear(2, 1)
    self.sigmoid = nn.Sigmoid()
```

`__init__()`은 모델이 생성될 때 실행되는 함수이다.

여기서는 모델 안에 필요한 층을 정의한다.

```text
self.linear:
입력 2개를 받아 출력 1개를 만드는 선형 계층

self.sigmoid:
선형 계층의 출력을 0과 1 사이로 바꾸는 시그모이드 함수
```

`super().__init__()`은 부모 클래스인 `nn.Module`의 초기화 기능을 불러오는 코드이다.

PyTorch 모델을 만들 때는 거의 항상 넣는다고 보면 된다.

---

#### forward()

```python
def forward(self, x):
    return self.sigmoid(self.linear(x))
```

`forward()`는 입력 데이터가 모델 안에서 어떤 순서로 계산되는지 정의한다.

여기서는 다음 순서로 계산된다.

```text
x
 ↓
self.linear(x)
 ↓
self.sigmoid(...)
 ↓
최종 예측값
```

즉, 수식으로는 다음과 같다.

$$
H(x) = \operatorname{sigmoid}(Wx + b)
$$

---

### 11.3 model(x_train)을 호출하면 forward가 자동 실행된다

모델 객체를 만든다.

```python
model = BinaryClassifier()
```

그리고 다음처럼 호출한다.

```python
hypothesis = model(x_train)
```

그러면 내부적으로는 다음 함수가 실행된다.

```python
forward(x_train)
```

즉, `model(x_train)`은 그냥 함수 호출처럼 보이지만 실제로는 모델의 forward 연산을 실행하는 것이다.

---

## 12. 클래스로 구현한 전체 코드

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

model = BinaryClassifier()

optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    hypothesis = model(x_train)

    cost = F.binary_cross_entropy(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 10 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5])
        correct_prediction = prediction.float() == y_train
        accuracy = correct_prediction.sum().item() / len(correct_prediction)

        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(
            epoch, nb_epochs, cost.item(), accuracy * 100,
        ))
```

---

## 13. nn.Sequential과 Class 구현 비교

| 구분 | nn.Sequential | Class 방식 |
|---|---|---|
| 장점 | 간단하다 | 복잡한 모델도 구현 가능하다 |
| 사용 난이도 | 쉽다 | 조금 더 어렵다 |
| 구조 | 층을 순서대로 쌓음 | `__init__`과 `forward`를 직접 정의 |
| 적합한 경우 | 단순한 모델 | 대부분의 실제 PyTorch 모델 |

이번 로지스틱 회귀에서는 두 방식이 거의 같은 결과를 낸다.

하지만 앞으로 인공 신경망을 더 깊게 배우면 클래스 방식이 훨씬 중요해진다.

---

## 14. 반드시 이해해야 할 핵심 정리

1. 이진 분류는 정답이 0 또는 1로 나뉘는 문제이다.

2. 선형 회귀의 가설식은 $H(x) = Wx + b$이다.

3. 선형 회귀는 출력값에 제한이 없어서 이진 분류에 그대로 쓰기 어렵다.

4. 로지스틱 회귀의 가설식은 $H(x) = \operatorname{sigmoid}(Wx + b)$이다.

5. 시그모이드 함수는 어떤 입력값이 들어와도 0과 1 사이의 값을 출력한다.

6. 시그모이드 출력값은 확률처럼 해석할 수 있다.

7. 보통 0.5를 기준으로 0 또는 1을 분류한다.

8. W는 그래프의 경사, b는 그래프의 좌우 이동을 담당한다.

9. 로지스틱 회귀에서는 MSE보다 Binary Cross Entropy 비용 함수를 사용한다.

10. PyTorch에서 BCE는 F.binary_cross_entropy(hypothesis, y_train)으로 계산할 수 있다.

11. nn.Linear(2, 1)은 입력 2개를 받아 출력 1개를 만드는 Wx + b 역할을 한다.

12. nn.Sigmoid()는 nn.Linear의 출력을 0과 1 사이로 바꿔준다.

13. nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())는 로지스틱 회귀 모델이다.

14. 클래스로 모델을 만들 때는 nn.Module을 상속받고 __init__과 forward를 정의한다.

15. model(x_train)을 호출하면 내부적으로 forward(x_train)이 자동 실행된다.

---

## 15. 헷갈리기 쉬운 부분 정리

### 15.1 로지스틱 회귀는 회귀인가 분류인가?

이름에는 회귀가 들어가지만 실제로는 분류에 사용된다.

```text
로지스틱 회귀 = 분류 알고리즘
```

---

### 15.2 sigmoid는 왜 쓰는가?

출력값을 0과 1 사이로 만들기 위해 사용한다.
$Wx + b$는 범위 제한이 없지만,
$\operatorname{sigmoid}(Wx + b)$는 0과 1 사이의 값을 출력한다.


---

### 15.3 0.5는 무조건 고정인가?

기본적으로는 0.5를 많이 사용한다.

하지만 문제 상황에 따라 기준을 바꿀 수도 있다.

예를 들어 질병 예측처럼 놓치면 위험한 문제에서는 0.5보다 낮은 기준을 사용할 수도 있다.

---

### 15.4 cost와 accuracy는 같은가?

다르다.

```text
cost:
예측값과 실제값의 차이를 수치로 나타낸 것
학습할 때 줄여야 하는 값

accuracy:
전체 데이터 중 몇 개를 맞혔는지 나타낸 비율
모델 성능을 확인할 때 보는 값
```

---

### 15.5 cost.backward()는 무엇인가?

`cost.backward()`는 역전파를 수행해서 W와 b에 대한 기울기를 계산한다.

즉, 다음 질문에 대한 답을 구하는 과정이다.

```text
cost를 줄이려면 W와 b를 어느 방향으로 바꿔야 하는가?
```

---

### 15.6 optimizer.step()은 무엇인가?

`optimizer.step()`은 계산된 기울기를 이용해서 실제로 W와 b 값을 바꾸는 코드이다.

```text
cost.backward()     → 기울기 계산
optimizer.step()    → 파라미터 업데이트
```

---

### 15.7 zero_grad()는 왜 필요한가?

PyTorch는 기울기를 자동으로 누적한다.

그래서 매번 학습하기 전에 이전 기울기를 지워줘야 한다.

```python
optimizer.zero_grad()
```

이 코드를 빼면 이전 기울기와 현재 기울기가 섞여서 학습이 이상해질 수 있다.

---

## 16. 전체 내용을 한 줄로 요약

```text
로지스틱 회귀는 Wx + b의 결과를 시그모이드 함수에 넣어 0과 1 사이의 확률값으로 바꾸고, 그 값을 0.5 기준으로 나누어 이진 분류를 수행하는 모델이다.
```
