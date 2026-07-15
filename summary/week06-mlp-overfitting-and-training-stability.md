# 다층 퍼셉트론, 과적합 방지와 딥러닝 학습 안정화 정리

## 0. 전체 흐름 먼저 보기

이번 정리는 `다층 퍼셉트론을 이용한 숫자 이미지 분류`, `과적합 방지`, `기울기 소실과 폭주`, `가중치 초기화`, `배치 정규화`, `층 정규화`까지 이어지는 내용이다.

전체 흐름은 다음과 같다.

```text
1. 사이킷런의 숫자 필기 데이터는 8 × 8 이미지이며, 하나의 이미지를 64차원 벡터로 사용한다.
2. 64차원 입력을 여러 선형 계층과 ReLU에 통과시켜 숫자 0~9를 분류하는 다층 퍼셉트론을 만든다.
3. MNIST는 28 × 28 이미지이므로 하나의 이미지를 784차원 벡터로 사용한다.
4. MNIST 데이터를 훈련 세트와 테스트 세트로 나누고 DataLoader로 미니 배치를 구성한다.
5. 모델은 클래스별 logit을 출력하고 CrossEntropyLoss로 손실을 계산한다.
6. 역전파와 옵티마이저를 이용해 가중치를 학습한 뒤, 평가 모드에서 테스트 정확도와 개별 예측을 확인한다.
7. 모델이 훈련 데이터를 지나치게 암기하면 과적합이 발생한다.
8. 과적합은 데이터 증강, 모델 단순화, 가중치 규제, 드롭아웃 등으로 완화할 수 있다.
9. 깊은 신경망에서는 역전파 중 기울기 소실 또는 기울기 폭주가 발생할 수 있다.
10. ReLU 계열 활성화 함수와 Xavier·He 초기화는 기울기 문제를 줄이는 데 도움이 된다.
11. 배치 정규화는 같은 특성의 미니 배치 통계를 이용하고, 층 정규화는 각 샘플 내부의 특성 통계를 이용한다.
```

숫자 이미지 분류의 기본 흐름은 다음과 같다.

```text
이미지 데이터
   ↓
픽셀값을 1차원 벡터로 변환
   ↓
다층 퍼셉트론에 입력
   ↓
숫자 0~9에 대한 10개의 logit 출력
   ↓
CrossEntropyLoss로 손실 계산
   ↓
역전파로 기울기 계산
   ↓
옵티마이저로 가중치와 편향 업데이트
   ↓
테스트 데이터로 일반화 성능 평가
```

과적합과 학습 불안정을 완화하는 흐름은 다음과 같다.

```text
과적합 발생
   ↓
데이터 증가·증강 / 모델 복잡도 감소 / L1·L2 규제 / 드롭아웃

기울기 소실·폭주 발생
   ↓
ReLU 계열 활성화 함수 / 적절한 가중치 초기화 / 정규화 계층
```

이번 정리에서 사용하는 주요 기호는 다음과 같다.

```text
X: 입력 데이터
Y 또는 y: 실제 정답 레이블
W: 가중치
b: 편향
logit: 출력층이 만든 클래스별 점수
n_in: 현재 층으로 들어오는 입력 뉴런의 수
n_out: 현재 층에서 나가는 출력 뉴런의 수
λ: 가중치 규제 강도를 조절하는 하이퍼파라미터
γ: 정규화된 값의 크기를 조정하는 스케일 매개변수
β: 정규화된 값을 이동시키는 시프트 매개변수
ε: 0으로 나누는 문제를 방지하기 위한 작은 값
```

---
## 1. 다층 퍼셉트론으로 숫자 필기 데이터 분류하기

앞에서는 소프트맥스 회귀를 이용해 다중 클래스 분류를 수행했다.

소프트맥스 회귀는 입력층과 출력층만 존재하는 단순한 신경망으로 볼 수 있다.

이번에는 입력층과 출력층 사이에 은닉층을 추가한 `다층 퍼셉트론(Multi-Layer Perceptron, MLP)`을 이용해 숫자 필기 데이터를 분류한다.

---

### 1.1 사이킷런 숫자 필기 데이터 소개

사이킷런(scikit-learn)의 `load_digits()`는 손으로 쓴 숫자 이미지를 제공하는 분류용 예제 데이터셋이다.

이 데이터는 MNIST와는 다른 데이터셋이다.

| 구분 | 사이킷런 digits | MNIST |
|---|---:|---:|
| 이미지 크기 | 8 × 8 | 28 × 28 |
| 입력 특성 수 | 64 | 784 |
| 픽셀값 범위 | 0~15 | 0~255 |
| 전체 샘플 수 | 1,797개 | 70,000개 |
| 클래스 | 0~9 | 0~9 |

사이킷런 digits 데이터의 특징은 다음과 같다.

```text
이미지 크기 = 8 × 8
픽셀 수 = 64
픽셀값 범위 = 0부터 15
전체 샘플 수 = 1,797개
클래스 개수 = 10개
```

데이터를 불러오는 코드는 다음과 같다.

```python
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
```

`digits` 객체에는 이미지, 벡터화된 입력 데이터, 정답 레이블 등이 함께 들어 있다.

```text
digits.images: 각 이미지를 8 × 8 행렬로 저장
digits.data: 각 이미지를 64차원 벡터로 저장
digits.target: 각 이미지의 정답 레이블 저장
```

---

### 1.2 첫 번째 이미지 행렬 확인하기

`.images[인덱스]`를 사용하면 해당 샘플의 이미지를 8 × 8 행렬로 확인할 수 있다.

```python
print(digits.images[0])
```

출력은 다음과 같다.

```text
[[ 0.  0.  5. 13.  9.  1.  0.  0.]
 [ 0.  0. 13. 15. 10. 15.  5.  0.]
 [ 0.  3. 15.  2.  0. 11.  8.  0.]
 [ 0.  4. 12.  0.  0.  8.  8.  0.]
 [ 0.  5.  8.  0.  0.  9.  8.  0.]
 [ 0.  4. 11.  0.  1. 12.  7.  0.]
 [ 0.  2. 14.  5. 10. 12.  0.  0.]
 [ 0.  0.  6. 13. 10.  0.  0.  0.]]
```

각 숫자는 해당 위치의 명암을 의미한다.

```text
0에 가까운 값: 배경에 가까운 픽셀
15에 가까운 값: 숫자 획에 가까운 픽셀
```

첫 번째 샘플의 정답 레이블은 다음과 같이 확인한다.

```python
print(digits.target[0])
```

출력은 다음과 같다.

```text
0
```

즉, 첫 번째 이미지에는 숫자 0이 들어 있다.

전체 샘플의 수는 다음과 같이 확인한다.

```python
print('전체 샘플의 수 : {}'.format(len(digits.images)))
```

출력은 다음과 같다.

```text
전체 샘플의 수 : 1797
```

---

### 1.3 상위 5개 샘플 시각화하기

이미지와 레이블을 함께 묶은 뒤 상위 5개 샘플을 시각화할 수 있다.

```python
images_and_labels = list(zip(digits.images, digits.target))

for index, (image, label) in enumerate(images_and_labels[:5]):
    plt.subplot(2, 5, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('sample: %i' % label)
```

<img width="360" height="88" alt="Image" src="https://github.com/user-attachments/assets/9746b343-6bbf-4bab-bdcb-12c6ba60bc41" />

상위 5개 샘플의 정답 레이블은 다음과 같이 확인할 수 있다.

```python
for i in range(5):
    print(i, '번 인덱스 샘플의 레이블 : ', digits.target[i])
```

출력은 다음과 같다.

```text
0 번 인덱스 샘플의 레이블 : 0
1 번 인덱스 샘플의 레이블 : 1
2 번 인덱스 샘플의 레이블 : 2
3 번 인덱스 샘플의 레이블 : 3
4 번 인덱스 샘플의 레이블 : 4
```

---

### 1.4 8 × 8 이미지를 64차원 벡터로 사용하기

`digits.images`는 각 이미지를 8 × 8 행렬로 저장한다.

하지만 완전 연결 계층인 `nn.Linear()`에 데이터를 넣으려면 하나의 이미지를 1차원 벡터로 표현하는 것이 편리하다.

`digits.data`에는 각 8 × 8 이미지가 이미 64차원 벡터로 펼쳐져 있다.

```python
print(digits.data[0])
```

출력 형태는 다음과 같다.

```text
[0. 0. 5. 13. 9. 1. ... 6. 13. 10. 0. 0. 0.]
```

크기 변환은 다음과 같다.

```text
8 × 8 이미지
   ↓ 펼치기
64차원 벡터
```

훈련에 사용할 입력과 정답을 다음과 같이 저장한다.

```python
X = digits.data
Y = digits.target
```

각 변수의 의미는 다음과 같다.

```text
X: 이미지 픽셀값을 64차원 벡터로 저장한 특성 행렬
Y: 각 이미지가 나타내는 숫자 레이블
```

전체 데이터의 크기는 다음과 같다.

```text
X.shape = (1797, 64)
Y.shape = (1797,)
```

---

### 1.5 다층 퍼셉트론 모델 구조

PyTorch를 사용하기 위해 필요한 모듈을 불러온다.

```python
import torch
import torch.nn as nn
from torch import optim
```

모델은 `nn.Sequential()`을 이용해 계층을 순서대로 연결한다.

```python
model = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 10)
)
```

전체 구조는 다음과 같다.

```text
입력층: 64
   ↓
첫 번째 선형 계층: Linear(64, 32)
   ↓
ReLU
   ↓
두 번째 선형 계층: Linear(32, 16)
   ↓
ReLU
   ↓
출력층: Linear(16, 10)
```

각 계층의 역할은 다음과 같다.

| 계층 | 역할 |
|---|---|
| `nn.Linear(64, 32)` | 64개의 입력 특성을 32개의 은닉 표현으로 변환 |
| `nn.ReLU()` | 비선형성을 추가하여 복잡한 패턴을 학습할 수 있게 함 |
| `nn.Linear(32, 16)` | 32개의 은닉 표현을 16개로 다시 변환 |
| `nn.ReLU()` | 두 번째 비선형 변환 수행 |
| `nn.Linear(16, 10)` | 숫자 0~9에 해당하는 10개의 logit 출력 |

마지막 `nn.Linear(16, 10)`은 은닉층이 아니라 `출력층`이다.

출력층 뒤에 softmax를 직접 붙이지 않는 이유는 이후 사용할 `nn.CrossEntropyLoss()`가 내부적으로 `log_softmax`를 처리하기 때문이다.

---

### 1.6 입력 데이터와 레이블을 텐서로 변환하기

사이킷런의 데이터는 NumPy 배열이므로 PyTorch 모델에 넣기 전에 텐서로 변환한다.

```python
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.int64)
```

자료형을 구분해야 하는 이유는 다음과 같다.

```text
입력 X:
실수형 픽셀값이므로 torch.float32 사용

정답 Y:
클래스 번호이므로 torch.int64 또는 torch.long 사용
```

`CrossEntropyLoss`의 정답 레이블은 원-핫 벡터가 아니라 정수형 클래스 번호여야 한다.

---

### 1.7 손실 함수와 옵티마이저 설정하기

손실 함수는 다중 클래스 분류에 사용하는 크로스 엔트로피를 사용한다.

```python
loss_fn = nn.CrossEntropyLoss()
```

옵티마이저는 Adam을 사용한다.

```python
optimizer = optim.Adam(model.parameters())
```

손실값의 변화를 기록하기 위한 리스트도 만든다.

```python
losses = []
```

각 요소의 역할은 다음과 같다.

```text
loss_fn:
모델이 출력한 logit과 실제 정답 레이블을 비교하여 손실 계산

optimizer:
역전파로 계산된 기울기를 이용해 모델의 매개변수 업데이트

losses:
에포크별 손실값을 저장하여 학습 곡선 시각화
```

---

### 1.8 전체 데이터를 이용한 학습 과정

이 예제에서는 1,797개의 전체 데이터를 한 번에 모델에 넣는 `전체 배치 학습`을 수행한다.

```python
for epoch in range(100):
    optimizer.zero_grad()

    y_pred = model(X)
    loss = loss_fn(y_pred, Y)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, 100, loss.item()
        ))

    losses.append(loss.item())
```

학습 과정은 다음과 같다.

```text
1. optimizer.zero_grad()
   이전 반복에서 남아 있는 기울기를 0으로 초기화한다.

2. y_pred = model(X)
   입력 데이터를 모델에 넣어 10개 클래스에 대한 logit을 계산한다.

3. loss = loss_fn(y_pred, Y)
   예측 logit과 실제 정답을 비교하여 손실을 계산한다.

4. loss.backward()
   역전파를 수행해 각 매개변수의 기울기를 계산한다.

5. optimizer.step()
   계산된 기울기를 이용해 가중치와 편향을 업데이트한다.
```

출력 예시는 다음과 같다.

```text
Epoch    0/100 Cost: 2.380815
Epoch   10/100 Cost: 2.059323
... 중략 ...
Epoch   90/100 Cost: 0.205398
```

손실값이 전반적으로 감소한다면 모델이 훈련 데이터의 패턴을 학습하고 있다는 뜻이다.

다만 이 코드에서는 훈련 데이터와 테스트 데이터를 분리하지 않았으므로, 이 손실만으로 새로운 데이터에 대한 일반화 성능을 판단할 수는 없다.

---

### 1.9 손실 곡선 시각화하기

저장한 손실값을 그래프로 그릴 수 있다.

```python
plt.plot(losses)
```

<img width="377" height="268" alt="Image" src="https://github.com/user-attachments/assets/8f7572dd-52ec-44d1-bd7b-10ead076d3fd" />

그래프에서 가로축은 에포크, 세로축은 손실값을 의미한다.

```text
가로축: 학습 반복 횟수
세로축: Cross Entropy Loss
```

손실이 감소한다고 해서 반드시 테스트 정확도도 계속 증가하는 것은 아니다.

훈련 손실은 감소하지만 검증 손실이 다시 증가한다면 과적합이 발생하고 있을 가능성이 있다.

---

## 2. 다층 퍼셉트론으로 MNIST 분류하기

앞에서 소프트맥스 회귀로 MNIST를 분류할 때는 입력층과 출력층만 사용했다.

이번에는 두 개의 은닉층을 추가해 다층 퍼셉트론으로 MNIST를 분류한다.

```text
소프트맥스 회귀:
784 → 10

다층 퍼셉트론:
784 → 100 → 100 → 10
```

은닉층과 ReLU 활성화 함수를 추가하면 선형 모델보다 복잡한 비선형 패턴을 학습할 수 있다.

---

### 2.1 MNIST 데이터 불러오기

필요한 라이브러리를 불러온다.

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.datasets import fetch_openml
```

OpenML에서 MNIST 데이터를 불러온다.

```python
mnist = fetch_openml(
    'mnist_784',
    version=1,
    cache=True,
    as_frame=False
)
```

각 옵션의 의미는 다음과 같다.

```text
'mnist_784': 784개의 픽셀 특성을 가진 MNIST 데이터셋
version=1: 데이터셋 버전 지정
cache=True: 한 번 다운로드한 데이터를 로컬에 저장
as_frame=False: pandas DataFrame이 아니라 NumPy 배열로 반환
```

MNIST 데이터의 크기는 다음과 같다.

```text
전체 이미지 수 = 70,000개
입력 특성 수 = 784개
클래스 수 = 10개
```

---

### 2.2 첫 번째 샘플과 레이블 확인하기

첫 번째 이미지의 픽셀 벡터는 다음과 같이 확인한다.

```python
mnist.data[0]
```

출력은 784개의 숫자로 이루어진 1차원 배열이다.

```text
28 × 28 이미지
   ↓ 펼쳐진 상태
784차원 벡터
```

첫 번째 샘플의 레이블을 확인한다.

```python
mnist.target[0]
```

출력은 다음과 같다.

```text
'5'
```

처음에는 레이블이 문자열 형태이므로 정수형으로 변환한다.

```python
mnist.target = mnist.target.astype(np.int8)
```

---

### 2.3 픽셀값 정규화하기

MNIST의 원래 픽셀값 범위는 0부터 255이다.

이를 255로 나누어 0과 1 사이로 변환한다.

```python
X = mnist.data / 255.0
y = mnist.target
```

정규화 전후의 범위는 다음과 같다.

```text
정규화 전: 0~255
정규화 후: 0~1
```

입력값의 크기를 일정한 범위로 맞추면 최적화 과정이 안정되고 학습이 더 효율적으로 진행될 수 있다.

여기서 말하는 정규화는 `입력 데이터 스케일 조정`이다.

뒤에서 다룰 `배치 정규화(Batch Normalization)`나 `층 정규화(Layer Normalization)`와는 적용 위치와 목적이 다르다.

---

### 2.4 첫 번째 MNIST 이미지 시각화하기

784차원 벡터를 다시 28 × 28 이미지로 바꿔 시각화한다.

```python
plt.imshow(X[0].reshape(28, 28), cmap='gray')
print('이 이미지 데이터의 레이블은 {:.0f}이다'.format(y[0]))
```

<img width="369" height="371" alt="Image" src="https://github.com/user-attachments/assets/f4c39a37-aff5-49cf-bcdd-f32819ef07b2" />

`reshape(28, 28)`은 784개의 원소를 다시 28행 28열의 이미지 형태로 바꾸는 연산이다.

---

### 2.5 훈련 데이터와 테스트 데이터 분리하기

필요한 모듈을 불러온다.

```python
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
```

전체 데이터를 훈련 세트와 테스트 세트로 분리한다.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=1/7,
    random_state=0
)
```

MNIST 전체 데이터는 70,000개이므로 `test_size=1/7`을 사용하면 대략 다음과 같이 나뉜다.

```text
훈련 데이터 = 60,000개
테스트 데이터 = 10,000개
```

`random_state=0`은 데이터 분할 결과를 재현할 수 있도록 난수 시드를 고정한다.

---

### 2.6 NumPy 배열을 PyTorch 텐서로 변환하기

분리한 데이터를 텐서로 변환한다.

```python
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)
```

입력은 실수형, 정답은 정수형 클래스 번호로 저장한다.

```text
X_train.shape = [60000, 784]
y_train.shape = [60000]
X_test.shape = [10000, 784]
y_test.shape = [10000]
```

---

### 2.7 TensorDataset과 DataLoader 만들기

입력과 정답을 하나의 데이터셋으로 묶는다.

```python
ds_train = TensorDataset(X_train, y_train)
ds_test = TensorDataset(X_test, y_test)
```

그 다음 미니 배치 단위로 데이터를 꺼낼 수 있도록 `DataLoader`를 만든다.

```python
loader_train = DataLoader(
    ds_train,
    batch_size=64,
    shuffle=True
)

loader_test = DataLoader(
    ds_test,
    batch_size=64,
    shuffle=False
)
```

각 설정의 의미는 다음과 같다.

```text
batch_size=64:
한 번의 매개변수 업데이트에 64개 샘플 사용

shuffle=True:
훈련할 때 매 에포크마다 데이터 순서를 섞음

shuffle=False:
평가할 때는 순서를 섞을 필요가 없음
```

전체 데이터를 한 번에 학습하지 않고 작은 묶음으로 나누어 학습하는 방식을 `미니 배치 학습(Mini-batch Training)`이라고 한다.

---

### 2.8 MNIST 다층 퍼셉트론 구조 만들기

필요한 모듈을 불러온다.

```python
from torch import nn
from torch import optim
```

모델을 정의한다.

```python
model = nn.Sequential()
model.add_module('fc1', nn.Linear(28 * 28, 100))
model.add_module('relu1', nn.ReLU())
model.add_module('fc2', nn.Linear(100, 100))
model.add_module('relu2', nn.ReLU())
model.add_module('fc3', nn.Linear(100, 10))
```

모델 구조를 출력하면 다음과 같다.

```python
print(model)
```

```text
Sequential(
  (fc1): Linear(in_features=784, out_features=100, bias=True)
  (relu1): ReLU()
  (fc2): Linear(in_features=100, out_features=100, bias=True)
  (relu2): ReLU()
  (fc3): Linear(in_features=100, out_features=10, bias=True)
)
```

전체 구조는 다음과 같다.

```text
입력: 784
   ↓
fc1: 784 → 100
   ↓
ReLU
   ↓
fc2: 100 → 100
   ↓
ReLU
   ↓
fc3: 100 → 10
```

마지막 출력 10개는 숫자 0부터 9까지 각 클래스에 대한 logit이다.

---

### 2.9 손실 함수와 옵티마이저 설정하기

손실 함수는 크로스 엔트로피를 사용한다.

```python
loss_fn = nn.CrossEntropyLoss()
```

옵티마이저는 Adam을 사용한다.

```python
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

`CrossEntropyLoss`에는 softmax가 포함되어 있으므로 모델의 마지막에 softmax를 추가하지 않는다.

```text
모델 출력: logit
   ↓
CrossEntropyLoss 내부에서 log_softmax 수행
   ↓
정답 레이블과 비교하여 손실 계산
```

---

### 2.10 미니 배치 단위로 모델 학습하기

총 3번의 에포크 동안 모델을 학습한다.

```python
epochs = 3

for epoch in range(epochs):
    for data, targets in loader_train:
        optimizer.zero_grad()

        y_pred = model(data)
        loss = loss_fn(y_pred, targets)

        loss.backward()
        optimizer.step()

    print('Epoch {:4d}/{} Cost: {:.6f}'.format(
        epoch + 1,
        epochs,
        loss.item()
    ))
```

학습 흐름은 다음과 같다.

```text
1. loader_train에서 64개 샘플을 가져온다.
2. 모델이 64개 샘플 각각에 대해 10개의 logit을 출력한다.
3. CrossEntropyLoss로 미니 배치의 손실을 계산한다.
4. backward()로 기울기를 계산한다.
5. optimizer.step()으로 매개변수를 업데이트한다.
6. 모든 미니 배치를 사용하면 한 에포크가 끝난다.
```

출력 예시는 다음과 같다.

```text
Epoch    1/3 Cost: 0.364392
Epoch    2/3 Cost: 0.048837
Epoch    3/3 Cost: 0.231157
```

여기서 출력한 손실은 각 에포크의 마지막 미니 배치에서 계산된 값이다.

에포크 전체의 평균 손실을 보고 싶다면 각 미니 배치의 손실을 누적한 뒤 평균을 계산해야 한다.

---

### 2.11 테스트 데이터로 정확도 평가하기

학습이 끝난 모델을 평가 모드로 전환한다.

```python
model.eval()
```

그 다음 기울기 계산을 비활성화한 상태에서 테스트 데이터를 예측한다.

```python
correct = 0

with torch.no_grad():
    for data, targets in loader_test:
        outputs = model(data)

        predicted = torch.argmax(outputs, dim=1)
        correct += predicted.eq(targets).sum().item()
```

전체 테스트 데이터 수와 정확도를 계산한다.

```python
data_num = len(loader_test.dataset)
accuracy = 100.0 * correct / data_num

print('\n테스트 데이터에서 예측 정확도: {}/{} ({:.0f}%)\n'.format(
    correct,
    data_num,
    accuracy
))
```

출력 예시는 다음과 같다.

```text
테스트 데이터에서 예측 정확도: 9566/10000 (96%)
```

각 코드의 의미는 다음과 같다.

```text
model.eval():
Dropout과 Batch Normalization처럼 학습과 평가에서 다르게 동작하는 계층을 평가 모드로 전환

with torch.no_grad():
기울기 추적을 중단하여 메모리와 연산량 절약

torch.argmax(outputs, dim=1):
각 샘플에서 가장 큰 logit을 가진 클래스 인덱스 선택

predicted.eq(targets):
예측 레이블과 실제 레이블이 같은지 비교
```

`outputs`는 확률이 아니라 logit이다.

하지만 softmax는 값의 순서를 바꾸지 않으므로, 최종 클래스만 구할 때는 softmax를 적용하지 않고 logit에서 바로 `argmax`를 사용해도 결과가 같다.

---

### 2.12 임의의 테스트 이미지 예측하기

테스트 데이터에서 하나의 샘플을 선택해 예측한다.

```python
index = 2018

model.eval()

with torch.no_grad():
    data = X_test[index]
    output = model(data)
    predicted = torch.argmax(output, dim=0)

print('예측 결과 : {}'.format(predicted.item()))

X_test_show = X_test[index].numpy()
plt.imshow(X_test_show.reshape(28, 28), cmap='gray')
print('이 이미지 데이터의 정답 레이블은 {:.0f}입니다'.format(
    y_test[index].item()
))
```

<img width="411" height="404" alt="Image" src="https://github.com/user-attachments/assets/63f2e7c4-ae06-4eed-b4a7-0fcad73d929f" />

하나의 샘플을 모델에 넣었으므로 출력의 크기는 다음과 같다.

```text
output.shape = [10]
```

따라서 클래스 방향인 `dim=0`에서 가장 큰 값을 찾는다.

미니 배치를 모델에 넣을 때는 출력 크기가 `[배치 크기, 10]`이므로 클래스 방향인 `dim=1`에서 가장 큰 값을 찾는다.

---

## 3. 과적합(Overfitting)을 막는 방법

`과적합(Overfitting)`은 모델이 훈련 데이터에 지나치게 맞춰져 새로운 데이터에서 성능이 떨어지는 현상이다.

과적합된 모델은 훈련 데이터의 일반적인 규칙뿐 아니라 우연히 포함된 노이즈와 세부 패턴까지 암기한다.

```text
훈련 데이터 성능: 매우 높음
검증·테스트 데이터 성능: 상대적으로 낮음
```

모델의 목적은 훈련 데이터를 완벽히 외우는 것이 아니라, 보지 못한 데이터에도 잘 적용되는 일반적인 패턴을 학습하는 것이다.

---

### 3.1 훈련 오차와 일반화 오차

과적합을 이해하려면 훈련 데이터와 평가 데이터의 역할을 구분해야 한다.

| 데이터 | 역할 |
|---|---|
| 훈련 데이터 | 모델의 가중치와 편향을 학습 |
| 검증 데이터 | 하이퍼파라미터 선택과 과적합 여부 확인 |
| 테스트 데이터 | 최종 모델의 일반화 성능 평가 |

과적합이 진행되면 보통 다음과 같은 현상이 나타난다.

```text
훈련 손실: 계속 감소
검증 손실: 어느 시점 이후 다시 증가
```

즉, 훈련 손실만 확인해서는 과적합 여부를 판단하기 어렵다.

---

### 3.2 데이터의 양 늘리기

데이터가 적으면 모델이 개별 샘플의 우연한 패턴과 노이즈를 쉽게 암기할 수 있다.

데이터의 양이 많아지면 여러 샘플에 공통으로 나타나는 일반적인 패턴을 학습할 가능성이 높아진다.

```text
데이터가 적음
   ↓
개별 샘플 암기 가능성 증가
   ↓
과적합 위험 증가

데이터가 많음
   ↓
공통 패턴 학습 가능성 증가
   ↓
일반화 성능 향상 가능
```

---

### 3.3 데이터 증강(Data Augmentation)

새로운 데이터를 직접 수집하기 어려울 때는 기존 데이터를 변형하여 학습 샘플을 늘릴 수 있다.

이를 `데이터 증강(Data Augmentation)`이라고 한다.

이미지에서는 다음과 같은 변형을 사용할 수 있다.

```text
이미지 회전
좌우 또는 상하 이동
일부 영역 자르기
크기 조절
밝기와 대비 조절
작은 노이즈 추가
```

데이터 증강은 단순히 같은 이미지를 복사하는 것이 아니라, 정답은 유지하면서 입력 모양을 조금씩 바꾸는 방법이다.

예를 들어 숫자 5 이미지를 약간 이동하거나 기울여도 여전히 정답은 숫자 5이다.

---

### 3.4 모델의 복잡도 줄이기

인공 신경망의 복잡도는 다음 요소들에 의해 커진다.

```text
은닉층의 수
각 은닉층의 뉴런 수
전체 가중치와 편향의 수
```

모델이 학습할 수 있는 복잡성의 정도를 `수용력(Capacity)`이라고 부르기도 한다.

수용력이 지나치게 크면 훈련 데이터를 세밀하게 암기할 수 있어 과적합 위험이 높아진다.

예를 들어 다음 모델은 세 개의 선형 계층을 가진다.

```python
class Architecture1(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out
```

과적합이 발생한다면 은닉층 하나를 제거해 모델을 단순하게 만들 수 있다.

```python
class Architecture2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
```

구조를 비교하면 다음과 같다.

```text
복잡한 모델:
입력 → 은닉층 → 은닉층 → 출력층

단순화한 모델:
입력 → 은닉층 → 출력층
```

모델을 무조건 작게 만드는 것이 정답은 아니다.

모델이 너무 단순하면 훈련 데이터의 패턴조차 충분히 학습하지 못하는 `과소적합(Underfitting)`이 발생할 수 있다.

---

### 3.5 가중치 규제(Weight Regularization)

가중치 규제는 기존 손실 함수에 가중치 크기에 대한 벌점 항을 추가하는 방법이다.

모델이 손실을 줄이는 동시에 가중치가 지나치게 커지지 않도록 제한한다.

```text
최종 손실
= 데이터에 대한 원래 손실
+ 가중치 크기에 대한 벌점
```

가중치가 매우 크면 입력의 작은 변화에도 출력이 크게 변할 수 있다.

가중치의 크기를 제한하면 모델이 일부 특성에 지나치게 의존하는 것을 줄이고 더 단순한 결정 규칙을 학습하도록 유도할 수 있다.

---

### 3.6 L1 규제

`L1 규제(L1 Regularization)`는 모든 가중치 절댓값의 합을 손실 함수에 더한다.

$$
L_{\mathrm{total}} = L_{\mathrm{data}} + \lambda \sum_i \lvert w_i \rvert
$$

각 기호의 의미는 다음과 같다.

```text
L_data: 원래의 데이터 손실
w_i: i번째 가중치
λ: 규제 강도
```

L1 규제는 일부 가중치를 정확히 0에 가깝게 만드는 경향이 있다.

따라서 중요하지 않은 특성의 가중치가 제거되는 `희소성(Sparsity)`을 만들 수 있다.

예를 들어 다음 가설이 있다고 하자.

$$
H(x) = w_1x_1 + w_2x_2 + w_3x_3 + w_4x_4
$$

L1 규제를 적용한 결과 $w_3$가 0에 가까워지면 $x_3$ 특성은 예측에 거의 사용되지 않는다.

---

### 3.7 L2 규제

`L2 규제(L2 Regularization)`는 모든 가중치 제곱의 합을 손실 함수에 더한다.

$$
L_{\mathrm{total}} = L_{\mathrm{data}} + \lambda \sum_i w_i^2
$$

교재나 구현에 따라 다음처럼 $\frac{1}{2}$가 포함되기도 한다.

$$
L_{\mathrm{total}} = L_{\mathrm{data}} + \frac{\lambda}{2}\sum_i w_i^2
$$

두 식은 규제 강도 $\lambda$의 정의가 달라질 뿐 핵심 의미는 같다.

L2 규제는 큰 가중치에 더 큰 벌점을 주므로, 가중치들이 전체적으로 작은 값을 갖도록 만든다.

L1 규제와 달리 많은 가중치를 정확히 0으로 만들기보다는 0에 가까운 작은 값으로 줄이는 경향이 있다.

| 구분 | L1 규제 | L2 규제 |
|---|---|---|
| 벌점 | $\sum_i \lvert w_i \rvert$ | $\sum w_i^2$ |
| 대표 효과 | 일부 가중치를 0에 가깝게 만듦 | 전체 가중치를 부드럽게 작게 만듦 |
| 특징 선택 | 비교적 강함 | 비교적 약함 |
| 다른 표현 | L1 norm penalty | weight decay와 관련 |

---

### 3.8 PyTorch에서 weight_decay 사용하기

PyTorch 옵티마이저의 `weight_decay` 매개변수를 이용하면 L2 형태의 가중치 감소를 적용할 수 있다.

```python
model = Architecture2(10, 20, 2)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)
```

각 값의 의미는 다음과 같다.

```text
lr=1e-4:
학습률

weight_decay=1e-5:
가중치 감소 강도
```

`weight_decay`의 기본값은 0이므로 값을 지정하지 않으면 가중치 감소가 적용되지 않는다.

일반적으로 `weight_decay`가 너무 크면 모델이 필요한 패턴까지 충분히 학습하지 못할 수 있고, 너무 작으면 과적합 방지 효과가 거의 없을 수 있다.

Adam 계열에서 가중치 감소를 명확히 분리해 적용하고 싶을 때는 `AdamW`를 사용하는 경우도 많다.

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)
```

---

### 3.9 드롭아웃(Dropout)

`드롭아웃(Dropout)`은 학습할 때 일부 뉴런의 출력을 무작위로 0으로 만드는 방법이다.

예를 들어 드롭아웃 비율이 0.5라면 학습 반복마다 대략 절반의 뉴런 출력을 무작위로 사용하지 않는다.

```text
원래 신경망
   ↓
학습 반복마다 일부 뉴런 무작위 제거
   ↓
남은 뉴런으로 순전파와 역전파 수행
```

<img width="705" height="277" alt="Image" src="https://github.com/user-attachments/assets/5d5859e3-4a0a-416a-addc-3345a23fa6a0" />

드롭아웃의 효과는 다음과 같다.

```text
특정 뉴런 하나에 대한 의존 감소
특정 뉴런 조합에 대한 의존 감소
여러 다른 부분 신경망을 학습하는 것과 비슷한 효과
과적합 완화
```

PyTorch에서는 다음과 같이 적용할 수 있다.

```python
model = nn.Sequential(
    nn.Linear(784, 100),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(100, 10)
)
```

`p=0.5`는 각 뉴런 출력이 0이 될 확률을 의미한다.

---

### 3.10 드롭아웃의 학습 모드와 평가 모드

드롭아웃은 학습할 때만 무작위로 뉴런을 제거한다.

평가할 때는 모든 뉴런을 사용해야 한다.

```python
model.train()
```

`model.train()`은 드롭아웃을 학습 모드로 설정한다.

```python
model.eval()
```

`model.eval()`은 드롭아웃을 평가 모드로 설정하여 무작위 제거를 중단한다.

```text
학습 단계:
일부 뉴런 출력 무작위 제거

평가 단계:
모든 뉴런 사용
```

PyTorch의 드롭아웃은 학습 시 남아 있는 출력값을 자동으로 보정하므로 평가 단계에서 별도의 수동 배율 조정이 필요하지 않다.

---

### 3.11 과적합 방지 방법 비교

| 방법 | 핵심 아이디어 | 주의점 |
|---|---|---|
| 데이터 증가 | 더 다양한 실제 샘플 수집 | 비용과 시간이 많이 들 수 있음 |
| 데이터 증강 | 기존 데이터를 의미가 유지되도록 변형 | 잘못된 변형은 정답 의미를 바꿀 수 있음 |
| 모델 단순화 | 층 또는 뉴런 수 감소 | 지나치면 과소적합 발생 |
| L1 규제 | 가중치 절댓값에 벌점 | 일부 가중치가 0에 가까워짐 |
| L2 규제 | 가중치 제곱에 벌점 | 규제 강도 조절 필요 |
| 드롭아웃 | 학습 중 일부 뉴런 무작위 제거 | 평가 전 `model.eval()` 필요 |

---
## 4. 기울기 소실(Gradient Vanishing)과 기울기 폭주(Gradient Exploding)

깊은 신경망은 여러 층의 연산을 순서대로 수행한다.

학습할 때는 출력층에서 계산한 손실을 기준으로 역전파를 수행하여 각 층의 기울기를 계산한다.

이 과정에서 기울기가 지나치게 작아지거나 지나치게 커질 수 있다.

```text
기울기가 계속 작아짐 → 기울기 소실
기울기가 계속 커짐 → 기울기 폭주
```

---

### 4.1 역전파와 기울기의 곱

역전파에서는 연쇄 법칙(Chain Rule)에 따라 여러 미분값을 곱한다.

간단히 표현하면 앞쪽 층의 기울기는 다음과 같은 곱으로 계산된다.

$$
\frac{\partial L}{\partial W_1}
=
\frac{\partial L}{\partial a_n}
\frac{\partial a_n}{\partial a_{n-1}}
\cdots
\frac{\partial a_2}{\partial a_1}
\frac{\partial a_1}{\partial W_1}
$$

여러 층을 거치며 작은 값이 계속 곱해지면 기울기가 0에 가까워질 수 있다.

반대로 1보다 큰 값이 반복해서 곱해지면 기울기가 매우 커질 수 있다.

---

### 4.2 기울기 소실

`기울기 소실(Gradient Vanishing)`은 역전파가 입력층 방향으로 진행될수록 기울기가 점점 작아지는 현상이다.

```text
출력층 근처:
기울기가 비교적 정상적으로 전달됨

입력층 근처:
기울기가 0에 가까워짐
```

입력층에 가까운 가중치의 기울기가 거의 0이 되면 다음과 같은 문제가 생긴다.

```text
가중치가 거의 업데이트되지 않음
앞쪽 층의 학습 속도가 매우 느려짐
깊은 층을 쌓아도 충분한 표현을 학습하지 못함
최적의 모델을 찾기 어려움
```

---

### 4.3 시그모이드 함수와 기울기 소실

시그모이드 함수는 다음과 같다.

$$
\sigma(x) = \frac{1}{1+e^{-x}}
$$

미분값은 다음과 같다.

$$
\sigma'(x)=\sigma(x)(1-\sigma(x))
$$

시그모이드의 미분값은 최대가 0.25이며, 입력의 절댓값이 커지면 0에 가까워진다.

```text
입력이 큰 양수:
출력은 1에 가까워지고 기울기는 0에 가까워짐

입력이 큰 음수:
출력은 0에 가까워지고 기울기는 0에 가까워짐
```

깊은 신경망에서 이런 작은 기울기가 반복해서 곱해지면 앞쪽 층으로 전달되는 기울기가 빠르게 사라질 수 있다.

하이퍼볼릭 탄젠트(tanh)도 시그모이드보다 중심이 0에 맞춰져 있다는 장점은 있지만, 입력의 절댓값이 클 때 포화되어 기울기가 작아질 수 있다.

---

### 4.4 기울기 폭주

`기울기 폭주(Gradient Exploding)`는 역전파 과정에서 기울기가 점점 커지는 현상이다.

```text
기울기 값 급격히 증가
   ↓
가중치가 비정상적으로 크게 업데이트됨
   ↓
손실값이 급격히 커지거나 발산
   ↓
NaN 또는 inf 발생 가능
```

기울기 폭주는 깊은 신경망뿐 아니라 긴 시퀀스를 처리하는 순환 신경망(Recurrent Neural Network, RNN)에서도 발생할 수 있다.

대표적인 완화 방법은 다음과 같다.

```text
적절한 가중치 초기화
정규화 계층 사용
학습률 조절
기울기 클리핑
```

기울기 클리핑은 기울기의 크기가 일정 기준을 넘으면 잘라내는 방법이다.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

보통 `loss.backward()` 이후, `optimizer.step()` 이전에 사용한다.

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

---

### 4.5 ReLU로 기울기 소실 완화하기

ReLU 함수는 다음과 같다.

$$
\mathrm{ReLU}(x)=\max(0,x)
$$

입력이 양수일 때 ReLU의 기울기는 1이다.

```text
x > 0:
출력 = x
기울기 = 1

x < 0:
출력 = 0
기울기 = 0
```

양수 영역에서 기울기가 1이므로 시그모이드처럼 모든 영역에서 작은 기울기를 반복해서 곱하는 문제를 줄일 수 있다.

따라서 은닉층에서는 시그모이드나 tanh 대신 ReLU 계열 함수를 사용하는 경우가 많다.

---

### 4.6 죽은 ReLU 문제

ReLU는 입력이 음수이면 출력과 기울기가 모두 0이다.

어떤 뉴런의 입력이 계속 음수가 되면 해당 뉴런은 이후에도 거의 업데이트되지 않을 수 있다.

이를 `죽은 ReLU(Dying ReLU)` 문제라고 한다.

```text
입력이 계속 음수
   ↓
출력 0
   ↓
기울기 0
   ↓
가중치 업데이트 중단
```

---

### 4.7 Leaky ReLU

Leaky ReLU는 음수 입력에서도 작은 기울기를 남겨 둔다.

$$
\mathrm{LeakyReLU}(x)=
\begin{cases}
x, & x \ge 0 \\
\alpha x, & x < 0
\end{cases}
$$

여기서 $\alpha$는 보통 0.01처럼 작은 양수이다.

```text
양수 입력:
기울기 = 1

음수 입력:
기울기 = α
```

PyTorch에서는 다음과 같이 사용한다.

```python
nn.LeakyReLU(negative_slope=0.01)
```

Leaky ReLU는 음수 영역에서도 기울기가 완전히 0이 되지 않으므로 죽은 ReLU 문제를 완화할 수 있다.

---

### 4.8 활성화 함수 선택 정리

```text
출력층이 이진 분류:
sigmoid를 사용할 수 있음

출력층이 다중 클래스 분류:
logit을 출력하고 CrossEntropyLoss 사용

은닉층:
ReLU 또는 Leaky ReLU와 같은 ReLU 계열을 주로 사용
```

은닉층에서 시그모이드를 절대 사용할 수 없는 것은 아니다.

다만 깊은 신경망에서는 기울기 소실 위험 때문에 ReLU 계열이 일반적으로 더 자주 사용된다.

---

## 5. 가중치 초기화(Weight Initialization)

신경망을 학습하기 전에는 각 층의 가중치에 초기값을 설정해야 한다.

같은 구조와 같은 데이터로 학습하더라도 초기 가중치에 따라 학습 속도와 최종 성능이 달라질 수 있다.

적절한 가중치 초기화의 목적은 다음과 같다.

```text
각 층의 출력값이 지나치게 작거나 커지는 것을 방지
역전파 기울기가 지나치게 작거나 커지는 것을 방지
뉴런들이 서로 다른 특징을 학습하도록 대칭성 깨기
학습을 빠르고 안정적으로 시작
```

---

### 5.1 모든 가중치를 0으로 초기화하면 안 되는 이유

같은 층의 모든 가중치를 0으로 초기화하면 모든 뉴런이 같은 출력을 만들고 같은 기울기를 받는다.

```text
같은 초기값
   ↓
같은 출력
   ↓
같은 기울기
   ↓
계속 같은 가중치
```

결과적으로 여러 뉴런이 있어도 모두 같은 특징만 학습한다.

따라서 가중치는 일반적으로 무작위 값으로 초기화하되, 분산이 너무 크거나 작지 않도록 조절한다.

편향은 0으로 초기화해도 대칭성 문제가 발생하지 않는 경우가 많다.

---

### 5.2 입력 뉴런 수와 출력 뉴런 수

가중치 초기화에서 자주 사용하는 기호 `n_in`과 `n_out`은 다음과 같다.

```text
n_in 또는 fan_in:
현재 층으로 들어오는 입력 뉴런의 수

n_out 또는 fan_out:
현재 층에서 나가는 출력 뉴런의 수
```

예를 들어 다음 계층이 있다고 하자.

```python
nn.Linear(784, 100)
```

이 경우 다음과 같다.

```text
n_in = 784
n_out = 100
```

---

### 5.3 세이비어 초기화(Xavier Initialization)

`세이비어 초기화(Xavier Initialization)`는 `글로럿 초기화(Glorot Initialization)`라고도 한다.

이 방법은 앞 층에서 들어오는 뉴런 수와 다음 층으로 나가는 뉴런 수를 함께 고려하여 가중치 분산을 결정한다.

시그모이드나 tanh와 같이 S자 형태의 활성화 함수와 함께 사용할 때 비교적 잘 동작한다.

---

### 5.4 Xavier 균등 분포 초기화

균등 분포를 사용할 때의 범위는 다음과 같다.

$$
W \sim \mathrm{Uniform}
\left(
-\sqrt{\frac{6}{n_{in}+n_{out}}},
+\sqrt{\frac{6}{n_{in}+n_{out}}}
\right)
$$

다음과 같이 둘 수 있다.

$$
m=\sqrt{\frac{6}{n_{in}+n_{out}}}
$$

그러면 가중치는 $-m$과 $+m$ 사이의 균등 분포에서 뽑는다.

```text
W ~ Uniform(-m, +m)
```

---

### 5.5 Xavier 정규 분포 초기화

정규 분포를 사용할 때 평균은 0이고 표준편차는 다음과 같다.

$$
\sigma=\sqrt{\frac{2}{n_{in}+n_{out}}}
$$

따라서 다음 정규 분포에서 가중치를 뽑는다.

$$
W \sim \mathcal{N}
\left(
0,
\frac{2}{n_{in}+n_{out}}
\right)
$$

여기서 두 번째 값은 분산이다.

---

### 5.6 Xavier 초기화의 목적

Xavier 초기화는 각 층의 입력과 출력 분산 사이의 균형을 맞추는 것을 목표로 한다.

```text
가중치 분산이 너무 작음:
층을 지날수록 출력과 기울기가 작아질 수 있음

가중치 분산이 너무 큼:
층을 지날수록 출력과 기울기가 커질 수 있음

적절한 분산:
여러 층을 지나도 값의 크기를 비교적 안정적으로 유지
```

다만 ReLU는 음수 입력을 0으로 만들기 때문에 Xavier 초기화보다 ReLU의 특성을 고려한 He 초기화가 더 잘 맞는 경우가 많다.

---

### 5.7 He 초기화(He Initialization)

`He 초기화(He Initialization)`는 ReLU 계열 활성화 함수에 적합하도록 설계된 초기화 방법이다.

He 초기화는 Xavier 초기화와 달리 기본적으로 $n_{out}$을 사용하지 않고 $n_{in}$을 중심으로 분산을 정한다.

---

### 5.8 He 균등 분포 초기화

균등 분포를 사용할 때의 범위는 다음과 같다.

$$
W \sim \mathrm{Uniform}
\left(
-\sqrt{\frac{6}{n_{in}}},
+\sqrt{\frac{6}{n_{in}}}
\right)
$$

---

### 5.9 He 정규 분포 초기화

정규 분포를 사용할 때 표준편차는 다음과 같다.

$$
\sigma=\sqrt{\frac{2}{n_{in}}}
$$

따라서 가중치는 다음 분포에서 뽑는다.

$$
W \sim \mathcal{N}
\left(
0,
\frac{2}{n_{in}}
\right)
$$

ReLU는 음수 영역을 0으로 만들기 때문에 활성값의 일부가 사라진다.

He 초기화는 이를 고려해 Xavier보다 비교적 큰 분산을 사용한다.

---

### 5.10 Xavier 초기화와 He 초기화 비교

| 구분 | Xavier 초기화 | He 초기화 |
|---|---|---|
| 다른 이름 | Glorot 초기화 | Kaiming 초기화 |
| 주로 고려하는 값 | $n_{in}$과 $n_{out}$ | 주로 $n_{in}$ |
| 잘 맞는 활성화 함수 | sigmoid, tanh | ReLU, Leaky ReLU |
| 정규 분포 표준편차 | $\sqrt{2/(n_{in}+n_{out})}$ | $\sqrt{2/n_{in}}$ |

일반적인 선택은 다음과 같다.

```text
시그모이드 또는 tanh 계열:
Xavier 초기화

ReLU 또는 Leaky ReLU 계열:
He 초기화
```

---

### 5.11 PyTorch에서 Xavier 초기화 적용하기

PyTorch에서는 `torch.nn.init` 함수를 사용할 수 있다.

```python
layer = nn.Linear(100, 50)

nn.init.xavier_uniform_(layer.weight)
nn.init.zeros_(layer.bias)
```

정규 분포 방식은 다음과 같다.

```python
nn.init.xavier_normal_(layer.weight)
```

---

### 5.12 PyTorch에서 He 초기화 적용하기

He 초기화는 PyTorch에서 Kaiming 초기화라는 이름으로 제공된다.

```python
layer = nn.Linear(100, 50)

nn.init.kaiming_uniform_(
    layer.weight,
    nonlinearity='relu'
)
nn.init.zeros_(layer.bias)
```

정규 분포 방식은 다음과 같다.

```python
nn.init.kaiming_normal_(
    layer.weight,
    nonlinearity='relu'
)
```

Leaky ReLU를 사용할 때는 음수 기울기 값도 전달할 수 있다.

```python
nn.init.kaiming_normal_(
    layer.weight,
    a=0.01,
    nonlinearity='leaky_relu'
)
```

---

## 6. 배치 정규화(Batch Normalization)

ReLU 계열 활성화 함수와 적절한 가중치 초기화는 기울기 소실과 폭주를 완화한다.

하지만 학습 중 가중치가 계속 바뀌므로 각 층에 전달되는 값의 분포도 계속 달라질 수 있다.

`배치 정규화(Batch Normalization)`는 미니 배치의 통계를 이용해 각 특성의 값을 정규화하고, 학습 가능한 스케일과 시프트를 적용하는 방법이다.

---

### 6.1 내부 공변량 변화(Internal Covariate Shift)

배치 정규화를 제안한 초기 설명에서는 학습 중 각 층의 입력 분포가 달라지는 현상을 `내부 공변량 변화(Internal Covariate Shift)`라고 불렀다.

```text
이전 층의 가중치 업데이트
   ↓
이전 층의 출력 분포 변화
   ↓
현재 층이 받는 입력 분포 변화
   ↓
현재 층이 계속 새로운 분포에 적응해야 함
```

일반적인 `공변량 변화(Covariate Shift)`와 내부 공변량 변화는 구분해야 한다.

```text
공변량 변화:
훈련 데이터와 테스트 데이터의 입력 분포가 다름

내부 공변량 변화:
학습 중 신경망 내부 층이 받는 입력 분포가 달라짐
```

배치 정규화가 효과적인 이유를 내부 공변량 변화 감소만으로 설명하기는 어렵다는 후속 연구도 있다.

현재는 손실 함수의 최적화 지형을 더 부드럽게 만들고 학습을 안정화하는 효과도 중요한 설명으로 본다.

---

### 6.2 배치 정규화의 기본 위치

배치 정규화는 일반적으로 선형 연산이나 합성곱 연산 뒤, 활성화 함수 앞에 배치한다.

```text
Linear 또는 Convolution
   ↓
Batch Normalization
   ↓
Activation Function
```

예를 들면 다음과 같다.

```python
model = nn.Sequential(
    nn.Linear(784, 100),
    nn.BatchNorm1d(100),
    nn.ReLU(),
    nn.Linear(100, 10)
)
```

구조와 연구 방식에 따라 활성화 함수 뒤에 배치하는 경우도 있지만, 기본적인 설명에서는 활성화 함수 전에 적용하는 형태를 많이 사용한다.

---

### 6.3 미니 배치 평균 계산

하나의 미니 배치를 다음과 같이 둔다.

$$
B=\{x^{(1)},x^{(2)},\ldots,x^{(m)}\}
$$

여기서 $m$은 미니 배치에 포함된 샘플 수이다.

미니 배치 평균은 다음과 같다.

$$
\mu_B=\frac{1}{m}\sum_{i=1}^{m}x^{(i)}
$$

배치 정규화는 일반적으로 각 특성마다 독립적으로 평균을 계산한다.

예를 들어 입력 모양이 다음과 같다고 하자.

```text
[배치 크기, 특성 수] = [64, 100]
```

그러면 100개의 각 특성에 대해 64개 샘플 방향으로 평균을 계산한다.

---

### 6.4 미니 배치 분산 계산

미니 배치 분산은 다음과 같다.

$$
\sigma_B^2
=
\frac{1}{m}
\sum_{i=1}^{m}
\left(x^{(i)}-\mu_B\right)^2
$$

분산은 각 값이 평균에서 얼마나 떨어져 있는지를 나타낸다.

배치 정규화의 훈련 계산에서는 보통 위와 같이 $m$으로 나누는 형태를 사용한다.

---

### 6.5 정규화 수행하기

평균을 빼고 표준편차로 나누어 정규화한다.

$$
\hat{x}^{(i)}
=
\frac{x^{(i)}-\mu_B}
{\sqrt{\sigma_B^2+\epsilon}}
$$

각 기호의 의미는 다음과 같다.

```text
x^(i): i번째 입력값
μ_B: 미니 배치 평균
σ_B²: 미니 배치 분산
ε: 0으로 나누는 문제를 막기 위한 작은 값
x̂^(i): 정규화된 값
```

$\epsilon$은 분산이 0이거나 매우 작을 때 분모가 0에 가까워지는 문제를 방지한다.

일반적으로 $10^{-5}$ 정도의 작은 값이 사용된다.

---

### 6.6 스케일과 시프트

정규화만 수행하면 각 층의 표현력이 제한될 수 있다.

따라서 학습 가능한 매개변수 $\gamma$와 $\beta$를 사용해 스케일과 위치를 다시 조절한다.

$$
y^{(i)}
=
\gamma\hat{x}^{(i)}+\beta
$$

전체 연산은 다음과 같이 나타낼 수 있다.

$$
y^{(i)}
=
BN_{\gamma,\beta}(x^{(i)})
$$

각 매개변수의 역할은 다음과 같다.

```text
γ:
정규화된 값을 확대하거나 축소하는 스케일 매개변수

β:
정규화된 값을 이동시키는 시프트 매개변수
```

$\gamma$와 $\beta$는 역전파를 통해 다른 가중치와 함께 학습된다.

---

### 6.7 배치 정규화 전체 수식

배치 정규화의 전체 흐름은 다음과 같다.

$$
\mu_B=\frac{1}{m}\sum_{i=1}^{m}x^{(i)}
$$

$$
\sigma_B^2
=
\frac{1}{m}
\sum_{i=1}^{m}
\left(x^{(i)}-\mu_B\right)^2
$$

$$
\hat{x}^{(i)}
=
\frac{x^{(i)}-\mu_B}
{\sqrt{\sigma_B^2+\epsilon}}
$$

$$
y^{(i)}
=
\gamma\hat{x}^{(i)}+\beta
$$

```text
입력
   ↓
미니 배치 평균 계산
   ↓
미니 배치 분산 계산
   ↓
평균 0, 분산 약 1이 되도록 정규화
   ↓
γ로 스케일 조정
   ↓
β로 위치 이동
   ↓
최종 출력
```

---

### 6.8 배치 정규화가 계산되는 방향

배치 크기가 3이고 특성이 4개라면 입력 행렬은 다음처럼 생각할 수 있다.

```text
샘플 1: [x11, x12, x13, x14]
샘플 2: [x21, x22, x23, x24]
샘플 3: [x31, x32, x33, x34]
```

배치 정규화는 같은 특성 위치에 있는 값들을 샘플 방향으로 묶는다.

```text
feature 1: x11, x21, x31
feature 2: x12, x22, x32
feature 3: x13, x23, x33
feature 4: x14, x24, x34
```

<img width="409" height="223" alt="Image" src="https://github.com/user-attachments/assets/c0b8af5a-ee6b-43f9-8b46-a80d2539a419" />

즉, 완전 연결 계층의 입력이 `[batch, feature]`라면 배치 정규화는 각 feature에 대해 batch 방향의 통계를 계산한다.

---

### 6.9 학습 단계에서의 배치 정규화

학습 단계에서는 현재 미니 배치의 평균과 분산을 사용한다.

또한 평가 단계에서 사용할 수 있도록 이동 평균 형태의 통계도 함께 저장한다.

```text
현재 미니 배치 평균과 분산:
현재 학습 계산에 사용

running mean과 running variance:
평가 단계에 사용할 통계로 누적
```

PyTorch에서는 `model.train()` 상태에서 이 동작이 수행된다.

---

### 6.10 평가 단계에서의 배치 정규화

평가 단계에서는 입력이 한 개일 수도 있고 배치 구성이 계속 달라질 수 있다.

따라서 현재 테스트 배치의 평균과 분산을 새로 계산하는 대신, 학습 중 누적한 이동 평균과 이동 분산을 사용한다.

```text
학습 모드:
현재 미니 배치 통계 사용

평가 모드:
학습 중 저장한 running statistics 사용
```

평가 전에 반드시 다음 코드를 호출해야 한다.

```python
model.eval()
```

`model.eval()`을 호출하지 않으면 배치 정규화가 계속 현재 배치의 통계를 사용하여 예측 결과가 불안정해질 수 있다.

---

### 6.11 배치 정규화의 장점

배치 정규화의 대표적인 장점은 다음과 같다.

```text
기울기 소실과 폭주 완화
가중치 초기값에 대한 민감도 감소
더 큰 학습률을 사용할 수 있는 경우가 많음
학습 수렴 속도 개선 가능
미니 배치 통계로 인한 약한 잡음 효과
```

미니 배치마다 평균과 분산이 조금씩 달라지므로 학습 과정에 약한 잡음이 들어간다.

이 때문에 일부 과적합 완화 효과가 나타날 수 있지만, 배치 정규화의 주목적은 학습 안정화이다.

강한 규제가 필요하면 드롭아웃이나 가중치 감소와 함께 사용할 수 있다.

---

### 6.12 배치 정규화의 한계: 작은 배치 크기

배치 정규화는 미니 배치의 평균과 분산에 의존한다.

배치 크기가 너무 작으면 통계가 불안정해진다.

특히 배치 크기가 1이면 해당 배치만으로 계산한 분산이 0이 될 수 있다.

```text
배치가 충분히 큼:
평균과 분산이 비교적 안정적

배치가 매우 작음:
평균과 분산이 샘플에 따라 크게 흔들림
```

따라서 메모리 제약 때문에 매우 작은 배치를 사용하는 환경에서는 배치 정규화가 잘 동작하지 않을 수 있다.

---

### 6.13 배치 정규화의 한계: RNN과 시퀀스 데이터

RNN은 시점마다 입력과 은닉 상태의 통계가 달라질 수 있다.

또한 시퀀스 길이가 서로 다를 수 있어 배치 방향 통계를 일관되게 계산하기 어렵다.

이 때문에 전통적인 배치 정규화를 RNN에 그대로 적용하는 것은 까다롭다.

시퀀스 모델에서는 배치 크기에 덜 의존하는 층 정규화를 자주 사용한다.

---

### 6.14 배치 정규화의 계산 비용

배치 정규화는 평균, 분산, 스케일, 시프트 연산을 추가하므로 정규화 계층이 없는 모델보다 연산이 늘어난다.

다만 평가 단계에서는 저장된 통계를 사용하므로 미니 배치 평균과 분산을 새로 계산하지 않는다.

또한 실제 배포 환경에서는 선형 계층이나 합성곱 계층과 배치 정규화 연산을 결합하여 최적화할 수 있다.

따라서 서비스 속도를 판단할 때는 모델 전체의 정확도, 지연 시간, 메모리 사용량을 함께 측정해야 한다.

---

## 7. 층 정규화(Layer Normalization)

`층 정규화(Layer Normalization)`는 하나의 샘플 안에서 여러 특성을 기준으로 평균과 분산을 계산하는 방법이다.

배치 정규화가 같은 특성을 여러 샘플에 걸쳐 정규화한다면, 층 정규화는 각 샘플의 특성들을 한 번에 정규화한다.

```text
배치 정규화:
샘플들 사이에서 같은 특성을 묶어 정규화

층 정규화:
각 샘플 안의 여러 특성을 묶어 정규화
```

---

### 7.1 층 정규화가 계산되는 방향

배치 크기가 3이고 특성이 4개라고 하자.

입력은 다음과 같다.

```text
샘플 1: [x11, x12, x13, x14]
샘플 2: [x21, x22, x23, x24]
샘플 3: [x31, x32, x33, x34]
```

층 정규화는 각 행 안에서 평균과 분산을 계산한다.

```text
샘플 1의 평균과 분산:
x11, x12, x13, x14로 계산

샘플 2의 평균과 분산:
x21, x22, x23, x24로 계산

샘플 3의 평균과 분산:
x31, x32, x33, x34로 계산
```

<img width="656" height="119" alt="Image" src="https://github.com/user-attachments/assets/52e8d4ef-91a7-403c-bf6a-7623ed9aea9b" />

---

### 7.2 층 정규화 수식

하나의 샘플이 $H$개의 특성을 가진다고 하자.

샘플 내부의 평균은 다음과 같다.

$$
\mu
=
\frac{1}{H}
\sum_{j=1}^{H}x_j
$$

분산은 다음과 같다.

$$
\sigma^2
=
\frac{1}{H}
\sum_{j=1}^{H}
(x_j-\mu)^2
$$

정규화된 값은 다음과 같다.

$$
\hat{x}_j
=
\frac{x_j-\mu}
{\sqrt{\sigma^2+\epsilon}}
$$

마지막으로 학습 가능한 $\gamma$와 $\beta$를 적용한다.

$$
y_j=\gamma_j\hat{x}_j+\beta_j
$$

---

### 7.3 층 정규화의 특징

층 정규화는 각 샘플 내부에서 통계를 계산하므로 배치 크기에 의존하지 않는다.

```text
배치 크기 64:
각 샘플을 독립적으로 정규화

배치 크기 1:
동일하게 각 샘플 내부에서 정규화 가능
```

따라서 다음과 같은 상황에서 유리하다.

```text
배치 크기가 매우 작은 경우
샘플마다 길이가 다른 시퀀스 데이터
RNN, LSTM과 같은 순환 신경망
Transformer와 같은 시퀀스 모델
```

---

### 7.4 PyTorch에서 층 정규화 사용하기

특성 수가 100인 완전 연결 계층의 출력에 층 정규화를 적용하면 다음과 같다.

```python
model = nn.Sequential(
    nn.Linear(784, 100),
    nn.LayerNorm(100),
    nn.ReLU(),
    nn.Linear(100, 10)
)
```

`nn.LayerNorm(100)`은 마지막 차원의 100개 특성을 기준으로 각 샘플을 정규화한다.

다차원 입력에서는 정규화할 마지막 차원들의 크기를 튜플로 전달할 수 있다.

```python
nn.LayerNorm((channels, height, width))
```

---

### 7.5 층 정규화의 학습과 평가

층 정규화는 현재 샘플 내부의 통계를 사용한다.

따라서 배치 정규화처럼 학습 중 이동 평균과 이동 분산을 저장할 필요가 없다.

```text
학습 단계:
현재 샘플의 통계 사용

평가 단계:
현재 샘플의 통계 사용
```

`model.eval()`은 드롭아웃이나 배치 정규화에는 중요한 영향을 주지만, 층 정규화의 통계 계산 방식은 학습과 평가에서 동일하다.

---

### 7.6 배치 정규화와 층 정규화 비교

| 구분 | 배치 정규화 | 층 정규화 |
|---|---|---|
| 영어 | Batch Normalization | Layer Normalization |
| 통계 계산 방향 | 같은 특성을 배치 방향으로 계산 | 한 샘플의 특성 방향으로 계산 |
| 배치 크기 의존성 | 있음 | 거의 없음 |
| 학습 시 통계 | 현재 미니 배치 평균·분산 | 현재 샘플 평균·분산 |
| 평가 시 통계 | running mean·variance | 현재 샘플 평균·분산 |
| 주 사용 분야 | CNN, 일반적인 대형 미니 배치 학습 | RNN, Transformer, 작은 배치 |
| PyTorch | `nn.BatchNorm1d()` 등 | `nn.LayerNorm()` |

---

### 7.7 정규화와 규제의 용어 구분

`Regularization`과 `Normalization`은 서로 다른 개념이다.

```text
Regularization:
과적합을 줄이기 위해 모델의 복잡도나 가중치 사용을 제한
예: L1, L2, Dropout

Normalization:
데이터 또는 중간 활성값의 스케일과 분포를 조정
예: 입력값 정규화, Batch Normalization, Layer Normalization
```

한국어로 둘 다 비슷하게 번역되는 경우가 있어 혼동할 수 있다.

이 정리에서는 다음 표현을 사용한다.

```text
Regularization → 규제 또는 정형화
Normalization → 정규화
```

---

## 8. 반드시 이해해야 할 핵심 정리

### 8.1 다층 퍼셉트론의 구조

다층 퍼셉트론(Multi-Layer Perceptron, MLP)은 입력층과 출력층 사이에 하나 이상의 은닉층을 가진 신경망이다.

```text
입력층
   ↓
Linear
   ↓
비선형 활성화 함수
   ↓
Linear
   ↓
비선형 활성화 함수
   ↓
출력층
```

선형 계층만 여러 개 연결하면 전체 모델도 하나의 선형 변환으로 합쳐질 수 있다. 따라서 은닉층 사이에 `ReLU`와 같은 비선형 활성화 함수가 필요하다.

---

### 8.2 숫자 필기 데이터와 MNIST의 차이

| 구분 | 사이킷런 digits | MNIST |
|---|---:|---:|
| 이미지 크기 | 8 × 8 | 28 × 28 |
| 입력 특성 수 | 64 | 784 |
| 픽셀값 범위 | 0~15 | 0~255 |
| 전체 샘플 수 | 1,797개 | 70,000개 |
| 클래스 수 | 10개 | 10개 |

완전 연결 계층에 입력할 때는 이미지를 다음과 같이 펼친다.

```text
8 × 8 이미지 → 64차원 벡터
28 × 28 이미지 → 784차원 벡터
```

---

### 8.3 출력층과 CrossEntropyLoss

숫자 분류에서 출력층은 숫자 0부터 9까지에 대응하는 10개의 값을 출력한다.

```text
출력 크기 = 10
각 출력값 = 해당 클래스에 대한 logit
```

`nn.CrossEntropyLoss()`는 내부적으로 `log_softmax`와 음의 로그 가능도 손실을 함께 처리한다.

따라서 모델의 출력층 뒤에 softmax를 직접 추가하지 않는다.

```python
y_pred = model(X)
loss = nn.CrossEntropyLoss()(y_pred, Y)
```

정답 레이블은 원-핫 벡터가 아니라 클래스 인덱스이며 자료형은 `torch.long`이어야 한다.

---

### 8.4 학습 과정의 순서

```text
1. optimizer.zero_grad()로 이전 기울기를 초기화한다.
2. model(X)로 순전파를 수행한다.
3. 손실 함수로 예측값과 정답의 차이를 계산한다.
4. loss.backward()로 각 매개변수의 기울기를 계산한다.
5. optimizer.step()으로 가중치와 편향을 업데이트한다.
```

PyTorch는 기본적으로 기울기를 누적하므로 `optimizer.zero_grad()`를 생략하면 이전 단계의 기울기가 계속 더해진다.

---

### 8.5 평가 과정의 핵심

평가할 때는 다음 두 설정을 함께 사용하는 것이 일반적이다.

```python
model.eval()

with torch.no_grad():
    outputs = model(X_test)
```

```text
model.eval():
Dropout과 Batch Normalization을 평가 방식으로 전환

torch.no_grad():
기울기 계산과 계산 그래프 저장을 중단
```

최종 예측 클래스는 가장 큰 logit의 인덱스로 구한다.

```python
predicted = torch.argmax(outputs, dim=1)
```

---

### 8.6 과적합과 일반화

과적합은 모델이 훈련 데이터의 일반적인 패턴뿐 아니라 노이즈와 세부 특징까지 지나치게 암기한 상태이다.

```text
훈련 데이터 성능: 매우 높음
검증·테스트 데이터 성능: 상대적으로 낮음
```

과적합을 완화하는 대표적인 방법은 다음과 같다.

```text
1. 데이터의 양 늘리기
2. 데이터 증강 사용하기
3. 은닉층이나 뉴런 수를 줄여 모델 단순화하기
4. L1 또는 L2 가중치 규제 적용하기
5. 드롭아웃 적용하기
```

---

### 8.7 L1 규제와 L2 규제

L1 규제는 가중치 절댓값의 합을 손실에 추가한다.

$$
L_{\mathrm{total}}
=
L_{\mathrm{data}}
+
\lambda\sum_i \lvert w_i \rvert
$$

L2 규제는 가중치 제곱합을 손실에 추가한다.

$$
L_{\mathrm{total}}
=
L_{\mathrm{data}}
+
\lambda\sum_i w_i^2
$$

```text
L1 규제:
일부 가중치를 정확히 0으로 만들기 쉬움
희소한 모델과 특성 선택 효과

L2 규제:
가중치를 전체적으로 작게 유지
딥러닝에서 더 일반적으로 사용
```

PyTorch에서는 옵티마이저의 `weight_decay`로 L2 계열 규제를 적용할 수 있다.

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)
```

---

### 8.8 드롭아웃의 동작

드롭아웃은 학습할 때 일부 뉴런의 출력을 무작위로 0으로 만든다.

```text
학습 단계:
일부 뉴런을 무작위로 제거

평가 단계:
모든 뉴런을 사용
```

드롭아웃은 모델이 특정 뉴런이나 특정 경로에 지나치게 의존하는 것을 줄여 과적합을 완화한다.

---

### 8.9 기울기 소실과 기울기 폭주

```text
기울기 소실:
입력층 방향으로 갈수록 기울기가 매우 작아져 앞쪽 층이 거의 학습되지 않는 현상

기울기 폭주:
역전파 과정에서 기울기가 지나치게 커져 가중치가 비정상적으로 커지고 학습이 불안정해지는 현상
```

은닉층에서 시그모이드나 하이퍼볼릭 탄젠트를 반복해서 사용하면 기울기 소실이 심해질 수 있다. 이를 완화하기 위해 ReLU 또는 Leaky ReLU 계열을 주로 사용한다.

---

### 8.10 Xavier 초기화와 He 초기화

가중치 초기화는 층을 지날 때 활성값과 기울기의 분산이 지나치게 커지거나 작아지지 않도록 설계해야 한다.

```text
Sigmoid 또는 tanh 계열 → Xavier 초기화
ReLU 또는 Leaky ReLU 계열 → He 초기화
```

대표적인 정규 분포 기준은 다음과 같다.

$$
\sigma_{\mathrm{Xavier}}
=
\sqrt{\frac{2}{n_{in}+n_{out}}}
$$

$$
\sigma_{\mathrm{He}}
=
\sqrt{\frac{2}{n_{in}}}
$$

---

### 8.11 배치 정규화의 계산

배치 정규화는 같은 특성에 대해 미니 배치 방향으로 평균과 분산을 계산한다.

$$
\mu_B
=
\frac{1}{m}\sum_{i=1}^{m}x^{(i)}
$$

$$
\sigma_B^2
=
\frac{1}{m}\sum_{i=1}^{m}(x^{(i)}-\mu_B)^2
$$

$$
\hat{x}^{(i)}
=
\frac{x^{(i)}-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}
$$

$$
y^{(i)}=\gamma\hat{x}^{(i)}+\beta
$$

훈련할 때는 현재 미니 배치의 통계와 이동 평균을 사용하고, 평가할 때는 저장해 둔 이동 평균과 이동 분산을 사용한다.

---

### 8.12 층 정규화의 계산

층 정규화는 한 샘플의 특성 방향으로 평균과 분산을 계산한다.

```text
배치 정규화:
같은 특성의 여러 샘플을 기준으로 계산

층 정규화:
한 샘플 안의 여러 특성을 기준으로 계산
```

층 정규화는 배치 크기에 거의 의존하지 않으며, RNN과 Transformer 같은 시퀀스 모델에서 널리 사용된다.

---

## 9. 헷갈리기 쉬운 부분 정리

### 9.1 사이킷런 digits와 MNIST는 같은 데이터인가?

아니다.

두 데이터 모두 숫자 0~9의 손글씨 이미지이지만 크기와 샘플 수가 다르다.

```text
digits: 8 × 8, 64차원, 1,797개
MNIST: 28 × 28, 784차원, 70,000개
```

---

### 9.2 `load_digits()`의 샘플 수는 1,979개인가?

아니다.

정확한 전체 샘플 수는 `1,797개`이다.

```python
len(digits.images)
# 1797
```

---

### 9.3 마지막 `Linear` 계층도 은닉층인가?

숫자 분류 모델의 마지막 `Linear` 계층은 출력층이다.

```python
nn.Linear(16, 10)
```

이 계층은 16개의 은닉 표현을 숫자 0~9에 대응하는 10개의 logit으로 변환한다.

---

### 9.4 출력층 뒤에 softmax를 넣어야 하는가?

`CrossEntropyLoss`를 사용할 때는 넣지 않는다.

```text
올바른 흐름:
Linear 출력 logit → CrossEntropyLoss

피해야 할 흐름:
Linear 출력 → Softmax → CrossEntropyLoss
```

평가할 때 클래스 인덱스만 필요하다면 softmax 없이 가장 큰 logit의 위치를 선택해도 결과가 같다.

---

### 9.5 `torch.max(outputs, 1)`은 가장 높은 확률을 찾는 것인가?

엄밀히 말하면 예제의 `outputs`는 확률이 아니라 logit이다.

```python
values, predicted = torch.max(outputs, dim=1)
```

softmax는 값의 순서를 바꾸지 않기 때문에 가장 큰 logit의 클래스와 가장 큰 확률의 클래스는 동일하다.

---

### 9.6 `model.eval()`과 `torch.no_grad()`는 같은 기능인가?

아니다.

```text
model.eval():
계층의 동작 방식을 평가용으로 변경

no_grad():
자동 미분과 계산 그래프 기록을 중단
```

정확하고 효율적인 평가를 위해 둘을 함께 사용하는 것이 좋다.

---

### 9.7 에포크가 끝난 뒤 출력한 `loss`는 에포크 평균 손실인가?

미니 배치 반복문 안에서 마지막으로 계산된 `loss`를 그대로 출력하면 마지막 미니 배치의 손실이다.

```python
for data, targets in loader_train:
    loss = loss_fn(model(data), targets)

print(loss.item())
```

에포크 전체 평균 손실을 구하려면 각 배치의 손실을 누적한 뒤 전체 샘플 수로 나눠야 한다.

---

### 9.8 훈련 손실이 줄어들면 항상 좋은 모델인가?

아니다.

훈련 손실이 계속 줄어들더라도 검증 손실이 증가한다면 과적합이 진행되고 있을 수 있다.

```text
훈련 손실 감소 + 검증 손실 감소:
일반적인 학습 진행

훈련 손실 감소 + 검증 손실 증가:
과적합 가능성
```

---

### 9.9 L1 규제와 L2 규제의 가장 큰 차이는 무엇인가?

```text
L1:
가중치 일부가 정확히 0이 되기 쉬움
희소성 유도

L2:
가중치를 0 근처의 작은 값으로 부드럽게 줄임
전체 가중치의 과도한 증가 억제
```

딥러닝에서 `weight_decay`라는 이름으로 흔히 사용하는 것은 일반적으로 L2 계열 규제이다.

---

### 9.10 Regularization과 Normalization은 같은 말인가?

아니다.

```text
Regularization:
과적합을 줄이기 위해 모델을 제한하는 방법
예: L1, L2, Dropout

Normalization:
데이터나 활성값의 분포와 크기를 조정하는 방법
예: 입력 정규화, BatchNorm, LayerNorm
```

한국어 번역에서 표현이 비슷해 혼동할 수 있지만 목적과 연산이 다르다.

---

### 9.11 드롭아웃은 평가할 때도 뉴런을 제거하는가?

아니다.

```text
model.train(): 드롭아웃 활성화
model.eval(): 드롭아웃 비활성화
```

평가할 때 드롭아웃을 끄지 않으면 같은 입력에도 예측이 무작위로 달라질 수 있다.

---

### 9.12 시그모이드에는 He 초기화를 사용하면 안 되는가?

반드시 오류가 발생하는 것은 아니지만 일반적인 조합은 아니다.

```text
Sigmoid·tanh → Xavier
ReLU 계열 → He
```

초기화 방법은 활성화 함수가 층을 통과하며 분산에 미치는 특성을 고려해 선택한다.

---

### 9.13 배치 정규화는 데이터셋 전체의 평균과 분산을 사용하는가?

훈련 중에는 현재 미니 배치의 평균과 분산을 사용한다.

평가 중에는 훈련 과정에서 누적한 이동 평균과 이동 분산을 사용한다.

```text
훈련: 미니 배치 통계 사용
평가: running mean, running variance 사용
```

---

### 9.14 배치 정규화는 배치 크기가 1이어도 잘 동작하는가?

일반적으로 안정적으로 동작하기 어렵다.

배치가 지나치게 작으면 평균과 분산 추정이 불안정하며, 특성마다 값이 하나뿐이면 분산을 제대로 추정할 수 없다.

작은 배치나 시퀀스 모델에서는 층 정규화가 더 적합할 수 있다.

---

### 9.15 배치 정규화와 층 정규화의 계산 방향은 어떻게 다른가?

샘플이 행이고 특성이 열인 행렬을 생각하면 다음과 같다.

```text
배치 정규화:
열 방향으로 계산
같은 특성을 가진 여러 샘플의 평균과 분산

층 정규화:
행 방향으로 계산
한 샘플 안에 있는 여러 특성의 평균과 분산
```

---

### 9.16 배치 정규화의 효과가 내부 공변량 변화 감소 때문이라고 확정할 수 있는가?

배치 정규화를 처음 제안한 설명은 내부 공변량 변화를 줄여 학습을 안정화한다는 것이었다.

하지만 이후 연구에서는 손실 함수의 최적화 지형을 더 부드럽게 만드는 효과가 중요하다는 다른 설명도 제시되었다.

따라서 내부 공변량 변화는 배치 정규화를 이해하는 전통적인 설명으로 보되, 효과의 원인이 그것 하나로 완전히 확정된 것은 아니라고 이해해야 한다.

---

### 9.17 배치 정규화가 드롭아웃을 완전히 대체하는가?

아니다.

배치 정규화에는 미니 배치 통계로 인한 약한 규제 효과가 있을 수 있지만, 주목적은 학습 안정화이다.

```text
BatchNorm과 LayerNorm:
학습 안정화가 중심

Dropout과 L1·L2:
과적합 완화가 중심
```

---

## 10. 전체 내용을 한 줄로 요약

```text
숫자 이미지를 벡터로 변환해 다층 퍼셉트론으로 분류하고, 데이터 증강·규제·드롭아웃으로 과적합을 줄이며, ReLU 계열 활성화 함수·적절한 가중치 초기화·배치 정규화·층 정규화로 깊은 신경망의 학습을 안정화한다.
```
