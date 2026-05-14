# 02. 파이토치 기초(PyTorch Basics)

## 1. 파이토치 패키지의 기본 구성

### 파이토치 패키지란?

파이토치(PyTorch)는 딥 러닝 모델을 만들고 학습시키기 위해 사용하는 파이썬 기반 프레임워크이다.

파이토치는 여러 하위 패키지와 모듈로 구성되어 있으며, 각 구성 요소는 텐서 연산, 자동 미분, 신경망 구성, 최적화, 데이터 처리, 모델 변환 등의 역할을 담당한다.

```text
PyTorch 패키지 구성
├─ torch
├─ torch.autograd
├─ torch.nn
├─ torch.optim
├─ torch.utils.data
└─ torch.onnx
```

<details>
<summary>프레임워크란?</summary>

프레임워크는 특정 목적의 프로그램을 더 쉽게 만들 수 있도록 기본 구조와 기능을 제공하는 도구이다.

딥 러닝 프레임워크는 텐서 연산, 신경망 구성, 학습, 최적화 같은 기능을 제공하여 딥 러닝 모델을 더 쉽게 구현할 수 있게 해준다.

</details>

---

### `torch`

`torch`는 파이토치의 **메인 네임스페이스**이다.

```text
torch = 파이토치의 기본 기능이 모여 있는 메인 네임스페이스
```

`torch`에는 텐서와 관련된 기능을 포함하여 다양한 수학 함수가 들어 있다.  
구조는 NumPy와 유사하다.

```text
torch
├─ 텐서 관련 기능
└─ 다양한 수학 함수
```

<details>
<summary>네임스페이스란?</summary>

네임스페이스는 함수, 클래스, 변수 등을 이름 충돌 없이 묶어서 관리하는 공간이다.

예를 들어 `torch` 안에는 파이토치에서 사용하는 여러 함수와 기능이 모여 있다.

</details>

---

### `torch.autograd`

`torch.autograd`는 **자동 미분(Automatic Differentiation)을 위한 함수들이 포함된 패키지**이다.

```text
torch.autograd = 자동 미분 기능 제공
```

`torch.autograd`에는 자동 미분의 사용 여부를 제어하는 기능과, 직접 미분 가능한 함수를 정의할 때 사용하는 기반 클래스 등이 포함된다.

주요 구성은 다음과 같다.

```text
torch.autograd
├─ enable_grad
├─ no_grad
└─ Function
```

`enable_grad`와 `no_grad`는 자동 미분의 on/off를 제어하는 콘텍스트 매니저이다.  
`Function`은 자체 미분 가능 함수를 정의할 때 사용하는 기반 클래스이다.

<details>
<summary>자동 미분이란?</summary>

자동 미분은 모델의 학습 과정에서 필요한 미분값을 자동으로 계산해주는 기능이다.

딥 러닝에서는 손실 함수의 값을 줄이기 위해 각 파라미터가 얼마나 영향을 주는지 계산해야 하는데, 이때 자동 미분이 사용된다.

</details>

---

### `torch.nn`

`torch.nn`은 **신경망을 구축하기 위한 다양한 데이터 구조와 레이어가 정의된 패키지**이다.

```text
torch.nn = 신경망 구성 요소 제공
```

`torch.nn`에는 신경망 레이어, 활성화 함수, 손실 함수 등이 포함된다.

예시:

```text
torch.nn
├─ RNN
├─ LSTM
├─ ReLU
└─ MSELoss
```

각 구성 요소의 역할은 다음과 같다.

| 구성 요소 | 역할 |
|---|---|
| `RNN` | 순차 데이터를 처리하는 신경망 레이어 |
| `LSTM` | 장기 의존성을 다루기 위한 순환 신경망 계열 레이어 |
| `ReLU` | 활성화 함수 |
| `MSELoss` | 평균 제곱 오차 손실 함수 |

<details>
<summary>레이어, 활성화 함수, 손실 함수란?</summary>

레이어는 신경망을 구성하는 층이다.  
활성화 함수는 신경망이 비선형적인 패턴을 학습할 수 있도록 돕는 함수이다.  
손실 함수는 모델의 예측값과 실제 정답이 얼마나 다른지를 계산하는 함수이다.

</details>

---

### `torch.optim`

`torch.optim`은 **파라미터 최적화 알고리즘이 구현된 패키지**이다.

```text
torch.optim = 모델 파라미터 최적화 기능 제공
```

대표적으로 확률적 경사 하강법(Stochastic Gradient Descent, SGD)을 중심으로 한 최적화 알고리즘이 포함되어 있다.

```text
torch.optim
└─ SGD 등 최적화 알고리즘
```

<details>
<summary>다른 최적화 알고리즘 예시</summary>

원문에서는 SGD를 중심으로 설명하지만, 실제 PyTorch에서는 `Adam`, `RMSprop` 같은 최적화 알고리즘도 자주 사용된다.

```text
torch.optim
├─ SGD
├─ Adam
└─ RMSprop 등
```

</details>

<details>
<summary>최적화란?</summary>

최적화는 모델의 예측이 실제 정답에 가까워지도록 파라미터를 조정하는 과정이다.

딥 러닝에서는 손실 함수의 값을 줄이는 방향으로 모델의 파라미터를 업데이트한다.

</details>

---

### `torch.utils.data`

`torch.utils.data`는 **SGD의 반복 연산을 실행할 때 사용하는 미니 배치용 유틸리티 함수가 포함된 패키지**이다.

```text
torch.utils.data = 데이터 로딩과 미니 배치 처리 지원
```

```text
torch.utils.data
└─ 미니 배치용 유틸리티 함수
```

<details>
<summary>Dataset과 DataLoader</summary>

실제 PyTorch에서는 `torch.utils.data`에서 `Dataset`과 `DataLoader`를 자주 사용한다.

```text
torch.utils.data
├─ Dataset
└─ DataLoader
```

`Dataset`은 데이터를 어떻게 가져올지 정의하는 객체이고, `DataLoader`는 데이터를 batch size 단위로 묶어서 반복적으로 불러오는 도구이다.

</details>

<details>
<summary>미니 배치란?</summary>

미니 배치는 전체 데이터를 한 번에 사용하지 않고, 작은 묶음 단위로 나누어 학습하는 방식이다.

전체 데이터를 한 번에 학습에 사용하면 계산량이 커질 수 있으므로, 딥 러닝에서는 미니 배치 단위로 데이터를 나누어 학습하는 경우가 많다.

</details>

---

### `torch.onnx`

`torch.onnx`는 **ONNX 포맷으로 모델을 익스포트(export)할 때 사용하는 패키지**이다.

```text
torch.onnx = PyTorch 모델을 ONNX 포맷으로 변환할 때 사용
```

ONNX(Open Neural Network Exchange)는 서로 다른 딥 러닝 프레임워크 간에 모델을 공유할 때 사용하는 포맷이다.

```text
ONNX = 서로 다른 딥 러닝 프레임워크 간 모델 공유를 위한 포맷
```

<details>
<summary>익스포트(export)란?</summary>

익스포트는 현재 사용 중인 모델이나 데이터를 다른 환경에서도 사용할 수 있도록 특정 형식으로 내보내는 과정이다.

예를 들어 PyTorch에서 만든 모델을 ONNX 형식으로 내보내면, ONNX를 지원하는 다른 프레임워크나 환경에서 활용할 수 있다.

</details>


---

## 2. 텐서 조작하기(Tensor Manipulation)

### 이번 절에서 배울 내용

이번 절에서는 벡터(Vector), 행렬(Matrix), 텐서(Tensor)의 개념을 이해하고, NumPy와 PyTorch로 벡터, 행렬, 텐서를 다루는 방법을 학습한다.

```text
텐서 조작하기
├─ 벡터, 행렬, 텐서
├─ NumPy 훑어보기
├─ PyTorch 텐서 선언하기
├─ 행렬 곱셈
└─ 기본 오퍼레이션
```

각 내용의 흐름은 다음과 같다.

| 내용 | 핵심 |
|---|---|
| 벡터, 행렬, 텐서 | 딥 러닝에서 다루는 기본 데이터 단위 |
| NumPy 훑어보기 | NumPy로 벡터와 행렬 생성 및 조작 |
| PyTorch 텐서 선언하기 | PyTorch로 텐서 생성 및 조작 |
| 행렬 곱셈 | `matmul()`과 원소별 곱셈의 차이 |
| 기본 오퍼레이션 | 평균, 합, 최대, 뷰, 스퀴즈, 연결, 스택킹 등 |

---

### 벡터, 행렬 그리고 텐서(Vector, Matrix and Tensor)

딥 러닝에서 다루는 가장 기본적인 단위는 **벡터, 행렬, 텐서**이다.

```text
스칼라 = 숫자 하나
벡터 = 1차원 텐서
행렬 = 2차원 텐서
텐서 = 다차원 배열
```

<img width="574" height="350" alt="Image" src="https://github.com/user-attachments/assets/c09eb5b9-4d2d-49cf-91af-353071a5ff57" />

차원이 없는 값은 **스칼라(Scalar)** 라고 한다.  
예를 들어 숫자 `47`은 스칼라이다.

숫자가 여러 개 나열되어 있으면 **벡터(Vector)** 라고 한다.  
예를 들어 `[3, 5, 7]`은 벡터이다.

행과 열로 이루어진 2차원 값은 **행렬(Matrix)** 이라고 한다.  
행렬은 2차원 텐서라고도 볼 수 있다.

3차원 이상부터는 보통 **텐서(Tensor)** 라는 표현을 사용한다.

<details>
<summary>벡터, 행렬, 텐서의 관계</summary>

데이터 사이언스 분야에서는 3차원 이상의 텐서를 다차원 행렬 또는 다차원 배열처럼 볼 수 있다.

또한 1차원 벡터와 2차원 행렬도 텐서라고 표현할 수 있다.

```text
벡터 = 1차원 텐서
행렬 = 2차원 텐서
3차원 이상 = 일반적으로 텐서라고 부름
```

</details>

---

### PyTorch Tensor Shape Convention

딥 러닝에서는 다루는 행렬이나 텐서의 크기(shape)를 정확히 파악하는 것이 중요하다.

PyTorch에서는 텐서의 크기를 보통 다음과 같은 방식으로 표현한다.

```text
shape = 텐서의 크기
```

---

### 2D Tensor

가장 전형적인 2차원 텐서는 행렬이다.

```text
|t| = (batch_size, dim)
```

<img width="394" height="300" alt="Image" src="https://github.com/user-attachments/assets/baa4c273-9f1a-4bc4-a70a-0c3833799744" />

위 그림에서 행의 크기는 `batch_size`, 열의 크기는 `dim`이다.

```text
batch_size = 한 번에 처리하는 데이터 개수
dim = 하나의 데이터가 가지는 벡터 차원
```

예를 들어 훈련 데이터 하나의 크기가 256이라고 하면, 데이터 하나는 길이 256의 벡터이다.  
이런 훈련 데이터가 3,000개 있다면 전체 훈련 데이터의 크기는 다음과 같다.

```text
전체 훈련 데이터 크기 = 3,000 × 256
```

이 중 64개씩 꺼내서 처리한다면 `batch_size`는 64이다.

```text
한 번에 처리하는 2D 텐서 크기
= batch_size × dim
= 64 × 256
```

<details>
<summary>배치 크기(batch size)란?</summary>

배치 크기(batch size)는 훈련 데이터가 많을 때 컴퓨터가 한 번에 가져가서 처리하는 데이터의 개수이다.

예를 들어 전체 데이터가 3,000개이고 64개씩 처리한다면, 배치 크기는 64이다.

</details>

---

### 3D Tensor: 컴퓨터 비전 분야

컴퓨터 비전(Computer Vision) 분야에서는 이미지나 영상 데이터를 다루기 때문에 3차원 텐서를 자주 사용한다.

```text
|t| = (batch_size, width, height)
```

<img width="405" height="308" alt="Image" src="https://github.com/user-attachments/assets/0a79293b-1dd1-4126-a7d6-f7097aa064e6" />

이미지는 가로와 세로가 존재한다.  
여러 장의 이미지를 batch size로 구성하면 3차원 텐서가 된다.

```text
batch_size = 이미지 개수
width = 이미지의 너비
height = 이미지의 높이
```

<details>
<summary>이미지 텐서에서 채널은 어디에 들어가나요?</summary>

이 자료에서는 개념 설명을 위해 이미지 텐서를 `(batch_size, width, height)`로 단순화한다.

하지만 실제 이미지 데이터에서는 색상 정보를 나타내는 채널(channel) 차원이 추가되는 경우가 많다.  
예를 들어 RGB 이미지는 빨강, 초록, 파랑의 3개 채널을 가진다.

```text
단순화된 이미지 텐서 = (batch_size, width, height)
일반적인 PyTorch 이미지 텐서 = (batch_size, channel, height, width)
다른 라이브러리에서 자주 쓰는 형식 = (batch_size, height, width, channel)
예: RGB 이미지 32장 = (32, 3, H, W)
```

즉, PyTorch에서는 이미지 데이터를 다룰 때 보통 `(N, C, H, W)` 형태를 자주 사용한다.

</details>

---

### 3D Tensor: 자연어 처리 분야

자연어 처리(Natural Language Processing, NLP) 분야에서도 3차원 텐서를 자주 사용한다.

```text
|t| = (batch_size, length, dim)
```

<img width="382" height="273" alt="Image" src="https://github.com/user-attachments/assets/540db281-909f-484e-98f5-466f64ccffd1" />

자연어 처리에서는 보통 다음과 같은 의미로 해석한다.

```text
batch_size = 한 번에 처리하는 문장 개수
length = 문장 길이
dim = 단어 벡터의 차원
```

---

### NLP 분야의 3D 텐서 예시

아래와 같이 4개의 문장으로 구성된 전체 훈련 데이터가 있다고 하자.

```text
[
  [나는 사과를 좋아해],
  [나는 바나나를 좋아해],
  [나는 사과를 싫어해],
  [나는 바나나를 싫어해]
]
```

컴퓨터가 텍스트를 입력으로 사용하려면 먼저 단어 단위로 나누어야 한다.

```text
[
  ['나는', '사과를', '좋아해'],
  ['나는', '바나나를', '좋아해'],
  ['나는', '사과를', '싫어해'],
  ['나는', '바나나를', '싫어해']
]
```

이제 데이터의 크기는 다음과 같다.

```text
문장 개수 × 문장 길이 = 4 × 3
```

각 단어를 3차원 벡터로 변환한다고 하자.

```text
'나는' = [0.1, 0.2, 0.9]
'사과를' = [0.3, 0.5, 0.1]
'바나나를' = [0.3, 0.5, 0.2]
'좋아해' = [0.7, 0.6, 0.5]
'싫어해' = [0.5, 0.6, 0.7]
```

그러면 전체 훈련 데이터는 다음과 같은 3D 텐서가 된다.

```text
[
  [[0.1, 0.2, 0.9], [0.3, 0.5, 0.1], [0.7, 0.6, 0.5]],
  [[0.1, 0.2, 0.9], [0.3, 0.5, 0.2], [0.7, 0.6, 0.5]],
  [[0.1, 0.2, 0.9], [0.3, 0.5, 0.1], [0.5, 0.6, 0.7]],
  [[0.1, 0.2, 0.9], [0.3, 0.5, 0.2], [0.5, 0.6, 0.7]]
]
```

```text
전체 텐서 크기 = 4 × 3 × 3
```

batch size를 2로 하면 데이터를 두 묶음으로 나누어 처리한다.

```text
첫 번째 배치 크기 = 2 × 3 × 3
두 번째 배치 크기 = 2 × 3 × 3
```

즉, 각 배치의 크기는 다음과 같다.

```text
(batch_size, 문장 길이, 단어 벡터 차원)
= (2, 3, 3)
```

<details>
<summary>단어 벡터란?</summary>

단어 벡터는 단어를 숫자 벡터로 표현한 것이다.  
컴퓨터는 텍스트 자체보다 숫자를 더 잘 처리하므로, 자연어 처리에서는 단어를 숫자 벡터로 변환해서 사용한다.

</details>

---

### shape 표기 방식

텐서의 크기(shape)는 쉼표로 표현하기도 하고, 곱하기 기호로 표현하기도 한다.

```text
(2, 3) = 2 × 3
(4, 3, 2) = 4 × 3 × 2
```

`(5,)`와 같은 형식은 원소가 5개인 1차원 벡터를 의미한다.

```text
(5,) = 길이가 5인 1차원 벡터
```

---

### NumPy로 텐서 만들기

### NumPy 임포트

PyTorch로 텐서를 만들어보기 전에 먼저 NumPy로 텐서를 만들어본다.

```python
import numpy as np
```

NumPy로 텐서를 만드는 방법은 간단하다.  
`[숫자, 숫자, 숫자]`와 같은 형식의 데이터를 만들고, 이를 `np.array()`로 감싸면 된다.

---

### 1D with NumPy

NumPy로 1차원 텐서인 벡터를 만들 수 있다.

```python
t = np.array([0., 1., 2., 3., 4., 5., 6.])
print(t)
```

출력:

```text
[0. 1. 2. 3. 4. 5. 6.]
```

1차원 텐서의 차원과 크기를 출력한다.

```python
print('Rank of t: ', t.ndim)
print('Shape of t: ', t.shape)
```

출력:

```text
Rank of t:  1
Shape of t:  (7,)
```

`.ndim`은 몇 차원인지를 출력한다.  
`.shape`는 텐서의 크기를 출력한다.

현재 `t`는 1차원 벡터이고, 원소는 7개이다.

```text
t.ndim = 1
t.shape = (7,)
```

---

### NumPy 인덱싱

NumPy에서 인덱스는 0부터 시작한다.

```python
print('t[0] t[1] t[-1] = ', t[0], t[1], t[-1])
```

출력:

```text
t[0] t[1] t[-1] =  0.0 1.0 6.0
```

```text
t[0] = 첫 번째 원소
t[1] = 두 번째 원소
t[-1] = 마지막 원소
```

---

### NumPy 슬라이싱

범위를 지정해서 원소를 가져오는 것을 슬라이싱(Slicing)이라고 한다.

```text
슬라이싱 = [시작 번호 : 끝 번호]
```

주의할 점은 끝 번호에 해당하는 인덱스는 포함하지 않는다는 것이다.

```python
print('t[2:5] t[4:-1]  = ', t[2:5], t[4:-1])
```

출력:

```text
t[2:5] t[4:-1]  =  [2. 3. 4.] [4. 5.]
```

```text
t[2:5] = 2번 인덱스부터 4번 인덱스까지
t[4:-1] = 4번 인덱스부터 마지막 직전까지
```

시작 번호나 끝 번호를 생략할 수도 있다.

```python
print('t[:2] t[3:]     = ', t[:2], t[3:])
```

출력:

```text
t[:2] t[3:]     =  [0. 1.] [3. 4. 5. 6.]
```

```text
t[:2] = 처음부터 1번 인덱스까지
t[3:] = 3번 인덱스부터 끝까지
```

<details>
<summary>인덱싱과 슬라이싱의 차이</summary>

인덱싱은 특정 위치의 원소 하나를 가져오는 방식이다.  
슬라이싱은 특정 범위의 원소 여러 개를 가져오는 방식이다.

```text
인덱싱 = 하나 선택
슬라이싱 = 범위 선택
```

</details>

---

### 2D with NumPy

NumPy로 2차원 텐서인 행렬을 만들 수 있다.

```python
t = np.array([
    [1., 2., 3.],
    [4., 5., 6.],
    [7., 8., 9.],
    [10., 11., 12.]
])

print(t)
```

출력:

```text
[[ 1.  2.  3.]
 [ 4.  5.  6.]
 [ 7.  8.  9.]
 [10. 11. 12.]]
```

차원과 크기를 출력한다.

```python
print('Rank  of t: ', t.ndim)
print('Shape of t: ', t.shape)
```

출력:

```text
Rank  of t:  2
Shape of t:  (4, 3)
```

현재 `t`는 2차원 행렬이고, 크기는 `(4, 3)`이다.

```text
(4, 3) = 4행 3열
```

---

### PyTorch 텐서 선언하기(PyTorch Tensor Allocation)

### PyTorch 텐서 선언

PyTorch는 NumPy와 매우 유사하다.  
PyTorch로 텐서를 사용하기 위해 먼저 `torch`를 임포트한다.

```python
import torch
```

---

### 1D with PyTorch

PyTorch로 1차원 텐서인 벡터를 만들 수 있다.

```python
t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
print(t)
```

`dim()`을 사용하면 현재 텐서의 차원을 확인할 수 있다.  
`shape`나 `size()`를 사용하면 텐서의 크기를 확인할 수 있다.

```python
print(t.dim())
print(t.shape)
print(t.size())
```

출력:

```text
1
torch.Size([7])
torch.Size([7])
```

현재 `t`는 1차원 텐서이며, 원소는 7개이다.

```text
t.dim() = 차원
t.shape = 크기
t.size() = 크기
```

---

### PyTorch 인덱싱과 슬라이싱

PyTorch 텐서도 NumPy처럼 인덱싱과 슬라이싱을 사용할 수 있다.

```python
print(t[0], t[1], t[-1])
print(t[2:5], t[4:-1])
print(t[:2], t[3:])
```

출력:

```text
tensor(0.) tensor(1.) tensor(6.)
tensor([2., 3., 4.]) tensor([4., 5.])
tensor([0., 1.]) tensor([3., 4., 5., 6.])
```

---

### 2D with PyTorch

PyTorch로 2차원 텐서인 행렬을 만들 수 있다.

```python
t = torch.FloatTensor([
    [1., 2., 3.],
    [4., 5., 6.],
    [7., 8., 9.],
    [10., 11., 12.]
])

print(t)
```

출력:

```text
tensor([[ 1.,  2.,  3.],
        [ 4.,  5.,  6.],
        [ 7.,  8.,  9.],
        [10., 11., 12.]])
```

차원과 크기를 출력한다.

```python
print(t.dim())
print(t.size())
```

출력:

```text
2
torch.Size([4, 3])
```

현재 텐서의 차원은 2차원이며, 크기는 `(4, 3)`이다.

---

### PyTorch 2D 텐서 슬라이싱

첫 번째 차원을 전체 선택하고, 두 번째 차원의 1번 인덱스 값만 가져올 수 있다.

```python
print(t[:, 1])
print(t[:, 1].size())
```

출력:

```text
tensor([ 2.,  5.,  8., 11.])
torch.Size([4])
```

이는 텐서에서 두 번째 열의 모든 값을 가져오는 것이다.

두 번째 차원에서 마지막 열을 제외하고 가져올 수도 있다.

```python
print(t[:, :-1])
```

출력:

```text
tensor([[ 1.,  2.],
        [ 4.,  5.],
        [ 7.,  8.],
        [10., 11.]])
```

---

### 브로드캐스팅(Broadcasting)

브로드캐스팅(Broadcasting)은 크기가 다른 텐서끼리 연산할 때, PyTorch가 자동으로 크기를 맞춰 연산을 수행하는 기능이다.

```text
브로드캐스팅 = 크기가 다른 텐서의 크기를 자동으로 맞춰 연산하는 기능
```

같은 크기일 때의 덧셈은 다음과 같다.

```python
m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2, 2]])

print(m1 + m2)
```

출력:

```text
tensor([[5., 5.]])
```

두 텐서의 크기는 모두 `(1, 2)`이므로 문제없이 덧셈이 가능하다.

---

### 벡터와 스칼라의 브로드캐스팅

크기가 다른 벡터와 스칼라도 PyTorch에서는 브로드캐스팅을 통해 연산할 수 있다.

```python
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([3])

print(m1 + m2)
```

출력:

```text
tensor([[4., 5.]])
```

원래 `m1`의 크기는 `(1, 2)`이고, `m2`의 크기는 `(1,)`이다.  
PyTorch는 `m2`를 `[3, 3]`처럼 확장하여 연산한다.

```text
[1, 2] + [3]
→ [1, 2] + [3, 3]
→ [4, 5]
```

---

### 벡터 간 브로드캐스팅

다음은 `(1, 2)` 크기의 벡터와 `(2, 1)` 크기의 벡터를 더하는 예시이다.

```python
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3], [4]])

print(m1 + m2)
```

출력:

```text
tensor([[4., 5.],
        [5., 6.]])
```

PyTorch는 두 텐서의 크기를 `(2, 2)`로 맞춘 뒤 연산한다.

```text
[1, 2]
→ [[1, 2],
   [1, 2]]

[[3],
 [4]]
→ [[3, 3],
   [4, 4]]
```

브로드캐스팅은 편리하지만 자동으로 실행되므로 주의해야 한다.  
사용자가 텐서의 크기가 같다고 착각한 상태에서 연산하면, 의도하지 않은 결과가 나올 수 있다.

<details>
<summary>브로드캐스팅을 주의해야 하는 이유</summary>

브로드캐스팅은 크기가 다른 텐서의 연산을 자동으로 처리해준다.  
하지만 사용자가 텐서 크기를 잘못 이해하고 있어도 에러가 발생하지 않을 수 있다.

그래서 결과가 이상하게 나와도 어느 부분에서 문제가 생겼는지 찾기 어려울 수 있다.

</details>

---

### 행렬 곱셈과 원소별 곱셈

행렬로 곱셈을 하는 방법은 크게 두 가지가 있다.

```text
matmul() = 행렬 곱셈
mul() 또는 * = 원소별 곱셈
```

---

### 행렬 곱셈(Matrix Multiplication)

행렬 곱셈은 `matmul()`을 통해 수행한다.

```python
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])

print('Shape of Matrix 1: ', m1.shape)
print('Shape of Matrix 2: ', m2.shape)
print(m1.matmul(m2))
```

출력:

```text
Shape of Matrix 1:  torch.Size([2, 2])
Shape of Matrix 2:  torch.Size([2, 1])
tensor([[ 5.],
        [11.]])
```

위 결과는 `(2, 2)` 행렬과 `(2, 1)` 행렬의 행렬 곱셈 결과이다.

---

### 원소별 곱셈(Element-wise Multiplication)

원소별 곱셈은 같은 위치에 있는 원소끼리 곱하는 연산이다.  
`*` 또는 `mul()`을 통해 수행한다.

```python
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])

print('Shape of Matrix 1: ', m1.shape)
print('Shape of Matrix 2: ', m2.shape)
print(m1 * m2)
print(m1.mul(m2))
```

출력:

```text
Shape of Matrix 1:  torch.Size([2, 2])
Shape of Matrix 2:  torch.Size([2, 1])
tensor([[1., 2.],
        [6., 8.]])
tensor([[1., 2.],
        [6., 8.]])
```

이 경우 `m2`는 브로드캐스팅되어 다음과 같이 변환된 뒤 원소별 곱셈이 수행된다.

```text
[[1],
 [2]]
→ [[1, 1],
   [2, 2]]
```

---

### 평균(Mean)

평균은 `mean()`을 사용해서 구할 수 있다.

1차원 벡터에서 평균을 구하는 예시이다.

```python
t = torch.FloatTensor([1, 2])
print(t.mean())
```

출력:

```text
tensor(1.5000)
```

2차원 행렬에서도 전체 원소의 평균을 구할 수 있다.

```python
t = torch.FloatTensor([[1, 2], [3, 4]])

print(t.mean())
```

출력:

```text
tensor(2.5000)
```

---

### `mean(dim=0)`

`dim=0`은 첫 번째 차원, 즉 행 방향을 기준으로 연산한다.  
그 결과 행 차원이 제거되고, 열별 평균이 남는다.

```python
print(t.mean(dim=0))
```

출력:

```text
tensor([2., 3.])
```

`dim=0`을 지정하면 첫 번째 차원이 제거되고, 열 방향의 결과가 남는다.

```text
[[1., 2.],
 [3., 4.]]

1과 3의 평균 = 2
2와 4의 평균 = 3

결과 = [2., 3.]
```

---

### `mean(dim=1)`과 `mean(dim=-1)`

`dim=1`은 두 번째 차원, 즉 열 방향을 기준으로 연산한다.  
그 결과 열 차원이 제거되고, 행별 평균이 남는다.

```python
print(t.mean(dim=1))
```

출력:

```text
tensor([1.5000, 3.5000])
```

```text
[[1., 2.],
 [3., 4.]]

1과 2의 평균 = 1.5
3과 4의 평균 = 3.5

결과 = [1.5, 3.5]
```

`dim=-1`은 마지막 차원을 의미하므로, 이 경우 `dim=1`과 같은 결과가 나온다.

```python
print(t.mean(dim=-1))
```

출력:

```text
tensor([1.5000, 3.5000])
```

<details>
<summary>dim 인자의 의미</summary>

`dim`은 어떤 차원을 기준으로 연산할지를 지정하는 인자이다.  
원문에서는 `dim`을 지정하면 해당 차원을 제거한다고 설명한다.

```text
dim=0 = 첫 번째 차원 기준 연산
dim=1 = 두 번째 차원 기준 연산
dim=-1 = 마지막 차원 기준 연산
```

</details>

---

### 덧셈(Sum)

덧셈은 `sum()`을 사용한다.  
인자의 의미는 `mean()`과 동일하며, 평균이 아니라 합을 구한다.

```python
t = torch.FloatTensor([[1, 2], [3, 4]])

print(t.sum())
print(t.sum(dim=0))
print(t.sum(dim=1))
print(t.sum(dim=-1))
```

출력:

```text
tensor(10.)
tensor([4., 6.])
tensor([3., 7.])
tensor([3., 7.])
```

---

### 최대(Max)와 아그맥스(ArgMax)

`max()`는 원소의 최대값을 리턴한다.  
`argmax`는 최대값을 가진 인덱스를 의미한다.

```python
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
```

출력:

```text
tensor([[1., 2.],
        [3., 4.]])
```

전체 원소 중 최대값을 구한다.

```python
print(t.max())
```

출력:

```text
tensor(4.)
```

`dim=0`을 주면 최대값과 아그맥스가 함께 리턴된다.

```python
print(t.max(dim=0))
```

출력:

```text
(tensor([3., 4.]), tensor([1, 1]))
```

```text
max = [3., 4.]
argmax = [1, 1]
```

첫 번째 열에서 최대값 3의 인덱스는 1이다.  
두 번째 열에서 최대값 4의 인덱스도 1이다.

최대값만 가져오려면 0번 인덱스를 사용하고, 아그맥스만 가져오려면 1번 인덱스를 사용한다.

```python
print('Max: ', t.max(dim=0)[0])
print('Argmax: ', t.max(dim=0)[1])
```

출력:

```text
Max:  tensor([3., 4.])
Argmax:  tensor([1, 1])
```

`dim=1`이나 `dim=-1`을 주면 다음과 같은 결과가 나온다.

```python
print(t.max(dim=1))
print(t.max(dim=-1))
```

출력:

```text
(tensor([2., 4.]), tensor([1, 1]))
(tensor([2., 4.]), tensor([1, 1]))
```

---

### 뷰(View)

뷰(View)는 원소의 수를 유지하면서 텐서의 크기(shape)를 변경하는 기능이다.

```text
view = 원소 수를 유지하면서 텐서의 크기 변경
```

NumPy의 `reshape()`과 같은 역할을 한다.

먼저 3차원 텐서를 만든다.

```python
t = np.array([
    [[0, 1, 2],
     [3, 4, 5]],
    [[6, 7, 8],
     [9, 10, 11]]
])

ft = torch.FloatTensor(t)
print(ft.shape)
```

출력:

```text
torch.Size([2, 2, 3])
```

<img width="326" height="297" alt="Image" src="https://github.com/user-attachments/assets/784b3e06-5d1d-424e-95d1-c76ab0701cdb" />

현재 텐서의 크기는 `(2, 2, 3)`이다.

---

### 3차원 텐서에서 2차원 텐서로 변경

`view()`를 사용해 3차원 텐서를 2차원 텐서로 변경할 수 있다.

```python
print(ft.view([-1, 3]))
print(ft.view([-1, 3]).shape)
```

출력:

```text
tensor([[ 0.,  1.,  2.],
        [ 3.,  4.,  5.],
        [ 6.,  7.,  8.],
        [ 9., 10., 11.]])
torch.Size([4, 3])
```

<img width="310" height="298" alt="Image" src="https://github.com/user-attachments/assets/d3283fb2-81a7-4b8d-b6a2-ce5446a17239" />

`view([-1, 3])`의 의미는 다음과 같다.

```text
첫 번째 차원 = PyTorch가 자동으로 추론
두 번째 차원 = 3으로 고정
```

결과적으로 `(4, 3)` 크기의 텐서가 된다.

```text
(2, 2, 3) → (2 × 2, 3) → (4, 3)
```

`view()`를 사용할 때 지켜야 할 규칙은 다음과 같다.

```text
변경 전후의 원소 개수가 유지되어야 한다.
-1은 다른 차원으로부터 자동으로 추론된다.
```

<details>
<summary>view에서 -1의 의미</summary>

`view()`에서 `-1`은 해당 차원의 크기를 PyTorch가 자동으로 계산하게 맡긴다는 의미이다.

예를 들어 원소가 12개인 텐서를 `view([-1, 3])`으로 바꾸면, 두 번째 차원이 3이므로 첫 번째 차원은 자동으로 4가 된다.

</details>

---

### 3차원 텐서의 크기 변경

차원 수를 유지한 채 3차원 텐서의 크기를 변경할 수도 있다.

```python
print(ft.view([-1, 1, 3]))
print(ft.view([-1, 1, 3]).shape)
```

출력:

```text
tensor([[[ 0.,  1.,  2.]],

        [[ 3.,  4.,  5.]],

        [[ 6.,  7.,  8.]],

        [[ 9., 10., 11.]]])
torch.Size([4, 1, 3])
```

기존 텐서의 원소 수는 12개이다.

```text
2 × 2 × 3 = 12
```

변경 후에도 원소 수가 12개여야 한다.

```text
? × 1 × 3 = 12
? = 4
```

따라서 결과 크기는 `(4, 1, 3)`이 된다.

---

### 스퀴즈(Squeeze)

스퀴즈(Squeeze)는 차원이 1인 경우 해당 차원을 제거한다.

```text
squeeze = 크기가 1인 차원 제거
```

예시로 `(3, 1)` 크기의 2차원 텐서를 만든다.

```python
ft = torch.FloatTensor([[0], [1], [2]])

print(ft)
print(ft.shape)
```

출력:

```text
tensor([[0.],
        [1.],
        [2.]])
torch.Size([3, 1])
```

`squeeze()`를 사용하면 두 번째 차원이 제거되어 `(3,)` 크기의 1차원 텐서가 된다.

```python
print(ft.squeeze())
print(ft.squeeze().shape)
```

출력:

```text
tensor([0., 1., 2.])
torch.Size([3])
```

---

### 언스퀴즈(Unsqueeze)

언스퀴즈(Unsqueeze)는 특정 위치에 크기가 1인 차원을 추가한다.

```text
unsqueeze = 특정 위치에 크기가 1인 차원 추가
```

예시로 `(3,)` 크기의 1차원 텐서를 만든다.

```python
ft = torch.Tensor([0, 1, 2])
print(ft.shape)
```

출력:

```text
torch.Size([3])
```

첫 번째 차원에 1인 차원을 추가한다.

```python
print(ft.unsqueeze(0))
print(ft.unsqueeze(0).shape)
```

출력:

```text
tensor([[0., 1., 2.]])
torch.Size([1, 3])
```

같은 작업은 `view()`로도 구현할 수 있다.

```python
print(ft.view(1, -1))
print(ft.view(1, -1).shape)
```

출력:

```text
tensor([[0., 1., 2.]])
torch.Size([1, 3])
```

두 번째 차원에 1인 차원을 추가할 수도 있다.

```python
print(ft.unsqueeze(1))
print(ft.unsqueeze(1).shape)
```

출력:

```text
tensor([[0.],
        [1.],
        [2.]])
torch.Size([3, 1])
```

마지막 차원에 1인 차원을 추가할 수도 있다.

```python
print(ft.unsqueeze(-1))
print(ft.unsqueeze(-1).shape)
```

출력:

```text
tensor([[0.],
        [1.],
        [2.]])
torch.Size([3, 1])
```

`view()`, `squeeze()`, `unsqueeze()`는 텐서의 원소 수를 유지하면서 모양과 차원을 조절한다.

---

### 타입 캐스팅(Type Casting)

텐서에는 자료형이 있다.  
자료형을 변환하는 것을 타입 캐스팅(Type Casting)이라고 한다.

<img width="598" height="254" alt="Image" src="https://github.com/user-attachments/assets/8a6bb80b-63ab-4d7c-9d75-2073c9e5f003" />

예를 들어 32비트 부동 소수점은 `torch.FloatTensor`, 64비트 부호 있는 정수는 `torch.LongTensor`를 사용한다.  
GPU 연산을 위한 자료형도 존재한다.

먼저 `long` 타입의 텐서를 선언한다.

```python
lt = torch.LongTensor([1, 2, 3, 4])
print(lt)
```

`.float()`를 붙이면 float 타입으로 변경된다.

```python
print(lt.float())
```

출력:

```text
tensor([1., 2., 3., 4.])
```

Byte 타입의 텐서를 만든다.

```python
bt = torch.ByteTensor([True, False, False, True])
print(bt)
```

출력:

```text
tensor([1, 0, 0, 1], dtype=torch.uint8)
```

`.long()`을 사용하면 long 타입으로, `.float()`을 사용하면 float 타입으로 변경된다.

```python
print(bt.long())
print(bt.float())
```

출력:

```text
tensor([1, 0, 0, 1])
tensor([1., 0., 0., 1.])
```

---

### 연결하기(Concatenate)

두 텐서를 연결할 때는 `torch.cat()`을 사용한다.

```text
torch.cat() = 텐서 연결
```

먼저 `(2, 2)` 크기의 텐서를 두 개 만든다.

```python
x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])
```

`dim=0`으로 연결하면 첫 번째 차원이 늘어난다.

```python
print(torch.cat([x, y], dim=0))
```

출력:

```text
tensor([[1., 2.],
        [3., 4.],
        [5., 6.],
        [7., 8.]])
```

```text
(2, 2) + (2, 2) → (4, 2)
```

`dim=1`로 연결하면 두 번째 차원이 늘어난다.

```python
print(torch.cat([x, y], dim=1))
```

출력:

```text
tensor([[1., 2., 5., 6.],
        [3., 4., 7., 8.]])
```

```text
(2, 2) + (2, 2) → (2, 4)
```

딥 러닝에서는 모델의 입력 또는 중간 연산에서 두 개의 텐서를 연결하는 경우가 많다.  
두 텐서를 연결해서 입력으로 사용하는 것은 두 가지 정보를 모두 사용한다는 의미를 가진다.

---

### 스택킹(Stacking)

스택킹(Stacking)은 텐서를 쌓는 방식의 연결이다.  
`torch.stack()`을 사용한다.

```text
torch.stack() = 텐서를 새 차원으로 쌓기
```

크기가 `(2,)`로 같은 3개의 벡터를 만든다.

```python
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])
```

3개의 벡터를 스택킹한다.

```python
print(torch.stack([x, y, z]))
```

출력:

```text
tensor([[1., 4.],
        [2., 5.],
        [3., 6.]])
```

결과적으로 `(3, 2)` 텐서가 된다.

<img width="267" height="398" alt="Image" src="https://github.com/user-attachments/assets/44325427-e824-400f-9b32-e517f3817eaf" />

스택킹은 다음 코드를 축약한 것과 같다.

```python
print(torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0))
```

출력:

```text
tensor([[1., 4.],
        [2., 5.],
        [3., 6.]])
```

`x`, `y`, `z`는 원래 `(2,)` 크기였지만, `unsqueeze(0)`을 통해 `(1, 2)` 크기의 2차원 텐서가 된다.  
이후 `cat()`으로 연결하면 `(3, 2)` 텐서가 된다.

`dim=1`을 사용해 두 번째 차원이 증가하도록 스택킹할 수도 있다.

```python
print(torch.stack([x, y, z], dim=1))
```

출력:

```text
tensor([[1., 2., 3.],
        [4., 5., 6.]])
```

결과적으로 `(2, 3)` 텐서가 된다.

<img width="435" height="204" alt="Image" src="https://github.com/user-attachments/assets/ef962fb2-ff96-4324-b2a1-5fdba44bdf5c" />

<details>
<summary>cat과 stack의 차이</summary>

`torch.cat()`은 기존 차원 방향으로 텐서를 이어 붙인다.  
`torch.stack()`은 새로운 차원을 만들어 텐서를 쌓는다.

```text
cat = 기존 차원으로 연결
stack = 새 차원을 만들어 쌓기
```

</details>

---

### `ones_like`와 `zeros_like`

`ones_like()`는 입력 텐서와 같은 크기이지만, 값이 모두 1인 텐서를 만든다.  
`zeros_like()`는 입력 텐서와 같은 크기이지만, 값이 모두 0인 텐서를 만든다.

```text
ones_like = 같은 크기의 1로 채워진 텐서 생성
zeros_like = 같은 크기의 0으로 채워진 텐서 생성
```

예시 텐서를 만든다.

```python
x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(x)
```

출력:

```text
tensor([[0., 1., 2.],
        [2., 1., 0.]])
```

같은 크기의 1로 채워진 텐서를 만든다.

```python
print(torch.ones_like(x))
```

출력:

```text
tensor([[1., 1., 1.],
        [1., 1., 1.]])
```

같은 크기의 0으로 채워진 텐서를 만든다.

```python
print(torch.zeros_like(x))
```

출력:

```text
tensor([[0., 0., 0.],
        [0., 0., 0.]])
```

---

### In-place Operation

In-place Operation은 기존 텐서의 값을 덮어쓰는 연산이다.

```text
In-place Operation = 기존 값을 덮어쓰는 연산
```

먼저 `(2, 2)` 크기의 텐서를 만든다.

```python
x = torch.FloatTensor([[1, 2], [3, 4]])
```

일반 곱셈 연산은 기존 `x` 값을 바꾸지 않는다.

```python
print(x.mul(2.))
print(x)
```

출력:

```text
tensor([[2., 4.],
        [6., 8.]])
tensor([[1., 2.],
        [3., 4.]])
```

연산 뒤에 `_`를 붙이면 기존 값을 덮어쓴다.

```python
print(x.mul_(2.))
print(x)
```

출력:

```text
tensor([[2., 4.],
        [6., 8.]])
tensor([[2., 4.],
        [6., 8.]])
```

`mul_(2.)`는 곱하기 2를 수행한 결과를 다시 `x`에 저장한다.

<details>
<summary>언더스코어(_)가 붙은 연산</summary>

PyTorch에서 함수 이름 뒤에 `_`가 붙은 연산은 보통 기존 텐서 값을 직접 변경하는 In-place Operation이다.

```text
mul() = 결과를 새로 반환
mul_() = 기존 텐서 값을 직접 변경
```

</details>

---

## 3. 파이썬 클래스(Python Class)

### 파이썬 클래스를 배우는 이유

대부분의 PyTorch 구현체들은 기본적으로 **클래스(Class)** 개념을 자주 사용한다.

이번 절에서는 함수(function)와 클래스(Class)의 차이를 덧셈기 예제를 통해 이해한다.

```text
파이썬 클래스 학습 흐름
├─ 함수로 덧셈기 구현
├─ 함수로 두 개의 덧셈기 구현
└─ 클래스로 덧셈기 구현
```

---

### 함수(function)와 클래스(Class)의 차이

함수와 클래스의 차이를 이해하기 위해, 덧셈을 지속적으로 수행할 수 있는 도구를 함수와 클래스로 각각 만들어본다.

```text
함수 = 특정 동작을 수행하는 코드 묶음
클래스 = 객체를 만들기 위한 설계도
```

<details>
<summary>객체란?</summary>

객체는 클래스를 바탕으로 실제로 만들어진 독립적인 대상을 의미한다.

예를 들어 `Calculator`라는 클래스를 만들고, `cal1 = Calculator()`를 실행하면 `cal1`이라는 객체가 만들어진다.

```text
클래스 = 설계도
객체 = 설계도로 만든 실제 대상
```

</details>

---

### 함수로 덧셈기 구현하기

먼저 함수(function)를 사용하여 덧셈기를 구현한다.

전역 변수 `result`를 선언한다.

```python
result = 0
```

그리고 `add()` 함수를 만든다.  
`add()` 함수는 기존 `result`에 함수의 인자로 들어온 숫자를 더하고, 그 값을 리턴한다.

```python
def add(num):
    global result
    result += num
    return result
```

함수를 두 번 실행한다.

```python
print(add(3))
print(add(4))
```

출력:

```text
3
7
```

처음에는 `result`가 0이다.  
여기에 3을 더하면 `result`가 3이 되므로 3이 출력된다.

그다음 4를 입력하면, 이미 `result`가 3으로 갱신된 상태이므로 `3 + 4`의 결과인 7이 출력된다.

```text
처음 result = 0
add(3) 실행 후 result = 3
add(4) 실행 후 result = 7
```

<details>
<summary>전역 변수(global variable)란?</summary>

전역 변수는 함수 밖에서 선언되어 여러 곳에서 접근할 수 있는 변수이다.

위 예제에서는 `result`가 함수 밖에서 선언되었고, `add()` 함수 안에서 `global result`를 통해 사용된다.

</details>

---

### 함수로 두 개의 덧셈기 구현하기

이번에는 독립적인 두 개의 덧셈기를 만든다.

비유하면 책상에 계산기 두 개를 두고, 서로 다른 계산을 하는 상황이다.

```text
첫 번째 계산기 = 3 + 4
두 번째 계산기 = 3 + 7
```

두 계산기는 서로 다른 계산기이므로 독립적이어야 한다.

함수로 이를 구현하려면 각각의 결과를 저장할 변수와 함수를 따로 만들어야 한다.

```python
result1 = 0
result2 = 0

def add1(num):
    global result1
    result1 += num
    return result1

def add2(num):
    global result2
    result2 += num
    return result2
```

각 함수를 실행한다.

```python
print(add1(3))
print(add1(4))
print(add2(3))
print(add2(7))
```

출력:

```text
3
7
3
10
```

`add1()`과 `add2()`는 서로 다른 전역 변수를 사용한다.  
그래서 서로의 값에 영향을 주지 않고 독립적으로 연산된다.

```text
add1의 결과 = 3 → 7
add2의 결과 = 3 → 10
```

하지만 덧셈기가 여러 개 필요해질수록 변수와 함수도 계속 늘어나게 된다.

---

### 클래스로 덧셈기 구현하기

이번에는 클래스(class)를 사용하여 덧셈기를 구현한다.

```python
class Calculator:
    def __init__(self):
        self.result = 0

    def add(self, num):
        self.result += num
        return self.result
```

`Calculator` 클래스는 덧셈기를 만들기 위한 설계도라고 볼 수 있다.

```text
Calculator 클래스 = 덧셈기 설계도
```

<details>
<summary>__init__이란?</summary>

`__init__`은 객체가 생성될 때 자동으로 실행되는 초기화 함수이다.  
이를 생성자(Constructor)라고도 한다.

위 코드에서는 객체가 생성될 때 `self.result = 0`을 실행하여 결과값을 0으로 초기화한다.

</details>

<details>
<summary>self란?</summary>

`self`는 생성된 객체 자기 자신을 가리킨다.

예를 들어 `cal1.add(3)`을 실행하면, 이때 `self`는 `cal1`을 의미한다고 볼 수 있다.  
그래서 `self.result`는 각 객체가 독립적으로 가지는 결과값이다.

</details>

---

### 객체 생성하기

클래스는 붕어빵 틀과 같다.  
클래스를 만든 후에는 그 클래스를 바탕으로 객체를 만들 수 있다.

먼저 `cal1` 객체를 만든다.

```python
cal1 = Calculator()
```

객체의 생성 방법은 다음과 같다.

```text
객체이름 = 클래스이름()
```

하나의 클래스로 여러 개의 객체를 만들 수 있다.  
이번에는 `cal2` 객체도 만든다.

```python
cal2 = Calculator()
```

```text
Calculator 클래스
├─ cal1 객체
└─ cal2 객체
```

---

### 클래스 객체로 독립적인 덧셈 연산하기

두 개의 객체에 대해 독립적으로 덧셈 연산을 수행한다.

```python
print(cal1.add(3))
print(cal1.add(4))
print(cal2.add(3))
print(cal2.add(7))
```

출력:

```text
3
7
3
10
```

`cal1`과 `cal2`는 같은 `Calculator` 클래스로 만들어졌지만, 서로 독립적인 객체이다.

```text
cal1.result = 3 → 7
cal2.result = 3 → 10
```

따라서 `cal1`의 계산 결과는 `cal2`에 영향을 주지 않는다.

함수로 독립적인 두 개의 덧셈기를 만들려면 함수와 변수를 각각 따로 만들어야 했다.  
하지만 클래스를 사용하면 하나의 클래스에서 여러 객체를 생성하여 더 간결하게 구현할 수 있다.

```text
함수 방식 = 덧셈기마다 변수와 함수가 필요
클래스 방식 = 하나의 클래스로 여러 객체 생성 가능
```

<details>
<summary>클래스를 사용하는 이유</summary>

클래스를 사용하면 같은 구조와 기능을 가진 객체를 여러 개 만들 수 있다.

각 객체는 자기만의 데이터를 가질 수 있으므로, 서로 독립적인 상태를 유지할 수 있다.

PyTorch에서는 모델, 레이어, 데이터셋 등을 클래스로 정의하는 경우가 많기 때문에 클래스 개념을 이해하는 것이 중요하다.

</details>

---

<details>
<summary>PyTorch에서 클래스가 중요한 이유</summary>

PyTorch에서는 신경망 모델을 만들 때 보통 `torch.nn.Module`을 상속받아 클래스로 모델을 정의한다.

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
```

즉, 클래스는 단순한 파이썬 문법이 아니라 PyTorch 모델 구조를 이해하기 위한 기본 개념이다.

`torch.nn.Module`은 PyTorch에서 신경망 모델이나 레이어를 만들 때 사용하는 기본 클래스이다.

직접 모델을 만들 때는 보통 `nn.Module`을 상속받고, `__init__()`에서 사용할 레이어를 정의한 뒤 `forward()`에서 입력 데이터가 어떤 순서로 지나가는지 작성한다.

```text
__init__() = 모델에 필요한 레이어 정의
forward() = 입력 데이터의 계산 흐름 정의
```

</details>

---

## 최종 핵심 정리

```text
PyTorch 패키지 구성
├─ torch = 텐서와 수학 함수가 포함된 메인 네임스페이스
├─ torch.autograd = 자동 미분 기능 제공
├─ torch.nn = 신경망 구성 요소 제공
├─ torch.optim = SGD 등 최적화 알고리즘 제공
├─ torch.utils.data = 미니 배치용 유틸리티 함수 제공
└─ torch.onnx = ONNX 포맷 익스포트 지원
```

```text
벡터, 행렬, 텐서
├─ 스칼라 = 숫자 하나
├─ 벡터 = 1차원 텐서
├─ 행렬 = 2차원 텐서
└─ 텐서 = 다차원 배열
```

```text
2D Tensor
= (batch_size, dim)
```

```text
3D Tensor - Computer Vision
= 개념 설명: (batch_size, width, height)
```

```text
3D Tensor - NLP
= (batch_size, length, dim)
```

```text
NumPy 텐서
├─ np.array() = 텐서 생성
├─ ndim = 차원 확인
├─ shape = 크기 확인
├─ indexing = 특정 원소 선택
└─ slicing = 범위 선택
```

```text
PyTorch 텐서
├─ torch.FloatTensor() = 텐서 생성
├─ dim() = 차원 확인
├─ shape = 크기 확인
└─ size() = 크기 확인
```

```text
브로드캐스팅
= 크기가 다른 텐서의 크기를 자동으로 맞춰 연산하는 기능
```

```text
곱셈
├─ matmul() = 행렬 곱셈
└─ mul() 또는 * = 원소별 곱셈
```

```text
기본 연산
├─ mean() = 평균
├─ sum() = 합
├─ max() = 최댓값
└─ argmax = 최댓값의 인덱스
```

```text
텐서 모양 조작
├─ view() = 원소 수를 유지하면서 shape 변경
├─ squeeze() = 크기가 1인 차원 제거
└─ unsqueeze() = 크기가 1인 차원 추가
```

```text
텐서 결합
├─ torch.cat() = 기존 차원으로 연결
└─ torch.stack() = 새 차원으로 쌓기
```

```text
기타 기능
├─ 타입 캐스팅 = 텐서 자료형 변환
├─ ones_like() = 같은 크기의 1 텐서 생성
├─ zeros_like() = 같은 크기의 0 텐서 생성
└─ In-place Operation = 기존 값을 덮어쓰는 연산
```

```text
파이썬 클래스
├─ 함수 = 특정 동작을 수행하는 코드 묶음
├─ 클래스 = 객체를 만들기 위한 설계도
└─ 객체 = 클래스로부터 만들어진 독립적인 대상
```

```text
함수 방식 덧셈기
= 독립적인 덧셈기가 여러 개 필요하면 변수와 함수도 각각 필요
```

```text
클래스 방식 덧셈기
= 하나의 클래스로 여러 객체를 만들어 독립적인 연산 가능
```

```text
클래스 주요 개념
├─ __init__ = 객체 생성 시 실행되는 초기화 함수
├─ self = 생성된 객체 자기 자신
└─ self.result = 각 객체가 독립적으로 가지는 값
```