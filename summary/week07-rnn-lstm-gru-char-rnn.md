# 순환 신경망(RNN), LSTM·GRU와 문자 단위 RNN 정리


## 0. 전체 흐름 먼저 보기

이번 정리는 `순환 신경망의 기본 구조`, `PyTorch의 RNN 구현`, `깊은·양방향 RNN`, `바닐라 RNN의 한계`, `LSTM과 GRU`, `문자 단위 RNN 실습`까지 이어지는 내용이다.

전체 흐름은 다음과 같다.

```text
1. RNN은 순서가 있는 시퀀스를 처리하며, 이전 시점의 은닉 상태를 현재 계산에 사용한다.
2. 하나의 RNN 셀을 시간축으로 펼치면 여러 시점의 입력, 은닉 상태, 출력이 연결된 구조가 된다.
3. 입력·출력 길이에 따라 one-to-many, many-to-one, many-to-many 구조로 설계할 수 있다.
4. 바닐라 RNN의 은닉 상태는 현재 입력과 이전 은닉 상태의 가중합에 tanh를 적용해 계산한다.
5. RNN의 가중치는 모든 시점에서 공유되며, NumPy 반복문으로 순전파를 직접 구현할 수 있다.
6. PyTorch의 nn.RNN은 모든 시점의 마지막 층 출력과 각 층의 최종 은닉 상태를 반환한다.
7. num_layers를 늘리면 깊은 RNN이 되고, bidirectional=True를 사용하면 양쪽 문맥을 처리한다.
8. 바닐라 RNN은 긴 시퀀스에서 앞부분의 정보를 유지하기 어려운 장기 의존성 문제가 있다.
9. LSTM은 셀 상태와 입력·망각·출력 게이트를 이용해 기억의 유지, 삭제, 추가, 출력을 조절한다.
10. GRU는 리셋·업데이트 게이트와 하나의 은닉 상태를 사용해 LSTM보다 단순한 구조로 정보를 갱신한다.
11. 문자 단위 RNN은 문자를 시점 단위로 입력하고 다음 문자를 예측하는 many-to-many 모델이다.
12. 문자 분류에서는 각 시점의 logit과 정답 인덱스를 펼쳐 CrossEntropyLoss로 학습한다.
```

RNN 계열 모델의 기본 정보 흐름은 다음과 같다.

```text
현재 입력 x_t + 이전 은닉 상태 h_{t-1}
                  ↓
             순환 셀 계산
                  ↓
          현재 은닉 상태 h_t
             ├─→ 현재 출력
             └─→ 다음 시점으로 전달
```

바닐라 RNN에서 LSTM·GRU로 이어지는 흐름은 다음과 같다.

```text
바닐라 RNN
   ↓
긴 시퀀스에서 장기 의존성 학습이 어려움
   ↓
게이트를 이용해 정보 흐름을 선택적으로 조절
   ├─→ LSTM: 셀 상태 + 입력·망각·출력 게이트
   └─→ GRU: 은닉 상태 + 리셋·업데이트 게이트
```

문자 단위 RNN의 학습 흐름은 다음과 같다.

```text
문자 시퀀스
   ↓
문자 인덱스 변환
   ↓
원-핫 벡터 변환
   ↓
RNN의 시점별 은닉 상태 계산
   ↓
Linear 계층으로 문자별 logit 출력
   ↓
다음 문자 인덱스와 CrossEntropyLoss 계산
```

이번 정리에서 사용하는 주요 기호는 다음과 같다.

```text
x_t: 시점 t의 입력 벡터
h_t: 시점 t의 은닉 상태
h_{t-1}: 이전 시점의 은닉 상태
y_t: 시점 t의 출력
W_x: 입력에서 은닉 상태로 연결되는 가중치
W_h: 이전 은닉 상태에서 현재 은닉 상태로 연결되는 가중치
W_y: 은닉 상태에서 출력으로 연결되는 가중치
b: 편향
T 또는 time_steps: 시퀀스의 전체 시점 수
d 또는 input_size: 각 시점의 입력 벡터 차원
D_h 또는 hidden_size: 은닉 상태의 차원
num_layers: 순환 은닉층의 수
num_directions: 단방향이면 1, 양방향이면 2
C_t: LSTM의 현재 셀 상태
f_t: LSTM의 망각 게이트
i_t: LSTM의 입력 게이트
o_t: LSTM의 출력 게이트
g_t: 새로 저장하거나 반영할 후보 정보
r_t: GRU의 리셋 게이트
z_t: GRU의 업데이트 게이트
vocab_size: 문자 집합의 크기이자 원-핫 입력 차원 및 문자 클래스 수
```

---

## 1. 순환 신경망(Recurrent Neural Network, RNN)

`순환 신경망(Recurrent Neural Network, RNN)`은 입력과 출력을 **시퀀스(sequence) 단위**로 처리하기 위해 설계된 신경망이다.

시퀀스는 순서가 있는 데이터의 나열을 의미한다.

```text
문장: 단어가 순서대로 나열된 시퀀스
음성: 시간에 따라 변화하는 신호 시퀀스
주가: 시간 순서대로 기록된 수치 시퀀스
영상: 프레임이 시간 순서대로 나열된 시퀀스
```

예를 들어 번역기는 입력 문장을 구성하는 단어 시퀀스를 받아, 번역된 문장의 단어 시퀀스를 출력한다.

```text
입력 문장
   ↓
단어 시퀀스
   ↓
시퀀스 모델
   ↓
번역된 단어 시퀀스
   ↓
출력 문장
```

RNN은 딥러닝에서 사용되는 가장 기본적인 시퀀스 모델 중 하나이다.

> `순환 신경망(Recurrent Neural Network)`과 `재귀 신경망(Recursive Neural Network)`은 이름이 비슷하지만 서로 다른 구조이다.

---

### 1.1 피드 포워드 신경망과 RNN의 차이

앞에서 다룬 일반적인 신경망에서는 은닉층의 출력이 다음 계층이나 출력층 방향으로만 전달된다.

이처럼 정보가 입력층에서 출력층 방향으로만 흐르는 신경망을 `피드 포워드 신경망(Feed Forward Neural Network)`이라고 한다.

```text
입력층
   ↓
은닉층
   ↓
출력층
```

반면 RNN의 은닉층에서는 현재 시점에서 계산한 결과를 출력층으로 전달하는 동시에, 다음 시점의 은닉층 계산에도 다시 사용한다.

```text
현재 시점의 입력
   ↓
현재 시점의 은닉 상태
   ├─→ 현재 시점의 출력
   └─→ 다음 시점의 은닉 상태 계산
```

따라서 RNN은 이전 시점의 정보를 이후 시점의 계산에 반영할 수 있다.

<img width="192" height="60" alt="Image" src="https://github.com/user-attachments/assets/a92455cc-5e5d-40f1-9d12-e2c858cb0773" />

---

### 1.2 RNN 셀과 메모리 셀

RNN의 은닉층에서 활성화 함수를 적용해 은닉 상태를 계산하는 단위를 `셀(cell)`이라고 한다.

이 셀은 이전 시점에서 계산된 값을 현재 시점의 입력으로 사용하므로, 과거 정보를 저장하고 전달하는 일종의 메모리 역할을 수행한다.

따라서 다음과 같은 표현을 사용한다.

```text
RNN 셀
메모리 셀
순환 셀
```

현재 시점을 \(t\)라고 하면, \(t\) 시점의 셀은 다음 두 값을 입력으로 사용한다.

```text
현재 시점의 입력값: x_t
이전 시점의 은닉 상태: h_{t-1}
```

그리고 다음 두 방향으로 값을 전달한다.

```text
현재 시점의 출력층 방향
다음 시점 t+1의 RNN 셀 방향
```

---

### 1.3 은닉 상태(Hidden State)

RNN의 메모리 셀이 출력층과 다음 시점의 셀로 전달하는 값을 `은닉 상태(hidden state)`라고 한다.

현재 시점 \(t\)의 은닉 상태를 \(h_t\)라고 하면, \(h_t\)는 다음 정보의 영향을 받아 계산된다.

```text
현재 입력 x_t
이전 은닉 상태 h_{t-1}
```

즉, 다음 관계가 성립한다.

```text
h_{t-1}
   ↓
현재 입력 x_t와 함께 사용
   ↓
현재 은닉 상태 h_t 계산
   ↓
다음 시점 h_{t+1} 계산에 사용
```

이 구조 때문에 현재 시점의 은닉 상태에는 이전 시점들에서 전달된 정보가 누적되어 반영될 수 있다.

---

### 1.4 접힌 표현과 펼친 표현

RNN은 크게 두 가지 방식으로 표현할 수 있다.

첫 번째는 하나의 셀에 순환 화살표를 그리는 방식이다.

```text
하나의 RNN 셀
   ↺
이전 은닉 상태가 다시 셀의 입력으로 전달됨
```

두 번째는 시간의 흐름에 따라 동일한 셀을 여러 개로 펼쳐서 표현하는 방식이다.

```text
시점 1          시점 2          시점 3
 x_1             x_2             x_3
  ↓               ↓               ↓
 h_1 ─────────→  h_2 ─────────→  h_3
  ↓               ↓               ↓
 y_1             y_2             y_3
```

두 그림은 서로 다른 모델이 아니다.

```text
순환 화살표 표현:
하나의 셀이 반복적으로 사용된다는 점을 강조

시간축으로 펼친 표현:
각 시점에서 입력, 은닉 상태, 출력이 어떻게 연결되는지 강조
```

<img width="428" height="198" alt="Image" src="https://github.com/user-attachments/assets/c90d21f6-a4d4-4e16-96cf-8635905e7c7e" />

---

### 1.5 RNN에서 사용하는 벡터 단위

피드 포워드 신경망에서는 각 계산 단위를 `뉴런`이라고 설명하는 경우가 많다.

RNN에서는 다음과 같이 벡터 단위의 표현을 주로 사용한다.

```text
입력층: 입력 벡터 x_t
은닉층: 은닉 상태 벡터 h_t
출력층: 출력 벡터 y_t
```

따라서 RNN 구조도에서 하나의 네모는 일반적으로 하나의 스칼라 뉴런이 아니라 여러 값을 포함하는 벡터를 나타낸다.

예를 들어 다음과 같은 RNN을 생각할 수 있다.

```text
입력 벡터 차원: 4
은닉 상태 크기: 2
출력 벡터 차원: 2
시점의 수: 2
```

이를 뉴런 단위로 해석하면 다음과 같다.

```text
입력층 뉴런 수: 4
은닉층 뉴런 수: 2
출력층 뉴런 수: 2
```

<img width="436" height="198" alt="Image" src="https://github.com/user-attachments/assets/97768f62-e94a-48c6-ab21-1cdab3a4502a" />

---

## 2. 입력과 출력 길이에 따른 RNN 구조

RNN은 입력 시퀀스와 출력 시퀀스의 길이를 서로 다르게 설계할 수 있다.

대표적인 형태는 다음과 같다.

| 구조 | 입력 길이 | 출력 길이 | 대표 작업 |
|---|---:|---:|---|
| 일 대 일(one-to-one) | 1 | 1 | 일반적인 고정 크기 입력 분류 |
| 일 대 다(one-to-many) | 1 | 여러 개 | 이미지 캡셔닝 |
| 다 대 일(many-to-one) | 여러 개 | 1 | 감성 분류, 스팸 메일 분류 |
| 다 대 다(many-to-many) | 여러 개 | 여러 개 | 번역, 챗봇, 개체명 인식, 품사 태깅 |

<img width="417" height="139" alt="Image" src="https://github.com/user-attachments/assets/003d0839-ed2c-49fa-a409-4ea0abd196c8" />

---

### 2.1 일 대 다(One-to-Many)

`일 대 다(one-to-many)` 구조는 하나의 입력으로부터 여러 시점의 출력을 생성한다.

```text
하나의 입력
   ↓
RNN
   ↓
출력 1 → 출력 2 → 출력 3 → ...
```

대표적인 예시는 `이미지 캡셔닝(Image Captioning)`이다.

이미지 캡셔닝에서는 하나의 이미지를 입력으로 받아, 이미지를 설명하는 여러 단어를 순서대로 출력한다.

```text
입력:
하나의 이미지

출력:
"A dog is running"
```

출력 문장은 여러 단어로 이루어진 시퀀스이므로 일 대 다 구조로 볼 수 있다.

---

### 2.2 다 대 일(Many-to-One)

`다 대 일(many-to-one)` 구조는 여러 시점의 입력을 모두 처리한 뒤 하나의 결과를 출력한다.

```text
입력 1 → 입력 2 → 입력 3 → ...
                         ↓
                    하나의 출력
```

대표적인 작업은 다음과 같다.

```text
감성 분류:
문장이 긍정인지 부정인지 분류

스팸 메일 분류:
메일이 정상 메일인지 스팸 메일인지 분류
```

문장을 구성하는 단어들이 순서대로 입력되고, 마지막에는 문장 전체에 대한 하나의 클래스가 출력된다.

<img width="216" height="216" alt="Image" src="https://github.com/user-attachments/assets/4ee3b5d1-511e-4243-be8a-f928f2dcd65e" />

---

### 2.3 다 대 다(Many-to-Many)

`다 대 다(many-to-many)` 구조는 여러 시점의 입력을 받아 여러 시점의 출력을 생성한다.

대표적인 작업은 다음과 같다.

```text
챗봇
기계 번역
개체명 인식
품사 태깅
```

다 대 다 구조는 출력 시점의 정렬 방식에 따라 다시 구분할 수 있다.

```text
입력과 출력이 시점별로 대응:
개체명 인식, 품사 태깅

입력 시퀀스를 모두 처리한 후 출력 시퀀스 생성:
기계 번역, 챗봇
```

개체명 인식에서는 각 입력 단어마다 하나의 태그를 출력한다.

```text
입력 단어: EU       rejects   GERMAN   call
출력 태그: 조직이름  해당없음  나라이름  해당없음
```

<img width="225" height="214" alt="Image" src="https://github.com/user-attachments/assets/93820bc2-e8b7-4bee-98a4-929fbc3a3a9f" />

---

## 3. RNN의 은닉 상태 계산

현재 시점 \(t\)의 RNN 셀은 다음 두 입력을 이용해 은닉 상태 \(h_t\)를 계산한다.

```text
현재 시점의 입력 벡터 x_t
이전 시점의 은닉 상태 h_{t-1}
```

이를 위해 RNN 셀에는 두 종류의 가중치가 존재한다.

```text
W_x:
현재 입력 x_t에 적용되는 가중치

W_h:
이전 은닉 상태 h_{t-1}에 적용되는 가중치
```

<img width="122" height="195" alt="Image" src="https://github.com/user-attachments/assets/cc5e2e6e-e62c-4fea-a5c5-b335b7b8676a" />

---

### 3.1 은닉 상태 수식

현재 시점의 은닉 상태는 다음과 같이 계산한다.

```math
h_t = \tanh\left(W_xx_t + W_hh_{t-1} + b\right)
```

각 기호의 의미는 다음과 같다.

```text
x_t:
현재 시점 t의 입력 벡터

h_{t-1}:
이전 시점 t-1의 은닉 상태 벡터

W_x:
입력 벡터에 적용되는 가중치 행렬

W_h:
이전 은닉 상태에 적용되는 가중치 행렬

b:
은닉층의 편향 벡터

h_t:
현재 시점 t의 은닉 상태 벡터
```

은닉 상태 계산 과정은 다음과 같다.

```text
현재 입력 x_t에 W_x를 곱함
   ↓
이전 은닉 상태 h_{t-1}에 W_h를 곱함
   ↓
두 결과와 편향 b를 더함
   ↓
tanh 활성화 함수 적용
   ↓
현재 은닉 상태 h_t 계산
```

---

### 3.2 은닉 상태 계산의 행렬 크기

입력 벡터의 차원을 \(d\), 은닉 상태의 크기를 \(D_h\)라고 하자.

각 벡터와 행렬의 크기는 다음과 같다.

```math
x_t \in \mathbb{R}^{d \times 1}
```

```math
W_x \in \mathbb{R}^{D_h \times d}
```

```math
W_h \in \mathbb{R}^{D_h \times D_h}
```

```math
h_{t-1} \in \mathbb{R}^{D_h \times 1}
```

```math
b \in \mathbb{R}^{D_h \times 1}
```

```math
h_t \in \mathbb{R}^{D_h \times 1}
```

행렬 곱의 크기를 확인하면 다음과 같다.

```text
W_x x_t:
(D_h × d) × (d × 1)
= D_h × 1

W_h h_{t-1}:
(D_h × D_h) × (D_h × 1)
= D_h × 1
```

두 결과와 편향의 크기가 모두 \(D_h \times 1\)이므로 서로 더할 수 있다.

```text
W_xx_t + W_hh_{t-1} + b
= (D_h × 1) + (D_h × 1) + (D_h × 1)
= D_h × 1
```

마지막으로 `tanh`를 원소별로 적용하면 \(h_t\)의 크기도 \(D_h \times 1\)이 된다.

<img width="604" height="118" alt="Image" src="https://github.com/user-attachments/assets/de1043d8-df24-4df3-baeb-29bf8bb24d27" />

---

### 3.3 예시: 입력 차원과 은닉 상태 크기가 모두 4인 경우

입력 차원 \(d=4\), 은닉 상태 크기 \(D_h=4\)라고 가정하면 각 크기는 다음과 같다.

```text
x_t:       4 × 1
W_x:       4 × 4
W_h:       4 × 4
h_{t-1}:   4 × 1
b:         4 × 1
h_t:       4 × 1
```

은닉 상태 계산은 다음과 같이 이루어진다.

```text
W_x x_t:
(4 × 4) × (4 × 1)
= 4 × 1

W_h h_{t-1}:
(4 × 4) × (4 × 1)
= 4 × 1

최종 합:
(4 × 1) + (4 × 1) + (4 × 1)
= 4 × 1

tanh 적용 결과:
h_t의 크기 = 4 × 1
```

---

### 3.4 은닉 상태 활성화 함수

RNN의 은닉 상태를 계산할 때는 일반적으로 `하이퍼볼릭 탄젠트 함수(tanh)`를 사용한다.

```math
\tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}
```

`tanh`의 출력 범위는 다음과 같다.

```text
-1 < tanh(z) < 1
```

따라서 은닉 상태의 값이 일정 범위 안에 있도록 제한할 수 있다.

자료에서 제시한 기본 RNN은 다음과 같이 표현된다.

```math
h_t = \tanh\left(W_xx_t + W_hh_{t-1} + b\right)
```

`tanh` 대신 ReLU를 사용하는 시도도 가능하지만, 기본적인 RNN 설명에서는 주로 `tanh`가 사용된다.

---

## 4. RNN의 출력 계산

현재 시점의 출력 \(y_t\)는 현재 은닉 상태 \(h_t\)를 이용해 계산한다.

자료의 표기 방식은 다음과 같다.

```math
y_t = f\left(W_yh_t + b\right)
```

편향을 은닉층과 출력층에서 구분해 표기하면 다음처럼 나타낼 수 있다.

```math
y_t = f\left(W_yh_t + b_y\right)
```

각 기호의 의미는 다음과 같다.

```text
h_t:
현재 시점의 은닉 상태

W_y:
은닉 상태를 출력 공간으로 변환하는 가중치 행렬

b 또는 b_y:
출력층의 편향 벡터

f:
출력 문제에 맞게 선택하는 비선형 활성화 함수

y_t:
현재 시점의 출력 벡터
```

---

### 4.1 문제에 따른 출력 활성화 함수

출력층의 활성화 함수는 수행하려는 작업에 따라 달라진다.

| 문제 유형 | 출력 형태 | 대표 활성화 함수 |
|---|---|---|
| 이진 분류 | 두 클래스 중 하나 | 시그모이드(sigmoid) |
| 다중 클래스 분류 | 여러 클래스 중 하나 | 소프트맥스(softmax) |
| 회귀 | 연속적인 수치 | 활성화 함수 생략 또는 문제에 맞는 함수 |

예를 들어 감성 분류에서 문장이 긍정인지 부정인지 판별한다면 시그모이드를 사용할 수 있다.

```text
출력값이 0에 가까움: 한 클래스
출력값이 1에 가까움: 다른 클래스
```

각 시점에서 여러 단어 후보 중 하나를 선택해야 하는 경우에는 소프트맥스를 사용할 수 있다.

```text
출력층의 각 원소:
각 단어 또는 클래스에 대한 logit

softmax 적용:
각 클래스에 대한 확률 분포
```

다만 PyTorch에서 `nn.CrossEntropyLoss()`로 학습할 때는 모델의 마지막에 softmax를 직접 추가하지 않는다.

```text
학습 시:
모델이 정규화되지 않은 logit 출력
   ↓
CrossEntropyLoss 내부에서 log-softmax 처리
   ↓
정답 클래스 인덱스와 손실 계산
```

최종 클래스만 선택할 때도 softmax는 값의 대소 관계를 바꾸지 않으므로 logit에 바로 `argmax`를 적용할 수 있다.

```text
확률 분포가 필요함:
softmax(logits) 사용

CrossEntropyLoss 학습 또는 클래스 선택만 필요함:
logit을 그대로 사용
```

---

## 5. 시간축에 따른 가중치 공유

RNN에서 사용하는 가중치 \(W_x\), \(W_h\), \(W_y\)는 모든 시점에서 동일하게 공유된다.

```text
시점 1:
W_x, W_h, W_y 사용

시점 2:
동일한 W_x, W_h, W_y 사용

시점 3:
동일한 W_x, W_h, W_y 사용
```

시간축으로 펼친 그림에서 RNN 셀이 여러 개 존재하는 것처럼 보이지만, 실제로는 같은 셀과 같은 가중치를 반복해서 사용하는 구조이다.

```text
시점마다 새로운 모델을 만드는 것이 아님
하나의 RNN 셀을 시간축을 따라 반복 적용
```

가중치를 공유하면 입력 시퀀스의 길이가 달라져도 동일한 연산 규칙을 적용할 수 있다.

또한 시점마다 별도의 가중치를 사용하는 것보다 학습해야 하는 매개변수 수를 줄일 수 있다.

---

### 5.1 여러 은닉층을 사용하는 경우

RNN의 은닉층을 두 층 이상 쌓을 수도 있다.

이 경우 같은 층의 가중치는 시간축 전체에서 공유되지만, 서로 다른 은닉층은 서로 다른 가중치를 사용한다.

```text
첫 번째 RNN 은닉층:
자신의 W_x^(1), W_h^(1) 사용

두 번째 RNN 은닉층:
자신의 W_x^(2), W_h^(2) 사용
```

즉, 다음 두 개념을 구분해야 한다.

```text
시간축 방향:
같은 층의 가중치를 모든 시점에서 공유

층 방향:
서로 다른 층은 서로 다른 가중치를 가짐
```

---

## 6. RNN의 시점별 계산 과정

시퀀스 입력이 \(x_1, x_2, x_3\)으로 주어졌다고 하자.

첫 번째 시점에서는 초기 은닉 상태 \(h_0\)와 첫 번째 입력 \(x_1\)을 사용한다.

```math
h_1 = \tanh\left(W_xx_1 + W_hh_0 + b\right)
```

```math
y_1 = f\left(W_yh_1 + b_y\right)
```

두 번째 시점에서는 이전에 계산한 \(h_1\)과 새로운 입력 \(x_2\)를 사용한다.

```math
h_2 = \tanh\left(W_xx_2 + W_hh_1 + b\right)
```

```math
y_2 = f\left(W_yh_2 + b_y\right)
```

세 번째 시점에서는 \(h_2\)와 \(x_3\)을 사용한다.

```math
h_3 = \tanh\left(W_xx_3 + W_hh_2 + b\right)
```

```math
y_3 = f\left(W_yh_3 + b_y\right)
```

전체 흐름은 다음과 같다.

```text
h_0 + x_1
   ↓
h_1 생성
   ├─→ y_1 계산
   └─→ 다음 시점으로 전달

h_1 + x_2
   ↓
h_2 생성
   ├─→ y_2 계산
   └─→ 다음 시점으로 전달

h_2 + x_3
   ↓
h_3 생성
   └─→ y_3 계산
```

이처럼 RNN은 이전 시점의 은닉 상태를 현재 시점의 계산에 반복적으로 사용한다.

---

## 7. NumPy로 RNN 계층 구현하기

앞에서 정의한 기본 RNN의 은닉 상태 계산식은 다음과 같다.

```math
h_t=\tanh\left(W_xx_t+W_hh_{t-1}+b\right)
```

현재 시점 \(t\)의 은닉 상태 \(h_t\)는 다음 두 정보를 함께 사용해 계산한다.

```text
현재 시점의 입력 벡터: x_t
이전 시점의 은닉 상태: h_{t-1}
```

동일한 연산을 시퀀스의 각 시점에 반복하면서, 이전에 계산한 은닉 상태를 다음 시점으로 전달한다.

---

### 7.1 RNN 계산의 의사 코드

RNN의 반복 구조를 의사 코드로 표현하면 다음과 같다.

```python
# 의사 코드이므로 실제로 바로 실행되는 코드는 아니다.

hidden_state_t = 0  # 초기 은닉 상태를 0 벡터로 초기화

for input_t in input_sequence:
    output_t = tanh(input_t, hidden_state_t)
    hidden_state_t = output_t
```

각 변수의 의미는 다음과 같다.

```text
input_sequence:
전체 입력 시퀀스

input_t:
현재 시점 t에 들어오는 입력 벡터

hidden_state_t:
현재까지 계산된 은닉 상태

output_t:
현재 입력과 이전 은닉 상태를 이용해 계산한 새로운 은닉 상태
```

실제 계산에서는 `tanh(input_t, hidden_state_t)`처럼 두 값을 함수에 직접 전달하는 것이 아니라, 가중치와 편향을 사용한다.

```math
h_t=\tanh\left(W_xx_t+W_hh_{t-1}+b\right)
```

반복 과정은 다음과 같다.

```text
초기 은닉 상태 h_0 설정
   ↓
x_1과 h_0으로 h_1 계산
   ↓
x_2와 h_1으로 h_2 계산
   ↓
x_3과 h_2으로 h_3 계산
   ↓
시퀀스의 마지막 시점까지 반복
```

입력 데이터의 길이는 RNN이 처리해야 하는 전체 시점의 수인 `timesteps`와 같다.

---

### 7.2 입력 크기와 초기 은닉 상태 정의하기

NumPy를 불러온다.

```python
import numpy as np
```

RNN 계산에 사용할 크기를 정의한다.

```python
timesteps = 10   # 전체 시점의 수
input_size = 4   # 각 시점에 들어오는 입력 벡터의 차원
hidden_size = 8  # 은닉 상태 벡터의 차원
```

자연어 처리에서는 각 값이 다음과 같이 대응될 수 있다.

```text
timesteps:
문장을 구성하는 토큰 또는 단어의 수

input_size:
각 토큰을 표현하는 단어 벡터의 차원

hidden_size:
RNN이 각 시점에서 유지하는 은닉 상태의 차원
```

10개 시점에서 각각 4차원 입력 벡터가 들어오도록 입력 배열을 생성한다.

```python
inputs = np.random.random((timesteps, input_size))
```

입력의 크기는 다음과 같다.

```text
inputs.shape = (10, 4)
```

이를 시점별로 해석하면 다음과 같다.

```text
inputs[0]: 첫 번째 시점의 4차원 입력
inputs[1]: 두 번째 시점의 4차원 입력
...
inputs[9]: 열 번째 시점의 4차원 입력
```

초기 은닉 상태는 모든 원소가 0인 벡터로 설정한다.

```python
hidden_state_t = np.zeros((hidden_size,))
```

출력하면 다음과 같다.

```python
print(hidden_state_t)
```

```text
[0. 0. 0. 0. 0. 0. 0. 0.]
```

은닉 상태의 크기를 8로 설정했으므로 초기 은닉 상태는 `(8,)` 크기의 0 벡터이다.

```text
hidden_state_t.shape = (8,)
```

---

### 7.3 가중치와 편향 정의하기

현재 입력에 적용할 가중치 \(W_x\), 이전 은닉 상태에 적용할 가중치 \(W_h\), 편향 \(b\)를 생성한다.

```python
Wx = np.random.random((hidden_size, input_size))
Wh = np.random.random((hidden_size, hidden_size))
b = np.random.random((hidden_size,))
```

각 매개변수의 크기는 다음과 같다.

```text
Wx.shape = (8, 4)
Wh.shape = (8, 8)
b.shape  = (8,)
```

일반적인 크기로 표현하면 다음과 같다.

```math
W_x\in\mathbb{R}^{D_h\times d}
```

```math
W_h\in\mathbb{R}^{D_h\times D_h}
```

```math
b\in\mathbb{R}^{D_h}
```

여기서 다음과 같다.

```text
d:
입력 벡터의 차원

D_h:
은닉 상태의 차원
```

크기를 출력하는 코드는 다음과 같다.

```python
print(np.shape(Wx))
print(np.shape(Wh))
print(np.shape(b))
```

출력은 다음과 같다.

```text
(8, 4)
(8, 8)
(8,)
```

---

### 7.4 한 시점의 은닉 상태 계산

한 시점의 은닉 상태는 다음 코드로 계산할 수 있다.

```python
output_t = np.tanh(
    np.dot(Wx, input_t)
    + np.dot(Wh, hidden_state_t)
    + b
)
```

각 연산의 크기는 다음과 같다.

```text
Wx:
(8, 4)

input_t:
(4,)

np.dot(Wx, input_t):
(8,)
```

```text
Wh:
(8, 8)

hidden_state_t:
(8,)

np.dot(Wh, hidden_state_t):
(8,)
```

따라서 세 항의 크기가 모두 `(8,)`이므로 원소별 덧셈이 가능하다.

```text
np.dot(Wx, input_t)          → (8,)
np.dot(Wh, hidden_state_t)   → (8,)
b                            → (8,)
------------------------------------------------
합산 결과                      → (8,)
tanh 적용 결과                 → (8,)
```

현재 시점의 출력 `output_t`는 다음 시점에서 이전 은닉 상태로 사용된다.

```python
hidden_state_t = output_t
```

이 구현에서는 별도의 출력층을 계산하지 않고, 각 시점의 은닉 상태 자체를 RNN 계층의 출력으로 사용한다.

---

### 7.5 모든 시점의 은닉 상태 계산하기

각 시점의 은닉 상태를 저장할 리스트를 생성한다.

```python
total_hidden_states = []
```

전체 입력 시퀀스를 순서대로 처리한다.

```python
for input_t in inputs:
    output_t = np.tanh(
        np.dot(Wx, input_t)
        + np.dot(Wh, hidden_state_t)
        + b
    )

    total_hidden_states.append(list(output_t))

    print(np.shape(total_hidden_states))

    hidden_state_t = output_t
```

반복문의 동작 순서는 다음과 같다.

```text
1. 현재 시점의 입력 input_t를 가져온다.
2. input_t와 이전 은닉 상태 hidden_state_t를 사용해 output_t를 계산한다.
3. 현재 은닉 상태 output_t를 리스트에 저장한다.
4. output_t를 다음 시점에서 사용할 은닉 상태로 갱신한다.
5. 마지막 시점까지 반복한다.
```

시점이 하나씩 처리될 때마다 저장된 은닉 상태의 크기는 다음처럼 증가한다.

```text
(1, 8)
(2, 8)
(3, 8)
(4, 8)
(5, 8)
(6, 8)
(7, 8)
(8, 8)
(9, 8)
(10, 8)
```

첫 번째 차원은 현재까지 처리한 시점의 수이고, 두 번째 차원은 은닉 상태의 크기이다.

마지막으로 리스트를 하나의 NumPy 배열로 변환한다.

```python
total_hidden_states = np.stack(
    total_hidden_states,
    axis=0
)
```

결과의 크기를 확인한다.

```python
print(total_hidden_states.shape)
```

```text
(10, 8)
```

즉, 10개 시점 각각에 대해 8차원의 은닉 상태가 저장되었다.

```text
total_hidden_states[t]:
t번째 시점의 은닉 상태

total_hidden_states.shape:
(timesteps, hidden_size)
```

`np.random.random()`으로 입력과 가중치를 임의 생성하므로 실제 은닉 상태의 숫자는 실행할 때마다 달라질 수 있다.

---

### 7.6 NumPy RNN 전체 코드

```python
import numpy as np

timesteps = 10
input_size = 4
hidden_size = 8

# (timesteps, input_size)
inputs = np.random.random((timesteps, input_size))

# 초기 은닉 상태: (hidden_size,)
hidden_state_t = np.zeros((hidden_size,))

# 가중치와 편향
Wx = np.random.random((hidden_size, input_size))
Wh = np.random.random((hidden_size, hidden_size))
b = np.random.random((hidden_size,))

total_hidden_states = []

for input_t in inputs:
    output_t = np.tanh(
        np.dot(Wx, input_t)
        + np.dot(Wh, hidden_state_t)
        + b
    )

    total_hidden_states.append(output_t)
    hidden_state_t = output_t

# (timesteps, hidden_size)
total_hidden_states = np.stack(
    total_hidden_states,
    axis=0
)

print(total_hidden_states)
print(total_hidden_states.shape)
```

출력 배열의 크기는 다음과 같다.

```text
(10, 8)
```

이 코드는 RNN의 순전파 구조를 이해하기 위한 단순화된 구현이다.

실제 딥러닝 프레임워크에서는 배치 차원을 포함한 3차원 입력을 주로 사용한다.

```text
단순화한 NumPy 예제:
(timesteps, input_size)

PyTorch에서 batch_first=True인 일반적인 입력:
(batch_size, timesteps, input_size)
```

---

## 8. PyTorch의 `nn.RNN`

PyTorch에서는 `torch.nn.RNN`을 이용해 기본 RNN 계층을 구현할 수 있다.

필요한 모듈을 불러온다.

```python
import torch
import torch.nn as nn
```

---

### 8.1 입력 크기와 은닉 상태 크기

입력 벡터와 은닉 상태의 크기를 정의한다.

```python
input_size = 5
hidden_size = 8
```

각 값의 의미는 다음과 같다.

```text
input_size=5:
각 시점에 입력되는 벡터의 차원

hidden_size=8:
각 시점에서 계산되는 은닉 상태의 차원
```

`hidden_size`는 RNN의 표현 용량을 결정하는 대표적인 하이퍼파라미터이다.

---

### 8.2 입력 텐서의 크기

`batch_first=True`를 사용하는 경우 입력 텐서의 크기는 다음과 같다.

```text
(batch_size, time_steps, input_size)
```

자료의 예제에서는 다음 크기의 입력을 사용한다.

```python
inputs = torch.randn(1, 10, 5)
```

```text
batch_size = 1
time_steps = 10
input_size = 5
```

따라서 입력 텐서의 크기는 다음과 같다.

```text
inputs.shape = (1, 10, 5)
```

> `torch.randn(1, 10, 5)`는 표준 정규 분포에서 값을 뽑아 입력 텐서를 생성한다. 값이 모두 0인 입력으로 shape만 확인하려면 `torch.zeros(1, 10, 5)`를 사용할 수 있다.

---

### 8.3 RNN 계층 생성하기

`nn.RNN`의 첫 번째 인자에는 입력 크기, 두 번째 인자에는 은닉 상태 크기를 전달한다.

```python
cell = nn.RNN(
    input_size,
    hidden_size,
    batch_first=True
)
```

각 설정의 의미는 다음과 같다.

```text
input_size:
각 시점의 입력 특성 수

hidden_size:
각 시점의 은닉 상태 크기

batch_first=True:
입력과 출력 텐서에서 배치 차원을 첫 번째에 배치
```

`batch_first=True`일 때 입력과 출력의 기본 축 순서는 다음과 같다.

```text
입력:
[batch, sequence, feature]

출력:
[batch, sequence, hidden]
```

---

### 8.4 `nn.RNN`의 두 반환값

입력 텐서를 RNN 계층에 전달한다.

```python
outputs, status = cell(inputs)
```

`nn.RNN`은 두 값을 반환한다.

```text
outputs:
마지막 RNN 층이 각 시점에서 출력한 은닉 상태

status:
각 RNN 층의 마지막 은닉 상태
```

은닉층이 한 개이고 단방향 RNN이라면 다음처럼 이해할 수 있다.

```text
outputs:
h_1, h_2, ..., h_T

status:
h_T
```

---

### 8.5 모든 시점의 출력 크기

첫 번째 반환값의 크기를 확인한다.

```python
print(outputs.shape)
```

출력은 다음과 같다.

```text
torch.Size([1, 10, 8])
```

각 차원의 의미는 다음과 같다.

```text
1:
배치 크기

10:
시퀀스의 시점 수

8:
각 시점의 은닉 상태 크기
```

즉, 하나의 샘플에 대해 10개 시점의 8차원 은닉 상태가 출력되었다.

```text
outputs.shape
= (batch_size, time_steps, hidden_size)
= (1, 10, 8)
```

---

### 8.6 마지막 은닉 상태의 크기

두 번째 반환값의 크기를 확인한다.

```python
print(status.shape)
```

출력은 다음과 같다.

```text
torch.Size([1, 1, 8])
```

`nn.RNN`이 반환하는 마지막 은닉 상태의 일반적인 크기는 다음과 같다.

```text
(num_layers × num_directions, batch_size, hidden_size)
```

현재 예제에서는 다음과 같다.

```text
num_layers = 1
num_directions = 1
batch_size = 1
hidden_size = 8
```

따라서 다음 크기가 된다.

```text
status.shape
= (1 × 1, 1, 8)
= (1, 1, 8)
```

여기서 첫 번째 차원은 시퀀스 길이가 아니라 `층의 수 × 방향의 수`를 의미한다.

---

### 8.7 `outputs`와 `status`의 관계

단일 층, 단방향 RNN에서는 마지막 시점의 `outputs`와 `status`가 같은 은닉 상태를 나타낸다.

```python
last_output = outputs[:, -1, :]
last_status = status[-1]
```

각 크기는 다음과 같다.

```text
last_output.shape = (batch_size, hidden_size)
last_status.shape = (batch_size, hidden_size)
```

현재 예제에서는 둘 다 다음 크기이다.

```text
(1, 8)
```

다만 다층 RNN이나 양방향 RNN에서는 각 반환값에 포함된 층과 방향 정보가 달라지므로 모양을 정확하게 확인해야 한다.

---

## 9. 깊은 순환 신경망(Deep RNN)

RNN도 여러 개의 순환 은닉층을 위아래로 쌓을 수 있다.

이를 `깊은 순환 신경망(Deep Recurrent Neural Network)`이라고 한다.

<img width="311" height="268" alt="Image" src="https://github.com/user-attachments/assets/1dbd63bc-3eeb-4f8d-a8ad-d71ef01c56e8" />

두 개의 은닉층을 가진 RNN의 흐름은 다음과 같다.

```text
입력 시퀀스
   ↓
첫 번째 RNN 층의 시점별 은닉 상태
   ↓
두 번째 RNN 층의 시점별 은닉 상태
   ↓
최종 시점별 출력
```

첫 번째 RNN 층은 각 시점의 은닉 상태를 같은 시점의 두 번째 RNN 층으로 전달한다.

```text
첫 번째 층의 h_t^(1)
   ↓
두 번째 층의 입력으로 사용
   ↓
두 번째 층의 h_t^(2) 계산
```

각 층은 시간축에서는 자신의 가중치를 반복해서 공유하지만, 서로 다른 층끼리는 별도의 가중치를 사용한다.

---

### 9.1 `num_layers`로 층 쌓기

PyTorch에서는 `nn.RNN`의 `num_layers` 인자를 사용해 순환 은닉층의 개수를 설정한다.

두 개의 층을 가진 RNN은 다음과 같이 만든다.

```python
inputs = torch.randn(1, 10, 5)

cell = nn.RNN(
    input_size=5,
    hidden_size=8,
    num_layers=2,
    batch_first=True
)

outputs, status = cell(inputs)
```

---

### 9.2 깊은 RNN의 `outputs`

첫 번째 반환값의 크기를 확인한다.

```python
print(outputs.shape)
```

출력은 다음과 같다.

```text
torch.Size([1, 10, 8])
```

층을 두 개로 늘려도 `outputs`의 크기는 이전과 같다.

```text
outputs.shape
= (batch_size, time_steps, hidden_size)
= (1, 10, 8)
```

그 이유는 `outputs`가 모든 층의 출력을 함께 반환하는 것이 아니라, **마지막 RNN 층의 모든 시점 출력**만 반환하기 때문이다.

```text
첫 번째 층의 시점별 은닉 상태:
내부적으로 두 번째 층에 전달

두 번째 층의 시점별 은닉 상태:
outputs로 반환
```

---

### 9.3 깊은 RNN의 `status`

마지막 은닉 상태의 크기를 확인한다.

```python
print(status.shape)
```

출력은 다음과 같다.

```text
torch.Size([2, 1, 8])
```

크기의 의미는 다음과 같다.

```text
2:
RNN 층의 수

1:
배치 크기

8:
은닉 상태의 크기
```

단방향 깊은 RNN에서 `status`의 크기는 다음과 같다.

```text
(num_layers, batch_size, hidden_size)
```

현재 예제에서는 다음과 같다.

```text
(2, 1, 8)
```

`status`에는 각 층의 마지막 시점 은닉 상태가 저장된다.

```text
status[0]:
첫 번째 RNN 층의 마지막 은닉 상태

status[1]:
두 번째 RNN 층의 마지막 은닉 상태
```

---

## 10. 양방향 순환 신경망(Bidirectional RNN)

기본 RNN은 입력 시퀀스를 앞에서 뒤로 읽으면서 이전 시점의 정보만 현재 계산에 사용한다.

```text
x_1 → x_2 → x_3 → ... → x_T
```

그러나 어떤 문제에서는 현재 위치를 판단하기 위해 이전 문맥뿐 아니라 이후 문맥도 필요하다.

예를 들어 다음 빈칸을 생각할 수 있다.

```text
Exercise is very effective at [          ] belly fat.

1) reducing
2) increasing
3) multiplying
```

빈칸 이전의 단어만 보면 정답을 확실히 결정하기 어렵다.

빈칸 뒤의 `belly fat`까지 확인하면 `reducing`이 자연스럽다는 정보를 얻을 수 있다.

이처럼 양쪽 문맥을 모두 사용하기 위해 고안된 구조가 `양방향 순환 신경망(Bidirectional Recurrent Neural Network)`이다.

---

### 10.1 정방향 상태와 역방향 상태

양방향 RNN은 서로 반대 방향으로 시퀀스를 처리하는 두 개의 RNN을 사용한다.

```text
정방향 RNN:
x_1 → x_2 → x_3 → ... → x_T

역방향 RNN:
x_T → ... → x_3 → x_2 → x_1
```

현재 시점 \(t\)에서는 다음 두 은닉 상태를 계산한다.

```text
정방향 은닉 상태:
앞쪽 시점들의 정보를 반영

역방향 은닉 상태:
뒤쪽 시점들의 정보를 반영
```

출력층은 두 방향의 은닉 상태를 모두 사용한다.

```text
정방향 은닉 상태
        ┐
        ├─→ 연결(concatenate) → 현재 시점의 출력
        ┘
역방향 은닉 상태
```

<img width="432" height="192" alt="Image" src="https://github.com/user-attachments/assets/4e7631b5-f1b5-47b6-a44a-a37824163da1" />

---

### 10.2 양방향 RNN의 시점별 표현

정방향 은닉 상태를 \(\overrightarrow{h_t}\), 역방향 은닉 상태를 \(\overleftarrow{h_t}\)라고 하면 각 방향은 다음처럼 계산된다.

```math
\overrightarrow{h_t}
=
\tanh\left(
W_x^{\rightarrow}x_t
+
W_h^{\rightarrow}\overrightarrow{h_{t-1}}
+
b^{\rightarrow}
\right)
```

```math
\overleftarrow{h_t}
=
\tanh\left(
W_x^{\leftarrow}x_t
+
W_h^{\leftarrow}\overleftarrow{h_{t+1}}
+
b^{\leftarrow}
\right)
```

현재 시점의 최종 은닉 표현은 일반적으로 두 방향을 연결해 만든다.

```math
h_t=
\left[
\overrightarrow{h_t};
\overleftarrow{h_t}
\right]
```

각 방향의 은닉 상태 크기가 `hidden_size`라면 연결 후 크기는 두 배가 된다.

```text
정방향 은닉 상태 크기: hidden_size
역방향 은닉 상태 크기: hidden_size

연결된 출력 크기:
hidden_size × 2
```

정방향과 역방향 RNN은 방향이 다르므로 서로 다른 가중치를 사용한다.

---

### 10.3 깊은 양방향 RNN

양방향 RNN도 여러 층으로 쌓을 수 있다.

<img width="435" height="262" alt="Image" src="https://github.com/user-attachments/assets/05224850-469b-42e2-8094-91e9bf76e520" />

깊은 양방향 RNN에서는 각 층마다 정방향 RNN과 역방향 RNN이 존재한다.

```text
첫 번째 층:
정방향 상태 + 역방향 상태

두 번째 층:
첫 번째 층의 양방향 출력을 입력으로 사용
정방향 상태 + 역방향 상태
```

은닉층의 수를 늘리면 모델이 더 복잡한 표현을 학습할 수 있지만, 층을 무조건 추가한다고 성능이 항상 좋아지는 것은 아니다.

```text
층 수 증가:
모델의 표현력과 매개변수 수 증가

동시에 발생할 수 있는 문제:
더 많은 훈련 데이터 필요
학습 시간 증가
최적화 어려움 증가
과적합 가능성 증가
```

---

### 10.4 PyTorch에서 양방향 RNN 구현하기

PyTorch에서는 `nn.RNN`의 `bidirectional` 인자를 `True`로 설정한다.

자료의 예제는 두 개의 층을 가진 깊은 양방향 RNN이다.

```python
inputs = torch.randn(1, 10, 5)

cell = nn.RNN(
    input_size=5,
    hidden_size=8,
    num_layers=2,
    batch_first=True,
    bidirectional=True
)

outputs, status = cell(inputs)
```

---

### 10.5 양방향 RNN의 `outputs`

첫 번째 반환값의 크기를 확인한다.

```python
print(outputs.shape)
```

출력은 다음과 같다.

```text
torch.Size([1, 10, 16])
```

양방향 RNN에서 `outputs`의 크기는 다음과 같다.

```text
(batch_size, time_steps, hidden_size × num_directions)
```

양방향인 경우 방향의 수는 2이다.

```text
num_directions = 2
```

현재 예제에서는 다음과 같다.

```text
batch_size = 1
time_steps = 10
hidden_size = 8
num_directions = 2
```

따라서 다음 크기가 된다.

```text
outputs.shape
= (1, 10, 8 × 2)
= (1, 10, 16)
```

각 시점의 출력에는 정방향 은닉 상태와 역방향 은닉 상태가 연결되어 있다.

```text
outputs[:, t, :8]:
t 시점의 정방향 출력

outputs[:, t, 8:]:
t 시점의 역방향 출력
```

---

### 10.6 양방향 RNN의 `status`

두 번째 반환값의 크기를 확인한다.

```python
print(status.shape)
```

출력은 다음과 같다.

```text
torch.Size([4, 1, 8])
```

양방향 RNN에서 마지막 은닉 상태의 크기는 다음과 같다.

```text
(num_layers × num_directions, batch_size, hidden_size)
```

현재 예제에서는 다음과 같다.

```text
num_layers = 2
num_directions = 2
batch_size = 1
hidden_size = 8
```

따라서 다음 크기가 된다.

```text
status.shape
= (2 × 2, 1, 8)
= (4, 1, 8)
```

첫 번째 차원에는 각 층과 각 방향의 마지막 은닉 상태가 저장된다.

PyTorch의 기본 배열 순서는 층별로 정방향과 역방향이 배치되는 형태로 이해할 수 있다.

```text
status[0]:
첫 번째 층의 정방향 마지막 은닉 상태

status[1]:
첫 번째 층의 역방향 마지막 은닉 상태

status[2]:
두 번째 층의 정방향 마지막 은닉 상태

status[3]:
두 번째 층의 역방향 마지막 은닉 상태
```

여기서 `마지막 은닉 상태`가 의미하는 시퀀스 위치는 방향에 따라 다르다.

```text
정방향 RNN:
원래 입력 기준 마지막 시점까지 처리한 은닉 상태

역방향 RNN:
역순 입력 기준 마지막까지 처리한 은닉 상태
즉, 원래 입력 기준 첫 번째 시점 쪽에서 완성된 은닉 상태
```

---

### 10.7 RNN 반환 텐서의 일반적인 크기

`batch_first=True`를 기준으로 `nn.RNN`의 반환 크기를 정리하면 다음과 같다.

```text
입력:
(batch_size, time_steps, input_size)
```

```text
outputs:
(batch_size,
 time_steps,
 hidden_size × num_directions)
```

```text
status:
(num_layers × num_directions,
 batch_size,
 hidden_size)
```

방향의 수는 다음과 같이 결정된다.

```text
단방향 RNN:
num_directions = 1

양방향 RNN:
num_directions = 2
```

예제별 크기를 비교하면 다음과 같다.

| 구조 | `outputs.shape` | `status.shape` |
|---|---|---|
| 단일 층 단방향 | `(1, 10, 8)` | `(1, 1, 8)` |
| 2층 단방향 | `(1, 10, 8)` | `(2, 1, 8)` |
| 2층 양방향 | `(1, 10, 16)` | `(4, 1, 8)` |

`num_layers`를 늘리면 `status`의 첫 번째 차원이 증가하지만, `outputs`에는 마지막 층의 출력만 들어가므로 층 수가 직접 표시되지 않는다.

`bidirectional=True`로 설정하면 각 시점에서 두 방향의 출력이 연결되므로 `outputs`의 마지막 차원이 두 배가 된다.

---

## 11. 바닐라 RNN의 한계와 LSTM

앞에서 다룬 가장 기본적인 형태의 RNN을 `바닐라 RNN(Vanilla RNN)`이라고 한다.

```text
Vanilla RNN:
가장 단순한 형태의 순환 신경망

Keras에서의 이름:
SimpleRNN
```

바닐라 RNN은 이전 시점의 은닉 상태를 현재 시점으로 전달하여 시퀀스 정보를 처리한다.

하지만 시퀀스가 길어지면 앞부분의 정보를 충분히 유지하기 어렵다는 한계가 있다.

이러한 한계를 완화하기 위해 여러 RNN 변형 구조가 제안되었으며, 대표적인 구조가 다음 두 가지이다.

```text
LSTM:
Long Short-Term Memory

GRU:
Gated Recurrent Unit
```

이후의 설명에서 별도의 구분 없이 `RNN`이라고 표현하는 경우에는 LSTM이나 GRU가 아닌 `바닐라 RNN`을 의미한다.

---

### 11.1 바닐라 RNN의 장기 의존성 문제

바닐라 RNN의 현재 은닉 상태는 이전 시점의 은닉 상태에 의존한다.

```text
h_1 → h_2 → h_3 → ... → h_t
```

따라서 이론적으로는 앞 시점의 정보가 은닉 상태를 통해 뒤 시점까지 전달될 수 있다.

그러나 시퀀스가 길어질수록 초기 입력의 정보가 반복적인 상태 갱신 과정에서 점차 약해질 수 있다.

<img width="326" height="198" alt="Image" src="https://github.com/user-attachments/assets/4a5e759b-3e3a-44f9-b09f-435291eeb159" />

그림에서 첫 번째 입력 \(x_1\)의 정보량을 짙은 남색으로 표현하면, 시점이 지날수록 색이 옅어지는 것으로 정보의 영향력이 감소하는 모습을 확인할 수 있다.

```text
초기 시점:
x_1의 정보가 은닉 상태에 강하게 반영됨

시점이 진행됨:
새로운 입력과 반복 연산을 거치며 x_1의 영향력이 감소

매우 먼 시점:
x_1의 정보가 현재 출력에 거의 영향을 주지 못할 수 있음
```

이 문제는 문장의 앞부분에 중요한 정보가 있고, 뒤쪽에서 그 정보를 다시 사용해야 할 때 특히 크게 나타난다.

예를 들어 다음 문장의 빈칸을 예측한다고 하자.

```text
모스크바에 여행을 왔는데 건물도 예쁘고 먹을 것도 맛있었어.
그런데 글쎄 직장 상사한테 전화가 왔어.
어디냐고 묻더라구. 그래서 나는 말했지.
저 여행 왔는데요. 여기 ___
```

빈칸 다음에 올 단어를 예측하려면 문장 앞부분의 장소 정보인 `모스크바`를 기억해야 한다.

하지만 바닐라 RNN이 긴 문장을 처리하는 동안 해당 정보를 충분히 유지하지 못하면 적절하지 않은 단어를 예측할 수 있다.

이처럼 멀리 떨어진 시점 사이의 관계를 학습하고 유지하기 어려운 문제를 `장기 의존성 문제(Long-Term Dependency Problem)`라고 한다.

---

### 11.2 바닐라 RNN의 내부 연산

바닐라 RNN의 현재 은닉 상태는 현재 입력과 이전 은닉 상태를 이용해 계산한다.

<img width="404" height="325" alt="Image" src="https://github.com/user-attachments/assets/11acf879-c3b3-45a6-b548-ae65bc5e429d" />

편향까지 포함한 은닉 상태 계산식은 다음과 같다.

```math
h_t
=
\tanh\left(
W_xx_t
+
W_hh_{t-1}
+
b
\right)
```

각 기호의 의미는 다음과 같다.

```text
x_t:
현재 시점 t의 입력 벡터

h_{t-1}:
이전 시점 t-1의 은닉 상태

W_x:
현재 입력에 적용되는 가중치 행렬

W_h:
이전 은닉 상태에 적용되는 가중치 행렬

b:
편향 벡터

h_t:
현재 시점의 은닉 상태
```

계산 과정은 다음과 같다.

```text
현재 입력 x_t에 W_x 적용
   ↓
이전 은닉 상태 h_{t-1}에 W_h 적용
   ↓
두 결과와 편향 b를 더함
   ↓
tanh 함수 적용
   ↓
현재 은닉 상태 h_t 계산
```

바닐라 RNN은 하나의 은닉 상태 \(h_t\)를 통해 과거 정보와 현재 정보를 함께 관리한다.

LSTM은 이 구조에 별도의 셀 상태와 여러 게이트를 추가하여 장기 정보를 더 선택적으로 유지한다.

---

### 11.3 LSTM(Long Short-Term Memory)

`LSTM(Long Short-Term Memory)`은 바닐라 RNN의 장기 의존성 문제를 완화하기 위해 제안된 순환 신경망 구조이다.

LSTM은 바닐라 RNN보다 내부 계산이 복잡하며, 다음 요소를 추가로 사용한다.

```text
입력 게이트(Input Gate)
망각 게이트 또는 삭제 게이트(Forget Gate)
출력 게이트(Output Gate)
셀 상태(Cell State)
```

<img width="422" height="330" alt="Image" src="https://github.com/user-attachments/assets/c939caa4-a4d6-4ae6-ac6d-6f516801ef9f" />

LSTM의 핵심은 정보를 무조건 다음 시점으로 전달하는 것이 아니라, 각 게이트를 이용해 다음을 선택적으로 결정하는 것이다.

```text
이전 기억 중 무엇을 유지할지
이전 기억 중 무엇을 삭제할지
현재 입력 중 무엇을 새로 기억할지
현재 셀 상태 중 무엇을 은닉 상태로 출력할지
```

LSTM은 바닐라 RNN의 은닉 상태 \(h_t\) 외에도 셀 상태 \(C_t\)를 사용한다.

```text
h_t:
은닉 상태
단기 상태라고도 표현

C_t:
셀 상태
장기 상태라고도 표현
```

---

### 11.4 셀 상태(Cell State)

셀 상태는 LSTM 내부에서 장기 정보를 전달하는 경로이다.

<img width="337" height="170" alt="Image" src="https://github.com/user-attachments/assets/a96d7227-8ade-4c2d-b1c8-1ca693c428bd" />

이전 시점의 셀 상태 \(C_{t-1}\)는 현재 시점의 셀 상태 \(C_t\)를 계산하는 데 사용되며, 계산된 \(C_t\)는 다시 다음 시점으로 전달된다.

```text
C_{t-1}
   ↓
망각 게이트로 이전 기억 조절
   ↓
입력 게이트로 새로운 기억 추가
   ↓
C_t
   ↓
다음 시점으로 전달
```

셀 상태 경로에서는 이전 정보를 단순한 비선형 변환으로 계속 덮어쓰는 대신, 원소별 곱셈과 덧셈을 통해 정보를 선택적으로 유지하고 갱신한다.

---

### 11.5 LSTM에서 사용하는 함수와 매개변수

LSTM의 게이트에서는 주로 시그모이드 함수와 하이퍼볼릭 탄젠트 함수가 사용된다.

#### 시그모이드 함수

```math
\sigma(z)
=
\frac{1}{1+e^{-z}}
```

출력 범위는 다음과 같다.

```text
0 < σ(z) < 1
```

시그모이드의 출력은 정보를 얼마나 통과시킬지 조절하는 게이트 값으로 사용된다.

```text
0에 가까운 값:
해당 정보를 거의 통과시키지 않음

1에 가까운 값:
해당 정보를 대부분 통과시킴
```

#### 하이퍼볼릭 탄젠트 함수

```math
\tanh(z)
=
\frac{e^z-e^{-z}}{e^z+e^{-z}}
```

출력 범위는 다음과 같다.

```text
-1 < tanh(z) < 1
```

`tanh`는 새로 저장할 후보 정보나 셀 상태를 변환할 때 사용된다.

LSTM의 각 게이트는 현재 입력 \(x_t\)와 이전 은닉 상태 \(h_{t-1}\)를 함께 사용하지만, 서로 다른 가중치와 편향을 가진다.

```text
현재 입력 x_t에 적용되는 가중치:
W_{xi}, W_{xg}, W_{xf}, W_{xo}

이전 은닉 상태 h_{t-1}에 적용되는 가중치:
W_{hi}, W_{hg}, W_{hf}, W_{ho}

각 연산의 편향:
b_i, b_g, b_f, b_o
```

---

### 11.6 입력 게이트(Input Gate)

입력 게이트는 현재 시점의 정보 중 어떤 내용을 셀 상태에 새로 저장할지 결정한다.

<img width="327" height="232" alt="Image" src="https://github.com/user-attachments/assets/5f1c73a7-51ca-467a-9b95-acc96c1d0b70" />

입력 게이트에서는 두 값을 계산한다.

```text
i_t:
현재 후보 정보 중 얼마나 저장할지 결정하는 게이트 값

g_t:
현재 시점에서 새로 저장할 후보 정보
```

#### 입력 게이트 값

```math
i_t
=
\sigma\left(
W_{xi}x_t
+
W_{hi}h_{t-1}
+
b_i
\right)
```

\(i_t\)는 시그모이드 함수를 통과하므로 각 원소가 0과 1 사이의 값을 가진다.

```text
i_t가 0에 가까움:
해당 후보 정보를 거의 저장하지 않음

i_t가 1에 가까움:
해당 후보 정보를 적극적으로 저장
```

#### 후보 기억

```math
g_t
=
\tanh\left(
W_{xg}x_t
+
W_{hg}h_{t-1}
+
b_g
\right)
```

\(g_t\)는 현재 입력과 이전 은닉 상태로부터 생성된 새로운 기억 후보이다.

`tanh`를 통과하므로 각 원소는 -1과 1 사이의 값을 가진다.

입력 게이트가 실제로 셀 상태에 추가할 정보는 다음 원소별 곱으로 결정된다.

```math
i_t\circ g_t
```

여기서 \(\circ\)는 같은 위치의 원소끼리 곱하는 `원소별 곱(Element-wise Product)`을 의미한다.

---

### 11.7 망각 게이트(Forget Gate)

망각 게이트는 이전 셀 상태의 정보 중 무엇을 유지하고 무엇을 삭제할지 결정한다.

자료에서는 `삭제 게이트`라는 표현도 사용한다.

<img width="336" height="241" alt="Image" src="https://github.com/user-attachments/assets/6b620d39-b72f-436b-95b3-2ca7a0bd1e73" />

망각 게이트 값은 다음과 같이 계산한다.

```math
f_t
=
\sigma\left(
W_{xf}x_t
+
W_{hf}h_{t-1}
+
b_f
\right)
```

\(f_t\)는 시그모이드 함수를 통과하므로 각 원소가 0과 1 사이의 값을 가진다.

```text
f_t가 0에 가까움:
이전 셀 상태의 해당 정보를 대부분 삭제

f_t가 1에 가까움:
이전 셀 상태의 해당 정보를 대부분 유지
```

이전 셀 상태에 대한 실제 조절은 다음 원소별 곱으로 이루어진다.

```math
f_t\circ C_{t-1}
```

---

### 11.8 셀 상태 갱신

현재 시점의 셀 상태는 다음 두 정보를 더하여 계산한다.

```text
이전 셀 상태 중 유지하기로 선택한 정보
현재 시점에서 새로 저장하기로 선택한 정보
```

<img width="337" height="245" alt="Image" src="https://github.com/user-attachments/assets/63733110-f780-4805-aa0b-f5055b9ecf38" />

셀 상태 계산식은 다음과 같다.

```math
C_t
=
f_t\circ C_{t-1}
+
i_t\circ g_t
```

각 항의 의미는 다음과 같다.

```text
f_t ∘ C_{t-1}:
이전 셀 상태에서 유지할 정보

i_t ∘ g_t:
현재 시점에서 새로 저장할 정보
```

전체 갱신 과정은 다음과 같다.

```text
이전 셀 상태 C_{t-1}
   ↓
망각 게이트 f_t와 원소별 곱
   ↓
유지할 과거 정보 계산
```

```text
후보 기억 g_t
   ↓
입력 게이트 i_t와 원소별 곱
   ↓
새로 저장할 현재 정보 계산
```

```text
유지할 과거 정보
+
새로 저장할 현재 정보
   ↓
현재 셀 상태 C_t
```

#### 망각 게이트가 완전히 닫힌 경우

\(f_t=0\)이라고 하면 다음과 같다.

```math
C_t
=
i_t\circ g_t
```

이 경우 이전 셀 상태 \(C_{t-1}\)의 영향은 사라지고, 현재 입력을 통해 선택한 새로운 기억만 셀 상태를 결정한다.

#### 입력 게이트가 완전히 닫힌 경우

\(i_t=0\)이라고 하면 다음과 같다.

```math
C_t
=
f_t\circ C_{t-1}
```

이 경우 현재 후보 정보는 추가되지 않고, 이전 셀 상태에서 유지하기로 선택한 정보만 현재 셀 상태를 결정한다.

즉, 두 게이트의 역할은 다음처럼 구분할 수 있다.

```text
망각 게이트:
이전 기억을 얼마나 유지할지 결정

입력 게이트:
현재 정보를 얼마나 새로 반영할지 결정
```

---

### 11.9 출력 게이트와 은닉 상태

출력 게이트는 현재 셀 상태의 정보 중 어느 부분을 현재 은닉 상태로 내보낼지 결정한다.

<img width="334" height="243" alt="Image" src="https://github.com/user-attachments/assets/ef1bd1b6-7a1d-4df6-b3c4-2d1afff435be" />

출력 게이트 값은 다음과 같이 계산한다.

```math
o_t
=
\sigma\left(
W_{xo}x_t
+
W_{ho}h_{t-1}
+
b_o
\right)
```

현재 은닉 상태는 셀 상태에 `tanh`를 적용한 값과 출력 게이트를 원소별로 곱하여 계산한다.

```math
h_t
=
o_t\circ\tanh(C_t)
```

계산 과정은 다음과 같다.

```text
현재 셀 상태 C_t
   ↓
tanh 적용
   ↓
-1과 1 사이의 값으로 변환
   ↓
출력 게이트 o_t와 원소별 곱
   ↓
현재 은닉 상태 h_t
```

은닉 상태 \(h_t\)는 다음 두 방향으로 전달된다.

```text
다음 시점의 LSTM 셀
현재 시점의 출력층
```

LSTM에서 두 상태의 역할은 다음과 같이 구분할 수 있다.

```text
셀 상태 C_t:
상대적으로 장기적인 정보를 전달하는 상태

은닉 상태 h_t:
현재 시점의 출력과 다음 시점 계산에 사용되는 상태
```

---

### 11.10 LSTM의 시점별 전체 계산

LSTM의 한 시점 계산을 순서대로 정리하면 다음과 같다.

#### 1단계: 망각 게이트 계산

```math
f_t
=
\sigma\left(
W_{xf}x_t
+
W_{hf}h_{t-1}
+
b_f
\right)
```

#### 2단계: 입력 게이트 계산

```math
i_t
=
\sigma\left(
W_{xi}x_t
+
W_{hi}h_{t-1}
+
b_i
\right)
```

#### 3단계: 후보 기억 계산

```math
g_t
=
\tanh\left(
W_{xg}x_t
+
W_{hg}h_{t-1}
+
b_g
\right)
```

#### 4단계: 셀 상태 갱신

```math
C_t
=
f_t\circ C_{t-1}
+
i_t\circ g_t
```

#### 5단계: 출력 게이트 계산

```math
o_t
=
\sigma\left(
W_{xo}x_t
+
W_{ho}h_{t-1}
+
b_o
\right)
```

#### 6단계: 은닉 상태 계산

```math
h_t
=
o_t\circ\tanh(C_t)
```

전체 흐름은 다음과 같다.

```text
x_t와 h_{t-1}
   ├─→ f_t 계산
   ├─→ i_t 계산
   ├─→ g_t 계산
   └─→ o_t 계산

C_{t-1}와 f_t
   ↓
유지할 과거 정보 계산

i_t와 g_t
   ↓
새로 저장할 정보 계산

두 결과를 더함
   ↓
C_t 계산

C_t와 o_t
   ↓
h_t 계산
```

---

### 11.11 PyTorch의 `nn.LSTM`

PyTorch에서는 `nn.LSTM`을 이용해 LSTM 계층을 생성한다.

기본 RNN 계층은 다음과 같이 생성했다.

```python
nn.RNN(
    input_dim,
    hidden_size,
    batch_first=True
)
```

LSTM 계층도 유사한 방식으로 생성한다.

```python
nn.LSTM(
    input_dim,
    hidden_size,
    batch_first=True
)
```

각 인자의 의미는 다음과 같다.

```text
input_dim:
각 시점에 입력되는 벡터의 차원

hidden_size:
은닉 상태와 셀 상태의 차원

batch_first=True:
입력 텐서의 첫 번째 차원을 배치 크기로 사용
```

`batch_first=True`를 사용하면 입력 텐서의 기본 크기는 다음과 같다.

```text
(batch_size, time_steps, input_dim)
```

LSTM은 모든 시점의 출력과 마지막 은닉 상태·셀 상태를 반환한다.

```python
lstm = nn.LSTM(
    input_size=input_dim,
    hidden_size=hidden_size,
    batch_first=True
)

inputs = torch.randn(1, 10, input_dim)
outputs, (h_n, c_n) = lstm(inputs)
```

단일 층·단방향 LSTM에서 각 크기는 다음과 같다.

```text
outputs.shape:
(batch_size, time_steps, hidden_size)

h_n.shape:
(num_layers × num_directions, batch_size, hidden_size)

c_n.shape:
(num_layers × num_directions, batch_size, hidden_size)
```

```text
outputs:
마지막 LSTM 층이 각 시점에서 출력한 은닉 상태

h_n:
각 층과 방향의 최종 은닉 상태

c_n:
각 층과 방향의 최종 셀 상태
```

`batch_first=True`는 `inputs`와 `outputs`의 축 순서에만 영향을 주며, `h_n`과 `c_n`의 축 순서는 바뀌지 않는다.

---

## 12. GRU(Gated Recurrent Unit)

`GRU(Gated Recurrent Unit)`는 LSTM과 마찬가지로 바닐라 RNN의 장기 의존성 문제를 완화하기 위한 순환 신경망 구조이다.

GRU는 LSTM의 핵심적인 게이트 구조를 유지하면서 내부 계산을 단순화한다.

자료에서는 GRU가 2014년 조경현 교수 등이 집필한 논문에서 제안된 구조라고 설명한다.

LSTM과 GRU의 상태 구조는 다음과 같이 구분된다.

```text
LSTM:
셀 상태 C_t와 은닉 상태 h_t를 별도로 사용

GRU:
별도의 셀 상태 없이 은닉 상태 h_t를 중심으로 정보 유지
```

LSTM에는 세 종류의 게이트가 있다.

```text
입력 게이트
망각 게이트
출력 게이트
```

GRU에는 두 종류의 게이트가 있다.

```text
리셋 게이트(Reset Gate)
업데이트 게이트(Update Gate)
```

<img width="344" height="287" alt="Image" src="https://github.com/user-attachments/assets/227ff907-7144-42ea-a038-ae4af4ea87cc" />

---

### 12.1 리셋 게이트(Reset Gate)

리셋 게이트는 새로운 후보 은닉 상태를 계산할 때 이전 은닉 상태를 얼마나 반영할지 결정한다.

```math
r_t
=
\sigma\left(
W_{xr}x_t
+
W_{hr}h_{t-1}
+
b_r
\right)
```

각 기호의 의미는 다음과 같다.

```text
r_t:
리셋 게이트 값

W_{xr}:
현재 입력에 적용되는 리셋 게이트 가중치

W_{hr}:
이전 은닉 상태에 적용되는 리셋 게이트 가중치

b_r:
리셋 게이트의 편향
```

\(r_t\)는 0과 1 사이의 값을 가지며, 후보 은닉 상태를 만들 때 이전 은닉 상태를 원소별로 조절한다.

```text
r_t가 0에 가까움:
이전 은닉 상태의 해당 정보를 후보 계산에서 거의 사용하지 않음

r_t가 1에 가까움:
이전 은닉 상태의 해당 정보를 후보 계산에 적극적으로 사용
```

---

### 12.2 업데이트 게이트(Update Gate)

업데이트 게이트는 이전 은닉 상태와 새로운 후보 은닉 상태를 어떤 비율로 결합할지 결정한다.

```math
z_t
=
\sigma\left(
W_{xz}x_t
+
W_{hz}h_{t-1}
+
b_z
\right)
```

각 기호의 의미는 다음과 같다.

```text
z_t:
업데이트 게이트 값

W_{xz}:
현재 입력에 적용되는 업데이트 게이트 가중치

W_{hz}:
이전 은닉 상태에 적용되는 업데이트 게이트 가중치

b_z:
업데이트 게이트의 편향
```

자료에서 제시한 은닉 상태 갱신식에서는 \(z_t\)가 클수록 이전 은닉 상태를 더 많이 유지한다.

```text
z_t가 1에 가까움:
이전 은닉 상태 h_{t-1}을 주로 유지

z_t가 0에 가까움:
새로운 후보 상태 g_t를 주로 반영
```

---

### 12.3 후보 은닉 상태

리셋 게이트로 조절한 이전 은닉 상태와 현재 입력을 이용해 새로운 후보 은닉 상태를 계산한다.

```math
g_t
=
\tanh\left(
W_{hg}
\left(
r_t\circ h_{t-1}
\right)
+
W_{xg}x_t
+
b_g
\right)
```

계산 과정은 다음과 같다.

```text
이전 은닉 상태 h_{t-1}
   ↓
리셋 게이트 r_t와 원소별 곱
   ↓
후보 계산에 사용할 과거 정보 결정
```

```text
조절된 과거 정보
+
현재 입력의 변환값
+
편향
   ↓
tanh 적용
   ↓
후보 은닉 상태 g_t
```

후보 은닉 상태 \(g_t\)는 현재 시점에서 새롭게 반영할 수 있는 정보이다.

위 수식은 자료에서 사용하는 GRU 표기이며, 리셋 게이트를 이전 은닉 상태에 먼저 원소별로 적용한 뒤 가중치 행렬을 곱한다.

```math
g_t
=
\tanh\left(
W_{hg}(r_t\circ h_{t-1})
+
W_{xg}x_t
+
b_g
\right)
```

PyTorch의 `nn.GRU`는 새로운 후보 상태를 계산할 때 리셋 게이트의 적용 위치가 조금 다르다.

```math
n_t
=
\tanh\left(
W_{in}x_t+b_{in}
+
r_t\circ\left(W_{hn}h_{t-1}+b_{hn}\right)
\right)
```

```text
자료의 수식:
r_t를 h_{t-1}에 먼저 적용한 뒤 은닉 가중치와 곱함

PyTorch nn.GRU:
h_{t-1}을 은닉 가중치로 변환한 결과에 r_t를 적용함
```

두 식은 GRU의 핵심 아이디어는 같지만 연산 순서와 매개변수화가 완전히 동일하지 않다. 따라서 앞의 수식을 PyTorch 내부 구현과 정확히 같은 식으로 해석해서는 안 된다.

---

### 12.4 현재 은닉 상태 계산

현재 은닉 상태는 이전 은닉 상태와 후보 은닉 상태를 업데이트 게이트로 결합하여 계산한다.

```math
h_t
=
(1-z_t)\circ g_t
+
z_t\circ h_{t-1}
```

각 항의 의미는 다음과 같다.

```text
(1-z_t) ∘ g_t:
새로운 후보 은닉 상태에서 반영할 정보

z_t ∘ h_{t-1}:
이전 은닉 상태에서 유지할 정보
```

업데이트 게이트의 값에 따른 동작은 다음과 같다.

#### \(z_t=1\)인 경우

```math
h_t=h_{t-1}
```

이전 은닉 상태를 그대로 유지하고 새로운 후보 정보는 반영하지 않는다.

#### \(z_t=0\)인 경우

```math
h_t=g_t
```

이전 은닉 상태 대신 새로운 후보 은닉 상태를 현재 상태로 사용한다.

즉, 업데이트 게이트 하나가 LSTM의 입력 게이트와 망각 게이트가 수행하던 일부 역할을 함께 담당하는 구조로 볼 수 있다.

---

### 12.5 GRU의 시점별 전체 계산

GRU의 한 시점 계산을 순서대로 정리하면 다음과 같다.

#### 1단계: 리셋 게이트 계산

```math
r_t
=
\sigma\left(
W_{xr}x_t
+
W_{hr}h_{t-1}
+
b_r
\right)
```

#### 2단계: 업데이트 게이트 계산

```math
z_t
=
\sigma\left(
W_{xz}x_t
+
W_{hz}h_{t-1}
+
b_z
\right)
```

#### 3단계: 후보 은닉 상태 계산

```math
g_t
=
\tanh\left(
W_{hg}
\left(
r_t\circ h_{t-1}
\right)
+
W_{xg}x_t
+
b_g
\right)
```

#### 4단계: 현재 은닉 상태 계산

```math
h_t
=
(1-z_t)\circ g_t
+
z_t\circ h_{t-1}
```

전체 흐름은 다음과 같다.

```text
x_t와 h_{t-1}
   ├─→ 리셋 게이트 r_t 계산
   └─→ 업데이트 게이트 z_t 계산

r_t로 h_{t-1} 조절
   ↓
현재 입력 x_t와 결합
   ↓
후보 은닉 상태 g_t 계산

g_t와 h_{t-1}
   ↓
z_t로 결합 비율 조절
   ↓
현재 은닉 상태 h_t 계산
```

---

### 12.6 LSTM과 GRU 비교

| 구분 | LSTM | GRU |
|---|---|---|
| 상태 | 셀 상태 \(C_t\), 은닉 상태 \(h_t\) | 은닉 상태 \(h_t\) |
| 주요 게이트 | 입력, 망각, 출력 게이트 | 리셋, 업데이트 게이트 |
| 내부 구조 | 상대적으로 복잡함 | 상대적으로 단순함 |
| 계산량과 매개변수 | 상대적으로 많음 | 상대적으로 적음 |
| 장기 의존성 대응 | 가능 | 가능 |
| 성능 | 문제와 설정에 따라 달라짐 | 문제와 설정에 따라 달라짐 |

GRU는 LSTM보다 구조가 단순하므로 일반적으로 계산해야 하는 매개변수와 연산이 적다.

자료에서는 GRU가 LSTM보다 학습 속도가 빠른 것으로 알려져 있지만, 여러 평가에서 두 구조가 비슷한 성능을 보인다고 설명한다.

따라서 다음과 같이 단정할 수는 없다.

```text
항상 LSTM이 더 좋다.
항상 GRU가 더 좋다.
```

모델 선택은 다음 조건에 따라 달라질 수 있다.

```text
데이터의 양
시퀀스 길이
문제의 특성
모델 크기 제한
학습 시간
하이퍼파라미터 설정
```

이미 LSTM으로 충분히 좋은 하이퍼파라미터와 성능을 확보했다면, 반드시 GRU로 교체해야 하는 것은 아니다.

GRU는 LSTM보다 매개변수와 연산이 상대적으로 적어 제한된 계산 자원이나 빠른 실험이 필요한 상황에서 유리할 수 있다.

다만 데이터의 양만으로 LSTM과 GRU의 우열을 결정할 수는 없다.

```text
실제 선택 기준:
검증 데이터 성능
학습 속도와 메모리 사용량
시퀀스 길이와 문제 특성
하이퍼파라미터 탐색 결과
```

따라서 동일한 데이터와 평가 조건에서 두 모델을 실험적으로 비교하는 것이 가장 안전하다.

---

### 12.7 PyTorch의 `nn.GRU`

PyTorch에서는 `nn.GRU`를 이용해 GRU 계층을 생성한다.

기본 RNN 계층은 다음과 같이 생성했다.

```python
nn.RNN(
    input_dim,
    hidden_size,
    batch_first=True
)
```

GRU 계층은 다음과 같이 생성한다.

```python
nn.GRU(
    input_dim,
    hidden_size,
    batch_first=True
)
```

각 인자의 의미는 다음과 같다.

```text
input_dim:
각 시점에 입력되는 벡터의 차원

hidden_size:
각 시점의 은닉 상태 차원

batch_first=True:
입력 텐서의 첫 번째 차원을 배치 크기로 사용
```

`batch_first=True`일 때 입력 텐서의 기본 크기는 다음과 같다.

```text
(batch_size, time_steps, input_dim)
```

GRU는 모든 시점의 출력과 각 층·방향의 최종 은닉 상태를 반환한다.

```python
gru = nn.GRU(
    input_size=input_dim,
    hidden_size=hidden_size,
    batch_first=True
)

inputs = torch.randn(1, 10, input_dim)
outputs, h_n = gru(inputs)
```

반환 텐서의 일반적인 크기는 다음과 같다.

```text
outputs.shape:
(batch_size,
 time_steps,
 hidden_size × num_directions)

h_n.shape:
(num_layers × num_directions,
 batch_size,
 hidden_size)
```

```text
outputs:
마지막 GRU 층이 각 시점에서 출력한 은닉 상태

h_n:
각 층과 방향의 최종 은닉 상태
```

LSTM과 달리 GRU에는 별도의 셀 상태가 없으므로 `c_n`을 반환하지 않는다.

---

## 13. 문자 단위 RNN(Char RNN)

RNN의 입력과 출력 단위를 단어가 아니라 `문자(character)`로 설정한 모델을 `문자 단위 RNN(Character-level RNN, Char RNN)`이라고 한다.

```text
단어 단위 RNN:
각 시점에서 하나의 단어 벡터를 입력하고 단어 단위 출력을 생성

문자 단위 RNN:
각 시점에서 하나의 문자 벡터를 입력하고 문자 단위 출력을 생성
```

문자 단위 RNN이라고 해서 RNN의 내부 구조 자체가 달라지는 것은 아니다.

```text
동일한 점:
RNN 셀, 은닉 상태 전달, 시간축 연산 방식

달라지는 점:
각 시점의 입력과 출력이 단어가 아니라 문자
```

이번 실습에서는 각 시점마다 문자를 하나씩 입력하고, 다음 문자를 예측하는 `다 대 다(many-to-many)` 형태의 RNN을 구현한다.

---

## 14. 실습 1: `apple`을 입력하여 `pple!` 예측하기

첫 번째 실습에서는 다음과 같은 문자 시퀀스 변환을 학습한다.

```text
입력:  a p p l e
정답:  p p l e !
```

각 입력 문자에 대해 바로 다음 위치의 문자를 예측하도록 학습하는 구조이다.

```text
a → p
p → p
p → l
l → e
e → !
```

이 실습은 실제 응용 문제를 해결하기보다는 문자 단위 RNN의 데이터 형태와 학습 과정을 이해하기 위한 예제이다.

---

### 14.1 필요한 라이브러리 불러오기

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
```

각 라이브러리의 역할은 다음과 같다.

```text
NumPy:
원-핫 벡터 생성

torch:
입력과 정답을 텐서로 변환

torch.nn:
RNN 계층, 선형 계층, 손실 함수 정의

torch.optim:
옵티마이저 정의
```

---

### 14.2 입력 시퀀스와 정답 시퀀스 정의하기

```python
input_str = "apple"
label_str = "pple!"
```

입력과 정답의 길이는 모두 5이다.

```text
입력 시퀀스 길이: 5
정답 시퀀스 길이: 5
```

RNN은 각 시점에서 현재 문자를 입력받아 대응하는 다음 문자를 예측한다.

---

### 14.3 문자 집합 만들기

입력 문자열과 정답 문자열에 등장하는 문자를 합친 뒤 중복을 제거한다.

```python
char_vocab = sorted(list(set(input_str + label_str)))
vocab_size = len(char_vocab)

print("문자 집합:", char_vocab)
print("문자 집합의 크기:", vocab_size)
```

문자 집합은 다음과 같다.

```text
['!', 'a', 'e', 'l', 'p']
```

문자 집합의 크기는 다음과 같다.

```text
vocab_size = 5
```

문자 집합의 크기는 다음 두 값과 직접 연결된다.

```text
입력 원-핫 벡터의 차원
출력층의 클래스 개수
```

---

### 14.4 하이퍼파라미터 정의하기

```python
input_size = vocab_size
hidden_size = 5
output_size = vocab_size
learning_rate = 0.1
```

각 하이퍼파라미터의 의미는 다음과 같다.

```text
input_size:
각 시점에 입력되는 벡터의 차원

hidden_size:
RNN 은닉 상태의 차원

output_size:
각 시점에서 출력할 클래스의 수

learning_rate:
옵티마이저의 학습률
```

현재 입력을 원-핫 벡터로 표현하므로 `input_size`는 문자 집합의 크기와 같다.

```text
문자 집합 크기 = 5
원-핫 벡터 차원 = 5
```

각 시점에서는 다섯 문자 중 하나를 예측하므로 `output_size`도 5이다.

---

### 14.5 문자와 정수 인덱스 매핑하기

각 문자에 고유한 정수 인덱스를 부여한다.

```python
char_to_index = {
    char: index
    for index, char in enumerate(char_vocab)
}

print(char_to_index)
```

출력은 다음과 같다.

```text
{'!': 0, 'a': 1, 'e': 2, 'l': 3, 'p': 4}
```

예측 결과를 다시 문자로 복원하기 위한 역방향 딕셔너리도 만든다.

```python
index_to_char = {
    index: char
    for char, index in char_to_index.items()
}

print(index_to_char)
```

출력은 다음과 같다.

```text
{0: '!', 1: 'a', 2: 'e', 3: 'l', 4: 'p'}
```

두 딕셔너리의 역할은 다음과 같다.

```text
char_to_index:
문자 → 정수 인덱스

index_to_char:
정수 인덱스 → 문자
```

---

### 14.6 입력과 정답을 정수 시퀀스로 변환하기

```python
x_data = [char_to_index[char] for char in input_str]
y_data = [char_to_index[char] for char in label_str]

print(x_data)
print(y_data)
```

출력은 다음과 같다.

```text
x_data = [1, 4, 4, 3, 2]
y_data = [4, 4, 3, 2, 0]
```

각 인덱스를 문자로 해석하면 다음과 같다.

```text
x_data:
a, p, p, l, e

y_data:
p, p, l, e, !
```

---

### 14.7 배치 차원 추가하기

PyTorch의 `nn.RNN`에 `batch_first=True`를 적용하면 입력은 다음과 같은 3차원 텐서여야 한다.

```text
(batch_size, sequence_length, input_size)
```

현재는 하나의 시퀀스만 사용하므로 배치 크기는 1이다.

```python
x_data = [x_data]
y_data = [y_data]

print(x_data)
print(y_data)
```

출력은 다음과 같다.

```text
[[1, 4, 4, 3, 2]]
[[4, 4, 3, 2, 0]]
```

현재 정수 시퀀스의 형태는 다음과 같다.

```text
x_data:
(batch_size, sequence_length)
= (1, 5)

y_data:
(batch_size, sequence_length)
= (1, 5)
```

---

### 14.8 입력 문자를 원-핫 벡터로 변환하기

각 입력 문자를 문자 집합 크기 5의 원-핫 벡터로 변환한다.

```python
x_one_hot = [
    np.eye(vocab_size)[sequence]
    for sequence in x_data
]

print(x_one_hot)
```

결과는 다음 형태를 가진다.

```text
a → [0, 1, 0, 0, 0]
p → [0, 0, 0, 0, 1]
p → [0, 0, 0, 0, 1]
l → [0, 0, 0, 1, 0]
e → [0, 0, 1, 0, 0]
```

따라서 입력 전체의 크기는 다음과 같다.

```text
(batch_size, sequence_length, vocab_size)
= (1, 5, 5)
```

---

### 14.9 PyTorch 텐서로 변환하기

```python
X = torch.tensor(
    np.array(x_one_hot),
    dtype=torch.float32
)

Y = torch.tensor(
    y_data,
    dtype=torch.long
)
```

각 텐서의 자료형은 다음과 같이 설정한다.

```text
X:
RNN에 입력되는 실수형 원-핫 벡터
dtype = torch.float32

Y:
정답 클래스의 정수 인덱스
dtype = torch.long
```

텐서 크기를 확인한다.

```python
print("훈련 데이터의 크기:", X.shape)
print("레이블의 크기:", Y.shape)
```

출력은 다음과 같다.

```text
훈련 데이터의 크기: torch.Size([1, 5, 5])
레이블의 크기: torch.Size([1, 5])
```

크기의 의미는 다음과 같다.

```text
X.shape = (1, 5, 5)
          배치, 시점, 입력 차원

Y.shape = (1, 5)
          배치, 시점
```

정답은 원-핫 벡터가 아니라 각 시점의 정답 문자 인덱스로 저장한다.

---

### 14.10 문자 단위 RNN 모델 정의하기

```python
class Net(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int
    ) -> None:
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(
            in_features=hidden_size,
            out_features=output_size,
            bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rnn_output, _hidden = self.rnn(x)
        logits = self.fc(rnn_output)
        return logits
```

모델 구조는 다음과 같다.

```text
원-핫 문자 벡터
   ↓
RNN 계층
   ↓
각 시점의 은닉 상태
   ↓
완전 연결 계층
   ↓
각 문자 클래스에 대한 logit
```

`nn.RNN`은 모든 시점의 은닉 상태를 반환한다.

```text
rnn_output.shape
= (batch_size, sequence_length, hidden_size)
```

완전 연결 계층은 각 시점의 은닉 상태를 문자 클래스 점수로 변환한다.

```text
logits.shape
= (batch_size, sequence_length, output_size)
```

---

### 14.11 모델 생성 및 출력 크기 확인하기

```python
net = Net(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size
)

outputs = net(X)

print(outputs.shape)
```

출력은 다음과 같다.

```text
torch.Size([1, 5, 5])
```

각 차원의 의미는 다음과 같다.

```text
1:
배치 크기

5:
시점의 수

5:
각 시점에서 예측하는 문자 클래스의 수
```

---

### 14.12 손실 계산을 위해 텐서 펼치기

`CrossEntropyLoss`는 다중 클래스 분류에서 일반적으로 다음 형태의 입력을 받는다.

```text
모델 출력:
(N, C)

정답:
(N)
```

여기서 다음과 같다.

```text
N:
분류해야 할 전체 항목 수

C:
클래스 수
```

현재 출력은 3차원이다.

```text
outputs.shape = (1, 5, 5)
```

배치 차원과 시점 차원을 하나로 합치면 다음과 같다.

```python
flat_outputs = outputs.reshape(-1, output_size)

print(flat_outputs.shape)
```

출력은 다음과 같다.

```text
torch.Size([5, 5])
```

정답도 같은 방식으로 펼친다.

```python
flat_targets = Y.reshape(-1)

print(flat_targets.shape)
```

출력은 다음과 같다.

```text
torch.Size([5])
```

변환 전후를 비교하면 다음과 같다.

```text
모델 출력:
(배치, 시점, 클래스)
(1, 5, 5)
   ↓
(전체 문자 위치, 클래스)
(5, 5)

정답:
(배치, 시점)
(1, 5)
   ↓
(전체 문자 위치)
(5)
```

> 자료의 설명처럼 배치 차원을 단순히 제거하는 것이 아니라, 정확히는 `배치 차원과 시점 차원을 하나의 차원으로 합치는 것`이다.

---

### 14.13 손실 함수와 옵티마이저 정의하기

```python
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    net.parameters(),
    lr=learning_rate
)
```

`CrossEntropyLoss`는 각 시점에서 모델이 출력한 문자별 logit과 실제 문자 인덱스를 비교한다.

모델의 마지막에 softmax를 직접 추가하지 않는 이유는 `CrossEntropyLoss`가 내부적으로 log-softmax 연산을 포함하기 때문이다.

---

### 14.14 모델 학습하기

```python
for epoch in range(100):
    optimizer.zero_grad()

    outputs = net(X)

    loss = criterion(
        outputs.reshape(-1, output_size),
        Y.reshape(-1)
    )

    loss.backward()
    optimizer.step()

    predicted_indices = outputs.argmax(dim=2)

    predicted_string = "".join(
        index_to_char[index]
        for index in predicted_indices.squeeze(0).tolist()
    )

    print(
        epoch,
        "loss:",
        loss.item(),
        "prediction:",
        predicted_indices.tolist(),
        "true Y:",
        y_data,
        "prediction str:",
        predicted_string
    )
```

학습 과정은 다음과 같다.

```text
1. 이전 기울기를 0으로 초기화한다.
2. 입력 시퀀스를 모델에 넣어 각 시점의 logit을 계산한다.
3. 출력과 정답을 2차원과 1차원 형태로 펼친다.
4. CrossEntropyLoss로 모든 시점의 손실을 계산한다.
5. 역전파로 기울기를 계산한다.
6. Adam 옵티마이저로 매개변수를 갱신한다.
7. 각 시점에서 가장 큰 logit의 인덱스를 예측 문자로 선택한다.
8. 예측 인덱스를 다시 문자열로 변환한다.
```

초기에는 잘못된 문자를 예측할 수 있다.

```text
초기 예측 예:
pp!p!
```

학습이 진행되면 손실이 감소하고 정답 문자열에 가까워진다.

```text
최종 예측 예:
pple!
```

---

### 14.15 실습 1 전체 코드

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Net(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int
    ) -> None:
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rnn_output, _hidden = self.rnn(x)
        logits = self.fc(rnn_output)
        return logits


input_str = "apple"
label_str = "pple!"

char_vocab = sorted(set(input_str + label_str))
vocab_size = len(char_vocab)

char_to_index = {
    char: index
    for index, char in enumerate(char_vocab)
}

index_to_char = {
    index: char
    for char, index in char_to_index.items()
}

x_data = [
    [char_to_index[char] for char in input_str]
]

y_data = [
    [char_to_index[char] for char in label_str]
]

x_one_hot = np.eye(vocab_size, dtype=np.float32)[x_data]

X = torch.tensor(
    x_one_hot,
    dtype=torch.float32
)

Y = torch.tensor(
    y_data,
    dtype=torch.long
)

input_size = vocab_size
hidden_size = 5
output_size = vocab_size
learning_rate = 0.1

net = Net(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size
)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    net.parameters(),
    lr=learning_rate
)

for epoch in range(100):
    optimizer.zero_grad()

    outputs = net(X)

    loss = criterion(
        outputs.reshape(-1, output_size),
        Y.reshape(-1)
    )

    loss.backward()
    optimizer.step()

    predicted_indices = outputs.argmax(dim=2)

    predicted_string = "".join(
        index_to_char[index]
        for index in predicted_indices.squeeze(0).tolist()
    )

    print(
        f"{epoch:3d} "
        f"loss: {loss.item():.6f} "
        f"prediction: {predicted_string}"
    )
```

---

## 15. 실습 2: 긴 문장으로 문자 단위 RNN 학습하기

두 번째 실습에서는 하나의 긴 문장을 여러 개의 고정 길이 시퀀스로 나누어 학습한다.

사용할 문장은 다음과 같다.

```python
sentence = (
    "if you want to build a ship, don't drum up people together to "
    "collect wood and don't assign them tasks and work, but rather "
    "teach them to long for the endless immensity of the sea."
)
```

모델은 각 길이 10의 문자열을 입력받아, 한 글자 뒤로 이동한 길이 10의 문자열을 예측하도록 학습한다.

```text
입력:
if you wan

정답:
f you want
```

---

### 15.1 필요한 라이브러리 불러오기

자료의 두 번째 예제에서는 원-핫 인코딩에 `np.eye()`를 사용하므로 NumPy도 함께 불러와야 한다.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
```

---

### 15.2 문자 집합과 정수 인코딩 만들기

문장에 등장하는 모든 고유 문자를 수집한다.

자료의 원래 코드는 다음과 같다.

```python
char_set = list(set(sentence))
```

하지만 Python의 `set`은 실행 환경에 따라 순서가 달라질 수 있으므로, 재현 가능한 인덱스를 원한다면 정렬하는 편이 안전하다.

```python
char_set = sorted(set(sentence))
```

각 문자에 정수 인덱스를 부여한다.

```python
char_dic = {
    char: index
    for index, char in enumerate(char_set)
}
```

역변환을 위한 딕셔너리도 만든다.

```python
index_to_char = {
    index: char
    for char, index in char_dic.items()
}
```

문자 집합에는 알파벳뿐 아니라 다음 문자도 포함된다.

```text
공백
쉼표
작은따옴표
마침표
```

문자 집합의 크기를 구한다.

```python
dic_size = len(char_dic)

print("문자 집합의 크기:", dic_size)
```

자료의 예제에서는 문자 집합의 크기가 25이다.

```text
dic_size = 25
```

---

### 15.3 하이퍼파라미터 설정하기

```python
hidden_size = dic_size
sequence_length = 10
learning_rate = 0.1
```

각 값의 의미는 다음과 같다.

```text
hidden_size:
RNN 은닉 상태의 크기

sequence_length:
하나의 훈련 샘플이 포함하는 문자 수

learning_rate:
옵티마이저의 학습률
```

자료에서는 편의를 위해 `hidden_size`를 문자 집합의 크기와 같게 설정하였다.

```text
hidden_size = dic_size = 25
```

하지만 두 값이 반드시 같아야 하는 것은 아니다.

```text
입력과 출력 차원:
문자 집합의 크기로 결정

은닉 상태 크기:
사용자가 선택하는 하이퍼파라미터
```

---

### 15.4 슬라이딩 윈도우로 훈련 샘플 만들기

입력 문장을 길이 10의 부분 문자열로 자른다.

정답은 입력보다 한 문자 뒤에서 시작하도록 만든다.

```python
x_data = []
y_data = []

for start in range(
    len(sentence) - sequence_length
):
    x_str = sentence[
        start:start + sequence_length
    ]

    y_str = sentence[
        start + 1:start + sequence_length + 1
    ]

    print(
        start,
        x_str,
        "->",
        y_str
    )

    x_data.append(
        [char_dic[char] for char in x_str]
    )

    y_data.append(
        [char_dic[char] for char in y_str]
    )
```

초기 샘플은 다음과 같다.

```text
0  if you wan  ->  f you want
1  f you want  ->   you want 
2   you want   ->  you want t
3  you want t  ->  ou want to
4  ou want to  ->  u want to 
```

각 샘플은 이전 샘플보다 한 문자씩 오른쪽으로 이동한다.

```text
원래 문장:
i f   y o u   w a n t ...

샘플 0 입력:
i f   y o u   w a n

샘플 0 정답:
f   y o u   w a n t

샘플 1 입력:
f   y o u   w a n t

샘플 1 정답:
  y o u   w a n t  
```

이와 같은 생성 방식을 `슬라이딩 윈도우(Sliding Window)` 방식으로 볼 수 있다.

자료에서는 총 170개의 샘플이 생성된다.

```text
샘플 수 = 170
각 샘플 길이 = 10
```

---

### 15.5 첫 번째 샘플 확인하기

```python
print(x_data[0])
print(y_data[0])
```

정수 인덱스의 실제 값은 문자 집합의 정렬 방식에 따라 달라질 수 있다.

중요한 것은 다음 문자 관계이다.

```text
x_data[0]:
"if you wan"

y_data[0]:
"f you want"
```

입력과 정답은 한 문자만큼 이동한 관계이다.

---

### 15.6 원-핫 인코딩과 텐서 변환

입력 정수 시퀀스를 원-핫 벡터로 변환한다.

```python
x_one_hot = np.eye(
    dic_size,
    dtype=np.float32
)[x_data]
```

PyTorch 텐서로 변환한다.

```python
X = torch.tensor(
    x_one_hot,
    dtype=torch.float32
)

Y = torch.tensor(
    y_data,
    dtype=torch.long
)
```

크기를 확인한다.

```python
print("훈련 데이터의 크기:", X.shape)
print("레이블의 크기:", Y.shape)
```

출력은 다음과 같다.

```text
훈련 데이터의 크기: torch.Size([170, 10, 25])
레이블의 크기: torch.Size([170, 10])
```

각 차원의 의미는 다음과 같다.

```text
X.shape:
(샘플 수, 시퀀스 길이, 문자 집합 크기)
= (170, 10, 25)

Y.shape:
(샘플 수, 시퀀스 길이)
= (170, 10)
```

이 예제에서는 생성한 170개의 시퀀스를 하나의 큰 배치처럼 한 번에 모델에 입력한다.

---

### 15.7 첫 번째 원-핫 인코딩 샘플 해석하기

```python
print(X[0])
```

첫 번째 샘플은 길이 10의 문자열 `"if you wan"`에 해당한다.

```text
X[0].shape
= (10, 25)
```

각 행은 하나의 문자를 나타내는 25차원 원-핫 벡터이다.

```text
첫 번째 행:
문자 i

두 번째 행:
문자 f

세 번째 행:
공백

...

열 번째 행:
문자 n
```

정답 시퀀스를 출력한다.

```python
print(Y[0])
```

`Y[0]`은 `"f you want"`에 해당하는 문자 인덱스 시퀀스이다.

---

### 15.8 2층 문자 단위 RNN 모델 정의하기

두 번째 실습에서는 RNN 은닉층을 두 개 쌓는다.

```python
class Net(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int
    ) -> None:
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            in_features=hidden_dim,
            out_features=output_dim,
            bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rnn_output, _hidden = self.rnn(x)
        logits = self.fc(rnn_output)
        return logits
```

모델을 생성한다.

```python
net = Net(
    input_dim=dic_size,
    hidden_dim=hidden_size,
    output_dim=dic_size,
    num_layers=2
)
```

자료의 코드에서는 `hidden_size`와 `dic_size`가 같기 때문에 출력층을 다음처럼 작성해도 동작한다.

```python
nn.Linear(hidden_dim, hidden_dim)
```

하지만 문자 클래스 수를 명확히 표현하려면 출력 차원을 `output_dim=dic_size`로 별도 표기하는 편이 구조를 이해하기 쉽다.

---

### 15.9 2층 RNN의 데이터 흐름

`num_layers=2`는 순환 은닉층을 두 층 쌓는다는 의미이다.

```text
문자 원-핫 벡터
   ↓
첫 번째 RNN 층
   ↓
두 번째 RNN 층
   ↓
완전 연결 출력층
   ↓
각 문자 클래스에 대한 logit
```

첫 번째 RNN 층의 모든 시점 출력이 두 번째 RNN 층의 입력으로 사용된다.

모델이 최종적으로 반환하는 것은 마지막 RNN 층의 모든 시점 출력이다.

---

### 15.10 모델 출력 크기 확인하기

```python
outputs = net(X)

print(outputs.shape)
```

출력은 다음과 같다.

```text
torch.Size([170, 10, 25])
```

각 차원의 의미는 다음과 같다.

```text
170:
훈련 샘플 수

10:
각 샘플의 시점 수

25:
각 시점의 문자 클래스 수
```

---

### 15.11 손실 계산을 위한 출력과 정답 펼치기

출력을 2차원으로 변환한다.

```python
flat_outputs = outputs.reshape(
    -1,
    dic_size
)

print(flat_outputs.shape)
```

출력은 다음과 같다.

```text
torch.Size([1700, 25])
```

정답도 1차원으로 변환한다.

```python
flat_targets = Y.reshape(-1)

print(flat_targets.shape)
```

출력은 다음과 같다.

```text
torch.Size([1700])
```

변환 과정은 다음과 같다.

```text
출력:
(170, 10, 25)
   ↓
(170 × 10, 25)
   ↓
(1700, 25)
```

```text
정답:
(170, 10)
   ↓
(170 × 10)
   ↓
(1700)
```

결국 170개 샘플의 10개 시점을 각각 하나의 문자 분류 문제로 취급한다.

```text
전체 문자 예측 위치 수:
170 × 10 = 1700
```

---

### 15.12 손실 함수와 옵티마이저 정의하기

```python
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    net.parameters(),
    lr=learning_rate
)
```

각 문자 위치에 대해 25개 클래스 중 정답 문자를 맞히도록 학습한다.

---

### 15.13 모델 학습하기

```python
for epoch in range(100):
    optimizer.zero_grad()

    outputs = net(X)

    loss = criterion(
        outputs.reshape(-1, dic_size),
        Y.reshape(-1)
    )

    loss.backward()
    optimizer.step()

    predicted_indices = outputs.argmax(dim=2)

    predicted_string = ""

    for sample_index, result in enumerate(
        predicted_indices
    ):
        if sample_index == 0:
            predicted_string += "".join(
                index_to_char[index]
                for index in result.tolist()
            )
        else:
            predicted_string += index_to_char[
                result[-1].item()
            ]

    print(
        f"{epoch:3d} "
        f"loss: {loss.item():.6f} "
        f"prediction: {predicted_string}"
    )
```

---

### 15.14 겹치는 예측 시퀀스를 하나의 문자열로 합치기

각 훈련 샘플의 예측 결과는 길이 10이다.

```text
샘플 0 예측:
10개 문자

샘플 1 예측:
10개 문자

샘플 2 예측:
10개 문자
```

하지만 각 샘플은 이전 샘플에서 한 문자만 이동한 형태이므로 대부분의 문자가 서로 겹친다.

따라서 다음 방식으로 하나의 긴 예측 문자열을 만든다.

```text
첫 번째 샘플:
예측된 10개 문자를 모두 사용

두 번째 샘플부터:
각 샘플의 마지막 예측 문자만 추가
```

코드에서는 다음 부분이 해당 동작을 수행한다.

```python
if sample_index == 0:
    predicted_string += "".join(
        index_to_char[index]
        for index in result.tolist()
    )
else:
    predicted_string += index_to_char[
        result[-1].item()
    ]
```

예를 들어 다음과 같다.

```text
샘플 0:
f you want

샘플 1:
 you want 

샘플 2:
you want t
```

첫 번째 예측은 전체를 사용하고 이후 예측에서는 마지막 문자만 이어 붙인다.

```text
f you want
          + 공백
          + t
          + ...
```

이 방식으로 원래 문장과 비슷한 길이의 연속된 문자열을 복원한다.

---

### 15.15 학습 결과 해석

학습 초기에는 모델의 가중치가 임의로 초기화되어 있으므로 의미 없는 문자열을 출력한다.

```text
초기 출력 예:
hahha...
```

학습이 진행되면 각 입력 문자 윈도우에 대응하는 다음 문자 패턴을 예측한다.

최종적으로는 겹치는 윈도우의 예측 결과를 연결하여 다음과 같이 원래 문장에 가까운 문자열을 복원할 수 있다.

```text
p you want to build a ship, don't drum up people together to
collect wood and don't assign them tasks and work, but rather
teach them to long for the endless immensity of the sea.
```

이 실습은 모델이 방금 예측한 문자를 다시 다음 입력으로 넣는 자유 생성 방식이 아니다.

```text
현재 실습:
원래 문장에서 만든 모든 슬라이딩 윈도우를 입력으로 제공
   ↓
각 윈도우의 다음 문자를 예측
   ↓
겹치는 예측 결과를 연결하여 문자열 복원

자기회귀 문자 생성:
초기 문자 또는 프롬프트만 제공
   ↓
예측한 문자를 다음 시점의 입력으로 다시 사용
   ↓
새 문자를 반복적으로 생성
```

첫 글자가 원문과 다를 수 있는 이유는 첫 입력 문자열에 대한 정답이 원문보다 한 문자 뒤에서 시작하기 때문이다.

```text
첫 입력:
if you wan

첫 정답:
f you want
```

자료의 출력 예시에서 첫 문자가 정답과도 다르게 나타난 것은 학습이 완전히 수렴하지 않았거나 해당 위치에서 예측 오차가 남아 있기 때문이다.

---

### 15.16 실습 2 전체 코드

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Net(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int
    ) -> None:
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_dim,
            output_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rnn_output, _hidden = self.rnn(x)
        logits = self.fc(rnn_output)
        return logits


sentence = (
    "if you want to build a ship, don't drum up people together to "
    "collect wood and don't assign them tasks and work, but rather "
    "teach them to long for the endless immensity of the sea."
)

char_set = sorted(set(sentence))

char_dic = {
    char: index
    for index, char in enumerate(char_set)
}

index_to_char = {
    index: char
    for char, index in char_dic.items()
}

dic_size = len(char_dic)

hidden_size = dic_size
sequence_length = 10
learning_rate = 0.1

x_data = []
y_data = []

for start in range(
    len(sentence) - sequence_length
):
    x_str = sentence[
        start:start + sequence_length
    ]

    y_str = sentence[
        start + 1:start + sequence_length + 1
    ]

    x_data.append(
        [char_dic[char] for char in x_str]
    )

    y_data.append(
        [char_dic[char] for char in y_str]
    )

x_one_hot = np.eye(
    dic_size,
    dtype=np.float32
)[x_data]

X = torch.tensor(
    x_one_hot,
    dtype=torch.float32
)

Y = torch.tensor(
    y_data,
    dtype=torch.long
)

net = Net(
    input_dim=dic_size,
    hidden_dim=hidden_size,
    output_dim=dic_size,
    num_layers=2
)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    net.parameters(),
    lr=learning_rate
)

for epoch in range(100):
    optimizer.zero_grad()

    outputs = net(X)

    loss = criterion(
        outputs.reshape(-1, dic_size),
        Y.reshape(-1)
    )

    loss.backward()
    optimizer.step()

    predicted_indices = outputs.argmax(dim=2)

    predicted_string = ""

    for sample_index, result in enumerate(
        predicted_indices
    ):
        if sample_index == 0:
            predicted_string += "".join(
                index_to_char[index]
                for index in result.tolist()
            )
        else:
            predicted_string += index_to_char[
                result[-1].item()
            ]

    print(
        f"{epoch:3d} "
        f"loss: {loss.item():.6f} "
        f"prediction: {predicted_string}"
    )
```

---

## 16. 두 문자 단위 RNN 실습 비교

| 구분 | 실습 1 | 실습 2 |
|---|---|---|
| 입력 문자열 | `apple` | 긴 영어 문장 |
| 정답 문자열 | `pple!` | 입력보다 한 문자 이동한 문장 |
| 문자 집합 크기 | 5 | 25 |
| 시퀀스 길이 | 5 | 10 |
| 샘플 수 | 1 | 170 |
| RNN 층 수 | 1 | 2 |
| 입력 크기 | `(1, 5, 5)` | `(170, 10, 25)` |
| 출력 크기 | `(1, 5, 5)` | `(170, 10, 25)` |
| 학습 목적 | 기본 동작 확인 | 긴 문장의 다음 문자 패턴 학습 |

두 실습 모두 핵심 학습 목표는 동일하다.

```text
현재까지 주어진 문자 시퀀스
   ↓
각 시점의 RNN 은닉 상태 계산
   ↓
완전 연결층으로 문자별 logit 계산
   ↓
다음 위치의 정답 문자와 비교
   ↓
모든 시점의 손실을 이용해 학습
```

---

## 17. 헷갈리기 쉬운 부분 정리

### 17.1 RNN을 펼친 그림의 셀들은 서로 다른 모델이 아니다

시간축으로 펼친 그림에서는 여러 개의 셀이 있는 것처럼 보이지만, 실제로는 하나의 RNN 셀과 같은 가중치를 각 시점에 반복 적용한 것이다.

```text
시간축 방향:
같은 가중치 공유

층 방향:
서로 다른 층은 서로 다른 가중치 사용
```

---

### 17.2 은닉 상태와 출력은 항상 같은 것이 아니다

`h_t`는 RNN 셀이 유지하고 다음 시점으로 전달하는 은닉 상태이다.

`y_t`는 필요에 따라 은닉 상태에 별도의 출력층을 적용해 만든 최종 출력이다.

```math
y_t=f(W_yh_t+b_y)
```

NumPy 직접 구현이나 PyTorch의 `nn.RNN` 설명에서는 은닉 상태 자체를 계층의 출력이라고 부르기도 하므로 문맥을 구분해야 한다.

---

### 17.3 `sequence_length`, `hidden_size`, `num_layers`는 서로 다른 값이다

```text
sequence_length:
입력 시퀀스에 포함된 시점의 수

hidden_size:
각 시점의 은닉 상태 벡터 차원

num_layers:
위아래로 쌓은 순환 은닉층의 수
```

시퀀스가 길어진다고 은닉 상태 차원이나 층 수가 자동으로 증가하는 것은 아니다.

---

### 17.4 `batch_first=True`는 모든 반환 텐서에 적용되지 않는다

`batch_first=True`를 사용하면 입력과 `outputs`의 배치 차원이 첫 번째가 된다.

```text
inputs, outputs:
(batch, sequence, feature)
```

하지만 최종 상태는 다음 축 순서를 유지한다.

```text
RNN·GRU의 h_n:
(num_layers × num_directions, batch, hidden)

LSTM의 h_n, c_n:
(num_layers × num_directions, batch, hidden)
```

---

### 17.5 `outputs`와 최종 상태의 차이

```text
outputs:
마지막 순환층의 모든 시점 출력

h_n 또는 status:
각 층과 방향의 최종 은닉 상태

c_n:
LSTM 각 층과 방향의 최종 셀 상태
```

다층 RNN에서 `outputs`에는 중간 층의 모든 출력이 포함되지 않는다.

---

### 17.6 양방향 RNN의 출력 차원은 두 배가 된다

정방향과 역방향의 은닉 상태를 연결하므로 `outputs`의 마지막 차원은 다음과 같다.

```text
hidden_size × num_directions
```

양방향에서는 `num_directions=2`이므로 마지막 차원이 `hidden_size × 2`가 된다.

---

### 17.7 softmax와 CrossEntropyLoss의 관계

다중 클래스 확률을 확인하려면 softmax를 사용할 수 있다.

하지만 PyTorch의 `CrossEntropyLoss`에는 softmax를 적용하기 전의 logit을 전달해야 한다.

```text
잘못된 학습 흐름:
모델 → softmax → CrossEntropyLoss

권장 학습 흐름:
모델 → logits → CrossEntropyLoss
```

클래스 선택만 필요하면 logit에서 바로 `argmax`를 사용해도 된다.

---

### 17.8 LSTM의 셀 상태와 은닉 상태는 역할이 다르다

```text
C_t:
장기 정보를 전달하는 셀 상태

h_t:
현재 출력과 다음 시점 계산에 사용하는 은닉 상태
```

LSTM은 두 상태를 모두 다음 시점으로 전달한다.

---

### 17.9 LSTM의 게이트 값 자체가 기억 내용은 아니다

시그모이드를 통과한 게이트 값은 0과 1 사이에서 정보의 통과 비율을 조절한다.

```text
f_t:
이전 셀 상태를 얼마나 유지할지 조절

i_t:
후보 기억을 얼마나 새로 저장할지 조절

o_t:
셀 상태를 은닉 상태로 얼마나 출력할지 조절
```

실제 기억 내용은 셀 상태와 후보 기억에 저장된다.

---

### 17.10 원소별 곱과 행렬 곱을 구분해야 한다

```text
행렬 곱:
W_xx_t, W_hh_{t-1}

원소별 곱:
f_t ∘ C_{t-1}, i_t ∘ g_t, r_t ∘ h_{t-1}
```

게이트는 같은 위치의 원소마다 정보의 통과량을 조절하므로 원소별 곱을 사용한다.

---

### 17.11 자료의 GRU 수식과 PyTorch `nn.GRU`의 후보 상태 계산은 미세하게 다르다

자료의 수식은 리셋 게이트를 이전 은닉 상태에 먼저 적용한다.

PyTorch는 이전 은닉 상태를 선형 변환한 결과에 리셋 게이트를 적용한다.

핵심 역할은 같지만 정확한 내부 수식이 동일하지 않으므로 구현 분석 시 구분해야 한다.

---

### 17.12 문자 집합 크기와 은닉 상태 크기는 같을 필요가 없다

```text
vocab_size:
원-핫 입력의 차원과 출력 문자 클래스 수

hidden_size:
RNN 내부 은닉 상태의 차원
```

실습에서는 편의상 두 값을 같게 설정한 경우가 있지만, 일반적으로는 서로 독립적인 하이퍼파라미터이다.

---

### 17.13 CrossEntropyLoss를 위한 `reshape`의 의미

문자 RNN 출력은 다음 3차원 형태이다.

```text
(batch, sequence, classes)
```

손실 계산에서는 배치와 시점 차원을 합쳐 다음 형태로 만든다.

```text
모델 출력:
(batch × sequence, classes)

정답:
(batch × sequence)
```

이는 배치 차원을 단순히 제거하는 것이 아니라 모든 문자 위치를 독립적인 클래스 분류 항목으로 펼치는 것이다.

---

### 17.14 두 번째 문자 RNN 실습은 자유 생성이 아니다

두 번째 실습은 원래 문장에서 만든 슬라이딩 윈도우를 모두 입력으로 사용한다.

따라서 학습 문장의 다음 문자를 예측하고 결과를 연결해 복원하는 실습이며, 예측 문자를 다시 입력으로 넣어 새로운 문장을 만드는 자기회귀 생성과는 다르다.

---

### 17.15 `set`으로 만든 문자 집합의 인덱스 순서는 고정되지 않을 수 있다

```python
char_set = list(set(sentence))
```

위 코드는 실행 환경에 따라 문자 인덱스 순서가 달라질 수 있다.

재현 가능한 정수 인코딩을 원하면 다음처럼 정렬한다.

```python
char_set = sorted(set(sentence))
```

---

## 18. 전체 내용을 한 줄로 요약

RNN은 이전 은닉 상태를 반복적으로 전달해 시퀀스를 처리하고, LSTM과 GRU는 게이트로 장기 정보의 흐름을 조절하며, 문자 단위 RNN은 이 구조를 문자별 다음 문자 예측에 적용한다.

