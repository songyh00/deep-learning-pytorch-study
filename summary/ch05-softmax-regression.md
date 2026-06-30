# 소프트맥스 회귀(Softmax Regression)와 원-핫 인코딩(One-hot Encoding) 정리

## 0. 전체 흐름 먼저 보기

이번 정리는 `원-핫 인코딩`, `소프트맥스 회귀`, `크로스 엔트로피 비용 함수`, `PyTorch 구현`, `MNIST 분류 실습`까지 이어지는 내용이다.

전체 흐름은 다음과 같다.

```text
1. 다중 클래스 분류에서는 정답 후보가 3개 이상이다.
2. 정답 클래스를 표현할 때 원-핫 인코딩을 사용할 수 있다.
3. 소프트맥스 회귀는 각 클래스에 대한 점수를 확률 벡터로 바꾼다.
4. 예측 확률 벡터와 실제 정답 벡터를 비교한다.
5. 비교에는 크로스 엔트로피 비용 함수를 사용한다.
6. PyTorch에서는 F.cross_entropy() 또는 nn.CrossEntropyLoss()로 구현할 수 있다.
7. MNIST처럼 이미지 데이터도 28 × 28 이미지를 784차원 벡터로 펼쳐 소프트맥스 회귀에 넣을 수 있다.
```

소프트맥스 회귀의 가장 큰 흐름은 다음과 같이 볼 수 있다.

```text
입력 데이터 X
   ↓
선형 연산 XW + B
   ↓
softmax 적용
   ↓
각 클래스에 대한 확률 벡터 출력
   ↓
실제 정답과 비교하여 Cross Entropy cost 계산
   ↓
역전파로 W와 B 업데이트
```

이번 정리에서 사용하는 주요 기호는 다음과 같다.

```text
X: 입력 데이터
W: 가중치
B 또는 b: 편향
Ŷ 또는 ŷ: 모델의 예측값
Y 또는 y: 실제 정답
z: softmax에 들어가기 전의 점수값, 즉 logit
```

여기서 $X$, $W$, $B$, $\hat{Y}$는 상황에 따라 벡터 또는 행렬로 표현한다.

---

## 1. 원-핫 인코딩(One-hot Encoding)

### 1.1 원-핫 인코딩이란?

`원-핫 인코딩(One-hot Encoding)`은 여러 개의 선택지를 벡터로 표현하는 방법이다.

선택해야 하는 선택지의 개수만큼 차원을 만들고, 해당 선택지의 위치에는 `1`, 나머지 위치에는 `0`을 넣는다.

예를 들어 선택지가 다음과 같이 3개 있다고 하자.

```text
강아지
고양이
냉장고
```

먼저 각 선택지에 인덱스를 부여한다.

| 선택지 | 인덱스 |
|---|:---:|
| 강아지 | 0 |
| 고양이 | 1 |
| 냉장고 | 2 |

선택지가 총 3개이므로 원-핫 벡터는 모두 3차원 벡터가 된다.

```text
강아지 = [1, 0, 0]
고양이 = [0, 1, 0]
냉장고 = [0, 0, 1]
```

여기서 중요한 점은 다음과 같다.

```text
1. 선택지 개수만큼 벡터 차원이 만들어진다.
2. 해당 선택지의 인덱스 위치만 1이 된다.
3. 나머지 위치는 모두 0이 된다.
```

이렇게 표현된 벡터를 `원-핫 벡터(One-hot Vector)`라고 한다.

---

### 1.2 원-핫 벡터 예시

붓꽃 품종 분류 문제에서 클래스가 다음과 같이 3개 있다고 하자.

```text
virginica
setosa
versicolor
```

임의로 다음과 같이 순서를 정할 수 있다.

| 클래스 | 인덱스 | 원-핫 벡터 |
|---|:---:|:---:|
| virginica | 0 | [1, 0, 0] |
| setosa | 1 | [0, 1, 0] |
| versicolor | 2 | [0, 0, 1] |

이때 실제 정답이 `setosa`라면 실제값은 숫자 2가 아니라 다음과 같은 벡터로 표현할 수 있다.

```text
setosa의 원-핫 벡터 = [0, 1, 0]
```

즉, 다중 클래스 분류에서는 정답을 하나의 숫자로만 보는 것이 아니라, 각 클래스에 대한 위치 정보를 가진 벡터로 표현할 수 있다.

---

## 2. 정수 인코딩과 원-핫 인코딩의 차이

### 2.1 정수 인코딩(Integer Encoding)

다중 클래스 분류에서 가장 단순하게 생각할 수 있는 방법은 각 클래스에 정수 번호를 붙이는 것이다.

예를 들어 다음과 같은 클래스가 있다고 하자.

```text
Banana
Tomato
Apple
```

이를 정수로 표현하면 다음과 같이 둘 수 있다.

```text
Banana = 1
Tomato = 2
Apple = 3
```

이런 방식을 `정수 인코딩(Integer Encoding)`이라고 볼 수 있다.

---

### 2.2 정수 인코딩의 문제점

정수 인코딩은 보기에는 단순하지만, 일반적인 분류 문제에서는 원하지 않는 의미를 줄 수 있다.

제곱 오차를 계산할 때 사용할 수 있는 평균 제곱 오차(MSE)는 다음과 같다.

$$
Loss\ function = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

예를 들어 실제값이 `Tomato`인데 모델이 `Banana`라고 예측했다고 하자.

```text
실제값: Tomato = 2
예측값: Banana = 1
```

제곱 오차를 계산하면 다음과 같다.

$$
(2 - 1)^2 = 1
$$

이번에는 실제값이 `Apple`인데 모델이 `Banana`라고 예측했다고 하자.

```text
실제값: Apple = 3
예측값: Banana = 1
```

제곱 오차는 다음과 같다.

$$
(3 - 1)^2 = 4
$$

즉, 정수 인코딩을 사용하면 다음과 같은 해석이 생긴다.

```text
Banana와 Tomato의 차이 = 1
Banana와 Apple의 차이 = 4
```

이는 모델에게 다음과 같은 잘못된 정보를 줄 수 있다.

```text
Banana는 Apple보다 Tomato에 더 가깝다.
```

하지만 일반적인 과일 분류 문제에서 사용자가 의도한 것은 이런 순서 관계가 아니다.

---

### 2.3 클래스에 순서 의미가 있는 경우

정수 인코딩이 항상 잘못된 것은 아니다.

클래스 자체에 순서 의미가 있는 경우에는 정수 인코딩이 자연스러울 수 있다.

예를 들어 다음과 같은 경우이다.

```text
1. baby / child / adolescent / adult
2. 1층 / 2층 / 3층 / 4층
3. 10대 / 20대 / 30대 / 40대
```

이런 경우에는 클래스 사이에 어느 정도 순서 관계가 존재한다.

하지만 일반적인 다중 클래스 분류에서는 클래스 사이에 순서가 없는 경우가 많다.

```text
setosa, versicolor, virginica 사이에는 숫자 순서의 의미가 없다.
```

따라서 이런 문제에서는 정수 인코딩보다 원-핫 인코딩이 더 적절하다.

---

## 3. 원-핫 벡터의 무작위성

### 3.1 원-핫 벡터는 클래스 간 거리를 균등하게 만든다

원-핫 인코딩을 사용하면 각 클래스는 서로 같은 거리 관계를 가진다.

예를 들어 세 개의 클래스를 다음과 같이 원-핫 벡터로 표현했다고 하자.

```text
Class 1 = [1, 0, 0]
Class 2 = [0, 1, 0]
Class 3 = [0, 0, 1]
```

Class 1과 Class 2 사이의 제곱 오차는 다음과 같다.

$$
((1,0,0) - (0,1,0))^2 = (1-0)^2 + (0-1)^2 + (0-0)^2 = 2
$$

Class 1과 Class 3 사이의 제곱 오차도 다음과 같다.

$$
((1,0,0) - (0,0,1))^2 = (1-0)^2 + (0-0)^2 + (0-1)^2 = 2
$$

즉, 원-핫 벡터에서는 서로 다른 클래스 사이의 거리가 동일하다.

```text
Class 1과 Class 2의 거리 = 동일
Class 1과 Class 3의 거리 = 동일
Class 2와 Class 3의 거리 = 동일
```

---

### 3.2 원-핫 벡터의 의미

원-핫 벡터가 클래스 간 거리를 균등하게 만든다는 것은 다음과 같은 의미를 가진다.

```text
각 클래스 사이에 특별한 순서 관계를 부여하지 않는다.
```

즉, 모델에게 다음과 같은 정보를 주지 않는다.

```text
Class 1은 Class 2와 더 가깝고,
Class 3과는 더 멀다.
```

이런 성질 때문에 일반적인 다중 클래스 분류에서는 원-핫 인코딩을 자주 사용한다.

다만 원-핫 벡터는 모든 클래스 사이의 관계를 균등하게 보기 때문에, 단어 사이의 의미적 유사성 같은 정보를 표현하지 못한다는 단점도 있다.

예를 들어 다음과 같은 관계를 표현하기 어렵다.

```text
강아지와 고양이는 냉장고보다 의미적으로 더 가깝다.
```

원-핫 벡터에서는 세 벡터가 서로 모두 같은 거리 관계를 가지기 때문이다.

---

## 4. 다중 클래스 분류(Multi-class Classification)

### 4.1 다중 클래스 분류란?

`다중 클래스 분류(Multi-class Classification)`는 세 개 이상의 클래스 중 하나를 선택하는 문제이다.

이진 분류가 다음과 같은 문제라면,

```text
0 또는 1
False 또는 True
불합격 또는 합격
```

다중 클래스 분류는 다음과 같은 문제이다.

```text
Class 1 또는 Class 2 또는 Class 3
setosa 또는 versicolor 또는 virginica
0 또는 1 또는 2 또는 ... 또는 9
```

---

### 4.2 붓꽃 품종 분류 문제

붓꽃 품종 분류 문제는 대표적인 다중 클래스 분류 문제이다.

입력 데이터는 꽃의 여러 특성값이고, 출력값은 붓꽃 품종이다.

| SepalLengthCm($x_1$) | SepalWidthCm($x_2$) | PetalLengthCm($x_3$) | PetalWidthCm($x_4$) | Species($y$) |
|---|---|---|---|---|
| 5.1 | 3.5 | 1.4 | 0.2 | setosa |
| 4.9 | 3.0 | 1.4 | 0.2 | setosa |
| 5.8 | 2.6 | 4.0 | 1.2 | versicolor |
| 6.7 | 3.0 | 5.2 | 2.3 | virginica |
| 5.6 | 2.8 | 4.9 | 2.0 | virginica |

여기서 입력 특성은 총 4개이다.

```text
x1 = SepalLengthCm
x2 = SepalWidthCm
x3 = PetalLengthCm
x4 = PetalWidthCm
```

출력 클래스는 총 3개이다.

```text
setosa
versicolor
virginica
```

따라서 이 문제는 다음과 같이 볼 수 있다.

```text
4개의 입력 특성을 이용해서
3개의 붓꽃 품종 중 하나를 예측하는 문제
```

---

## 5. 로지스틱 회귀 복습

### 5.1 로지스틱 회귀의 가설식

로지스틱 회귀에서는 선형식에 시그모이드 함수를 적용한다.

$$
H(X) = sigmoid(WX + B)
$$

흐름은 다음과 같다.

```text
입력 X
   ↓
WX + B 계산
   ↓
sigmoid 적용
   ↓
0과 1 사이의 값 출력
   ↓
0.5 기준으로 두 클래스 중 하나로 분류
```

<img width="563" height="140" alt="Image" src="https://github.com/user-attachments/assets/7e789a33-3e68-4840-90ec-95a5e86ba1aa" />

예를 들어 출력값이 다음과 같다고 하자.

```text
H(X) = 0.75
```

그러면 이를 다음처럼 해석할 수 있다.

```text
Class 1일 확률이 75%이다.
Class 1이 아닐 확률은 25%이다.
```

보통 이진 분류에서는 다음과 같이 판단한다.

```text
H(X) >= 0.5  → Class 1
H(X) < 0.5   → Class 2
```

---

### 5.2 로지스틱 회귀의 한계

로지스틱 회귀는 기본적으로 두 개의 클래스 중 하나를 고르는 문제에 적합하다.

하지만 붓꽃 품종 분류처럼 클래스가 3개 이상이면 하나의 확률값만으로는 부족하다.

예를 들어 붓꽃 품종 분류에서는 다음 세 가지 확률이 필요하다.

```text
setosa일 확률
versicolor일 확률
virginica일 확률
```

따라서 다중 클래스 분류에서는 각 클래스마다 확률을 출력할 수 있는 방법이 필요하다.

---

## 6. 소프트맥스 회귀(Softmax Regression)

### 6.1 소프트맥스 회귀란?

`소프트맥스 회귀(Softmax Regression)`는 다중 클래스 분류 문제에 사용되는 모델이다.

로지스틱 회귀가 이진 분류에 사용된다면, 소프트맥스 회귀는 세 개 이상의 클래스를 분류할 때 사용한다.

```text
로지스틱 회귀  → 이진 분류
소프트맥스 회귀 → 다중 클래스 분류
```

소프트맥스 회귀의 가설식은 다음과 같다.

$$
H(X) = softmax(WX + B)
$$

---

### 6.2 소프트맥스 회귀의 출력

소프트맥스 회귀는 각 클래스에 대한 확률값을 출력한다.

예를 들어 클래스가 3개라면 출력은 다음과 같은 벡터가 된다.

```text
[0.15, 0.75, 0.10]
```

각 원소는 해당 클래스가 정답일 확률을 의미한다.

```text
0.15 → Class 1일 확률
0.75 → Class 2일 확률
0.10 → Class 3일 확률
```

소프트맥스 함수의 출력값은 전체 합이 1이 되도록 정규화된다.

```text
0.15 + 0.75 + 0.10 = 1
```

즉, 소프트맥스 회귀의 출력값은 다음 조건을 만족한다.

```text
1. 각 원소는 0과 1 사이의 값이다.
2. 모든 원소의 합은 1이다.
3. 각 원소는 각 클래스가 정답일 확률로 해석할 수 있다.
```

---

### 6.3 소프트맥스 회귀의 분류 기준

소프트맥스 회귀에서는 가장 큰 확률값을 가진 클래스를 최종 예측값으로 선택한다.

예를 들어 다음과 같은 출력이 있다고 하자.

```text
[0.15, 0.75, 0.10]
```

가장 큰 값은 `0.75`이다.

```text
Class 1 확률 = 0.15
Class 2 확률 = 0.75
Class 3 확률 = 0.10
```

따라서 최종 예측은 다음과 같다.

```text
예측 클래스 = Class 2
```

---

## 7. 소프트맥스 함수(Softmax Function)

### 7.1 소프트맥스 함수가 필요한 이유

다중 클래스 분류에서는 각 클래스마다 하나의 점수가 나온다.

예를 들어 클래스가 3개라면 모델은 다음과 같은 값들을 만들 수 있다.

```text
z1, z2, z3
```

이 값들은 아직 확률이 아니다.

```text
1. 음수가 될 수도 있다.
2. 1보다 클 수도 있다.
3. 전체 합이 1이 아닐 수도 있다.
```

따라서 이 점수들을 확률처럼 해석할 수 있도록 바꾸는 함수가 필요하다.

그 역할을 하는 함수가 `소프트맥스 함수(Softmax Function)`이다.

---

### 7.2 소프트맥스 함수 식

클래스의 개수가 `k`개이고, 각 클래스에 대한 점수를 다음과 같이 둔다.

```text
z1, z2, ..., zk
```

이때 i번째 클래스가 정답일 확률을 $p_i$라고 하면 소프트맥스 함수는 다음과 같이 정의된다.

$$
p_i = \frac{e^{z_i}}{\sum_{j=1}^{k} e^{z_j}} \quad for \quad i = 1, 2, ..., k
$$

여기서 각 기호의 의미는 다음과 같다.

```text
zi:
 i번째 클래스에 대한 점수

exp(zi) 또는 e^zi:
 zi를 양수 값으로 변환한 값

Σ e^zj:
 모든 클래스 점수에 exp를 적용한 뒤 더한 값

pi:
 i번째 클래스가 정답일 확률
```

---

### 7.3 클래스가 3개일 때의 소프트맥스

붓꽃 품종 분류 문제에서는 클래스가 3개이다.

```text
virginica
setosa
versicolor
```

클래스가 3개이므로 소프트맥스 함수의 입력도 3차원 벡터로 볼 수 있다.

$$
z = [z_1, z_2, z_3]
$$

소프트맥스 출력은 다음과 같다.

$$
softmax(z) =
\left[
\frac{e^{z_1}}{\sum_{j=1}^{3} e^{z_j}},
\frac{e^{z_2}}{\sum_{j=1}^{3} e^{z_j}},
\frac{e^{z_3}}{\sum_{j=1}^{3} e^{z_j}}
\right]
$$

이를 확률 벡터로 쓰면 다음과 같다.

$$
softmax(z) = [p_1, p_2, p_3] = \hat{y}
$$

여기서 $\hat{y}$는 모델의 예측값이다.

---

### 7.4 붓꽃 품종 분류에서의 확률 해석

클래스 순서를 다음과 같이 정했다고 하자.

```text
1번 클래스 = virginica
2번 클래스 = setosa
3번 클래스 = versicolor
```

그러면 소프트맥스 출력은 다음과 같이 해석된다.

$$
softmax(z) = [p_{virginica}, p_{setosa}, p_{versicolor}]
$$

각 원소의 의미는 다음과 같다.

```text
p_virginica:
 입력 데이터가 virginica일 확률

p_setosa:
 입력 데이터가 setosa일 확률

p_versicolor:
 입력 데이터가 versicolor일 확률
```

예를 들어 출력값이 다음과 같다고 하자.

```text
[0.26, 0.71, 0.03]
```

위 클래스 순서에 따라 해석하면 다음과 같다.

```text
virginica일 확률 = 0.26
setosa일 확률 = 0.71
versicolor일 확률 = 0.03
```

가장 큰 값은 `0.71`이므로 모델은 해당 입력을 `setosa`로 예측한다.

---

## 8. 그림을 통한 소프트맥스 회귀 이해

### 8.1 입력 데이터의 형태

붓꽃 품종 분류 문제에서 하나의 샘플 데이터는 4개의 특성을 가진다.

```text
x1 = SepalLengthCm
x2 = SepalWidthCm
x3 = PetalLengthCm
x4 = PetalWidthCm
```

따라서 하나의 입력 데이터는 다음과 같은 4차원 벡터로 볼 수 있다.

$$
X = [x_1, x_2, x_3, x_4]
$$

---

### 8.2 소프트맥스 함수의 입력은 클래스 개수와 같은 차원이어야 한다

소프트맥스 함수는 각 클래스에 대한 확률을 출력해야 한다.

따라서 클래스가 3개라면 소프트맥스 함수의 출력도 3차원 벡터가 된다.

```text
출력 = [p1, p2, p3]
```

이때 소프트맥스 함수에 들어가기 전의 점수 벡터도 클래스 개수와 같은 3차원 벡터가 되어야 한다.

```text
z = [z1, z2, z3]
```

즉, 입력 데이터는 4차원이지만 소프트맥스에 들어가는 값은 3차원이어야 한다.

```text
입력 데이터 X: 4차원
소프트맥스 입력 z: 3차원
소프트맥스 출력: 3차원
```

<img width="495" height="234" alt="Image" src="https://github.com/user-attachments/assets/4184bc60-01ee-48e7-87be-61fae0434fd5" />

이 변환은 가중치 행렬 $W$와 편향 $B$를 이용해 수행한다.

---

### 8.3 4차원 입력을 3차원 점수로 변환하기

하나의 입력 데이터가 4개의 특성을 가지고 있고, 분류할 클래스가 3개라면 각 클래스마다 하나의 점수를 만들어야 한다.

```text
입력 특성 개수 = 4
클래스 개수 = 3
필요한 출력 점수 개수 = 3
```

따라서 모델은 다음과 같은 점수들을 만든다.

```text
z1 = Class 1에 대한 점수
z2 = Class 2에 대한 점수
z3 = Class 3에 대한 점수
```

이 점수들은 가중치 곱과 편향을 통해 계산된다.

```text
X
 ↓
WX + B
 ↓
z
 ↓
softmax(z)
 ↓
[p1, p2, p3]
```

---

### 8.4 가중치의 개수

입력 특성이 4개이고 클래스가 3개라면 가중치는 총 `4 × 3 = 12개`가 필요하다.

각 입력 특성이 각 클래스의 점수를 계산하는 데 영향을 주기 때문이다.

```text
x1 → z1, z2, z3에 영향
x2 → z1, z2, z3에 영향
x3 → z1, z2, z3에 영향
x4 → z1, z2, z3에 영향
```

따라서 총 연결 개수는 다음과 같다.

```text
4개의 입력 특성 × 3개의 클래스 점수 = 12개의 가중치
```

<img width="294" height="138" alt="Image" src="https://github.com/user-attachments/assets/6cd0dce6-360c-45a4-8606-e670836cdeb6" />

이 가중치들은 처음부터 정답을 알고 있는 값이 아니다.

학습 과정에서 오차가 줄어드는 방향으로 점점 업데이트된다.

---

## 9. 예측값과 실제값 비교

### 9.1 소프트맥스 출력은 예측값이다

소프트맥스 회귀에서 모델의 출력은 각 클래스에 대한 확률 벡터이다.

예를 들어 다음과 같은 출력이 있다고 하자.

```text
예측값 = [0.26, 0.71, 0.03]
```

클래스 순서를 다음과 같이 정했다면,

```text
1번 클래스 = virginica
2번 클래스 = setosa
3번 클래스 = versicolor
```

예측값은 다음과 같이 해석된다.

```text
virginica일 확률 = 0.26
setosa일 확률 = 0.71
versicolor일 확률 = 0.03
```

따라서 모델의 최종 예측은 다음과 같다.

```text
setosa
```

---

### 9.2 실제값은 원-핫 벡터로 표현한다

실제 정답이 `setosa`라면, 클래스 순서에 따라 실제값은 다음과 같이 표현된다.

```text
setosa = [0, 1, 0]
```

따라서 예측값과 실제값은 다음과 같이 비교할 수 있다.

```text
예측값 = [0.26, 0.71, 0.03]
실제값 = [0,    1,    0]
```

모델은 예측값이 실제값에 가까워지도록 학습한다.

가장 이상적인 예측은 다음과 같다.

```text
예측값 = [0, 1, 0]
실제값 = [0, 1, 0]
```

이 경우 모델은 setosa를 확실하게 맞힌 것이다.

---

### 9.3 오차를 이용한 학습

모델의 예측값과 실제값이 다르면 오차가 발생한다.

```text
예측값 = [0.26, 0.71, 0.03]
실제값 = [0,    1,    0]
```

이 오차를 이용해서 가중치 $W$와 편향 $B$를 업데이트한다.

<img width="558" height="191" alt="Image" src="https://github.com/user-attachments/assets/fdfdca7c-a4e6-45d9-8a8a-5aa72b7177ce" />

흐름은 다음과 같다.

```text
1. 입력 X를 모델에 넣는다.
2. WX + B를 계산해서 z를 만든다.
3. softmax(z)를 계산해서 예측 확률을 얻는다.
4. 예측값과 실제값을 비교해서 오차를 구한다.
5. 오차가 줄어드는 방향으로 W와 B를 업데이트한다.
```

소프트맥스 회귀에서는 보통 `크로스 엔트로피(Cross Entropy)` 비용 함수를 사용해서 오차를 계산한다.

---

## 10. 벡터와 행렬로 보는 소프트맥스 회귀

### 10.1 하나의 샘플을 열벡터로 볼 때

하나의 입력 데이터가 4개의 특성을 가진다고 하자.

이를 열벡터로 표현하면 다음과 같다.

$$
X = \begin{bmatrix}
x_1 \\
x_2 \\
x_3 \\
x_4
\end{bmatrix}
$$

입력 특성의 개수를 $f$, 클래스의 개수를 $c$라고 하면 다음과 같이 둘 수 있다.

```text
f = feature의 수
c = class의 수
```

붓꽃 품종 분류 문제에서는 다음과 같다.

```text
f = 4
c = 3
```

---

### 10.2 행렬 크기

하나의 샘플을 열벡터로 본다면 행렬 크기는 다음과 같이 맞출 수 있다.

```text
W: c × f
X: f × 1
B: c × 1
WX + B: c × 1
```

수식으로 쓰면 다음과 같다.

$$
softmax(WX + B) = \hat{y}
$$

행렬 크기를 함께 쓰면 다음과 같다.

```text
W        ×  X       +  B       =  예측값
(c × f)     (f × 1)    (c × 1)    (c × 1)
```

붓꽃 품종 분류 문제에 대입하면 다음과 같다.

```text
W        ×  X       +  B       =  예측값
(3 × 4)     (4 × 1)    (3 × 1)    (3 × 1)
```

<img width="485" height="163" alt="Image" src="https://github.com/user-attachments/assets/50c97662-4e01-42fb-ac19-599d1c2b6905" />

즉, 4개의 입력 특성을 받아 3개의 클래스 확률을 출력하는 구조이다.

---

### 10.3 각 점수의 계산

클래스가 3개이고 입력 특성이 4개라면, 소프트맥스에 들어가기 전 점수는 다음과 같이 계산된다.

$$
z = WX + B
$$

이를 풀어서 보면 다음과 같다.

$$
z_1 = w_{11}x_1 + w_{12}x_2 + w_{13}x_3 + w_{14}x_4 + b_1
$$

$$
z_2 = w_{21}x_1 + w_{22}x_2 + w_{23}x_3 + w_{24}x_4 + b_2
$$

$$
z_3 = w_{31}x_1 + w_{32}x_2 + w_{33}x_3 + w_{34}x_4 + b_3
$$

여기서 각 점수는 다음 의미를 가진다.

```text
z1 = 1번 클래스에 대한 점수
z2 = 2번 클래스에 대한 점수
z3 = 3번 클래스에 대한 점수
```

그 다음 이 점수 벡터를 소프트맥스 함수에 넣는다.

$$
\hat{y} = softmax(z)
$$

---

### 10.4 PyTorch에서의 행렬 방향 주의

위 설명은 하나의 샘플을 열벡터 $X$로 두는 방식이다.

하지만 PyTorch에서는 보통 여러 개의 데이터를 한 번에 다루기 때문에 입력을 다음과 같은 행렬로 둔다.

```text
X: 데이터 개수 × 특성 개수
```

예를 들어 데이터가 5개이고 특성이 4개라면 다음과 같다.

```text
X: 5 × 4
```

PyTorch의 `nn.Linear(4, 3)`은 다음 의미이다.

```text
입력 특성 4개를 받아
클래스 점수 3개를 출력한다.
```

따라서 개념적으로는 동일하지만, 수식에서 행렬을 열벡터로 볼지 행벡터로 볼지에 따라 $W$의 모양이 다르게 보일 수 있다.

중요한 것은 다음이다.

```text
입력 특성 개수 4개 → 클래스 점수 3개로 변환한다.
```

---

## 11. 소프트맥스 회귀와 로지스틱 회귀 비교

| 구분 | 로지스틱 회귀 | 소프트맥스 회귀 |
|---|---|---|
| 사용하는 문제 | 이진 분류 | 다중 클래스 분류 |
| 클래스 개수 | 2개 | 3개 이상 |
| 사용하는 함수 | sigmoid | softmax |
| 출력 형태 | 하나의 확률값 | 클래스 개수만큼의 확률 벡터 |
| 출력 예시 | 0.75 | [0.15, 0.75, 0.10] |
| 분류 방식 | 0.5 기준 | 가장 큰 확률값 선택 |
| 가설식 | $H(X)=sigmoid(WX+B)$ | $H(X)=softmax(WX+B)$ |

---

## 12. 붓꽃 품종 분류하기를 행렬 연산으로 이해하기

### 12.1 입력 데이터 X의 크기

붓꽃 품종 분류 문제에서는 하나의 샘플이 다음 4개의 특성을 가진다.

```text
1. SepalLengthCm
2. SepalWidthCm
3. PetalLengthCm
4. PetalWidthCm
```

예시 데이터가 총 5개라면 입력 데이터는 `5 × 4` 크기의 행렬로 표현할 수 있다.

$$
X =
\begin{pmatrix}
5.1 & 3.5 & 1.4 & 0.2 \\
4.9 & 3.0 & 1.4 & 0.2 \\
5.8 & 2.6 & 4.0 & 1.2 \\
6.7 & 3.0 & 5.2 & 2.3 \\
5.6 & 2.8 & 4.9 & 2.0
\end{pmatrix}
$$

이를 각 원소의 위치를 나타내는 변수로 표현하면 다음과 같다.

$$
X =
\begin{pmatrix}
x_{11} & x_{12} & x_{13} & x_{14} \\
x_{21} & x_{22} & x_{23} & x_{24} \\
x_{31} & x_{32} & x_{33} & x_{34} \\
x_{41} & x_{42} & x_{43} & x_{44} \\
x_{51} & x_{52} & x_{53} & x_{54}
\end{pmatrix}
$$

여기서 행(row)은 샘플 하나를 의미하고, 열(column)은 특성 하나를 의미한다.

```text
X의 크기 = 5 × 4

5: 샘플의 개수
4: 각 샘플이 가진 특성의 개수
```

---

### 12.2 예측값 행렬 \hat{Y}의 크기

붓꽃 품종 분류 문제의 정답 클래스는 3개이다.

```text
1. virginica
2. setosa
3. versicolor
```

소프트맥스 회귀에서는 각 샘플마다 3개의 클래스에 대한 확률을 출력해야 한다.

예를 들어 하나의 샘플에 대한 출력은 다음과 같은 형태가 된다.

```text
[virginica일 확률, setosa일 확률, versicolor일 확률]
```

따라서 샘플이 5개이고 클래스가 3개라면 예측값 행렬 $\hat{Y}$의 크기는 `5 × 3`이 된다.

$$
\hat{Y} =
\begin{pmatrix}
y_{11} & y_{12} & y_{13} \\
y_{21} & y_{22} & y_{23} \\
y_{31} & y_{32} & y_{33} \\
y_{41} & y_{42} & y_{43} \\
y_{51} & y_{52} & y_{53}
\end{pmatrix}
$$

여기서 각 행은 하나의 샘플에 대한 예측 결과이다.

```text
\hat{Y}의 크기 = 5 × 3

5: 샘플의 개수
3: 클래스의 개수
```

예를 들어 첫 번째 행이 다음과 같다고 하자.

```text
[0.26, 0.71, 0.03]
```

그러면 이 값은 첫 번째 샘플에 대해 다음과 같이 해석할 수 있다.

```text
virginica일 확률   = 0.26
setosa일 확률      = 0.71
versicolor일 확률  = 0.03
```

---

### 12.3 가중치 행렬 W의 크기

소프트맥스 회귀에서 예측값은 입력 행렬 $X$와 가중치 행렬 $W$의 곱으로 만들어진다.

입력 행렬 $X$의 크기는 다음과 같다.

```text
X = 5 × 4
```

예측값 행렬 $\hat{Y}$의 크기는 다음과 같아야 한다.

```text
\hat{Y} = 5 × 3
```

행렬 곱셈의 크기를 맞추기 위해서는 $W$의 크기가 `4 × 3`이어야 한다.

```text
XW = (5 × 4)(4 × 3) = 5 × 3
```

따라서 가중치 행렬 $W$는 다음과 같이 표현할 수 있다.

$$
W =
\begin{pmatrix}
w_{11} & w_{12} & w_{13} \\
w_{21} & w_{22} & w_{23} \\
w_{31} & w_{32} & w_{33} \\
w_{41} & w_{42} & w_{43}
\end{pmatrix}
$$

여기서 $W$의 행 개수는 입력 특성의 개수와 같고, 열 개수는 클래스의 개수와 같다.

```text
W의 크기 = 4 × 3

4: 입력 특성의 개수
3: 클래스의 개수
```

---

### 12.4 편향 B의 크기

편향 $B$는 각 클래스에 더해지는 값이다.

클래스가 3개이므로 기본적으로 편향은 다음과 같은 크기를 가진다.

```text
B = 1 × 3
```

PyTorch에서는 보통 다음과 같은 형태로 선언한다.

```python
b = torch.zeros((1, 3), requires_grad=True)
```

다만 행렬식으로 전체 샘플에 더해지는 모습을 표현할 때는 같은 편향 벡터가 각 행마다 반복되는 것처럼 생각할 수 있다.

$$
B =
\begin{pmatrix}
b_1 & b_2 & b_3 \\
b_1 & b_2 & b_3 \\
b_1 & b_2 & b_3 \\
b_1 & b_2 & b_3 \\
b_1 & b_2 & b_3
\end{pmatrix}
$$

이처럼 보면 $B$도 예측값 행렬과 같은 `5 × 3` 크기처럼 표현할 수 있다.

하지만 실제 구현에서는 `1 × 3` 편향 벡터가 자동으로 각 샘플에 더해지는 브로드캐스팅(broadcasting)이 일어난다고 이해하면 된다.

```text
실제 선언되는 b의 크기 = 1 × 3
계산할 때 더해지는 형태 = 5 × 3처럼 자동 확장
```

---

### 12.5 소프트맥스 회귀의 행렬식

결과적으로 소프트맥스 회귀의 가설식은 다음과 같다.

$$
\hat{Y} = softmax(XW + B)
$$

전체 행렬의 크기만 보면 다음과 같다.

```text
XW + B
= (5 × 4)(4 × 3) + (1 × 3)
= (5 × 3) + (1 × 3)
= 5 × 3
```

여기서 $B$는 브로드캐스팅되어 각 샘플의 결과에 더해진다.

이를 행렬 형태로 쓰면 다음과 같다.

$$
\begin{pmatrix}
y_{11} & y_{12} & y_{13} \\
y_{21} & y_{22} & y_{23} \\
y_{31} & y_{32} & y_{33} \\
y_{41} & y_{42} & y_{43} \\
y_{51} & y_{52} & y_{53}
\end{pmatrix}
=
softmax
\left(
\begin{pmatrix}
x_{11} & x_{12} & x_{13} & x_{14} \\
x_{21} & x_{22} & x_{23} & x_{24} \\
x_{31} & x_{32} & x_{33} & x_{34} \\
x_{41} & x_{42} & x_{43} & x_{44} \\
x_{51} & x_{52} & x_{53} & x_{54}
\end{pmatrix}
\begin{pmatrix}
w_{11} & w_{12} & w_{13} \\
w_{21} & w_{22} & w_{23} \\
w_{31} & w_{32} & w_{33} \\
w_{41} & w_{42} & w_{43}
\end{pmatrix}
+
\begin{pmatrix}
b_1 & b_2 & b_3 \\
b_1 & b_2 & b_3 \\
b_1 & b_2 & b_3 \\
b_1 & b_2 & b_3 \\
b_1 & b_2 & b_3
\end{pmatrix}
\right)
$$

소프트맥스 함수는 보통 각 행마다 적용된다.

즉, 한 샘플에 대해 나온 3개의 값이 0과 1 사이의 확률값으로 바뀌고, 그 합이 1이 된다.

```text
한 행의 출력 예시:
[0.26, 0.71, 0.03]

0.26 + 0.71 + 0.03 = 1.00
```

---

## 13. 비용 함수(Cost Function)

소프트맥스 회귀에서는 비용 함수로 크로스 엔트로피(Cross Entropy)를 사용한다.

비용 함수는 모델의 예측값과 실제 정답이 얼마나 다른지를 숫자로 나타낸 것이다.

```text
모델의 목표:
예측값과 실제값의 차이가 작아지도록 W와 B를 업데이트하는 것
```

---

### 13.1 크로스 엔트로피 함수

소프트맥스 회귀에서 하나의 샘플에 대한 크로스 엔트로피 비용 함수는 다음과 같다.

$$
cost(W) = -\sum_{j=1}^{k} y_j \log(p_j)
$$

각 기호의 의미는 다음과 같다.

```text
k: 클래스의 개수
j: 클래스 인덱스
y_j: 실제 정답 원-핫 벡터의 j번째 값
p_j: 모델이 j번째 클래스일 확률로 예측한 값
```

예를 들어 클래스가 3개이고 실제 정답이 setosa라고 하자.

클래스 순서를 다음과 같이 정했다고 가정한다.

```text
0번: virginica
1번: setosa
2번: versicolor
```

그러면 실제값은 다음과 같은 원-핫 벡터로 표현된다.

```text
y = [0, 1, 0]
```

모델의 예측값이 다음과 같다고 하자.

```text
p = [0.26, 0.71, 0.03]
```

그러면 비용은 다음과 같이 계산된다.

$$
cost(W) = -[0\log(0.26) + 1\log(0.71) + 0\log(0.03)]
$$

$$
cost(W) = -\log(0.71)
$$

원-핫 벡터에서는 실제 정답 위치만 1이고 나머지는 0이므로, 결국 정답 클래스에 대한 예측 확률만 비용 계산에 영향을 준다.

```text
정답 클래스의 예측 확률이 높을수록 cost가 작아진다.
정답 클래스의 예측 확률이 낮을수록 cost가 커진다.
```

---

### 13.2 전체 데이터에 대한 비용 함수

데이터가 $n$개 있을 때는 각 데이터의 비용을 구한 뒤 평균을 낸다.

$$
cost(W) = -\frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{k} y_j^{(i)} \log(p_j^{(i)})
$$

각 기호의 의미는 다음과 같다.

```text
n: 전체 데이터의 개수
k: 클래스의 개수
i: 데이터 인덱스
j: 클래스 인덱스
y_j^{(i)}: i번째 데이터의 실제 정답 원-핫 벡터에서 j번째 값
p_j^{(i)}: i번째 데이터가 j번째 클래스일 확률이라고 모델이 예측한 값
```

이 식은 다음 흐름으로 이해하면 된다.

```text
1. 각 샘플마다 클래스별 예측 확률을 구한다.
2. 실제 정답 원-핫 벡터와 비교한다.
3. 정답 클래스에 해당하는 확률의 -log 값을 구한다.
4. 모든 샘플의 손실값을 평균낸다.
```

---

### 13.3 이진 분류의 크로스 엔트로피와의 관계

로지스틱 회귀에서 사용한 이진 분류의 크로스 엔트로피는 다음과 같았다.

$$
cost(W) = -[y\log(H(X)) + (1-y)\log(1-H(X))]
$$

이진 분류에서는 클래스가 2개라고 볼 수 있다.

예를 들어 다음과 같이 치환할 수 있다.

```text
y_1 = y
p_1 = H(X)

y_2 = 1 - y
p_2 = 1 - H(X)
```

그러면 위 식은 다음과 같이 쓸 수 있다.

$$
-[y_1\log(p_1) + y_2\log(p_2)]
$$

이는 합 기호를 사용하면 다음과 같다.

$$
-\sum_{j=1}^{2} y_j\log(p_j)
$$

소프트맥스 회귀에서는 클래스 개수가 2개로 고정된 것이 아니라 $k$개로 확장된다.

$$
-\sum_{j=1}^{k} y_j\log(p_j)
$$

따라서 이진 분류의 크로스 엔트로피는 소프트맥스 회귀의 크로스 엔트로피에서 클래스 개수가 2개인 특수한 경우로 이해할 수 있다.

---

### 13.4 PyTorch 코드로 크로스 엔트로피 직접 구현하기

소프트맥스 회귀의 비용 함수는 다음과 같았다.

$$
cost(W) = -\frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{k} y_j^{(i)} \log(p_j^{(i)})
$$

이를 다음처럼 바꾸어 코드로 구현할 수 있다.

$$
cost(W) = \frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{k} y_j^{(i)} \times (-\log(p_j^{(i)}))
$$

PyTorch 코드로 직접 구현하면 다음과 같다.

```python
cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
print(cost)
```

각 부분의 의미는 다음과 같다.

```text
y_one_hot:
실제 정답을 원-핫 인코딩한 값

hypothesis:
softmax를 통과한 예측 확률값

-torch.log(hypothesis):
예측 확률에 -log를 취한 값

y_one_hot * -torch.log(hypothesis):
정답 클래스 위치의 손실값만 남김

.sum(dim=1):
각 샘플별로 클래스 방향의 손실값을 합함

.mean():
전체 샘플의 평균 손실값을 구함
```

출력 예시는 다음과 같다.

```text
tensor(1.4689, grad_fn=<MeanBackward1>)
```

---

## 14. PyTorch로 소프트맥스 비용 함수 구현하기

이번에는 소프트맥스 비용 함수를 PyTorch에서 로우 레벨 방식과 하이 레벨 방식으로 구현하는 과정을 정리한다.

앞으로의 실습에서는 아래 코드가 먼저 실행되었다고 가정한다.

```python
import torch
import torch.nn.functional as F

torch.manual_seed(1)
```

---

### 14.1 소프트맥스 함수 적용하기

먼저 3개의 원소를 가진 벡터 텐서를 만든다.

```python
z = torch.FloatTensor([1, 2, 3])
```

이 텐서에 소프트맥스 함수를 적용한다.

```python
hypothesis = F.softmax(z, dim=0)
print(hypothesis)
```

출력은 다음과 같다.

```text
tensor([0.0900, 0.2447, 0.6652])
```

소프트맥스 함수는 입력값들을 0과 1 사이의 값으로 바꾸고, 전체 합이 1이 되도록 만든다.

```python
hypothesis.sum()
```

출력은 다음과 같다.

```text
tensor(1.)
```

즉, 소프트맥스 출력은 확률 벡터처럼 해석할 수 있다.

---

### 14.2 3 × 5 텐서에 소프트맥스 적용하기

이번에는 임의의 `3 × 5` 크기를 가진 텐서를 만든다.

```python
z = torch.rand(3, 5, requires_grad=True)
```

여기서 의미는 다음과 같이 볼 수 있다.

```text
3: 샘플의 개수
5: 클래스의 개수
```

즉, 3개의 샘플 각각에 대해 5개의 클래스 점수(logit)가 있는 상황이다.

각 샘플마다 5개 클래스에 대한 확률을 구해야 하므로 `dim=1` 방향으로 소프트맥스를 적용한다.

```python
hypothesis = F.softmax(z, dim=1)
print(hypothesis)
```

출력 예시는 다음과 같다.

```text
tensor([[0.2645, 0.1639, 0.1855, 0.2585, 0.1277],
        [0.2430, 0.1624, 0.2322, 0.1930, 0.1694],
        [0.2226, 0.1986, 0.2326, 0.1594, 0.1868]], grad_fn=<SoftmaxBackward>)
```

각 행의 합은 1이 된다.

```text
첫 번째 행: 5개 클래스에 대한 확률
두 번째 행: 5개 클래스에 대한 확률
세 번째 행: 5개 클래스에 대한 확률
```

---

### 14.3 정답 레이블 만들기

각 샘플에 대한 정답 레이블을 임의로 만든다.

```python
y = torch.randint(5, (3,)).long()
print(y)
```

출력 예시는 다음과 같다.

```text
tensor([0, 2, 1])
```

의미는 다음과 같다.

```text
첫 번째 샘플의 정답 클래스 = 0
두 번째 샘플의 정답 클래스 = 2
세 번째 샘플의 정답 클래스 = 1
```

`F.cross_entropy()`나 `F.nll_loss()`를 사용할 때는 이런 정수 레이블을 그대로 사용한다.

---

### 14.4 원-핫 인코딩 직접 만들기

로우 레벨 방식으로 크로스 엔트로피를 직접 계산하려면 정답 레이블을 원-핫 벡터로 바꿔야 한다.

먼저 예측값과 같은 크기의 0 텐서를 만든다.

```python
y_one_hot = torch.zeros_like(hypothesis)
```

그 다음 `scatter_()`를 사용해서 정답 클래스 위치에 1을 넣는다.

```python
y_one_hot.scatter_(1, y.unsqueeze(1), 1)
print(y_one_hot)
```

출력은 다음과 같다.

```text
tensor([[1., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0.],
        [0., 1., 0., 0., 0.]])
```

---

### 14.5 y.unsqueeze(1)의 의미

처음에 `y`의 크기는 다음과 같다.

```text
y.shape = torch.Size([3])
```

값은 다음과 같았다.

```text
tensor([0, 2, 1])
```

여기에 `unsqueeze(1)`을 적용하면 두 번째 차원이 추가된다.

```python
print(y.unsqueeze(1))
```

출력은 다음과 같다.

```text
tensor([[0],
        [2],
        [1]])
```

즉, 크기가 다음처럼 바뀐다.

```text
(3,) → (3, 1)
```

`scatter_(1, y.unsqueeze(1), 1)`은 `dim=1` 방향에서 각 행의 정답 클래스 위치에 1을 넣는 연산이다.

```text
첫 번째 행: 0번 위치에 1
두 번째 행: 2번 위치에 1
세 번째 행: 1번 위치에 1
```

`scatter_`처럼 이름 뒤에 `_`가 붙은 함수는 PyTorch에서 in-place operation을 의미한다.

즉, 새로운 텐서를 만드는 것이 아니라 기존 `y_one_hot` 값을 직접 수정한다.

---

### 14.6 로우 레벨 비용 함수 구현

소프트맥스 회귀의 비용 함수는 다음과 같다.

$$
cost(W) = \frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{k} y_j^{(i)} \times (-\log(p_j^{(i)}))
$$

이를 코드로 구현하면 다음과 같다.

```python
cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
print(cost)
```

출력 예시는 다음과 같다.

```text
tensor(1.4689, grad_fn=<MeanBackward1>)
```

---

### 14.7 F.softmax() + torch.log()와 F.log_softmax()

로우 레벨 구현에서는 다음처럼 소프트맥스 결과에 로그를 적용했다.

```python
torch.log(F.softmax(z, dim=1))
```

PyTorch에서는 이 두 과정을 합친 함수가 제공된다.

```python
F.log_softmax(z, dim=1)
```

두 코드는 의미상 다음과 같다.

```text
torch.log(F.softmax(z, dim=1))
= F.log_softmax(z, dim=1)
```

코드로 비교하면 다음과 같다.

```python
# Low level
torch.log(F.softmax(z, dim=1))

# High level
F.log_softmax(z, dim=1)
```

출력 예시는 다음과 같다.

```text
tensor([[-1.3301, -1.8084, -1.6846, -1.3530, -2.0584],
        [-1.4147, -1.8174, -1.4602, -1.6450, -1.7758],
        [-1.5025, -1.6165, -1.4586, -1.8360, -1.6776]],
       grad_fn=<LogSoftmaxBackward>)
```

`F.log_softmax()`는 수치적으로 더 안정적인 방식으로 계산되므로 실제 구현에서는 직접 `softmax`와 `log`를 따로 쓰는 것보다 권장된다.

---

### 14.8 F.log_softmax() + F.nll_loss() = F.cross_entropy()

앞서 로우 레벨 비용 함수는 다음과 같았다.

```python
(y_one_hot * -torch.log(F.softmax(z, dim=1))).sum(dim=1).mean()
```

이를 `F.log_softmax()`로 바꾸면 다음과 같다.

```python
(y_one_hot * -F.log_softmax(z, dim=1)).sum(dim=1).mean()
```

출력은 동일하다.

```text
tensor(1.4689, grad_fn=<MeanBackward0>)
```

그런데 PyTorch에서는 이 과정을 더 간단하게 처리할 수 있다.

```python
F.nll_loss(F.log_softmax(z, dim=1), y)
```

출력은 다음과 같다.

```text
tensor(1.4689, grad_fn=<NllLossBackward>)
```

여기서 `nll`은 `Negative Log Likelihood`의 약자이다.

`F.nll_loss()`를 사용할 때는 원-핫 벡터를 넣지 않고, 정수 레이블 `y`를 그대로 넣는다.

그리고 이를 더 간단하게 만든 함수가 `F.cross_entropy()`이다.

```python
F.cross_entropy(z, y)
```

출력은 다음과 같다.

```text
tensor(1.4689, grad_fn=<NllLossBackward>)
```

정리하면 다음 관계가 성립한다.

```text
F.cross_entropy(z, y)
= F.nll_loss(F.log_softmax(z, dim=1), y)
```

중요한 점은 `F.cross_entropy()` 안에 `softmax` 과정이 포함되어 있다는 것이다.

따라서 `F.cross_entropy()`에 넣는 입력값은 softmax를 통과한 확률값이 아니라, softmax를 적용하기 전의 점수값인 `z`, 즉 logit이어야 한다.

```text
올바른 방식:
F.cross_entropy(z, y)

피해야 할 방식:
F.cross_entropy(F.softmax(z, dim=1), y)
```

---

### 14.9 nn.CrossEntropyLoss() 사용하기

`F.cross_entropy()`는 함수 방식이다.

```python
F.cross_entropy(z, y)
```

같은 기능을 클래스 방식으로 사용할 수도 있다.

```python
nn.CrossEntropyLoss()(z, y)
```

출력 예시는 다음과 같다.

```text
tensor(1.4689, grad_fn=<NllLossBackward0>)
```

실제로는 보통 다음처럼 손실 함수 객체를 먼저 만든 뒤 사용한다.

```python
criterion = nn.CrossEntropyLoss()

loss = criterion(z, y)
print(loss)
```

함수 방식과 클래스 방식의 차이는 다음과 같다.

| 구분 | F.cross_entropy() | nn.CrossEntropyLoss() |
|---|---|---|
| 형태 | 함수 | 클래스 |
| 사용 방식 | 호출할 때마다 직접 사용 | 객체를 만들어두고 반복 사용 |
| 예시 | `F.cross_entropy(z, y)` | `criterion(z, y)` |
| 장점 | 간단한 한 번 계산에 편함 | 같은 설정을 반복해서 쓸 때 편함 |

예를 들어 손실값의 평균이 아니라 합계를 구하고 싶다면 클래스 방식에서는 객체를 만들 때 설정을 저장해둘 수 있다.

```python
criterion = nn.CrossEntropyLoss(reduction='sum')

loss1 = criterion(z, y)
loss2 = criterion(z, y)
```

함수 방식에서는 매번 설정을 전달해야 한다.

```python
loss_sum1 = F.cross_entropy(z, y, reduction='sum')
loss_sum2 = F.cross_entropy(z2, y2, reduction='sum')
```

`nn.CrossEntropyLoss()`도 `F.cross_entropy()`와 마찬가지로 내부에 `log_softmax`와 `nll_loss`를 포함한다.

따라서 입력으로는 softmax를 통과한 확률값이 아니라 logit을 넣어야 한다.

---

## 15. MNIST 데이터 이해하기

이번에는 소프트맥스 회귀를 MNIST 데이터에 적용하는 예제를 생각해본다.

MNIST 데이터는 손글씨 숫자 이미지로 구성된 대표적인 머신러닝 데이터셋이다.

```text
입력: 손글씨 숫자 이미지
출력: 0부터 9까지의 숫자 레이블
```

MNIST는 과거 우체국에서 편지의 우편 번호를 인식하기 위한 목적으로 만들어진 데이터셋이며, 머신러닝을 처음 배울 때 자주 사용하는 기본 예제이다. 사람에게는 손글씨 숫자를 알아보는 일이 비교적 간단하지만, 모델에게는 픽셀값만 보고 숫자를 분류해야 하는 문제이다.

MNIST 데이터는 아래 링크에 공개되어 있다.

```text
http://yann.lecun.com/exdb/mnist
```

---

### 15.1 MNIST 데이터의 구성

MNIST는 숫자 0부터 9까지의 이미지로 구성되어 있다.

```text
클래스 개수 = 10

0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

일반적으로 MNIST 데이터는 다음과 같이 구성되어 있다.

```text
훈련 데이터: 60,000개
테스트 데이터: 10,000개
이미지 크기: 28 × 28
클래스 개수: 10개
```

<img width="291" height="298" alt="Image" src="https://github.com/user-attachments/assets/e3ed5450-e1c5-43b4-b3f8-2102b459e121" />

모델은 손글씨 숫자 이미지를 입력받고, 그 이미지가 어떤 숫자인지 예측해야 한다.

예를 들어 숫자 5의 이미지가 입력으로 들어오면 모델은 다음과 같이 판단해야 한다.

```text
이 이미지는 숫자 5일 확률이 가장 높다.
```

---

### 15.2 이미지를 벡터로 바꾸기

MNIST의 각 이미지는 `28 × 28` 픽셀이다.

```text
이미지 크기 = 28 × 28
```

이를 소프트맥스 회귀에 넣기 위해서는 2차원 이미지를 1차원 벡터로 펼쳐야 한다.

```text
28 × 28 = 784
```

따라서 하나의 이미지는 784개의 원소를 가진 벡터가 된다.

```text
하나의 MNIST 이미지
   ↓
28 × 28 픽셀
   ↓
784차원 벡터
```

<img width="461" height="448" alt="Image" src="https://github.com/user-attachments/assets/79c6838a-5ea4-4223-adb9-8c9ca162e990" />

즉, MNIST 문제에서 입력 특성의 개수는 784개이다.

```text
입력 차원 = 784
출력 클래스 개수 = 10
```

---

### 15.3 MNIST를 소프트맥스 회귀로 분류하기

MNIST를 소프트맥스 회귀로 분류한다면 모델의 구조는 다음과 같다.

```text
입력 이미지 28 × 28
   ↓
784차원 벡터로 펼침
   ↓
Linear(784, 10)
   ↓
10개 숫자 클래스에 대한 점수 출력
   ↓
CrossEntropyLoss로 학습
```

수식으로는 다음과 같이 볼 수 있다.

$$
\hat{Y} = softmax(XW + B)
$$

여기서 각 크기는 다음과 같다.

```text
X: n × 784
W: 784 × 10
B: 1 × 10
\hat{Y}: n × 10
```

여기서 $n$은 배치 크기(batch size)이다.

예를 들어 배치 크기가 100이라면 다음과 같다.

```text
X: 100 × 784
W: 784 × 10
B: 1 × 10
출력: 100 × 10
```

---

## 16. 소프트맥스 회귀 구현하기

이제 PyTorch로 소프트맥스 회귀를 구현하는 여러 방법을 정리한다.

앞으로의 실습에서는 아래 코드가 먼저 실행되었다고 가정한다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
```

---

### 16.1 데이터셋 준비

훈련 데이터와 레이블을 텐서로 선언한다.

```python
x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]

y_train = [2, 2, 2, 1, 1, 1, 0, 0]

x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)
```

입력 데이터의 의미는 다음과 같다.

```text
샘플 개수 = 8
특성 개수 = 4
클래스 개수 = 3
```

`x_train`의 각 행은 하나의 샘플이고, 각 샘플은 4개의 특성을 가진다.

`y_train`은 각 샘플의 정답 레이블이다.

```text
y_train = [2, 2, 2, 1, 1, 1, 0, 0]
```

레이블 값이 0, 1, 2 중 하나이므로 클래스 개수는 3개이다.

---

### 16.2 데이터 크기 확인하기

```python
print(x_train.shape)
print(y_train.shape)
```

출력은 다음과 같다.

```text
torch.Size([8, 4])
torch.Size([8])
```

의미는 다음과 같다.

```text
x_train: 8개의 샘플, 각 샘플은 4개의 특성
          shape = [8, 4]

y_train: 8개의 샘플에 대한 정답 레이블
          shape = [8]
```

`y_train`은 `[8, 1]`이 아니라 `[8]` 형태이다.

PyTorch의 `F.cross_entropy()`는 정답을 원-핫 벡터가 아니라 정수 클래스 레이블로 받기 때문에 이런 형태가 자연스럽다.

---

### 16.3 소프트맥스 회귀 구현하기: 로우 레벨 방식

로우 레벨 방식에서는 원-핫 인코딩을 직접 만들고, 비용 함수도 직접 계산한다.

먼저 정답 레이블을 원-핫 인코딩한다.

```python
y_one_hot = torch.zeros(8, 3)
y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)
print(y_one_hot.shape)
```

출력은 다음과 같다.

```text
torch.Size([8, 3])
```

의미는 다음과 같다.

```text
8: 샘플의 개수
3: 클래스의 개수
```

이제 가중치 $W$와 편향 $b$를 선언한다.

```python
W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros((1, 3), requires_grad=True)
```

각 크기의 의미는 다음과 같다.

```text
x_train: 8 × 4
W:       4 × 3
b:       1 × 3
출력:    8 × 3
```

옵티마이저는 SGD를 사용한다.

```python
optimizer = optim.SGD([W, b], lr=0.1)
```

전체 학습 코드는 다음과 같다.

```python
nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    hypothesis = F.softmax(x_train.matmul(W) + b, dim=1)

    cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
```

코드 흐름은 다음과 같다.

```text
1. x_train.matmul(W) + b
   각 샘플에 대해 3개 클래스 점수를 계산한다.

2. F.softmax(..., dim=1)
   각 샘플의 클래스 점수를 확률로 바꾼다.

3. cost = ...
   원-핫 정답과 예측 확률을 이용해 크로스 엔트로피를 직접 계산한다.

4. optimizer.zero_grad()
   이전 기울기를 초기화한다.

5. cost.backward()
   역전파로 W와 b의 기울기를 계산한다.

6. optimizer.step()
   W와 b를 업데이트한다.
```

---

### 16.4 소프트맥스 회귀 구현하기: 하이 레벨 방식

하이 레벨 방식에서는 `F.cross_entropy()`를 사용한다.

`F.cross_entropy()`는 내부적으로 다음 두 과정을 포함한다.

```text
1. F.log_softmax()
2. F.nll_loss()
```

따라서 모델의 출력값에 softmax를 직접 적용하지 않는다.

먼저 W와 b를 다시 초기화한다.

```python
W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros((1, 3), requires_grad=True)

optimizer = optim.SGD([W, b], lr=0.1)
```

전체 학습 코드는 다음과 같다.

```python
nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    z = x_train.matmul(W) + b
    cost = F.cross_entropy(z, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
```

로우 레벨 방식과 비교하면 다음과 같다.

| 구분 | 로우 레벨 방식 | 하이 레벨 방식 |
|---|---|---|
| 예측값 | `F.softmax(z, dim=1)` | `z` 그대로 사용 |
| 정답 | 원-핫 벡터 | 정수 레이블 |
| 비용 함수 | 직접 구현 | `F.cross_entropy(z, y_train)` |
| 코드 길이 | 길다 | 짧다 |

하이 레벨 방식에서 가장 중요한 점은 다음과 같다.

```text
F.cross_entropy()를 사용할 때는 softmax를 미리 적용하지 않는다.
```

---

### 16.5 nn.Linear로 소프트맥스 회귀 구현하기

이번에는 `W`와 `b`를 직접 선언하지 않고 `nn.Linear()`를 사용한다.

```python
model = nn.Linear(4, 3)
```

이 코드는 다음 연산을 수행하는 계층을 만든다.

$$
XW + B
$$

각 숫자의 의미는 다음과 같다.

```text
nn.Linear(4, 3)

4: 입력 특성 개수
3: 출력 클래스 개수
```

`F.cross_entropy()`를 사용할 것이므로 모델 뒤에 softmax를 붙이지 않는다.

```python
optimizer = optim.SGD(model.parameters(), lr=0.1)
```

전체 학습 코드는 다음과 같다.

```python
nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    prediction = model(x_train)

    cost = F.cross_entropy(prediction, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
```

여기서 `prediction`은 확률값이 아니라 softmax 적용 전 점수값이다.

```text
prediction = model(x_train)
           = x_train에 nn.Linear(4, 3)을 적용한 결과
           = softmax 이전의 logit
```

---

### 16.6 클래스로 소프트맥스 회귀 구현하기

실제 PyTorch 모델은 보통 `nn.Module`을 상속받은 클래스로 정의한다.

```python
class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x):
        return self.linear(x)
```

각 부분의 의미는 다음과 같다.

```text
class SoftmaxClassifierModel(nn.Module):
PyTorch 모델 클래스를 정의한다.

super().__init__():
nn.Module의 초기화 기능을 불러온다.

self.linear = nn.Linear(4, 3):
입력 특성 4개를 받아 클래스 3개에 대한 점수를 출력하는 선형 계층이다.

forward(self, x):
입력 x가 모델 안에서 어떻게 계산되는지 정의한다.
```

모델 객체를 생성한다.

```python
model = SoftmaxClassifierModel()
```

옵티마이저를 설정한다.

```python
optimizer = optim.SGD(model.parameters(), lr=0.1)
```

전체 학습 코드는 다음과 같다.

```python
nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    prediction = model(x_train)

    cost = F.cross_entropy(prediction, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
```

클래스 방식에서도 `forward()` 안에 softmax를 넣지 않는다.

그 이유는 `F.cross_entropy()`가 내부적으로 softmax 역할까지 포함하기 때문이다.

```text
model(x_train)
   ↓
Linear 결과, 즉 logit 출력
   ↓
F.cross_entropy(prediction, y_train)
   ↓
내부에서 log_softmax + nll_loss 수행
```


---

## 17. MNIST 이미지를 784차원 벡터로 만들기

### 17.1 MNIST 이미지 예시

MNIST 데이터는 손글씨 숫자 이미지이다. 하나의 이미지는 `28 × 28` 픽셀로 구성되어 있다.

예를 들어 아래와 같은 숫자 5 이미지가 있다고 하자.

<img width="244" height="246" alt="Image" src="https://github.com/user-attachments/assets/9f74d8d1-c53d-4cad-95b5-53ddd34cbc7f" />

MNIST 이미지는 사람에게는 숫자 그림처럼 보이지만, 모델에게는 픽셀값들의 묶음으로 입력된다.

```text
사람이 보는 형태:
손글씨 숫자 이미지

모델이 받는 형태:
픽셀값으로 이루어진 텐서
```

---

### 17.2 28 × 28 이미지를 784차원 벡터로 펼치기

MNIST 이미지 하나의 크기는 다음과 같다.

```text
28 × 28
```

이를 계산하면 다음과 같다.

$$
28 \times 28 = 784
$$

따라서 하나의 이미지는 784개의 픽셀값을 가진다.

소프트맥스 회귀에 넣기 위해서는 2차원 이미지를 1차원 벡터처럼 펼쳐야 한다.

```text
28 × 28 이미지
   ↓
784차원 벡터
```

즉, MNIST 분류 문제에서는 입력 특성의 개수가 784개가 된다.

```text
입력 차원 = 784
출력 클래스 개수 = 10
```

---

### 17.3 view()를 이용한 형태 변환

MNIST 데이터를 `DataLoader`에서 꺼내면 입력 이미지 `X`는 보통 다음과 같은 크기를 가진다.

```text
X.shape = [batch_size, 1, 28, 28]
```

각 의미는 다음과 같다.

```text
batch_size: 한 번에 학습할 이미지 개수
1: 채널 수
28: 이미지의 세로 픽셀 수
28: 이미지의 가로 픽셀 수
```

이 이미지를 소프트맥스 회귀 모델에 넣기 위해 다음과 같이 펼친다.

```python
for X, Y in data_loader:
    X = X.view(-1, 28 * 28)
```

위 코드의 의미는 다음과 같다.

```text
변환 전:
[batch_size, 1, 28, 28]

변환 후:
[batch_size, 784]
```

여기서 `-1`은 PyTorch가 앞쪽 차원을 자동으로 계산하라는 뜻이다.

예를 들어 배치 크기가 100이면 다음과 같이 변환된다.

```text
변환 전: [100, 1, 28, 28]
변환 후: [100, 784]
```

즉, 한 번에 100개의 이미지를 가져오고, 각 이미지를 784차원 벡터로 펼치는 것이다.

---

## 18. 토치비전(torchvision)

### 18.1 torchvision이란?

`torchvision`은 PyTorch에서 이미지 데이터를 다룰 때 자주 사용하는 패키지이다.

torchvision에는 다음과 같은 기능들이 포함되어 있다.

```text
1. 유명한 이미지 데이터셋
2. 이미 구현된 유명한 모델
3. 이미지 전처리 도구
```

공식 문서 링크는 다음과 같다.

```text
https://pytorch.org/docs/stable/torchvision/index.html
```

MNIST 데이터셋도 torchvision을 통해 쉽게 불러올 수 있다.

```python
import torchvision.datasets as dsets
import torchvision.transforms as transforms
```

여기서 각 모듈의 의미는 다음과 같다.

```text
torchvision.datasets:
MNIST 같은 이미지 데이터셋을 불러올 때 사용한다.

torchvision.transforms:
이미지를 텐서로 변환하거나 정규화하는 전처리에 사용한다.
```

---

### 18.2 torchtext와의 비교

이미지 처리를 위해 `torchvision`을 사용하듯이, 자연어 처리를 위해서는 `torchtext`라는 패키지를 사용할 수 있다.

```text
이미지 처리 관련 도구 → torchvision
자연어 처리 관련 도구 → torchtext
```

---

## 19. MNIST 분류기 구현을 위한 사전 설정

### 19.1 필요한 라이브러리 import

MNIST 분류기를 구현하기 위해 필요한 도구들을 먼저 import한다.

```python
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import random
```

각 코드의 의미는 다음과 같다.

```text
1. torch
   PyTorch의 기본 기능을 사용한다.

2. torchvision.datasets
   MNIST 같은 이미지 데이터셋을 불러온다.

3. torchvision.transforms
   이미지 데이터를 텐서로 변환하는 전처리를 수행한다.

4. DataLoader
   데이터를 미니 배치 단위로 불러온다.

5. torch.nn
   Linear, CrossEntropyLoss 같은 신경망 구성 요소를 사용한다.

6. matplotlib.pyplot
   이미지를 시각화할 때 사용한다.

7. random
   테스트 데이터 중 임의의 샘플을 뽑을 때 사용한다.
```

---

### 19.2 CPU 또는 GPU 설정

현재 환경에서 GPU 연산이 가능하면 GPU를 사용하고, 그렇지 않으면 CPU를 사용하도록 설정한다.

```python
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print("다음 기기로 학습합니다:", device)
```

각 코드의 의미는 다음과 같다.

```text
torch.cuda.is_available():
현재 환경에서 CUDA GPU를 사용할 수 있으면 True를 반환한다.

torch.device("cuda" if USE_CUDA else "cpu"):
GPU를 사용할 수 있으면 cuda, 아니면 cpu를 연산 장치로 선택한다.
```

출력 예시는 다음과 같다.

```text
다음 기기로 학습합니다: cuda
```

또는 다음과 같이 나올 수 있다.

```text
다음 기기로 학습합니다: cpu
```

Colab에서는 다음 경로에서 GPU를 사용할 수 있다.

```text
런타임
   ↓
런타임 유형 변경
   ↓
하드웨어 가속기
   ↓
GPU 선택
```

---

### 19.3 랜덤 시드 고정

실행할 때마다 비슷한 결과가 나오도록 랜덤 시드를 고정한다.

```python
random.seed(777)
torch.manual_seed(777)
if USE_CUDA:
    torch.cuda.manual_seed_all(777)
```

각 코드의 의미는 다음과 같다.

```text
random.seed(777):
파이썬 random 모듈의 랜덤성을 고정한다.

torch.manual_seed(777):
PyTorch의 랜덤성을 고정한다.

torch.cuda.manual_seed_all(777):
GPU 연산에서 사용되는 랜덤성도 고정한다.
```

여기서 `if USE_CUDA:`를 사용하는 이유는 GPU를 사용할 수 있을 때만 CUDA용 랜덤 시드를 고정하기 위해서이다.

---

### 19.4 하이퍼파라미터 설정

학습에 사용할 하이퍼파라미터를 변수로 둔다.

```python
training_epochs = 15
batch_size = 100
```

각 의미는 다음과 같다.

```text
training_epochs = 15:
전체 훈련 데이터를 15번 반복해서 학습한다.

batch_size = 100:
한 번의 학습 단계에서 100개의 이미지를 사용한다.
```

하이퍼파라미터는 모델이 학습으로 찾는 값이 아니라, 사람이 직접 정해주는 설정값이다.

---

## 20. MNIST 분류기 구현하기

### 20.1 MNIST 데이터셋 불러오기

`torchvision.datasets`의 `MNIST`를 사용하면 MNIST 데이터셋을 쉽게 불러올 수 있다.

```python
mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)
```

각 인자의 의미는 다음과 같다.

```text
root:
MNIST 데이터를 저장할 경로이다.

train=True:
훈련 데이터를 불러온다.

train=False:
테스트 데이터를 불러온다.

transform=transforms.ToTensor():
이미지 데이터를 PyTorch 텐서로 변환한다.

download=True:
지정한 경로에 데이터가 없으면 다운로드한다.
```

훈련 데이터와 테스트 데이터의 역할은 다음과 같다.

```text
훈련 데이터:
모델의 가중치와 편향을 학습하는 데 사용한다.

테스트 데이터:
학습이 끝난 모델이 새로운 데이터도 잘 맞히는지 평가하는 데 사용한다.
```

---

### 20.2 DataLoader 만들기

훈련 데이터를 미니 배치 단위로 불러오기 위해 `DataLoader`를 사용한다.

```python
data_loader = DataLoader(dataset=mnist_train,
                         batch_size=batch_size,
                         shuffle=True,
                         drop_last=True)
```

각 인자의 의미는 다음과 같다.

```text
dataset:
불러올 데이터셋이다.

batch_size:
한 번에 몇 개의 데이터를 가져올지 정한다.

shuffle=True:
매 에포크마다 데이터를 섞어서 불러온다.

drop_last=True:
마지막 배치의 크기가 batch_size보다 작으면 버린다.
```

예를 들어 데이터가 1,000개이고 배치 크기가 128이라고 하자.

```text
1,000 ÷ 128 = 7 ... 104
```

이 경우 128개짜리 배치 7개가 만들어지고, 마지막에 104개가 남는다.

`drop_last=True`를 사용하면 마지막 104개짜리 배치를 버린다.

```text
drop_last=True:
크기가 부족한 마지막 배치를 버림

drop_last=False:
크기가 부족해도 마지막 배치를 사용함
```

크기가 작은 마지막 배치를 사용하면 다른 배치와 크기가 달라져 학습에서 상대적으로 영향이 달라질 수 있다. 그래서 실습에서는 `drop_last=True`를 사용한다.

---

### 20.3 모델 설계하기

MNIST 이미지는 `28 × 28 = 784`개의 픽셀값을 가진다.

분류해야 하는 숫자는 0부터 9까지 총 10개이다.

따라서 입력 차원과 출력 차원은 다음과 같다.

```text
input_dim = 784
output_dim = 10
```

모델은 `nn.Linear` 하나로 구현할 수 있다.

```python
linear = nn.Linear(784, 10, bias=True).to(device)
```

각 부분의 의미는 다음과 같다.

```text
nn.Linear(784, 10):
784개의 입력값을 받아 10개의 출력값을 만든다.

bias=True:
편향 b를 사용한다.

.to(device):
모델의 파라미터를 CPU 또는 GPU로 보낸다.
```

이 모델이 하는 일은 다음과 같다.

```text
784차원 입력 벡터
   ↓
Linear(784, 10)
   ↓
10개 숫자 클래스에 대한 점수 출력
```

수식으로 보면 다음과 같다.

$$
Z = XW + B
$$

여기서 출력 `Z`는 아직 softmax를 통과하지 않은 점수값이다.

---

### 20.4 비용 함수와 옵티마이저 정의

비용 함수로는 `nn.CrossEntropyLoss()`를 사용한다.

```python
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)
```

각 코드의 의미는 다음과 같다.

```text
criterion = nn.CrossEntropyLoss():
크로스 엔트로피 비용 함수를 만든다.

.to(device):
비용 함수도 모델과 같은 장치에서 사용하도록 한다.

optimizer = torch.optim.SGD(...):
SGD 방식으로 모델 파라미터를 업데이트한다.

linear.parameters():
linear 모델 내부의 W와 b를 학습 대상으로 지정한다.

lr=0.1:
학습률을 0.1로 설정한다.
```

`nn.CrossEntropyLoss()`는 내부적으로 softmax를 포함한다.

따라서 모델의 출력에 softmax를 따로 적용하지 않는다.

```text
올바른 흐름:
linear(X) → CrossEntropyLoss

피해야 할 흐름:
linear(X) → softmax → CrossEntropyLoss
```

---

### 20.5 전체 학습 코드

```python
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)

    for X, Y in data_loader:
        X = X.view(-1, 28 * 28).to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = linear(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning finished')
```

---

### 20.6 학습 코드 흐름 설명

전체 학습 코드는 다음 순서로 진행된다.

```text
1. epoch 반복을 시작한다.
2. DataLoader에서 미니 배치 X, Y를 꺼낸다.
3. X를 [batch_size, 784] 형태로 펼친다.
4. X와 Y를 device로 보낸다.
5. 이전 기울기를 초기화한다.
6. linear(X)로 예측 점수를 계산한다.
7. criterion으로 cost를 계산한다.
8. cost.backward()로 기울기를 계산한다.
9. optimizer.step()으로 W와 b를 업데이트한다.
10. 배치별 cost를 누적해서 에포크 평균 cost를 출력한다.
```

각 코드의 의미를 조금 더 자세히 보면 다음과 같다.

```python
X = X.view(-1, 28 * 28).to(device)
```

이 코드는 입력 이미지를 다음 형태로 바꾼다.

```text
[batch_size, 1, 28, 28]
   ↓
[batch_size, 784]
```

```python
Y = Y.to(device)
```

레이블도 모델과 같은 장치로 보낸다.

```python
optimizer.zero_grad()
```

이전 반복에서 계산된 기울기를 초기화한다.

```python
hypothesis = linear(X)
```

모델의 예측값을 계산한다. 여기서 `hypothesis`는 확률값이 아니라 softmax 이전의 점수값이다.

```python
cost = criterion(hypothesis, Y)
```

예측 점수와 실제 레이블을 비교해 비용을 계산한다.

```python
cost.backward()
optimizer.step()
```

역전파로 기울기를 계산하고, 그 기울기를 이용해 모델 파라미터를 업데이트한다.

---

### 20.7 학습 결과 예시

학습을 진행하면 다음과 같이 cost가 점점 감소하는 모습을 볼 수 있다.

```text
Epoch: 0001 cost = 0.535468459
Epoch: 0002 cost = 0.359274209
Epoch: 0003 cost = 0.331187516
Epoch: 0004 cost = 0.316578060
Epoch: 0005 cost = 0.307158142
Epoch: 0006 cost = 0.300180763
Epoch: 0007 cost = 0.295130193
Epoch: 0008 cost = 0.290851474
Epoch: 0009 cost = 0.287417054
Epoch: 0010 cost = 0.284379572
Epoch: 0011 cost = 0.281825274
Epoch: 0012 cost = 0.279800713
Epoch: 0013 cost = 0.277808994
Epoch: 0014 cost = 0.276154339
Epoch: 0015 cost = 0.274440885
Learning finished
```

cost가 감소한다는 것은 모델이 훈련 데이터에 대해 점점 더 잘 맞추는 방향으로 학습되고 있다는 뜻이다.

---

### 20.8 테스트 데이터로 모델 평가하기

학습이 끝난 뒤에는 테스트 데이터를 사용해 모델을 평가한다.

```python
with torch.no_grad():
    X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = linear(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
    Y_single_data = mnist_test.test_labels[r:r + 1].to(device)

    print('Label: ', Y_single_data.item())
    single_prediction = linear(X_single_data)
    print('Prediction: ', torch.argmax(single_prediction, 1).item())

    plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()
```

위 코드에서 중요한 부분은 `torch.no_grad()`이다.

```python
with torch.no_grad():
```

이 블록 안에서는 기울기를 계산하지 않는다.

테스트 단계에서는 모델을 학습시키는 것이 아니라 예측 성능만 확인하므로, 기울기 계산이 필요 없다.

```text
torch.no_grad() 사용 이유:
1. 불필요한 기울기 계산을 막는다.
2. 메모리 사용량을 줄인다.
3. 평가 속도를 높인다.
```

---

### 20.9 정확도 계산

```python
prediction = linear(X_test)
correct_prediction = torch.argmax(prediction, 1) == Y_test
accuracy = correct_prediction.float().mean()
print('Accuracy:', accuracy.item())
```

각 코드의 의미는 다음과 같다.

```text
prediction = linear(X_test):
테스트 데이터 전체에 대한 예측 점수를 계산한다.

torch.argmax(prediction, 1):
각 샘플에서 가장 큰 점수를 가진 클래스 인덱스를 선택한다.

== Y_test:
예측한 클래스와 실제 정답이 같은지 비교한다.

correct_prediction.float().mean():
맞힌 비율을 평균으로 계산한다.
```

출력 예시는 다음과 같다.

```text
Accuracy: 0.8883000016212463
```

이는 테스트 데이터에서 약 88.83%를 맞혔다는 의미이다.

---

### 20.10 임의의 테스트 이미지 예측

테스트 데이터에서 임의의 이미지 하나를 뽑아 예측할 수 있다.

```python
r = random.randint(0, len(mnist_test) - 1)
X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
Y_single_data = mnist_test.test_labels[r:r + 1].to(device)

print('Label: ', Y_single_data.item())
single_prediction = linear(X_single_data)
print('Prediction: ', torch.argmax(single_prediction, 1).item())
```

예를 들어 출력이 다음과 같다면,

```text
Label:  5
Prediction:  5
```

실제 정답도 5이고, 모델의 예측도 5라는 뜻이다.

```text
Label:
실제 정답

Prediction:
모델이 예측한 값
```

이미지를 직접 확인하기 위해 다음 코드를 사용한다.

```python
plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28),
           cmap='Greys',
           interpolation='nearest')
plt.show()
```

각 인자의 의미는 다음과 같다.

```text
cmap='Greys':
이미지를 회색조로 보여준다.

interpolation='nearest':
이미지를 확대할 때 픽셀 경계를 비교적 그대로 보여준다.
```

---

## 21. 반드시 이해해야 할 핵심 정리

1. 다중 클래스 분류는 세 개 이상의 클래스 중 하나를 선택하는 문제이다.

2. 원-핫 인코딩은 클래스 개수만큼 차원을 만들고, 정답 클래스 위치만 1로 표현하는 방식이다.

3. 정수 인코딩은 클래스 사이에 원하지 않는 순서 관계를 줄 수 있다.

4. 일반적인 다중 클래스 분류에서는 클래스 사이에 순서가 없으므로 원-핫 인코딩이 적절하다.

5. 소프트맥스 회귀는 다중 클래스 분류에 사용된다.

6. 소프트맥스 회귀의 가설식은 $H(X)=softmax(XW+B)$이다.

7. 소프트맥스 함수는 여러 개의 점수를 각 클래스에 대한 확률 벡터로 바꾼다.

8. 소프트맥스 출력값은 각 원소가 0과 1 사이이고, 전체 합이 1이다.

9. 소프트맥스 회귀에서는 가장 큰 확률값을 가진 클래스를 최종 예측값으로 선택한다.

10. 크로스 엔트로피는 실제 정답 클래스의 예측 확률이 높을수록 cost가 작아지도록 만든다.

11. 소프트맥스 회귀의 비용 함수는 $-\sum y_j\log(p_j)$ 형태이다.

12. 전체 데이터에 대해서는 각 샘플의 크로스 엔트로피 값을 평균낸다.

13. PyTorch에서 로우 레벨로 구현할 때는 `F.softmax()`와 `torch.log()`를 직접 사용할 수 있다.

14. `F.log_softmax()`는 `softmax`와 `log`를 합친 함수이다.

15. `F.cross_entropy()`는 내부적으로 `F.log_softmax()`와 `F.nll_loss()`를 포함한다.

16. `F.cross_entropy()`에 넣는 입력은 softmax를 거친 확률값이 아니라 softmax 이전의 logit이어야 한다.

17. `nn.CrossEntropyLoss()`도 `F.cross_entropy()`와 마찬가지로 softmax 과정을 내부에 포함한다.

18. MNIST 데이터는 0부터 9까지 손글씨 숫자 이미지로 구성된 다중 클래스 분류 데이터셋이다.

19. MNIST 이미지 하나는 28 × 28 픽셀이므로 784차원 벡터로 펼칠 수 있다.

20. MNIST를 소프트맥스 회귀로 분류할 때는 `nn.Linear(784, 10)`을 사용할 수 있다.

21. `DataLoader`는 데이터를 미니 배치 단위로 불러오는 역할을 한다.

22. `X.view(-1, 28*28)`은 이미지를 `[batch_size, 784]` 형태로 바꾸는 코드이다.

23. `torch.no_grad()`는 평가 단계에서 기울기 계산을 하지 않도록 한다.

24. `torch.argmax(prediction, 1)`은 각 샘플에서 가장 점수가 높은 클래스를 선택한다.

---

## 22. 헷갈리기 쉬운 부분 정리

### 22.1 소프트맥스 회귀는 회귀인가 분류인가?

이름에는 회귀가 들어가지만 실제로는 다중 클래스 분류에 사용된다.

```text
소프트맥스 회귀 = 다중 클래스 분류 모델
```

---

### 22.2 원-핫 인코딩과 정수 인코딩은 무엇이 다른가?

정수 인코딩은 클래스를 하나의 숫자로 표현한다.

```text
Banana = 1
Tomato = 2
Apple = 3
```

원-핫 인코딩은 클래스 개수만큼 차원을 만들고 정답 위치만 1로 표현한다.

```text
Banana = [1, 0, 0]
Tomato = [0, 1, 0]
Apple = [0, 0, 1]
```

정수 인코딩은 숫자 크기 때문에 클래스 사이에 순서 관계가 생길 수 있다.

반면 원-핫 인코딩은 서로 다른 클래스 사이의 거리를 균등하게 만든다.

---

### 22.3 softmax 출력값은 무엇인가?

softmax 출력값은 각 클래스가 정답일 확률처럼 해석할 수 있는 값이다.

```text
[0.26, 0.71, 0.03]
```

위 값은 다음처럼 해석할 수 있다.

```text
1번 클래스일 확률 = 0.26
2번 클래스일 확률 = 0.71
3번 클래스일 확률 = 0.03
```

최종 예측은 가장 큰 값인 0.71에 해당하는 2번 클래스이다.

---

### 22.4 F.cross_entropy()에는 softmax를 먼저 적용해야 하는가?

적용하면 안 된다.

`F.cross_entropy()`는 내부적으로 `log_softmax`를 포함하고 있다.

따라서 입력으로는 softmax 이전 값인 logit을 넣어야 한다.

```python
# 올바른 방식
cost = F.cross_entropy(z, y)

# 피해야 할 방식
cost = F.cross_entropy(F.softmax(z, dim=1), y)
```

---

### 22.5 CrossEntropyLoss의 정답 y는 원-핫 벡터인가?

`F.cross_entropy()`나 `nn.CrossEntropyLoss()`를 사용할 때 정답은 원-핫 벡터가 아니라 정수 레이블이다.

```python
y_train = torch.LongTensor([2, 2, 2, 1, 1, 1, 0, 0])
```

즉, 정답은 다음처럼 넣는다.

```text
정답 클래스 번호
```

로우 레벨로 직접 비용 함수를 구현할 때만 원-핫 벡터를 직접 만들어 사용한다.

---

### 22.6 hypothesis와 prediction은 항상 확률값인가?

아니다.

로우 레벨 구현에서 다음 코드는 확률값이다.

```python
hypothesis = F.softmax(x_train.matmul(W) + b, dim=1)
```

하지만 `F.cross_entropy()`를 사용할 때 다음 코드는 확률값이 아니다.

```python
prediction = model(x_train)
```

이 값은 softmax 이전의 점수값, 즉 logit이다.

---

### 22.7 y_train의 shape는 왜 [8, 1]이 아니라 [8]인가?

`F.cross_entropy()`는 정답을 정수 클래스 레이블 형태로 받는다.

따라서 정답 텐서는 보통 다음과 같은 형태이다.

```text
y_train.shape = [8]
```

각 원소가 해당 샘플의 정답 클래스 번호를 의미한다.

---

### 22.8 view(-1, 28*28)에서 -1은 무엇인가?

`-1`은 해당 차원의 크기를 PyTorch가 자동으로 계산하라는 뜻이다.

```python
X = X.view(-1, 28 * 28)
```

배치 크기가 100이면 결과는 다음과 같다.

```text
[100, 1, 28, 28]
   ↓
[100, 784]
```

즉, 이미지 하나를 784차원 벡터로 펼치되, 배치 크기는 자동으로 유지한다.

---

### 22.9 torch.no_grad()는 왜 쓰는가?

평가 단계에서는 모델을 학습시키지 않는다.

따라서 기울기를 계산할 필요가 없다.

```python
with torch.no_grad():
    prediction = linear(X_test)
```

이렇게 하면 불필요한 기울기 계산을 막아 메모리와 연산량을 줄일 수 있다.

---

### 22.10 argmax는 무엇을 하는가?

`torch.argmax(prediction, 1)`은 각 샘플에서 가장 큰 값을 가진 클래스 인덱스를 반환한다.

예를 들어 예측 점수가 다음과 같다고 하자.

```text
[1.2, 0.3, 4.8]
```

가장 큰 값은 4.8이고, 위치는 2번 인덱스이다.

따라서 argmax 결과는 다음과 같다.

```text
2
```

소프트맥스 회귀에서는 이 인덱스를 최종 예측 클래스로 사용한다.

---

## 23. 전체 내용을 한 줄로 요약

```text
소프트맥스 회귀는 여러 클래스에 대한 점수 XW+B를 softmax로 확률 벡터로 바꾸고, 크로스 엔트로피 비용 함수로 실제 정답과의 차이를 계산하여 다중 클래스 분류를 수행하는 모델이다.
```

