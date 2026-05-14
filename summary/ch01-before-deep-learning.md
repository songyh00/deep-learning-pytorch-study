# 01. 딥 러닝 시작 전 기초 정리

## 1. 머신 러닝 워크플로우(Machine Learning Workflow)

### 머신 러닝 워크플로우란?

머신 러닝 워크플로우는 **데이터를 수집하고, 점검하고, 전처리한 뒤 모델을 훈련·평가·배포하는 전체 과정**이다.

```text
머신 러닝 워크플로우 = 데이터 수집부터 모델 배포까지의 과정
```

데이터 사이언스(Data Science)나 머신 러닝(Machine Learning)에서는 일반적으로 이러한 흐름을 거친다.  
딥 러닝(Deep Learning)도 머신 러닝의 한 갈래이므로, 딥 러닝 워크플로우도 머신 러닝 워크플로우로 볼 수 있다.

<details>
<summary>머신 러닝과 딥 러닝의 관계</summary>

머신 러닝: 컴퓨터가 데이터를 통해 규칙이나 패턴을 학습하고, 이를 바탕으로 예측이나 분류 같은 작업을 수행하도록 하는 방법  
딥 러닝: 인공 신경망을 여러 층으로 깊게 쌓아 복잡한 패턴을 학습하는 방법

```text
머신 러닝
└─ 딥 러닝
```

</details>

---

### 머신 러닝 워크플로우의 전체 흐름

머신 러닝을 하는 과정은 크게 6단계로 나눌 수 있다.

```text
머신 러닝 워크플로우
├─ 수집
├─ 점검 및 탐색
├─ 전처리 및 정제
├─ 모델링 및 훈련
├─ 평가
└─ 배포
```

<img width="829" height="237" alt="Image" src="https://github.com/user-attachments/assets/514fa784-4254-403f-ad30-a02496865419" />

---

### 수집(Acquisition)이란?

수집은 머신 러닝을 하기 위해 **기계에 학습시켜야 할 데이터를 모으는 단계**이다.

```text
수집 = 머신 러닝에 사용할 데이터를 모으는 단계
```

자연어 처리에서는 자연어 데이터를 **말뭉치** 또는 **코퍼스(Corpus)** 라고 부른다.  
코퍼스는 조사나 연구 목적에 의해 특정 도메인으로부터 수집된 텍스트 집합을 의미한다.

---

### 텍스트 데이터의 형식과 출처

텍스트 데이터의 파일 형식은 다양하다.

예시:

```text
txt 파일, csv 파일, xml 파일
```

데이터의 출처도 다양하다.

예시:

```text
음성 데이터, 웹 수집기를 통해 수집된 데이터, 영화 리뷰
```

즉, 머신 러닝을 시작하기 위해서는 먼저 목적에 맞는 데이터를 확보해야 한다.

---

### 점검 및 탐색(Inspection and Exploration)이란?

점검 및 탐색은 수집한 데이터를 **확인하고 분석하는 단계**이다.

```text
점검 및 탐색 = 데이터의 상태와 특징을 파악하는 단계
```

이 단계에서는 다음과 같은 내용을 확인한다.

- 데이터의 구조
- 노이즈 데이터
- 머신 러닝 적용을 위해 데이터를 어떻게 정제해야 하는지

---

### 탐색적 데이터 분석(EDA)이란?

점검 및 탐색 단계는 **탐색적 데이터 분석(Exploratory Data Analysis, EDA)** 단계라고도 한다.

```text
EDA = 데이터의 특징과 구조적 관계를 파악하는 과정
```

EDA에서는 다음과 같은 요소를 점검한다.

- 독립 변수
- 종속 변수
- 변수 유형
- 변수의 데이터 타입

이 과정에서 시각화와 간단한 통계 테스트를 진행하기도 한다.

<details>
<summary>독립 변수와 종속 변수</summary>

독립 변수: 결과에 영향을 주는 입력에 해당하는 변수  
종속 변수: 독립 변수의 영향을 받아 결정되는 결과에 해당하는 변수

</details>

---

### 전처리 및 정제(Preprocessing and Cleaning)란?

전처리 및 정제는 데이터를 머신 러닝에 사용할 수 있도록 **가공하고 정리하는 단계**이다.

```text
전처리 및 정제 = 데이터를 학습에 적합한 형태로 다듬는 과정
```

데이터에 대한 파악이 끝나면 전처리 과정에 들어간다.  
이 단계는 머신 러닝 워크플로우에서 가장 까다로운 작업 중 하나이다.

---

### 자연어 처리에서의 전처리 예시

자연어 처리에서는 전처리 단계에 다음과 같은 작업이 포함될 수 있다.

```text
토큰화, 정제, 정규화, 불용어 제거
```

빠르고 정확한 데이터 전처리를 위해서는 사용하는 툴과 라이브러리에 대한 지식이 필요하다.  
이 책에서는 파이썬을 기준으로 설명한다.

전처리가 매우 까다로운 경우에는 전처리 과정에서 머신 러닝이 사용되기도 한다.

---

### 모델링 및 훈련(Modeling and Training)이란?

모델링 및 훈련은 머신 러닝 알고리즘을 선택하고, 데이터를 이용해 **기계가 학습하도록 만드는 단계**이다.

```text
모델링 및 훈련 = 모델을 만들고 데이터를 통해 학습시키는 단계
```

데이터 전처리가 끝나면 머신 러닝 코드를 작성하는 모델링 단계에 들어간다.  
적절한 머신 러닝 알고리즘을 선택하여 모델링을 진행한 뒤, 전처리가 완료된 데이터를 이용해 기계에게 학습시킨다.

이 과정을 **학습(training)** 또는 **훈련**이라고 부른다.  
두 용어는 혼용해서 사용한다.

---

### 훈련 후 수행할 수 있는 작업

기계가 데이터에 대한 학습을 마치고 훈련이 제대로 되었다면, 원하는 태스크를 수행할 수 있다.

자연어 처리 작업 예시:

```text
기계 번역, 음성 인식, 텍스트 분류
```

즉, 모델은 학습을 통해 주어진 작업을 수행할 수 있는 상태가 된다.

---

### 모든 데이터를 학습에 사용하면 안 되는 이유

모델링 및 훈련 단계에서 주의해야 할 점은 **대부분의 경우 모든 데이터를 기계에게 학습시켜서는 안 된다**는 것이다.

데이터 중 일부는 테스트용으로 남겨두고, 훈련용 데이터만 훈련에 사용해야 한다.

```text
훈련용 데이터 = 모델 학습에 사용
테스트용 데이터 = 모델 성능 평가에 사용
```

이렇게 나누는 이유는 다음과 같다.

- 학습 후 테스트용 데이터를 통해 현재 성능을 측정할 수 있다.
- 과적합(overfitting) 상황을 막을 수 있다.

더 좋은 방법은 훈련용, 검증용, 테스트용 데이터로 나누는 것이다.

```text
전체 데이터
├─ 훈련용 데이터
├─ 검증용 데이터
└─ 테스트용 데이터
```

<img width="346" height="175" alt="Image" src="https://github.com/user-attachments/assets/2861fa08-2950-41e7-87b0-9902c4a80b3f" />

<details>
<summary>과적합의 의미</summary>

과적합은 모델이 훈련용 데이터에 지나치게 맞춰져서, 새로운 데이터에 대해서는 성능이 떨어지는 상황을 말한다.

</details>

---

### 훈련용, 검증용, 테스트용 데이터의 차이

훈련용, 검증용, 테스트용 데이터의 차이는 시험에 비유하면 이해하기 쉽다.

| 구분 | 역할 | 시험 비유 |
|---|---|---|
| 훈련용 데이터(Training Data) | 모델을 학습시키는 데 사용 | 학습지 |
| 검증용 데이터(Validation Data) | 현재 모델의 성능을 판단하고 개선하는 데 사용 | 모의고사 |
| 테스트용 데이터(Test Data) | 모델의 최종 성능을 평가하는 데 사용 | 수능 시험 |

검증용 데이터는 현재 모델의 성능을 판단하고 개선하는 데 사용된다.  
반면 테스트용 데이터는 모델의 최종 성능을 수치화하여 평가하기 위해 사용된다.

```text
검증용 데이터 = 모델 개선에 사용
테스트용 데이터 = 최종 평가에 사용
```

현업에서는 검증용 데이터를 사용하는 것이 거의 필수적이다.

---

### 평가(Evaluation)란?

평가는 학습이 끝난 모델의 성능을 **테스트용 데이터로 확인하는 단계**이다.

```text
평가 = 테스트용 데이터로 모델 성능을 측정하는 단계
```

평가 방법은 기계가 예측한 데이터가 테스트용 데이터의 실제 정답과 얼마나 가까운지를 측정한다.

즉, 모델이 새로운 데이터에 대해 얼마나 잘 맞히는지를 확인하는 과정이다.

---

### 배포(Deployment)란?

배포는 평가 단계에서 성공적으로 훈련되었다고 판단된 모델을 **실제로 사용할 수 있도록 내보내는 단계**이다.

```text
배포 = 완성된 모델을 실제 사용 가능한 형태로 제공하는 단계
```

다만 배포가 끝났다고 해서 모든 과정이 완전히 종료되는 것은 아니다.  
완성된 모델에 대한 전체적인 피드백으로 인해 모델을 업데이트해야 하는 상황이 올 수 있다.

이 경우 다시 수집 단계로 돌아갈 수 있다.

```text
피드백 발생 → 모델 업데이트 필요 → 수집 단계로 돌아갈 수 있음
```

---

## 2. 데이터 분석 필수 패키지(Pandas, NumPy, Matplotlib)

### 데이터 분석 필수 패키지 삼대장

파이썬으로 데이터 분석을 할 때 자주 사용하는 필수 패키지는 **Pandas, NumPy, Matplotlib**이다.

```text
데이터 분석 필수 패키지
├─ Pandas
├─ NumPy
└─ Matplotlib
```

아나콘다를 설치했다면 세 패키지는 추가 설치 없이 사용할 수 있다.  
아나콘다를 설치하지 않았다면 각각 `pip install` 명령어로 별도 설치할 수 있다.

---

### 세 패키지의 역할

각 패키지는 데이터 분석 과정에서 서로 다른 역할을 한다.

| 패키지 | 주요 역할 |
|---|---|
| Pandas | 데이터를 표 형태로 처리하고 분석 |
| NumPy | 수치 데이터와 배열, 행렬 연산 처리 |
| Matplotlib | 데이터를 그래프나 차트로 시각화 |

```text
Pandas = 데이터 처리
NumPy = 수치 계산
Matplotlib = 데이터 시각화
```

<details>
<summary>패키지와 라이브러리</summary>

패키지나 라이브러리는 특정 기능을 쉽게 사용할 수 있도록 미리 만들어 둔 코드 모음이다.  
예를 들어 Pandas를 사용하면 표 형태의 데이터를 직접 처음부터 구현하지 않고도 쉽게 다룰 수 있다.

</details>

---

### Pandas란?

Pandas는 **파이썬에서 데이터를 처리하기 위한 라이브러리**이다.

```text
Pandas = 파이썬 데이터 처리를 위한 라이브러리
```

파이썬을 이용한 데이터 분석 작업에서 필수 라이브러리로 알려져 있다.

아나콘다를 설치하지 않았다면 아래 명령어로 Pandas를 설치할 수 있다.

```bash
pip install pandas
```

Pandas는 보통 `pd`라는 이름으로 임포트한다.

```python
import pandas as pd
```

---

### Pandas의 데이터 구조

Pandas는 총 세 가지 데이터 구조를 사용한다.

```text
Pandas 데이터 구조
├─ Series
├─ DataFrame
└─ Panel
```

이 중 **DataFrame**이 가장 많이 사용된다.  
여기서는 **Series**와 **DataFrame**을 중심으로 다룬다.

---

### 시리즈(Series)란?

시리즈는 **1차원 배열의 값에 인덱스를 붙인 구조**이다.

```text
시리즈(Series) = 값(values) + 인덱스(index)
```

예시:

```python
sr = pd.Series([17000, 18000, 1000, 5000],
               index=["피자", "치킨", "콜라", "맥주"])

print(sr)
```

출력:

```text
피자    17000
치킨    18000
콜라     1000
맥주     5000
dtype: int64
```

시리즈는 값과 인덱스를 각각 확인할 수 있다.

```python
print(sr.values)
print(sr.index)
```

```text
시리즈의 값 : [17000 18000  1000  5000]
시리즈의 인덱스 : Index(['피자', '치킨', '콜라', '맥주'], dtype='object')
```

---

### 데이터프레임(DataFrame)이란?

데이터프레임은 **행과 열을 가지는 2차원 자료구조**이다.

```text
데이터프레임(DataFrame) = 값(values) + 인덱스(index) + 열(columns)
```

시리즈가 값과 인덱스로 구성된다면, 데이터프레임은 여기에 열 이름이 추가된다.

예시:

```python
values = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
index = ['one', 'two', 'three']
columns = ['A', 'B', 'C']

df = pd.DataFrame(values, index=index, columns=columns)

print(df)
```

출력:

```text
       A  B  C
one    1  2  3
two    4  5  6
three  7  8  9
```

---

### 데이터프레임의 구성 요소

데이터프레임은 인덱스, 열 이름, 값을 확인할 수 있다.

```python
print(df.index)
print(df.columns)
print(df.values)
```

```text
데이터프레임의 인덱스 : Index(['one', 'two', 'three'], dtype='object')
데이터프레임의 열이름: Index(['A', 'B', 'C'], dtype='object')
데이터프레임의 값 :
[[1 2 3]
 [4 5 6]
 [7 8 9]]
```

<details>
<summary>Series와 DataFrame의 차이</summary>

Series는 한 줄짜리 1차원 데이터 구조이다.  
DataFrame은 행과 열을 가진 2차원 데이터 구조이다.

```text
Series = 1차원
DataFrame = 2차원
```

예를 들어 학생 한 명의 점수 목록은 Series처럼 볼 수 있고, 여러 학생의 학번·이름·점수를 모은 표는 DataFrame처럼 볼 수 있다.

</details>

---

### 데이터프레임 생성 방법

데이터프레임은 여러 자료로부터 생성할 수 있다.

```text
데이터프레임 생성 가능 자료
├─ 리스트
├─ 시리즈
├─ 딕셔너리
├─ NumPy ndarray
└─ 또 다른 데이터프레임
```

여기서는 리스트와 딕셔너리를 사용한 생성 방법을 다룬다.

---

### 리스트로 데이터프레임 생성하기

이중 리스트를 사용하면 데이터프레임을 만들 수 있다.

```python
data = [
    ['1000', 'Steve', 90.72],
    ['1001', 'James', 78.09],
    ['1002', 'Doyeon', 98.43],
    ['1003', 'Jane', 64.19],
    ['1004', 'Pilwoong', 81.30],
    ['1005', 'Tony', 99.14],
]

df = pd.DataFrame(data)
print(df)
```

출력:

```text
      0         1      2
0  1000     Steve  90.72
1  1001     James  78.09
2  1002    Doyeon  98.43
3  1003      Jane  64.19
4  1004  Pilwoong  81.30
5  1005      Tony  99.14
```

열 이름을 지정하면 더 의미 있는 데이터프레임을 만들 수 있다.

```python
df = pd.DataFrame(data, columns=['학번', '이름', '점수'])
print(df)
```

출력:

```text
    학번       이름    점수
0  1000     Steve  90.72
1  1001     James  78.09
2  1002    Doyeon  98.43
3  1003      Jane  64.19
4  1004  Pilwoong  81.30
5  1005      Tony  99.14
```

---

### 딕셔너리로 데이터프레임 생성하기

파이썬 딕셔너리를 사용해서도 데이터프레임을 만들 수 있다.

```python
data = {
    '학번' : ['1000', '1001', '1002', '1003', '1004', '1005'],
    '이름' : ['Steve', 'James', 'Doyeon', 'Jane', 'Pilwoong', 'Tony'],
    '점수' : [90.72, 78.09, 98.43, 64.19, 81.30, 99.14]
}

df = pd.DataFrame(data)
print(df)
```

출력:

```text
    학번       이름    점수
0  1000     Steve  90.72
1  1001     James  78.09
2  1002    Doyeon  98.43
3  1003      Jane  64.19
4  1004  Pilwoong  81.30
5  1005      Tony  99.14
```

---

### 데이터프레임 조회하기

데이터프레임에서 원하는 부분만 확인할 때 자주 사용하는 명령어가 있다.

| 명령어 | 의미 |
|---|---|
| `df.head(n)` | 앞부분 n개 확인 |
| `df.tail(n)` | 뒷부분 n개 확인 |
| `df['열이름']` | 특정 열 확인 |

앞부분 3개 확인:

```python
print(df.head(3))
```

뒷부분 3개 확인:

```python
print(df.tail(3))
```

특정 열 확인:

```python
print(df['학번'])
```

---

### 외부 데이터 읽기

Pandas는 다양한 외부 데이터 파일을 읽어 데이터프레임을 생성할 수 있다.

```text
CSV, 텍스트, Excel, SQL, HTML, JSON
```

CSV 파일을 읽을 때는 `read_csv()`를 사용한다.

```python
df = pd.read_csv('example.csv')
print(df)
```

<img width="243" height="177" alt="Image" src="https://github.com/user-attachments/assets/90c15e21-c818-4ac0-aaab-65fc9aabc94c" />

CSV 파일을 읽으면 인덱스가 자동으로 부여된다.

```python
print(df.index)
```

출력:

```text
RangeIndex(start=0, stop=6, step=1)
```

<details>
<summary>CSV 파일 형식</summary>

CSV는 데이터를 쉼표로 구분해서 저장하는 텍스트 파일 형식이다.  
표 형태의 데이터를 저장하거나 교환할 때 자주 사용된다.

```text
CSV = Comma-Separated Values
```

</details>

---

### NumPy란?

NumPy는 **수치 데이터를 다루는 파이썬 패키지**이다.

```text
NumPy = 수치 데이터와 배열 연산을 다루는 패키지
```

NumPy의 핵심은 다차원 행렬 자료구조인 **ndarray**이다.  
ndarray는 벡터와 행렬을 사용하는 선형대수 계산에서 주로 사용된다.

NumPy는 편의성뿐 아니라 속도 면에서도 순수 파이썬에 비해 빠르다는 장점이 있다.

아나콘다를 설치하지 않았다면 아래 명령어로 NumPy를 설치할 수 있다.

```bash
pip install numpy
```

NumPy는 보통 `np`라는 이름으로 임포트한다.

```python
import numpy as np
```

---

### `np.array()`

`np.array()`는 리스트, 튜플, 배열로부터 ndarray를 생성한다.

```text
np.array() = ndarray 생성
```

1차원 배열 예시:

```python
vec = np.array([1, 2, 3, 4, 5])
print(vec)
```

출력:

```text
[1 2 3 4 5]
```

2차원 배열 예시:

```python
mat = np.array([[10, 20, 30], [60, 70, 80]])
print(mat)
```

출력:

```text
[[10 20 30]
 [60 70 80]]
```

두 배열의 타입은 모두 `numpy.ndarray`이다.

```python
print(type(vec))
print(type(mat))
```

---

### `ndim`과 `shape`

NumPy 배열에는 축의 개수와 크기라는 개념이 있다.

```text
ndim = 축의 개수
shape = 배열의 크기
```

예시:

```python
print(vec.ndim)
print(vec.shape)
```

출력:

```text
1
(5,)
```

```python
print(mat.ndim)
print(mat.shape)
```

출력:

```text
2
(2, 3)
```

배열의 크기를 정확히 아는 것은 딥 러닝에서 매우 중요하다.

<details>
<summary>shape 확인이 중요한 이유</summary>

딥 러닝에서는 데이터가 보통 벡터나 행렬 형태로 모델에 입력된다.  
이때 입력 데이터의 크기가 맞지 않으면 연산이 제대로 수행되지 않을 수 있다.

예를 들어 `shape`가 `(2, 3)`이라는 것은 2행 3열 구조라는 뜻이다.

</details>

---

### ndarray의 초기화

ndarray를 만드는 다양한 방법이 있다.

| 함수 | 의미 |
|---|---|
| `np.zeros()` | 모든 원소가 0인 배열 생성 |
| `np.ones()` | 모든 원소가 1인 배열 생성 |
| `np.full()` | 사용자가 지정한 값으로 채운 배열 생성 |
| `np.eye()` | 대각선은 1, 나머지는 0인 2차원 배열 생성 |
| `np.random.random()` | 임의의 값으로 채운 배열 생성 |

예시:

```python
zero_mat = np.zeros((2, 3))
one_mat = np.ones((2, 3))
same_value_mat = np.full((2, 2), 7)
eye_mat = np.eye(3)
random_mat = np.random.random((2, 2))
```

---

### `np.arange()`

`np.arange(n)`은 0부터 n-1까지의 값을 가지는 배열을 생성한다.

```python
range_vec = np.arange(10)
print(range_vec)
```

출력:

```text
[0 1 2 3 4 5 6 7 8 9]
```

`np.arange(i, j, k)`는 i부터 j-1까지 k씩 증가하는 배열을 생성한다.

```python
range_n_step_vec = np.arange(1, 10, 2)
print(range_n_step_vec)
```

출력:

```text
[1 3 5 7 9]
```

---

### `np.reshape()`

`np.reshape()`은 내부 데이터는 변경하지 않고 배열의 구조를 바꾼다.

```text
reshape = 데이터는 그대로 두고 배열 모양만 변경
```

예시:

```python
reshape_mat = np.array(np.arange(30)).reshape((5, 6))
print(reshape_mat)
```

출력:

```text
[[ 0  1  2  3  4  5]
 [ 6  7  8  9 10 11]
 [12 13 14 15 16 17]
 [18 19 20 21 22 23]
 [24 25 26 27 28 29]]
```

---

### NumPy 슬라이싱

NumPy 배열은 리스트처럼 슬라이싱 기능을 지원한다.  
슬라이싱을 사용하면 특정 행이나 열의 원소에 접근할 수 있다.

```text
슬라이싱 = 배열의 일부를 잘라서 가져오는 기능
```

예시:

```python
mat = np.array([[1, 2, 3], [4, 5, 6]])
print(mat[0, :])
print(mat[:, 1])
```

출력:

```text
[1 2 3]
[2 5]
```

---

### NumPy 정수 인덱싱(Integer Indexing)

정수 인덱싱은 원하는 위치의 원소들을 직접 뽑는 방식이다.

```text
정수 인덱싱 = 원하는 위치의 원소를 선택해서 가져오는 방법
```

슬라이싱은 연속적인 부분 배열을 추출할 때 유용하다.  
하지만 연속적이지 않은 원소들을 뽑아 새로운 배열을 만들 때는 정수 인덱싱을 사용할 수 있다.

예시:

```python
mat = np.array([[1, 2], [4, 5], [7, 8]])

print(mat[1, 0])

indexing_mat = mat[[2, 1], [0, 1]]
print(indexing_mat)
```

출력:

```text
4
[7 5]
```

---

### NumPy 연산

NumPy를 사용하면 배열 간 연산을 쉽게 수행할 수 있다.

| 연산 | 연산자 | 함수 |
|---|---|---|
| 덧셈 | `+` | `np.add()` |
| 뺄셈 | `-` | `np.subtract()` |
| 곱셈 | `*` | `np.multiply()` |
| 나눗셈 | `/` | `np.divide()` |

예시:

```python
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

print(x + y)
print(x - y)
print(x * y)
print(x / y)
```

`*`를 통해 수행하는 것은 요소별 곱이다.  
벡터와 행렬곱 또는 행렬곱을 수행하려면 `dot()`을 사용해야 한다.

```python
mat1 = np.array([[1, 2], [3, 4]])
mat2 = np.array([[5, 6], [7, 8]])

mat3 = np.dot(mat1, mat2)
print(mat3)
```

출력:

```text
[[19 22]
 [43 50]]
```

<details>
<summary>요소별 곱과 행렬곱의 차이</summary>

`*`는 같은 위치에 있는 원소끼리 곱하는 요소별 곱이다.  
`np.dot()`은 행렬의 곱셈 규칙에 따라 계산하는 행렬곱이다.

```text
* = 요소별 곱
np.dot() = 행렬곱
```

</details>

---

### Matplotlib이란?

Matplotlib은 **데이터를 차트나 플롯으로 시각화하는 패키지**이다.

```text
Matplotlib = 데이터 시각화 패키지
```

데이터 분석에서는 분석 전 데이터를 이해하기 위한 시각화나, 분석 후 결과를 시각화하기 위해 사용된다.

아나콘다를 설치하지 않았다면 아래 명령어로 Matplotlib을 설치할 수 있다.

```bash
pip install matplotlib
```

Matplotlib의 주요 모듈인 `pyplot`은 보통 `plt`라는 이름으로 임포트한다.

```python
import matplotlib.pyplot as plt
```

---

### 라인 플롯(Line Plot) 그리기

`plot()`은 라인 플롯을 그리는 기능을 수행한다.

```text
plot() = 라인 플롯 생성
```

`title()`을 사용하면 그래프 제목을 지정할 수 있고, `show()`를 사용하면 그래프를 화면에 표시할 수 있다.

예시:

```python
plt.title('test')
plt.plot([1, 2, 3, 4], [2, 4, 8, 6])
plt.show()
```

<img width="324" height="243" alt="Image" src="https://github.com/user-attachments/assets/1fc6a373-31e2-4982-9093-3a085ebd67d5" />

---

### 축 레이블 삽입하기

x축과 y축에 이름을 넣고 싶다면 `xlabel()`과 `ylabel()`을 사용한다.

```text
xlabel() = x축 이름 설정
ylabel() = y축 이름 설정
```

예시:

```python
plt.title('test')
plt.plot([1, 2, 3, 4], [2, 4, 8, 6])
plt.xlabel('hours')
plt.ylabel('score')
plt.show()
```

<img width="348" height="247" alt="Image" src="https://github.com/user-attachments/assets/1bfa288f-95c5-43b8-b675-a3eb8afbb9c9" />

---

### 라인 추가와 범례 삽입하기

여러 개의 `plot()`을 사용하면 하나의 그래프에 여러 라인을 나타낼 수 있다.

```text
여러 개의 plot() = 하나의 그래프에 여러 라인 추가
```

여러 라인 플롯을 동시에 사용할 경우, 각 선이 어떤 데이터를 나타내는지 보여주기 위해 범례를 사용한다.

```text
legend() = 범례 삽입
```

예시:

```python
plt.title('students')
plt.plot([1, 2, 3, 4], [2, 4, 8, 6])
plt.plot([1.5, 2.5, 3.5, 4.5], [3, 5, 8, 10])
plt.xlabel('hours')
plt.ylabel('score')
plt.legend(['A student', 'B student'])
plt.show()
```

<img width="344" height="242" alt="Image" src="https://github.com/user-attachments/assets/65d0f8ed-fcf9-4f07-8796-33d21e5b5e3f" />

<details>
<summary>범례의 역할</summary>

범례는 그래프에서 각 선이나 막대가 어떤 데이터를 의미하는지 알려주는 설명이다.  
여러 데이터를 하나의 그래프에 함께 표시할 때 유용하다.

</details>

---

## 3. 데이터의 분리(Splitting Data)

### 데이터 분리란?

머신 러닝 모델을 학습시키고 평가하기 위해서는 데이터를 적절하게 분리하는 작업이 필요하다.

이 책에서는 대부분의 경우 **지도 학습(Supervised Learning)** 을 다룬다.  
따라서 여기서는 지도 학습을 위한 데이터 분리 작업을 중심으로 다룬다.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
```

---

### 지도 학습(Supervised Learning)이란?

지도 학습의 훈련 데이터는 문제지를 연상하면 이해하기 쉽다.

지도 학습의 훈련 데이터는 정답이 무엇인지 맞혀야 하는 **문제**에 해당되는 데이터와, 레이블이라고 부르는 **정답**이 적혀 있는 데이터로 구성된다.

```text
지도 학습 데이터 = 문제 데이터 + 정답 데이터
```

기계는 정답이 적혀 있는 문제지를 문제와 정답을 함께 보면서 학습한다.  
그리고 향후에는 정답이 없는 문제에 대해서도 정답을 잘 예측해야 한다.

<details>
<summary>지도 학습과 레이블</summary>

지도 학습에서 레이블(label)은 데이터에 붙어 있는 정답을 의미한다.  
예를 들어 메일이 스팸 메일인지 정상 메일인지를 나타내는 값이 레이블이다.

```text
입력 데이터 = 문제
레이블 = 정답
```

</details>

---

### 스팸 메일 분류 예시

스팸 메일 분류기 데이터는 메일 본문과 해당 메일이 정상 메일인지 스팸 메일인지 적혀 있는 레이블로 구성된다.

예를 들어 약 20,000개의 데이터가 있다고 가정하면, 데이터는 다음과 같은 형태가 된다.

| 텍스트(메일의 내용) | 레이블(스팸 여부) |
|---|---|
| 당신에게 드리는 마지막 혜택! ... | 스팸 메일 |
| 내일 뵐 수 있을지 확인 부탁... | 정상 메일 |
| ... | ... |
| (광고) 멋있어질 수 있는... | 스팸 메일 |

이 데이터는 두 개의 열로 구성된다.

```text
첫 번째 열 = 메일의 본문
두 번째 열 = 정상 메일인지 스팸 메일인지에 대한 정답
```

---

### X와 y

기계를 훈련시키기 위해 데이터를 문제와 정답으로 나눈다.

메일의 내용이 담긴 첫 번째 열은 `X`에 저장한다.  
메일이 스팸인지 정상인지 정답이 적혀 있는 두 번째 열은 `y`에 저장한다.

```text
X = 문제지 데이터
y = 정답 데이터
```

예를 들어 20,000개의 데이터가 있다면 다음과 같이 나눌 수 있다.

```text
X = 20,000개의 문제 데이터
y = 20,000개의 정답 데이터
```

이때 중요한 점은 `X`와 `y`의 매핑 관계가 유지되어야 한다는 것이다.  
즉, 어떤 `X`에 대한 정답이 어떤 `y`인지 바로 찾을 수 있어야 한다.

<details>
<summary>X와 y를 분리하는 이유</summary>

머신 러닝에서는 모델이 입력 데이터를 보고 정답을 예측하도록 학습한다.  
그래서 입력 데이터는 `X`, 정답 데이터는 `y`로 분리해서 다루는 경우가 많다.

```text
X = 모델이 보고 학습하는 입력
y = 모델이 맞혀야 하는 정답
```

</details>

---

### 훈련 데이터와 테스트 데이터

`X`와 `y`를 만든 뒤에는 일부 데이터를 시험용으로 따로 분리한다.  
이는 문제지를 다 공부하고 난 뒤 실력을 평가하기 위해 일부 문제와 정답지를 따로 빼놓는 것과 같다.

예를 들어 20,000개의 데이터 중 2,000개를 테스트 데이터로 분리한다고 하면 다음과 같다.

```text
훈련 데이터 = 18,000개의 X, y 쌍
테스트 데이터 = 2,000개의 X, y 쌍
```

이 책에서는 일반적으로 다음과 같은 변수명을 사용한다.

| 구분 | 변수명 | 의미 |
|---|---|---|
| 훈련 데이터 | `X_train` | 문제지 데이터 |
| 훈련 데이터 | `y_train` | 문제지에 대한 정답 데이터 |
| 테스트 데이터 | `X_test` | 시험지 데이터 |
| 테스트 데이터 | `y_test` | 시험지에 대한 정답 데이터 |

기계는 `X_train`과 `y_train`을 함께 보면서 학습한다.  
학습이 끝나면 `y_test`는 보여주지 않고 `X_test`에 대해서 정답을 예측하게 한다.

이후 기계가 예측한 답과 실제 정답인 `y_test`를 비교하여 성능을 평가한다.  
이때 기계가 정답을 얼마나 맞혔는지를 나타내는 대표적인 평가 지표가 **정확도(Accuracy)** 이다.

```text
정확도(Accuracy)
= 모델이 예측한 답과 실제 정답을 비교해 얼마나 맞혔는지 나타내는 대표적인 평가 지표
```

---

### X와 y 분리하기

데이터에서 `X`와 `y`를 분리하는 방법은 여러 가지가 있다.

```text
X와 y 분리 방법
├─ zip 함수 사용
├─ 데이터프레임 사용
└─ NumPy 사용
```

---

### zip 함수로 분리하기

zip() 함수는 리스트처럼 순서가 있는 데이터에서 각 순서에 등장하는 원소들끼리 묶어주는 역할을 한다.

```python
X, y = zip(['a', 1], ['b', 2], ['c', 3])

print('X 데이터 :', X)
print('y 데이터 :', y)
```

출력:

```text
X 데이터 : ('a', 'b', 'c')
y 데이터 : (1, 2, 3)
```

각 데이터에서 첫 번째로 등장한 원소끼리 묶이고, 두 번째로 등장한 원소끼리 묶인 것을 볼 수 있다.

리스트의 리스트 구조에서는 `zip(*sequences)` 형태를 사용할 수 있다.

```python
sequences = [['a', 1], ['b', 2], ['c', 3]]

X, y = zip(*sequences)

print('X 데이터 :', X)
print('y 데이터 :', y)
```

출력:

```text
X 데이터 : ('a', 'b', 'c')
y 데이터 : (1, 2, 3)
```

<details>
<summary>zip(*sequences)의 의미</summary>

`*sequences`는 리스트 안에 들어 있는 여러 데이터를 각각의 인자로 풀어서 전달한다는 의미이다.

```python
zip(*sequences)
```

위 코드는 다음처럼 동작한다고 볼 수 있다.

```python
zip(['a', 1], ['b', 2], ['c', 3])
```

</details>

---

### 데이터프레임으로 분리하기

Pandas 데이터프레임은 열 이름으로 각 열에 접근할 수 있다.  
따라서 열 이름을 이용하면 손쉽게 `X` 데이터와 `y` 데이터를 분리할 수 있다.

```python
values = [
    ['당신에게 드리는 마지막 혜택!', 1],
    ['내일 뵐 수 있을지 확인 부탁드...', 0],
    ['도연씨. 잘 지내시죠? 오랜만입...', 0],
    ['(광고) AI로 주가를 예측할 수 있다!', 1]
]

columns = ['메일 본문', '스팸 메일 유무']

df = pd.DataFrame(values, columns=columns)
df
```

출력 예시:

```text
            메일 본문              스팸 메일 유무
 0   당신에게 드리는 마지막 혜택!           1
 1   내일 뵐 수 있을지 확인 부탁드...       0
 2   도연씨. 잘 지내시죠? 오랜만입...       0
 3   (광고) AI로 주가를 예측할 수 있다!     1
```

`메일 본문` 열은 `X`, `스팸 메일 유무` 열은 `y`로 분리한다.

```python
X = df['메일 본문']
y = df['스팸 메일 유무']
```

출력 코드:

```python
print('X 데이터 :', X.to_list())
print('y 데이터 :', y.to_list())
```

출력:

```text
X 데이터 : ['당신에게 드리는 마지막 혜택!', '내일 뵐 수 있을지 확인 부탁드...', '도연씨. 잘 지내시죠? 오랜만입...', '(광고) AI로 주가를 예측할 수 있다!']
y 데이터 : [1, 0, 0, 1]
```

---

### NumPy로 분리하기

NumPy의 슬라이싱을 사용해서도 `X`와 `y`를 분리할 수 있다.

```python
np_array = np.arange(0, 16).reshape((4, 4))

print('전체 데이터 :')
print(np_array)
```

출력:

```text
전체 데이터 :
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]]
```

마지막 열을 제외한 부분은 `X` 데이터에 저장하고, 마지막 열만 `y` 데이터에 저장한다.

```python
X = np_array[:, :3]
y = np_array[:, 3]

print('X 데이터 :')
print(X)
print('y 데이터 :', y)
```

출력:

```text
X 데이터 :
[[ 0  1  2]
 [ 4  5  6]
 [ 8  9 10]
 [12 13 14]]
y 데이터 : [ 3  7 11 15]
```

```text
X = np_array[:, :3]  → 모든 행에서 0열부터 2열까지 선택
y = np_array[:, 3]   → 모든 행에서 3열만 선택
```

---

### 테스트 데이터 분리하기

`X`와 `y`가 분리된 뒤에는 테스트 데이터를 따로 분리해야 한다.

테스트 데이터를 분리하는 방법은 크게 두 가지가 있다.

```text
테스트 데이터 분리 방법
├─ 사이킷런의 train_test_split() 사용
└─ 수동으로 분리
```

---

### 사이킷런으로 분리하기

사이킷런(scikit-learn)은 학습용 데이터와 테스트용 데이터를 쉽게 분리할 수 있도록 `train_test_split()`을 제공한다.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1234
)
```

각 인자의 의미는 다음과 같다.

| 인자 | 의미 |
|---|---|
| `X` | 독립 변수 데이터 |
| `y` | 종속 변수 데이터 또는 레이블 데이터 |
| `test_size` | 테스트용 데이터의 개수 또는 비율 |
| `train_size` | 학습용 데이터의 개수 또는 비율 |
| `random_state` | 난수 시드 |

`train_size`와 `test_size`는 둘 중 하나만 기재해도 된다.

<details>
<summary>random_state의 역할</summary>

`random_state`는 데이터를 섞는 방식이 매번 달라지지 않도록 고정하는 값이다.  
같은 `random_state` 값을 사용하면 실행할 때마다 같은 방식으로 데이터가 섞인다.

```text
random_state 고정 = 실습 결과 재현 가능
random_state 변경 = 다른 순서로 데이터 분리
```

</details>

---

### train_test_split() 예시

임의로 `X` 데이터와 `y` 데이터를 생성한다.

```python
X, y = np.arange(10).reshape((5, 2)), range(5)

print('X 전체 데이터 :')
print(X)
print('y 전체 데이터 :')
print(list(y))
```

출력:

```text
X 전체 데이터 :
[[0 1]
 [2 3]
 [4 5]
 [6 7]
 [8 9]]
y 전체 데이터 :
[0, 1, 2, 3, 4]
```

여기서는 7:3의 비율로 훈련 데이터와 테스트 데이터를 분리한다.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=1234
)
```

70% 비율로 분리된 `X` 훈련 데이터와 30% 비율로 분리된 `X` 테스트 데이터는 다음과 같다.

```python
print('X 훈련 데이터 :')
print(X_train)
print('X 테스트 데이터 :')
print(X_test)
```

출력:

```text
X 훈련 데이터 :
[[2 3]
 [4 5]
 [6 7]]
X 테스트 데이터 :
[[8 9]
 [0 1]]
```

`y` 훈련 데이터와 `y` 테스트 데이터는 다음과 같다.

```python
print('y 훈련 데이터 :')
print(y_train)
print('y 테스트 데이터 :')
print(y_test)
```

출력:

```text
y 훈련 데이터 :
[1, 2, 3]
y 테스트 데이터 :
[4, 0]
```

출력 결과를 보면 데이터를 단순히 앞과 뒤로 자른 것이 아니라, 데이터 순서가 섞인 뒤 분리된 것을 확인할 수 있다.

---

### random_state 값 변경

`random_state` 값을 변경하면 다른 순서로 섞인 데이터가 출력된다.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=1
)

print('y 훈련 데이터 :')
print(y_train)
print('y 테스트 데이터 :')
print(y_test)
```

출력:

```text
y 훈련 데이터 :
[4, 0, 3]
y 테스트 데이터 :
[2, 1]
```

다시 `random_state=1234`로 설정하면 이전과 동일한 결과가 나온다.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=1234
)

print('y 훈련 데이터 :')
print(y_train)
print('y 테스트 데이터 :')
print(y_test)
```

출력:

```text
y 훈련 데이터 :
[1, 2, 3]
y 테스트 데이터 :
[4, 0]
```

즉, `random_state` 값을 고정해두면 동일한 코드를 다음에도 재현할 수 있다.

---

### 수동으로 분리하기

데이터를 직접 앞부분과 뒷부분으로 나누어 훈련 데이터와 테스트 데이터를 만들 수도 있다.

먼저 임의로 `X` 데이터와 `y` 데이터를 생성한다.

```python
X, y = np.arange(0, 24).reshape((12, 2)), range(12)

print('X 전체 데이터 :')
print(X)
print('y 전체 데이터 :')
print(list(y))
```

출력:

```text
X 전체 데이터 :
[[ 0  1]
 [ 2  3]
 [ 4  5]
 [ 6  7]
 [ 8  9]
 [10 11]
 [12 13]
 [14 15]
 [16 17]
 [18 19]
 [20 21]
 [22 23]]
y 전체 데이터 :
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
```

훈련 데이터의 개수와 테스트 데이터의 개수를 정한다.

```python
num_of_train = int(len(X) * 0.8)
num_of_test = int(len(X) - num_of_train)

print('훈련 데이터의 크기 :', num_of_train)
print('테스트 데이터의 크기 :', num_of_test)
```

출력:

```text
훈련 데이터의 크기 : 9
테스트 데이터의 크기 : 3
```

아직 실제 데이터를 나눈 것이 아니라, 몇 개로 나눌지만 정한 상태이다.

---

### 수동 분리 시 주의할 점

`num_of_test`를 `len(X) * 0.2`로 계산하면 데이터 누락이 발생할 수 있다.

예를 들어 전체 데이터가 4,518개라고 하면 다음과 같다.

```text
4,518의 80% = 3,614.4 → 정수 변환 후 3,614
4,518의 20% = 903.6   → 정수 변환 후 903

3,614 + 903 = 4,517
```

이 경우 전체 4,518개 중 데이터 1개가 누락된다.

따라서 어느 한쪽을 먼저 계산하고, 전체 길이에서 그 값을 빼는 방식으로 계산하는 것이 좋다.

```text
num_of_train = int(len(X) * 0.8)
num_of_test = len(X) - num_of_train
```

---

### 수동으로 훈련 데이터와 테스트 데이터 나누기

앞에서 구한 `num_of_train`을 기준으로 훈련 데이터와 테스트 데이터를 분리한다.

```python
X_test = X[num_of_train:]
y_test = y[num_of_train:]

X_train = X[:num_of_train]
y_train = y[:num_of_train]
```

테스트 데이터를 출력하여 정상적으로 분리되었는지 확인한다.

```python
print('X 테스트 데이터 :')
print(X_test)
print('y 테스트 데이터 :')
print(list(y_test))
```

출력:

```text
X 테스트 데이터 :
[[18 19]
 [20 21]
 [22 23]]
y 테스트 데이터 :
[9, 10, 11]
```

수동 분리는 `train_test_split()`과 다르게 데이터가 섞이지 않은 채 어느 지점에서 앞과 뒤로 분리된다.  
따라서 수동으로 분리할 때는 데이터를 나누기 전에 직접 데이터를 섞는 과정이 필요할 수 있다.

---

## 최종 핵심 정리

```text
머신 러닝 워크플로우
= 데이터를 수집하고 전처리한 뒤 모델을 훈련, 평가, 배포하는 과정
```

```text
머신 러닝 워크플로우
├─ 수집
├─ 점검 및 탐색
├─ 전처리 및 정제
├─ 모델링 및 훈련
├─ 평가
└─ 배포
```

```text
데이터 분리
├─ 훈련용 데이터(Training Data) = 모델 학습
├─ 검증용 데이터(Validation Data) = 모델 성능 판단과 개선
└─ 테스트용 데이터(Test Data) = 모델 최종 성능 평가
```

```text
데이터 분석 필수 패키지
├─ Pandas = 데이터 처리
├─ NumPy = 수치 계산
└─ Matplotlib = 데이터 시각화
```

```text
Pandas
├─ Series = 값 + 인덱스
└─ DataFrame = 값 + 인덱스 + 열
```

```text
NumPy
├─ ndarray = 다차원 배열 자료구조
├─ shape = 배열의 크기
└─ np.dot() = 행렬곱
```

```text
Matplotlib
├─ plot() = 라인 플롯 생성
├─ xlabel() = x축 이름 설정
├─ ylabel() = y축 이름 설정
└─ legend() = 범례 삽입
```

```text
지도 학습 데이터 = 문제 데이터(X) + 정답 데이터(y)
```

```text
훈련 데이터
├─ X_train = 문제지 데이터
└─ y_train = 문제지에 대한 정답 데이터
```

```text
테스트 데이터
├─ X_test = 시험지 데이터
└─ y_test = 시험지에 대한 정답 데이터
```

```text
데이터 분리 방법
├─ X와 y 분리
│  ├─ zip 함수 사용
│  ├─ 데이터프레임 사용
│  └─ NumPy 사용
└─ 훈련/테스트 데이터 분리
   ├─ train_test_split() 사용
   └─ 수동 분리
```

```text
train_test_split()
├─ test_size = 테스트 데이터 비율 또는 개수
├─ train_size = 훈련 데이터 비율 또는 개수
└─ random_state = 동일한 결과를 재현하기 위한 난수 시드
```

```text
수동 분리 시 주의
num_of_train = int(len(X) * 0.8)
num_of_test = len(X) - num_of_train
```

```text
정확도(Accuracy)
= 모델이 예측한 답과 실제 정답을 비교해 얼마나 맞혔는지 나타내는 대표적인 평가 지표
```