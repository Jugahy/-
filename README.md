# 빅데이터분석기사
😠 빅데이터분석기사 자격증을 따기위한 험난한 사투를 벌이며 함께 스터디하는 저장소입니다.

</br>

### Table of contents 

1. [기출 및 체험 문제 분석](#1️⃣-기출-및-체험-문제-분석)
2. [유형별 공략](#2️⃣-유형별-공략)
3. [9회 실기 후기](#3️⃣-9회-실기-후기)

</br>

## 1️⃣ 기출 및 체험 문제 분석

### 1-1. 유형 요약
* 제 1유형 : MinMaxScaler 또는 Z-Score 진행하고 pandas를 활용해서 특정 값을 찾는 문제가 주로 나왔습니다.
  > pandas를 활용해 특정 값을 복잡하게 찾는 문제에서 </br>데이터 전처리 기법을 적용하게 하고 pandas 활용은 쉬워지고 있습니다.
* 제 2유형 : Regressor 또는 Classifier을 진행합니다.
  > Regressor, Classifier에서 변한적이 없습니다.
* 제 3유형 : 검정(ttest, chi2_contigency, ...) / 회귀(OLS, Logit, ...)
  > 검정, 회귀를 주로 활용하고 잔차 이탈도, 로짓 우도값 찾기 같은 접해보지 않으면 어려운 문제 유형에서 </br>statistic, coef와 같은 대부분 접해봤을 값들을 찾는 문제로 변하고 있습니다.

</br>

### 1-2. 체험, 8, 7, 6회 기출 키워드
| 문제   | [공식홈페이지 제공 문제 체험](https://github.com/Jugahy/Big_Data_Analyzer/blob/main/%EB%B9%85%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B6%84%EC%84%9D%EA%B8%B0%EC%82%AC%20%EC%8B%A4%EA%B8%B0%20%EC%B2%B4%ED%97%98.ipynb)                                                         | [8회 기출 변형](https://github.com/Jugahy/Big_Data_Analyzer/blob/main/%EA%B8%B0%EC%B6%9C/8%ED%9A%8C%20%EA%B8%B0%EC%B6%9C%20%EB%B3%80%ED%98%95.ipynb)                                                                                         | [7회 기출 변형](https://github.com/Jugahy/Big_Data_Analyzer/blob/main/%EA%B8%B0%EC%B6%9C/7%ED%9A%8C%20%EA%B8%B0%EC%B6%9C%20%EB%B3%80%ED%98%95.ipynb)                                                                                              | [6회 기출 변형](https://github.com/Jugahy/Big_Data_Analyzer/blob/main/%EA%B8%B0%EC%B6%9C/6%ED%9A%8C%20%EA%B8%B0%EC%B6%9C%20%EB%B3%80%ED%98%95.ipynb)                                      |
|--------|-----------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|------------------------------------------|
| 제 1유형 | MinMax                                                                          | groupby, sort_values(ascending=False), MinMaxScaler, std(표준편차)                            | z-score, corr                                                                                    | 시간 데이터, groupby, 어려움              |
| 제 2유형 | Classifier(StandardScaler, LabelEncoder, roc_auc_score)                          | Regressor                                                                                   | Regressor                                                                                        | Classifier                               |
| 제 3유형 | chi2_contingency(statistic), Logit(coef), Logit(Odds ratio)                      | Logit(유의하지 않은 변수, coef, 5배 Odds ratio), OLS(p-value, R-squared, prediction)         | corr, OLS(p-value), Logit(Odd ratio, 잔차 이탈도, 로짓 우도값, test 데이터로 target 예측 후 오류율) | 직접 컬럼 만듬, chisquare, OLS(coef, p-value, prediction) |

</br>

## 2️⃣ 유형별 공략

### 제 1유형
* MinMaxScaler, Z-Score 코드 암기
* pandas를 통한 특정 값 추출하는 예제 많이 풀기


### 제 2유형
* 데이터의 유형 파악 (info, describe)</br>
* 데이터 전처리 (x, y, train, test set 분리 / 결측치 처리 / 수치형 변수 - StandardScaler, 범주형 변수 - LabelEncoder OneHotEncoder)
* 데이터 분리 (train_test_split)
* 모델 학습 및 검증 (RandomForest, LightGBM, ...)
* 평가 (from sklearn.metrics import roc_auc_score, accuracy_score, ...)
* 모델로 예측 결과 (to_csv)
* 생성 결과 확인


### 제 3유형
* 검정과 회귀 부분 이 두 가지로 코드 스타일이 달라지는 둘 다 숙지
* summary를 통해 추출되는 값의 의미 알기
* 3유형의 준비가 부족하여 추가적으로 스터디 진행
  * [빅분기 실기 3유형 1](https://github.com/Jugahy/Big_Data_Analyzer/blob/main/%EC%A0%9C%203%EC%9C%A0%ED%98%95%20%EC%A0%95%EB%A6%AC/%EB%B9%85%EB%B6%84%EA%B8%B0%20%EC%8B%A4%EA%B8%B0%203%EC%9C%A0%ED%98%95_1.ipynb)
  * [빅분기 실기 3유형 2](https://github.com/Jugahy/Big_Data_Analyzer/blob/main/%EC%A0%9C%203%EC%9C%A0%ED%98%95%20%EC%A0%95%EB%A6%AC/%EB%B9%85%EB%B6%84%EA%B8%B0%20%EC%8B%A4%EA%B8%B0%203%EC%9C%A0%ED%98%95_2.ipynb)
  * [빅분기 실기 3유형 알짜베기](https://github.com/Jugahy/Big_Data_Analyzer/blob/main/%EC%A0%9C%203%EC%9C%A0%ED%98%95%20%EC%A0%95%EB%A6%AC/%EB%B9%85%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B6%84%EC%84%9D%EA%B8%B0%EC%82%AC%20%EC%8B%A4%EA%B8%B0%203%EC%9C%A0%ED%98%95%20%EC%95%8C%EC%A7%9C%EB%B2%A0%EA%B8%B0.ipynb)

</br>

## 3️⃣ 9회 실기 후기
### 제 1유형
> 7, 8회 기출을 풀어봤을 때 1유형이 쉬워지는 경향이 있어서 쉬울거라 예측했지만 나름 고전했습니다.</br>
MinMaxScaler, StandardScaler와 같은 데이터 전처리는 나오지 않았고 특정 데이터를 찾는 문제로 3문제 나왔습니다.

**필요 기술 Keyword**
- 기본적인 pandas 문법
- groupby
- agg


### 제 2유형
> 기존 기출과 동일하게 분류를 진행하면 됐고 결측치는 존재하지 않았지만 </br>
수치형, 범주형 변수 스케일링은 필요했습니다. 성능 평가 지표로는 macro f1 score를 사용했습니다.

**필요 기술 Keyword**
- Classifier
- StandardScaler
- LabelEncoder
- get_dummies


### 제 3유형
> 3유형 8회 3유형에 비해 문제를 조금 꼬았지만 기본 코드를 알면 충분히 대처할 수 있었습니다.</br>
검정 문제는 나오지 않았고 회귀 문제만 나왔습니다.

**필요 기술 Keyword**
- 다중선형회귀
- 로지스틱회귀
- 유의마한 변수 찾기(p-value)
- 피어슨 상관계수
- train, test 데이터로 분리 후 test 데이터 모델 평가 후 rmse 평가
- odd 계산

