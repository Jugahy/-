# 빅데이터분석기사
😠 빅데이터분석기사 자격증을 따기위한 험난한 사투를 다루는 저장소입니다.

## 1️⃣ 기출 및 체험 문제 분석

* 제 1유형 : MinMaxScaler 또는 Z-Score 진행하고 pandas를 활용해서 특정 값을 찾는 문제가 주로 나왔습니다.
  > pandas를 활용해 특정 값을 복잡하게 찾는 문제에서 </br>데이터 전처리 기법을 적용하게 하고 pandas 활용은 쉬워지고 있습니다.
* 제 2유형 : Regressor 또는 Classifier을 진행합니다.
  > Regressor, Classifier에서 변한적이 없습니다.
* 제 3유형 : 검정(ttest, chi2_contigency, ...) / 회귀(OLS, Logit, ...)
  > 검정, 회귀를 주로 활용하고 잔차 이탈도, 로짓 우도값 찾기 같은 접해보지 않으면 어려운 문제 유형에서 </br>statistic, coef와 같은 대부분 접해봤을 값들을 찾는 문제로 변하고 있습니다.
  
| 문제   | 공식홈페이지 제공 문제 체험                                                         | 8회                                                                                         | 7회                                                                                              | 6회                                      |
|--------|-----------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|------------------------------------------|
| 제 1유형 | MinMax                                                                          | groupby, sort_values(ascending=False), MinMaxScaler, std(표준편차)                            | z-score, corr                                                                                    | 시간 데이터, groupby, 어려움              |
| 제 2유형 | Classifier(StandardScaler, LabelEncoder, roc_auc_score)                          | Regressor                                                                                   | Regressor                                                                                        | Classifier                               |
| 제 3유형 | chi2_contingency(statistic), Logit(coef), Logit(Odds ratio)                      | Logit(유의하지 않은 변수, coef, 5배 Odds ratio), OLS(p-value, R-squared, prediction)         | corr, OLS(p-value), Logit(Odd ratio, 잔차 이탈도, 로짓 우도값, test 데이터로 target 예측 후 오류율) | 직접 컬럼 만듬, chisquare, OLS(coef, p-value, prediction) |




</br>
</br>

>  제 2유형 절차
> 1. 데이터의 유형 파악 (info, describe)</br>
> 2. 데이터 전처리 (x, y, train, test set 분리 / 결측치 처리 / 수치형 변수 - StandardScaler, 범주형 변수 - LabelEncoder OneHotEncoder)
> 3. 데이터 분리 (train_test_split)
> 4. 모델 학습 및 검증 (RandomForest, LightGBM, ...)
> 5. 평가 (from sklearn.metrics import roc_auc_score, accuracy_score, ...)
> 6. 모델로 예측 결과 (to_csv)
> 7. 생성 결과 확인
