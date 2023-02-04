---
title:  "[M.L.] 📖 Introduction to Ensemble"
excerpt: "사공이 많으면 오히려 좋다?"

categories:
  - M_L_
tags:
  - [A.I., M.L., Ensemble, Bagging, Boosting]

toc: true
toc_sticky: true
 
date: 2023-01-20
last_modified_at: 2023-01-31
---

**⚠ 제 개인적인 해석과 견해, 즉, '뇌피셜'이 가득한 글이므로 주의해서 읽어주시길 바랍니다. ⚠**

## **🤝 Ensemble이란?**
 ```Ensemble(앙상블)```이란 ***다수의 Weak Model을 결합하여 Strong Model을 만들어내는 방법***을 말한다.<br><br>
 Model이 특정 데이터 셋을 학습하는 경우, model이 해당 데이터에 대해 overfitting(과적합)되거나 특정한 데이터 유형에 대해 잘못된 예측을 도출할 수 있다. 또한 model의 유형에 따라 특정 데이터 셋에 대한 뚜렷한 장단점을 가질 수 있다. 이러한 맥락에서 여러 model을 ***적절한 방법으로 결합***하였을 때, 더욱 좋은 성능을 가진 model을 얻을 수 있다는 가설이 등장하게 되었다.<br><br>
 ```Weak Model```이란 특정한 유형의 단일 model을 말하고, ```Strong Model```이란 다수의 Weak Model을 적절한 방법으로 결합하여 만들어낸 model을 말한다. 그리고 이렇게 Weak Model을 통해 Strong Model을 만들어내는 방법을 ```Ensemble```이라고 한다.<br><br>

## **🎯 Ensemble의 핵심**

### *Bias-Variance Tradeoff* ###
---
 ```Bias-Variance Tradeoff```는 supervised learning(지도 학습)을 통해 생성된 model의 예측 결과에 대한 ```bias(편향)```와 ```variance(분산)```가 ***반비례***하는 경향을 말하고, 이와 같은 경향 때문에 supervised learning을 통해 model을 학습시킬 때 model의 예측 값에 대한 bias와 variance의 균형을 맞추는 것이 매우 중요하다.<br><br>
 ```Bias```는 model의 특정 데이터에 대한 예측 값과 정답의 차이에 대한 평균을 말한다. Model의 예측 값은 bias가 낮을 수록 정답과 가까우며, 높을 수록 정답과 멀다. ```Variance```는 model의 특정 데이터에 대한 예측 값과 모든 예측 값에 대한 평균의 차이에 대한 제곱의 평균을 말한다. 즉, variance는 model이 예측한 값들의 분포를 나타낸다. Model의 예측 값은 variance가 낮을 수록 좁은 범위에 분포되어 있고, 높을 수록 넓은 범위에 분포되어 있다.<br><br>
 Model의 예측 값에 대한 bias가 너무 높으면 훈련 데이터 셋의 중요한 규칙성을 포함하지 못하는 underfitting(과소적합) 문제가 발생하고, variance가 너무 높으면 훈련 데이터 셋에 포함된 outlier와 잘못된 데이터의 영향을 받아 overfitting(과적합) 문제가 발생한다. ```Underfitting```의 경우 training 데이터 셋을 적절히 학습하지 못하여 model이 제대로 기능을 할 수 없는 경우를 말하고, ```Overfitting```의 경우 훈련 데이터 셋을 과도하게 학습하여 오히려 실제 데이터에 대한 예측의 오류가 커지는 경우를 말한다.<br><br>
 따라서 supervised learning을 통해 model을 학습시킬 때, underfitting과 overfitting을 모두 피할 수 있도록 예측 값에 대한 bias와 variance의 균형을 맞추는 방향으로 model을 학습시켜야 한다.<br><br>

### *Bias*와 *Variance*의 균형을 위한 접근 ###
---
 ```Ensemble```은 bias와 variance의 균형을 맞추기 위한 목적으로 사용된다. 
Ensemble의 종류에는 variance가 높은 weak model을 결합하여 variance를 낮추는 방법, bias가 높은 weak model을 결합하여 bias를 낮추는 방법 등이 있다.<br><br>

 ```Bagging(Bootstrap Aggregating)```, ```Boosting```은 Ensemble의 대표적인 두 가지 방식이다.

> ```Bagging(Bootstrap Aggregating)```<br>
 훈련 데이터 셋에 대한 복원 추출을 통해 다수의 훈련 데이터 셋을 만들고(***bootstrap***), 각 데이터 셋마다 하나의 model을 학습시켜 각 model의 예측 값에 대한 평균 또는 투표를 통해 최종 예측 값을 결정(***aggregate***)하는 결합 방식

> ```Boosting```<br>
 하나의 weak model을 시작으로, 이전 weak model의 잘못된 예측을 수정하는 방향으로 여러 weak model을 반복하여 만든 뒤, 이렇게 만들어진 다수의 weak model의 데이터 셋에 대한 예측을 바탕으로 최종 예측 값을 결정하는 결합 방식

 이 두 가지 방식 중 ```Bagging(Bootstrap Aggregating)```은 결합 과정에서 ***variance를 줄이는 방향***으로 작용하고, ```Boosting```은 주로 ***bias를 줄이는 방향***으로 작용한다.<br><br>

## **🔎 Ensemble Algorithm의 종류**

### *Bagging* ###
---
 ```Bagging```은 ***bootstrap과 aggregate 단계로 구성***되어 있으며, ```bootstrap``` 단계에서는 ***모집단으로부터 다수의 표본을 복원 추출***하고, ```aggregate``` 단계에서는 복원 추출된 표본을 학습한 ***각 model의 주어진 데이터에 대한 예측을 종합하여 최종적인 예측을 도출***해낸다. Bagging을 통해서 weak model은 예측에 대한 variance가 감소하는 방향으로 사용된다.<br><br>
 ```Bootstrap``` 단계에서는 모집단, 즉, 훈련 데이터 셋과 같은 크기를 가진 다수의 표본을 복원 추출을 통해 만들어 낸다. 이와 같은 방법을 통해 서로 다른 분포의 데이터 셋을 학습한 다수의 weak model이 만들어지게 되고, 이렇게 만들어진 다수의 weak model이 Aggregate 단계를 통해 결합되어 strong model이 생성된다. ```Aggregate``` 단계에서는 각 weak model의 예측 결과를 종합하여 strong model의 최종적인 예측을 만들어 낸다. Weak model의 예측을 종합할 때는 투표의 방식을 따르는 Voting, 예측 값의 평균을 계산하는 Averaging과 같은 다양한 방법이 사용된다.<br><br>
 다수의 표본을 이용하여 weak model을 학습시키고 그 결과를 종합하면 데이터 셋에 포함된 outlier의 영향을 덜 받을 수 있고, 이에 따라 strong model의 예측 값에 대한 variance는 weak model에 비해 감소하여 overfitting의 가능성을 낮출 수 있다.<br><br> 

### *Boosting* ###
---
 ```Boosting```은 ***다수의 weak model(stump)을 sequential한 방법으로 학습시켜 strong model을 만들어내는 결합 방법***을 말한다. 이 때, weak model은 바로 직전에 생성된 weak model에 영향을 받는 방법으로 생성되고, 영향을 받는 방법에 따라 훈련 데이터 셋의 분포에 변화를 주는 ```AdaBoost(Adaptive Boosting)```, pseudo residual을 통해 weak model 자체에 변화를 주는 ```Gradient Boosting```으로 그 방법이 나뉘게 된다. Boosting은 weak model의 예측에 대한 bias를 줄이는 방향으로 동작한다. 

## **👋 Summary**

 ```Ensemble(앙상블)```이란 ***다수의 Weak Model을 결합하여 Strong Model을 만들어내는 방법***을 말하고, ```Bias-Variance Tradeoff```로 인해 bias와 variance의 균형을 고려하여 model을 학습시켜야 한다. 대표적인 결합 방식으로는 ```Bagging```과 ```Boosting```이 있으며, bagging의 경우 weak model의 variance를 줄이는 방향, boosting의 경우 weak model의 bias를 줄이는 방향으로 작용한다.

## **🙏 Reference**

<https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205><br><br>
<https://ko.wikipedia.org/wiki/%ED%8E%B8%ED%96%A5-%EB%B6%84%EC%82%B0_%ED%8A%B8%EB%A0%88%EC%9D%B4%EB%93%9C%EC%98%A4%ED%94%84> 
 