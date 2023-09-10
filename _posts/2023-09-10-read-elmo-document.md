---
layout: post
title: elmo 어필 문서를 읽은 후
date: 2023-09-10 18:16 +0800
last_modified_at: 2023-09-10 18:16 +0800
tags: [shogi, machine-learning]
toc: true
---

세계 컴퓨터 쇼기 선수권에서 우승한 elmo의 어필 문서를 읽었습니다만, 이해하기 쉽지 않았습니다.

> 승률이 이항 분포를 따른다면, 평가값은 로지스틱 분포를 따를 것(※1)이라고 판단하고 로지스틱 회귀를 적용하고 있습니다 (※2).

이 부분은 특정 상황에서의 승패가 평가 함수의 값에 따라 결정되는 확률 분포를 가정하고, 주어진 평가값에 대한 사후 확률을 로지스틱 함수로 표현한다는 것으로 이해됩니다.

만약 승패가 평가값에 의해 결정된다면, 이는 베르누이 시도에 따라 이항 분포로 나타납니다.

로지스틱 함수는 다음 식으로 주어지며, 값의 범위는 0부터 1까지입니다. 입력 값이 크면 1에 가까워지고, 입력 값이 작으면 0에 가까워집니다.

$$
\sigma(a)=\frac {1}{1+\exp(−a)}
$$

$\mathbf {a}$는 평가치이고, KPP 관계의 가중치를 $\mathbf {w}$로, 학습 데이터의 KPP 관계의 입력 벡터를 $\mathbf {x}$로 설정했을 때의 선형 합은 수식 입니다.

{% highlight text %}
※ 대부분의 쇼기 소프트웨어에서는 기물 보(歩) 하나의 가치를 대략 100점으로 설정하기 위해 x축을 스케일링하고 있습니다.
{% endhighlight %}

![시그모이드 함수](https://th.bing.com/th/id/OIP.q58sDjMhXtMPQPGxp2Qd9wHaES?pid=ImgDet&rs=1){: .align-center}

사람이 평가값이 1000을 넘어가면 거의 이긴 것으로 판단하는 감각과도 일치하기 때문에, 평가치에 따른 승률이 로지스틱 함수를 따른다고 가정하는 것은 납득할 수 있습니다.

국면의 승패 결과로부터, 기계 학습의 최대우도추정 방법을 사용하여, 손실 함수(교차 엔트로피 오차 함수)를 최소화하도록 학습함으로써, 평가 함수의 파라미터를 최적화할 수 있습니다.

교차 엔트로피 오차 함수는 아래 식으로 표현됩니다.

$$
L(\mathbf {w}, \mathbf {X}, \mathbf {t}) = − \sum_{i=1}^{N}(t_i \mathbf {w}^T \mathbf {x}_i − \log(1+\exp(\mathbf {w}^T \mathbf {x}_i)))
$$

$\mathbf {X}$는 학습 데이터의 집합$(\mathbf {x}_1, \mathbf {x}_2, ...)$, $\mathbf {t}$는 승패(0이나 1) 교사 데이터입니다.

교차 엔트로피 오차 함수를 파라미터 $\mathbf {w}$에 대해 편미분하면...

$$
\frac{\partial L(\mathbf {w},\mathbf {X},\mathbf {t})}{\partial \mathbf {w}} = \sum_{i=1}^{N}\mathbf {x}_i (\sigma (\mathbf {w}^T \mathbf {x}_i) − t_i)
$$