---
layout: post
title: elmo 어필 문서를 읽은 후
date: 2023-09-10
last_modified_at: 2025-04-18
tags: [일본장기, 머신러닝]
math: true
toc: true
githubRepo: mk-takizawa/elmo_for_learn
---

세계 컴퓨터 쇼기 선수권에서 우승한 elmo의 어필 문서를 읽었습니다만, 이해하기 쉽지 않았습니다.

> 승률이 이항 분포를 따른다면, 평가치는 로지스틱 분포를 따를 것[^1]이라고 판단하고 로지스틱 회귀를 적용하고 있습니다[^2].

이 부분은 특정 상황에서의 승패가 평가 함수의 값에 따라 결정되는 확률 분포를 가정하고, 주어진 평가치에 대한 사후 확률을 로지스틱 함수로 표현한다는 것으로 이해됩니다.

만약 승패가 평가치에 의해 결정된다면, 이는 베르누이 시도에 따라 이항 분포로 나타납니다.

로지스틱 함수는 다음 식으로 주어지며, 값의 범위는 0부터 1까지입니다. 입력 값이 크면 1에 가까워지고, 입력 값이 작으면 0에 가까워집니다.

$$
\sigma (a) = \frac{1}{1 + \exp(−a)}
$$

$\mathbf {a}$는 평가치이고, KPP 관계의 가중치를 $\mathbf{w}$로, 학습 데이터의 KPP 관계의 입력 벡터를 $\mathbf {x}$로 설정했을 때의 선형 합은 수식 입니다.

※ 대부분의 쇼기 소프트웨어에서는 보(歩) 기물 하나의 가치를 대략 100점으로 설정하기 위해 x축을 스케일링하고 있습니다.

![시그모이드 함수](https://th.bing.com/th/id/OIP.q58sDjMhXtMPQPGxp2Qd9wHaES?pid=ImgDet&rs=1)

평가치가 1000을 넘어가면 거의 이긴 것으로 판단하는 감각과도 일치하기 때문에, 평가치에 따른 승률이 로지스틱 함수를 따른다고 가정하는 것은 납득할 수 있습니다.

국면의 승패 결과로부터, 기계 학습의 최대우도추정 방법을 사용하여, 손실 함수(교차 엔트로피 오차 함수)를 최소화하도록 학습함으로써, 평가 함수의 파라미터를 최적화할 수 있습니다.

교차 엔트로피 오차 함수는 아래 식으로 표현됩니다.

$$
L(\mathbf {w}, \mathbf {X}, \mathbf {t}) = − \sum_{i=1}^{N}(t_i \mathbf {w}^T \mathbf {x}_i − \log(1+\exp(\mathbf {w}^T \mathbf {x}_i)))
$$

$\mathbf {X}$는 학습 데이터의 집합 $(\mathbf {x}_1, \mathbf {x}_2, \cdots)$, $\mathbf {t}$는 승패(0이나 1) 교사 데이터입니다.

교차 엔트로피 오차 함수를 파라미터 $\mathbf {w}$에 대해 편미분하면

$$
\frac{\partial L(\mathbf {w}, \mathbf {X}, \mathbf {t})}{\partial \mathbf {w}} = \sum_{i=1}^{N} \mathbf {x}_i (\sigma (\mathbf {w}^T \mathbf {x}_i) − t_i)
$$

와 같이 계산하기 쉬운 식이 됩니다.

어필 문서에서는 이에 정규화 항을 추가하고 있습니다.

> 단순히 최대우도추정의 로지스틱 회귀를 적용하는 것이 아니라, 깊은 탐색 결과를 얕은 탐색 결과에 피드백하는 방법[^3]을 정규화 항으로 사용하고 있습니다.

교차 엔트로피 오차 함수에 그 상황의 깊은 탐색 결과에 따른 평가치를 추가함으로써, 그 평가치에 가까워지는 정규화를 수행하고 있습니다.

$$
L'(\mathbf {w}, \mathbf {X}, \mathbf {t}) = L(\mathbf {w}, \mathbf {X}, \mathbf {t}) + \lambda R(\mathbf {X})
$$

$\lambda$는 정규화 계수이며, $R(\mathbf {x})$는 얕은 탐색에 의한 평가치와 깊은 탐색에 의한 평가치의 차이입니다.

더불어, 정규화 항에 다음 변형을 추가하고 있습니다.

> 정규화 항에는, 제4회 전왕전 토너먼트에서 †白美神†님이 사용하던 동일한 교차 엔트로피[^4]를 사용하고 있습니다. 이것은 단순히 로지스틱 회귀의 손실 항이 일반적으로 교차 엔트로피를 사용하기 때문에, 두 항의 순서를 맞추기 위한 목적으로 사용하고 있습니다. 계산이 간단하고 직관적으로 값이 이해하기 쉬운 점도 장점입니다.

즉, 아래의 식의 식이 됩니다.

$$
\begin {align}
L'(\mathbf {w}, \mathbf {X}, \mathbf {t}) & = L(\mathbf {w}, \mathbf {X}, \mathbf {t}) + \lambda H(\mathbf {p}, \mathbf {q}) \\
H(p, q) & = \sum_{t} p(t) \log q(t) \\
& = -p \log q - (1 - p) \log (1 - q) \\
\end {align}
$$

p는 평가치에서의 승률, q는 깊은 탐색의 평가치에서의 승률, H(p, q)는 두 확률 변수의 교차 엔트로피입니다.

H(p, q)의 편미분은 다음과 같습니다.

$$
\frac{\partial H(p,q)}{\partial \mathbf {w}} = \mathbf {x}_i(q − p)
$$

$H(p, q)$의 미분식의 전개 세부사항은 [인용된 자료](https://denou.jp/tournament2016/img/PR/Hakubishin.pdf)를 참조 바랍니다.

로지스틱 함수의 미분이 $\sigma'(a) = \sigma(a) \sigma(1 − a)$임을 활용하고 있습니다.

학습 데이터로 사용하는 것은 특정 국면에서 얕은 탐색 결과와 깊은 탐색 결과의 평가치 및 승패 데이터입니다.

1 에포크 분량의 파라미터를 업데이트 한 후에는 자체 대국을 통해 다시 학습 데이터를 생성해야 하므로, 여러 에포크 동안 학습을 반복하는 것은 상당히 어려워 보입니다.

elmo는 어필 문서에서 탐색 깊이 6, 약 50억 국면에서 단 한 번 최적화한 것으로 설명되어 있습니다. (대국 시에는 더 많이 반복했을 수도 있습니다.)

### 추가 내용

{% include embedding-github-cards.html id=page.githubRepo %}

위의 설명과 대응하는 부분은 아래와 같습니다.

#### src/usi.cpp (703~719행)

```cpp
const double eval_winrate = sigmoidWinningRate(eval);
const double teacher_winrate = sigmoidWinningRate(teacherEval);
const double t = ( (hcpe.isWin) ? 1.0 : 0.0 ); // mk-takizawa: 이기고 있으면 1, 지고 있으면 0

const double LAMBDA = 0.5; // mk-takizawa: 적당히 변경해주세요.
const double dsig = (eval_winrate -t) + LAMBDA * (eval_winrate - teacher_winrate);
```

`eval_winrate`는 얕은 탐색의 평가치에 의한 예측 승률이고, `teacher_winrate`는 깊은 탐색의 평가치에 의한 예측 승률이며, `t`는 승패 데이터입니다. `LAMBDA`는 정규화 계수로, 0.5가 사용되었습니다.

`dsig`에는 위에서 설명한 "승패의 교차 엔트로피 오차의 기울기" + "정규화 계수" × "정규화 항의 기울기"를 대입하고 있습니다.

이후 처리에서 `dsig`에 학습률을 곱한 값으로, 입력 벡터의 요소 중 값이 있는 항목의 매개변수를 업데이트합니다.

또한, 원래 Apery 소스 코드에서는 얕은 탐색과 깊은 탐색의 평가치에 의한 예측 승률의 제곱 오차가 손실 함수로 사용되었습니다.

[^1]: [GPS将棋](https://ipsj.ixsq.nii.ac.jp/ej/?action=pages_view_main&active_action=repository_view_main_item_detail&item_id=9786&item_no=1&page_id=13&block_id=8)는 이 점을 직접 평가하지 않아 적절히 설정되지 않았다고 생각됩니다.

[^2]: WCSC26의 쇼기 소프트 "激指"을 참고하고 있습니다. 또한, Ponanza가 로지스틱 회귀를 사용하고 있다는 기록이 있습니다(더 이상 사용하지 않을 수도 있습니다).

[^3]: Apery, YaneuraOu 등 많은 곳에서 채택되었습니다. NDF의 방법을 단순화하여 현재의 형태가 된 것은 Ponanza에서 시작된 것으로 이해하고 있습니다.

[^4]: Apery의 제곱 오차에 비교하면, 경사가 더욱 완만하게 작아지는 것 같습니다. 그 부분에서도 좋은 인상을 가지고 있습니다.
