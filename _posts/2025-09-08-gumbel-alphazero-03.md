---
layout: post
title: "Gumbel AlphaZero 핵심 알고리즘 2: 정책 학습"
date: 2025-09-08 23:45:00 +0900
last_modified_at: 2025-09-08 23:45:00 +0900
tags: [AlphaZero, Gumbel, ReinforcementLearning, 일본장기, 튜닝]
math: true
toc: true
---

## 📑 **Table Of Contents**

- [1. ⚙ 정책 학습이란?](#policy-learning)
- [2. ⚙ 완성된 Q값 (Completed Q-values)](#completed-q-values)
- [3. ⚙ 학습 목표 설정](#target-policy)
- [4. ⚙ 코드 심층 분석](#code-analysis)
- [5. 🏁 마치며](#conclusion)

---

![gumbel_03.png](/images/posts/2025-09-08-gumbel-alphazero-03/gumbel_03.png)

## 1. ⚙ 정책 학습이란? {#policy-learning}

`Gumbel AlphaZero`에서 **정책 학습(Policy Learning)**은 탐색을 통해 얻은 더 나은 행동 정보를 신경망에 반영하여 다음 탐색에서 처음부터 더 유망한 수를 예측할 수 있도록 만드는 과정입니다.

### 1.1. 자기 대국 학습의 핵심

`AlphaZero` 계열의 알고리즘은 **자기 대국(Self-play)** 학습을 통해 스스로 강해집니다:

1. **탐색**: 현재 정책 네트워크를 사용하여 MCTS 탐색 수행
2. **대국**: 탐색 결과로 얻은 행동으로 게임 진행  
3. **학습**: 대국 결과를 바탕으로 정책 네트워크 업데이트
4. **반복**: 개선된 네트워크로 다시 탐색 시작

이 순환 과정에서 **정책 학습**은 3단계에 해당하며, 탐색에서 발견한 '더 좋은 수'를 네트워크가 기억하도록 만드는 핵심 역할을 합니다.

### 1.2. 기존 AlphaZero와의 차이점

**기존 AlphaZero**:
- 탐색 후 루트 노드의 **방문 횟수 분포**를 목표 정책으로 사용
- 방문하지 않은 행동에 대한 정보 부족
- 시뮬레이션 횟수가 적을 때 정책 개선 보장 불가

**Gumbel AlphaZero**:
- **완성된 Q값(Completed Q-values)**을 활용한 더 정확한 목표 정책 생성
- 모든 행동(방문한 행동과 방문하지 않은 행동)에 대한 완전한 가치 정보 활용
- 이론적으로 보장된 정책 개선

### 1.3. 정책 개선의 이론적 보장

`Gumbel AlphaZero`의 강력함은 **적은 시뮬레이션만으로도 정책이 개선될 것임을 이론적으로 보장**한다는 점입니다. 이는 `Gumbel-Top-k` 트릭과 `순차적 반감법`을 통해 얻은 탐색 결과가 수학적으로 검증된 방식으로 정책 학습에 활용되기 때문입니다.

## 2. ⚙ 완성된 Q값 (Completed Q-values) {#completed-q-values}

`Gumbel AlphaZero`의 정책 학습에서 가장 핵심적인 개념은 **완성된 Q값(Completed Q-values)**입니다. 이는 탐색 과정에서 방문하지 않은 행동들의 Q값을 추정하여, 모든 행동에 대해 완전한 가치 정보를 제공하는 혁신적인 방법입니다.

### 2.1. 완성된 Q값이 필요한 이유

탐색 과정에서는 `Gumbel-Top-k`와 `순차적 반감법`을 통해 일부 유망한 행동들만 시뮬레이션됩니다. 따라서 시뮬레이션되지 않은 행동들은 정확한 가치(Q-value)를 알 수 없습니다.

하지만 정책 학습을 위해서는 **모든 행동**에 대한 가치 정보가 필요합니다. 완성된 Q값은 이 문제를 해결하기 위해 다음과 같은 방식으로 동작합니다:

1. **방문한 행동**: 실제 시뮬레이션을 통해 얻은 정확한 Q값 사용
2. **방문하지 않은 행동**: 혼합 가치(Mixed Value)로 Q값을 추정하여 보완

### 2.2. 혼합 가치 (Mixed Value) 계산

혼합 가치는 다음 두 요소를 결합하여 계산됩니다:

$$\text{Mixed Value} = \frac{\text{raw_value} + \sum_{\text{visited}} N(a) \cdot \text{weighted_q}}{\sum_{\text{visited}} N(a) + 1}$$

여기서:
- **raw_value**: 가치 네트워크가 예측한 현재 상태의 가치
- **weighted_q**: 방문한 행동들의 Q값을 사전 확률로 가중평균한 값
- **N(a)**: 각 행동의 방문 횟수

이 공식은 가치 네트워크의 예측과 실제 탐색 결과를 균형있게 결합하여, 더 정확한 가치 추정을 제공합니다.

### 2.3. Q값 보완 과정

완성된 Q값 생성 과정:

```python
def complete_qvalues(qvalues, visit_counts, mixed_value):
    """방문하지 않은 행동의 Q값을 혼합 가치로 보완"""
    completed_qvalues = jnp.where(
        visit_counts > 0,  # 방문한 행동인가?
        qvalues,           # 실제 Q값 사용
        mixed_value        # 혼합 가치로 보완
    )
    return completed_qvalues
```

### 2.4. Q값 정규화 및 스케일링

완성된 Q값은 다음 단계를 거쳐 최종 처리됩니다:

1. **Min-Max 정규화**: 모든 Q값을 [0, 1] 범위로 정규화
2. **방문 횟수 기반 스케일링**: 탐색이 많이 진행될수록 Q값의 영향력을 증가

$$\text{Final Q-values} = (c_{\text{visit}} + \max_a N(a)) \cdot c_{\text{scale}} \cdot \text{normalized_q}$$

여기서 $$c_{\text{visit}} = 50.0$$, $$c_{\text{scale}} = 0.1$$은 하이퍼파라미터입니다.

## 3. ⚙ 학습 목표 설정 {#target-policy}

완성된 Q값을 얻었다면, 이제 이를 바탕으로 정책 네트워크를 학습시켜야 합니다. `Gumbel AlphaZero`는 완성된 Q값을 직접 목표 정책으로 변환하여 사용합니다.

### 3.1. 목표 정책 생성

완성된 Q값으로부터 목표 정책을 생성하는 과정:

```python
def create_target_policy(old_logits, completed_qvalues, visit_counts, *, c_visit=50.0, c_scale=0.1, temperature=1.0):
    """logits + σ(completedQ) → softmax로 목표 정책 생성"""
    max_visits = jnp.max(visit_counts, axis=-1)
    sigma = (c_visit + max_visits) * c_scale * completed_qvalues
    target_logits = old_logits + sigma

    # temperature를 사용해 부드러운 확률 분포 유지
    return jax.nn.softmax(target_logits / temperature)
```

수식으로는:

$$\sigma(\hat q(a)) = (c_{\text{visit}} + \max_b N(b)) \cdot c_{\text{scale}} \cdot \hat q(a) \quad\Rightarrow\quad \pi_{\text{target}}(a) = \mathrm{softmax}\left(\text{logits}(a) + \sigma(\text{completedQ}(a))\right)$$

### 3.2. 정책 학습 손실 함수

정책 네트워크는 목표 정책 $$\pi_{\text{target}}$$과 현재 정책 $$\pi_{\theta}$$ 사이의 **교차엔트로피 손실(Cross-Entropy Loss)**을 최소화하도록 학습됩니다. 이는 KL divergence를 최소화하는 것과 **이론적으로 동치**입니다.

$$\mathcal{L}_{\text{policy}} = -\sum_{a} \pi_{\text{target}}(a) \log \pi_{\theta}(a)$$

### 3.3. 기존 AlphaZero와의 차이점

| 구분 | AlphaZero | Gumbel AlphaZero |
|------|-----------|------------------|
| 목표 정책 | 방문 횟수 분포 | 완성된 Q값 분포 |
| 미방문 행동 | 정보 없음 | 혼합 가치로 추정 |
| 이론적 보장 | 제한적 | 정책 개선 보장 |

### 3.4. 정책 개선의 이론적 보장

`Gumbel AlphaZero`의 핵심 장점은 **이론적으로 정책 개선이 보장**된다는 점입니다. 이는 다음 조건들이 만족될 때 성립합니다:

1. **Gumbel-Top-k 샘플링**: 올바른 확률 분포에서 후보 선택
2. **순차적 반감법**: 효율적인 시뮬레이션 예산 배분
3. **완성된 Q값**: 모든 행동에 대한 완전한 가치 정보

이러한 조건들이 만족되면, 새로운 정책은 이전 정책보다 항상 더 나은 성능을 보장받습니다.

## 4. ⚙ 코드 심층 분석 {#code-analysis}

이제 `Gumbel AlphaZero`의 정책 학습이 실제 코드에서 어떻게 구현되어 있는지 자세히 살펴보겠습니다. 핵심 함수들을 중심으로 단계별로 분석해보겠습니다.

### 4.1. qtransform_completed_by_mix_value 함수

완성된 Q값 생성의 핵심 함수입니다:

```python
def qtransform_completed_by_mix_value(
    tree: tree_lib.Tree,
    node_index: chex.Numeric,
    *,
    value_scale: chex.Numeric = 0.1,
    maxvisit_init: chex.Numeric = 50.0,
    rescale_values: bool = True,
    use_mixed_value: bool = True,
    epsilon: chex.Numeric = 1e-8,
) -> chex.Array:
    # 1. 기본 정보 수집
    qvalues = tree.qvalues(node_index)
    visit_counts = tree.children_visits[node_index]
    raw_value = tree.raw_values[node_index]
    prior_probs = jax.nn.softmax(tree.children_prior_logits[node_index])
    
    # 2. 혼합 가치 계산
    if use_mixed_value:
        value = _compute_mixed_value(
            raw_value, qvalues=qvalues, 
            visit_counts=visit_counts, prior_probs=prior_probs)
    else:
        value = raw_value
    
    # 3. Q값 보완
    completed_qvalues = _complete_qvalues(
        qvalues, visit_counts=visit_counts, value=value)
    
    # 4. 정규화 및 스케일링
    if rescale_values:
        completed_qvalues = _rescale_qvalues(completed_qvalues, epsilon)
    
    maxvisit = jnp.max(visit_counts, axis=-1)
    visit_scale = maxvisit_init + maxvisit
    return visit_scale * value_scale * completed_qvalues
```

### 4.2. _compute_mixed_value 함수

혼합 가치 계산의 핵심 로직:

```python
def _compute_mixed_value(raw_value, qvalues, visit_counts, prior_probs):
    """가치 네트워크 예측과 탐색 결과를 결합한 혼합 가치 계산"""
    sum_visit_counts = jnp.sum(visit_counts, axis=-1)
    
    # 방문한 행동들의 사전 확률 합계
    sum_probs = jnp.sum(jnp.where(visit_counts > 0, prior_probs, 0.0), axis=-1)
    
    # 방문한 행동들의 Q값을 사전 확률로 가중평균
    weighted_q = jnp.sum(jnp.where(
        visit_counts > 0,
        prior_probs * qvalues / jnp.where(visit_counts > 0, sum_probs, 1.0),
        0.0), axis=-1)
    
    # 가치 네트워크 예측과 탐색 결과를 결합
    return (raw_value + sum_visit_counts * weighted_q) / (sum_visit_counts + 1)
```

### 4.3. Tree.qvalues 함수

각 행동의 Q값을 계산하는 함수:

```python
def _unbatched_qvalues(tree: Tree, index: int) -> chex.Array:
    """벨만 방정식에 따른 Q값 계산: Q(s,a) = R(s,a) + γ * V(s')"""
    return (
        tree.children_rewards[index] +  # 즉시 보상
        tree.children_discounts[index] * tree.children_values[index]  # 할인된 미래 가치
    )
```

### 4.4. 정책 학습 통합 과정

완성된 Q값을 사용한 정책 학습의 전체 흐름:

```python
def policy_learning_step(tree, node_index):
    """정책 학습의 전체 과정"""
    # 1. 완성된 Q값 계산
    completed_qvalues = qtransform_completed_by_mix_value(tree, node_index)
    
    # 2. 목표 정책 생성
    target_policy = jax.nn.softmax(completed_qvalues)
    
    # 3. 정책 네트워크 손실 계산
    current_policy = policy_network(state)
    policy_loss = -jnp.sum(target_policy * jnp.log(current_policy + 1e-8))
    
    return policy_loss, target_policy
```

### 4.5. 핵심 구현 특징

1. **효율적인 배치 처리**: `jax.vmap`을 활용한 벡터화 연산
2. **수치적 안정성**: 0으로 나누기 방지 및 로그 연산 안정화
3. **하이퍼파라미터 조정**: `value_scale`, `maxvisit_init` 등을 통한 세밀한 제어
4. **조건부 처리**: `use_mixed_value`, `rescale_values` 플래그를 통한 유연한 설정

## 5. 🏁 마치며 {#conclusion}

이번 포스트에서는 `Gumbel AlphaZero`의 정책 학습 메커니즘을 심층적으로 살펴보았습니다. 특히 **완성된 Q값(Completed Q-values)**이라는 혁신적인 개념이 어떻게 기존 `AlphaZero`의 한계를 극복하고, 이론적으로 보장된 정책 개선을 가능하게 하는지 알아보았습니다.

### 주요 포인트 정리:

1. **완성된 Q값**: 방문하지 않은 행동의 가치를 혼합 가치로 추정하여 모든 행동에 대한 완전한 정보 제공

2. **혼합 가치**: 가치 네트워크 예측과 탐색 결과를 균형있게 결합한 더 정확한 가치 추정

3. **이론적 보장**: `Gumbel-Top-k`와 순차적 반감법을 통한 정책 개선의 수학적 보장

4. **효율적 구현**: JAX 기반의 벡터화 연산과 수치적 안정성을 고려한 실제 구현

`Gumbel AlphaZero`의 정책 학습은 단순히 탐색 결과를 모방하는 것이 아니라, 탐색을 통해 얻은 지식을 체계적으로 정제하고 일반화하여 신경망에 전달하는 정교한 과정입니다. 이러한 접근 방식이 바로 `Gumbel AlphaZero`가 적은 시뮬레이션만으로도 높은 성능을 달성할 수 있는 핵심 비결입니다.

다음 포스트에서는 지금까지 다룬 모든 알고리즘들이 실제 구현에서 어떻게 통합되어 동작하는지, 전체적인 아키텍처와 구현 세부사항을 중심으로 살펴보겠습니다.
