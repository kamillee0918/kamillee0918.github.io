---
layout: post
title: "Gumbel AlphaZero 핵심 알고리즘 1: 행동 선택"
date: 2025-09-04 23:18:00 +0900
last_modified_at: 2025-09-04 23:18:00 +0900
tags: [AlphaZero, Gumbel, ReinforcementLearning, 일본장기, 튜닝]
math: true
toc: true
---

## 📑 **Table Of Contents**

- [1. ⚙ 행동 선택, 왜 중요할까?](#action-selection)
- [2. ⚙ Gumbel-Max 트릭: 똑똑하게 샘플링하기](#gumbel-max-trick)
  - [2.1. Gumbel 분포란?](#gumbel-distribution)
  - [2.2. Gumbel-Max 트릭의 동작 원리](#gumbel-max-principle)
  - [2.3. Gumbel-Softmax의 미분 가능성](#gumbel-softmax)
- [3. ⚙ Gumbel-Top-k: 최고의 후보 그룹 찾기](#gumbel-top-k)
  - [3.1. 카테고리컬 분포에서 Top-k 샘플링](#categorical-top-k)
  - [3.2. Gumbel-Top-k의 구현](#gumbel-top-k-implementation)
- [4. ⚙ 순차적 반감법: 한정된 자원의 효율적 배분](#sequential-halving)
  - [4.1. Multi-Armed Bandit 문제와의 연관성](#multi-armed-bandit)
  - [4.2. 순차적 반감법 알고리즘](#sequential-halving-algorithm)
  - [4.3. 각 페이즈별 시뮬레이션 배분](#simulation-allocation)
  - [4.4. 실제 구현 예시](#implementation-example)
- [5. ⚙ 루트 노드와 내부 노드의 전략적 차이](#root-vs-interior)
  - [5.1. 루트 노드: 신중한 탐험가](#root-node-selection)
  - [5.2. 내부 노드: 빠른 검증자](#interior-node-selection)
- [6. ⚙ Mctx 코드 심층 분석](#mctx-implementation)
  - [6.1. gumbel_muzero_root_action_selection](#root-action-selection)
  - [6.2. gumbel_muzero_interior_action_selection](#interior-action-selection)
- [7. 🏁 마치며](#conclusion)

---

![gumbel_02.png](/images/posts/2025-09-04-gumbel-alphazero-02/gumbel_02.png)

## 1. ⚙ 행동 선택, 왜 중요할까? {#action-selection}

`Gumbel AlphaZero`의 핵심적인 개선은 **탐색(Search)** 과정, 그중에서도 어떤 행동(Action)에 더 깊게 파고들지 결정하는 **행동 선택(Action Selection)** 알고리즘에 있습니다. 효율적인 행동 선택은 제한된 시간 안에 더 깊고 정확한 수읽기를 가능하게 하여 AI 성능을 극대화하는, 그야말로 AI의 '두뇌'와 같은 역할을 합니다.

[지난 포스트](/2025/09/02/gumbel-alphazero-01/)에서 살펴보았듯, 기존 `AlphaZero`는 PUCT 알고리즘을 사용했지만 시뮬레이션 횟수가 적을 때는 정책 개선을 이론적으로 보장하지 못하는 한계가 있었습니다. `Gumbel AlphaZero`는 이 문제를 해결하기 위해 탐색 트리의 시작점인 **루트 노드(Root Node)**와 탐색이 진행되는 **내부 노드(Interior Node)**에서 각기 다른 영리한 전략을 사용하여 탐색의 효율과 정확성을 모두 높입니다.

이번 포스트에서는 바로 이 행동 선택 알고리즘을 파헤쳐 보겠습니다.

## 2. ⚙ Gumbel-Max 트릭: 똑똑하게 샘플링하기 {#gumbel-max-trick}

이야기를 시작하기에 앞서, `Gumbel AlphaZero` 이름의 유래이기도 한 'Gumbel'에 대해 먼저 알아볼 필요가 있습니다. Gumbel 분포와 이를 활용한 Gumbel-Max 트릭은 행동 선택 과정의 근간을 이루는 중요한 아이디어입니다.

### 2.1. Gumbel 분포란? {#gumbel-distribution}

Gumbel 분포는 **극값 분포(Extreme Value Distribution)**의 한 종류로, 여러 값 중 최댓값이나 최솟값이 어떤 분포를 따르는지 모델링할 때 사용됩니다. 표준 Gumbel 분포는 아래와 같은 확률밀도함수(PDF)와 누적분포함수(CDF)를 가집니다.

- **확률밀도함수(PDF)**: $$f(x) = e^{-(x + e^{-x})}$$
- **누적분포함수(CDF)**: $$F(x) = e^{-e^{-x}}$$

Gumbel 분포의 가장 큰 특징은 0 근처에서 가장 높은 확률을 가지면서도, 오른쪽으로 긴 꼬리를 가져 때로는 예상치 못한 큰 값을 만들어낸다는 점입니다. 이러한 특성은 강화학습에서 **탐험(Exploration)**과 **활용(Exploitation)**의 균형을 맞추는 데 매우 유용합니다. 즉, 주로 좋은 행동을 선택하되(활용), 가끔은 예상 밖의 행동도 시도(탐험)하게 만드는 데 적합합니다.

### 2.2. Gumbel-Max 트릭의 동작 원리 {#gumbel-max-principle}

신경망이 여러 행동에 대한 확률(logits)을 출력했을 때, 그중 하나를 샘플링하는 가장 일반적인 방법은 Softmax를 사용하는 것입니다. 하지만 이 과정은 미분이 불가능하고 계산 비용이 발생한다는 단점이 있습니다.

**Gumbel-Max 트릭**은 이 문제를 해결합니다. 방법은 간단합니다.

1.  신경망이 출력한 각 행동의 **logits** 값에 **표준 Gumbel 분포에서 추출한 노이즈**를 더합니다.
2.  그 결과값들 중에서 **가장 큰 값을 가진 행동을 선택(argmax)**합니다.

```python
import jax.numpy as jnp
import jax.random as jrandom

def gumbel_max_sample(logits, rng_key):
    """Gumbel-Max 트릭을 사용한 샘플링"""
    # 1. Gumbel 분포에서 노이즈 생성
    gumbel_noise = jrandom.gumbel(rng_key, shape=logits.shape)
    
    # 2. 로짓에 Gumbel 노이즈를 더한 후 최댓값의 인덱스 선택
    perturbed_logits = logits + gumbel_noise
    return jnp.argmax(perturbed_logits)
```

이 간단한 방법은 Softmax 확률 분포에서 직접 샘플링하는 것과 수학적으로 동일한 결과를 제공합니다. 복잡한 정규화 과정 없이, 결정론적인 `argmax` 연산만으로 확률적 샘플링을 구현할 수 있다는 점에서 매우 효율적입니다.

### 2.3. Gumbel-Softmax의 미분 가능성 {#gumbel-softmax}

Gumbel-Max 트릭의 `argmax` 연산은 여전히 미분이 불가능하다는 한계가 있습니다. 만약 학습 과정에서 샘플링 과정 자체를 미분해야 한다면, `argmax`를 Softmax로 근사한 **Gumbel-Softmax** 기법을 사용할 수 있습니다.

```python
def gumbel_softmax(logits, temperature, rng_key):
    """미분 가능한 Gumbel-Softmax"""
    gumbel_noise = jrandom.gumbel(rng_key, shape=logits.shape)
    y = (logits + gumbel_noise) / temperature
    return jax.nn.softmax(y)
```

여기서 `temperature`는 분포의 뾰족한 정도를 조절하는 매개변수로, 값이 낮을수록 `argmax`와 유사한 원핫(one-hot) 벡터에 가까워집니다.

## 3. ⚙ Gumbel-Top-k: 최고의 후보 그룹 찾기 {#gumbel-top-k}

Gumbel-Max 트릭이 단 하나의 최선책을 뽑는 방법이었다면, **Gumbel-Top-k**는 한 걸음 더 나아가 **가장 유망한 후보 k개**를 한 번에 뽑는 방법입니다.

### 3.1. 카테고리컬 분포에서 Top-k 샘플링 {#categorical-top-k}

만약 여러 행동 후보 중 k개를 **하나씩 순서대로 뽑는다면(비복원 추출)** 과정이 매우 번거로울 것입니다. 첫 번째 샘플을 뽑고, 해당 행동의 확률을 0으로 만든 뒤 다시 정규화하고, 두 번째 샘플을 뽑는 과정을 k번 반복해야 하기 때문입니다.

### 3.2. Gumbel-Top-k의 구현 {#gumbel-top-k-implementation}

Gumbel-Top-k 트릭은 이 비효율적인 과정을 단 한 번의 연산으로 해결합니다. Gumbel-Max 트릭과 거의 동일하지만, `argmax` 대신 상위 k개의 인덱스를 가져온다는 점만 다릅니다.

```python
def gumbel_top_k(logits, k, rng_key):
    """Gumbel-Top-k 트릭을 사용한 상위 k개 선택"""
    # Gumbel 노이즈 생성
    gumbel_noise = jrandom.gumbel(rng_key, shape=logits.shape)
    
    # 로짓에 Gumbel 노이즈 추가
    perturbed_logits = logits + gumbel_noise
    
    # 상위 k개의 인덱스 선택
    _, top_k_indices = jax.lax.top_k(perturbed_logits, k)
    
    return top_k_indices
```

이 방법 역시 원래 확률 분포에서 k번의 비복원 추출을 수행한 것과 수학적으로 동일한 결과를 보장하면서도 훨씬 효율적입니다. `Gumbel AlphaZero`는 바로 이 Gumbel-Top-k를 사용해 탐색할 후보 행동들을 선정합니다.

## 4. ⚙ 순차적 반감법: 한정된 자원의 효율적 배분 {#sequential-halving}

Gumbel-Top-k로 유망한 후보들을 뽑았다면, 이제 한정된 시뮬레이션 자원을 이 후보들에게 어떻게 배분해야 가장 효율적일까요? `Gumbel AlphaZero`는 이 질문에 **순차적 반감법(Sequential Halving)**이라는 해법을 제시합니다.

### 4.1. Multi-Armed Bandit 문제와의 연관성 {#multi-armed-bandit}

순차적 반감법은 여러 슬롯머신 중 가장 돈을 많이 버는 머신을 찾는 **Multi-Armed Bandit** 문제에서 영감을 얻은 알고리즘입니다.

* 기존 AlphaZero의 **UCB1 알고리즘**은 탐색 과정 전체의 손실(누적 후회, Cumulative Regret)을 최소화하는 데 목적이 있습니다. 그래서 모든 행동에 대해 조심스럽게 시뮬레이션을 배분합니다.

* 반면 **순차적 반감법**은 최종적으로 최고의 행동 하나만 정확히 찾으면 되는 상황(단순 후회, Simple Regret 최소화)에 특화되어 있습니다. 마치 토너먼트처럼, 가능성이 낮은 후보를 과감하게 탈락시키고 유망한 후보에 자원을 집중하는 방식입니다.

### 4.2. 순차적 반감법 알고리즘 {#sequential-halving-algorithm}

순차적 반감법은 일종의 토너먼트처럼 동작합니다.

1) **초기 후보 설정**: Gumbel-Top-k로 선택된 m개의 행동을 후보로 설정합니다.

2) **균등 배분**: 현재 라운드에 남은 후보들에게 시뮬레이션을 균등하게 배분합니다.

3) **성과 평가**: 각 후보의 평균 보상(Q-value)을 계산합니다.

4) **하위 50% 제거**: 성과가 낮은 하위 절반의 후보를 탈락시킵니다.

5) **반복**: 최종 후보가 하나 남을 때까지 2~4 과정을 반복합니다.

### 4.3. 각 페이즈별 시뮬레이션 배분 {#simulation-allocation}

총 n번의 시뮬레이션을 m개의 후보에 배분하는 경우, 각 라운드(페이즈)별 시뮬레이션 횟수는 다음과 같이 계산됩니다.

- **총 페이즈 수**: $$\lceil\log_2(m)\rceil$$
- **각 페이즈의 후보당 시뮬레이션 수**: $$\frac{n}{\lceil \log_2(m) \rceil \times \text{현재 후보 수}}$$

이 방식을 통해 초반에는 넓게 탐색하고, 라운드가 진행될수록 유망한 후보에 시뮬레이션을 집중할 수 있습니다.

### 4.4. 실제 구현 예시 {#implementation-example}

Mctx 라이브러리에는 이러한 방문 계획을 미리 생성하는 함수가 구현되어 있습니다. 각 시뮬레이션 단계에서 어떤 행동 그룹을 탐색해야 할지 미리 계산해두는 것입니다.

```python
def get_sequence_of_considered_visits(max_num_considered_actions, num_simulations):
    """Sequential Halving의 방문 스케줄 생성"""
    if max_num_considered_actions <= 1:
        return tuple(range(num_simulations))
    
    log2max = int(math.ceil(math.log2(max_num_considered_actions)))
    sequence = []
    visits = [0] * max_num_considered_actions
    num_considered = max_num_considered_actions
    
    while len(sequence) < num_simulations:
        # 현재 페이즈에서 각 후보당 추가 방문 횟수
        num_extra_visits = max(1, int(num_simulations / (log2max * num_considered)))
        
        # ... (생략) ...
```

## 5. ⚙ 루트 노드와 내부 노드의 전략적 차이 {#root-vs-interior}

이제 `Gumbel AlphaZero`가 위에서 소개한 도구들을 어떻게 상황에 맞게 사용하는지 살펴보겠습니다. 핵심은 탐색의 시작점인 **루트 노드**와, 이미 탐색이 진행된 **내부 노드**에서의 전략이 다르다는 점입니다.

### 5.1. 루트 노드: 신중한 탐험가 {#root-node-selection}

루트 노드는 탐색의 방향을 결정하는 가장 중요한 지점입니다. 따라서 이곳에서는 **정책 개선**을 목표로 신중하고 체계적인 탐험을 수행합니다.

1) `Gumbel-Top-k`: 정책 네트워크의 예측에 Gumbel 노이즈를 더해 탐험할 k개의 후보를 선정합니다.

2) `순차적 반감법`: 선정된 후보들에게 시뮬레이션 예산을 체계적으로 배분합니다.

3) **최종 행동 결정**: 각 후보의 점수는 Gumbel 값 + 정책 예측(logits) + 정규화된 Q값을 합산하여 계산됩니다. 순차적 반감법 계획에 따라 현재 방문해야 할 후보들 중에서 이 점수가 가장 높은 행동을 선택합니다.

### 5.2. 내부 노드: 빠른 검증자 {#interior-node-selection}

일단 특정 경로를 따라 탐색을 시작한 내부 노드에서는 상황이 다릅니다. 여기서는 넓은 탐험보다 현재 경로의 가치를 빠르고 정확하게 평가하는 것이 중요합니다. 따라서 **결정론적인 방식**으로 행동을 선택합니다.

1) **개선된 정책 구성**: `사전 정책(prior_logits) + 현재까지의 Q값(completed_qvalues)`을 더해 현재 시점에서 가장 유망해 보이는 정책을 즉석에서 구성합니다.

2) **방문 비율 기반 선택**: 이 개선된 정책의 확률 분포와 현재까지의 방문 횟수 분포를 비교하여, 목표 확률에 비해 방문이 덜 된 행동을 우선적으로 선택합니다. 이를 통해 방문 횟수가 목표 정책 분포를 따라가도록 유도합니다.

## 6. ⚙ Mctx 코드 심층 분석 {#mctx-code-analysis}

이러한 전략들이 Mctx 라이브러리에서는 어떻게 구현되어 있는지 실제 함수를 통해 확인해 보겠습니다.

### 6.1. gumbel_muzero_root_action_selection {#gumbel-muzero-root-action-selection}

루트 노드의 행동 선택은 이 함수에서 이루어집니다. Gumbel 노이즈와 순차적 반감법 계획에 따라 다음에 방문할 행동을 결정합니다.

```python
def gumbel_muzero_root_action_selection(
    # ... (인수 생략) ...
) -> chex.Array:
    
    # ... (생략) ...
    
    # 순차적 반감법 테이블에서 현재 시뮬레이션에서 방문해야 할 횟수를 가져옴
    considered_visit = table[num_considered, simulation_index]
    
    # 저장된 Gumbel 노이즈를 사용
    gumbel = tree.extra_data.root_gumbel
    
    # 최종 스코어 계산 및 행동 선택
    to_argmax = seq_halving.score_considered(
        considered_visit, gumbel, prior_logits, completed_qvalues, visit_counts)
    
    return masked_argmax(to_argmax, tree.root_invalid_actions)
```

### 6.2. gumbel_muzero_interior_action_selection {#gumbel-muzero-interior-action-selection}

내부 노드의 행동 선택은 이 함수가 담당합니다. `prior + Q값`으로 개선된 정책을 만들고, 방문 횟수를 고려하여 결정론적으로 다음 행동을 선택합니다.

```python
def gumbel_muzero_interior_action_selection(
    # ... (인수 생략) ...
) -> chex.Array:

    # ... (생략) ...
    
    # 개선된 정책 구성: prior + Q값
    improved_policy_logits = prior_logits + completed_qvalues
    probs = jax.nn.softmax(improved_policy_logits)
    
    # 방문 빈도가 목표 분포에 근사하도록 스코어 계산
    to_argmax = _prepare_argmax_input(
        probs=probs, 
        visit_counts=visit_counts
    )
    
    # 최댓값 선택 (결정론적)
    return jnp.argmax(to_argmax, axis=-1).astype(jnp.int32)
```

### 7. 🏁 마치며 {#conclusion}

이번 포스트에서는 `Gumbel AlphaZero`의 심장이라 할 수 있는 행동 선택 알고리즘을 깊이 있게 살펴보았습니다.

주요 포인트 정리:

* **Gumbel-Top-k 트릭**으로 효율적인 후보 샘플링을 수행합니다.

* **순차적 반감법**을 통해 한정된 시뮬레이션 예산을 최적으로 배분합니다.

* **루트 노드**에서는 탐험을, **내부 노드**에서는 빠른 검증을 목표로 하는 이원화된 전략을 사용합니다.

이러한 정교한 기법들의 조합이 바로 `Gumbel AlphaZero`가 적은 시뮬레이션만으로도 높은 성능을 발휘하는 비결입니다.

다음 포스트에서는 이렇게 탐색한 결과를 바탕으로 신경망을 어떻게 더 똑똑하게 만드는지, 즉 **정책 학습(Policy Learning)** 방법에 대해 자세히 알아보겠습니다.