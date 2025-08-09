---
layout: post
title: "알고리즘 복잡도 분석 - 시간 복잡도와 공간 복잡도"
date: 2025-08-06 12:00:00 +0900
last_modified_at: 2025-08-06 12:00:00 +0900
tags: [CS, Algorithm, Big-O, Time Complexity, Space Complexity, 시간복잡도, 공간복잡도, 빅오]
toc: true
---

## 📑 **Table Of Contents**

- [1. ⚙ 복잡도 분석의 개념](#complexity-basics)
  - [복잡도 분석의 필요성](#why-complexity-analysis)
  - [시간 복잡도와 공간 복잡도](#time-space-complexity)
- [2. ⚙ 시간 복잡도 (Time Complexity)](#time-complexity)
  - [Big-O 표기법](#big-o-notation)
  - [주요 시간 복잡도 유형](#time-complexity-types)
- [3. ⚙ 공간 복잡도 (Space Complexity)](#space-complexity)
  - [공간 복잡도의 구성 요소](#space-complexity-components)
  - [주요 공간 복잡도 예시](#space-complexity-examples)
- [4. ⚙ 시간 복잡도 vs 공간 복잡도](#time-vs-space)
  - [트레이드오프 관계](#tradeoff-relationship)
  - [피보나치 수열 예시](#fibonacci-example)
- [5. ⚙ 복잡도 분석 방법론](#analysis-methodology)
  - [복잡도 분석 기법](#analysis-techniques)
  - [실무 적용 가이드라인](#practical-guidelines)
- [6. ⚙ 실제 개발에서의 적용](#practical-applications)
- [7. 🏁 마치며](#conclusion)

---

![algorithm-complexity.png](/images/posts/2025-08-06-time-space-complexity/algorithm-complexity.png)

## 1. ⚙ 복잡도 분석의 개념 {#complexity-basics}

### 복잡도 분석의 필요성 {#why-complexity-analysis}

프로그래밍에서 알고리즘을 설계할 때 성능을 측정하는 가장 중요한 지표 중 하나가 바로 **복잡도(Complexity)**입니다. 복잡도는 크게 시간 복잡도와 공간 복잡도로 나누어지며, 이를 통해 알고리즘의 효율성을 객관적으로 평가할 수 있습니다.

### 시간 복잡도와 공간 복잡도 {#time-space-complexity}

- **시간 복잡도**: 입력 크기에 따라 알고리즘의 실행 시간이 어떻게 증가하는지를 나타냅니다.
- **공간 복잡도**: 알고리즘 실행 중 사용되는 메모리 공간이 입력 크기에 따라 어떻게 변하는지를 나타냅니다.

---

## 2. ⚙ 시간 복잡도 (Time Complexity) {#time-complexity}

시간 복잡도는 입력 크기에 따라 알고리즘이 실행되는 시간이 얼마나 증가하는지를 나타내는 지표입니다. 절대적인 실행 시간이 아닌, 입력 크기 n에 대한 상대적인 증가율을 의미합니다.

### Big-O 표기법 {#big-o-notation}

시간 복잡도는 주로 **Big-O 표기법**으로 표현됩니다. 이는 최악의 경우(worst case)에서의 성능을 나타냅니다.

### 주요 시간 복잡도 유형 {#time-complexity-types}

1. **O(1) - 상수 시간**
   - 입력 크기와 관계없이 일정한 시간
   - 예: 배열의 특정 인덱스 접근
   ```python
   def get_first_element(arr):
       return arr[0]  # O(1)
   ```

2. **O(log n) - 로그 시간**
   - 입력 크기가 증가해도 실행 시간은 로그적으로 증가
   - 예: 이진 탐색
   ```python
   def binary_search(arr, target):
       left, right = 0, len(arr) - 1
       while left <= right:
           mid = (left + right) // 2
           if arr[mid] == target:
               return mid
           elif arr[mid] < target:
               left = mid + 1
           else:
               right = mid - 1
       return -1  # O(log n)
   ```

3. **O(n) - 선형 시간**
   - 입력 크기에 비례하여 실행 시간 증가
   - 예: 배열 순회
   ```python
   def linear_search(arr, target):
       for i, value in enumerate(arr):
           if value == target:
               return i
       return -1  # O(n)
   ```

4. **O(n log n) - 선형 로그 시간**
   - 효율적인 정렬 알고리즘의 시간 복잡도
   - 예: 병합 정렬, 힙 정렬
   ```python
   def merge_sort(arr):
       if len(arr) <= 1:
           return arr
       mid = len(arr) // 2
       left = merge_sort(arr[:mid])
       right = merge_sort(arr[mid:])
       return merge(left, right)  # O(n log n)
   ```

5. **O(n²) - 제곱 시간**
   - 중첩 반복문이 있는 경우
   - 예: 버블 정렬, 선택 정렬
   ```python
   def bubble_sort(arr):
       n = len(arr)
       for i in range(n):
           for j in range(0, n - i - 1):
               if arr[j] > arr[j + 1]:
                   arr[j], arr[j + 1] = arr[j + 1], arr[j]  # O(n²)
   ```

6. **O(2^n) - 지수 시간**
   - 매우 비효율적인 알고리즘
   - 예: 재귀로 구현한 피보나치 수열
   ```python
   def fibonacci(n):
       if n <= 1:
           return n
       return fibonacci(n-1) + fibonacci(n-2)  # O(2^n)
   ```

## 3. ⚙ 공간 복잡도 (Space Complexity) {#space-complexity}

공간 복잡도는 알고리즘이 실행되는 동안 사용하는 메모리 공간의 양을 나타냅니다. 입력 크기에 따라 필요한 추가 메모리 공간이 얼마나 증가하는지를 측정합니다.

### 공간 복잡도의 구성 요소 {#space-complexity-components}

1. **고정 공간**: 알고리즘 자체가 사용하는 공간 (코드, 상수, 변수 등)
2. **가변 공간**: 입력 크기에 따라 달라지는 공간 (동적 할당, 재귀 호출 스택 등)

### 주요 공간 복잡도 예시 {#space-complexity-examples}

1. **O(1) - 상수 공간**
   ```python
   def swap(arr, i, j):
       temp = arr[i]  # 추가 변수 하나만 사용
       arr[i] = arr[j]
       arr[j] = temp  # O(1) 공간
   ```

2. **O(n) - 선형 공간**
   ```python
   def create_copy(arr):
       new_arr = []
       for item in arr:
           new_arr.append(item)  # 입력 크기만큼 새 배열 생성
       return new_arr  # O(n) 공간
   ```

3. **O(log n) - 로그 공간**
   ```python
   def binary_search_recursive(arr, target, left, right):
       if left > right:
           return -1
       mid = (left + right) // 2
       if arr[mid] == target:
           return mid
       elif arr[mid] < target:
           return binary_search_recursive(arr, target, mid + 1, right)
       else:
           return binary_search_recursive(arr, target, left, mid - 1)
       # 재귀 호출 깊이가 log n이므로 O(log n) 공간
   ```

## 4. ⚙ 시간 복잡도 vs 공간 복잡도 {#time-vs-space}

### 트레이드오프 관계 {#tradeoff-relationship}

알고리즘을 설계할 때는 시간과 공간 사이의 **트레이드오프(Trade-off)**를 고려해야 합니다.

### 피보나치 수열 예시 {#fibonacci-example}

1. **재귀 방식** - 시간: O(2^n), 공간: O(n)
2. **메모이제이션** - 시간: O(n), 공간: O(n)
3. **반복문** - 시간: O(n), 공간: O(1)

```python
# 1. 재귀 방식 (비효율적)
def fib_recursive(n):
    if n <= 1:
        return n
    return fib_recursive(n-1) + fib_recursive(n-2)

# 2. 메모이제이션 (시간 효율적, 공간 사용)
def fib_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]

# 3. 반복문 (시간, 공간 모두 효율적)
def fib_iterative(n):
    if n <= 1:
        return n
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    return curr
```

## 5. ⚙ 복잡도 분석 방법론 {#analysis-methodology}

### 복잡도 분석 기법 {#analysis-techniques}

1. **최악의 경우를 고려하라**: Big-O는 최악의 시나리오를 기준으로 합니다.

2. **상수 항은 무시하라**: O(2n)은 O(n)으로, O(n²/2)는 O(n²)로 표현합니다.

3. **가장 큰 항만 고려하라**: O(n² + n + 1)은 O(n²)입니다.

4. **중첩 루프를 주의하라**: 이중 반복문은 보통 O(n²)입니다.

5. **재귀의 깊이와 너비를 분석하라**: 재귀 호출의 패턴을 파악해야 합니다.

### 실무 적용 가이드라인 {#practical-guidelines}

앞서 언급한 분석 기법들은 다음과 같은 상황에서 특히 중요합니다:

1. **최악의 경우를 고려하라**: Big-O는 최악의 시나리오를 기준으로 합니다.
2. **상수 항은 무시하라**: O(2n)은 O(n)으로, O(n²/2)는 O(n²)로 표현합니다.
3. **가장 큰 항만 고려하라**: O(n² + n + 1)은 O(n²)입니다.
4. **중첩 루프를 주의하라**: 이중 반복문은 보통 O(n²)입니다.
5. **재귀의 깊이와 너비를 분석하라**: 재귀 호출의 패턴을 파악해야 합니다.

---

## 6. ⚙ 실제 개발에서의 적용 {#practical-applications}

복잡도 분석은 다음과 같은 상황에서 중요합니다:

- **대용량 데이터 처리**: 입력 크기가 클 때 성능 차이가 극명해집니다.
- **실시간 시스템**: 응답 시간이 중요한 시스템에서는 시간 복잡도가 핵심입니다.
- **메모리 제약 환경**: 임베디드 시스템에서는 공간 복잡도가 중요합니다.
- **코딩 테스트**: 알고리즘 문제 해결 시 필수적인 분석 도구입니다.

---

## 7. 🏁 마치며 {#conclusion}

시간 복잡도와 공간 복잡도는 효율적인 알고리즘을 설계하고 선택하는 데 필수적인 개념입니다. 단순히 동작하는 코드를 작성하는 것을 넘어서, 확장 가능하고 성능이 우수한 솔루션을 만들기 위해서는 복잡도 분석이 반드시 필요합니다.

각 상황에 맞는 최적의 알고리즘을 선택하고, 필요에 따라 시간과 공간 사이의 균형을 잘 맞추는 것이 좋은 개발자가 되는 길입니다.
