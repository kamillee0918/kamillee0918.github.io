---
layout: post
title: 컴퓨터 아키텍처
date: 2025-04-28
last_modified_at: 2025-04-28
tags: [CS]
toc: true
---

## 📑 **Table Of Contents**

- [1. ⚙ CPU](#cpu)
  - [PC (Program Counter)](#pc-program-counter)
  - [레지스터 (Register)](#register)
  - [상태 코드 (Condition Code)](#condition-code)
- [2. ⚙ 컴파일 과정 (Compilation Process)](#compilation-process)
  - [전처리 (Preprocessing)](#preprocessing)
  - [컴파일 (Compilation)](#compilation)
  - [어셈블 (Assembly)](#assembly)
  - [링크 (Linking)](#linking)
- [3. ⚙ 어셈블리 언어와 명령어 실행](#assembly-language-and-instruction-execution)
  - [명령어 단위 실행 (Instruction-Level Execution)](#instruction-level-execution)
  - [자바 바이트코드 (Java Bytecode)](#java-bytecode)
- [4. 🏁 마치며](#conclusion)

이번시간에는 컴퓨터 아키텍처에서 CPU의 주요 구성 요소 프로그램 카운터(PC), 레지스터, 상태 코드를 살펴보도록 하겠습니다. 이어서 컴파일 과정에 대해 살펴볼 것이며, 또 마지막으로 어셈블리 언어와 명령어 실행을 컴파일러에 의해 최적화가 수행되는 과정까지 이어질 것입니다. 이러한 과정들을 이해함으로써 컴퓨터 시스템의 작동 원리를 깊이 있게 파악할 수 있을 것입니다.

![computer_architecture.png](/images/posts/2025-04-28-computer-architecture/computer_architecture.png)

## 1. ⚙ CPU {#cpu}

### PC (Program Counter) {#pc-program-counter}

- **역할**

  다음에 실행할 명령어의 메모리 주소를 저장하는 레지스터입니다. CPU는 PC에 저장된 주소를 참조하여 순차적으로 명령어를 가져와 실행합니다.

- **동작**

  명령어 실행이 완료될 때마다 PC는 자동으로 다음 명령어의 주소로 업데이트됩니다. 분기 명령어나 함수 호출 등의 경우 PC 값이 변경되어 프로그램 실행 흐름을 제어합니다.

### 레지스터 (Register) {#register}

- **역할**

  CPU 내부에 있는 작고 빠른 임시 저장 공간입니다. 연산에 필요한 데이터나 중간 결과, 메모리 주소 등을 저장하며, CPU는 메모리보다 레지스터에 접근하는 것이 훨씬 빠르기 때문에 프로그램 실행 속도를 높이는 데 중요한 역할을 합니다.

- **종류:**
  - **범용 레지스터:** 다양한 용도로 사용되는 레지스터 (예: 데이터 연산, 주소 저장)
  - **특수 목적 레지스터:** 특정 용도로 사용되는 레지스터 (예: PC, 상태 레지스터)

### 상태 코드 (Condition Code) {#condition-code}

- **역할**

  CPU 내부의 상태 레지스터 (Status Register)에 저장되는 플래그 (flag)들의 집합입니다. 이 플래그들은 CPU가 수행한 연산의 결과에 따라 설정되며, 이후 실행될 명령어의 흐름을 제어하는 데 사용됩니다.

- **주요 Condition Code Flag**
  - **Zero Flag (Z):** 연산 결과가 0인 경우 1로 설정됩니다.
  - **Carry Flag (C):** 연산 결과에서 자리 올림(carry)이 발생한 경우 1로 설정됩니다. 주로 부호 없는 정수 연산에서 사용됩니다.
  - **Overflow Flag (V):** 연산 결과가 표현 가능한 범위를 넘어 오버플로우가 발생한 경우 1로 설정됩니다. 주로 부호 있는 정수 연산에서 사용됩니다.
  - **Sign Flag (S):** 연산 결과가 음수인 경우 1로 설정됩니다.
  - **Parity Flag (P):** 연산 결과에서 1의 개수가 짝수인 경우 1로 설정됩니다.

- **활용**

  주로 조건 분기 명령어와 함께 사용되어 프로그램의 실행 흐름을 제어합니다. 예를 들어, 두 수를 비교하는 명령어를 실행한 후, Zero Flag, Carry Flag, Overflow Flag 등을 확인하여 점프(jump) 명령어를 통해 특정 위치로 이동할 수 있습니다.

- **예시**

  ```text
  CMP R1, R2  ; R1과 R2 레지스터 값 비교
  JZ  EQUAL   ; Zero Flag가 1이면 EQUAL 레이블로 점프 (R1 == R2)
  JG  GREATER ; Sign Flag가 0이고 Overflow Flag가 0이면 GREATER 레이블로 점프 (R1 > R2)
  JL  LESS    ; Sign Flag가 1이면 LESS 레이블로 점프 (R1 < R2)
  ```

  위 어셈블리 코드는 두 레지스터의 값을 비교하고, 비교 결과에 따라 다른 레이블로 점프하는 예시입니다. 이처럼 Condition Code를 활용하면 프로그램의 실행 흐름을 다양하게 제어할 수 있습니다.

- **중요성**

  프로그램의 논리적인 흐름을 제어하는 데 필수적인 역할을 합니다. 조건문, 반복문, 함수 호출 등 다양한 제어 구조를 구현하는 데 사용되며, 효율적인 프로그램 실행을 위해 Condition Code를 적절히 활용하는 것이 중요합니다.

## 2. ⚙ 컴파일 과정 (Compilation Process) {#compilation-process}

### 전처리 (Preprocessing) {#preprocessing}

소스 코드에서 `#include`, `#define` 등의 전처리 지시자를 처리하여 하나의 소스 파일로 통합합니다.

### 컴파일 (Compilation) {#compilation}

전처리된 소스 코드를 어셈블리 코드로 변환합니다. 이 과정에서 문법 오류를 검사하고, 변수 타입, 함수 호출 등을 분석합니다.

### 어셈블 (Assembly) {#assembly}

어셈블리 코드를 기계어 (0과 1로 이루어진 명령어)로 변환합니다. 이때 각 어셈블리 명령어는 CPU가 실행할 수 있는 기계어 명령어로 변환됩니다.

### 링크 (Linking) {#linking}

여러 개의 목적 파일 (object file)과 라이브러리를 연결하여 실행 가능한 파일 (executable file)을 생성합니다. 외부 함수 호출 등을 해결하고, 프로그램 실행에 필요한 모든 코드를 하나로 묶습니다.

## 3. ⚙ 어셈블리 언어와 명령어 실행 {#assembly-language-and-instruction-execution}

### 명령어 단위 실행 (Instruction-Level Execution) {#instruction-level-execution}

컴퓨터는 CPU (중앙처리장치)를 통해 프로그램을 실행합니다. CPU는 프로그램을 구성하는 명령어들을 순차적으로 하나씩 가져와서 해독하고 실행하는 과정을 반복합니다. 이러한 방식을 명령어 단위 실행이라고 합니다.

- **명령어 실행 과정**
  1. **Fetch (가져오기):** CPU는 PC (Program Counter) 레지스터에 저장된 메모리 주소를 참조하여 다음에 실행할 명령어를 메모리에서 가져옵니다.
  2. **Decode (해독):** CPU는 가져온 명령어를 해독하여 어떤 연산을 수행해야 하는지 파악합니다. 명령어는 연산 코드 (opcode)와 피연산자 (operand)로 구성됩니다.
  3. **Execute (실행):** CPU는 해독된 명령어에 따라 필요한 연산을 수행합니다. 연산 결과는 레지스터에 저장되거나 메모리에 쓰여질 수 있습니다.

- **명령어 집합 (Instruction Set)**

  각 CPU는 고유한 명령어 집합을 가지고 있습니다. 명령어 집합은 CPU가 실행할 수 있는 명령어들의 종류와 형식을 정의합니다. 명령어 집합은 CPU의 성능과 기능에 큰 영향을 미치며, 컴퓨터 아키텍처의 중요한 요소입니다.

### 자바 바이트코드 (Java Bytecode) {#java-bytecode}

자바는 컴파일 언어이지만, 기계어로 직접 변환되지 않고 중간 단계의 코드인 자바 바이트코드로 변환됩니다. 자바 바이트코드는 JVM (Java Virtual Machine)이라는 가상 머신에서 실행됩니다.

- **자바 바이트코드의 장점:**
  - **플랫폼 독립성:** JVM은 다양한 운영체제와 하드웨어 플랫폼에서 동작하므로, 자바 프로그램은 한 번 컴파일하면 어떤 플랫폼에서든 실행될 수 있습니다.
  - **보안성:** JVM은 자바 바이트코드를 실행하기 전에 안전성 검사를 수행하여 악성 코드 실행을 방지합니다.
  - **최적화:** JVM은 실행 시간에 자바 바이트코드를 분석하여 성능을 향상시키는 JIT (Just-In-Time) 컴파일과 같은 최적화 기법을 적용할 수 있습니다.

- **자바 바이트코드 실행 과정:**
  1. **자바 컴파일러:** 자바 소스 코드를 자바 바이트코드로 변환합니다.
  2. **클래스 로더:** JVM은 필요한 클래스 파일 (자바 바이트코드가 저장된 파일)을 로드합니다.
  3. **바이트코드 검증기:** JVM은 로드된 클래스 파일의 유효성을 검사하여 악성 코드를 차단합니다.
  4. **인터프리터 또는 JIT 컴파일러:** JVM은 자바 바이트코드를 해석하여 실행하거나, JIT 컴파일을 통해 기계어로 변환하여 실행합니다.

### 컴파일러 최적화 (Compiler Optimization) {#compiler-optimization}

컴파일러는 자바 소스 코드를 자바 바이트코드로 변환하는 과정에서 다양한 최적화 기법을 적용하여 프로그램의 실행 속도를 향상시키고 메모리 사용량을 줄입니다.

- **주요 컴파일러 최적화 기법:**
  - **상수 전파 (Constant Propagation):** 컴파일 시간에 결정될 수 있는 상수 값을 미리 계산하여 코드를 간소화합니다.
  - **데드 코드 제거 (Dead Code Elimination):** 실행되지 않는 코드를 제거하여 프로그램 크기를 줄입니다.
  - **루프 언롤링 (Loop Unrolling):** 반복문의 일부를 복제하여 반복 횟수를 줄이고 실행 속도를 높입니다.
  - **인라인 함수 확장 (Inline Function Expansion):** 함수 호출 부분을 함수 내용으로 대체하여 함수 호출 오버헤드를 줄입니다.

- **최적화 레벨**

  컴파일러는 다양한 최적화 레벨을 제공하여 개발자가 원하는 수준의 최적화를 선택할 수 있도록 합니다. 일반적으로 최적화 레벨이 높을수록 컴파일 시간은 길어지지만, 실행 속도는 빨라집니다.

## 🏁 마치며 {#conclusion}

이러한 개념들은 컴퓨터의 작동 방식과 프로그래밍의 기초를 이해하는 데 중요합니다. 각 단계와 구성 요소는 프로그램이 효율적으로 실행되도록 하는 데 필수적인 역할을 합니다. 또한, 컴파일러 최적화는 프로그램의 성능을 향상시키는 데 중요한 기법입니다. 이러한 지식은 프로그래밍과 시스템 설계에 있어 근본적인 이해를 제공합니다.
