---
layout: post
title: CPU 스케줄러
date: 2025-04-30
last_modified_at: 2025-04-30
tags: [CS]
toc: true
---

## 📑 **Table Of Contents**
- [1. ⚙ CPU 스케줄링 개요](#cpu-scheduling-overview)
  - [CPU 스케줄링의 기본 개념](#basic-concepts)
  - [스케줄링의 역할과 작동 시점](#scheduling-role-timing)
  - [디스패처와 컨텍스트 스위칭](#dispatcher-context-switching)
- [2. ⚙ 주요 스케줄링 알고리즘](#scheduling-algorithms)
  - [FCFS (First-Come, First-Served)](#fcfs)
  - [SJF (Shortest Job First)](#sjf)
  - [라운드 로빈 (Round Robin)](#round-robin)
- [3. ⚙ 고급 스케줄링 기법](#advanced-scheduling)
  - [다단계 큐 (Multilevel Queue)](#multilevel-queue)
- [4. 🏁 마치며](#conclusion)

<<<<<<< HEAD
![cpu_scheduler_1.png](/images/posts/2025-04-30-cpu-scheduler/cpu_scheduler_1.png)
=======
![cpu_scheduler_1.png](/images/posts/cpu_scheduler_1.png)
>>>>>>> 424e1e2ea7ce8b20dfc4fe2ec09cba4e219939ae

## 1. ⚙ CPU 스케줄링 개요 {#cpu-scheduling-overview}

### CPU 스케줄링의 기본 개념 {#basic-concepts}

- CPU 스케줄링은 여러 프로세스가 CPU를 효율적으로 사용할 수 있도록 CPU 사용 권한을 할당하는 운영체제의 핵심 기능입니다.
- CPU burst와 I/O request의 빈도와 지속 시간을 고려하여 적절히 조율하는 것이 중요합니다.
- CPU burst duration은 일반적으로 밀리초 단위이며, CPU bound 작업과 I/O bound 작업이 존재합니다.
- CPU 스케줄러는 단기 스케줄러(short-term scheduler)라고도 불립니다.

<<<<<<< HEAD
![cpu_scheduler_2.png](/images/posts/2025-04-30-cpu-scheduler/cpu_scheduler_2.png)
=======
![cpu_scheduler_2.png](/images/posts/cpu_scheduler_2.png)
>>>>>>> 424e1e2ea7ce8b20dfc4fe2ec09cba4e219939ae

### 스케줄링의 역할과 작동 시점 {#scheduling-role-timing}

- CPU 스케줄러는 다음과 같은 상황에서 동작합니다:
  - 프로세스가 입장(admitted)하거나 종료(exit)할 때
  - 인터럽트 발생 시
  - 프로세스가 입출력(I/O) 대기 상태로 전환될 때
- 스케줄링은 선점형(preemptive)과 비선점형(non-preemptive)으로 나뉘며, 인터럽트나 커널 모드에서 선점이 고려됩니다.
- 프로세스 상태 변화에 따른 스케줄링 흐름도는 다음과 같습니다:

<<<<<<< HEAD
![cpu_scheduler_3.png](/images/posts/2025-04-30-cpu-scheduler/cpu_scheduler_3.png)
=======
![cpu_scheduler_3.png](/images/posts/cpu_scheduler_3.png)
>>>>>>> 424e1e2ea7ce8b20dfc4fe2ec09cba4e219939ae

이 중 1번과 4번 전환 시 선점형 스케줄링이 발생합니다.

### 디스패처와 컨텍스트 스위칭 {#dispatcher-context-switching}

- **디스패처**는 Ready 리스트에서 실행할 프로세스를 선택하여 CPU에 할당하는 역할을 합니다.
- 프로세스 정보를 담은 프로세스 디스크립터(Process Descriptor)를 활용하여 컨텍스트 스위칭이 이루어집니다.
- **컨텍스트 스위칭**은 현재 실행 중인 프로세스의 상태를 저장하고, 새 프로세스의 상태를 복원하는 과정으로, 사용자 모드 전환과 프로그램 재시작 위치 점프를 포함합니다.
- 컨텍스트 스위칭은 시스템 오버헤드를 발생시킵니다.

<<<<<<< HEAD
![cpu_scheduler_4.png](/images/posts/2025-04-30-cpu-scheduler/cpu_scheduler_4.png)
=======
![cpu_scheduler_4.png](/images/posts/cpu_scheduler_4.png)
>>>>>>> 424e1e2ea7ce8b20dfc4fe2ec09cba4e219939ae

## 2. ⚙ 주요 스케줄링 알고리즘 {#scheduling-algorithms}

### FCFS (First-Come, First-Served) {#fcfs}

- 가장 간단한 비선점형 스케줄링 알고리즘으로, 먼저 도착한 프로세스가 먼저 CPU를 할당받습니다.
- 단점으로는 긴 작업이 앞에 있으면 뒤에 있는 짧은 작업들이 대기하는 ‘호위 효과(Convoy Effect)’가 발생합니다.
- 이를 해결하기 위해 뒤에 소개할 SJF나 Round Robin 알고리즘이 사용됩니다.

<<<<<<< HEAD
![cpu_scheduler_5.png](/images/posts/2025-04-30-cpu-scheduler/cpu_scheduler_5.png)
=======
![cpu_scheduler_5.png](/images/posts/cpu_scheduler_5.png)
>>>>>>> 424e1e2ea7ce8b20dfc4fe2ec09cba4e219939ae

### SJF (Shortest Job First) {#sjf}

- 실행 시간이 가장 짧은 작업을 우선 실행하는 알고리즘입니다.
- 이론적으로 평균 대기 시간을 최소화하지만, 긴 작업이 계속 대기하는 기아 현상(Starvation)이 발생할 수 있습니다.
- 실행 시간 예측이 어려운 문제는 과거 실행 시간을 기반으로 한 예측 알고리즘으로 해결합니다.

<<<<<<< HEAD
![cpu_scheduler_6.png](/images/posts/2025-04-30-cpu-scheduler/cpu_scheduler_6.png)
=======
![cpu_scheduler_6.png](/images/posts/cpu_scheduler_6.png)
>>>>>>> 424e1e2ea7ce8b20dfc4fe2ec09cba4e219939ae

### 라운드 로빈 (Round Robin) {#round-robin}

- 각 프로세스에 동일한 시간 할당량(Time Quantum)을 주고 순환하며 실행하는 선점형 스케줄링입니다.
- 컨텍스트 스위칭 오버헤드가 발생할 수 있으나, 다단계 큐 같은 차등적 시간 할당량 적용으로 문제를 완화할 수 있습니다.
- 짧은 시간 할당량은 응답성을 높이지만 오버헤드를 증가시키고, 긴 시간 할당량은 FCFS와 유사한 동작을 합니다.

<<<<<<< HEAD
![cpu_scheduler_7.png](/images/posts/2025-04-30-cpu-scheduler/cpu_scheduler_7.png)
=======
![cpu_scheduler_7.png](/images/posts/cpu_scheduler_7.png)
>>>>>>> 424e1e2ea7ce8b20dfc4fe2ec09cba4e219939ae

## 3. ⚙ 고급 스케줄링 기법 {#advanced-scheduling}

### 다단계 큐 (Multilevel Queue) {#multilevel-queue}

- 프로세스 유형별로 여러 개의 큐를 구성하고, 각 큐마다 다른 스케줄링 알고리즘과 우선순위를 적용합니다.
- 예를 들어, 시스템 프로세스는 높은 우선순위의 큐에서 Round Robin으로, 배치 프로세스는 낮은 우선순위의 큐에서 FCFS로 스케줄링할 수 있습니다.
- 각 큐는 시분할(Time Slice) 방식으로 CPU 시간을 분배받아 사용합니다.
- 다단계 큐는 사용자와 시스템 요구에 맞춰 유연한 스케줄링 정책을 구현할 수 있습니다.

<<<<<<< HEAD
![cpu_scheduler_8.png](/images/posts/2025-04-30-cpu-scheduler/cpu_scheduler_8.png)
=======
![cpu_scheduler_8.png](/images/posts/cpu_scheduler_8.png)
>>>>>>> 424e1e2ea7ce8b20dfc4fe2ec09cba4e219939ae

## 🏁 마치며 {#conclusion}

CPU 스케줄러는 운영체제의 핵심 구성 요소로, 다양한 스케줄링 알고리즘과 기법을 통해 시스템 자원을 효율적으로 관리합니다. 기본적인 FCFS, SJF, Round Robin 알고리즘부터 다단계 큐와 같은 고급 기법까지, 각각의 장단점과 적용 상황을 이해하는 것이 중요합니다. 이를 통해 시스템 성능 향상과 공정한 자원 분배를 달성할 수 있습니다.
