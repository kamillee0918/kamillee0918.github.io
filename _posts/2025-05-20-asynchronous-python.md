---
layout: post
title: "비동기 파이썬 기반 LLM API 구현: OpenAI API, Redis, Asyncio 활용"
date: 2025-05-20 12:27:10 +0900
last_modified_at: 2025-05-20 12:27:10 +0900
tags: [비동기 프로그래밍, FastAPI, LLM]
toc: true
---

## 📑 **Table Of Contents**

- [1. ⚙ 비동기 LLM API의 개념](#async-llm-basics)
  - [비동기 프로그래밍의 필요성](#why-async)
  - [Python 비동기 기초와 Asyncio](#asyncio-basics)
  - [애플리케이션 다이어그램](#application-diagram)
- [2. ⚙ OpenAI API 비동기 호출](#openai-async)
  - [AsyncOpenAI 및 aiohttp 활용](#asyncopenai-usage)
  - [동시성/배치 처리](#concurrency-batching)
- [3. ⚙ FastAPI 기반 비동기 LLM 서비스 설계](#fastapi-design)
  - [엔드포인트 설계 및 예시](#endpoint-design)
- [4. ⚙ Redis를 활용한 상태 관리 및 캐싱](#redis-integration)
  - [비동기 Redis 클라이언트 활용](#async-redis)
  - [작업 상태 추적 및 TTL 관리](#task-tracking)
- [5. ⚙ 전체 구조 및 코드 예시](#full-example)
- [6. ⚙ 고려사항 및 베스트 프랙티스](#considerations)
- [7. 🏁 마치며](#conclusion)

---

![async-python-1.png](/images/posts/2025-05-20-asynchronous-python/async-python-1.png)

## 1. ⚙ 비동기 LLM API의 개념 {#async-llm-basics}

### 비동기 프로그래밍의 필요성 {#why-async}

대규모 언어모델(LLM) API를 활용할 때, 다수의 프롬프트를 순차적으로 처리하면 네트워크 지연과 응답 대기 때문에 전체 처리 시간이 크게 늘어납니다.

비동기 프로그래밍을 적용하면 여러 요청을 동시에 처리할 수 있어 처리량(throughput)이 비약적으로 증가하고, 응답 대기 시간을 줄일 수 있습니다.

### Python 비동기 기초와 Asyncio {#asyncio-basics}

Python의 **asyncio** 모듈은 코루틴, 이벤트 루프, Future 객체를 활용해 비동기 코드를 작성할 수 있게 해줍니다.

- **코루틴**: `async def`로 정의, `await`로 일시 중단 및 재개
- **이벤트 루프**: 비동기 작업을 관리
- **awaitables**: `await`로 대기 가능한 객체
- **워커(Worker)**: 작업 큐에서 작업을 가져와 실행하는 소비자(consumer) 역할을 하는 비동기 함수로, 여러 워커가 동시에 실행되어 병렬 처리를 구현

```python
import asyncio

async def greet(name):
    await asyncio.sleep(1)
    print(f"Hello, {name}!")


async def main():
    await asyncio.gather(
        greet("Kamil Lee"),
        greet("Dowonna Lee"),
        greet("Zorn")
    )

asyncio.run(main())
```

이처럼 여러 작업을 동시에 실행할 수 있습니다.

### 애플리케이션 다이어그램 {#application-diagram}

![async-python-2.png](/images/posts/2025-05-20-asynchronous-python/async-python-2.png)

---

## 2. ⚙ OpenAI API 비동기 호출 {#openai-async}

### AsyncOpenAI 활용 {#asyncopenai-usage}

OpenAI 공식 Python 클라이언트는 비동기 사용을 지원합니다.

- `AsyncOpenAI`와 같은 비동기 메서드를 사용하면 여러 프롬프트에 대한 응답을 병렬로 받을 수 있습니다.

```python
# app/routers/router.py
from openai import AsyncOpenAI

async def process_llm_task(task_id: str, prompt: str):
    """LLM 작업 처리 함수"""
    try:
        client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        # OpenAI가 지원하는 모델 호출
        response = await client.responses.create(
            model="gpt-3.5-turbo",
            input=prompt,
            max_output_tokens=100, # 최대 출력 토큰 수
            temperature=0.7 # 응답 다양성 조절
        )

    except Exception as e:
        print("Error processing task:", e)
        return None
```

```python
# app/main.py
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI

import os
import redis.asyncio as redis


# 환경변수 로드
load_dotenv()

# Redis 클라이언트 전역 변수 선언
redis_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST'),
        port=os.getenv('REDIS_PORT'),
        db=os.getenv('REDIS_DB'),
        decode_responses=True,
    )
    yield
    await app.state.redis_client.aclose()

# FastAPI 애플리케이션 초기화
app = FastAPI(
    title="Asynchronous LLM API",
    description="비동기 LLM 처리를 위한 API 서버",
    version="1.0.0",
    swagger_ui_parameters={
        "syntaxHighlight": {
            "theme": "obsidian"
        }
    },
    lifespan=lifespan,
)

# 라우터 등록
from routers.router import router
app.include_router(router, prefix="/api", tags=["LLM Background"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="localhost", port=8000)
```

이 방식으로 여러 요청을 동시에 처리할 수 있습니다.

### 동시성/배치 처리 {#concurrency-batching}

- **동시성 제어**: `asyncio.Queue`와 고정된 수의 워커(consumer)를 활용해, 큐에 쌓인 작업을 병렬로 처리합니다. 각 워커는 큐에서 작업을 가져와 비동기적으로 실행하는 독립적인 소비자(consumer)로, 워커 수를 적절히 조절함으로써 동시에 처리되는 작업 수를 제한할 수 있습니다.  
- **배치 처리**: 단일 API 호출로 여러 프롬프트를 처리하는 것이 아닌, 다수의 개별 프롬프트 작업을 큐에 넣고 여러 워커가 이를 병렬로 처리하는 방식입니다. 이를 통해 많은 양의 작업을 보다 효율적으로 분산 처리할 수 있으며, 자원을 최적으로 활용할 수 있습니다.

---

## 3. ⚙ FastAPI 기반 비동기 LLM 서비스 설계 {#fastapi-design}

FastAPI는 파이썬 비동기 코드를 자연스럽게 지원하는 프레임워크이며, [Starlette](https://www.starlette.io/)과 [Pydantic](https://docs.pydantic.dev/latest/)을 기반으로 비동기 I/O의 장점을 최대한 활용할 수 있도록 설계되었습니다.

- **Non-blocking I/O**: 전통적인 동기식 웹 서버와 달리, FastAPI는 각 요청이 I/O 작업을 기다리는 동안 다른 요청을 처리할 수 있어 동일 하드웨어에서 더 많은 동시 연결을 처리
- **이벤트 루프 활용**: asyncio의 이벤트 루프를 기반으로 하여 대량의 동시 연결을 효율적으로 관리
- **비동기 라우팅**: 라우트 핸들러를 `async def`로 정의하여 비동기 처리가 필요한 작업(외부 API 호출, DB 쿼리 등)을 효율적으로 처리

### 엔드포인트 설계 및 예시 {#endpoint-design}

FastAPI는 Python 비동기 코드를 자연스럽게 지원하는 프레임워크입니다.

- **POST /api/tasks**: 여러 프롬프트를 받아 비동기 작업 생성, 각 작업에 고유 task_id 부여
- **GET /api/tasks**: 작업 상태(progress/result) 조회

```python
# app/routers/router.py
from dotenv import load_dotenv
from fastapi import APIRouter, Request
from typing import List
from pydantic import BaseModel
from openai import AsyncOpenAI

import uuid
import os
import asyncio
import redis.asyncio as redis

# ... 환경변수 호출 및 Redis 연결 생략 ...

# 고정값 지정
NUM_WORKERS = 4 # 쓰레드 풀 크기와 일치

# 비동기 방식의 큐 구현
task_queue = asyncio.Queue()


class PromptList(BaseModel):
    """PromptList 모델 클래스 정의"""
    prompts: List[str] # 프롬프트 리스트


async def background_worker():
    """큐에서 작업을 가져와 처리하는 워커(소비자) 함수"""
    while True:
        try:
            task_id, prompt = await task_queue.get()
            await process_llm_task(task_id, prompt)
            task_queue.task_done()
        except Exception as e:
            print(f"Background worker error: {e}")
            continue


async def process_llm_task(task_id: str, prompt: str):
    """LLM 작업 처리 함수
    
    Args:
        task_id (str): 작업 ID
        prompt (str): 프롬프트
    """
    try:
        client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        # ... Redis 상태 업데이트 관련 생략 ...

        # OpenAI가 지원하는 모델 호출
        response = await client.responses.create(
            model="gpt-3.5-turbo",
            input=prompt,
            max_output_tokens=100, # 최대 출력 토큰 수
            temperature=0.7 # 응답 다양성 조절
        )
    except Exception as e:
        print("Error processing task:", e)
        return None


@router.post("/tasks")
async def create_tasks(prompts: PromptList):
    """프롬프트 리스트를 큐에 추가하고 작업 ID를 반환하는 함수
    
    Args:
        prompts (PromptList): 프롬프트 리스트
    
    Returns:
        dict: 작업 ID 리스트와 상태
    """
    task_ids = []
    
    if not hasattr(router, "worker_tasks"):
        router.worker_tasks = [
            asyncio.create_task(background_worker())
            for _ in range(NUM_WORKERS)
        ]
    
    for prompt in prompts.prompts:
        task_id = str(uuid.uuid4())
        # 큐에 작업 추가
        await task_queue.put((task_id, prompt))
        task_ids.append(task_id)

    return {"task_ids": task_ids, "status": "queued"}


@router.get("/tasks")
async def get_task_status(task_ids: str, request: Request):
    """작업 상태를 반환하는 함수
    
    Args:
        task_ids (str): 쉼표로 구분된 작업 ID 리스트
        request (Request): FastAPI Request 객체
    
    Returns:
        dict: 작업 ID별 상태 정보
    """
    # 쉼표로 구분된 작업 ID 리스트를 분리
    task_id_list = task_ids.split(",")
    res = {}

    for task_id in task_id_list:
        redis_client = request.app.state.redis_client
        task_data = await redis_client.hgetall(f"task:{task_id}")
        res[task_id] = task_data if task_data else {"status": "not_found"}
    
    return res
```

이 구조를 확장해 작업 큐, 상태 추적, 캐싱 등을 구현할 수 있습니다.

---

## 4. ⚙ Redis를 활용한 상태 관리 및 캐싱 {#redis-integration}

### 비동기 Redis 클라이언트 활용 {#async-redis}

- **aioredis** 또는 **redis.asyncio**를 사용하면 Redis와 비동기적으로 통신할 수 있습니다.
- 작업별 상태(progress/result) 저장 및 TTL(Time-To-Live)로 만료 관리

```python
from dotenv import load_dotenv
from fastapi import FastAPI
from contextlib import asynccontextmanager

import redis.asyncio as redis


# 환경변수 로드
load_dotenv()

# Redis 클라이언트 전역 변수 선언
redis_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST'),
        port=os.getenv('REDIS_PORT'),
        db=os.getenv('REDIS_DB'),
        decode_responses=True,
    )
    yield
    await app.state.redis_client.aclose()
```

이처럼 작업별로 상태/결과를 저장하고 만료시킬 수 있습니다.

### 작업 상태 추적 및 TTL 관리 {#task-tracking}

- 작업 생성 시 UUID로 task_id 생성, Redis에 `status: pending` 저장
- 작업 완료 후 `status: completed`, 결과 저장
- TTL로 24시간 등 자동 만료 설정
- GET 엔드포인트에서 task_id로 상태/결과 조회

---

## 5. ⚙ 전체 구조 및 코드 예시 {#full-example}

아래는 OpenAI API, FastAPI, Redis, Asyncio를 결합한 비동기 LLM API의 전체 구조 예시입니다. 이 시스템은 다음과 같은 비동기 워크플로우를 따릅니다.

1. 클라이언트가 `/api/tasks` 엔드포인트를 통해 프롬프트 목록 전송
2. 각 프롬프트마다 고유 task_id 생성 후 비동기 큐에 추가
3. 여러 워커(소비자)가 동시에 큐를 모니터링하며 작업 처리
4. Redis를 통한 작업 상태 관리 및 결과 저장
5. 클라이언트가 `/api/tasks` GET 요청으로 작업 상태 및 결과 조회

이러한 구조는 I/O 바운드 작업(API 호출, DB 쿼리 등)을 효율적으로 병렬 처리하여, 시스템 자원을 최대한 활용하면서 높은 처리량을 제공합니다.

```python
# app/routers/router.py
from fastapi import APIRouter, Request
from pydantic import BaseModel
from openai import AsyncOpenAI
import uuid, os, asyncio, redis.asyncio as redis

router = APIRouter()
NUM_WORKERS = 4 # 워커 수 설정
task_queue = asyncio.Queue() # 비동기 작업 큐


class PromptList(BaseModel):
    """PromptList 모델 클래스 정의"""
    prompts: list[str] # 처리할 프롬프트 목록


async def background_worker():
    """큐에서 작업을 가져와 처리하는 워커(소비자) 함수"""
    while True:
        try:
            task_id, prompt = await task_queue.get() # 큐에서 작업 가져오기
            await process_llm_task(task_id, prompt) # 작업 처리
            task_queue.task_done() # 작업 완료 표시
        except Exception as e:
            print(f"워커 오류: {e}")
            continue # 오류가 발생해도 워커는 계속 실행


async def process_llm_task(task_id, prompt):
    client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    # ... Redis 상태 업데이트 및 OpenAI 호출 생략 ...
    # ... 결과/오류 처리 및 TTL 설정 생략 ...


@router.post("/tasks")
async def create_tasks(prompts: PromptList):
    if not hasattr(router, "worker_tasks"):
        router.worker_tasks = [asyncio.create_task(background_worker()) for _ in range(NUM_WORKERS)]
    task_ids = []
    for prompt in prompts.prompts:
        task_id = str(uuid.uuid4())
        await task_queue.put((task_id, prompt))
        task_ids.append(task_id)
    return {"task_ids": task_ids, "status": "queued"}


@router.get("/tasks")
async def get_task_status(task_ids: str, request: Request):
    task_id_list = task_ids.split(",")
    res = {}
    for task_id in task_id_list:
        redis_client = request.app.state.redis_client
        task_data = await redis_client.hgetall(f"task:{task_id}")
        res[task_id] = task_data if task_data else {"status": "not_found"}
    return res
```

- POST `/api/tasks`로 여러 프롬프트를 비동기 처리 요청
- 각 작업은 별도 task_id로 Redis에 상태 저장
- GET `/api/tasks/{task_id}`로 상태/결과 조회
- Redis TTL로 자동 만료

---

## 6. ⚙ 고려사항 및 베스트 프랙티스 {#considerations}

- **동시성 제어**:
  - `asyncio.Queue`와 고정된 수의 워커를 통해 동시에 처리되는 작업의 수를 제한
  - 시스템 리소스에 맞게 `NUM_WORKERS` 값 최적화 (일반적으로 CPU 코어 수 × 2 정도로 시작)
  - 워커 수가 많으면 동시성은 높아지나 CPU 오버헤드 발생, 적으면 큐 대기 시간 증가

- **에러 처리**:
  - LLM API 호출 관련 오류(토큰 한도 초과, 모델 로드 실패 등)에 대해 세분화된 예외 처리
  - Redis 연결 실패 시 지수 백오프 전략으로 재시도 구현
  - 치명적 오류 시 경고 알림 시스템 연동

- **캐싱 전략**:
  - 동일 프롬프트에 대한 결과 해시 기반 캐싱으로 중복 요청 방지
  - 캐시 적중률 모니터링 및 최적화
  - 캐시 무효화 주기 설정

- **작업 만료(TTL)**:
  - Redis TTL 계층화: 작업 상태는 짧게(24시간), 결과는 길게(7일) 유지
  - 중요 작업 결과 영구 저장소 연동

- **로깅/모니터링**:
  - Prometheus/Grafana로 큐 길이, 처리 시간, 오류율 등 실시간 모니터링
  - 구조화된 로그로 디버깅 용이성 향상
  - 평균 응답 시간, P95/P99 지연 시간 등 성능 지표 추적

- **부하 테스트 및 스케일링**:
  - 최대 부하 테스트로 시스템 한계 파악
  - 필요시 Redis Cluster 구성으로 수평적 확장
  - 급격한 트래픽 증가에 대비한 동적 워커 스케일링 구현

---

## 7. 🏁 마치며 {#conclusion}

비동기 파이썬, FastAPI, Redis, Asyncio, 그리고 OpenAI API를 결합하면 대규모 LLM 기반 API를 효율적으로 설계할 수 있습니다.

- **동시성**을 통한 빠른 처리
- **Redis**로 작업 상태 추적 및 캐싱
- **FastAPI**로 확장성 높은 API 제공
- **Asyncio**로 네트워크/IO 병목 최소화

이 구조는 의미 검색, 문서 요약, 질의응답 등 다양한 LLM 기반 서비스에 적용할 수 있으며, 대량 요청 처리와 실시간 응답이 중요한 환경에서 특히 강력한 선택지입니다.
