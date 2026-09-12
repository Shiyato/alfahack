"""Спайк: замер накладных расходов Python на горячем пути (ADR-001).

Вопрос, на который отвечает спайк: сколько стоит проксирование SSE на Python
и во сколько обходится *разбор каждого кадра* — то единственное место, где
интерпретатор реально виден (см. обсуждение стека).

Три режима, чтобы отделить вклад каждого слоя:

  /passthrough  — байты апстрима идут наружу как есть. Нижняя граница.
  /parsed       — каждый SSE-кадр разбирается как JSON, считается токен,
                  кадр пересобирается. Верхняя граница: так выглядит
                  настоящий stream pipeline (§3.3.9) с учётом токенов.
  /counted      — кадр не пересобирается, но токены считаются по границам
                  кадров без разбора JSON. Компромисс, который, вероятно,
                  и пойдёт в продакшн-путь.

Спайк намеренно не содержит ни роутинга, ни квот: измеряется стоимость
самого факта «Python стоит в разрыве стрима», а не нашей логики.
"""

import json
import os
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

UPSTREAM = os.environ.get("SPIKE_UPSTREAM", "http://127.0.0.1:9001")

app = FastAPI()
client: httpx.AsyncClient | None = None


@app.on_event("startup")
async def _startup() -> None:
    global client
    # Пул держим большим: иначе мы измерим лимит пула, а не Python.
    limits = httpx.Limits(max_connections=2000, max_keepalive_connections=2000)
    client = httpx.AsyncClient(base_url=UPSTREAM, limits=limits, timeout=httpx.Timeout(300.0))


@app.on_event("shutdown")
async def _shutdown() -> None:
    if client is not None:
        await client.aclose()


SSE_HEADERS = {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}


async def _upstream_stream(body: bytes):
    req = client.build_request(
        "POST", "/v1/chat/completions", content=body,
        headers={"Content-Type": "application/json"},
    )
    return await client.send(req, stream=True)


@app.post("/passthrough/v1/chat/completions")
async def passthrough(request: Request):
    body = await request.body()
    resp = await _upstream_stream(body)

    async def gen():
        try:
            async for chunk in resp.aiter_raw():
                yield chunk
        finally:
            await resp.aclose()

    return StreamingResponse(gen(), headers=SSE_HEADERS)


@app.post("/counted/v1/chat/completions")
async def counted(request: Request):
    """Считаем токены по границам кадров, без разбора JSON."""
    body = await request.body()
    resp = await _upstream_stream(body)

    async def gen():
        tokens = 0
        try:
            async for chunk in resp.aiter_raw():
                tokens += chunk.count(b"\ndata: ") + chunk.startswith(b"data: ")
                yield chunk
        finally:
            await resp.aclose()

    return StreamingResponse(gen(), headers=SSE_HEADERS)


@app.post("/parsed/v1/chat/completions")
async def parsed(request: Request):
    """Полный разбор каждого кадра: худший реалистичный случай."""
    body = await request.body()
    resp = await _upstream_stream(body)

    async def gen():
        tokens = 0
        buf = b""
        try:
            async for chunk in resp.aiter_raw():
                buf += chunk
                while b"\n\n" in buf:
                    frame, buf = buf.split(b"\n\n", 1)
                    if not frame.startswith(b"data: "):
                        continue
                    payload = frame[6:]
                    if payload == b"[DONE]":
                        yield b"data: [DONE]\n\n"
                        continue
                    obj = json.loads(payload)
                    choices = obj.get("choices") or []
                    if choices and choices[0].get("delta", {}).get("content"):
                        tokens += 1
                    yield b"data: " + json.dumps(obj, separators=(",", ":")).encode() + b"\n\n"
        finally:
            await resp.aclose()

    return StreamingResponse(gen(), headers=SSE_HEADERS)


@app.get("/healthz")
async def healthz():
    return {"ok": True, "ts": time.time()}


# --- Разделение вклада слоёв -------------------------------------------------
# /synthetic не ходит в апстрим вообще: он сам генерирует такой же SSE-поток.
# Сравнение /synthetic с /passthrough отделяет стоимость ASGI-стриминга
# (uvicorn + starlette) от стоимости HTTP-клиента (httpx).
import asyncio


@app.post("/synthetic/v1/chat/completions")
async def synthetic(request: Request):
    body = await request.body()
    req = json.loads(body)
    n = int(req.get("mock_output_tokens") or 6)
    prefill = float(os.environ.get("SPIKE_PREFILL_MS", "9.1")) / 1000
    itl = float(os.environ.get("SPIKE_ITL_MS", "10")) / 1000

    async def gen():
        await asyncio.sleep(prefill)
        yield b'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n'
        for i in range(n):
            await asyncio.sleep(itl)
            yield b'data: {"choices":[{"index":0,"delta":{"content":"tok "}}]}\n\n'
        yield b"data: [DONE]\n\n"

    return StreamingResponse(gen(), headers=SSE_HEADERS)


# --- Альтернативный HTTP-клиент ---------------------------------------------
# Замер показал, что узкое место горячего пути — не интерпретатор и не ASGI,
# а сам HTTP-клиент. Поэтому клиент проверяется отдельно и по тем же правилам.
import aiohttp

aio_session: "aiohttp.ClientSession | None" = None


@app.on_event("startup")
async def _startup_aiohttp() -> None:
    global aio_session
    conn = aiohttp.TCPConnector(limit=0, limit_per_host=0, ttl_dns_cache=300)
    aio_session = aiohttp.ClientSession(
        base_url=UPSTREAM, connector=conn,
        timeout=aiohttp.ClientTimeout(total=None, sock_connect=5),
    )


@app.on_event("shutdown")
async def _shutdown_aiohttp() -> None:
    if aio_session is not None:
        await aio_session.close()


@app.post("/aiohttp/v1/chat/completions")
async def aiohttp_passthrough(request: Request):
    body = await request.body()

    async def gen():
        async with aio_session.post(
            "/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"},
        ) as resp:
            async for chunk in resp.content.iter_any():
                yield chunk

    return StreamingResponse(gen(), headers=SSE_HEADERS)


@app.post("/aiohttp-parsed/v1/chat/completions")
async def aiohttp_parsed(request: Request):
    """aiohttp + полный разбор каждого кадра: реалистичный горячий путь."""
    body = await request.body()

    async def gen():
        tokens = 0
        buf = b""
        async with aio_session.post(
            "/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"},
        ) as resp:
            async for chunk in resp.content.iter_any():
                buf += chunk
                while b"\n\n" in buf:
                    frame, buf = buf.split(b"\n\n", 1)
                    if not frame.startswith(b"data: "):
                        continue
                    payload = frame[6:]
                    if payload == b"[DONE]":
                        yield b"data: [DONE]\n\n"
                        continue
                    obj = json.loads(payload)
                    choices = obj.get("choices") or []
                    if choices and choices[0].get("delta", {}).get("content"):
                        tokens += 1
                    yield b"data: " + json.dumps(obj, separators=(",", ":")).encode() + b"\n\n"

    return StreamingResponse(gen(), headers=SSE_HEADERS)
