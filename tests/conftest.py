"""Общие приспособления для тестов.

Главное здесь — **фейковый апстрим**. Он позволяет прогонять весь горячий
путь (§3.2) без сети, без Docker и без поднятого стенда.

Зачем это важно именно на хакатоне: без такого приспособления единственный
способ проверить цепочку целиком — поднять стенд и дождаться его. Цикл
«правка → проверка» растягивается с полусекунды до полуминуты, и под
давлением проверять перестают вовсе.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator

import pytest

from gateway.core.domain import Upstream
from gateway.upstream.adapter import (
    StreamEvent,
    UpstreamError,
    UpstreamResponse,
)


@dataclass
class FakeUpstreamBehaviour:
    """Как фейковый апстрим отвечает. Ровно те режимы, что нужны для
    проверки resilience: успех, ошибка, зависание, обрыв посреди стрима."""

    tokens: int = 5
    ttft_s: float = 0.0
    itl_s: float = 0.0
    status: int = 200
    fail_with: UpstreamError | None = None
    hang: bool = False
    break_after: int | None = None     # оборвать стрим после N токенов
    usage: bool = True


class FakeUpstreamClient:
    """Подменяет UpstreamClient, не выходя в сеть.

    Ведёт журнал вызовов: по нему проверяется не только «что вернулось»,
    но и **куда ушёл запрос** и **закрыли ли поток** — а это ровно те
    свойства, на которых система ломалась (утечка ёмкости, §3.3.9).
    """

    def __init__(self) -> None:
        self.behaviour: dict[str, FakeUpstreamBehaviour] = {}
        self.default = FakeUpstreamBehaviour()
        self.calls: list[str] = []
        self.closed: list[str] = []
        self.started = False

    def set(self, upstream_id: str, **kwargs) -> None:
        self.behaviour[upstream_id] = FakeUpstreamBehaviour(**kwargs)

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        self.started = False

    def _behaviour(self, upstream_id: str) -> FakeUpstreamBehaviour:
        return self.behaviour.get(upstream_id, self.default)

    async def open_stream(self, request, upstream: Upstream, adapter) -> UpstreamResponse:
        self.calls.append(upstream.id)
        b = self._behaviour(upstream.id)

        if b.fail_with is not None:
            raise b.fail_with
        if b.hang:
            await asyncio.sleep(3600)

        closed_flag: list[str] = self.closed

        async def events() -> AsyncIterator[StreamEvent]:
            if b.ttft_s:
                await asyncio.sleep(b.ttft_s)
            yield StreamEvent(kind="delta", raw=b'data: {"delta":{"role":"assistant"}}\n\n')
            for i in range(b.tokens):
                if b.break_after is not None and i >= b.break_after:
                    raise UpstreamError(
                        f"апстрим {upstream.id} оборвал стрим",
                        upstream_id=upstream.id,
                    )
                if b.itl_s:
                    await asyncio.sleep(b.itl_s)
                yield StreamEvent(kind="delta", content=f"т{i} ",
                                  raw=f'data: {{"t":{i}}}\n\n'.encode())
            if b.usage:
                yield StreamEvent(kind="usage", raw=b'data: {"usage":{}}\n\n',
                                  prompt_tokens=request.prompt_tokens_est,
                                  completion_tokens=b.tokens)
            yield StreamEvent(kind="done", raw=b"data: [DONE]\n\n")

        async def close() -> None:
            closed_flag.append(upstream.id)

        return UpstreamResponse(
            status=b.status, headers={}, events=events(),
            close=close, upstream=upstream,
        )


@pytest.fixture
def fake_client() -> FakeUpstreamClient:
    return FakeUpstreamClient()


@pytest.fixture
def gateway(fake_client, tmp_path):
    """Собранный гейтвей с фейковым апстримом и своей копией конфига.

    Копия нужна, чтобы тест мог править конфигурацию (демо hot-reload)
    и не трогать рабочие файлы репозитория.
    """
    import shutil
    from pathlib import Path

    from gateway.core.gateway import Gateway
    from gateway.core.registry import ConfigRegistry

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    for f in Path("config").glob("*.yaml"):
        shutil.copy(f, cfg_dir / f.name)

    registry = ConfigRegistry(cfg_dir, poll_interval_s=0.05)
    registry.load_now()
    gw = Gateway(registry)
    gw.client = fake_client
    gw.registry_dir = cfg_dir
    return gw


async def collect(response) -> list[bytes]:
    """Дочитывает стрим до конца, как это делает реальный клиент."""
    return [chunk async for chunk in response.stream]


def chat_body(*, model: str = "main", stream: bool = True,
              content: str = "привет", turns: int = 0, **extra) -> dict:
    """Тело запроса; при turns > 0 — многоходовая сессия."""
    messages = [{"role": "system", "content": "системный промпт " * 100}]
    for i in range(turns):
        messages.append({"role": "user", "content": f"вопрос {i} " * 30})
        messages.append({"role": "assistant", "content": f"ответ {i} " * 30})
    messages.append({"role": "user", "content": content})
    return {"model": model, "stream": stream, "messages": messages, **extra}
