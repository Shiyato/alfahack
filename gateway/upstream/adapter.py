"""Upstream-адаптеры (§3.3.7).

Назначение слоя — изоляция различий провайдеров в одном месте. Новый
провайдер должен стоить «новый адаптер плюс запись в реестре», без правок
роутера и остальной цепочки. Это и есть требование «гибкость» из §1.2.

О выборе HTTP-клиента. Здесь используется aiohttp, и это не вкусовщина,
а результат замера (ADR-001): на 50 одновременных стримах httpx даёт
TTFT p95 в 1189 мс при загрузке процессора 58%, то есть упирается не в
CPU, а во внутреннюю сериализацию; aiohttp на той же нагрузке — 31 мс.
Разница в два порядка. Запрет на httpx в горячем пути зафиксирован в ADR.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import aiohttp

from ..core.domain import ChatRequest, Upstream

log = logging.getLogger(__name__)


class UpstreamError(Exception):
    """Ошибка обращения к апстриму, пригодная для решения о деградации."""

    def __init__(self, message: str, *, status: int | None = None,
                 retryable: bool = False, upstream_id: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.upstream_id = upstream_id


class UpstreamTimeout(UpstreamError):
    pass


@dataclass(slots=True)
class StreamEvent:
    """Событие потока в нормализованном виде.

    Цепочка обработки не должна разбирать провайдерский JSON — иначе
    формат конкретного провайдера снова протечёт наружу.
    """

    kind: str                       # "delta" | "done" | "usage"
    content: str = ""
    raw: bytes = b""                # исходный кадр: отдаём клиенту как есть
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None


class Adapter(ABC):
    """Контракт адаптера."""

    name: str = "base"

    @abstractmethod
    def build_payload(self, request: ChatRequest, upstream: Upstream) -> dict[str, Any]:
        ...

    @abstractmethod
    def parse_frame(self, payload: bytes) -> StreamEvent | None:
        ...

    def headers(self, upstream: Upstream) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if upstream.api_key:
            h["Authorization"] = f"Bearer {upstream.api_key}"
        return h

    def endpoint(self, upstream: Upstream) -> str:
        return upstream.base_url.rstrip("/") + "/v1/chat/completions"


class OpenAIAdapter(Adapter):
    """OpenAI-совместимый провайдер — де-факто стандарт.

    Служит и адаптером по умолчанию, и образцом: любой новый провайдер
    реализует те же три метода.
    """

    name = "openai"

    def build_payload(self, request: ChatRequest, upstream: Upstream) -> dict[str, Any]:
        payload: dict[str, Any] = {
            # Имя модели — то, под которым её знает провайдер, а не
            # логическое имя из нашего реестра.
            "model": upstream.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            # Всегда стрим: см. пояснение у ChatRequest.stream.
            "stream": True,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        # Просим usage в последнем кадре: иначе расход токенов придётся
        # оценивать, а по нему считаются квоты (§3.3.3.5).
        payload["stream_options"] = {"include_usage": True}
        payload.update(request.passthrough)
        return payload

    def parse_frame(self, payload: bytes) -> StreamEvent | None:
        if payload == b"[DONE]":
            return StreamEvent(kind="done", raw=b"data: [DONE]\n\n")
        try:
            obj = json.loads(payload)
        except (ValueError, UnicodeDecodeError):
            return None
        if not isinstance(obj, dict):
            # Валидный JSON, но не объект: `null`, `[]`, строка, число.
            # Апстрим не обязан присылать только то, что мы ожидаем, а
            # исключение здесь роняет весь стрим — один странный кадр
            # обрывает нормально идущую генерацию.
            return None

        raw = b"data: " + payload + b"\n\n"
        usage = obj.get("usage")
        if isinstance(usage, dict) and usage:
            return StreamEvent(
                kind="usage", raw=raw,
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
            )
        choices = obj.get("choices") or []
        if not isinstance(choices, list) or not choices:
            return StreamEvent(kind="delta", raw=raw)
        choice = choices[0]
        if not isinstance(choice, dict):
            return StreamEvent(kind="delta", raw=raw)
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            delta = {}
        content = delta.get("content")
        return StreamEvent(
            kind="delta",
            content=content if isinstance(content, str) else "",
            raw=raw,
            finish_reason=choice.get("finish_reason"),
        )


ADAPTERS: dict[str, type[Adapter]] = {OpenAIAdapter.name: OpenAIAdapter}


def get_adapter(name: str) -> Adapter:
    cls = ADAPTERS.get(name)
    if cls is None:
        raise ValueError(f"неизвестный адаптер {name!r}, доступны: {sorted(ADAPTERS)}")
    return cls()


@dataclass
class UpstreamResponse:
    """Открытый поток от апстрима.

    `close` обязателен: без него соединение и слот на стороне апстрима
    остаются занятыми. Именно это и есть утечка ёмкости из §3.3.9 —
    брошенные генерации продолжают жечь ресурсы, и проявляется это только
    под нагрузкой.
    """

    status: int
    headers: dict[str, str]
    events: AsyncIterator[StreamEvent]
    close: Any
    upstream: Upstream


class UpstreamClient:
    """HTTP-клиент к апстримам.

    Один общий `ClientSession` на процесс: создание сессии на запрос
    убивает keep-alive и добавляет полный TCP-хендшейк в TTFT.
    """

    def __init__(self, *, total_limit: int = 0, per_host_limit: int = 0) -> None:
        self._session: aiohttp.ClientSession | None = None
        self._total_limit = total_limit
        self._per_host_limit = per_host_limit

    async def start(self) -> None:
        connector = aiohttp.TCPConnector(
            limit=self._total_limit,            # 0 — без ограничения: лимиты у нас свои
            limit_per_host=self._per_host_limit,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
        )
        self._session = aiohttp.ClientSession(
            connector=connector,
            # Общий таймаут не задаём: стрим длинного ответа легитимно живёт
            # минуты. Раздельные таймауты выставляются на каждый запрос —
            # один общий либо режет длинные ответы, либо не ловит зависший
            # префилл (§3.3.8).
            timeout=aiohttp.ClientTimeout(total=None),
            auto_decompress=False,
        )

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def open_stream(
        self, request: ChatRequest, upstream: Upstream, adapter: Adapter
    ) -> UpstreamResponse:
        """Открывает поток и возвращает управление на первом байте.

        Важно: метод не ждёт конца ответа. Всё, что после — забота
        stream pipeline, который отдаёт кадры клиенту по мере поступления.
        """
        if self._session is None:
            raise RuntimeError("клиент не запущен")

        payload = adapter.build_payload(request, upstream)
        timeout = aiohttp.ClientTimeout(
            total=upstream.timeout_total_s,
            connect=upstream.timeout_connect_s,
            # Пауза между чтениями: ловит и зависший префилл, и застрявший
            # посреди генерации стрим.
            sock_read=upstream.timeout_ttft_s,
        )

        try:
            resp = await self._session.post(
                adapter.endpoint(upstream),
                json=payload,
                headers=adapter.headers(upstream),
                timeout=timeout,
            )
        except aiohttp.ServerTimeoutError as exc:
            raise UpstreamTimeout(f"таймаут соединения с {upstream.id}",
                                  retryable=True, upstream_id=upstream.id) from exc
        except aiohttp.ClientError as exc:
            raise UpstreamError(f"ошибка соединения с {upstream.id}: {exc}",
                                retryable=True, upstream_id=upstream.id) from exc

        if resp.status >= 400:
            body = (await resp.read())[:2048]
            resp.release()
            raise UpstreamError(
                f"апстрим {upstream.id} ответил {resp.status}: "
                f"{body.decode('utf-8', 'ignore')}",
                status=resp.status,
                # Восстановимые коды: имеет смысл повторить или деградировать.
                retryable=resp.status in (429, 500, 502, 503, 504),
                upstream_id=upstream.id,
            )

        async def events() -> AsyncIterator[StreamEvent]:
            buf = b""
            try:
                async for chunk in resp.content.iter_any():
                    buf += chunk
                    while b"\n\n" in buf:
                        frame, buf = buf.split(b"\n\n", 1)
                        for line in frame.split(b"\n"):
                            if not line.startswith(b"data: "):
                                continue
                            ev = adapter.parse_frame(line[6:])
                            if ev is not None:
                                yield ev
            except aiohttp.ServerTimeoutError as exc:
                raise UpstreamTimeout(
                    f"апстрим {upstream.id} замолчал дольше "
                    f"{upstream.timeout_ttft_s} с",
                    upstream_id=upstream.id,
                ) from exc

        async def close() -> None:
            if not resp.closed:
                resp.close()

        return UpstreamResponse(
            status=resp.status,
            headers=dict(resp.headers),
            events=events(),
            close=close,
            upstream=upstream,
        )
