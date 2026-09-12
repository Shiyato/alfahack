"""Stream pipeline (§3.3.9).

Отвечает за то, что происходит между открытым потоком от апстрима и
клиентом: подсчёт токенов на лету, пометка деградации, корректная
обработка разрыва соединения.

**Главное требование — отмена запроса к апстриму при отключении клиента.**
Иначе брошенные генерации продолжают жечь ёмкость. Это классическая
утечка, которая проявляется именно под нагрузкой: пока запросов мало,
её не видно, а под нагрузкой мощность утекает именно туда, где её не
ищут.

**Ретрай возможен только до первого отданного токена.** После первого
токена стрим не идемпотентен: повтор породит дубли в уже начатом ответе.
Флаг `first_token_sent` — это граница, за которой деградация уже
невозможна, и её приходится соблюдать явно.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

from ..upstream.adapter import StreamEvent, UpstreamResponse

log = logging.getLogger(__name__)


@dataclass(slots=True)
class StreamResult:
    """Итог стрима — то, по чему считаются квоты и метрики."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    itl_ms: float = 0.0
    finished: bool = False          # дошёл ли стрим до конца
    client_disconnected: bool = False
    upstream_id: str = ""
    degraded: bool = False
    degraded_reason: str = ""

    @property
    def goodput_counted(self) -> bool:
        """В goodput засчитываются только полностью завершённые запросы
        (§2.1.2). Если запрос отвалился на полпути, потраченные ресурсы
        списываются впустую, и считать их полезной работой нельзя."""
        return self.finished and not self.client_disconnected


class StreamPipeline:
    """Перекладывает кадры апстрима клиенту, считая по дороге.

    Кадры отдаются **как есть** (`ev.raw`): клиент получает ровно то,
    что прислал провайдер, включая поля, которых мы не понимаем. Разбор
    нужен нам для учёта, а не для переписывания ответа.
    """

    def __init__(self, *, count_tokens_locally: bool = True) -> None:
        self._count_locally = count_tokens_locally

    async def relay(
        self,
        response: UpstreamResponse,
        result: StreamResult,
        *,
        started_at: float,
        degraded_notice: dict | None = None,
    ) -> AsyncIterator[bytes]:
        """Основной цикл. Генератор: отдаёт байты клиенту по мере
        поступления, ничего не буферизуя целиком.

        Отмена приходит сюда как GeneratorExit или CancelledError — ASGI
        закрывает генератор, когда клиент отвалился. Оба случая обязаны
        закрыть поток к апстриму.
        """
        result.upstream_id = response.upstream.id
        first_token_at = 0.0
        last_token_at = 0.0
        emitted = 0

        try:
            # Пометка деградации идёт первым кадром, до содержимого.
            # Клиент не должен молча получать ответ другой модели —
            # для банковского контура это обязательное требование (§3.3.8).
            if degraded_notice is not None:
                result.degraded = True
                result.degraded_reason = degraded_notice.get("reason", "")
                yield b"data: " + json.dumps(
                    degraded_notice, ensure_ascii=False, separators=(",", ":")
                ).encode() + b"\n\n"

            async for ev in response.events:
                now = time.monotonic()
                if ev.kind == "usage":
                    # Провайдер сообщил точный расход — он надёжнее нашей
                    # оценки, по нему и корректируем квоту.
                    result.prompt_tokens = ev.prompt_tokens or result.prompt_tokens
                    result.completion_tokens = ev.completion_tokens or result.completion_tokens
                    yield ev.raw
                    continue

                if ev.kind == "done":
                    result.finished = True
                    yield ev.raw
                    continue

                if ev.content:
                    if not emitted:
                        first_token_at = now
                        result.ttft_ms = (now - started_at) * 1000.0
                    emitted += 1
                    last_token_at = now

                if ev.finish_reason:
                    result.finished = True

                yield ev.raw

            if self._count_locally and not result.completion_tokens:
                result.completion_tokens = emitted

        except (GeneratorExit, Exception) as exc:
            # Клиент отвалился или упал апстрим — в обоих случаях поток
            # к апстриму обязан закрыться, иначе генерация продолжится
            # в пустоту.
            if isinstance(exc, GeneratorExit):
                result.client_disconnected = True
                log.debug("клиент отключился, запрос к %s отменён", response.upstream.id)
            raise
        finally:
            await response.close()
            end = time.monotonic()
            result.total_ms = (end - started_at) * 1000.0
            if emitted > 1 and first_token_at and last_token_at > first_token_at:
                result.itl_ms = (last_token_at - first_token_at) * 1000.0 / (emitted - 1)


@dataclass(slots=True)
class RetryGuard:
    """Граница, за которой повтор запрещён (§3.3.8).

    Вынесено в отдельный объект нарочно: это правило легко нарушить,
    добавив ретрай в другом месте цепочки, и тогда клиент получит
    дубликат посреди уже начатого ответа.
    """

    first_token_sent: bool = False
    attempts: int = 0
    max_attempts: int = 2

    def may_retry(self) -> bool:
        return (not self.first_token_sent) and self.attempts < self.max_attempts

    def mark_attempt(self) -> None:
        self.attempts += 1

    def mark_first_token(self) -> None:
        self.first_token_sent = True


def degradation_notice(
    *, original_model: str, actual_model: str, reason: str
) -> dict:
    """Кадр-предупреждение о деградации.

    Промышленные шлюзы позволяют при подмене модели менять параметры
    вроде temperature и max_tokens. Технически удобно, но семантически
    означает, что клиент **молча получает ответ на другой запрос**. Для
    банковского контура так нельзя: подмена обязана быть видимой.
    """
    return {
        "object": "gateway.degradation",
        "degraded": True,
        "requested_model": original_model,
        "served_model": actual_model,
        "reason": reason,
    }
