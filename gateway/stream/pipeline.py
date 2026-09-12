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

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

from ..upstream.adapter import StreamEvent, UpstreamResponse

log = logging.getLogger(__name__)

# Сколько служебных кадров копить, прежде чем отдать их клиенту, не
# дождавшись содержимого. Предел нужен против апстрима, который шлёт
# метаданные бесконечно: иначе буфер растёт молча.
MAX_STARTUP_FRAMES = 64


@dataclass(slots=True)
class StreamResult:
    """Итог стрима — то, по чему считаются квоты и метрики."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: float = 0.0
    # TTFT, отсчитанный от момента, когда запрос ушёл апстриму. Разница
    # с `ttft_ms` — и есть наши накладные расходы: очередь, auth,
    # admission, роутинг, хеширование. Единственная метрика, за которую
    # отвечаем мы (§2.1).
    upstream_ttft_ms: float = 0.0
    total_ms: float = 0.0
    itl_ms: float = 0.0
    finished: bool = False          # дошёл ли стрим до конца
    client_disconnected: bool = False
    upstream_id: str = ""
    degraded: bool = False
    degraded_reason: str = ""

    @property
    def overhead_ms(self) -> float:
        """Наше время минус время апстрима.

        Если эта величина растёт, проблема в инфраструктуре, а не в
        модели. Именно её разделение и есть ключевой приём дашборда
        (§3.3.10): без него рост latency невозможно атрибутировать.
        """
        if not self.ttft_ms or not self.upstream_ttft_ms:
            return 0.0
        return max(0.0, self.ttft_ms - self.upstream_ttft_ms)

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
        upstream_started_at: float | None = None,
        degraded_notice: dict | None = None,
        guard: "RetryGuard | None" = None,
    ) -> AsyncIterator[bytes]:
        """Основной цикл. Генератор: отдаёт байты клиенту по мере
        поступления, ничего не буферизуя целиком.

        Отмена приходит сюда как GeneratorExit или CancelledError — ASGI
        закрывает генератор, когда клиент отвалился. Оба случая обязаны
        закрыть поток к апстриму.
        """
        result.upstream_id = response.upstream.id
        cancelled = False
        first_token_at = 0.0
        last_token_at = 0.0
        emitted = 0

        # Буфер стартовых кадров.
        #
        # Стрим начинается со служебных кадров — роль ассистента,
        # идентификатор ответа, метаданные запуска. Содержимого в них нет,
        # и если апстрим упадёт сразу после них, попытку можно повторить
        # на другом. Но кадры, уже отданные клиенту, отозвать нельзя:
        # после переподключения он получил бы служебные кадры обеих
        # попыток, включая два кадра с ролью.
        #
        # Поэтому начало стрима копится в буфере и уходит клиенту вместе
        # с первым кадром содержимого — то есть в момент, когда стало
        # ясно, что эта попытка и есть окончательная. Буфер неудачной
        # попытки просто отбрасывается.
        #
        # Приём взят из Bifrost (Apache-2.0), где стартовые метаданные
        # буферизуются ровно для того, чтобы ошибка после них ещё могла
        # попасть в логику повторов.
        startup: list[bytes] = []
        released = False

        def flush() -> list[bytes]:
            """Отдаёт накопленное начало стрима вместе с пометкой деградации."""
            nonlocal released
            released = True
            out: list[bytes] = []
            if degraded_notice is not None:
                # Пометка идёт перед содержимым: клиент не должен молча
                # получать ответ другой модели (§3.3.8).
                result.degraded = True
                result.degraded_reason = degraded_notice.get("reason", "")
                out.append(b"data: " + json.dumps(
                    degraded_notice, ensure_ascii=False, separators=(",", ":")
                ).encode() + b"\n\n")
            out.extend(startup)
            startup.clear()
            return out

        try:
            async for ev in response.events:
                now = time.monotonic()
                if guard is not None:
                    # Граница повторов сдвигается здесь, а не у потребителя:
                    # только тут видно, есть ли в кадре содержимое.
                    guard.observe_event(ev)
                if ev.kind == "usage":
                    # Провайдер сообщил точный расход — он надёжнее нашей
                    # оценки, по нему и корректируем квоту.
                    result.prompt_tokens = ev.prompt_tokens or result.prompt_tokens
                    result.completion_tokens = ev.completion_tokens or result.completion_tokens

                if ev.content:
                    if not emitted:
                        first_token_at = now
                        result.ttft_ms = (now - started_at) * 1000.0
                        if upstream_started_at is not None:
                            result.upstream_ttft_ms = (now - upstream_started_at) * 1000.0
                    emitted += 1
                    last_token_at = now

                if ev.finish_reason or ev.kind == "done":
                    result.finished = True

                if released:
                    yield ev.raw
                    continue

                # Содержимое, конец потока или расход токенов означают,
                # что начало стрима состоялось и его пора отдать.
                if ev.content or ev.kind in ("done", "usage") or ev.finish_reason:
                    for chunk in flush():
                        yield chunk
                    yield ev.raw
                    continue

                startup.append(ev.raw)
                # Предохранитель: апстрим, бесконечно шлющий служебные
                # кадры, не должен копиться в памяти молча.
                if len(startup) > MAX_STARTUP_FRAMES:
                    for chunk in flush():
                        yield chunk

            if not released and startup:
                # Поток кончился, не дав ни одного кадра содержимого.
                # Отдаём накопленное: клиент должен увидеть хоть что-то.
                for chunk in flush():
                    yield chunk

            if self._count_locally and not result.completion_tokens:
                result.completion_tokens = emitted

        except BaseException as exc:
            cancelled = True
            # Клиент отвалился или упал апстрим — в обоих случаях поток
            # к апстриму обязан закрыться, иначе генерация продолжится
            # в пустоту.
            #
            # Отмена приходит сюда двумя способами, и различать их нужно.
            # Когда потребитель закрывает генератор напрямую, приходит
            # GeneratorExit. Но в реальной цепочке `relay` обёрнут в другой
            # генератор (ASGI отдаёт клиенту его), и тогда вложенный
            # генератор получает **CancelledError**, а не GeneratorExit.
            #
            # Первая версия проверяла только GeneratorExit, и разрывы
            # клиентом учитывались как «неполный ответ»: ёмкость
            # освобождалась правильно, но метрика разрывов всегда была
            # нулевой, а неполные ответы засчитывались размыкателю как
            # отказы апстрима — то есть клиент, закрывший вкладку, портил
            # репутацию исправному апстриму.
            if isinstance(exc, (GeneratorExit, asyncio.CancelledError)):
                result.client_disconnected = True
                log.debug("клиент отключился, запрос к %s отменён", response.upstream.id)
            raise
        finally:
            # Закрытие апстрима: способ зависит от того, как мы сюда попали.
            #
            # При нормальном завершении ждём обычным await — так закрытие
            # гарантированно произошло к моменту возврата.
            #
            # При отмене ждать нельзя. Когда потребитель закрывает
            # генератор, блок finally выполняется **асинхронно, уже после
            # возврата из aclose()**, и любой await внутри него
            # откладывается на неопределённый срок: поток к апстриму
            # остаётся открытым, а брошенная генерация продолжает жечь
            # ёмкость ровно так, как запрещает §3.3.9. Поэтому закрытие
            # ставится задачей в цикл событий и выполняется независимо от
            # судьбы текущей корутины.
            if cancelled:
                _schedule_close(response)
            else:
                await response.close()
            end = time.monotonic()
            result.total_ms = (end - started_at) * 1000.0
            if emitted > 1 and first_token_at and last_token_at > first_token_at:
                result.itl_ms = (last_token_at - first_token_at) * 1000.0 / (emitted - 1)


def _schedule_close(response: UpstreamResponse) -> None:
    """Закрывает поток к апстриму, не завися от того, доживёт ли текущая
    корутина до следующего await."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    task = loop.create_task(response.close())
    # Ссылку держим до завершения: без неё сборщик мусора может забрать
    # задачу раньше, чем она выполнится.
    _pending_closes.add(task)
    task.add_done_callback(_pending_closes.discard)


_pending_closes: set[asyncio.Task] = set()


@dataclass(slots=True)
class RetryGuard:
    """Граница, за которой повтор запрещён (§3.3.8).

    Вынесено в отдельный объект нарочно: правило легко нарушить, добавив
    ретрай в другом месте цепочки, и тогда клиент получит дубликат
    посреди уже начатого ответа.

    **Где именно проходит граница** — тонкость, которую пришлось
    исправлять. Стрим начинается со служебных кадров: роль ассистента,
    идентификатор ответа, у некоторых провайдеров — метаданные запуска.
    Содержимого в них нет, клиент по ним ничего не увидел, и повтор в
    этот момент совершенно безопасен.

    Первая версия помечала границу по **любому** отданному кадру, из-за
    чего отказ сразу после кадра с ролью делал деградацию невозможной,
    хотя терять было нечего. Граница проходит по первому кадру
    **с содержимым**.

    Идея взята из Bifrost (Apache-2.0), где стартовые метаданные
    буферизуются именно для того, чтобы ошибка, пришедшая после них,
    ещё могла попасть в логику повторов.
    """

    first_content_sent: bool = False
    attempts: int = 0
    max_attempts: int = 2

    def may_retry(self) -> bool:
        return (not self.first_content_sent) and self.attempts < self.max_attempts

    def mark_attempt(self) -> None:
        self.attempts += 1

    def observe_event(self, event: StreamEvent) -> None:
        """Сдвигает границу только на кадре с содержимым."""
        if event.content:
            self.first_content_sent = True

    def backoff_s(self, *, initial: float = 0.2, maximum: float = 5.0,
                  retry_after_s: float | None = None) -> float:
        """Задержка перед следующей попыткой.

        Если апстрим назвал время сам — берём его: он знает, когда
        сдвинется окно квоты, а мы только догадываемся.

        Иначе экспоненциальная задержка с джиттером. Джиттер обязателен:
        без него все запросы, отвалившиеся одновременно, повторятся тоже
        одновременно и создадут вторую волну ровно той же формы.
        """
        if retry_after_s is not None:
            return min(retry_after_s, maximum)
        base = min(initial * (2 ** max(0, self.attempts - 1)), maximum)
        return base * random.uniform(0.8, 1.2)


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
