"""Приоритетная очередь и селективная отдача (§3.3.3.4, §3.3.3.7).

Два механизма, которые оказываются одной сущностью.

**Селективная отдача.** Слепая отдача — маршрутизировать запрос апстриму
сразу по прибытии. Так работают round-robin, least-load и большинство
прокси. Для CPU-нагрузок с однородным временем обработки это нормально;
для LLM — нет, потому что длительность запроса зависит от длины вывода,
которую невозможно предсказать. Инстанс с короткой на вид очередью может
обрабатывать её долго, и запросы копятся у него, пока соседи простаивают.

Значит, очередь должна жить **на прокси**, а не размазываться по
апстримам: только запрос, который ещё не начали обрабатывать, можно
перенаправить.

**Приоритеты.** А как только очередь оказалась у нас, в ней появляется
место, где живут классы обслуживания (§2.2). Без собственной очереди
приоритизацию реализовать физически негде.

**Fair-share внутри класса** нужен, чтобы один шумный сосед не съел весь
класс. Реализован через выбор тенанта с наименьшим числом обслуженных
запросов в текущем окне — дёшево и достаточно.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from ..core.domain import ServiceClass


@dataclass(slots=True, order=False)
class QueuedRequest:
    """Запрос, ожидающий свободного апстрима."""

    service_class: ServiceClass
    tenant_id: str
    enqueued_at: float
    estimated_tokens: int
    payload: Any = None
    future: asyncio.Future | None = None
    seq: int = 0

    @property
    def waited_s(self) -> float:
        return time.monotonic() - self.enqueued_at


class QueueFull(Exception):
    """Очередь переполнена: отказ до постановки, а не после ожидания."""


class QueueTimeout(Exception):
    """Ожидание превысило отведённое время."""


class PriorityQueue:
    """Очередь с приоритетом по классу и fair-share между тенантами.

    Устройство: по одной FIFO-очереди на пару (класс, тенант). При выдаче
    берётся самый приоритетный непустой класс, а внутри него — тенант с
    наименьшим числом обслуженных в текущем окне.

    **Почему не одна куча с ключом при постановке.** Первая версия считала
    ключ сортировки в момент `put`, подмешивая туда счётчик обслуженных.
    Это не работает: если шумный тенант выложил пять запросов до того, как
    обслужили хоть один, все пять получили одинаковый нулевой счётчик и
    прошли подряд — fair-share не сработал вовсе. Решение о справедливости
    обязано приниматься **в момент выдачи**, когда счётчики актуальны.

    Голодания между тенантами нет по построению. Между классами голодание
    возможно и намеренно: по §4 под перегрузкой `batch` растягивается, а
    `background` отбрасывается. Это описанное поведение, а не дефект.
    """

    def __init__(self, *, max_depth: int = 1000) -> None:
        self._max_depth = max_depth
        self._counter = itertools.count()
        # (класс → тенант → дек запросов)
        self._lanes: dict[ServiceClass, dict[str, deque[QueuedRequest]]] = {
            c: {} for c in ServiceClass
        }
        self._served: dict[str, int] = {}
        self._depth_by_class: dict[ServiceClass, int] = {c: 0 for c in ServiceClass}
        self._total = 0
        self._waiters: list[asyncio.Future] = []

    def __len__(self) -> int:
        return self._total

    def depth(self, service_class: ServiceClass | None = None) -> int:
        if service_class is None:
            return self._total
        return self._depth_by_class[service_class]

    def put(self, req: QueuedRequest) -> None:
        if self._total >= self._max_depth:
            raise QueueFull(f"глубина очереди достигла {self._max_depth}")
        req.seq = next(self._counter)
        lane = self._lanes[req.service_class].setdefault(req.tenant_id, deque())
        lane.append(req)
        self._depth_by_class[req.service_class] += 1
        self._total += 1
        self._wake_one()

    def _select(self) -> tuple[ServiceClass, str] | None:
        """Какой запрос вышел бы следующим. Вынесено отдельно, чтобы
        `peek` и `get_nowait` не разъезжались: правило выбора должно
        быть одно."""
        for cls in sorted(ServiceClass, key=lambda c: c.priority):
            lanes = self._lanes[cls]
            if not self._depth_by_class[cls]:
                continue
            best_tenant = min(
                (t for t, d in lanes.items() if d),
                key=lambda t: (self._served.get(t, 0), lanes[t][0].seq),
            )
            return cls, best_tenant
        return None

    def peek(self) -> QueuedRequest | None:
        """Голова очереди без изъятия.

        Нужна диспетчеру: ожидающий обязан убедиться, что следующий —
        именно его запрос, прежде чем забирать слот. Забирать чужой и
        пытаться его кому-то передать — верный способ потерять запрос
        и устроить живую блокировку.
        """
        sel = self._select()
        if sel is None:
            return None
        cls, tenant = sel
        return self._lanes[cls][tenant][0]

    def get_nowait(self) -> QueuedRequest | None:
        for cls in sorted(ServiceClass, key=lambda c: c.priority):
            lanes = self._lanes[cls]
            if not self._depth_by_class[cls]:
                continue
            # Тенант с наименьшим числом обслуженных; при равенстве —
            # тот, чей запрос пришёл раньше.
            best_tenant = min(
                (t for t, d in lanes.items() if d),
                key=lambda t: (self._served.get(t, 0), lanes[t][0].seq),
            )
            req = lanes[best_tenant].popleft()
            if not lanes[best_tenant]:
                del lanes[best_tenant]
            self._depth_by_class[cls] -= 1
            self._total -= 1
            self._served[best_tenant] = self._served.get(best_tenant, 0) + 1
            return req
        return None

    def remove(self, req: QueuedRequest) -> bool:
        """Изъятие конкретного запроса: клиент отвалился, пока тот ждал.

        Без этого брошенный запрос дождётся своей очереди и займёт слот
        апстрима, который никому не нужен, — та же утечка ёмкости, что и
        зомби-запрос в стриме (§3.3.9), только на входе.
        """
        lane = self._lanes[req.service_class].get(req.tenant_id)
        if not lane:
            return False
        try:
            lane.remove(req)
        except ValueError:
            return False
        if not lane:
            del self._lanes[req.service_class][req.tenant_id]
        self._depth_by_class[req.service_class] -= 1
        self._total -= 1
        return True

    def reset_fairshare(self) -> None:
        """Сброс окна fair-share.

        Без сброса счётчик обслуженных растёт вечно, и тенант, активный с
        утра, навсегда уступает тому, кто подключился вечером.
        """
        self._served.clear()

    # --- Асинхронное ожидание ---

    def _wake_one(self) -> None:
        while self._waiters:
            fut = self._waiters.pop(0)
            if not fut.done():
                fut.set_result(None)
                return

    async def wait_for_item(self, timeout: float | None = None) -> None:
        if self._total:
            return
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._waiters.append(fut)
        try:
            await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError as exc:
            raise QueueTimeout("нет запросов в очереди") from exc
        finally:
            if fut in self._waiters:
                self._waiters.remove(fut)

    def snapshot(self) -> dict[str, int]:
        return {c.value: self._depth_by_class[c] for c in ServiceClass}


class SelectiveDispatcher:
    """Отдаёт запросы только готовым апстримам (§3.3.3.7).

    Ключевая деталь — **какой порог считать признаком занятости**.
    Естественная идея ограничить число одновременных запросов константой
    работает плохо: сколько запросов вмещает движок, зависит от суммы
    входных и выходных токенов, а выход заранее неизвестен. На одной
    модели и одном датасете предельное число одновременных запросов
    гуляло от 20 до 50. Любая константа будет либо душить, либо
    перегружать.

    Поэтому признак — **бинарный и адаптивный**: занят ли апстрим прямо
    сейчас. Источник признака зависит от того, что нам видно (§3.3.6.8),
    и подставляется извне функцией `ready_upstreams`.

    **Каждый ожидающий забирает только свой запрос.** Первая версия
    вынимала из очереди голову и, если та принадлежала другому, пыталась
    передать ей слот через future, которого ни у кого не было. Чужой
    запрос при этом терялся, а его собственная корутина продолжала
    крутиться. Под нагрузкой это дало живую блокировку: 14 миллионов
    оборотов на четыре тысячи запросов и среднее ожидание в очереди
    30 секунд при пустых апстримах. Ожидающий обязан лишь **смотреть**
    на голову очереди и забирать её, только если это он сам.
    """

    def __init__(
        self,
        queue: PriorityQueue,
        *,
        max_wait_s: float = 30.0,
        poll_interval_s: float = 0.002,
    ) -> None:
        self._queue = queue
        self._max_wait = max_wait_s
        self._poll = poll_interval_s
        self._dispatched = 0
        self._timed_out = 0
        self._spins = 0
        self._wait_samples: list[float] = []

    async def acquire_slot(self, req: QueuedRequest, ready_upstreams):
        """Ждёт, пока появится готовый апстрим И подойдёт очередь.

        Запрос, стоящий в очереди, ещё не тратит ресурсы апстрима —
        поэтому отказ здесь дёшев, а отказ после начала обработки
        означал бы выброшенную работу (§2.1.2).
        """
        self._queue.put(req)
        deadline = time.monotonic() + self._max_wait

        try:
            while True:
                if self._queue.peek() is req:
                    candidates = ready_upstreams()
                    if candidates:
                        taken = self._queue.get_nowait()
                        # Между peek и get никто не мог вклиниться:
                        # обе операции синхронны и цикл событий их не
                        # разрывает. Проверка — на случай будущих правок.
                        if taken is not req:
                            if taken is not None:
                                self._queue.put(taken)
                            await asyncio.sleep(0)
                            continue
                        wait = req.waited_s
                        self._dispatched += 1
                        self._wait_samples.append(wait)
                        if len(self._wait_samples) > 2000:
                            del self._wait_samples[:1000]
                        return candidates[0], wait

                if time.monotonic() >= deadline:
                    self._timed_out += 1
                    raise QueueTimeout(
                        f"не дождались готового апстрима за {self._max_wait} с"
                    )
                self._spins += 1
                await asyncio.sleep(self._poll)
        except BaseException:
            # Клиент отвалился или истекло время — запрос обязан покинуть
            # очередь, иначе он дождётся своей очереди и займёт слот
            # апстрима, который уже никому не нужен.
            self._queue.remove(req)
            raise

    def stats(self) -> dict[str, float]:
        samples = self._wait_samples[-1000:]
        return {
            "dispatched": self._dispatched,
            "timed_out": self._timed_out,
            "spins": self._spins,
            "queue_wait_avg_ms": (sum(samples) / len(samples) * 1000) if samples else 0.0,
            "queue_wait_max_ms": (max(samples) * 1000) if samples else 0.0,
        }
