"""Квоты по токенам (§3.3.3.5).

Три решения, каждое с обоснованием:

1. **Считаем токены, а не запросы.** Ограничение по RPM не отражает риск:
   запрос с контекстом на 100k токенов и запрос на 200 нагружают систему
   на два порядка по-разному. При этом счёт запросов тоже нужен — он ловит
   шквал лёгких запросов, который по токенам выглядит безобидно.

2. **Token bucket, а не окно.** Fixed window даёт эффект границы: двойная
   нагрузка на стыке окон. Главное же — агентская нагрузка принципиально
   всплесковая: агент выпускает серию запросов подряд по мере обхода
   инструментов, и окно резало бы легитимные серии. У ведра всплеск
   ограничен его размером, что честнее.

3. **Списываем оценку вперёд, корректируем по факту.** Число выходных
   токенов заранее неизвестно. Без предварительного списания квота не
   защищает вовсе: все параллельные запросы пройдут проверку до того,
   как хоть один завершится.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(slots=True)
class TokenBucket:
    """Ведро с непрерывным пополнением.

    Реализовано без фонового таймера: уровень пересчитывается лениво в
    момент обращения. Это важно для горячего пути — тысяча тенантов не
    порождает тысячу задач в цикле событий.
    """

    capacity: float
    refill_per_second: float
    level: float = field(default=0.0)
    updated_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        if self.level <= 0:
            self.level = self.capacity

    def _refill(self, now: float) -> None:
        elapsed = now - self.updated_at
        if elapsed > 0:
            self.level = min(self.capacity, self.level + elapsed * self.refill_per_second)
            self.updated_at = now

    def try_take(self, amount: float, now: float | None = None) -> bool:
        now = now if now is not None else time.monotonic()
        self._refill(now)
        if self.level >= amount:
            self.level -= amount
            return True
        return False

    def give_back(self, amount: float, now: float | None = None) -> None:
        """Возврат переоценённого. Вызывается по завершении запроса, когда
        фактический расход известен."""
        now = now if now is not None else time.monotonic()
        self._refill(now)
        self.level = min(self.capacity, self.level + amount)

    def retry_after_s(self, amount: float, now: float | None = None) -> float:
        """Через сколько секунд в ведре хватит места под `amount`.

        Это содержимое заголовка Retry-After. Смысл — чтобы клиент мог
        выстроить автоматическую логику повтора, а не гадать (§3.3.3.6).
        """
        now = now if now is not None else time.monotonic()
        self._refill(now)
        missing = amount - self.level
        if missing <= 0:
            return 0.0
        if self.refill_per_second <= 0:
            return float("inf")
        return missing / self.refill_per_second

    def remaining(self, now: float | None = None) -> float:
        self._refill(now if now is not None else time.monotonic())
        return self.level


@dataclass(slots=True)
class QuotaDecision:
    allowed: bool
    reason: str = ""
    retry_after_s: float = 0.0
    limit_kind: str = ""
    remaining_tokens: float = 0.0
    remaining_requests: float = 0.0
    reserved_tokens: int = 0


class QuotaLedger:
    """Учёт квот по измерениям: тенант, команда, модель (§3.3.3.5).

    Проверка идёт по всем применимым вёдрам, и **списание происходит только
    если прошли все**. Иначе запрос, отклонённый по последнему измерению,
    успел бы списать квоту по первым — и чужие запросы получили бы отказ
    из-за него.
    """

    def __init__(self) -> None:
        self._tokens: dict[str, TokenBucket] = {}
        self._requests: dict[str, TokenBucket] = {}
        self._concurrent: dict[str, int] = {}

    def _bucket(
        self, store: dict[str, TokenBucket], key: str, per_minute: float, burst: float
    ) -> TokenBucket:
        b = store.get(key)
        if b is None or b.refill_per_second != per_minute / 60.0:
            # Смена лимита конфигом создаёт ведро заново, сохраняя долю
            # заполнения: иначе hot-reload квот либо дарил бы полное ведро,
            # либо обнулял бы легитимный остаток.
            fill = (b.level / b.capacity) if b and b.capacity > 0 else 1.0
            b = TokenBucket(capacity=per_minute * burst / 60.0 * 60.0,
                            refill_per_second=per_minute / 60.0)
            b.capacity = per_minute * burst
            b.level = b.capacity * fill
            store[key] = b
        return b

    def check_and_reserve(
        self,
        *,
        keys: list[str],
        tokens_per_minute: float | None,
        requests_per_minute: float | None,
        max_concurrent: int | None,
        estimated_tokens: int,
        burst_multiplier: float = 2.0,
    ) -> QuotaDecision:
        now = time.monotonic()

        # 1. Конкурентность — самая дешёвая проверка, делаем первой.
        if max_concurrent is not None:
            for k in keys:
                if self._concurrent.get(k, 0) >= max_concurrent:
                    return QuotaDecision(
                        allowed=False,
                        reason=f"превышен лимит одновременных запросов ({max_concurrent})",
                        retry_after_s=1.0,
                        limit_kind="concurrency",
                    )

        # 2. Проверяем все вёдра, ничего не списывая.
        token_buckets: list[TokenBucket] = []
        request_buckets: list[TokenBucket] = []
        if tokens_per_minute:
            token_buckets = [self._bucket(self._tokens, k, tokens_per_minute, burst_multiplier)
                             for k in keys]
        if requests_per_minute:
            request_buckets = [self._bucket(self._requests, k, requests_per_minute, burst_multiplier)
                               for k in keys]

        for b in token_buckets:
            if b.remaining(now) < estimated_tokens:
                return QuotaDecision(
                    allowed=False,
                    reason=f"исчерпана квота по токенам ({tokens_per_minute}/мин)",
                    retry_after_s=b.retry_after_s(estimated_tokens, now),
                    limit_kind="tokens",
                    remaining_tokens=b.remaining(now),
                )
        for b in request_buckets:
            if b.remaining(now) < 1:
                return QuotaDecision(
                    allowed=False,
                    reason=f"исчерпана квота по запросам ({requests_per_minute}/мин)",
                    retry_after_s=b.retry_after_s(1, now),
                    limit_kind="requests",
                    remaining_requests=b.remaining(now),
                )

        # 3. Все прошли — списываем.
        for b in token_buckets:
            b.try_take(estimated_tokens, now)
        for b in request_buckets:
            b.try_take(1, now)
        for k in keys:
            self._concurrent[k] = self._concurrent.get(k, 0) + 1

        return QuotaDecision(
            allowed=True,
            reserved_tokens=estimated_tokens,
            remaining_tokens=token_buckets[0].remaining(now) if token_buckets else float("inf"),
            remaining_requests=request_buckets[0].remaining(now) if request_buckets else float("inf"),
        )

    def settle(self, keys: list[str], reserved: int, actual: int) -> None:
        """Корректировка по факту завершения (§3.3.3.5).

        Если фактический расход меньше оценки — разница возвращается.
        Если больше — списывается дополнительно, и ведро может уйти в
        минус. Это правильно: перерасход должен отразиться на следующих
        запросах того же тенанта, а не раствориться.
        """
        delta = reserved - actual
        now = time.monotonic()
        for k in keys:
            if (b := self._tokens.get(k)) is not None:
                if delta > 0:
                    b.give_back(delta, now)
                elif delta < 0:
                    b._refill(now)
                    b.level -= -delta
            self._concurrent[k] = max(0, self._concurrent.get(k, 1) - 1)

    def concurrent(self, key: str) -> int:
        return self._concurrent.get(key, 0)
