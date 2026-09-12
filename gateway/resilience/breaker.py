"""Circuit breaker и правила деградации (§3.3.8).

Три решения с обоснованием.

**1. Размыкаем не только по ошибкам, но и по латентности.** Апстрим,
который отвечает успешно, но в десять раз медленнее бюджета, вреднее
того, который честно отдаёт 503: он держит наши слоты и тянет вниз
общий TTFT, не давая никакого сигнала.

**2. Ретрай допустим только до первого отданного токена.** После первого
токена стрим не идемпотентен: повтор породит дубли в уже начатом ответе.
Это та тонкость, которую большинство реализаций забывает.

**3. Гистерезис при восстановлении.** Полуоткрытое состояние пропускает
пробные запросы поштучно. Без него размыкание превращается в качели:
закрылись, получили залп, снова размокли.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

log = logging.getLogger(__name__)


class BreakerState(str, Enum):
    CLOSED = "closed"        # трафик идёт
    OPEN = "open"            # апстрим исключён
    HALF_OPEN = "half_open"  # пропускаем пробные запросы


@dataclass(slots=True)
class BreakerConfig:
    error_rate_threshold: float = 0.5    # доля ошибок в окне для размыкания
    min_samples: int = 20                # меньше — статистика недостоверна
    # Подряд идущие отказы — сигнал куда более сильный, чем доля в окне,
    # и ждать накопления статистики для явно мёртвого апстрима незачем:
    # каждая попытка стоит таймаута соединения. Порог низкий нарочно;
    # ложное срабатывание стоит одного полуоткрытого пробного запроса.
    consecutive_failures_to_open: int = 3
    latency_multiplier: float = 5.0      # во сколько раз хуже бюджета — уже отказ
    open_duration_s: float = 10.0
    half_open_successes: int = 3         # столько удач подряд для закрытия
    window_s: float = 30.0


@dataclass(slots=True)
class _Window:
    """Скользящее окно результатов. Хранит отметки времени, а не счётчики:
    иначе старые ошибки влияют на решение вечно."""

    window_s: float
    ok: list[float] = field(default_factory=list)
    fail: list[float] = field(default_factory=list)

    def add(self, success: bool, now: float) -> None:
        (self.ok if success else self.fail).append(now)
        self._trim(now)

    def _trim(self, now: float) -> None:
        edge = now - self.window_s
        while self.ok and self.ok[0] < edge:
            self.ok.pop(0)
        while self.fail and self.fail[0] < edge:
            self.fail.pop(0)

    def stats(self, now: float) -> tuple[int, float]:
        self._trim(now)
        total = len(self.ok) + len(self.fail)
        return total, (len(self.fail) / total if total else 0.0)


class CircuitBreaker:
    """Размыкатель на один апстрим."""

    def __init__(self, upstream_id: str, cfg: BreakerConfig | None = None) -> None:
        self.upstream_id = upstream_id
        self.cfg = cfg or BreakerConfig()
        self._state = BreakerState.CLOSED
        self._window = _Window(window_s=self.cfg.window_s)
        self._opened_at = 0.0
        self._half_open_ok = 0
        self._consecutive_failures = 0
        self._transitions = 0
        # Почему разомкнут: "errors", "consecutive" или "latency".
        # Различие нужно снаружи: медленный апстрим под общей перегрузкой
        # всё же лучше, чем отсутствие апстрима вовсе.
        self.open_reason = ""

    @property
    def state(self) -> BreakerState:
        return self._state

    @property
    def transitions(self) -> int:
        return self._transitions

    def allows(self, *, now: float | None = None) -> bool:
        now = now if now is not None else time.monotonic()
        if self._state is BreakerState.CLOSED:
            return True
        if self._state is BreakerState.OPEN:
            if now - self._opened_at >= self.cfg.open_duration_s:
                self._to(BreakerState.HALF_OPEN)
                self._half_open_ok = 0
                return True
            return False
        # HALF_OPEN: пропускаем, но по одному — решение о параллелизме
        # принимает вызывающий, здесь только разрешение.
        return True

    def record_success(self, *, latency_ms: float = 0.0, budget_ms: float = 0.0,
                       now: float | None = None) -> None:
        now = now if now is not None else time.monotonic()
        # Успех, который в разы хуже бюджета, считается отказом: иначе
        # медленный апстрим никогда не будет исключён.
        too_slow = (
            budget_ms > 0
            and latency_ms > budget_ms * self.cfg.latency_multiplier
        )
        if too_slow:
            self._window.add(False, now)
            log.debug("апстрим %s: ответ за %.0f мс при бюджете %.0f мс — засчитан отказ",
                      self.upstream_id, latency_ms, budget_ms)
            self._maybe_open(now, reason="latency")
            return

        self._window.add(True, now)
        self._consecutive_failures = 0
        if self._state is BreakerState.HALF_OPEN:
            self._half_open_ok += 1
            if self._half_open_ok >= self.cfg.half_open_successes:
                self._to(BreakerState.CLOSED)
                self._window = _Window(window_s=self.cfg.window_s)

    def record_failure(self, *, hard: bool = True, now: float | None = None) -> None:
        """Регистрирует отказ.

        `hard` отличает отказ апстрима (соединение не установилось, пришёл
        5xx) от неполного результата, у которого может быть множество
        причин на нашей стороне — наш таймаут, отмена, разрыв в середине.

        Различие не косметическое. Правило «три отказа подряд размыкают»
        задумано против явно мёртвого апстрима, где каждая попытка стоит
        таймаута соединения. Применённое к неполным стримам, оно под
        нагрузкой исключало живые апстримы: несколько оборванных ответов
        подряд — обычное дело при перегрузке, а не признак поломки.
        """
        now = now if now is not None else time.monotonic()
        self._window.add(False, now)
        if hard:
            self._consecutive_failures += 1
        if self._state is BreakerState.HALF_OPEN:
            # Пробный запрос провалился — снова размыкаем, не дожидаясь
            # накопления статистики.
            self._open(now)
            return
        self._maybe_open(now)

    def _maybe_open(self, now: float, *, reason: str = "errors") -> None:
        if self._state is not BreakerState.CLOSED:
            return
        if self._consecutive_failures >= self.cfg.consecutive_failures_to_open:
            self._open(now, reason="consecutive")
            return
        total, rate = self._window.stats(now)
        if total >= self.cfg.min_samples and rate >= self.cfg.error_rate_threshold:
            self._open(now, reason=reason)

    def _open(self, now: float, *, reason: str = "errors") -> None:
        self._opened_at = now
        self._consecutive_failures = 0
        self.open_reason = reason
        self._to(BreakerState.OPEN)
        log.warning("апстрим %s исключён на %.0f с", self.upstream_id,
                    self.cfg.open_duration_s)

    def _to(self, state: BreakerState) -> None:
        if state is not self._state:
            self._state = state
            self._transitions += 1

    def snapshot(self, *, now: float | None = None) -> dict[str, object]:
        now = now if now is not None else time.monotonic()
        total, rate = self._window.stats(now)
        return {
            "upstream": self.upstream_id,
            "state": self._state.value,
            "samples": total,
            "error_rate": round(rate, 3),
            "consecutive_failures": self._consecutive_failures,
            "open_reason": self.open_reason,
            "transitions": self._transitions,
        }


class BreakerRegistry:
    def __init__(self, cfg: BreakerConfig | None = None) -> None:
        self._cfg = cfg or BreakerConfig()
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(self, upstream_id: str) -> CircuitBreaker:
        b = self._breakers.get(upstream_id)
        if b is None:
            b = CircuitBreaker(upstream_id, self._cfg)
            self._breakers[upstream_id] = b
        return b

    def healthy(self, upstream_ids: list[str]) -> list[str]:
        """Апстримы, которым можно слать запросы.

        **Никогда не возвращает пустой список, если хоть один апстрим
        разомкнут лишь по латентности.** Это защита от лавины: под общей
        перегрузкой медленными становятся все апстримы сразу, размыкатели
        исключают всех, и деградация «медленно» превращается в отказ
        «недоступно».

        Перегрузка — это работа admission control (§3.3.3), а не
        размыкателя. Размыкатель существует для того, чтобы обойти
        *сломанный* апстрим, а не чтобы выключить сервис, когда нагрузка
        выше расчётной. Механизмы, разумные по отдельности, вместе дают
        лавину — это тот самый случай.
        """
        live = [uid for uid in upstream_ids if self.get(uid).allows()]
        if live:
            return live
        # Все разомкнуты. Возвращаем тех, кто разомкнут только за
        # медлительность: медленный ответ лучше пятисотки.
        slow_only = [
            uid for uid in upstream_ids
            if self.get(uid).open_reason == "latency"
        ]
        return slow_only

    def snapshot(self) -> list[dict[str, object]]:
        return [b.snapshot() for b in self._breakers.values()]
