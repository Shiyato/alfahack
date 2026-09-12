"""Оценка нагрузки и раннее отклонение (§3.3.3.1–3.3.3.3).

Три идеи, и третья — самая ценная.

**1. Нагрузка измеряется соблюдением SLO, а не долей ёмкости.**
«80% ёмкости» — величина, которую невозможно честно определить для LLM,
где запросы различаются на порядки. «Прогнозируемый TTFT превысит SLO» —
проверяемое утверждение, напрямую связанное с тем, что видит пользователь.

**2. Отклонять надо до начала дорогой работы.** В goodput засчитываются
только полностью завершённые запросы (§2.1.2): если запрос отвалился на
полпути, потраченные ресурсы списываются впустую. Отсюда правило: не
начинаем то, что не сможем закончить. Admission control из защитного
механизма превращается в оптимизатор полезной работы.

**3. Наивное раннее отклонение вызывает автоколебания.** Это неочевидно
и стоит отдельного внимания. Механизм качелей:

    низкая загрузка → принимаем много → пачка доходит до следующей стадии
    → её загрузка взлетает → начинаем отклонять → первая стадия пустеет
    → новых запросов нет → загрузка падает → снова принимаем много

Корень — лаг между предсказанием нагрузки и её фактическим возникновением:
решение по текущей нагрузке принципиально запаздывает. Итог — пила вместо
полки и плохая утилизация.

Три демпфера, все три обязательны:
  • EWMA вместо мгновенного значения — убирает дребезг;
  • гистерезис (порог включения выше порога выключения) — убирает
    переключения на границе;
  • предсказание нагрузки на момент завершения текущей стадии, а не
    оценка текущей — убирает собственно лаг.

Предсказание намеренно **системного уровня, а не позапросного**:
предсказывать длину вывода конкретного запроса дорого и неточно, особенно
под перегрузкой, а оценить агрегированное состояние пула через время —
проще и требует меньшей точности.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(slots=True)
class EWMA:
    """Экспоненциальное сглаживание.

    alpha — вес нового наблюдения. Чем меньше, тем инертнее оценка:
    инерция здесь полезна, она и есть демпфер.
    """

    alpha: float
    value: float = 0.0
    initialized: bool = False

    def update(self, sample: float) -> float:
        if not self.initialized:
            self.value = sample
            self.initialized = True
        else:
            self.value += self.alpha * (sample - self.value)
        return self.value


@dataclass(slots=True)
class UpstreamLoad:
    """Наблюдаемое состояние одного апстрима.

    Поля разделены на две группы по §3.3.6.8. Прямые сигналы доступны
    только когда апстрим отдаёт своё состояние (сценарий 1). Наблюдаемые
    заменители работают всегда, потому что считаются нами самими.
    """

    upstream_id: str

    # --- Наблюдаемые нами всегда ---
    inflight_requests: int = 0
    inflight_tokens: int = 0        # счёт токенов, а не запросов: запросы несравнимы
    queued_requests: int = 0
    observed_ttft_ms: EWMA = field(default_factory=lambda: EWMA(alpha=0.2))
    error_rate: EWMA = field(default_factory=lambda: EWMA(alpha=0.2))
    last_429_at: float = 0.0

    # --- Прямые, если апстрим их отдаёт ---
    pending_prefill_tokens: int | None = None
    has_pending_queue: bool | None = None
    state_fresh_at: float = 0.0

    def busy(self, *, now: float | None = None, stale_after_s: float = 5.0) -> bool:
        """Бинарный признак занятости (§3.3.3.7).

        Естественная идея — ограничить число одновременных запросов
        константой — работает плохо: сколько запросов вмещает движок,
        зависит от суммы входных и выходных токенов, а выход заранее
        неизвестен. На одной модели и одном датасете предельное число
        одновременных запросов гуляло от 20 до 50. Любая фиксированная
        константа будет либо душить, либо перегружать.

        Поэтому признак — не порог, а факт: есть ли необслуженная очередь.
        Если апстрим этого не сообщает, заменяем наблюдаемым: недавний 429
        — явное признание перегрузки самим апстримом.
        """
        now = now if now is not None else time.monotonic()
        if self.has_pending_queue is not None and (now - self.state_fresh_at) < stale_after_s:
            return self.has_pending_queue
        if self.last_429_at and (now - self.last_429_at) < 2.0:
            return True
        return False

    def load_metric(self) -> float:
        """Метрика нагрузки для сравнения апстримов (§3.3.6.3).

        Предпочтительна — число pending prefill-токенов: латентность
        префилла масштабируется с числом токенов. Если апстрим её не
        отдаёт, считаем свои in-flight токены: это тот же смысл,
        измеренный снаружи.
        """
        if self.pending_prefill_tokens is not None:
            return float(self.pending_prefill_tokens)
        return float(self.inflight_tokens)


class LoadEstimator:
    """Сводная оценка нагрузки системы и решение о раннем отклонении.

    Состояние мягкое: при потере экземпляра прокси оно восстанавливается
    за секунды наблюдения. Деградирует эффективность, но не корректность
    (§3.3.8.1).
    """

    def __init__(
        self,
        *,
        ewma_alpha: float = 0.2,
        reject_threshold: float = 1.0,
        resume_threshold: float = 0.8,
        decode_time_estimate_s: float = 2.0,
    ) -> None:
        if resume_threshold >= reject_threshold:
            raise ValueError(
                "resume_threshold должен быть строго меньше reject_threshold: "
                "без гистерезиса контур будет колебаться (§3.3.3.3)"
            )
        self._alpha = ewma_alpha
        self._reject = reject_threshold
        self._resume = resume_threshold
        self._t_decode = decode_time_estimate_s
        self._upstreams: dict[str, UpstreamLoad] = {}
        # Сглаженная оценка отношения «прогноз TTFT / бюджет SLO».
        self._pressure = EWMA(alpha=ewma_alpha)
        # Состояние гистерезиса: включён ли режим отказов.
        self._rejecting = False
        self._flips = 0

    def upstream(self, upstream_id: str) -> UpstreamLoad:
        u = self._upstreams.get(upstream_id)
        if u is None:
            u = UpstreamLoad(
                upstream_id=upstream_id,
                observed_ttft_ms=EWMA(alpha=self._alpha),
                error_rate=EWMA(alpha=self._alpha),
            )
            self._upstreams[upstream_id] = u
        return u

    def all_upstreams(self) -> list[UpstreamLoad]:
        return list(self._upstreams.values())

    # --- Предсказание ---

    def predicted_pressure(
        self, *, queue_depth: int, capacity_tokens_per_s: float, ttft_budget_ms: float
    ) -> float:
        """Отношение прогнозируемого TTFT к бюджету SLO.

        Предсказывается нагрузка, которая возникнет **после** завершения
        текущей стадии, а не текущая: решение по текущей принципиально
        запаздывает и потому качает систему.

        Эвристика системного уровня: считаем, что каждый обрабатываемый
        сейчас запрос освободит ёмкость примерно через `t_decode`, и
        оцениваем, что останется в пуле к этому моменту.
        """
        total_pending = sum(u.load_metric() for u in self._upstreams.values())
        inflight = sum(u.inflight_requests for u in self._upstreams.values())

        if capacity_tokens_per_s <= 0:
            return float("inf")

        # Что уйдёт из пула за время t_decode, и что в него добавится из очереди.
        drain = capacity_tokens_per_s * self._t_decode
        future_pending = max(0.0, total_pending - drain)
        # Очередь доедет до префилла — добавляем её вклад консервативно.
        future_pending += queue_depth * max(1.0, total_pending / max(1, inflight)) if inflight else 0.0

        predicted_ttft_ms = future_pending / capacity_tokens_per_s * 1000.0
        if ttft_budget_ms <= 0:
            return float("inf")
        return predicted_ttft_ms / ttft_budget_ms

    def observe(
        self, *, queue_depth: int, capacity_tokens_per_s: float, ttft_budget_ms: float
    ) -> float:
        """Обновляет сглаженную оценку давления и возвращает её."""
        raw = self.predicted_pressure(
            queue_depth=queue_depth,
            capacity_tokens_per_s=capacity_tokens_per_s,
            ttft_budget_ms=ttft_budget_ms,
        )
        return self._pressure.update(raw)

    def should_reject(self) -> bool:
        """Решение с гистерезисом.

        Порог включения отказов выше порога выключения, поэтому система
        не дребезжит на границе: войдя в режим отказов, она выходит из
        него только заметно ниже точки входа.
        """
        p = self._pressure.value
        if self._rejecting:
            if p < self._resume:
                self._rejecting = False
                self._flips += 1
        else:
            if p > self._reject:
                self._rejecting = True
                self._flips += 1
        return self._rejecting

    @property
    def pressure(self) -> float:
        return self._pressure.value

    @property
    def flips(self) -> int:
        """Сколько раз контур переключался.

        Это диагностическая метрика: частые переключения под постоянной
        нагрузкой означают, что демпфирование не работает. На графике
        мониторинга должна быть полка, а не пила (§4).
        """
        return self._flips
