"""Контракт роутера (§3.3.6).

Ключевое проектное решение — **интерфейс, а не алгоритм**. Из §3.3.6.8:
подход агностичен к внутренней микроархитектуре сервинга и требует только
оценки TTFT для каждого кандидата. То есть контракт роутера — одна
функция, а разные её реализации соответствуют разным сценариям видимости:

  1. видим пул инстансов и их состояние → DualMap как есть;
  2. апстрим непрозрачен, но их несколько → сессионная маршрутизация,
     недостающие сигналы заменяем наблюдаемыми (наш основной сценарий);
  3. апстрим один и непрозрачен → роутинг вырождается, ценность
     смещается в admission control.

До старта хакатона неизвестно, какой сценарий будет нашим (§1.3 п.2).
Поэтому стратегия выбирается строкой в конфиге и меняется без передеплоя.

Отдельная оговорка, которую стоит помнить: среди одиночных балансировщиков
простой least-load достигает 97,38% пропускной способности продвинутого
решения (§3.3.6.3g). Весь этот модуль имеет смысл **только потому**, что
агентская нагрузка даёт очень высокую общность префикса (>80%, §5.3).
Если трафик окажется чатовым и разнородным, честный ответ — least_load.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..core.domain import ChatRequest, Upstream


@dataclass(slots=True)
class RoutingDecision:
    upstream: Upstream
    reason: str                 # почему выбран именно он — для отладки и дашборда
    expected_hit_blocks: int = 0
    strategy: str = ""
    candidates_considered: int = 0


class RoutingStrategy(ABC):
    """Базовый контракт. Стратегия обязана быть быстрой и не расти
    со размером кластера (§3.3.6.7): решение принимается по локальным
    метаданным кандидатов, а не опросом всего пула."""

    name: str = "base"

    @abstractmethod
    def select(
        self,
        request: ChatRequest,
        candidates: list[Upstream],
        ctx: "RoutingContext",
    ) -> RoutingDecision:
        ...

    def on_dispatched(
        self, request: ChatRequest, upstream: Upstream, ctx: "RoutingContext"
    ) -> None:
        """Уведомление о том, что запрос ушёл на апстрим.

        Здесь стратегия обновляет своё представление о том, что где лежит.
        Вынесено из `select`, потому что запрос может быть отклонён между
        выбором и отправкой, и тогда обновлять таблицу нельзя.
        """
        return None


class RoutingContext:
    """Всё, что стратегии нужно знать о системе.

    Собран как единый объект нарочно: стратегии не должны лазить по
    глобальному состоянию, иначе их нельзя протестировать поодиночке
    и нельзя честно сравнить между собой на одном стенде.
    """

    def __init__(self, *, load_estimator, prefix_table, slo, block_tokens: int) -> None:
        self.load = load_estimator
        self.prefix = prefix_table
        self.slo = slo
        self.block_tokens = block_tokens

    def load_of(self, upstream_id: str) -> float:
        return self.load.upstream(upstream_id).load_metric()

    def busy(self, upstream_id: str) -> bool:
        return self.load.upstream(upstream_id).busy()

    def observed_ttft_ms(self, upstream_id: str) -> float:
        return self.load.upstream(upstream_id).observed_ttft_ms.value

    def estimated_ttft_ms(self, upstream_id: str, compute_tokens: int) -> float:
        """Оценка TTFT кандидата — центральная функция роутера (§3.3.6.7a).

        Рецепт из Mooncake: время префилла предсказывается моделью,
        построенной на офлайн-замерах, по двум входам — длине запроса и
        длине совпавшего префикса; время ожидания складывается из времён
        префилла запросов в очереди инстанса.

        Важное свойство, объясняющее форму модели: время префилла растёт
        **суперлинейно** с длиной входа (внимание квадратично, MLP
        линейно). Поэтому линейная аппроксимация плоха на длинных
        контекстах, и здесь есть квадратичный член.

        Коэффициенты берутся из калибровочного прогона
        (`loadtest/calibrate.py`), а не назначаются. До калибровки
        работают значения по умолчанию, и это честно помечено.
        """
        u = self.load.upstream(upstream_id)
        queue_tokens = u.load_metric()
        total = queue_tokens + compute_tokens
        k = total / 1000.0
        return (
            self.prefill_const_ms
            + self.prefill_per_token_ms * total
            + self.prefill_quadratic_ms * k * k
        )

    # Коэффициенты модели префилла. Значения по умолчанию — заглушка,
    # пригодная только для того, чтобы система работала до калибровки.
    prefill_const_ms: float = 15.0
    prefill_per_token_ms: float = 0.04
    prefill_quadratic_ms: float = 1.5
    calibrated: bool = False

    def apply_calibration(self, coeffs: dict[str, float]) -> None:
        self.prefill_const_ms = coeffs["const_ms"]
        self.prefill_per_token_ms = coeffs["per_token_ms"]
        self.prefill_quadratic_ms = coeffs["quadratic_ms"]
        self.calibrated = True
