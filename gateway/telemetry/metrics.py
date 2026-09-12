"""Метрики (§3.3.10).

Ключевой приём для дашборда — **разделять время пользователя и время
инференса**. Разница между ними есть сеть и прокси; если она растёт,
проблема не в модели, а в инфраструктуре. Именно эта разница — метрика
нашей ответственности, и потому она выведена отдельной серией, а не
считается глазами по двум графикам.

Гистограммы размечены под три графика защиты (§3.3.10):
  1. TTFT p50/p95/p99 против RPS — доказательство SLA;
  2. поведение при отказе апстрима — доказательство отказоустойчивости;
  3. `interactive` держит SLO, `batch` деградирует — доказательство
     осмысленности архитектуры.
"""

from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

REGISTRY = CollectorRegistry()

# Границы подобраны под LLM: от единиц миллисекунд (попадание в кэш) до
# десятков секунд (длинный префилл). Стандартные границы Prometheus здесь
# бесполезны — они заканчиваются на 10 секундах.
TTFT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0,
                5.0, 10.0, 30.0, 60.0, 120.0)
ITL_BUCKETS = (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0)
OVERHEAD_BUCKETS = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5)

ttft = Histogram(
    "gateway_ttft_seconds", "Время до первого токена",
    ["service_class", "model", "upstream"],
    buckets=TTFT_BUCKETS, registry=REGISTRY,
)
itl = Histogram(
    "gateway_itl_seconds", "Средняя межтокенная задержка",
    ["service_class", "model"],
    buckets=ITL_BUCKETS, registry=REGISTRY,
)
overhead = Histogram(
    "gateway_overhead_seconds",
    "Накладные расходы гейтвея: наше время минус время апстрима. "
    "Единственная метрика, за которую отвечаем мы (§2.1)",
    ["service_class"],
    buckets=OVERHEAD_BUCKETS, registry=REGISTRY,
)
queue_wait = Histogram(
    "gateway_queue_wait_seconds", "Ожидание в очереди прокси до отправки",
    ["service_class"],
    buckets=TTFT_BUCKETS, registry=REGISTRY,
)

requests_total = Counter(
    "gateway_requests_total", "Запросы",
    ["service_class", "model", "outcome"], registry=REGISTRY,
)
goodput_total = Counter(
    "gateway_goodput_total",
    "Запросы, полностью завершённые в пределах SLO. Незавершённый запрос "
    "тратит ресурсы, но полезной работой не является (§2.1.2)",
    ["service_class", "model"], registry=REGISTRY,
)
tokens_total = Counter(
    "gateway_tokens_total", "Токены",
    ["direction", "tenant", "model"], registry=REGISTRY,
)
rejected_total = Counter(
    "gateway_rejected_total", "Отклонённые запросы",
    ["service_class", "reason"], registry=REGISTRY,
)
retries_total = Counter(
    "gateway_retries_total", "Повторы и деградации",
    ["kind"], registry=REGISTRY,
)
client_disconnects = Counter(
    "gateway_client_disconnects_total",
    "Разрывы со стороны клиента. Каждый обязан сопровождаться отменой "
    "запроса к апстриму, иначе течёт ёмкость (§3.3.9)",
    ["service_class"], registry=REGISTRY,
)

queue_depth = Gauge(
    "gateway_queue_depth", "Глубина очереди на прокси",
    ["service_class"], registry=REGISTRY,
)
inflight = Gauge(
    "gateway_inflight_requests", "Запросы в работе", ["upstream"], registry=REGISTRY,
)
inflight_tokens = Gauge(
    "gateway_inflight_tokens", "Токены в работе на апстриме", ["upstream"], registry=REGISTRY,
)
pressure = Gauge(
    "gateway_load_pressure",
    "Сглаженное отношение прогноза TTFT к бюджету SLO. Под постоянной "
    "нагрузкой обязана быть полка, а не пила (§3.3.3.3)",
    registry=REGISTRY,
)
breaker_state = Gauge(
    "gateway_breaker_state",
    "Состояние размыкателя: 0 закрыт, 1 полуоткрыт, 2 разомкнут",
    ["upstream"], registry=REGISTRY,
)
cache_hit_blocks = Counter(
    "gateway_prefix_hit_blocks_total", "Блоки префикса, найденные на апстриме",
    ["upstream"], registry=REGISTRY,
)
cache_total_blocks = Counter(
    "gateway_prefix_total_blocks_total", "Блоки префикса всего",
    ["upstream"], registry=REGISTRY,
)
config_version = Gauge(
    "gateway_config_version", "Версия применённой конфигурации", registry=REGISTRY,
)
quota_blocked = Counter(
    "gateway_quota_blocked_total",
    "Запросы, заблокированные квотой. Панель отвечает на вопрос, не "
    "задушили ли мы легитимный трафик (§3.3.10)",
    ["tenant", "limit_kind"], registry=REGISTRY,
)


def render() -> bytes:
    return generate_latest(REGISTRY)


BREAKER_CODES = {"closed": 0, "half_open": 1, "open": 2}
