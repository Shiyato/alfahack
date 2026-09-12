"""Декларативные политики и hot-reload (§3.1.1).

Ценность конфигурации в том, что она **демонстрируема**: правка файла,
применение, изменение поведения в прямом эфире, без передеплоя. Это прямой
ответ на критерий «гибкость» из §1.2.

Семантика правил — «применяется первое совпавшее», как в ACL или правилах
файрвола. Отсюда ловушка: **порядок становится частью семантики**, и
перестановка строк молча меняет поведение. Поэтому здесь же живёт валидатор,
предупреждающий о недостижимых правилах.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .domain import ServiceClass, Tenant, Upstream

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Правила
# --------------------------------------------------------------------------


@dataclass(slots=True)
class RuleWhen:
    """Условие срабатывания правила. Пустое поле означает «любое»."""

    subjects: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    service_classes: tuple[str, ...] = ()
    status_codes: tuple[int, ...] = ()

    def matches(
        self,
        *,
        subject: str | None = None,
        team: str | None = None,
        model: str | None = None,
        service_class: str | None = None,
        status: int | None = None,
    ) -> bool:
        if self.subjects:
            candidates = {f"user:{subject}", f"team:{team}", subject or ""}
            if not candidates & set(self.subjects):
                return False
        if self.models and model not in self.models:
            return False
        if self.service_classes and service_class not in self.service_classes:
            return False
        if self.status_codes and status not in self.status_codes:
            return False
        return True

    def covers(self, other: "RuleWhen") -> bool:
        """Истина, если это условие перекрывает `other` целиком.

        Используется валидатором: правило, чьё условие полностью покрыто
        вышестоящим, недостижимо — до него никогда не дойдёт очередь.
        """

        def field_covers(mine: tuple, theirs: tuple) -> bool:
            if not mine:          # «любое» покрывает что угодно
                return True
            if not theirs:        # конкретное не покрывает «любое»
                return False
            return set(theirs) <= set(mine)

        return (
            field_covers(self.subjects, other.subjects)
            and field_covers(self.models, other.models)
            and field_covers(self.service_classes, other.service_classes)
            and field_covers(self.status_codes, other.status_codes)
        )


@dataclass(slots=True)
class RateLimitRule:
    """Правило квотирования (§3.3.3.5).

    Лимиты по токенам и по запросам нужны одновременно: токены ловят
    тяжёлые запросы, счёт запросов ловит шквал лёгких.
    """

    id: str
    when: RuleWhen
    tokens_per_minute: int | None = None
    requests_per_minute: int | None = None
    max_concurrent: int | None = None
    # Размер ведра: всплеск ограничен им. Агентская нагрузка принципиально
    # всплесковая — агент выпускает серию запросов по мере обхода инструментов,
    # и fixed window резал бы легитимные серии.
    burst_multiplier: float = 2.0


@dataclass(slots=True)
class FallbackRule:
    """Правило деградации (§3.3.8)."""

    id: str
    when: RuleWhen
    targets: tuple[str, ...] = ()
    # 429 как триггер конфликтует с нашим admission control: перекладывая
    # нагрузку на соседа, мы переносим перегрузку, а не лечим её. Допустимо
    # только на апстрим с независимым пулом ёмкости — отсюда явный флаг,
    # который валидатор требует подтвердить.
    independent_capacity: bool = False


# --------------------------------------------------------------------------
# Сводная конфигурация
# --------------------------------------------------------------------------


@dataclass(slots=True)
class SLOConfig:
    """Целевые показатели (§2.1.4).

    TTFT-SLO линейный по числу входных токенов: запрос на 100k токенов
    физически не может ответить так же быстро, как на 500, и единый
    абсолютный порог наказывал бы длинные запросы ни за что.

        ttft_slo = base_ms + per_token_ms * input_tokens

    Коэффициенты берутся из калибровочного прогона, а не из головы.
    """

    ttft_base_ms: float = 500.0
    ttft_per_token_ms: float = 0.05
    tpot_ms: float = 30.0
    # Множители относительного SLO (§2.1.1): порог = N × времени того же
    # запроса в одиночку. TTFT терпит большую деградацию, чем TPOT: рваный
    # стрим человек замечает сразу, лишнюю секунду ожидания — спокойнее.
    relative_ttft_multiplier: float = 10.0
    relative_tpot_multiplier: float = 5.0

    def ttft_budget_ms(self, input_tokens: int) -> float:
        return self.ttft_base_ms + self.ttft_per_token_ms * input_tokens


@dataclass(slots=True)
class RouterConfig:
    """Настройки роутера (§3.3.6).

    `strategy` выбирается строкой, потому что до старта хакатона неизвестно,
    какие сигналы будут доступны (§3.3.6.8). Все стратегии реализуют один
    интерфейс, переключение — правкой конфига без передеплоя.
    """

    strategy: str = "session"          # least_load | consistent_hash | session | dualmap
    block_tokens: int = 256            # размер блока для цепочечного хеширования
    prefix_ttl_s: float = 600.0        # 90% переиспользований укладываются в ~100 с (§5.3)
    # Гиперпараметры SMetric. Их плато широкое — TPS меняется в пределах 6%
    # при OVERLOAD от 1 до бесконечности, — поэтому подбирать не нужно.
    overload_factor: float = 2.0
    hit_ratio_factor: float = 0.5


@dataclass(slots=True)
class AdmissionConfig:
    """Настройки admission control (§3.3.3)."""

    max_queue_depth: int = 1000
    # Контур «нагрузка → решение о приёме» без сглаживания и гистерезиса
    # даёт автоколебания (§3.3.3.3). Оба демпфера включены по умолчанию.
    ewma_alpha: float = 0.2
    reject_threshold: float = 1.0      # доля бюджета SLO, выше которой отказываем
    resume_threshold: float = 0.8      # гистерезис: порог выключения ниже порога включения
    # Оценка выхода для предварительного списания квоты: число выходных
    # токенов заранее неизвестно, поэтому списываем оценку и корректируем
    # по факту (§3.3.3.5).
    default_output_estimate: int = 512


@dataclass(slots=True)
class ResilienceConfig:
    """Параметры устойчивости (§3.3.8).

    Вынесены в конфигурацию, а не оставлены в коде, по конкретной причине:
    поведение незнакомой платформы при отказах заранее неизвестно. Её
    таймауты, коды ошибок и скорость восстановления выясняются только на
    месте, и крутить их придётся быстро — правкой YAML, а не передеплоем.

    Значения по умолчанию совпадают с проверенными в замерах Б-5 и Б-6.
    """

    # --- Размыкатель ---
    error_rate_threshold: float = 0.5
    min_samples: int = 20
    # Серия отказов подряд — сигнал сильнее доли в окне: ждать статистики
    # для явно мёртвого апстрима незачем, каждая попытка стоит таймаута.
    consecutive_failures_to_open: int = 3
    # Ответ во столько раз хуже бюджета считается отказом: успешный, но
    # очень медленный апстрим вреднее честной пятисотки.
    latency_multiplier: float = 5.0
    open_duration_s: float = 10.0
    half_open_successes: int = 3
    window_s: float = 30.0

    # --- Ретраи ---
    # Повтор возможен только до первого отданного токена; это ограничение
    # жёсткое и конфигом не снимается (§3.3.8).
    max_attempts: int = 2

    # --- Селективная отдача ---
    queue_max_wait_s: float = 30.0
    queue_poll_interval_s: float = 0.002

    # --- Опрос состояния апстримов ---
    # Замер Б-2: устаревание сигнала рушит admission control сильнее, чем
    # отсутствие любых демпферов — доля запросов в SLO падала со 100% до
    # 10%. Поэтому частота опроса здесь несущий параметр, а не деталь.
    probe_interval_s: float = 0.25
    probe_timeout_s: float = 1.0
    fairshare_window_s: float = 10.0


@dataclass(slots=True)
class CalibrationConfig:
    """Коэффициенты модели времени префилла (§3.3.6.7a).

    `measured` отличает результат калибровочного прогона от значений по
    умолчанию. Это не украшение: роутер, работающий на выдуманных
    коэффициентах, принимает решения, которые выглядят обоснованными, но
    ничем не обоснованы. Отличать одно от другого надо явно.
    """

    const_ms: float = 15.0
    per_token_ms: float = 0.04
    quadratic_ms: float = 1.5
    measured: bool = False
    mean_rel_err: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {"const_ms": self.const_ms, "per_token_ms": self.per_token_ms,
                "quadratic_ms": self.quadratic_ms}


@dataclass(slots=True)
class GatewayConfig:
    slo: SLOConfig = field(default_factory=SLOConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    admission: AdmissionConfig = field(default_factory=AdmissionConfig)
    resilience: ResilienceConfig = field(default_factory=ResilienceConfig)
    upstreams: dict[str, Upstream] = field(default_factory=dict)
    model_aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    tenants: dict[str, Tenant] = field(default_factory=dict)
    rate_limits: tuple[RateLimitRule, ...] = ()
    fallbacks: tuple[FallbackRule, ...] = ()
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    version: int = 0

    def upstreams_for(self, model: str) -> list[Upstream]:
        """Апстримы, обслуживающие логическое имя модели."""
        ids = self.model_aliases.get(model, ())
        return [u for i in ids if (u := self.upstreams.get(i)) and u.enabled]

    def rate_limit_for(self, tenant: Tenant, model: str) -> RateLimitRule | None:
        """Первое совпавшее правило — семантика ACL (§3.1.1)."""
        for rule in self.rate_limits:
            if rule.when.matches(
                subject=tenant.name, team=tenant.team, model=model,
                service_class=tenant.service_class.value,
            ):
                return rule
        return None

    def fallback_for(self, model: str, status: int) -> FallbackRule | None:
        for rule in self.fallbacks:
            if rule.when.matches(model=model, status=status):
                return rule
        return None


# --------------------------------------------------------------------------
# Загрузка
# --------------------------------------------------------------------------


def _when(raw: dict[str, Any] | None) -> RuleWhen:
    raw = raw or {}
    return RuleWhen(
        subjects=tuple(raw.get("subjects", ())),
        models=tuple(raw.get("models", ())),
        service_classes=tuple(raw.get("service_classes", ())),
        status_codes=tuple(raw.get("response_status_codes", ())),
    )


def load_config(config_dir: str | Path) -> tuple[GatewayConfig, list[str]]:
    """Читает конфигурацию из каталога. Возвращает конфиг и список замечаний.

    Замечания — не ошибки: конфиг применим, но валидатор нашёл в нём то,
    что почти наверняка не соответствует намерению автора.
    """
    d = Path(config_dir)
    cfg = GatewayConfig()

    models = _read_yaml(d / "models.yaml")
    for raw in models.get("upstreams", ()):
        up = Upstream(
            id=raw["id"],
            base_url=raw["base_url"],
            model=raw.get("model", raw["id"]),
            adapter=raw.get("adapter", "openai"),
            api_key=raw.get("api_key"),
            weight=float(raw.get("weight", 1.0)),
            state_url=raw.get("state_url"),
            timeout_connect_s=float(raw.get("timeout_connect_s", 5.0)),
            timeout_ttft_s=float(raw.get("timeout_ttft_s", 30.0)),
            timeout_total_s=float(raw.get("timeout_total_s", 600.0)),
            enabled=bool(raw.get("enabled", True)),
        )
        cfg.upstreams[up.id] = up
    cfg.model_aliases = {
        name: tuple(ids) for name, ids in (models.get("models") or {}).items()
    }
    if slo := models.get("slo"):
        cfg.slo = SLOConfig(**slo)
    if router := models.get("router"):
        cfg.router = RouterConfig(**router)
    if adm := models.get("admission"):
        cfg.admission = AdmissionConfig(**adm)
    if res := models.get("resilience"):
        cfg.resilience = ResilienceConfig(**res)

    auth = _read_yaml(d / "tenants.yaml")
    for raw in auth.get("tenants", ()):
        t = Tenant(
            id=raw["id"],
            name=raw.get("name", raw["id"]),
            team=raw.get("team", "default"),
            service_class=ServiceClass(raw.get("service_class", "interactive")),
            allowed_models=tuple(raw.get("allowed_models", ())),
            tokens_per_minute=raw.get("tokens_per_minute"),
            requests_per_minute=raw.get("requests_per_minute"),
            max_concurrent=raw.get("max_concurrent"),
            semantic_cache_enabled=bool(raw.get("semantic_cache_enabled", False)),
            cross_tenant_cache=bool(raw.get("cross_tenant_cache", False)),
        )
        for key in raw.get("keys", ()):
            cfg.tenants[key] = t

    rl = _read_yaml(d / "ratelimit.yaml")
    cfg.rate_limits = tuple(
        RateLimitRule(
            id=raw["id"],
            when=_when(raw.get("when")),
            tokens_per_minute=raw.get("tokens_per_minute"),
            requests_per_minute=raw.get("requests_per_minute"),
            max_concurrent=raw.get("max_concurrent"),
            burst_multiplier=float(raw.get("burst_multiplier", 2.0)),
        )
        for raw in rl.get("rules", ())
    )

    fb = _read_yaml(d / "fallback.yaml")
    cfg.fallbacks = tuple(
        FallbackRule(
            id=raw["id"],
            when=_when(raw.get("when")),
            targets=tuple(t["target"] if isinstance(t, dict) else t
                          for t in raw.get("fallback_models", ())),
            independent_capacity=bool(raw.get("independent_capacity", False)),
        )
        for raw in fb.get("rules", ())
    )

    cal = _read_yaml(d / "calibration.yaml")
    if prefill := cal.get("prefill"):
        cfg.calibration = CalibrationConfig(
            const_ms=float(prefill["const_ms"]),
            per_token_ms=float(prefill["per_token_ms"]),
            quadratic_ms=float(prefill["quadratic_ms"]),
            measured=True,
            mean_rel_err=float((cal.get("quality") or {}).get("mean_rel_err", 0.0)),
        )

    return cfg, validate(cfg)


# ${VAR} и ${VAR:-значение по умолчанию}
_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(text: str) -> str:
    """Подстановка переменных окружения в текст конфигурации.

    Один и тот же файл должен работать и локально, и в контейнере, где
    апстримы доступны под именами сервисов, а не по localhost. Без
    подстановки пришлось бы держать две копии конфига, которые неизбежно
    разъедутся.

    Значение по умолчанию обязательно там, где оно осмысленно: конфиг
    должен запускаться без единой переменной окружения, иначе локальная
    разработка превращается в обряд.
    """

    def repl(m: re.Match[str]) -> str:
        name, default = m.group(1), m.group(2)
        value = os.environ.get(name)
        if value is not None:
            return value
        if default is not None:
            return default
        log.warning("переменная окружения %s не задана и не имеет значения "
                    "по умолчанию; подставлена пустая строка", name)
        return ""

    return _ENV_RE.sub(repl, text)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(expand_env(f.read())) or {}


# --------------------------------------------------------------------------
# Валидатор
# --------------------------------------------------------------------------


def validate(cfg: GatewayConfig) -> list[str]:
    """Проверки, которые стоят получаса и отвечают на вопрос защиты
    «как вы гарантируете корректность политик» (§3.1.1)."""
    issues: list[str] = []

    # Недостижимые правила: перекрытые вышестоящими.
    for rules, kind in ((cfg.rate_limits, "квотирования"), (cfg.fallbacks, "деградации")):
        for i, rule in enumerate(rules):
            for earlier in rules[:i]:
                if earlier.when.covers(rule.when):
                    issues.append(
                        f"правило {kind} {rule.id!r} недостижимо: "
                        f"полностью перекрыто правилом {earlier.id!r} выше"
                    )
                    break

    # Модели без апстримов и апстримы-сироты.
    for model, ids in cfg.model_aliases.items():
        missing = [i for i in ids if i not in cfg.upstreams]
        if missing:
            issues.append(f"модель {model!r} ссылается на несуществующие апстримы: {missing}")
        if not ids:
            issues.append(f"модель {model!r} не имеет ни одного апстрима")
    referenced = {i for ids in cfg.model_aliases.values() for i in ids}
    for orphan in set(cfg.upstreams) - referenced:
        issues.append(f"апстрим {orphan!r} не используется ни одной моделью")

    # Цели fallback обязаны существовать.
    for rule in cfg.fallbacks:
        for target in rule.targets:
            if target not in cfg.model_aliases and target not in cfg.upstreams:
                issues.append(f"правило деградации {rule.id!r} ссылается на неизвестную цель {target!r}")
        # 429 как триггер требует независимой ёмкости, иначе каскад (§3.3.8).
        if 429 in rule.when.status_codes and not rule.independent_capacity:
            issues.append(
                f"правило деградации {rule.id!r} срабатывает на 429, но цель не помечена "
                "independent_capacity: перекладывание нагрузки на общий пул даст каскад"
            )

    # Гистерезис обязан быть настоящим, иначе контур колеблется (§3.3.3.3).
    adm = cfg.admission
    if adm.resume_threshold >= adm.reject_threshold:
        issues.append(
            f"гистерезис отсутствует: resume_threshold={adm.resume_threshold} "
            f">= reject_threshold={adm.reject_threshold}; контур будет колебаться"
        )
    if not 0 < adm.ewma_alpha <= 1:
        issues.append(f"ewma_alpha={adm.ewma_alpha} вне (0, 1]")

    # Роутер на неоткалиброванной модели принимает решения, которые
    # выглядят обоснованными, но ничем не обоснованы (§3.3.6.7a).
    if not cfg.calibration.measured and cfg.router.strategy in ("dualmap",):
        issues.append(
            "стратегия dualmap опирается на оценку TTFT, а модель префилла не "
            "откалибрована: запустите loadtest/calibrate.py"
        )
    if cfg.calibration.measured and cfg.calibration.mean_rel_err > 0.25:
        issues.append(
            f"калибровка низкого качества: средняя относительная ошибка "
            f"{cfg.calibration.mean_rel_err:.0%}; оценкам TTFT доверять нельзя"
        )

    r = cfg.resilience
    if r.probe_interval_s > 2.0:
        issues.append(
            f"probe_interval_s={r.probe_interval_s} с: по замеру Б-2 устаревание "
            "сигнала нагрузки рушит admission control сильнее, чем отсутствие "
            "демпферов — доля запросов в SLO падает со 100% до 10%"
        )
    if r.half_open_successes < 1:
        issues.append("half_open_successes должен быть не меньше 1, "
                      "иначе размыкатель закроется на первом же пробном запросе")
    if r.open_duration_s <= 0:
        issues.append("open_duration_s должен быть положительным, "
                      "иначе размыкатель не исключает апстрим вовсе")
    if r.max_attempts < 1:
        issues.append("max_attempts должен быть не меньше 1")

    known = {"least_load", "consistent_hash", "session", "dualmap"}
    if cfg.router.strategy not in known:
        issues.append(f"неизвестная стратегия роутинга {cfg.router.strategy!r}, допустимы: {sorted(known)}")

    # Тенант, которому разрешены несуществующие модели.
    for key, t in cfg.tenants.items():
        for m in t.allowed_models:
            if m not in cfg.model_aliases:
                issues.append(f"тенанту {t.id!r} разрешена неизвестная модель {m!r}")

    return issues
