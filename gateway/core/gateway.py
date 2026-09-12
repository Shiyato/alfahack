"""Горячий путь целиком (§3.2).

Цепочка из десяти блоков, собранная в одном месте, чтобы порядок и
границы были видны сразу:

    приём → auth → admission → роутинг → апстрим → стрим → учёт

Что здесь важно и неочевидно:

**Порядок не произволен.** Auth идёт до admission, потому что квота
принадлежит тенанту. Admission идёт до роутинга, потому что отклонять
надо **до** начала дорогой работы (§3.3.3.2): решение после префилла
означало бы выброшенные вычисления. Роутинг идёт до отправки, потому что
очередь живёт у нас, а не на апстримах (§3.3.3.7).

**Квота списывается вперёд и корректируется по факту.** Число выходных
токенов заранее неизвестно; без предварительного списания квота не
защищает вовсе — все параллельные запросы пройдут проверку раньше, чем
завершится хоть один.

**Расчёт по квоте обязан произойти при любом исходе.** Отсюда try/finally
вокруг всего стрима: клиент отвалился, апстрим упал, сработала отмена —
зарезервированные токены всё равно возвращаются, иначе квота утекает.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import AsyncIterator

from ..admission.load import LoadEstimator
from ..admission.queue import (
    PriorityQueue,
    QueueFull,
    QueuedRequest,
    QueueTimeout,
    SelectiveDispatcher,
)
from ..admission.quota import QuotaLedger
from ..auth.resolver import authorize_model, resolve
from ..resilience.breaker import BreakerConfig, BreakerRegistry, BreakerState
from ..router.base import RoutingContext
from ..router.prefix import PrefixTable, block_hashes, estimate_tokens
from ..router.strategies import build_strategy
from ..stream.pipeline import (
    RetryGuard,
    StreamPipeline,
    StreamResult,
    degradation_notice,
)
from ..telemetry import metrics as m
from ..upstream.adapter import (
    FailureKind,
    UpstreamClient,
    UpstreamError,
    get_adapter,
)
from .config import GatewayConfig
from .domain import ChatRequest, ServiceClass, Upstream
from .registry import ConfigRegistry

log = logging.getLogger(__name__)


class AdmissionRejected(Exception):
    """Ранний отказ. Несёт всё, что нужно для честного 429 (§3.3.3.6)."""

    def __init__(self, message: str, *, retry_after_s: float = 1.0,
                 limit_kind: str = "", remaining_tokens: float = 0.0) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s
        self.limit_kind = limit_kind
        self.remaining_tokens = remaining_tokens


@dataclass(slots=True)
class GatewayResponse:
    stream: AsyncIterator[bytes]
    result: StreamResult
    headers: dict[str, str]


class Gateway:
    """Сборка всех компонентов в один горячий путь."""

    def __init__(self, registry: ConfigRegistry) -> None:
        self.registry = registry
        cfg = registry.config

        self.load = LoadEstimator(
            ewma_alpha=cfg.admission.ewma_alpha,
            reject_threshold=cfg.admission.reject_threshold,
            resume_threshold=cfg.admission.resume_threshold,
        )
        self.prefix = PrefixTable(ttl_s=cfg.router.prefix_ttl_s)
        self.ctx = RoutingContext(
            load_estimator=self.load, prefix_table=self.prefix,
            slo=cfg.slo, block_tokens=cfg.router.block_tokens,
        )
        if cfg.calibration.measured:
            self.ctx.apply_calibration(cfg.calibration.as_dict())

        self.strategy = build_strategy(cfg.router)
        self.queue = PriorityQueue(max_depth=cfg.admission.max_queue_depth)
        self.dispatcher = SelectiveDispatcher(
            self.queue,
            max_wait_s=cfg.resilience.queue_max_wait_s,
            poll_interval_s=cfg.resilience.queue_poll_interval_s,
        )
        self.quotas = QuotaLedger()
        self.breakers = BreakerRegistry(self._breaker_config(cfg))
        self.client = UpstreamClient()
        self.pipeline = StreamPipeline()

        # Смена стратегии или блока — событие control-plane; горячий путь
        # о ней не знает и не должен (§3.1).
        registry.subscribe(self._on_config_change)

    @staticmethod
    def _breaker_config(cfg: GatewayConfig) -> BreakerConfig:
        r = cfg.resilience
        return BreakerConfig(
            error_rate_threshold=r.error_rate_threshold,
            min_samples=r.min_samples,
            consecutive_failures_to_open=r.consecutive_failures_to_open,
            latency_multiplier=r.latency_multiplier,
            open_duration_s=r.open_duration_s,
            half_open_successes=r.half_open_successes,
            window_s=r.window_s,
        )

    # ------------------------------------------------------------------
    # Жизненный цикл
    # ------------------------------------------------------------------

    async def start(self) -> None:
        await self.client.start()
        r = self.registry.config.resilience
        self._probe_task = asyncio.create_task(
            self._probe_loop(r.probe_interval_s), name="upstream-probe")
        self._fairshare_task = asyncio.create_task(
            self._fairshare_loop(r.fairshare_window_s), name="fairshare-reset")

    async def stop(self) -> None:
        for name in ("_probe_task", "_fairshare_task"):
            task = getattr(self, name, None)
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await self.client.close()

    def _on_config_change(self, old: GatewayConfig, new: GatewayConfig) -> None:
        if old.router.strategy != new.router.strategy:
            log.info("стратегия роутинга: %s → %s", old.router.strategy, new.router.strategy)
            self.strategy = build_strategy(new.router)
        if old.resilience != new.resilience:
            # Новая настройка применяется к вновь создаваемым размыкателям;
            # существующие доживают на прежней, чтобы правка конфига не
            # обнуляла накопленную статистику по живым апстримам.
            self.breakers._cfg = self._breaker_config(new)
        self.ctx.slo = new.slo
        self.ctx.block_tokens = new.router.block_tokens
        if new.calibration.measured:
            self.ctx.apply_calibration(new.calibration.as_dict())
        m.config_version.set(new.version)

    # ------------------------------------------------------------------
    # Фоновые циклы
    # ------------------------------------------------------------------

    async def _probe_loop(self, interval_s: float) -> None:
        """Опрос состояния апстримов.

        Частота выбрана не наугад: замер Б-2 показал, что устаревание
        сигнала нагрузки рушит admission control сильнее, чем отсутствие
        любых демпферов — доля запросов в SLO падала со 100% до 10%.
        Поэтому опрос частый, а бюджет усилий вложен именно сюда.

        Апстрим, не отдающий состояние, не является проблемой: роутер
        переходит на наблюдаемые заменители (§3.3.6.8, сценарий 2).
        """
        while True:
            try:
                await asyncio.sleep(interval_s)
                cfg = self.registry.config
                targets = [u for u in cfg.upstreams.values() if u.enabled and u.state_url]
                if not targets:
                    continue
                await asyncio.gather(
                    *(self._probe_one(u) for u in targets), return_exceptions=True
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("сбой цикла опроса апстримов")

    async def _probe_one(self, upstream: Upstream) -> None:
        import aiohttp

        try:
            assert self.client._session is not None
            async with self.client._session.get(
                upstream.state_url,
                timeout=aiohttp.ClientTimeout(
                    total=self.registry.config.resilience.probe_timeout_s),
            ) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()
        except Exception:
            # Недоступное состояние — не ошибка запроса: просто переходим
            # на наблюдаемые сигналы.
            return

        u = self.load.upstream(upstream.id)
        u.pending_prefill_tokens = data.get("pending_prefill_tokens")
        u.has_pending_queue = data.get("has_pending_queue")
        u.state_fresh_at = time.monotonic()
        m.inflight_tokens.labels(upstream.id).set(u.load_metric())

    async def _fairshare_loop(self, interval_s: float) -> None:
        """Сброс окна fair-share.

        Без сброса счётчик обслуженных растёт вечно, и тенант, активный
        с утра, навсегда уступает подключившемуся вечером.
        """
        while True:
            try:
                await asyncio.sleep(interval_s)
                self.queue.reset_fairshare()
            except asyncio.CancelledError:
                raise

    # ------------------------------------------------------------------
    # Горячий путь
    # ------------------------------------------------------------------

    async def handle(self, body: dict, authorization: str | None) -> GatewayResponse:
        cfg = self.registry.config
        started = time.monotonic()

        from ..api.schema import parse_chat_request

        request = parse_chat_request(body)
        request.request_id = uuid.uuid4().hex[:16]
        request.received_at = started

        # 1. Auth и разрешения (§3.3.2)
        tenant = resolve(cfg, authorization)
        authorize_model(tenant, request.model, cfg)
        request.tenant = tenant

        prompt_text = request.prompt_text()
        request.prompt_tokens_est = estimate_tokens(prompt_text)
        sc = tenant.service_class

        # 2. Admission control (§3.3.3)
        estimate = request.prompt_tokens_est + (
            request.max_tokens or cfg.admission.default_output_estimate
        )
        keys = [f"tenant:{tenant.id}", f"team:{tenant.team}", f"model:{request.model}"]
        rule = cfg.rate_limit_for(tenant, request.model)

        decision = self.quotas.check_and_reserve(
            keys=keys,
            tokens_per_minute=(rule.tokens_per_minute if rule else tenant.tokens_per_minute),
            requests_per_minute=(rule.requests_per_minute if rule else tenant.requests_per_minute),
            max_concurrent=(rule.max_concurrent if rule else tenant.max_concurrent),
            estimated_tokens=estimate,
            burst_multiplier=rule.burst_multiplier if rule else 2.0,
        )
        if not decision.allowed:
            m.quota_blocked.labels(tenant.id, decision.limit_kind).inc()
            m.rejected_total.labels(sc.value, decision.limit_kind).inc()
            raise AdmissionRejected(
                decision.reason, retry_after_s=decision.retry_after_s,
                limit_kind=decision.limit_kind,
                remaining_tokens=decision.remaining_tokens,
            )

        reserved = decision.reserved_tokens
        result = StreamResult()
        settled = False

        def settle() -> None:
            nonlocal settled
            if not settled:
                settled = True
                actual = result.prompt_tokens + result.completion_tokens
                self.quotas.settle(keys, reserved, actual or reserved)

        try:
            # 3. Раннее отклонение по прогнозу (§3.3.3.2)
            budget_ms = cfg.slo.ttft_budget_ms(request.prompt_tokens_est)
            self._update_pressure(budget_ms)
            if sc in (ServiceClass.BATCH, ServiceClass.BACKGROUND) and self.load.should_reject():
                # Под перегрузкой первым деградирует batch, а background
                # отбрасывается — это описанное поведение матрицы (§4),
                # а не побочный эффект.
                m.rejected_total.labels(sc.value, "overload").inc()
                raise AdmissionRejected(
                    "система перегружена, класс обслуживания деградирует первым",
                    retry_after_s=2.0, limit_kind="overload",
                )

            # 4. Выбор апстрима и ожидание готового (§3.3.3.7, §3.3.6)
            upstream, queue_wait = await self._pick_upstream(request, cfg, sc)
            m.queue_wait.labels(sc.value).observe(queue_wait)

            # 5. Отправка с деградацией при отказе (§3.3.8)
            return await self._dispatch(
                request, upstream, cfg, result, started, settle, budget_ms
            )
        except Exception:
            settle()
            raise

    def _update_pressure(self, budget_ms: float) -> None:
        capacity = max(1.0, sum(
            max(1.0, u.load_metric()) for u in self.load.all_upstreams()
        ))
        p = self.load.observe(
            queue_depth=len(self.queue),
            capacity_tokens_per_s=capacity,
            ttft_budget_ms=budget_ms,
        )
        m.pressure.set(p)
        for sc in ServiceClass:
            m.queue_depth.labels(sc.value).set(self.queue.depth(sc))

    async def _pick_upstream(
        self, request: ChatRequest, cfg: GatewayConfig, sc: ServiceClass
    ) -> tuple[Upstream, float]:
        candidates = cfg.upstreams_for(request.model)
        if not candidates:
            raise AdmissionRejected(f"нет живых апстримов для модели {request.model!r}",
                                    retry_after_s=5.0, limit_kind="no_upstream")

        # Размыкатель исключает апстрим до всякого выбора: незачем
        # маршрутизировать на заведомо сломанный.
        healthy_ids = set(self.breakers.healthy([u.id for u in candidates]))
        healthy = [u for u in candidates if u.id in healthy_ids]
        if not healthy:
            # Если апстримы сами назвали время восстановления, отдаём его
            # клиенту вместо выдуманной константы: наша оценка заведомо
            # хуже, чем прямое указание источника.
            hints = [
                self.load.upstream(u.id).retry_after_s
                for u in candidates
                if self.load.upstream(u.id).retry_after_s
            ]
            raise AdmissionRejected("все апстримы модели исключены размыкателем",
                                    retry_after_s=min(hints) if hints else 5.0,
                                    limit_kind="all_open")

        qreq = QueuedRequest(
            service_class=sc, tenant_id=request.tenant.id,
            enqueued_at=time.monotonic(), estimated_tokens=request.prompt_tokens_est,
            payload=request,
        )
        try:
            # Селективная отдача: ждём апстрим, который готов принять.
            # Готовность — бинарный признак, а не порог: предельное число
            # одновременных запросов у движка гуляет в разы (§3.3.3.7).
            def ready() -> list[Upstream]:
                free = [u for u in healthy if not self.load.upstream(u.id).busy()]
                return free or ([] if len(self.queue) > 0 else healthy)

            _, waited = await self.dispatcher.acquire_slot(qreq, ready)
        except QueueFull as exc:
            m.rejected_total.labels(sc.value, "queue_full").inc()
            raise AdmissionRejected(str(exc), retry_after_s=1.0,
                                    limit_kind="queue_full") from exc
        except QueueTimeout as exc:
            m.rejected_total.labels(sc.value, "queue_timeout").inc()
            raise AdmissionRejected(str(exc), retry_after_s=3.0,
                                    limit_kind="queue_timeout") from exc

        decision = self.strategy.select(request, healthy, self.ctx)
        return decision.upstream, waited

    async def _dispatch(
        self, request: ChatRequest, upstream: Upstream, cfg: GatewayConfig,
        result: StreamResult, started: float, settle, budget_ms: float,
    ) -> GatewayResponse:
        """Отправляет запрос и возвращает поток клиенту.

        Повтор возможен в двух местах, и это не дублирование логики, а
        два разных момента отказа:

        1. **При открытии потока** — апстрим отказал сразу (4xx/5xx, сеть).
        2. **Внутри уже открытого потока, до первого содержимого** — поток
           открылся, пришли служебные кадры, и только потом отказ. Так
           ведут себя провайдеры, сообщающие об ошибке внутри HTTP 200
           после стартовых метаданных.

        Второй случай важен и неочевиден: без него отказ, случившийся на
        миллисекунду позже, доходит до клиента ошибкой, хотя терять
        нечего — содержимого он ещё не видел. Первая версия обрабатывала
        только первый случай.

        Граница в обоих случаях одна: первый кадр **с содержимым**. После
        него стрим не идемпотентен, и повтор породил бы дубли (§3.3.8).
        """
        guard = RetryGuard(max_attempts=cfg.resilience.max_attempts)
        opened = await self._open_with_retry(
            request, upstream, cfg, guard, started
        )

        async def body() -> AsyncIterator[bytes]:
            response, target, notice = opened
            try:
                while True:
                    u_load = self.load.upstream(target.id)
                    try:  # noqa: PERF203
                        async for chunk in self.pipeline.relay(
                            response, result, started_at=started,
                            upstream_started_at=time.monotonic(),
                            degraded_notice=notice, guard=guard,
                        ):
                            yield chunk
                        return
                    except UpstreamError as exc:
                        self._record_failure(target, exc, u_load, cfg)
                        if not (exc.retryable and guard.may_retry()):
                            raise
                        nxt = self._fallback_target(request, cfg, exc, target)
                        if nxt is None:
                            raise
                        # Ни одного кадра с содержимым отдано не было —
                        # переоткрываем поток на другом апстриме прозрачно
                        # для клиента.
                        await self._wait_before_retry(exc, guard, cfg)
                        m.retries_total.labels("mid_stream").inc()
                        notice = degradation_notice(
                            original_model=request.model, actual_model=nxt.model,
                            reason=f"апстрим {target.id} оборвал поток: {exc}",
                        )
                        # Учёт занятости обязан быть симметричным: старая
                        # цель освобождается ровно там, где новая
                        # занимается. Иначе при каждом переподключении
                        # у прежнего апстрима остаётся занятый слот, и
                        # он постепенно выглядит перегруженным, не
                        # обрабатывая ничего.
                        self._release_one(request, target)
                        target = nxt
                        self._acquire_inflight(request, target)
                        try:
                            response = await self.client.open_stream(
                                request, target, get_adapter(target.adapter)
                            )
                        except UpstreamError:
                            self._release_one(request, target)
                            raise
                        self.strategy.on_dispatched(request, target, self.ctx)
            except (GeneratorExit, asyncio.CancelledError):
                result.client_disconnected = True
                raise
            finally:
                self._release_one(request, target)
                self._finish(request, result, self.breakers.get(target.id),
                             budget_ms, started)
                settle()

        response, target, _ = opened
        return GatewayResponse(
            stream=body(),
            result=result,
            headers={
                "X-Gateway-Request-Id": request.request_id,
                "X-Gateway-Upstream": target.id,
                "X-Gateway-Strategy": self.strategy.name,
                "X-Gateway-Config-Version": str(cfg.version),
            },
        )

    async def _open_with_retry(
        self, request: ChatRequest, upstream: Upstream, cfg: GatewayConfig,
        guard: RetryGuard, started: float,
    ):
        """Открывает поток, при необходимости перебирая цели."""
        target = upstream
        notice = None
        while True:
            guard.mark_attempt()
            u_load = self.load.upstream(target.id)
            self._acquire_inflight(request, target)
            try:
                response = await self.client.open_stream(
                    request, target, get_adapter(target.adapter)
                )
            except UpstreamError as exc:
                self._release_one(request, target)
                self._record_failure(target, exc, u_load, cfg)
                nxt = self._fallback_target(request, cfg, exc, target)
                if not (exc.retryable and guard.may_retry() and nxt is not None):
                    raise
                await self._wait_before_retry(exc, guard, cfg)
                m.retries_total.labels("fallback").inc()
                notice = degradation_notice(
                    original_model=request.model, actual_model=nxt.model,
                    reason=f"апстрим {target.id} недоступен: {exc}",
                )
                target = nxt
                continue

            self.strategy.on_dispatched(request, target, self.ctx)
            self._record_prefix(request, target)
            return response, target, notice

    def _acquire_inflight(self, request: ChatRequest, target: Upstream) -> None:
        u = self.load.upstream(target.id)
        u.inflight_requests += 1
        u.inflight_tokens += request.prompt_tokens_est
        m.inflight.labels(target.id).set(u.inflight_requests)

    def _release_one(self, request: ChatRequest, target: Upstream) -> None:
        u = self.load.upstream(target.id)
        u.inflight_requests = max(0, u.inflight_requests - 1)
        u.inflight_tokens = max(0, u.inflight_tokens - request.prompt_tokens_est)
        m.inflight.labels(target.id).set(u.inflight_requests)

    def _record_failure(self, target: Upstream, exc: UpstreamError,
                        u_load, cfg: GatewayConfig) -> None:
        """Учитывает отказ по его природе (§3.3.8).

        Отказ по учётным данным не засчитывается размыкателю: апстрим жив
        и исправен, у нас неверный ключ. Иначе один просроченный ключ
        исключил бы здоровый апстрим для всех тенантов сразу.
        """
        breaker = self.breakers.get(target.id)
        if exc.kind is not FailureKind.CREDENTIAL:
            breaker.record_failure(hard=exc.kind is FailureKind.TRANSIENT)
        m.breaker_state.labels(target.id).set(m.BREAKER_CODES[breaker.state.value])
        if exc.kind is FailureKind.RATE_LIMIT:
            u_load.last_429_at = time.monotonic()
            u_load.retry_after_s = exc.retry_after_s
        m.retries_total.labels(f"failure_{exc.kind.value}").inc()

    async def _wait_before_retry(self, exc: UpstreamError, guard: RetryGuard,
                                 cfg: GatewayConfig) -> None:
        """Пауза перед следующей попыткой.

        Мёртвый ключ от ожидания не оживёт, поэтому смена цели при отказе
        по учётным данным идёт немедленно. Исчерпанная квота и упавший
        апстрим требуют паузы: иначе повтор придётся на то же окно, что
        и отказ.
        """
        if exc.kind is FailureKind.CREDENTIAL:
            return
        delay = guard.backoff_s(
            initial=cfg.resilience.retry_backoff_initial_s,
            maximum=cfg.resilience.retry_backoff_max_s,
            retry_after_s=exc.retry_after_s,
        )
        if delay > 0:
            await asyncio.sleep(delay)

    def _fallback_target(
        self, request: ChatRequest, cfg: GatewayConfig, exc: UpstreamError,
        failed: Upstream,
    ) -> Upstream | None:
        """Цель деградации по правилам конфигурации (§3.3.8).

        Осторожность с 429: перекладывая нагрузку на соседа с общим пулом
        ёмкости, мы переносим перегрузку, а не лечим её. Поэтому правило
        обязано явно подтвердить независимость ёмкости — валидатор
        конфигурации это требует.
        """
        # У обрыва соединения или потока кода ответа нет, но по существу
        # это отказ апстрима. Сопоставляем такие случаи с правилами как
        # 503: иначе настроенная резервная модель не сработает именно
        # там, где она нужнее всего — когда апстрим умер молча.
        status = exc.status if exc.status is not None else 503
        if True:
            rule = cfg.fallback_for(request.model, status)
            if rule is not None:
                if status == 429 and not rule.independent_capacity:
                    return None
                for target in rule.targets:
                    for u in cfg.upstreams_for(target):
                        if u.id != failed.id and self.breakers.get(u.id).allows():
                            return u
        # Правила нет — пробуем живого соседа той же модели.
        for u in cfg.upstreams_for(request.model):
            if u.id != failed.id and self.breakers.get(u.id).allows():
                return u
        return None

    def _record_prefix(self, request: ChatRequest, upstream: Upstream) -> None:
        hashes = block_hashes(
            request.prompt_text(), self.ctx.block_tokens,
            salt=request.tenant.id if request.tenant else "anon",
        )
        if not hashes:
            return
        hit = self.prefix.hit_len(hashes, upstream.id)
        m.cache_hit_blocks.labels(upstream.id).inc(hit)
        m.cache_total_blocks.labels(upstream.id).inc(len(hashes))

    def _finish(self, request: ChatRequest, result: StreamResult,
                breaker, budget_ms: float, started: float) -> None:
        sc = request.tenant.service_class if request.tenant else ServiceClass.INTERACTIVE
        labels = (sc.value, request.model)

        if result.client_disconnected:
            # Разрыв клиентом не является отказом апстрима и не должен
            # влиять на размыкатель: иначе закрытая вкладка портит
            # репутацию исправному апстриму.
            m.client_disconnects.labels(sc.value).inc()
            m.requests_total.labels(*labels, "disconnected").inc()
            return

        if result.ttft_ms:
            m.ttft.labels(sc.value, request.model, result.upstream_id).observe(
                result.ttft_ms / 1000.0
            )
            u = self.load.upstream(result.upstream_id)
            u.observed_ttft_ms.update(result.ttft_ms)
        if result.itl_ms:
            m.itl.labels(sc.value, request.model).observe(result.itl_ms / 1000.0)
        if (oh := result.overhead_ms) > 0:
            m.overhead.labels(sc.value).observe(oh / 1000.0)

        if result.finished:
            breaker.record_success(latency_ms=result.ttft_ms, budget_ms=budget_ms)
            m.requests_total.labels(*labels, "ok").inc()
            # В goodput засчитывается только то, что и завершилось,
            # и уложилось в SLO (§2.1.2).
            if result.ttft_ms <= budget_ms:
                m.goodput_total.labels(*labels).inc()
        else:
            # Неполный стрим — мягкий отказ: причин у него много, и
            # большинство из них на нашей стороне, а не у апстрима.
            breaker.record_failure(hard=False)
            m.requests_total.labels(*labels, "incomplete").inc()

        m.breaker_state.labels(result.upstream_id).set(
            m.BREAKER_CODES[breaker.state.value]
        )
        if request.tenant:
            m.tokens_total.labels("in", request.tenant.id, request.model).inc(
                result.prompt_tokens or request.prompt_tokens_est
            )
            m.tokens_total.labels("out", request.tenant.id, request.model).inc(
                result.completion_tokens
            )
