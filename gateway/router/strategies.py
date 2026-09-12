"""Стратегии маршрутизации (§3.3.6).

Четыре реализации одного контракта, от простейшей к самой требовательной
по доступным сигналам. Порядок здесь — это порядок деградации: если
сигналы недоступны, спускаемся на строчку выше и ничего не ломаем.
"""

from __future__ import annotations

import bisect
import hashlib

from ..core.domain import ChatRequest, Upstream
from .base import RoutingContext, RoutingDecision, RoutingStrategy
from .prefix import block_hashes, estimate_tokens


class _RoundRobinTiebreak:
    """Разрыв ничьей по кругу.

    Нужен потому, что детерминированный разрыв ничьей по идентификатору —
    настоящий дефект, а не мелочь. На простаивающей системе нагрузка всех
    апстримов равна нулю, `min()` по `(нагрузка, id)` всегда возвращает
    первый по алфавиту, и весь трафик идёт на один узел, пока тот не
    нагрузится достаточно, чтобы проиграть сравнение.

    Дефект нашёлся симуляцией: least_load отправил 2586 запросов из 2589
    на один апстрим из четырёх. Под нагрузкой он маскируется, под низкой —
    проявляется полностью.
    """

    __slots__ = ("_n",)

    def __init__(self) -> None:
        self._n = 0

    def pick(self, candidates: list[Upstream], ctx: RoutingContext,
             *, epsilon: float = 1.0) -> Upstream:
        """Наименее загруженный; при примерном равенстве — по кругу.

        `epsilon` в токенах: разница меньше неё считается ничьёй, иначе
        шум в оценке нагрузки даёт ложную определённость.
        """
        loads = [(ctx.load_of(u.id), u) for u in candidates]
        lo = min(l for l, _ in loads)
        tied = [u for l, u in loads if l - lo <= epsilon]
        if len(tied) == 1:
            return tied[0]
        self._n += 1
        return tied[self._n % len(tied)]


class LeastLoadStrategy(RoutingStrategy):
    """Наименее загруженный. Двадцать строк, никаких требований к видимости.

    Присутствует не как заглушка, а как честный вариант ответа: среди
    одиночных балансировщиков least-load достигает 97,38% пропускной
    способности продвинутого решения, когда общность префиксов невысока
    (§3.3.6.3g). Это же — базовая линия, относительно которой измеряются
    остальные: без неё выигрыш умных стратегий не с чем сравнить.
    """

    name = "least_load"

    def __init__(self) -> None:
        self._tiebreak = _RoundRobinTiebreak()

    def select(self, request, candidates, ctx) -> RoutingDecision:
        free = [u for u in candidates if not ctx.busy(u.id)] or candidates
        best = self._tiebreak.pick(free, ctx)
        return RoutingDecision(
            upstream=best,
            reason=f"наименьшая нагрузка ({ctx.load_of(best.id):.0f} токенов)",
            strategy=self.name,
            candidates_considered=len(candidates),
        )


class ConsistentHashStrategy(RoutingStrategy):
    """Хеш-кольцо по сессии с пропуском занятых узлов (§3.3.6.3f).

    Неявно префикс-осведомлённая стратегия: запросы одной сессии разделяют
    контекст и хешируются на один апстрим безо всякого анализа промптов.
    Одно расширение делает её пригодной под нагрузкой — **виртуальные узлы
    пропускаются по признаку доступности**, и обход продолжается к
    следующему. Кольцо даёт affinity, пропуск даёт устойчивость.

    Три известные слабости, о которых надо знать: теряется межпользовательское
    разделение префиксов (−16,5% попаданий), всплеск от одного пользователя
    перегружает один апстрим (−7,1%), разнородные паттерны внутри
    пользователя (−8,8%).

    Держим как запасной вариант: дёшево, объяснимо, работает.
    """

    name = "consistent_hash"

    def __init__(self, *, vnodes: int = 160) -> None:
        self._vnodes = vnodes
        self._ring: list[tuple[int, str]] = []
        self._ring_keys: list[int] = []
        self._built_for: tuple[str, ...] = ()

    def _build(self, candidates: list[Upstream]) -> None:
        ids = tuple(sorted(u.id for u in candidates))
        if ids == self._built_for:
            return
        ring = []
        for uid in ids:
            for v in range(self._vnodes):
                h = hashlib.blake2b(f"{uid}#{v}".encode(), digest_size=8).digest()
                ring.append((int.from_bytes(h, "big"), uid))
        ring.sort()
        self._ring = ring
        self._ring_keys = [k for k, _ in ring]
        self._built_for = ids

    def select(self, request, candidates, ctx) -> RoutingDecision:
        self._build(candidates)
        by_id = {u.id: u for u in candidates}
        tenant_id = request.tenant.id if request.tenant else "anon"
        key = request.session_key(tenant_id)
        h = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")

        start = bisect.bisect_left(self._ring_keys, h)
        n = len(self._ring)
        first_choice = None
        for step in range(n):
            _, uid = self._ring[(start + step) % n]
            u = by_id.get(uid)
            if u is None:
                continue
            if first_choice is None:
                first_choice = u
            if not ctx.busy(uid):
                reason = ("кольцо: первый свободный узел"
                          if u is first_choice else "кольцо: пропущены занятые узлы")
                return RoutingDecision(upstream=u, reason=reason, strategy=self.name,
                                       candidates_considered=len(candidates))
        # Все заняты — отдаём «своему», лучше подождать у него, чем размазать кэш.
        best = first_choice or candidates[0]
        return RoutingDecision(upstream=best, reason="кольцо: все узлы заняты",
                               strategy=self.name, candidates_considered=len(candidates))


class SessionStrategy(RoutingStrategy):
    """Сессионная маршрутизация по SMetric (§3.3.6.3a) — основной слой.

    Наблюдение, из которого всё следует: под cache-aware маршрутизацией
    96,6% последующих запросов возвращаются на тот же инстанс, что
    обслужил первый запрос сессии (против 4,0% под чистой балансировкой).
    Значит, **размещение первого запроса определяет размещение всей
    сессии**, и чтобы сбалансировать кластер, достаточно балансировать
    только первые запросы — а это малая доля трафика.

    Почему именно эта стратегия основная: она требует только номера хода,
    который извлекается из тела запроса (§3.3.6.3b). Ей не нужна никакая
    видимость внутрь апстрима — а мы до старта не знаем, будет ли она.

    Два предохранителя:
      • `not_overloaded` — страховка от случая, когда две изначально
        короткие сессии на одном инстансе обе разрослись. Откат на
        балансировку фактически мигрирует длинную сессию.
      • `session_not_evicted` — если сессия долго простаивала, её KV мог
        быть вытеснен; такой запрос надо трактовать как первый в новой
        сессии.

    Гиперпараметры подбирать не нужно: их плато широкое — TPS остаётся в
    пределах 6% при OVERLOAD от 1 до бесконечности и 4% при HIT_RATIO от
    0 до 0,75.
    """

    name = "session"

    def __init__(self, *, overload_factor: float = 2.0, hit_ratio_factor: float = 0.5) -> None:
        self._overload = overload_factor
        self._hit_ratio = hit_ratio_factor
        self._tiebreak = _RoundRobinTiebreak()

    def select(self, request, candidates, ctx) -> RoutingDecision:
        tenant_id = request.tenant.id if request.tenant else "anon"
        hashes = block_hashes(request.prompt_text(), ctx.block_tokens, salt=tenant_id)

        # Первый запрос сессии — чистая балансировка. Это и есть весь
        # приём: балансируем малую долю трафика, остальное залипает само.
        if request.turn == 0 or not hashes:
            best = self._tiebreak.pick(candidates, ctx)
            return RoutingDecision(upstream=best, reason="первый ход сессии: балансировка",
                                   strategy=self.name, candidates_considered=len(candidates))

        sticky_id, hit_blocks = ctx.prefix.best_upstream(hashes)
        by_id = {u.id: u for u in candidates}
        sticky = by_id.get(sticky_id) if sticky_id else None

        if sticky is not None:
            loads = [ctx.load_of(u.id) for u in candidates]
            mean_load = sum(loads) / len(loads) if loads else 0.0

            overloaded = mean_load > 0 and ctx.load_of(sticky.id) > self._overload * mean_load
            # Ожидаемое попадание по истории: все блоки, кроме относящихся
            # к свежедобавленному ходу. Если фактическое заметно ниже —
            # сессия, скорее всего, вытеснена.
            expected = self._expected_hit_blocks(request, ctx)
            evicted = hit_blocks < self._hit_ratio * expected

            if not overloaded and not evicted:
                return RoutingDecision(
                    upstream=sticky,
                    reason=f"залипание сессии (попадание {hit_blocks} блоков)",
                    expected_hit_blocks=hit_blocks,
                    strategy=self.name,
                    candidates_considered=len(candidates),
                )
            reason_detail = "перегружен" if overloaded else "кэш сессии вытеснен"
        else:
            reason_detail = "сессия неизвестна"

        best = self._tiebreak.pick(candidates, ctx)
        return RoutingDecision(
            upstream=best,
            reason=f"балансировка: {reason_detail}",
            strategy=self.name,
            candidates_considered=len(candidates),
        )

    def _expected_hit_blocks(self, request: ChatRequest, ctx: RoutingContext) -> int:
        """Сколько блоков должно было бы попасть, если сессия жива.

        Оценивается по истории в самом запросе, исключая последний ход:
        именно он добавлен сейчас и в кэше заведомо отсутствует.
        """
        if len(request.messages) <= 1:
            return 0
        history = "".join(f"{m.role}\n{m.content}\n" for m in request.messages[:-1])
        return estimate_tokens(history) // ctx.block_tokens

    def on_dispatched(self, request, upstream, ctx) -> None:
        tenant_id = request.tenant.id if request.tenant else "anon"
        hashes = block_hashes(request.prompt_text(), ctx.block_tokens, salt=tenant_id)
        if hashes:
            ctx.prefix.record(hashes, upstream.id)


class DualMapStrategy(RoutingStrategy):
    """Двойное отображение по DualMap (§3.3.6.1, §3.3.6.3).

    Каждый запрос отображается двумя независимыми хеш-функциями от
    префикса в двух кандидатов, из которых выбирается лучший по текущему
    состоянию.

    Почему это разрывает конфликт affinity и баланса: по теореме Power of
    Two Choices максимальное отклонение нагрузки падает с `Θ(sqrt(m·log n / n))`
    при одном кандидате до `log log n` при двух — экспоненциально меньше.
    При этом для m запросов с общим префиксом cache hit rate составляет
    `max(0, 1 − 2/m)` против `max(0, 1 − 1/m)` у чистой affinity: при
    больших m разница исчезает.

    Почему ровно два, а не больше: рост числа кандидатов уменьшает
    отклонение лишь логарифмически, зато разбрасывает одинаковые префиксы
    по большему числу инстансов и разрушает локальность кэша. Глобальная
    стратегия «опросить всех» эквивалентна d = n: выигрыша по нагрузке
    почти нет, а reuse деградирует сильно.

    Правило выбора — не Min TTFT. Поштучная минимизация TTFT осциллирует:
    очередь на кэширующем инстансе растёт, запрос уходит на второй с
    промахом, очередь спадает, следующий возвращается обратно. Вместо неё
    — **держаться affinity, пока прогноз укладывается в SLO**.

    Требует видимости состояния инстансов (сценарий 1 из §3.3.6.8),
    поэтому включается флагом, а не по умолчанию.
    """

    name = "dualmap"

    def select(self, request, candidates, ctx) -> RoutingDecision:
        n = len(candidates)
        if n == 1:
            return RoutingDecision(upstream=candidates[0], reason="единственный апстрим",
                                   strategy=self.name, candidates_considered=1)

        tenant_id = request.tenant.id if request.tenant else "anon"
        hashes = block_hashes(request.prompt_text(), ctx.block_tokens, salt=tenant_id)
        key = hashes[-1] if hashes else request.session_key(tenant_id).encode()

        i1 = int.from_bytes(hashlib.blake2b(key, digest_size=8, person=b"dualmap1").digest(), "big") % n
        i2 = int.from_bytes(hashlib.blake2b(key, digest_size=8, person=b"dualmap2").digest(), "big") % n
        if i1 == i2:
            # Кандидатов всегда ровно два — иначе теряется весь смысл P2C.
            i2 = (i1 + 1) % n
        c1, c2 = candidates[i1], candidates[i2]

        hit1 = ctx.prefix.hit_len(hashes, c1.id) if hashes else 0
        hit2 = ctx.prefix.hit_len(hashes, c2.id) if hashes else 0

        if hit1 == hit2:
            # При равном попадании всегда выбираем менее загруженного.
            best = c1 if ctx.load_of(c1.id) <= ctx.load_of(c2.id) else c2
            return RoutingDecision(upstream=best, reason="равное попадание: менее загруженный",
                                   expected_hit_blocks=hit1, strategy=self.name,
                                   candidates_considered=2)

        best_cache, best_hit = (c1, hit1) if hit1 > hit2 else (c2, hit2)
        other = c2 if best_cache is c1 else c1

        compute_tokens = max(0, request.prompt_tokens_est - best_hit * ctx.block_tokens)
        budget = ctx.slo.ttft_budget_ms(request.prompt_tokens_est)
        if ctx.estimated_ttft_ms(best_cache.id, compute_tokens) <= budget:
            return RoutingDecision(upstream=best_cache,
                                   reason=f"приоритет affinity (попадание {best_hit} блоков)",
                                   expected_hit_blocks=best_hit, strategy=self.name,
                                   candidates_considered=2)

        # Прогноз вышел за SLO — деградируем в load-aware.
        best = best_cache if ctx.load_of(best_cache.id) <= ctx.load_of(other.id) else other
        return RoutingDecision(upstream=best,
                               reason="прогноз TTFT вне SLO: переход на балансировку",
                               expected_hit_blocks=best_hit if best is best_cache else 0,
                               strategy=self.name, candidates_considered=2)

    def on_dispatched(self, request, upstream, ctx) -> None:
        tenant_id = request.tenant.id if request.tenant else "anon"
        hashes = block_hashes(request.prompt_text(), ctx.block_tokens, salt=tenant_id)
        if hashes:
            ctx.prefix.record(hashes, upstream.id)


STRATEGIES: dict[str, type[RoutingStrategy]] = {
    LeastLoadStrategy.name: LeastLoadStrategy,
    ConsistentHashStrategy.name: ConsistentHashStrategy,
    SessionStrategy.name: SessionStrategy,
    DualMapStrategy.name: DualMapStrategy,
}


def build_strategy(cfg) -> RoutingStrategy:
    cls = STRATEGIES.get(cfg.strategy)
    if cls is None:
        raise ValueError(f"неизвестная стратегия {cfg.strategy!r}")
    if cls is SessionStrategy:
        return SessionStrategy(overload_factor=cfg.overload_factor,
                               hit_ratio_factor=cfg.hit_ratio_factor)
    return cls()
