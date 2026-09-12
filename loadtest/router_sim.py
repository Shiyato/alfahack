"""Сравнение стратегий маршрутизации на агентском профиле (§3.3.6, §5.3).

Что доказывается. В §2.1 сказано: пара метрик «cache hit rate + CV»
образует Парето-фронт, и один график, где наша точка лежит выше и левее
baseline-ов, — законченное доказательство качества роутера.

Генератор воспроизводит форму агентской нагрузки из §5.3, а не
«случайные независимые промпты»:

  • общий системный промпт у всех сессий (18–20% переиспользований);
  • сессия из нескольких ходов, каждый расширяет историю предыдущего
    (~67% переиспользований — внутрисессионные);
  • тяжёлый хвост по длине сессий (верхние 25% дают >80% токенов);
  • много одновременных сессий относительно числа апстримов — иначе
    балансировка не работает и выводы нерепрезентативны.

Важная калибровка ожиданий (§3.3.6.3d): при нормальной нагрузке разница
между стратегиями в пределах 1%, потому что у кластера есть запас и
ошибки маршрутизации ничего не стоят. Выигрыш появляется у колена
насыщения. Поэтому здесь свип по интенсивности, а не один замер.

Запуск:  PYTHONPATH=. uv run python loadtest/router_sim.py
"""

from __future__ import annotations

import random
import statistics
from collections import OrderedDict
from dataclasses import dataclass, field

from gateway.admission.load import LoadEstimator
from gateway.core.config import SLOConfig
from gateway.core.domain import ChatRequest, Message, Tenant, Upstream
from gateway.router.base import RoutingContext
from gateway.router.prefix import PrefixTable, block_hashes, estimate_tokens
from gateway.router.strategies import (
    ConsistentHashStrategy,
    DualMapStrategy,
    LeastLoadStrategy,
    SessionStrategy,
)

SYSTEM_PROMPT = "Ты — агент разработки. " * 400          # ~2000 токенов, общий у всех
BLOCK_TOKENS = 256


@dataclass
class Session:
    sid: int
    turns_total: int
    turn: int = 0
    messages: list[Message] = field(default_factory=list)

    def next_request(self, tenant: Tenant, rnd: random.Random) -> ChatRequest:
        if not self.messages:
            self.messages = [
                Message("system", SYSTEM_PROMPT),
                Message("user", f"задача {self.sid}: " + "контекст файла " * rnd.randint(150, 400)),
            ]
        else:
            # Каждый ход дописывает историю — так и растёт общий префикс.
            self.messages.append(Message("assistant", "ответ " * rnd.randint(30, 80)))
            self.messages.append(Message("user", "следующий шаг " * rnd.randint(20, 60)))
        self.turn += 1
        req = ChatRequest(model="main", messages=list(self.messages), tenant=tenant)
        req.prompt_tokens_est = estimate_tokens(req.prompt_text())
        return req


def make_sessions(n: int, rnd: random.Random) -> list[Session]:
    """Тяжёлый хвост: большинство сессий короткие, меньшинство — очень длинные."""
    out = []
    for i in range(n):
        turns = 2 if rnd.random() < 0.75 else rnd.randint(8, 30)
        out.append(Session(sid=i, turns_total=turns))
    return out


@dataclass
class UpstreamSim:
    """Модель инстанса: очередь префилла и KV-кэш ограниченного размера.

    Ограничение кэша принципиально. Первая версия держала кэш безграничным
    множеством, и через несколько сотен запросов общий системный префикс
    оседал на всех инстансах — доля попаданий у всех стратегий сходилась,
    а выигрыш affinity занижался. Безграничный кэш измеряет не роутер,
    а собственную щедрость.
    """

    uid: str
    capacity_tokens_per_s: float
    cache_blocks: int = 600            # ёмкость KV-кэша в блоках
    cache: "OrderedDict[bytes, None]" = field(default_factory=OrderedDict)
    pending_tokens: float = 0.0
    served: int = 0
    served_tokens: int = 0
    evicted: int = 0

    def hit_blocks(self, hashes: list[bytes]) -> int:
        n = 0
        for h in hashes:
            if h not in self.cache:
                break
            self.cache.move_to_end(h)
            n += 1
        return n

    def admit(self, hashes: list[bytes], prompt_tokens: int) -> tuple[int, float]:
        hit = self.hit_blocks(hashes)
        compute = max(0, prompt_tokens - hit * BLOCK_TOKENS)
        self.pending_tokens += compute
        for h in hashes:
            self.cache[h] = None
            self.cache.move_to_end(h)
        while len(self.cache) > self.cache_blocks:
            self.cache.popitem(last=False)
            self.evicted += 1
        self.served += 1
        self.served_tokens += prompt_tokens
        return hit, compute

    def drain(self, dt: float) -> None:
        self.pending_tokens = max(0.0, self.pending_tokens - self.capacity_tokens_per_s * dt)


# Ёмкость откалибрована так, чтобы интенсивность 40 запросов/с давала
# загрузку около колена насыщения. Замер на низкой загрузке показал бы,
# что все стратегии одинаковы, и обесценил бы сравнение (§3.3.6.3d).
# Расчёт: средняя длина входа 5031 токенов, при 93% попаданий вычислять
# надо ~350 токенов на запрос, при 40 запросах/с — 14000 токенов/с на
# кластер, то есть 3500 на апстрим.
def run_strategy(strategy, *, n_upstreams=4, n_sessions=400, arrival_per_s=40.0,
                 capacity=3800.0, cache_blocks=600, seed=7) -> dict:
    rnd = random.Random(seed)
    ups = [UpstreamSim(f"u{i}", capacity, cache_blocks=cache_blocks)
           for i in range(n_upstreams)]
    upstreams = [Upstream(id=u.uid, base_url=f"http://{u.uid}", model="m") for u in ups]
    by_id = {u.uid: u for u in ups}

    est = LoadEstimator()
    table = PrefixTable(ttl_s=600.0)
    ctx = RoutingContext(load_estimator=est, prefix_table=table,
                         slo=SLOConfig(), block_tokens=BLOCK_TOKENS)
    tenant = Tenant(id="agent", name="agent")

    sessions = make_sessions(n_sessions, rnd)
    active = [s for s in sessions]
    rnd.shuffle(active)

    total_blocks = hit_blocks = 0
    load_samples: list[list[float]] = []
    dt = 1.0 / arrival_per_s
    processed = 0

    while active:
        s = active[rnd.randrange(len(active))]
        req = s.next_request(tenant, rnd)
        hashes = block_hashes(req.prompt_text(), BLOCK_TOKENS, salt=tenant.id)

        # Синхронизируем взгляд роутера с состоянием инстансов.
        for u in ups:
            est.upstream(u.uid).pending_prefill_tokens = int(u.pending_tokens)

        decision = strategy.select(req, upstreams, ctx)
        target = by_id[decision.upstream.id]
        hit, _ = target.admit(hashes, req.prompt_tokens_est)
        strategy.on_dispatched(req, decision.upstream, ctx)

        total_blocks += len(hashes)
        hit_blocks += hit
        processed += 1

        for u in ups:
            u.drain(dt)
        if processed > 200:      # прогрев исключаем (§5.1)
            load_samples.append([u.pending_tokens for u in ups])

        if s.turn >= s.turns_total:
            active.remove(s)

    cvs = []
    for sample in load_samples:
        m = statistics.fmean(sample)
        if m > 0:
            cvs.append(statistics.pstdev(sample) / m)
    served = [u.served for u in ups]
    tokens = [u.served_tokens for u in ups]

    return {
        "strategy": strategy.name,
        "cache_hit_rate": hit_blocks / total_blocks if total_blocks else 0.0,
        "cv": statistics.fmean(cvs) if cvs else 0.0,
        "max_over_mean": max(tokens) / statistics.fmean(tokens) if tokens else 0.0,
        "requests": processed,
        "per_upstream": served,
        "evicted": sum(u.evicted for u in ups),
    }


def main() -> None:
    print("Сравнение стратегий маршрутизации на агентском профиле (§3.3.6)")
    print("4 апстрима, 400 сессий с тяжёлым хвостом, общий системный промпт.\n")

    strategies = [LeastLoadStrategy(), ConsistentHashStrategy(),
                  SessionStrategy(), DualMapStrategy()]

    print("--- Свип по интенсивности: где вообще проявляется разница (§3.3.6.3d) ---\n")
    print(f"{'интенсивность':>14} " + "".join(f"{s.name:>18}" for s in strategies))
    print("-" * (14 + 18 * len(strategies)))
    for rate, label in ((20.0, "0.5x (запас)"), (40.0, "1.0x (норма)"),
                        (60.0, "1.5x (колено)"), (80.0, "2.0x (перегрузка)")):
        cells = []
        for s in strategies:
            r = run_strategy(type(s)() if not isinstance(s, SessionStrategy) else SessionStrategy(),
                             arrival_per_s=rate)
            cells.append(f"{r['cache_hit_rate']:>8.1%}/{r['cv']:>8.2f}")
        print(f"{label:>14} " + "".join(f"{c:>18}" for c in cells))
    print("\n(в ячейке: доля попаданий в кэш / CV нагрузки — чем выше и левее, тем лучше)")

    print("\n--- Детально у колена насыщения (1.5x) ---\n")
    print(f"{'стратегия':<18} {'cache hit':>10} {'CV':>7} {'max/mean':>9}  распределение запросов")
    print("-" * 78)
    for s in strategies:
        fresh = SessionStrategy() if isinstance(s, SessionStrategy) else type(s)()
        r = run_strategy(fresh, arrival_per_s=60.0)
        print(f"{r['strategy']:<18} {r['cache_hit_rate']:>10.1%} {r['cv']:>7.2f} "
              f"{r['max_over_mean']:>9.2f}  {r['per_upstream']}")


if __name__ == "__main__":
    main()
