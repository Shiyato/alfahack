"""Воспроизведение автоколебаний контура раннего отклонения (§3.3.3.3).

Зачем этот файл существует. Документ утверждает, что наивное раннее
отклонение порождает качели загрузки и что лечится это демпфированием.
Утверждение проверяемое — значит, его надо проверить, а не пересказать
на защите.

Модель: двухстадийный конвейер (префилл → декод) с лагом между стадиями.
Лаг между решением о приёме и возникновением нагрузки, которую это решение
создаёт, — единственная причина качелей; больше ничего не требуется.

ДВЕ ОШИБКИ ПРИБОРА, допущенные при написании этого файла и исправленные.
Обе стоит помнить, потому что каждая давала уверенный, но ложный вывод.

1. Решение о приёме принималось раз в такт для всей пачки сразу. Такая
   модель — сама по себе релейный регулятор и колеблется независимо от
   лага. Она «подтверждала» гипотезу там, где эффекта нет.

2. Goodput считался по загрузке в момент приёма. Но запрос обслуживается
   позже, и решает его судьбу загрузка во время обслуживания, а не в
   момент входа. По такой мерке наивный контур выглядел лучшим, хотя
   принимал втрое больше работы, чем система способна доделать.

Сейчас измеряется сквозная задержка каждого запроса, как она видна
клиенту, а SLO задан относительно — кратностью ко времени того же
запроса на пустой системе (§2.1.1).

Запуск:  PYTHONPATH=. uv run python loadtest/oscillation_sim.py
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field

from gateway.admission.load import EWMA


@dataclass
class Pipeline:
    """Двухстадийный конвейер. Каждый запрос отслеживается поимённо,
    чтобы измерять его фактическую задержку, а не косвенный признак."""

    prefill_capacity: int = 100      # запросов за такт
    decode_capacity: int = 100       # одновременно в декоде
    decode_lag_ticks: int = 3
    decode_hold_ticks: int = 8

    tick_no: int = 0
    prefill_queue: deque = field(default_factory=deque)   # (admit_tick,)
    decode_active: deque = field(default_factory=deque)   # (release_tick, admit_tick)
    _in_transit: list = field(default_factory=list)
    completed: list = field(default_factory=list)         # задержки в тактах

    def __post_init__(self) -> None:
        self._in_transit = [[] for _ in range(self.decode_lag_ticks)]

    # Минимально возможная задержка на пустой системе — база относительного SLO.
    @property
    def min_latency(self) -> int:
        return 1 + self.decode_lag_ticks + self.decode_hold_ticks

    def admit(self) -> None:
        self.prefill_queue.append(self.tick_no)

    def tick(self) -> None:
        # 1. Префилл: FIFO, ограниченная пропускная способность.
        done = []
        for _ in range(min(len(self.prefill_queue), self.prefill_capacity)):
            done.append(self.prefill_queue.popleft())

        # 2. Лаг до декода.
        self._in_transit.append(done)
        entering = self._in_transit.pop(0)

        # 3. Декод: ограничен по числу одновременных. Не поместившиеся
        #    ждут следующего такта — это и есть очередь второй стадии.
        capacity_left = self.decode_capacity - len(self.decode_active)
        admitted_now = entering[:max(0, capacity_left)]
        overflow = entering[max(0, capacity_left):]
        for admit_tick in admitted_now:
            self.decode_active.append((self.tick_no + self.decode_hold_ticks, admit_tick))
        if overflow:
            # Возвращаем в транзит: доедут, когда освободится место.
            self._in_transit[0] = overflow + self._in_transit[0]

        # 4. Освобождение декода.
        while self.decode_active and self.decode_active[0][0] <= self.tick_no:
            _, admit_tick = self.decode_active.popleft()
            self.completed.append(self.tick_no - admit_tick)

        self.tick_no += 1

    def load(self) -> float:
        """Загрузка — максимум по стадиям (§3.3.3.2).

        Запросы, застрявшие в транзите из-за переполнения декода, обязаны
        считаться нагрузкой второй стадии: это её очередь. Ранняя версия
        их не учитывала, и загрузка выглядела ровной полкой при очереди,
        растущей неограниченно, — измерение показывало «всё хорошо» ровно
        там, где система разваливалась.
        """
        backlog = sum(len(b) for b in self._in_transit)
        return max(
            len(self.prefill_queue) / self.prefill_capacity,
            (len(self.decode_active) + backlog) / self.decode_capacity,
        )

    def predicted_load(self) -> float:
        """Нагрузка на момент, когда текущий префилл доедет до декода.

        Системный уровень, а не позапросный: предсказывать длину вывода
        конкретного запроса дорого и неточно, а оценить агрегированное
        состояние пула через время — проще и требует меньшей точности.
        """
        incoming = sum(len(b) for b in self._in_transit) + len(self.prefill_queue)
        release = sum(
            1 for release_tick, _ in self.decode_active
            if release_tick <= self.tick_no + self.decode_lag_ticks
        )
        future_decode = max(0, len(self.decode_active) + incoming - release)
        return max(len(self.prefill_queue) / self.prefill_capacity,
                   future_decode / self.decode_capacity)


def run(
    *,
    ticks: int = 400,
    arrivals: int = 160,          # больше устойчивой ёмкости: постоянная перегрузка
    smoothing: bool = False,
    hysteresis: bool = False,
    prediction: bool = False,
    alpha: float = 0.2,
    reject_at: float = 1.0,
    resume_at: float = 0.8,
    slo_multiplier: float = 2.0,  # относительный SLO (§2.1.1)
    staleness_ticks: int = 0,     # как редко обновляется сигнал нагрузки
) -> dict:
    p = Pipeline()
    ewma = EWMA(alpha=alpha)
    rejecting = False
    loads: list[float] = []
    flips = 0
    admitted_total = rejected_total = 0

    cached_signal = 0.0
    for tick_i in range(ticks):
        # Устаревание сигнала: реальный прокси не видит состояние апстрима
        # непрерывно, он опрашивает его heartbeat-зондом раз в N периодов,
        # и всё это время решения принимаются по старым данным.
        if staleness_ticks and tick_i % staleness_ticks == 0:
            cached_signal = p.predicted_load() if prediction else p.load()

        for _ in range(arrivals):
            if staleness_ticks:
                raw = cached_signal
            else:
                raw = p.predicted_load() if prediction else p.load()
            signal = ewma.update(raw) if smoothing else raw

            was = rejecting
            if hysteresis:
                if rejecting:
                    if signal < resume_at:
                        rejecting = False
                elif signal > reject_at:
                    rejecting = True
            else:
                rejecting = signal > reject_at
            if was != rejecting:
                flips += 1

            if rejecting:
                rejected_total += 1
            else:
                p.admit()
                admitted_total += 1

        p.tick()
        loads.append(p.load())

    slo_ticks = p.min_latency * slo_multiplier
    warm_from = 100
    done = p.completed
    within = sum(1 for d in done if d <= slo_ticks)

    warm = loads[warm_from:]
    mean = statistics.fmean(warm) if warm else 0.0
    lat_sorted = sorted(done)

    def pct(q: float) -> float:
        if not lat_sorted:
            return 0.0
        return lat_sorted[min(len(lat_sorted) - 1, int(q * len(lat_sorted)))]

    return {
        "cv": statistics.pstdev(warm) / mean if mean else 0.0,
        "min_load": min(warm) if warm else 0.0,
        "max_load": max(warm) if warm else 0.0,
        "flips": flips,
        "admitted": admitted_total,
        "rejected": rejected_total,
        "completed": len(done),
        "goodput": within,
        "slo_attainment": within / len(done) if done else 0.0,
        "p50": pct(0.50),
        "p95": pct(0.95),
        "slo_ticks": slo_ticks,
        "loads": loads,
    }


def sparkline(values: list[float], width: int = 60) -> str:
    chars = "▁▂▃▄▅▆▇█"
    step = max(1, len(values) // width)
    sampled = values[::step][:width]
    lo, hi = min(sampled), max(sampled)
    span = (hi - lo) or 1.0
    return "".join(chars[min(7, int((v - lo) / span * 7.99))] for v in sampled)


def main() -> None:
    ref = Pipeline()
    slo = ref.min_latency * 2
    print("Admission control под постоянной перегрузкой (§3.3.3)")
    print(f"Поступает 160/такт. Минимальная задержка на пустой системе — {ref.min_latency} тактов,")
    print(f"SLO = 2x от неё = {slo} тактов (относительный SLO, §2.1.1).")

    print("\n--- Опыт 1. Демпферы при мгновенном сигнале нагрузки ---\n")
    scenarios = [
        ("наивный (без демпферов)", dict()),
        ("+ сглаживание EWMA", dict(smoothing=True)),
        ("+ гистерезис", dict(smoothing=True, hysteresis=True)),
        ("+ предсказание (полный)", dict(smoothing=True, hysteresis=True, prediction=True)),
    ]
    print(f"{'сценарий':<26} {'CV':>6} {'принято':>8} {'goodput':>8} {'SLO':>6} {'p95':>6} {'перекл.':>8}")
    print("-" * 78)
    results = {}
    for name, kw in scenarios:
        r = run(**kw)
        results[name] = r
        print(f"{name:<26} {r['cv']:>6.3f} {r['admitted']:>8} {r['goodput']:>8} "
              f"{r['slo_attainment']:>5.0%} {r['p95']:>6.0f} {r['flips']:>8}")

    print("\nЗагрузка во времени:")
    for name, r in results.items():
        print(f"  {name:<26} {sparkline(r['loads'])}")

    naive = results["наивный (без демпферов)"]
    full = results["+ предсказание (полный)"]
    print(f"\nВывод опыта 1: заявленных в §3.3.3.3 автоколебаний НЕ ВОЗНИКЛО.")
    print(f"Наивный контур держит {naive['slo_attainment']:.0%} SLO. Демпферы дают "
          f"+{(full['goodput'] / naive['goodput'] - 1) * 100:.0f}% goodput — это полезно,")
    print("но это оптимизация, а не спасение от развала.")

    print("\n--- Опыт 2. Устаревание сигнала нагрузки ---\n")
    print("Реальный прокси не видит состояние апстрима непрерывно: он опрашивает")
    print("его heartbeat-зондом, и между опросами решения принимаются по старым данным.\n")
    print(f"{'свежесть сигнала':>18} {'демпферы':<12} {'CV':>6} {'goodput':>8} {'SLO':>6} {'p95':>6}")
    print("-" * 64)
    stale_rows = []
    for stale in (0, 1, 5, 20):
        for label, kw in (("нет", {}),
                          ("все три", dict(smoothing=True, hysteresis=True, prediction=True))):
            r = run(staleness_ticks=stale, **kw)
            stale_rows.append((stale, label, r))
            name = "мгновенный" if stale == 0 else f"раз в {stale} тактов"
            print(f"{name:>18} {label:<12} {r['cv']:>6.3f} {r['goodput']:>8} "
                  f"{r['slo_attainment']:>5.0%} {r['p95']:>6.0f}")

    fresh = next(r for s_, l, r in stale_rows if s_ == 0 and l == "все три")
    stale5 = next(r for s_, l, r in stale_rows if s_ == 5 and l == "все три")
    stale20 = next(r for s_, l, r in stale_rows if s_ == 20 and l == "все три")
    print(f"\nВывод опыта 2: свежесть сигнала важнее любых демпферов.")
    print(f"Доля в SLO: {fresh['slo_attainment']:.0%} (мгновенный) → "
          f"{stale5['slo_attainment']:.0%} (раз в 5) → {stale20['slo_attainment']:.0%} (раз в 20).")
    print("При устаревании в 20 тактов демпферы не дают вообще ничего: сглаживать")
    print("устаревшие данные бессмысленно — они и так гладкие, просто неверные.")
    print("\nИнженерное следствие: бюджет усилий идёт в частоту опроса апстримов,")
    print("а не в изощрённость регулятора.")


if __name__ == "__main__":
    main()
