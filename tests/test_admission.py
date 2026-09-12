"""Тесты admission control (§3.3.3).

Проверяется поведение, которое легко сломать незаметно: порядок выдачи,
справедливость между тенантами, учёт квот. Каждый тест утверждает
что-то, что должно быть правдой по документу, а не «что функция
что-то вернула».
"""

from __future__ import annotations

import asyncio
import time

import pytest

from gateway.admission.load import EWMA, LoadEstimator, UpstreamLoad
from gateway.admission.queue import (
    PriorityQueue,
    QueueFull,
    QueuedRequest,
    SelectiveDispatcher,
)
from gateway.admission.quota import QuotaLedger, TokenBucket
from gateway.core.domain import ServiceClass


def q_req(sc: ServiceClass, tenant: str, tokens: int = 100) -> QueuedRequest:
    return QueuedRequest(
        service_class=sc, tenant_id=tenant,
        enqueued_at=time.monotonic(), estimated_tokens=tokens,
    )


# --------------------------------------------------------------------------
# Очередь
# --------------------------------------------------------------------------


def test_interactive_obgonyaet_batch():
    """Интерактивный запрос не ждёт за батчем — ради этого всё и строится.

    Реальная боль, из-за которой такой прокси пишут: пользователь ждёт
    минуты, потому что его запрос стоит в одной очереди с массовой
    обработкой (§2.2).
    """
    q = PriorityQueue()
    for _ in range(10):
        q.put(q_req(ServiceClass.BATCH, "batch-tenant"))
    q.put(q_req(ServiceClass.INTERACTIVE, "human"))

    first = q.get_nowait()
    assert first.service_class is ServiceClass.INTERACTIVE, (
        "интерактивный запрос, поставленный последним, обязан выйти первым"
    )


def test_poryadok_klassov_polnyi():
    q = PriorityQueue()
    for sc in (ServiceClass.BACKGROUND, ServiceClass.BATCH,
               ServiceClass.AGENT, ServiceClass.INTERACTIVE):
        q.put(q_req(sc, f"t-{sc.value}"))
    got = [q.get_nowait().service_class for _ in range(4)]
    assert got == [ServiceClass.INTERACTIVE, ServiceClass.AGENT,
                   ServiceClass.BATCH, ServiceClass.BACKGROUND]


def test_fairshare_ne_daet_shumnomu_zabrat_vsyo():
    """Шумный сосед не должен съесть весь класс.

    Тонкость, на которой сломалась первая реализация: если считать
    справедливость в момент постановки, тенант, выложивший пачку разом,
    получит одинаковый счётчик на все запросы и пройдёт подряд. Решение
    обязано приниматься при выдаче.
    """
    q = PriorityQueue()
    for _ in range(5):
        q.put(q_req(ServiceClass.AGENT, "шумный"))
    for _ in range(2):
        q.put(q_req(ServiceClass.AGENT, "тихий"))

    order = []
    while (r := q.get_nowait()) is not None:
        order.append(r.tenant_id)

    # Тихий обязан получить оба слота в первой половине выдачи.
    assert order.index("тихий") <= 2, f"тихий тенант задвинут в конец: {order}"
    assert order.count("тихий") == 2


def test_fifo_vnutri_tenanta():
    q = PriorityQueue()
    reqs = [q_req(ServiceClass.AGENT, "t") for _ in range(5)]
    for r in reqs:
        q.put(r)
    got = [q.get_nowait() for _ in range(5)]
    assert got == reqs, "внутри одного тенанта порядок обязан быть FIFO"


def test_pereplonenie_otkaz_srazu():
    """Отказ до постановки, а не после ожидания: честный быстрый 429
    лучше таймаута на 120-й секунде (§3.3.3)."""
    q = PriorityQueue(max_depth=3)
    for _ in range(3):
        q.put(q_req(ServiceClass.AGENT, "t"))
    with pytest.raises(QueueFull):
        q.put(q_req(ServiceClass.AGENT, "t"))


def test_izyatie_broshennogo_zaprosa():
    """Клиент отвалился, пока запрос ждал в очереди.

    Без изъятия он дождётся очереди и займёт слот апстрима, который
    никому не нужен, — утечка ёмкости на входе.
    """
    q = PriorityQueue()
    a, b = q_req(ServiceClass.AGENT, "t"), q_req(ServiceClass.AGENT, "t")
    q.put(a)
    q.put(b)
    assert q.remove(a) is True
    assert len(q) == 1
    assert q.get_nowait() is b
    assert q.remove(a) is False, "повторное изъятие обязано вернуть False"


def test_sbros_okna_fairshare():
    q = PriorityQueue()
    for _ in range(3):
        q.put(q_req(ServiceClass.AGENT, "старожил"))
    while q.get_nowait():
        pass
    q.reset_fairshare()
    q.put(q_req(ServiceClass.AGENT, "старожил"))
    q.put(q_req(ServiceClass.AGENT, "новичок"))
    # После сброса окна оба в равных условиях, порядок — FIFO.
    assert q.get_nowait().tenant_id == "старожил"


# --------------------------------------------------------------------------
# Селективная отдача
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_selektivnaya_otdacha_zhdet_gotovogo():
    """Запрос не уходит апстриму, пока тот занят (§3.3.3.7).

    Смысл: запрос, ещё не отданный апстриму, можно перенаправить;
    отданный — уже нет.
    """
    q = PriorityQueue()
    disp = SelectiveDispatcher(q, max_wait_s=2.0, poll_interval_s=0.005)
    ready: list[str] = []

    task = asyncio.create_task(
        disp.acquire_slot(q_req(ServiceClass.AGENT, "t"), lambda: ready)
    )
    await asyncio.sleep(0.05)
    assert not task.done(), "запрос ушёл, хотя готовых апстримов не было"

    ready.append("mock-a")
    upstream, waited = await task
    assert upstream == "mock-a"
    assert waited >= 0.04, "время ожидания в очереди не учтено"


@pytest.mark.asyncio
async def test_selektivnaya_otdacha_taimaut():
    q = PriorityQueue()
    disp = SelectiveDispatcher(q, max_wait_s=0.1, poll_interval_s=0.005)
    from gateway.admission.queue import QueueTimeout

    with pytest.raises(QueueTimeout):
        await disp.acquire_slot(q_req(ServiceClass.AGENT, "t"), lambda: [])


# --------------------------------------------------------------------------
# Квоты
# --------------------------------------------------------------------------


def test_vedro_ogranichivaet_vsplesk_razmerom():
    """Token bucket выбран потому, что агентская нагрузка всплесковая:
    агент выпускает серию запросов по мере обхода инструментов (§3.3.3.5)."""
    b = TokenBucket(capacity=1000, refill_per_second=100)
    assert b.try_take(1000) is True, "всплеск в размер ведра обязан пройти целиком"
    assert b.try_take(1) is False, "сверх ведра — отказ"


def test_retry_after_schitaetsya_iz_skorosti_popolneniya():
    """429 без подсказки, когда повторять, заставляет клиента гадать (§3.3.3.6)."""
    b = TokenBucket(capacity=100, refill_per_second=10)
    b.try_take(100)
    assert b.retry_after_s(50) == pytest.approx(5.0, abs=0.1)


def test_kvota_po_tokenam_a_ne_po_zaprosam():
    """Запрос на 100k токенов и на 200 нагружают систему на два порядка
    по-разному — счёт запросов этого не видит."""
    led = QuotaLedger()
    kw = dict(keys=["t"], tokens_per_minute=10000, requests_per_minute=100000,
              max_concurrent=None, burst_multiplier=1.0)

    heavy = led.check_and_reserve(estimated_tokens=10000, **kw)
    assert heavy.allowed
    light = led.check_and_reserve(estimated_tokens=1, **kw)
    assert not light.allowed, "один тяжёлый запрос обязан исчерпать токенную квоту"
    assert light.limit_kind == "tokens"


def test_schet_zaprosov_lovit_shkval_legkih():
    """Пара «токены + запросы» нужна одновременно: токены ловят тяжёлые
    запросы, счёт запросов — шквал лёгких."""
    led = QuotaLedger()
    kw = dict(keys=["t"], tokens_per_minute=10_000_000, requests_per_minute=5,
              max_concurrent=None, burst_multiplier=1.0)
    for _ in range(5):
        assert led.check_and_reserve(estimated_tokens=1, **kw).allowed
    d = led.check_and_reserve(estimated_tokens=1, **kw)
    assert not d.allowed and d.limit_kind == "requests"


def test_korrektirovka_po_faktu_vozvrashaet_pereotsenku():
    """Выход заранее неизвестен: списываем оценку, корректируем по факту."""
    led = QuotaLedger()
    kw = dict(keys=["t"], tokens_per_minute=6000, requests_per_minute=1000,
              max_concurrent=None, burst_multiplier=1.0)
    d = led.check_and_reserve(estimated_tokens=1000, **kw)
    assert d.remaining_tokens == pytest.approx(5000, abs=1)
    led.settle(["t"], reserved=1000, actual=100)
    after = led.check_and_reserve(estimated_tokens=0, **kw)
    assert after.remaining_tokens == pytest.approx(5900, abs=1)


def test_pererashod_spisyvaetsya_a_ne_rastvoryaetsya():
    """Если фактический расход превысил оценку, разница обязана
    отразиться на следующих запросах того же тенанта."""
    led = QuotaLedger()
    kw = dict(keys=["t"], tokens_per_minute=6000, requests_per_minute=1000,
              max_concurrent=None, burst_multiplier=1.0)
    led.check_and_reserve(estimated_tokens=100, **kw)
    led.settle(["t"], reserved=100, actual=1000)
    after = led.check_and_reserve(estimated_tokens=0, **kw)
    assert after.remaining_tokens == pytest.approx(5000, abs=1)


def test_otkaz_ne_spisyvaet_kvotu_po_drugim_izmereniyam():
    """Запрос, отклонённый по последнему измерению, не должен списать
    квоту по первым — иначе он накажет чужие запросы."""
    led = QuotaLedger()
    kw = dict(keys=["tenant", "team"], tokens_per_minute=1000,
              requests_per_minute=0, max_concurrent=None, burst_multiplier=1.0)
    denied = led.check_and_reserve(estimated_tokens=5000, **kw)
    assert not denied.allowed
    ok = led.check_and_reserve(estimated_tokens=1000, **kw)
    assert ok.allowed, "отклонённый запрос списал чужую квоту"


# --------------------------------------------------------------------------
# Оценка нагрузки
# --------------------------------------------------------------------------


def test_konstruktor_otvergaet_konfig_bez_gisterezisa():
    """Контур без гистерезиса дребезжит на границе порога (§3.3.3.3).
    Это конфигурационная ошибка, которую нельзя пропустить молча."""
    with pytest.raises(ValueError, match="гистерезис"):
        LoadEstimator(reject_threshold=0.8, resume_threshold=0.9)


def test_gisterezis_ne_perklyuchaetsya_na_granitse():
    est = LoadEstimator(ewma_alpha=1.0, reject_threshold=1.0, resume_threshold=0.8)

    est._pressure.update(1.2)
    assert est.should_reject() is True

    # Значение между порогами: режим обязан сохраниться.
    est._pressure.update(0.9)
    assert est.should_reject() is True, "вышли из режима отказов раньше нижнего порога"

    est._pressure.update(0.7)
    assert est.should_reject() is False


def test_ewma_sglazhivaet_vybros():
    e = EWMA(alpha=0.2)
    for _ in range(10):
        e.update(1.0)
    e.update(10.0)
    assert e.value < 3.0, "одиночный выброс не должен рвать оценку"


def test_priznak_zanyatosti_binarnyi_a_ne_porogovyi():
    """Фиксированный порог параллелизма не работает: предельное число
    одновременных запросов гуляет от 20 до 50 на одной модели (§3.3.3.7)."""
    u = UpstreamLoad(upstream_id="u1")
    now = time.monotonic()

    u.has_pending_queue = False
    u.state_fresh_at = now
    assert u.busy(now=now) is False

    u.has_pending_queue = True
    assert u.busy(now=now) is True

    # Устаревший сигнал не используется.
    u.state_fresh_at = now - 100
    assert u.busy(now=now) is False


def test_429_ot_apstrima_priznak_zanyatosti():
    """Если апстрим сам сказал «слишком много», это явное признание
    перегрузки — самый честный из доступных сигналов."""
    u = UpstreamLoad(upstream_id="u1")
    now = time.monotonic()
    u.last_429_at = now
    assert u.busy(now=now) is True


def test_metrika_nagruzki_predpochitaet_pending_prefill():
    """Латентность префилла масштабируется с числом токенов, поэтому
    pending prefill-токены — лучшая метрика, если апстрим её отдаёт."""
    u = UpstreamLoad(upstream_id="u1", inflight_tokens=500)
    assert u.load_metric() == 500.0
    u.pending_prefill_tokens = 12000
    assert u.load_metric() == 12000.0


# --------------------------------------------------------------------------
# Живая блокировка диспетчера — дефект, найденный под нагрузкой
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ozhidayushchii_zabiraet_tolko_svoi_zapros():
    """Первая версия диспетчера вынимала из очереди голову и, если та
    принадлежала другому, пыталась передать ей слот через future,
    которого ни у кого не было. Чужой запрос терялся, а его корутина
    продолжала крутиться.

    Под нагрузкой это дало живую блокировку: 14 миллионов оборотов на
    четыре тысячи запросов и среднее ожидание 30 секунд при полностью
    свободных апстримах.
    """
    q = PriorityQueue()
    disp = SelectiveDispatcher(q, max_wait_s=3.0, poll_interval_s=0.002)
    ready = ["u1"]

    reqs = [q_req(ServiceClass.AGENT, f"t{i}") for i in range(5)]
    tasks = [asyncio.create_task(disp.acquire_slot(r, lambda: ready)) for r in reqs]
    done = await asyncio.gather(*tasks)

    assert len(done) == 5, "часть запросов потерялась"
    assert len(q) == 0, "в очереди остались запросы после обслуживания всех"
    stats = disp.stats()
    assert stats["dispatched"] == 5
    # Оборотов должно быть на порядки меньше числа запросов, а не наоборот.
    assert stats["spins"] < 200, f"диспетчер крутится вхолостую: {stats['spins']} оборотов"


@pytest.mark.asyncio
async def test_prioritetnyi_zapros_obgonyaet_v_dispetchere():
    """Приоритет обязан работать и в ожидании слота, а не только
    в самой очереди."""
    q = PriorityQueue()
    disp = SelectiveDispatcher(q, max_wait_s=3.0, poll_interval_s=0.002)
    ready: list[str] = []
    order: list[str] = []

    async def run(req, name):
        await disp.acquire_slot(req, lambda: ready)
        order.append(name)

    batch = [asyncio.create_task(run(q_req(ServiceClass.BATCH, f"b{i}"), f"batch-{i}"))
             for i in range(3)]
    await asyncio.sleep(0.02)
    inter = asyncio.create_task(
        run(q_req(ServiceClass.INTERACTIVE, "human"), "interactive")
    )
    await asyncio.sleep(0.02)

    ready.append("u1")
    await asyncio.gather(*batch, inter)
    assert order[0] == "interactive", f"интерактивный не обогнал батч: {order}"


@pytest.mark.asyncio
async def test_taimaut_ubiraet_zapros_iz_ocheredi():
    """Брошенный запрос, оставшийся в очереди, дождётся своей очереди и
    займёт слот апстрима, который уже никому не нужен."""
    from gateway.admission.queue import QueueTimeout

    q = PriorityQueue()
    disp = SelectiveDispatcher(q, max_wait_s=0.05, poll_interval_s=0.002)
    with pytest.raises(QueueTimeout):
        await disp.acquire_slot(q_req(ServiceClass.AGENT, "t"), lambda: [])
    assert len(q) == 0, "запрос остался в очереди после таймаута"


@pytest.mark.asyncio
async def test_otmena_ubiraet_zapros_iz_ocheredi():
    q = PriorityQueue()
    disp = SelectiveDispatcher(q, max_wait_s=5.0, poll_interval_s=0.002)
    task = asyncio.create_task(
        disp.acquire_slot(q_req(ServiceClass.AGENT, "t"), lambda: [])
    )
    await asyncio.sleep(0.02)
    assert len(q) == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(q) == 0, "отменённый запрос остался в очереди"
