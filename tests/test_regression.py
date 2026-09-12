"""Защита от возврата дефектов, которые стоили коллапса.

Обычные тесты проверяют, что механизм работает. Эти — что он работает
**с тем же порядком стоимости**. Разница существенна: живая блокировка
диспетчера возвращала корректный результат, проходила все проверки и
при этом роняла пропускную способность впятеро.

Пороги намеренно щедрые — в разы, а не в проценты. Цель не в том, чтобы
ловить микрорегрессии (на общей машине это дало бы мигающие тесты), а в
том, чтобы поймать возврат дефекта, меняющего порядок величины.

Каждый тест назван по дефекту, который он сторожит, и содержит цифры
исходного симптома.
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter

import pytest

from gateway.admission.queue import PriorityQueue, QueuedRequest, SelectiveDispatcher
from gateway.core.domain import ServiceClass
from gateway.router.prefix import PrefixTable, block_hashes
from tests.conftest import chat_body, collect

AGENT = "Bearer sk-agent-demo"


def q_req(sc=ServiceClass.AGENT, tenant="t") -> QueuedRequest:
    return QueuedRequest(service_class=sc, tenant_id=tenant,
                         enqueued_at=time.monotonic(), estimated_tokens=100)


# --------------------------------------------------------------------------
# Живая блокировка диспетчера
# Симптом: 14 021 378 отдач на 4 тысячи запросов, ожидание 30 с при
# свободных апстримах, пропускная способность с 36,5 до 7,7 RPS.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispetcher_ne_zavisaet_pri_prioritetnoi_konkurentsii():
    """Сторожит живую блокировку диспетчера.

    Условие воспроизведения оказалось неочевидным, и найти его удалось
    только возвратом исходного кода из истории. Нужны **одновременно**:

      • много ожидающих, уже стоящих в очереди;
      • расхождение между порядком очереди и порядком запуска корутин.

    Второе даёт приоритет: низкоприоритетные запросы встают первыми,
    поэтому их корутины запускаются раньше, но в очереди впереди них
    оказывается высокоприоритетный. Ожидающий, добравшийся до головы,
    видит там чужой запрос — и в сломанной версии забирал его себе,
    теряя навсегда.

    Без этого расхождения дефект не проявляется вовсе: каждая корутина
    находит в голове собственный запрос. Наивные версии этого теста —
    с одинаковым приоритетом и со свободным с самого начала апстримом —
    проходили при намеренно возвращённом дефекте. Проверено возвратом
    кода из коммита 0972190.

    Контрольный опыт: на этом сценарии сломанная версия **зависает**,
    исправленная обслуживает 40 из 40 за 540 оборотов.
    """
    q = PriorityQueue()
    disp = SelectiveDispatcher(q, max_wait_s=3.0, poll_interval_s=0.001)
    ready: list[str] = []

    low = [q_req(ServiceClass.BATCH, f"b{i}") for i in range(20)]
    high = [q_req(ServiceClass.INTERACTIVE, f"h{i}") for i in range(20)]

    tasks = [asyncio.create_task(disp.acquire_slot(r, lambda: ready)) for r in low]
    await asyncio.sleep(0.01)
    tasks += [asyncio.create_task(disp.acquire_slot(r, lambda: ready)) for r in high]
    await asyncio.sleep(0.01)

    n = len(low) + len(high)
    assert len(q) == n, f"в очередь встали {len(q)} запросов из {n}"

    ready.append("u1")
    try:
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=15.0)
    except asyncio.TimeoutError:
        for t in tasks:
            t.cancel()
        pytest.fail(
            "диспетчер завис при приоритетной конкуренции — вернулась живая "
            "блокировка (исходный симптом: 14 млн отдач на 4 тыс. запросов, "
            "ожидание 30 с при свободных апстримах)"
        )

    assert len(results) == n
    assert len(q) == 0, f"в очереди осталось {len(q)} запросов"
    assert disp.stats()["dispatched"] == n, (
        f"диспетчер отчитался о {disp.stats()['dispatched']} отдачах при {n} "
        "запросах — запросы теряются или обслуживаются дважды"
    )

    spins = disp.stats()["spins"]
    assert spins < n * 100, (
        f"{spins} холостых оборотов на {n} запросов — диспетчер крутится вхолостую"
    )


@pytest.mark.asyncio
async def test_ozhidanie_v_ocheredi_pri_svobodnyh_apstrimah_pochti_nulevoe():
    """Если апстримы свободны, запрос не должен ждать вовсе.
    Исходный симптом дефекта — 30 секунд ожидания при пустом кластере."""
    q = PriorityQueue()
    disp = SelectiveDispatcher(q, max_wait_s=5.0, poll_interval_s=0.001)
    ready = ["u1"]

    tasks = [asyncio.create_task(disp.acquire_slot(q_req(), lambda: ready))
             for _ in range(100)]
    await asyncio.gather(*tasks)

    avg_ms = disp.stats()["queue_wait_avg_ms"]
    assert avg_ms < 100, (
        f"среднее ожидание {avg_ms:.0f} мс при полностью свободных апстримах"
    )


@pytest.mark.asyncio
async def test_ochered_pustaya_posle_obsluzhivaniya():
    """Потерянный в очереди запрос — вторая половина того же дефекта:
    он дождётся своей очереди и займёт слот, который уже никому не нужен."""
    q = PriorityQueue()
    disp = SelectiveDispatcher(q, max_wait_s=3.0, poll_interval_s=0.001)
    ready = ["u1"]
    await asyncio.gather(*[disp.acquire_slot(q_req(), lambda: ready) for _ in range(50)])
    assert len(q) == 0, f"в очереди осталось {len(q)} запросов после обслуживания всех"


# --------------------------------------------------------------------------
# Лавина размыкателей
# Симптом: 3 успешных ответа из 2073 на интенсивности 300 запросов/с.
# --------------------------------------------------------------------------


def test_reestr_ne_ostavlyaet_bez_apstrimov_pri_obshchei_medlitelnosti():
    """Под общей перегрузкой медленными становятся все апстримы сразу.
    Размыкатель не должен превращать «медленно» в «недоступно»."""
    from gateway.resilience.breaker import BreakerConfig, BreakerRegistry

    reg = BreakerRegistry(BreakerConfig(min_samples=3, latency_multiplier=2.0,
                                        error_rate_threshold=0.5, open_duration_s=60))
    ids = [f"u{i}" for i in range(4)]
    now = time.monotonic()
    for uid in ids:
        for _ in range(5):
            reg.get(uid).record_success(latency_ms=5000, budget_ms=1000, now=now)

    assert reg.healthy(ids), (
        "все апстримы исключены за медлительность — вернулась лавина "
        "(исходный симптом: 3 успешных ответа из 2073)"
    )


def test_nepolnyi_otvet_ne_razmykaet_zhivoi_apstrim():
    """Оборванные ответы под нагрузкой — норма, а не поломка."""
    from gateway.resilience.breaker import BreakerConfig, BreakerState, CircuitBreaker

    b = CircuitBreaker("u", BreakerConfig(min_samples=100, consecutive_failures_to_open=3))
    now = time.monotonic()
    for _ in range(20):
        b.record_failure(hard=False, now=now)
    assert b.state is BreakerState.CLOSED


# --------------------------------------------------------------------------
# Утечка ёмкости при разрывах клиентом
# Симптом: 60 брошенных стримов продолжали бы генерировать 15 минут.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_massovye_razryvy_ne_ostavlyayut_inflight(gateway, fake_client):
    """После разрыва всех стримов ни один слот не должен остаться занятым."""
    fake_client.default.tokens = 1000

    gens = []
    for i in range(30):
        resp = await gateway.handle(chat_body(content=f"разрыв {i}"), AGENT)
        gen = resp.stream
        await gen.__anext__()
        gens.append(gen)

    for gen in gens:
        await gen.aclose()
    for _ in range(50):
        await asyncio.sleep(0)

    stuck = {u.upstream_id: u.inflight_requests
             for u in gateway.load.all_upstreams() if u.inflight_requests}
    assert not stuck, f"после 30 разрывов остались занятые слоты: {stuck}"

    assert len(fake_client.closed) == 30, (
        f"закрыто {len(fake_client.closed)} потоков из 30 — брошенные "
        "генерации продолжают жечь ёмкость апстрима (§3.3.9)"
    )


@pytest.mark.asyncio
async def test_razryvy_ne_portyat_reputatsiyu_apstrima(gateway, fake_client):
    """Клиент, закрывший вкладку, не должен исключать исправный апстрим."""
    fake_client.default.tokens = 1000

    for i in range(20):
        resp = await gateway.handle(chat_body(content=f"вкладка {i}"), AGENT)
        gen = resp.stream
        await gen.__anext__()
        await gen.aclose()
        for _ in range(5):
            await asyncio.sleep(0)

    opened = [b for b in gateway.breakers.snapshot() if b["state"] != "closed"]
    assert not opened, (
        f"разрывы клиентом разомкнули размыкатели: {opened}"
    )


# --------------------------------------------------------------------------
# Концентрация трафика
# Симптомы: 2586 запросов из 2589 на один апстрим (разрыв ничьей),
# 800 из 800 (ключ сессии), 0 из 400 у одного узла (голодание).
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trafik_ne_konsentriruetsya_na_prostaivayushchei_sisteme(gateway, fake_client):
    """Три разных дефекта давали один и тот же симптом. Сторожим симптом,
    а не конкретную причину: следующий дефект этого класса, скорее всего,
    будет новым."""
    for i in range(90):
        resp = await gateway.handle(chat_body(content=f"новая сессия {i}"), AGENT)
        await collect(resp)

    counts = Counter(fake_client.calls)
    n = len(gateway.registry.config.upstreams_for("main"))
    mean = sum(counts.values()) / n
    assert max(counts.values()) / mean < 2.0, (
        f"перекос {max(counts.values()) / mean:.2f}x на простаивающей системе: "
        f"{dict(counts)}"
    )
    assert len(counts) == n, f"часть апстримов не получила трафика: {dict(counts)}"


# --------------------------------------------------------------------------
# Рост состояния
# --------------------------------------------------------------------------


def test_tablitsa_prefiksov_ogranichena():
    """Таблица префиксов не должна расти неограниченно: сессия никогда
    не сообщает о своём завершении (§3.3.6.3b)."""
    t = PrefixTable(ttl_s=600.0, max_entries=5000)
    for i in range(3000):
        t.record(block_hashes(f"сессия {i} " * 400, 256, salt="t"), f"u{i % 4}")
    assert t.stats()["entries"] <= 5000, (
        f"таблица выросла до {t.stats()['entries']} записей при лимите 5000"
    )


def test_tablitsa_prefiksov_bystra_pri_bolshom_obeme():
    """Поиск по таблице лежит в горячем пути и не должен деградировать
    с её наполнением: роутинг обязан укладываться в доли миллисекунды и
    не расти с размером кластера (§3.3.6.7)."""
    t = PrefixTable(ttl_s=600.0, max_entries=200_000)
    for i in range(2000):
        t.record(block_hashes(f"сессия {i} " * 300, 256, salt="t"), f"u{i % 8}")

    probe = block_hashes("сессия 1000 " * 300, 256, salt="t")
    start = time.perf_counter()
    for _ in range(1000):
        t.best_upstream(probe)
    per_call_ms = (time.perf_counter() - start) * 1000 / 1000

    assert per_call_ms < 1.0, (
        f"поиск в таблице занимает {per_call_ms:.3f} мс — роутер сам стал "
        "узким местом"
    )


@pytest.mark.asyncio
async def test_ocheredi_ne_rastut_pri_normalnoi_nagruzke(gateway, fake_client):
    """Очередь на прокси — не место для накопления. Рост при нормальной
    нагрузке означает утечку: запросы не изымаются после обслуживания."""
    for i in range(60):
        resp = await gateway.handle(chat_body(content=f"нагрузка {i}"), AGENT)
        await collect(resp)

    assert len(gateway.queue) == 0, (
        f"в очереди осело {len(gateway.queue)} запросов: {gateway.queue.snapshot()}"
    )


@pytest.mark.asyncio
async def test_schetchik_konkurrentnosti_vozvrashchaetsya_k_nulyu(gateway, fake_client):
    """Незакрытый счётчик постепенно исчерпает лимит конкурентности
    тенанта, и запросы начнут отклоняться без видимой причины."""
    for i in range(40):
        resp = await gateway.handle(chat_body(content=f"счётчик {i}"), AGENT)
        await collect(resp)

    key = "tenant:ide-agent"
    assert gateway.quotas.concurrent(key) == 0, (
        f"счётчик одновременных запросов застрял на "
        f"{gateway.quotas.concurrent(key)}"
    )


# --------------------------------------------------------------------------
# Стоимость горячего пути
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stoimost_odnogo_zaprosa_v_goryachem_puti(gateway, fake_client):
    """Собственная работа гейтвея без сети. Порог щедрый: ловим не
    микрорегрессию, а появление случайного O(n) прохода или тяжёлой
    операции в цепочке.

    Замер на стенде даёт 1,2-4,1 мс с учётом сети (Б-5), так что
    собственная работа обязана быть заметно дешевле.
    """
    fake_client.default.tokens = 1

    # Прогрев: первый запрос строит структуры.
    for _ in range(5):
        await collect(await gateway.handle(chat_body(), AGENT))

    n = 100
    start = time.perf_counter()
    for i in range(n):
        await collect(await gateway.handle(chat_body(content=f"стоимость {i}"), AGENT))
    per_request_ms = (time.perf_counter() - start) * 1000 / n

    assert per_request_ms < 10.0, (
        f"{per_request_ms:.2f} мс собственной работы на запрос — бюджет "
        "накладных расходов 5-20 мс (§2.1.2) съеден целиком"
    )
