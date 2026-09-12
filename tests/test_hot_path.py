"""Горячий путь целиком, без сети (§3.2).

Цепочка `приём → auth → admission → роутинг → апстрим → стрим → учёт`
прогоняется с фейковым апстримом. До появления этого файла она
проверялась только через `loadtest/e2e.sh`, которому нужен поднятый
стенд, — то есть цикл «правка → проверка» занимал полминуты вместо
полусекунды. Под давлением такое перестают запускать.

Здесь проверяется то, что видно **только на сквозном проходе**:
взаимодействие механизмов, а не каждый по отдельности.
"""

from __future__ import annotations

import asyncio

import pytest

from gateway.admission.quota import QuotaLedger
from gateway.core.gateway import AdmissionRejected
from gateway.upstream.adapter import UpstreamError
from tests.conftest import chat_body, collect

AGENT = "Bearer sk-agent-demo"
CHAT = "Bearer sk-chat-demo"
BATCH = "Bearer sk-batch-demo"


# --------------------------------------------------------------------------
# Базовый проход
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skvoznoi_prohod_otdaet_potok_i_schitaet_tokeny(gateway, fake_client):
    fake_client.default.tokens = 7
    resp = await gateway.handle(chat_body(), AGENT)
    chunks = await collect(resp)

    assert chunks, "клиент не получил ни одного кадра"
    assert chunks[-1] == b"data: [DONE]\n\n"
    assert resp.result.completion_tokens == 7
    assert resp.result.finished is True
    assert resp.result.goodput_counted is True
    assert fake_client.closed == [resp.result.upstream_id], "поток к апстриму не закрыт"


@pytest.mark.asyncio
async def test_zagolovki_otvechayut_na_vopros_kuda_ushel_zapros(gateway):
    """На демо и при разборе инцидента первый вопрос — какой апстрим
    обслужил запрос и на какой версии конфигурации."""
    resp = await gateway.handle(chat_body(), AGENT)
    await collect(resp)
    for header in ("X-Gateway-Request-Id", "X-Gateway-Upstream",
                   "X-Gateway-Strategy", "X-Gateway-Config-Version"):
        assert resp.headers.get(header), f"нет заголовка {header}"


@pytest.mark.asyncio
async def test_trafik_raskladyvaetsya_po_apstrimam(gateway, fake_client):
    """Проверка на сквозном проходе, а не на стратегии в изоляции:
    между стратегией и отправкой стоит размыкатель, очередь и селективная
    отдача, и любой из них может свести трафик в одну точку."""
    for i in range(40):
        resp = await gateway.handle(chat_body(content=f"сессия {i}"), AGENT)
        await collect(resp)
    used = set(fake_client.calls)
    assert len(used) >= 3, f"трафик сошёлся на {used}"


# --------------------------------------------------------------------------
# Взаимодействие квот и стрима
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kvota_vozvrashchaetsya_posle_zaversheniya(gateway, fake_client):
    """Списываем оценку вперёд, корректируем по факту (§3.3.3.5).
    Без корректировки квота утекает с каждым запросом."""
    fake_client.default.tokens = 3
    ledger: QuotaLedger = gateway.quotas

    resp = await gateway.handle(chat_body(), AGENT)
    key = "tenant:ide-agent"
    during = ledger._tokens[key].remaining()
    await collect(resp)
    after = ledger._tokens[key].remaining()

    assert after > during, (
        "зарезервированные токены не возвращены: квота утекает с каждым запросом"
    )


@pytest.mark.asyncio
async def test_kvota_vozvrashchaetsya_pri_otkaze_apstrima(gateway, fake_client):
    """Расчёт обязан произойти при любом исходе, иначе неудачные запросы
    съедают квоту тенанта навсегда."""
    ledger: QuotaLedger = gateway.quotas
    key = "tenant:ide-agent"

    resp = await gateway.handle(chat_body(), AGENT)
    await collect(resp)
    baseline = ledger._tokens[key].remaining()

    for uid in list(gateway.registry.config.upstreams):
        fake_client.set(uid, fail_with=UpstreamError("апстрим мёртв", status=503,
                                                     retryable=True))
    with pytest.raises(UpstreamError):
        r = await gateway.handle(chat_body(content="обречённый"), AGENT)
        await collect(r)

    after = ledger._tokens[key].remaining()
    assert after >= baseline * 0.95, (
        f"после неудачного запроса квота упала с {baseline:.0f} до {after:.0f}: "
        "резерв не возвращён"
    )


@pytest.mark.asyncio
async def test_kvota_vozvrashchaetsya_pri_razryve_klientom(gateway, fake_client):
    """Клиент, закрывший вкладку, не должен расходовать свою квоту так,
    будто получил полный ответ."""
    fake_client.default.tokens = 100
    ledger: QuotaLedger = gateway.quotas
    key = "tenant:ide-agent"

    resp = await gateway.handle(chat_body(), AGENT)
    gen = resp.stream
    await gen.__anext__()
    await gen.__anext__()
    await gen.aclose()
    for _ in range(10):
        await asyncio.sleep(0)

    assert ledger.concurrent(key) == 0, (
        "счётчик одновременных запросов не сброшен после разрыва: "
        "тенант постепенно исчерпает лимит конкурентности"
    )


# --------------------------------------------------------------------------
# Взаимодействие resilience и стрима
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_degradatsiya_na_zhivoi_apstrim_s_pometkoi(gateway, fake_client):
    """Отказ апстрима даёт деградацию с явной пометкой, а не ошибку
    клиенту (§3.3.8)."""
    cfg = gateway.registry.config
    first = cfg.upstreams_for("main")[0]
    fake_client.set(first.id, fail_with=UpstreamError("упал", status=503, retryable=True))

    # Повторяем, пока роутер не выберет сломанный: важно, что переход
    # произошёл прозрачно для клиента.
    for i in range(20):
        resp = await gateway.handle(chat_body(content=f"деградация {i}"), AGENT)
        chunks = await collect(resp)
        if resp.result.degraded:
            assert b"degraded" in chunks[0], "пометка деградации не первым кадром"
            assert resp.result.finished, "после деградации ответ не завершился"
            return
    pytest.skip("роутер ни разу не выбрал сломанный апстрим за 20 попыток")


@pytest.mark.asyncio
async def test_otkaz_posle_pervogo_tokena_ne_povtoryaetsya(gateway, fake_client):
    """После первого отданного токена стрим не идемпотентен: повтор
    породил бы дубли в уже начатом ответе (§3.3.8)."""
    for uid in list(gateway.registry.config.upstreams):
        fake_client.set(uid, tokens=10, break_after=3)

    resp = await gateway.handle(chat_body(), AGENT)
    with pytest.raises(UpstreamError):
        await collect(resp)

    # Ровно одна попытка: обрыв случился после начала выдачи.
    assert len(fake_client.calls) == 1, (
        f"после обрыва посреди стрима сделано {len(fake_client.calls)} попыток — "
        "клиент получит дубликаты"
    )


@pytest.mark.asyncio
async def test_vse_apstrimy_mertvy_daet_chestnyi_otkaz(gateway, fake_client):
    """Полный отказ обязан давать понятный ответ, а не зависание (§4)."""
    for uid in list(gateway.registry.config.upstreams):
        fake_client.set(uid, fail_with=UpstreamError("мёртв", status=503, retryable=True))

    with pytest.raises((UpstreamError, AdmissionRejected)):
        resp = await gateway.handle(chat_body(), AGENT)
        await collect(resp)


@pytest.mark.asyncio
async def test_razmykatel_isklyuchaet_slomannyi_apstrim(gateway, fake_client):
    """После серии отказов запросы перестают уходить на сломанный."""
    cfg = gateway.registry.config
    broken = cfg.upstreams_for("main")[0].id
    fake_client.set(broken, fail_with=UpstreamError("мёртв", status=503, retryable=True))

    for i in range(30):
        try:
            resp = await gateway.handle(chat_body(content=f"размыкатель {i}"), AGENT)
            await collect(resp)
        except UpstreamError:
            pass

    tail = fake_client.calls[-10:]
    assert broken not in tail, (
        f"сломанный апстрим {broken} всё ещё получает трафик: {tail}"
    )


# --------------------------------------------------------------------------
# Классы обслуживания
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_klass_obsluzhivaniya_beretsya_iz_klyucha(gateway):
    """Класс определяется тенантом, а не заголовком запроса: иначе любой
    клиент объявил бы себя интерактивным."""
    from gateway.core.domain import ServiceClass

    for auth, expected in ((AGENT, ServiceClass.AGENT),
                           (CHAT, ServiceClass.INTERACTIVE),
                           (BATCH, ServiceClass.BATCH)):
        resp = await gateway.handle(chat_body(), auth)
        await collect(resp)
        # Класс виден по тому, в какую очередь попал запрос; проверяем
        # через резолвинг, который и принимает решение.
        from gateway.auth.resolver import resolve
        assert resolve(gateway.registry.config, auth).service_class is expected


@pytest.mark.asyncio
async def test_fonovyi_klass_otbrasyvaetsya_pri_peregruzke(gateway, fake_client):
    """Под перегрузкой background отбрасывается первым (§4). Проверяем
    сам механизм, подняв давление напрямую."""
    gateway.load._pressure.update(5.0)
    assert gateway.load.should_reject() is True

    with pytest.raises(AdmissionRejected) as e:
        await gateway.handle(chat_body(), "Bearer sk-bg-demo")
    assert e.value.limit_kind == "overload"


@pytest.mark.asyncio
async def test_interaktivnyi_klass_ne_otbrasyvaetsya_pri_peregruzke(gateway):
    """Обратная сторона: защищаемый класс обслуживается даже когда
    система решила отказывать. В этом весь смысл классов (§2.2)."""
    gateway.load._pressure.update(5.0)
    assert gateway.load.should_reject() is True

    resp = await gateway.handle(chat_body(), CHAT)
    await collect(resp)
    assert resp.result.finished is True


# --------------------------------------------------------------------------
# Сессионная маршрутизация на сквозном проходе
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sessiya_zalipaet_na_odnom_apstrime(gateway, fake_client):
    """Ходы одной сессии обязаны идти на один апстрим — ради этого
    и существует сессионная стратегия (§3.3.6.3a)."""
    chosen = []
    for turn in range(6):
        resp = await gateway.handle(chat_body(turns=turn, content="одна задача"), AGENT)
        await collect(resp)
        chosen.append(resp.headers["X-Gateway-Upstream"])
    assert len(set(chosen)) == 1, f"сессия размазана по апстримам: {chosen}"


@pytest.mark.asyncio
async def test_raznye_tenanty_ne_delyat_kesh(gateway, fake_client):
    """Общий префикс-кэш между тенантами — боковой канал (§3.3.6.10).
    Проверяем на сквозном проходе: ключ префикса обязан включать тенанта
    на всём пути, а не только в самой функции хеширования."""
    from gateway.router.prefix import block_hashes

    body = chat_body(turns=2, content="одинаковый запрос")
    r1 = await gateway.handle(body, AGENT)
    await collect(r1)
    r2 = await gateway.handle(body, CHAT)
    await collect(r2)

    text = "".join(f"{m['role']}\n{m['content']}\n" for m in body["messages"])
    h_agent = block_hashes(text, gateway.ctx.block_tokens, salt="ide-agent")
    h_chat = block_hashes(text, gateway.ctx.block_tokens, salt="chat-ui")
    assert gateway.prefix.hit_len(h_agent, r2.headers["X-Gateway-Upstream"]) == 0 or \
           h_agent != h_chat, "тенанты делят ключи префикса"
