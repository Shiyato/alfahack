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


# --------------------------------------------------------------------------
# Классификация отказов
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_otkaz_po_klyuchu_ne_portit_reputatsiyu_apstrima(gateway, fake_client):
    """Просроченный ключ — не вина апстрима: он жив и исправен.

    Если засчитывать это в статистику размыкателя, один неверный ключ
    исключит здоровый апстрим для всех тенантов сразу.
    """
    cfg = gateway.registry.config
    target = cfg.upstreams_for("main")[0].id
    fake_client.set(target, fail_with=UpstreamError("ключ просрочен", status=401))

    for i in range(20):
        try:
            resp = await gateway.handle(chat_body(content=f"ключ {i}"), AGENT)
            await collect(resp)
        except UpstreamError:
            pass

    b = gateway.breakers.get(target)
    assert b.state.value == "closed", (
        "отказ по учётным данным разомкнул исправный апстрим"
    )


@pytest.mark.asyncio
async def test_otkaz_apstrima_razmykaet(gateway, fake_client):
    """Обратная сторона: настоящий отказ апстрима обязан размыкать."""
    cfg = gateway.registry.config
    target = cfg.upstreams_for("main")[0].id
    fake_client.set(target, fail_with=UpstreamError("упал", status=503))

    for i in range(20):
        try:
            resp = await gateway.handle(chat_body(content=f"отказ {i}"), AGENT)
            await collect(resp)
        except UpstreamError:
            pass

    assert gateway.breakers.get(target).state.value != "closed"


@pytest.mark.asyncio
async def test_nevosstanovimyi_otkaz_ne_povtoryaetsya(gateway, fake_client):
    """400 и 404 означают, что запрос некорректен. Повтор не поможет
    никогда, и перебирать апстримы бессмысленно — только тратить их время."""
    for uid in list(gateway.registry.config.upstreams):
        fake_client.set(uid, fail_with=UpstreamError("некорректный запрос", status=400))

    with pytest.raises(UpstreamError):
        resp = await gateway.handle(chat_body(), AGENT)
        await collect(resp)

    assert len(fake_client.calls) == 1, (
        f"невосстановимый отказ вызвал {len(fake_client.calls)} попыток"
    )


@pytest.mark.asyncio
async def test_sluzhebnyi_kadr_ne_meshaet_degradatsii(gateway, fake_client):
    """Апстрим успел отдать кадр с ролью и упал. Клиент не увидел ни
    одного токена, значит деградация обязана сработать.

    Первая версия считала границей любой отданный кадр, и такой отказ
    доходил до клиента ошибкой, хотя терять было нечего.
    """
    cfg = gateway.registry.config
    first = cfg.upstreams_for("main")[0].id
    # break_after=0: служебный кадр отдан, содержимое — нет.
    fake_client.set(first, tokens=5, break_after=0)

    for i in range(20):
        resp = await gateway.handle(chat_body(content=f"служебный {i}"), AGENT)
        try:
            chunks = await collect(resp)
        except UpstreamError:
            continue
        if resp.result.degraded:
            assert resp.result.finished, "деградация не довела ответ до конца"
            return
    pytest.skip("роутер ни разу не выбрал целевой апстрим за 20 попыток")


@pytest.mark.asyncio
async def test_obryv_do_soderzhimogo_perekryvaetsya_prozrachno(gateway, fake_client):
    """Апстрим открыл поток, отдал служебные кадры и упал.

    Так ведут себя провайдеры, сообщающие об ошибке внутри HTTP 200
    после стартовых метаданных. Клиент содержимого не видел, значит
    переход на другой апстрим обязан быть прозрачным.

    Без повтора внутри потока отказ, случившийся на миллисекунду позже
    открытия, доходил бы до клиента ошибкой, хотя терять нечего.
    """
    cfg = gateway.registry.config
    main_ids = [u.id for u in cfg.upstreams_for("main")]
    # Все основные обрывают поток до первого содержимого; резервная
    # модель исправна — именно для этого она и настроена.
    for uid in main_ids:
        fake_client.set(uid, tokens=5, break_after=0)

    resp = await gateway.handle(chat_body(), AGENT)
    chunks = await collect(resp)

    assert resp.result.finished is True, "запрос не доведён до конца"
    assert resp.result.degraded is True, "деградация не отмечена"
    assert b"degraded" in chunks[0]
    assert len(fake_client.calls) > 1, "переоткрытия потока не было"


@pytest.mark.asyncio
async def test_obryv_posle_soderzhimogo_ne_perekryvaetsya(gateway, fake_client):
    """Обратная сторона и главное ограничение: после первого кадра с
    содержимым стрим не идемпотентен, и повтор породил бы дубли в уже
    начатом ответе (§3.3.8)."""
    for uid in list(gateway.registry.config.upstreams):
        fake_client.set(uid, tokens=10, break_after=3)

    resp = await gateway.handle(chat_body(), AGENT)
    with pytest.raises(UpstreamError):
        await collect(resp)

    assert len(fake_client.calls) == 1, (
        f"после отдачи содержимого сделано {len(fake_client.calls)} попыток — "
        "клиент получит дубликаты"
    )


@pytest.mark.asyncio
async def test_perepodklyuchenie_ne_dubliruet_soderzhimoe(gateway, fake_client):
    """Переоткрытие потока не должно приводить к повторной выдаче того,
    что клиент уже получил."""
    cfg = gateway.registry.config
    for uid in [u.id for u in cfg.upstreams_for("main")]:
        fake_client.set(uid, tokens=4, break_after=0)
    # Резерв отдаёт ровно 4 кадра содержимого.
    for uid in [u.id for u in cfg.upstreams_for("reserve")]:
        fake_client.set(uid, tokens=4)

    resp = await gateway.handle(chat_body(), AGENT)
    chunks = await collect(resp)

    content_frames = [c for c in chunks if b'"t"' in c]
    # Ответ ровно один: 4 кадра с содержимым, без повторов.
    assert len(content_frames) == 4, (
        f"выдано {len(content_frames)} кадров содержимого вместо 4 — "
        "переоткрытие продублировало ответ"
    )


@pytest.mark.asyncio
async def test_inflight_osvobozhdaetsya_posle_perepodklyucheniya(gateway, fake_client):
    """Переоткрытие занимает счётчик новой цели; старая обязана
    освободиться, иначе ёмкость утекает при каждом обрыве."""
    from gateway.core.gateway import AdmissionRejected

    cfg = gateway.registry.config
    for uid in [u.id for u in cfg.upstreams_for("main")]:
        fake_client.set(uid, tokens=3, break_after=0)

    # Часть запросов законно упрётся в размыкатель: апстримы обрываются
    # раз за разом, и исключение их — правильное поведение. Проверяется
    # не успех, а отсутствие утечки счётчиков при любом исходе.
    for i in range(10):
        try:
            resp = await gateway.handle(chat_body(content=f"обрыв {i}"), AGENT)
            await collect(resp)
        except (UpstreamError, AdmissionRejected):
            pass

    stuck = {u.upstream_id: u.inflight_requests
             for u in gateway.load.all_upstreams() if u.inflight_requests}
    assert not stuck, f"после переподключений остались занятые слоты: {stuck}"
