"""Контрактные тесты: проверки для кода, которого ещё нет.

Зачем этот файл. До старта неизвестно, к какой платформе мы подключаемся
(§1.3 п. 2) и какие сигналы будут доступны от апстрима (§3.3.6.8). Значит,
на хакатоне почти наверняка придётся написать **новый адаптер** и, может
быть, **новую стратегию маршрутизации** — под давлением и быстро.

Тесты здесь проверяют не конкретную реализацию, а **контракт, которому
обязана удовлетворять любая**. Они автоматически распространяются на всё,
что появится в реестрах `ADAPTERS` и `STRATEGIES`: добавил класс — он сразу
под проверкой, ничего дописывать не нужно.

Это и есть ответ на вопрос «какие тесты можно написать заранее»: те, что
формулируют требования к будущему коду, а не описывают поведение
нынешнего.
"""

from __future__ import annotations

import inspect

import pytest

from gateway.core.config import RouterConfig, SLOConfig, load_config
from gateway.core.domain import ChatRequest, Message, Tenant, Upstream
from gateway.router.base import RoutingDecision, RoutingStrategy
from gateway.router.prefix import estimate_tokens
from gateway.router.strategies import STRATEGIES, build_strategy
from gateway.upstream.adapter import ADAPTERS, Adapter, StreamEvent
from tests.test_router import make_ctx


def make_request(turns: int = 2, *, tenant: str = "t1") -> ChatRequest:
    msgs = [Message("system", "системный промпт " * 200)]
    for i in range(turns):
        msgs.append(Message("user", f"вопрос {i} " * 60))
        msgs.append(Message("assistant", f"ответ {i} " * 60))
    msgs.append(Message("user", "текущий вопрос " * 60))
    r = ChatRequest(model="main", messages=msgs, tenant=Tenant(id=tenant, name=tenant))
    r.prompt_tokens_est = estimate_tokens(r.prompt_text())
    return r


# ==========================================================================
# Контракт адаптера провайдера
#
# Если платформа окажется с нестандартным API, на хакатоне придётся
# написать новый адаптер. Эти тесты применятся к нему автоматически.
# ==========================================================================


ADAPTER_IDS = sorted(ADAPTERS)


@pytest.fixture(scope="module")
def cfg():
    c, _ = load_config("config")
    return c


@pytest.mark.parametrize("name", ADAPTER_IDS)
def test_adapter_realizuet_polnyi_interfeis(name):
    """Адаптер обязан реализовать все три метода контракта.

    Частичная реализация — самая дорогая ошибка спешки: код собирается,
    падает в рантайме под нагрузкой.
    """
    adapter = ADAPTERS[name]()
    assert isinstance(adapter, Adapter)
    for method in ("build_payload", "parse_frame", "headers", "endpoint"):
        impl = getattr(type(adapter), method, None)
        assert impl is not None, f"адаптер {name} не реализует {method}"
        assert not getattr(impl, "__isabstractmethod__", False), (
            f"адаптер {name}: метод {method} остался абстрактным"
        )
    assert adapter.name == name, (
        f"адаптер зарегистрирован как {name!r}, но сообщает имя {adapter.name!r} — "
        "конфигурация будет ссылаться на несуществующее"
    )


@pytest.mark.parametrize("name", ADAPTER_IDS)
def test_adapter_vsegda_prosit_strim(name, cfg):
    """К апстриму всегда идём стримом независимо от желания клиента.

    Иначе появляется вторая ветка кода, которая почти не используется и
    потому тихо гниёт — так и вышло при первой сборке: нестриминговый
    путь возвращал пустой ответ.
    """
    adapter = ADAPTERS[name]()
    up = cfg.upstreams["mock-a"]
    for client_wants_stream in (True, False):
        payload = adapter.build_payload(
            ChatRequest(model="main", messages=[Message("user", "x")],
                        stream=client_wants_stream),
            up,
        )
        assert payload.get("stream") is True, (
            f"адаптер {name} не запросил стрим при stream={client_wants_stream}"
        )


@pytest.mark.parametrize("name", ADAPTER_IDS)
def test_adapter_ne_puskaet_logicheskoe_imya_modeli_naruzhu(name, cfg):
    """Логическое имя модели из нашего реестра — наша внутренняя сущность.
    Провайдер о ней не знает и знать не должен."""
    adapter = ADAPTERS[name]()
    up = cfg.upstreams["mock-a"]
    payload = adapter.build_payload(
        ChatRequest(model="main", messages=[Message("user", "x")]), up
    )
    assert payload.get("model") == up.model, (
        f"адаптер {name} отправил провайдеру логическое имя вместо {up.model!r}"
    )


@pytest.mark.parametrize("name", ADAPTER_IDS)
def test_adapter_donosit_neizvestnye_polya(name, cfg):
    """Провайдер может понимать больше, чем мы. Терять это нельзя —
    иначе гейтвей ограничивает возможности апстрима (§3.3.1)."""
    adapter = ADAPTERS[name]()
    req = ChatRequest(
        model="main", messages=[Message("user", "x")],
        passthrough={"top_p": 0.9, "seed": 42, "неизвестное_поле": "значение"},
    )
    payload = adapter.build_payload(req, cfg.upstreams["mock-a"])
    for k, v in req.passthrough.items():
        assert payload.get(k) == v, f"адаптер {name} потерял поле {k!r}"


@pytest.mark.parametrize("name", ADAPTER_IDS)
def test_adapter_ne_padaet_na_musore(name):
    """Апстрим может прислать что угодно, включая обрывок кадра.
    Падение разбора в горячем пути роняет весь стрим."""
    adapter = ADAPTERS[name]()
    junk_frames = [b"", "не json".encode(), b"{", b"null", b"[]",
                   b'{"choices":null}', b'{"choices":[{}]}', b"\xff\xfe",
                   b'{"choices":[]}']
    for junk in junk_frames:
        try:
            ev = adapter.parse_frame(junk)
        except Exception as exc:
            pytest.fail(f"адаптер {name} упал на кадре {junk!r}: {exc!r}")
        assert ev is None or isinstance(ev, StreamEvent)


@pytest.mark.parametrize("name", ADAPTER_IDS)
def test_adapter_opoznaet_konets_potoka(name):
    """Без опознания конца стрим не закроется, и запрос не попадёт
    в goodput, даже если завершился успешно (§2.1.2)."""
    adapter = ADAPTERS[name]()
    ev = adapter.parse_frame(b"[DONE]")
    assert ev is not None and ev.kind == "done", (
        f"адаптер {name} не опознал конец потока"
    )


@pytest.mark.parametrize("name", ADAPTER_IDS)
def test_adapter_otdaet_raw_dlya_klienta(name):
    """Кадры отдаются клиенту как есть: разбор нужен нам для учёта,
    а не для переписывания ответа."""
    adapter = ADAPTERS[name]()
    ev = adapter.parse_frame(b'{"choices":[{"index":0,"delta":{"content":"x"}}]}')
    assert ev is not None and ev.raw, (
        f"адаптер {name} не сохранил исходный кадр — клиент получит пустоту"
    )


@pytest.mark.parametrize("name", ADAPTER_IDS)
def test_adapter_ne_techet_klyuchom_v_logi(name, cfg):
    """Реальные ключи провайдеров не покидают гейтвей (§3.3.2).
    Ключ в строковом представлении объекта — прямой путь в лог."""
    adapter = ADAPTERS[name]()
    secret = "СЕКРЕТНЫЙ-КЛЮЧ"
    up = Upstream(id="u", base_url="http://u", model="m", api_key=secret)
    headers = adapter.headers(up)
    assert any(secret in v for v in headers.values()), (
        f"адаптер {name} не передал ключ провайдеру"
    )
    assert secret not in repr(adapter), "ключ осел в состоянии адаптера"


# ==========================================================================
# Контракт стратегии маршрутизации
#
# Если окажется, что доступны метаданные KV-кэша или, наоборот, апстрим
# один и непрозрачен, стратегию придётся менять. Эти проверки применятся
# к любой новой автоматически.
# ==========================================================================


STRATEGY_IDS = sorted(STRATEGIES)


@pytest.mark.parametrize("name", STRATEGY_IDS)
def test_strategiya_stroitsya_iz_konfiga(name):
    """Стратегия выбирается строкой в конфиге и обязана строиться из него
    без дополнительных аргументов — иначе смена без передеплоя невозможна."""
    s = build_strategy(RouterConfig(strategy=name))
    assert isinstance(s, RoutingStrategy)
    assert s.name == name


@pytest.mark.parametrize("name", STRATEGY_IDS)
def test_strategiya_vsegda_vozvrashchaet_kandidata(name):
    """Стратегия не имеет права вернуть None или выбрать апстрим,
    которого ей не давали: и то и другое уронит запрос уже в отправке."""
    ctx, ups = make_ctx(4)
    s = build_strategy(RouterConfig(strategy=name))
    allowed = {u.id for u in ups}
    for turns in (0, 1, 5):
        d = s.select(make_request(turns), ups, ctx)
        assert isinstance(d, RoutingDecision)
        assert d.upstream is not None, f"стратегия {name} не выбрала апстрим"
        assert d.upstream.id in allowed, (
            f"стратегия {name} выбрала {d.upstream.id!r}, которого не было в кандидатах"
        )
        assert d.reason, f"стратегия {name} не объяснила выбор — отладка станет гаданием"


@pytest.mark.parametrize("name", STRATEGY_IDS)
def test_strategiya_vyzhivaet_pri_odnom_apstrime(name):
    """Сценарий 3 из §3.3.6.8: апстрим один и непрозрачен. Роутинг
    вырождается, но не ломается."""
    ctx, ups = make_ctx(1)
    s = build_strategy(RouterConfig(strategy=name))
    d = s.select(make_request(), ups, ctx)
    assert d.upstream.id == ups[0].id


@pytest.mark.parametrize("name", STRATEGY_IDS)
def test_strategiya_vyzhivaet_na_pustom_zaprose(name):
    """Вырожденный вход: агент отбросил историю, промпт короче блока.
    Обязано обрабатываться как первый запрос новой сессии, а не падать."""
    ctx, ups = make_ctx(3)
    s = build_strategy(RouterConfig(strategy=name))
    req = ChatRequest(model="main", messages=[Message("user", "?")],
                      tenant=Tenant(id="t", name="t"))
    req.prompt_tokens_est = 1
    d = s.select(req, ups, ctx)
    assert d.upstream is not None


@pytest.mark.parametrize("name", STRATEGY_IDS)
def test_strategiya_ne_hranit_rastushchee_sostoyanie(name):
    """Состояние маршрутизации обязано быть **мягким и ограниченным**.

    Таблица «сессия → апстрим» растёт с числом сессий, требует сборки
    мусора и никогда не узнаёт, что сессия закончилась — сессия об этом
    не сообщает (§3.3.6.3b). Стратегия, накапливающая такое, под нагрузкой
    съест память.

    Проверяем: после тысячи разных сессий собственное состояние стратегии
    не выросло линейно.
    """
    ctx, ups = make_ctx(4)
    s = build_strategy(RouterConfig(strategy=name))

    def own_state_size() -> int:
        return sum(len(v) for v in vars(s).values()
                   if isinstance(v, (dict, set, list)))

    def run(n: int, offset: int) -> None:
        for i in range(n):
            req = make_request(1, tenant=f"tenant-{offset + i}")
            d = s.select(req, ups, ctx)
            s.on_dispatched(req, d.upstream, ctx)

    # Сравниваем ПРИРОСТ, а не абсолютный размер: у стратегии может быть
    # законное состояние, зависящее от числа апстримов (например, хеш-кольцо
    # из виртуальных узлов). Оно ограничено размером кластера и расти с
    # нагрузкой не должно.
    run(100, 0)
    after_100 = own_state_size()
    run(900, 100)
    after_1000 = own_state_size()

    assert after_1000 - after_100 < 100, (
        f"стратегия {name}: состояние выросло на {after_1000 - after_100} "
        "записей за 900 дополнительных сессий — оно растёт с числом сессий, "
        "а сессия никогда не сообщает о своём завершении (§3.3.6.3b)"
    )


@pytest.mark.parametrize("name", STRATEGY_IDS)
def test_strategiya_bystra(name):
    """Роутинг обязан укладываться в доли миллисекунды и не расти с
    размером кластера (§3.3.6.7): решение принимается по локальным
    метаданным кандидатов, а не опросом всего пула.

    Порог намеренно щедрый — ловим не микрооптимизацию, а появление
    случайного O(n) прохода по кластеру или тяжёлого хеширования.
    """
    import time

    ctx, ups = make_ctx(32)
    s = build_strategy(RouterConfig(strategy=name))
    req = make_request(3)

    # Прогрев: первый вызов может строить кольцо.
    s.select(req, ups, ctx)

    start = time.perf_counter()
    n = 200
    for _ in range(n):
        s.select(req, ups, ctx)
    per_call_ms = (time.perf_counter() - start) / n * 1000

    assert per_call_ms < 5.0, (
        f"стратегия {name}: {per_call_ms:.2f} мс на решение при 32 апстримах — "
        "роутер сам стал узким местом"
    )


@pytest.mark.parametrize("name", STRATEGY_IDS)
def test_strategiya_ne_trebuet_nedostupnyh_signalov(name):
    """Основной рабочий сценарий — непрозрачный апстрим (§3.3.6.8).

    Стратегия обязана работать, когда состояние апстрима недоступно:
    нет pending_prefill_tokens, нет признака очереди. Падение здесь
    означает, что стратегия неприменима к нашему сценарию.
    """
    ctx, ups = make_ctx(4)
    for u in ups:
        load = ctx.load.upstream(u.id)
        load.pending_prefill_tokens = None
        load.has_pending_queue = None
    s = build_strategy(RouterConfig(strategy=name))
    d = s.select(make_request(2), ups, ctx)
    assert d.upstream is not None


def test_reestr_strategii_sootvetstvuet_validatoru():
    """Валидатор конфигурации знает список допустимых стратегий отдельно.
    Разъезд означает: либо новая стратегия отвергается как неизвестная,
    либо конфиг проходит валидацию и падает при сборке."""
    from gateway.core.config import validate, GatewayConfig

    for name in STRATEGY_IDS:
        cfg = GatewayConfig()
        cfg.router = RouterConfig(strategy=name)
        issues = [i for i in validate(cfg) if "неизвестная стратегия" in i]
        assert not issues, (
            f"стратегия {name} зарегистрирована, но валидатор её не знает: {issues}"
        )
