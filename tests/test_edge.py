"""Тесты внешнего контракта, авторизации, размыкателя и стрима.

Сетевых вызовов нет: проверяется логика, а не HTTP. Сквозные проверки
живут в loadtest/e2e.sh, потому что требуют поднятых апстримов.
"""

from __future__ import annotations

import asyncio

import pytest

from gateway.api.schema import ValidationError, parse_chat_request
from gateway.auth.resolver import AuthError, authorize_model, resolve
from gateway.core.config import load_config
from gateway.core.domain import ServiceClass
from gateway.resilience.breaker import BreakerConfig, BreakerState, CircuitBreaker
from gateway.stream.pipeline import (
    RetryGuard,
    StreamPipeline,
    StreamResult,
    degradation_notice,
)
from gateway.upstream.adapter import OpenAIAdapter, StreamEvent, UpstreamResponse


# --------------------------------------------------------------------------
# Внешний контракт
# --------------------------------------------------------------------------


def test_razbor_minimalnogo_zaprosa():
    r = parse_chat_request({"model": "main", "messages": [{"role": "user", "content": "привет"}]})
    assert r.model == "main"
    assert r.messages[0].content == "привет"
    assert r.stream is False


def test_neizvestnye_polya_edut_v_passthrough():
    """Провайдер может понимать больше, чем мы. Терять это нельзя —
    иначе гейтвей ограничивает возможности апстрима (§3.3.1)."""
    r = parse_chat_request({
        "model": "main", "messages": [{"role": "user", "content": "x"}],
        "top_p": 0.9, "tools": [{"type": "function"}], "seed": 42,
    })
    assert r.passthrough == {"top_p": 0.9, "tools": [{"type": "function"}], "seed": 42}


def test_multimodalnyi_kontent_skleivaetsya_v_tekst():
    """Для хеширования префикса нужен текст; исходная структура
    сохраняется в passthrough и уезжает провайдеру нетронутой."""
    r = parse_chat_request({
        "model": "main",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "опиши "},
            {"type": "image_url", "image_url": {"url": "..."}},
            {"type": "text", "text": "картинку"},
        ]}],
    })
    assert r.messages[0].content == "опиши картинку"


@pytest.mark.parametrize("body,fragment", [
    ({}, "model"),
    ({"model": "main"}, "messages"),
    ({"model": "main", "messages": []}, "messages"),
    ({"model": "main", "messages": [{"content": "x"}]}, "role"),
    ({"model": "main", "messages": [{"role": "user", "content": "x"}], "max_tokens": -5},
     "max_tokens"),
])
def test_nevalidnye_zaprosy_otklonyayutsya(body, fragment):
    with pytest.raises(ValidationError, match=fragment):
        parse_chat_request(body)


# --------------------------------------------------------------------------
# Авторизация
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg():
    c, _ = load_config("config")
    return c


def test_klyuch_rezolvitsya_v_tenanta(cfg):
    t = resolve(cfg, "Bearer sk-agent-demo")
    assert t.id == "ide-agent"
    assert t.service_class is ServiceClass.AGENT


def test_klyuch_bez_prefiksa_bearer(cfg):
    assert resolve(cfg, "sk-chat-demo").id == "chat-ui"


@pytest.mark.parametrize("header", [None, "", "Bearer ", "Bearer sk-нет-такого"])
def test_nevernyi_klyuch_daet_401(cfg, header):
    with pytest.raises(AuthError) as e:
        resolve(cfg, header)
    assert e.value.status == 401


def test_model_ne_razreshena_tenantu_daet_403(cfg):
    """chat-ui имеет доступ только к main."""
    t = resolve(cfg, "Bearer sk-chat-demo")
    with pytest.raises(AuthError) as e:
        authorize_model(t, "reserve", cfg)
    assert e.value.status == 403


def test_neizvestnaya_model_daet_404(cfg):
    t = resolve(cfg, "Bearer sk-agent-demo")
    with pytest.raises(AuthError) as e:
        authorize_model(t, "нет-такой-модели", cfg)
    assert e.value.status == 404


# --------------------------------------------------------------------------
# Адаптер
# --------------------------------------------------------------------------


def test_adapter_vsegda_prosit_strim(cfg):
    """К апстриму идём стримом независимо от того, чего хочет клиент:
    так остаётся один горячий путь, а нестриминговый ответ собирается
    у нас. Иначе вторая ветка тихо гниёт."""
    from gateway.core.domain import ChatRequest, Message
    a = OpenAIAdapter()
    up = cfg.upstreams["mock-a"]
    payload = a.build_payload(
        ChatRequest(model="main", messages=[Message("user", "x")], stream=False), up
    )
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}


def test_adapter_podstavlyaet_imya_modeli_provaidera(cfg):
    """Логическое имя из реестра не должно утекать провайдеру."""
    from gateway.core.domain import ChatRequest, Message
    payload = OpenAIAdapter().build_payload(
        ChatRequest(model="main", messages=[Message("user", "x")]), cfg.upstreams["mock-a"]
    )
    assert payload["model"] == "mock-llm"


def test_adapter_razbiraet_kadry():
    a = OpenAIAdapter()
    ev = a.parse_frame('{"choices":[{"index":0,"delta":{"content":"привет"}}]}'.encode())
    assert ev.kind == "delta" and ev.content == "привет"

    ev = a.parse_frame(b'{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5}}')
    assert ev.kind == "usage" and ev.prompt_tokens == 10

    assert a.parse_frame(b"[DONE]").kind == "done"
    assert a.parse_frame("не json".encode()) is None


# --------------------------------------------------------------------------
# Размыкатель
# --------------------------------------------------------------------------


def test_seriya_otkazov_razmykaet_bystree_statistiki():
    """Подряд идущие отказы — сигнал сильнее доли в окне. Ждать двадцати
    замеров для явно мёртвого апстрима незачем: каждая попытка стоит
    таймаута соединения."""
    b = CircuitBreaker("u", BreakerConfig(min_samples=20, consecutive_failures_to_open=3))
    t = 1000.0
    for i in range(3):
        b.record_failure(now=t + i)
    assert b.state is BreakerState.OPEN


def test_uspekh_sbrasyvaet_seriyu():
    b = CircuitBreaker("u", BreakerConfig(min_samples=100, consecutive_failures_to_open=3))
    t = 1000.0
    b.record_failure(now=t)
    b.record_failure(now=t + 1)
    b.record_success(now=t + 2)
    b.record_failure(now=t + 3)
    b.record_failure(now=t + 4)
    assert b.state is BreakerState.CLOSED, "серия не была сброшена успехом"


def test_medlennyi_otvet_schitaetsya_otkazom():
    """Апстрим, отвечающий успешно, но в разы хуже бюджета, вреднее того,
    что честно отдаёт 503: он держит слоты и не даёт сигнала."""
    cfg = BreakerConfig(min_samples=5, latency_multiplier=5.0, consecutive_failures_to_open=3)
    b = CircuitBreaker("u", cfg)
    t = 1000.0
    for i in range(5):
        b.record_success(latency_ms=9000, budget_ms=1000, now=t + i)
    assert b.state is BreakerState.OPEN


def test_vosstanovlenie_cherez_poluotkrytoe():
    cfg = BreakerConfig(consecutive_failures_to_open=2, open_duration_s=5,
                        half_open_successes=2)
    b = CircuitBreaker("u", cfg)
    t = 1000.0
    b.record_failure(now=t)
    b.record_failure(now=t)
    assert b.allows(now=t + 1) is False
    assert b.allows(now=t + 6) is True
    assert b.state is BreakerState.HALF_OPEN
    b.record_success(now=t + 6)
    b.record_success(now=t + 7)
    assert b.state is BreakerState.CLOSED


def test_proval_probnogo_zaprosa_snova_razmykaet():
    """Без этого восстановление превращается в качели: закрылись,
    получили залп, снова размокли."""
    cfg = BreakerConfig(consecutive_failures_to_open=2, open_duration_s=5)
    b = CircuitBreaker("u", cfg)
    t = 1000.0
    b.record_failure(now=t)
    b.record_failure(now=t)
    b.allows(now=t + 6)
    assert b.state is BreakerState.HALF_OPEN
    b.record_failure(now=t + 6)
    assert b.state is BreakerState.OPEN


# --------------------------------------------------------------------------
# Ограничение ретраев
# --------------------------------------------------------------------------


def test_retrai_zapreshchen_posle_pervogo_tokena():
    """После первого токена стрим не идемпотентен: повтор породит дубли
    в уже начатом ответе (§3.3.8). Тонкость, которую большинство
    реализаций забывает."""
    g = RetryGuard(max_attempts=3)
    g.mark_attempt()
    assert g.may_retry() is True
    g.mark_first_token()
    assert g.may_retry() is False


def test_retrai_ogranichen_chislom_popytok():
    g = RetryGuard(max_attempts=2)
    g.mark_attempt()
    assert g.may_retry() is True
    g.mark_attempt()
    assert g.may_retry() is False


def test_pometka_degradatsii_yavnaya():
    """Клиент не должен молча получать ответ другой модели: для
    банковского контура подмена обязана быть видимой (§3.3.8)."""
    n = degradation_notice(original_model="main", actual_model="qwen", reason="апстрим упал")
    assert n["degraded"] is True
    assert n["requested_model"] == "main"
    assert n["served_model"] == "qwen"


# --------------------------------------------------------------------------
# Stream pipeline
# --------------------------------------------------------------------------


def _fake_response(events: list[StreamEvent], closed: list[bool]):
    from gateway.core.domain import Upstream

    async def gen():
        for e in events:
            yield e

    async def close():
        closed.append(True)

    return UpstreamResponse(
        status=200, headers={}, events=gen(), close=close,
        upstream=Upstream(id="u1", base_url="http://u1", model="m"),
    )


@pytest.mark.asyncio
async def test_pipeline_schitaet_tokeny_i_zakryvaet_potok():
    closed: list[bool] = []
    events = [
        StreamEvent(kind="delta", content="a", raw=b"1"),
        StreamEvent(kind="delta", content="b", raw=b"2"),
        StreamEvent(kind="usage", raw=b"3", prompt_tokens=7, completion_tokens=2),
        StreamEvent(kind="done", raw=b"4"),
    ]
    result = StreamResult()
    out = [c async for c in StreamPipeline().relay(
        _fake_response(events, closed), result, started_at=0.0
    )]
    assert out == [b"1", b"2", b"3", b"4"]
    assert result.prompt_tokens == 7
    assert result.completion_tokens == 2
    assert result.finished is True
    assert result.goodput_counted is True
    assert closed == [True], "поток к апстриму не закрыт"


@pytest.mark.asyncio
async def test_otmena_klientom_zakryvaet_potok_k_apstrimu():
    """Без этого брошенные генерации продолжают жечь ёмкость —
    утечка, которая проявляется только под нагрузкой (§3.3.9)."""
    closed: list[bool] = []
    events = [StreamEvent(kind="delta", content=str(i), raw=b"x") for i in range(100)]
    result = StreamResult()
    gen = StreamPipeline().relay(_fake_response(events, closed), result, started_at=0.0)

    await gen.__anext__()
    await gen.__anext__()
    await gen.aclose()          # клиент отвалился

    assert closed == [True], "поток к апстриму остался открытым после разрыва"
    assert result.client_disconnected is True
    assert result.goodput_counted is False, "незавершённый запрос попал в goodput"


@pytest.mark.asyncio
async def test_nezavershennyi_zapros_ne_schitaetsya_goodput():
    """В goodput засчитываются только полностью завершённые запросы:
    если запрос отвалился на полпути, ресурсы потрачены впустую (§2.1.2)."""
    closed: list[bool] = []
    result = StreamResult()
    events = [StreamEvent(kind="delta", content="a", raw=b"1")]
    async for _ in StreamPipeline().relay(
        _fake_response(events, closed), result, started_at=0.0
    ):
        pass
    assert result.finished is False
    assert result.goodput_counted is False


@pytest.mark.asyncio
async def test_pometka_degradatsii_idet_pervym_kadrom():
    closed: list[bool] = []
    events = [StreamEvent(kind="delta", content="a", raw="полезное".encode())]
    result = StreamResult()
    notice = degradation_notice(original_model="main", actual_model="r", reason="упал")
    out = [c async for c in StreamPipeline().relay(
        _fake_response(events, closed), result, started_at=0.0, degraded_notice=notice
    )]
    assert b"degraded" in out[0]
    assert out[1] == "полезное".encode()
    assert result.degraded is True
