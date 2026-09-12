"""Тесты роутера (§3.3.6).

Проверяется то, что легко сломать незаметно: свойства хеширования,
изоляция тенантов, правила выбора кандидата и — отдельно — разрыв ничьей,
на котором реализация уже один раз сломалась.
"""

from __future__ import annotations

import pytest

from gateway.admission.load import LoadEstimator
from gateway.core.config import SLOConfig
from gateway.core.domain import ChatRequest, Message, Tenant, Upstream
from gateway.router.base import RoutingContext
from gateway.router.prefix import (
    PrefixTable,
    block_hashes,
    common_prefix_len,
    estimate_tokens,
)
from gateway.router.strategies import (
    ConsistentHashStrategy,
    DualMapStrategy,
    LeastLoadStrategy,
    SessionStrategy,
)

BLOCK = 256


def make_ctx(n_upstreams: int = 4):
    est = LoadEstimator()
    table = PrefixTable(ttl_s=600.0)
    ctx = RoutingContext(load_estimator=est, prefix_table=table,
                         slo=SLOConfig(), block_tokens=BLOCK)
    ups = [Upstream(id=f"u{i}", base_url=f"http://u{i}", model="m") for i in range(n_upstreams)]
    return ctx, ups


def make_request(turns: int = 0, *, tenant: str = "t1", body: str = "x") -> ChatRequest:
    msgs = [Message("system", "системный промпт " * 300)]
    for i in range(turns):
        msgs.append(Message("user", f"вопрос {i} " * 100))
        msgs.append(Message("assistant", f"ответ {i} " * 100))
    msgs.append(Message("user", f"{body} текущий вопрос " * 100))
    req = ChatRequest(model="main", messages=msgs, tenant=Tenant(id=tenant, name=tenant))
    req.prompt_tokens_est = estimate_tokens(req.prompt_text())
    return req


# --------------------------------------------------------------------------
# Хеширование префикса
# --------------------------------------------------------------------------


def test_prodolzhenie_sessii_sohranyaet_prefiks():
    """Следующий ход дописывает историю, поэтому обязан попасть в кэш
    ровно на длину общего префикса — на этом держится весь роутинг."""
    base = "общий контекст " * 500
    h1 = block_hashes(base, BLOCK, salt="t1")
    h2 = block_hashes(base + "продолжение " * 200, BLOCK, salt="t1")
    assert common_prefix_len(h1, h2) == len(h1)


def test_tsepochechnoe_hashirovanie_rvetsya_na_rashozhdenii():
    """Одинаковый блок в разных контекстах — это разные KV-состояния.
    Если бы блоки хешировались независимо, совпадение «перепрыгнуло» бы
    разрыв, и роутер обещал бы попадание там, где его нет."""
    base = "текст " * 1000
    h1 = block_hashes(base, BLOCK, salt="t1")
    changed = base[: BLOCK * 4] + "РАЗРЫВ" + base[BLOCK * 4 + 6 :]
    h2 = block_hashes(changed, BLOCK, salt="t1")
    assert common_prefix_len(h1, h2) == 1, "цепочка обязана порваться на первом же несовпадении"


def test_tenanty_izolirovany_v_klyuche_prefiksa():
    """Общий префикс-кэш между тенантами — боковой канал: чужой промпт
    угадывается по времени ответа (§3.3.6.10). Ключ обязан включать тенанта."""
    text = "одинаковый текст " * 500
    assert common_prefix_len(
        block_hashes(text, BLOCK, salt="tenant-a"),
        block_hashes(text, BLOCK, salt="tenant-b"),
    ) == 0


def test_nepolnyi_blok_ne_popadaet_v_klyuch():
    """Частично посчитанный блок не даёт переиспользуемого KV-состояния."""
    exactly_two = "a" * (BLOCK * 4 * 2)
    assert len(block_hashes(exactly_two, BLOCK)) == 2
    assert len(block_hashes(exactly_two + "хвост", BLOCK)) == 2


# --------------------------------------------------------------------------
# Таблица префиксов
# --------------------------------------------------------------------------


def test_tablitsa_ttl_zabyvaet_staroe():
    """TTL порядка минут покрывает ~90% переиспользований (§5.3);
    хранить дольше бессмысленно, а помнить вечно — вредно."""
    t = PrefixTable(ttl_s=10.0)
    h = block_hashes("текст " * 500, BLOCK)
    t.record(h, "u1", now=1000.0)
    assert t.hit_len(h, "u1", now=1005.0) == len(h)
    assert t.hit_len(h, "u1", now=1020.0) == 0, "протухшая запись обязана перестать считаться"


def test_tablitsa_ne_putaet_apstrimy():
    t = PrefixTable()
    h = block_hashes("текст " * 500, BLOCK)
    t.record(h, "u1")
    assert t.hit_len(h, "u1") == len(h)
    assert t.hit_len(h, "u2") == 0


# --------------------------------------------------------------------------
# Разрыв ничьей — дефект, найденный симуляцией
# --------------------------------------------------------------------------


def test_least_load_ne_konsentriruet_trafik_pri_ravnoi_nagruzke():
    """На простаивающей системе нагрузка всех апстримов равна нулю.
    Детерминированный разрыв ничьей по идентификатору отправлял бы
    весь трафик на первый по алфавиту — в симуляции это дало 2586
    запросов из 2589 на один апстрим из четырёх."""
    ctx, ups = make_ctx(4)
    s = LeastLoadStrategy()
    chosen = [s.select(make_request(body=f"r{i}"), ups, ctx).upstream.id for i in range(40)]
    assert len(set(chosen)) == 4, f"трафик сконцентрирован: {set(chosen)}"
    counts = [chosen.count(u.id) for u in ups]
    assert max(counts) - min(counts) <= 1, f"распределение неравномерно: {counts}"


def test_least_load_vybiraet_menee_zagruzhennyi():
    """Разрыв ничьей не должен ломать основную функцию стратегии."""
    ctx, ups = make_ctx(3)
    ctx.load.upstream("u0").pending_prefill_tokens = 50_000
    ctx.load.upstream("u1").pending_prefill_tokens = 100
    ctx.load.upstream("u2").pending_prefill_tokens = 80_000
    assert LeastLoadStrategy().select(make_request(), ups, ctx).upstream.id == "u1"


def test_least_load_obhodit_zanyatye():
    ctx, ups = make_ctx(3)
    for uid in ("u0", "u1"):
        u = ctx.load.upstream(uid)
        u.has_pending_queue = True
        u.state_fresh_at = 1e18  # свежий сигнал
    assert LeastLoadStrategy().select(make_request(), ups, ctx).upstream.id == "u2"


# --------------------------------------------------------------------------
# Сессионная стратегия
# --------------------------------------------------------------------------


def test_pervyi_hod_idet_na_balansirovku():
    """Размещение первого запроса определяет размещение всей сессии,
    поэтому балансировать достаточно только первые запросы (§3.3.6.3a)."""
    ctx, ups = make_ctx(4)
    # Все нагружены, один заметно меньше: иначе «нулевые» апстримы окажутся
    # ещё свободнее ожидаемого, и тест проверял бы не то, что заявлено.
    for u in ups:
        ctx.load.upstream(u.id).pending_prefill_tokens = 100_000
    ctx.load.upstream("u3").pending_prefill_tokens = 10
    req = make_request(turns=0)
    assert req.turn == 0
    d = SessionStrategy().select(req, ups, ctx)
    assert d.upstream.id == "u3"
    assert "первый ход" in d.reason


def test_posleduyushchie_hody_zalipayut():
    """96,6% последующих запросов возвращаются на тот же инстанс —
    ради этого эффекта стратегия и существует."""
    ctx, ups = make_ctx(4)
    s = SessionStrategy()

    first = make_request(turns=0)
    d1 = s.select(first, ups, ctx)
    s.on_dispatched(first, d1.upstream, ctx)

    second = make_request(turns=1)
    # Ровный фон и небольшая добавка залипшему: он загружен сильнее прочих,
    # но не кратно, поэтому предохранитель от перегрузки срабатывать не
    # должен, а affinity — должна перевесить.
    for u in ups:
        ctx.load.upstream(u.id).pending_prefill_tokens = 10_000
    ctx.load.upstream(d1.upstream.id).pending_prefill_tokens = 12_000

    d2 = s.select(second, ups, ctx)
    assert d2.upstream.id == d1.upstream.id, "сессия не залипла"
    assert "залипание" in d2.reason


def test_predohranitel_ot_peregruzki_migriruet_sessiyu():
    """Страховка от случая, когда две изначально короткие сессии на одном
    инстансе обе разрослись: откат на балансировку мигрирует длинную."""
    ctx, ups = make_ctx(4)
    s = SessionStrategy(overload_factor=2.0)

    first = make_request(turns=0)
    d1 = s.select(first, ups, ctx)
    s.on_dispatched(first, d1.upstream, ctx)

    # Залипший перегружен кратно относительно среднего.
    for u in ups:
        ctx.load.upstream(u.id).pending_prefill_tokens = 1000
    ctx.load.upstream(d1.upstream.id).pending_prefill_tokens = 500_000

    d2 = s.select(make_request(turns=1), ups, ctx)
    assert d2.upstream.id != d1.upstream.id
    assert "перегружен" in d2.reason


def test_vyrozhdennyi_sluchai_bez_istorii_bezopasen():
    """Если агент отбросил историю, запрос обрабатывается как первый в
    новой сессии — и это правильно, потому что в кэш он всё равно
    почти не попадёт (§3.3.6.3b)."""
    req = ChatRequest(model="main",
                      messages=[Message("user", "вопрос без истории " * 100)],
                      tenant=Tenant(id="t1", name="t1"))
    assert req.turn == 0
    ctx, ups = make_ctx(4)
    d = SessionStrategy().select(req, ups, ctx)
    assert "первый ход" in d.reason


# --------------------------------------------------------------------------
# Consistent hashing
# --------------------------------------------------------------------------


def test_koltso_daet_stabilnoe_otobrazhenie():
    """Одна сессия обязана попадать на один апстрим — в этом весь смысл."""
    ctx, ups = make_ctx(4)
    s = ConsistentHashStrategy()
    req = make_request(turns=2)
    chosen = {s.select(req, ups, ctx).upstream.id for _ in range(10)}
    assert len(chosen) == 1


def test_koltso_propuskaet_zanyatye():
    """Кольцо даёт affinity, пропуск занятых даёт устойчивость (§3.3.6.3f)."""
    ctx, ups = make_ctx(4)
    s = ConsistentHashStrategy()
    req = make_request(turns=2)
    natural = s.select(req, ups, ctx).upstream.id

    u = ctx.load.upstream(natural)
    u.has_pending_queue = True
    u.state_fresh_at = 1e18
    d = s.select(req, ups, ctx)
    assert d.upstream.id != natural
    assert "пропущены занятые" in d.reason


def test_koltso_pri_dobavlenii_uzla_pereotobrazhaet_malo():
    """Добавление инстанса не должно вызывать глобальный ремаппинг:
    иначе эластичность разрушает cache affinity (§3.3.6.5)."""
    ctx, ups4 = make_ctx(4)
    reqs = [make_request(turns=2, body=f"s{i}") for i in range(200)]

    s4 = ConsistentHashStrategy()
    before = [s4.select(r, ups4, ctx).upstream.id for r in reqs]

    ups5 = ups4 + [Upstream(id="u4", base_url="http://u4", model="m")]
    s5 = ConsistentHashStrategy()
    after = [s5.select(r, ups5, ctx).upstream.id for r in reqs]

    moved = sum(1 for a, b in zip(before, after) if a != b)
    # При идеальном consistent hashing переезжает ~1/5 ключей.
    assert moved < len(reqs) * 0.4, f"переехало {moved} из {len(reqs)} — кольцо работает плохо"


# --------------------------------------------------------------------------
# DualMap
# --------------------------------------------------------------------------


def test_dualmap_rassmatrivaet_rovno_dvuh_kandidatov():
    """Рост числа кандидатов уменьшает отклонение лишь логарифмически,
    зато разбрасывает одинаковые префиксы и разрушает локальность (§3.3.6.1)."""
    ctx, ups = make_ctx(8)
    d = DualMapStrategy().select(make_request(turns=2), ups, ctx)
    assert d.candidates_considered == 2


def test_dualmap_pri_ravnom_popadanii_beret_menee_zagruzhennyi():
    ctx, ups = make_ctx(4)
    for i, u in enumerate(ups):
        ctx.load.upstream(u.id).pending_prefill_tokens = (i + 1) * 1000
    d = DualMapStrategy().select(make_request(turns=2), ups, ctx)
    assert "менее загруженный" in d.reason


def warm_all(strategy, ups, ctx) -> None:
    """Отмечает все апстримы прогретыми.

    Нужно там, где проверяется правило выбора, а не прогревочная гарантия:
    холодный кандидат забирает запрос вне очереди и маскирует проверяемое
    поведение.
    """
    if hasattr(strategy, "_warmed"):
        strategy._warmed.update(u.id for u in ups)


def test_dualmap_derzhit_affinity_poka_ukladyvaetsya_v_slo():
    """Правило DualMap, а не Min TTFT: поштучная минимизация осциллирует
    между cache-aware и load-aware решениями (§3.3.6.3)."""
    ctx, ups = make_ctx(4)
    s = DualMapStrategy()
    req = make_request(turns=2)

    d1 = s.select(req, ups, ctx)
    s.on_dispatched(req, d1.upstream, ctx)
    warm_all(s, ups, ctx)

    # Кэширующий кандидат слегка загружен, но в пределах SLO.
    ctx.load.upstream(d1.upstream.id).pending_prefill_tokens = 500
    d2 = s.select(req, ups, ctx)
    assert d2.upstream.id == d1.upstream.id
    assert "affinity" in d2.reason


def test_dualmap_degradiruet_v_load_aware_pri_narushenii_slo():
    ctx, ups = make_ctx(4)
    s = DualMapStrategy()
    req = make_request(turns=2)
    d1 = s.select(req, ups, ctx)
    s.on_dispatched(req, d1.upstream, ctx)
    warm_all(s, ups, ctx)

    # Прогноз TTFT кэширующего кандидата уходит далеко за бюджет.
    ctx.load.upstream(d1.upstream.id).pending_prefill_tokens = 5_000_000
    d2 = s.select(req, ups, ctx)
    assert "вне SLO" in d2.reason


def test_dualmap_edinstvennyi_apstrim_ne_padaet():
    """Сценарий 3 из §3.3.6.8: апстрим один и непрозрачен — роутинг
    вырождается, но не ломается."""
    ctx, ups = make_ctx(1)
    d = DualMapStrategy().select(make_request(turns=2), ups, ctx)
    assert d.upstream.id == "u0"


# --------------------------------------------------------------------------
# Оценка TTFT
# --------------------------------------------------------------------------


def test_otsenka_ttft_superlineina_po_dline_vhoda():
    """Время префилла растёт суперлинейно: внимание квадратично.
    Линейная аппроксимация врала бы на длинных контекстах (§3.3.6.7a)."""
    ctx, _ = make_ctx(1)
    t1 = ctx.estimated_ttft_ms("u0", 32_000) - ctx.prefill_const_ms
    t2 = ctx.estimated_ttft_ms("u0", 64_000) - ctx.prefill_const_ms
    assert t2 > 2.0 * t1


def test_kalibrovka_pomechaet_model_kak_izmerennuyu():
    """До калибровочного прогона коэффициенты — заглушка, и это должно
    быть видно, а не подразумеваться."""
    ctx, _ = make_ctx(1)
    assert ctx.calibrated is False
    ctx.apply_calibration({"const_ms": 1.0, "per_token_ms": 0.01, "quadratic_ms": 0.5})
    assert ctx.calibrated is True
    assert ctx.prefill_const_ms == 1.0
