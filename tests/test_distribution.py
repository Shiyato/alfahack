"""Тесты распределения трафика (§3.3.6).

Отдельный файл, потому что здесь проверяется свойство, которое не видно
ни в одном тесте вида «функция вернула апстрим»: **куда расходится поток
запросов**. Оба дефекта, найденных на этом проекте, были именно такими —
код возвращал валидный результат, но весь трафик собирался в одной точке.

Общее у обоих: на простаивающей или слабо нагруженной системе перекос
проявляется полностью, а под нагрузкой маскируется, потому что нагрузка
сама разводит запросы. Поэтому все проверки здесь идут **на нулевой
нагрузке** — там, где дефект виден.
"""

from __future__ import annotations

from collections import Counter

import pytest

from gateway.core.domain import ChatRequest, Message, Tenant
from gateway.router.prefix import estimate_tokens
from gateway.router.strategies import (
    ConsistentHashStrategy,
    DualMapStrategy,
    LeastLoadStrategy,
    SessionStrategy,
)
from tests.test_router import make_ctx

ALL_STRATEGIES = [LeastLoadStrategy, ConsistentHashStrategy, SessionStrategy, DualMapStrategy]


def session_turns(task: str, n_turns: int, *, tenant: str = "t1") -> list[ChatRequest]:
    """Настоящая многоходовая сессия.

    Важная деталь: первое сообщение пользователя неизменно на всех ходах,
    история наращивается сверху. Именно так выглядит агентский трафик
    (§5.3), и именно на этом держится устойчивость ключа сессии.
    """
    msgs = [
        Message("system", "общий системный промпт " * 300),
        Message("user", f"{task} " * 200),
    ]
    out = []
    for i in range(n_turns):
        if i:
            msgs = msgs + [
                Message("assistant", f"ответ {i} " * 80),
                Message("user", f"шаг {i} " * 80),
            ]
        r = ChatRequest(model="main", messages=list(msgs),
                        tenant=Tenant(id=tenant, name=tenant))
        r.prompt_tokens_est = estimate_tokens(r.prompt_text())
        out.append(r)
    return out


# --------------------------------------------------------------------------
# Ключ сессии — основание для consistent hashing
# --------------------------------------------------------------------------


def test_klyuch_sessii_ustoichiv_po_hodam():
    """Все запросы одной сессии обязаны давать один ключ.

    Первая версия брала хеш истории без последнего хода, и ключ менялся
    от хода к ходу: залипания не возникало вовсе.
    """
    keys = {r.session_key("t1") for r in session_turns("задача", 10)}
    assert len(keys) == 1, f"ключ сессии изменился по ходам: {len(keys)} различных"


def test_klyuch_sessii_razlichaet_sessii_na_pervom_hode():
    """Запросы разных сессий обязаны давать разные ключи — включая
    первый ход, когда истории ещё нет.

    Первая версия на первом ходе хешировала только системный промпт,
    общий у всех. В замере это дало 800 запросов из 800 на один узел
    из восьми: каждая новая сессия уходила туда же.
    """
    keys = {session_turns(f"задача {i}", 1)[0].session_key("t1") for i in range(50)}
    assert len(keys) == 50, f"из 50 сессий получилось {len(keys)} различных ключей"


def test_klyuch_sessii_izoliruet_tenantov():
    """Общий ключ между тенантами открывает боковой канал (§3.3.6.10)."""
    a = session_turns("одна и та же задача", 1, tenant="tenant-a")[0]
    b = session_turns("одна и та же задача", 1, tenant="tenant-b")[0]
    assert a.session_key("tenant-a") != b.session_key("tenant-b")


# --------------------------------------------------------------------------
# Распределение новых сессий
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL_STRATEGIES, ids=lambda c: c.name)
def test_novye_sessii_ne_sobirayutsya_na_odnom_apstrime(cls):
    """Главная проверка этого файла.

    На простаивающей системе нагрузка всех апстримов равна нулю, и любая
    детерминированная привязка — разрыв ничьей по идентификатору или
    одинаковый ключ у разных сессий — собирает весь трафик в одну точку.
    Под нагрузкой это маскируется, поэтому мерить надо именно здесь.
    """
    ctx, ups = make_ctx(8)
    s = cls()
    counts: Counter[str] = Counter()
    n = 400
    for i in range(n):
        req = session_turns(f"задача {i}", 1)[0]
        d = s.select(req, ups, ctx)
        s.on_dispatched(req, d.upstream, ctx)
        counts[d.upstream.id] += 1

    per_upstream = [counts.get(f"u{i}", 0) for i in range(8)]
    mean = n / 8
    max_over_mean = max(per_upstream) / mean
    assert max_over_mean <= 2.0, (
        f"перекос {max_over_mean:.2f}x при распределении новых сессий: {per_upstream}"
    )
    assert min(per_upstream) > 0, f"есть простаивающие апстримы: {per_upstream}"


@pytest.mark.parametrize("cls", ALL_STRATEGIES, ids=lambda c: c.name)
def test_ravnomernost_ne_lomaetsya_pri_odnom_apstrime(cls):
    """Вырожденный случай: апстрим один (сценарий 3 из §3.3.6.8)."""
    ctx, ups = make_ctx(1)
    s = cls()
    for i in range(10):
        req = session_turns(f"задача {i}", 1)[0]
        assert s.select(req, ups, ctx).upstream.id == "u0"


# --------------------------------------------------------------------------
# Залипание сессии
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [ConsistentHashStrategy, SessionStrategy],
                         ids=lambda c: c.name)
def test_sessiya_ne_razmazyvaetsya_po_apstrimam(cls):
    """Сессия обязана целиком жить на одном апстриме, пока тот не
    перегружен: ради этого cache-affinity и существует."""
    ctx, ups = make_ctx(8)
    s = cls()
    chosen = []
    for req in session_turns("длинная задача", 12):
        d = s.select(req, ups, ctx)
        s.on_dispatched(req, d.upstream, ctx)
        chosen.append(d.upstream.id)
    assert len(set(chosen)) == 1, f"сессия размазана по {len(set(chosen))} апстримам: {chosen}"


@pytest.mark.parametrize("cls", [ConsistentHashStrategy, SessionStrategy],
                         ids=lambda c: c.name)
def test_raznye_sessii_zhivut_na_raznyh_apstrimah(cls):
    """Залипание не должно вырождаться в «все на одном»: разные сессии
    обязаны расходиться."""
    ctx, ups = make_ctx(8)
    s = cls()
    placement: dict[int, set[str]] = {}
    for k in range(24):
        picks = set()
        for req in session_turns(f"задача {k}", 4):
            d = s.select(req, ups, ctx)
            s.on_dispatched(req, d.upstream, ctx)
            picks.add(d.upstream.id)
        placement[k] = picks

    assert all(len(p) == 1 for p in placement.values()), "какая-то сессия размазалась"
    used = {next(iter(p)) for p in placement.values()}
    assert len(used) >= 4, f"24 сессии уместились всего на {len(used)} апстримах: {used}"


def test_holodnyi_apstrim_poluchaet_trafik():
    """Апстрим с пустым кэшем обязан получить шанс.

    Без прогревочной гарантии возникает замкнутый круг: попадание у
    холодного узла нулевое → он проигрывает любое сравнение по кэшу →
    кэш у него не появляется. В замере это дало один узел из восьми с
    нулём запросов из 400.

    Особенно важно для эластичности (§3.3.6.5): инстанс, добавленный при
    масштабировании, приходит холодным. Без гарантии масштабирование не
    работает ровно тогда, когда оно нужно.
    """
    ctx, ups = make_ctx(4)
    s = DualMapStrategy()

    # Прогреваем три узла из четырёх.
    for i in range(30):
        req = session_turns(f"разогрев {i}", 1)[0]
        d = s.select(req, ups[:3], ctx)
        s.on_dispatched(req, d.upstream, ctx)

    cold = ups[3].id
    assert cold not in s._warmed

    got_cold = False
    for i in range(60):
        req = session_turns(f"после расширения {i}", 1)[0]
        d = s.select(req, ups, ctx)
        s.on_dispatched(req, d.upstream, ctx)
        if d.upstream.id == cold:
            got_cold = True
            break
    assert got_cold, "холодный апстрим не получил ни одного запроса за 60 попыток"
