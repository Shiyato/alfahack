"""Несколько экземпляров прокси (§3.3.8.1).

Проблема, о которой документ говорит прямо: как только появляется
состояние маршрутизации — таблица префиксов, счётчики очередей, кольцо, —
несколько экземпляров начинают конфликтовать. Без координации они
независимо выберут один и тот же апстрим как лучший и создадут горячую
точку.

При этом полная синхронизация требует порядка `O(N_прокси × N_апстримов)`
соединений и не масштабируется, поэтому в MVP её нет. Значит, нужно
проверить, что её отсутствие **не ломает корректность** — а именно это и
есть заявление на защите: «состояние маршрутизации мягкое, при потере
экземпляра деградирует эффективность, но не корректность».

Заявление проверяемое. Здесь оно проверяется.
"""

from __future__ import annotations

import asyncio
from collections import Counter

import pytest

from gateway.core.gateway import Gateway
from gateway.core.registry import ConfigRegistry
from tests.conftest import FakeUpstreamClient, chat_body, collect

AGENT = "Bearer sk-agent-demo"


def make_instance(cfg_dir) -> tuple[Gateway, FakeUpstreamClient]:
    """Отдельный экземпляр со своим состоянием, как в реальности."""
    registry = ConfigRegistry(cfg_dir, poll_interval_s=0.05)
    registry.load_now()
    gw = Gateway(registry)
    client = FakeUpstreamClient()
    gw.client = client
    return gw, client


@pytest.fixture
def cfg_dir(tmp_path):
    import shutil
    from pathlib import Path

    d = tmp_path / "config"
    d.mkdir()
    for f in Path("config").glob("*.yaml"):
        shutil.copy(f, d / f.name)
    return d


# --------------------------------------------------------------------------
# Горячая точка
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dva_ekzemplyara_ne_sozdayut_goryachuyu_tochku(cfg_dir):
    """Главная проверка файла.

    Два экземпляра стартуют с пустым состоянием и видят одинаковую
    картину. Если стратегия детерминирована по состоянию, оба выберут
    один и тот же апстрим — и трафик, распределённый балансировщиком
    поровну, снова сойдётся в одну точку.
    """
    gw1, c1 = make_instance(cfg_dir)
    gw2, c2 = make_instance(cfg_dir)

    # Балансировщик раскладывает запросы поровну, экземпляры не общаются.
    for i in range(120):
        gw = gw1 if i % 2 == 0 else gw2
        resp = await gw.handle(chat_body(content=f"сессия {i}"), AGENT)
        await collect(resp)

    combined = Counter(c1.calls) + Counter(c2.calls)
    total = sum(combined.values())
    n_upstreams = len(gw1.registry.config.upstreams_for("main"))
    max_over_mean = max(combined.values()) / (total / n_upstreams)

    assert max_over_mean < 2.0, (
        f"два экземпляра создали горячую точку, перекос {max_over_mean:.2f}x: "
        f"{dict(combined)}"
    )
    assert len(combined) == n_upstreams, (
        f"часть апстримов простаивает: {dict(combined)}"
    )


@pytest.mark.asyncio
async def test_odna_sessiya_cherez_raznye_ekzemplyary(cfg_dir):
    """Реалистичный случай: балансировщик не знает о сессиях и разложит
    её ходы по разным экземплярам.

    Идеального залипания здесь быть не может — у экземпляров разные
    таблицы префиксов. Проверяем, что это не приводит к разбросу по
    всем апстримам: при совпадающем состоянии решение должно совпадать.
    """
    gw1, c1 = make_instance(cfg_dir)
    gw2, c2 = make_instance(cfg_dir)

    chosen = []
    for turn in range(8):
        gw = gw1 if turn % 2 == 0 else gw2
        resp = await gw.handle(chat_body(turns=turn, content="одна задача"), AGENT)
        await collect(resp)
        chosen.append(resp.headers["X-Gateway-Upstream"])

    n_upstreams = len(gw1.registry.config.upstreams_for("main"))
    assert len(set(chosen)) <= max(2, n_upstreams - 1), (
        f"сессия размазана по всем апстримам: {chosen}"
    )


# --------------------------------------------------------------------------
# Мягкость состояния
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_poterya_ekzemplyara_ne_lomaet_korrektnost(cfg_dir):
    """Ровно то, что заявляется на защите: при потере экземпляра
    деградирует эффективность, но не корректность."""
    gw1, c1 = make_instance(cfg_dir)
    gw2, c2 = make_instance(cfg_dir)

    # Первый экземпляр обслужил сессии и накопил таблицу префиксов.
    for turn in range(4):
        resp = await gw1.handle(chat_body(turns=turn, content="долгая задача"), AGENT)
        await collect(resp)
    assert gw1.prefix.stats()["entries"] > 0

    # Экземпляр «умер». Продолжаем на втором, у которого состояния нет.
    for turn in range(4, 8):
        resp = await gw2.handle(chat_body(turns=turn, content="долгая задача"), AGENT)
        chunks = await collect(resp)
        assert resp.result.finished is True, "запрос не обслужен после потери экземпляра"
        assert chunks[-1] == b"data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_sostoyanie_vosstanavlivaetsya_nablyudeniem(cfg_dir):
    """Состояние мягкое — значит, новый экземпляр набирает его сам,
    без переноса и без координации."""
    gw, _ = make_instance(cfg_dir)
    assert gw.prefix.stats()["entries"] == 0

    for i in range(20):
        resp = await gw.handle(chat_body(turns=2, content=f"задача {i}"), AGENT)
        await collect(resp)

    assert gw.prefix.stats()["entries"] > 0, (
        "новый экземпляр не набрал таблицу префиксов наблюдением"
    )


@pytest.mark.asyncio
async def test_kvoty_ne_delyatsya_mezhdu_ekzemplyarami(cfg_dir):
    """Честная граница MVP, которую надо назвать вслух на защите.

    Квоты считаются локально, поэтому N экземпляров дают тенанту
    фактически N-кратный лимит. Это осознанное упрощение: общий счётчик
    требует внешнего хранилища в горячем пути, то есть новой точки
    отказа и сетевого вызова на каждый запрос (§3.1).

    Тест фиксирует поведение, а не одобряет его: если завтра появится
    общий счётчик, тест упадёт и напомнит обновить заявление.
    """
    gw1, _ = make_instance(cfg_dir)
    gw2, _ = make_instance(cfg_dir)

    key = "tenant:ide-agent"
    r1 = await gw1.handle(chat_body(), AGENT)
    await collect(r1)
    r2 = await gw2.handle(chat_body(), AGENT)
    await collect(r2)

    assert key in gw1.quotas._tokens and key in gw2.quotas._tokens
    assert gw1.quotas._tokens[key] is not gw2.quotas._tokens[key], (
        "вёдра квот оказались общими — поведение изменилось, обновите "
        "формулировку про границы MVP"
    )


# --------------------------------------------------------------------------
# Согласованность конфигурации
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_oba_ekzemplyara_podhvatyvayut_pravku(cfg_dir):
    """Экземпляры читают один каталог конфигурации. Расхождение версий
    означало бы, что половина трафика идёт по старым правилам, а
    диагностировать это по метрикам почти невозможно."""
    gw1, _ = make_instance(cfg_dir)
    gw2, _ = make_instance(cfg_dir)
    await gw1.registry.start()
    await gw2.registry.start()
    try:
        v = gw1.registry.config.version
        text = (cfg_dir / "models.yaml").read_text()
        (cfg_dir / "models.yaml").write_text(
            text.replace("strategy: session", "strategy: least_load")
        )

        deadline = asyncio.get_running_loop().time() + 2.0
        while asyncio.get_running_loop().time() < deadline:
            if (gw1.registry.config.router.strategy == "least_load"
                    and gw2.registry.config.router.strategy == "least_load"):
                break
            await asyncio.sleep(0.02)

        assert gw1.strategy.name == gw2.strategy.name == "least_load", (
            f"экземпляры разошлись: {gw1.strategy.name} против {gw2.strategy.name}"
        )
    finally:
        await gw1.registry.stop()
        await gw2.registry.stop()


@pytest.mark.asyncio
async def test_ekzemplyary_nezavisimy_pri_otkaze_apstrima(cfg_dir):
    """Размыкатели локальны. Это означает, что каждый экземпляр обнаружит
    отказ самостоятельно — медленнее, чем при общем состоянии, но без
    координации и без общей точки отказа."""
    gw1, c1 = make_instance(cfg_dir)
    gw2, c2 = make_instance(cfg_dir)

    from gateway.upstream.adapter import UpstreamError

    broken = gw1.registry.config.upstreams_for("main")[0].id
    for c in (c1, c2):
        c.set(broken, fail_with=UpstreamError("мёртв", status=503, retryable=True))

    for i in range(30):
        for gw in (gw1, gw2):
            try:
                resp = await gw.handle(chat_body(content=f"отказ {i}"), AGENT)
                await collect(resp)
            except UpstreamError:
                pass

    for name, gw in (("первый", gw1), ("второй", gw2)):
        state = gw.breakers.get(broken).state.value
        assert state != "closed", (
            f"{name} экземпляр не обнаружил отказ апстрима самостоятельно"
        )
