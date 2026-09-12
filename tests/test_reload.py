"""Hot-reload конфигурации (§3.1, §3.1.1).

До этого файла модуль `registry.py` не был покрыт ни одним тестом, хотя
это **демо в прямом эфире**: правка файла меняет поведение без передеплоя.
Если оно сломается на показе, объяснять будет нечего.

Отдельная тема — правки, которые применять **нельзя**. Гейтвей, упавший
из-за опечатки в YAML во время демонстрации, — худший исход из возможных,
поэтому сломанная конфигурация обязана отвергаться с сохранением
последней рабочей версии.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

from gateway.core.config import expand_env, load_config, validate
from gateway.core.registry import ConfigRegistry


@pytest.fixture
def cfg_dir(tmp_path):
    d = tmp_path / "config"
    d.mkdir()
    for f in Path("config").glob("*.yaml"):
        shutil.copy(f, d / f.name)
    return d


@pytest.fixture
async def registry(cfg_dir):
    r = ConfigRegistry(cfg_dir, poll_interval_s=0.02)
    await r.start()
    yield r
    await r.stop()


async def wait_version(registry, above: int, timeout: float = 2.0) -> bool:
    """Ждёт применения новой версии. По условию, а не фиксированной паузой:
    иначе тест либо флакает, либо тормозит."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if registry.config.version > above:
            return True
        await asyncio.sleep(0.01)
    return False


# --------------------------------------------------------------------------
# Что должно применяться
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_smena_strategii_podhvatyvaetsya(registry, cfg_dir):
    """Ровно то, что показывается на защите: стратегия роутинга меняется
    строкой конфига без передеплоя."""
    assert registry.config.router.strategy == "session"
    v = registry.config.version

    text = (cfg_dir / "models.yaml").read_text()
    (cfg_dir / "models.yaml").write_text(
        text.replace("strategy: session", "strategy: least_load")
    )
    assert await wait_version(registry, v), "конфигурация не перечитана"
    assert registry.config.router.strategy == "least_load"


@pytest.mark.asyncio
async def test_dobavlenie_modeli_na_hodu(registry, cfg_dir):
    v = registry.config.version
    text = (cfg_dir / "models.yaml").read_text()
    (cfg_dir / "models.yaml").write_text(
        text.replace("models:\n  main:", "models:\n  express: [mock-a]\n  main:")
    )
    assert await wait_version(registry, v)
    assert "express" in registry.config.model_aliases
    assert registry.config.upstreams_for("express")


@pytest.mark.asyncio
async def test_izmenenie_kvoty_na_hodu(registry, cfg_dir):
    v = registry.config.version
    text = (cfg_dir / "ratelimit.yaml").read_text()
    (cfg_dir / "ratelimit.yaml").write_text(
        text.replace("tokens_per_minute: 105000000", "tokens_per_minute: 1000")
    )
    assert await wait_version(registry, v)
    rule = next(r for r in registry.config.rate_limits if r.id == "agent-main-generous")
    assert rule.tokens_per_minute == 1000


@pytest.mark.asyncio
async def test_podpischiki_uvedomlyayutsya(registry, cfg_dir):
    """Компоненты с состоянием обязаны узнавать о смене: роутеру, например,
    нужно расширить хеш-кольцо при добавлении апстрима, а не строить его
    заново — иначе ремаппинг разрушит cache affinity (§3.3.6.5)."""
    seen: list[tuple[int, int]] = []
    registry.subscribe(lambda old, new: seen.append((old.version, new.version)))

    v = registry.config.version
    text = (cfg_dir / "models.yaml").read_text()
    (cfg_dir / "models.yaml").write_text(text.replace("block_tokens: 256", "block_tokens: 512"))
    assert await wait_version(registry, v)
    assert seen, "подписчик не уведомлён о смене конфигурации"


# --------------------------------------------------------------------------
# Что применяться НЕ должно
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bityi_yaml_ne_primenyaetsya(registry, cfg_dir):
    """Гейтвей, упавший из-за опечатки во время показа, — худший исход."""
    good = registry.config
    v = good.version
    (cfg_dir / "models.yaml").write_text("upstreams: [ это: не: yaml\n")
    await asyncio.sleep(0.3)

    assert registry.config.version == v, "битая конфигурация применена"
    assert registry.config.upstreams, "реестр апстримов потерян"
    assert registry.stats()["failed_reloads"] >= 1
    assert registry.stats()["last_error"], "ошибка не сохранена для диагностики"


@pytest.mark.asyncio
async def test_posle_pochinki_konfig_podhvatyvaetsya(registry, cfg_dir):
    """Отвергнув битую версию, реестр не должен «залипнуть»: починка
    обязана примениться."""
    original = (cfg_dir / "models.yaml").read_text()
    v = registry.config.version

    (cfg_dir / "models.yaml").write_text("не: [ yaml\n")
    await asyncio.sleep(0.2)
    assert registry.config.version == v

    (cfg_dir / "models.yaml").write_text(original)
    assert await wait_version(registry, v), "после починки конфигурация не применилась"


@pytest.mark.asyncio
async def test_konfig_bez_gisterezisa_otvergaetsya(registry, cfg_dir):
    """Контур без гистерезиса дребезжит на границе порога (§3.3.3.3).
    Это не предупреждение, а причина не применять конфигурацию."""
    v = registry.config.version
    text = (cfg_dir / "models.yaml").read_text()
    (cfg_dir / "models.yaml").write_text(
        text.replace("resume_threshold: 0.8", "resume_threshold: 1.5")
    )
    await asyncio.sleep(0.3)
    assert registry.config.version == v, "конфигурация без гистерезиса применена"


@pytest.mark.asyncio
async def test_neizvestnaya_strategiya_otvergaetsya(registry, cfg_dir):
    """Иначе гейтвей применит конфиг и упадёт при первой же сборке
    стратегии — то есть на первом запросе после правки."""
    v = registry.config.version
    text = (cfg_dir / "models.yaml").read_text()
    (cfg_dir / "models.yaml").write_text(
        text.replace("strategy: session", "strategy: волшебная")
    )
    await asyncio.sleep(0.3)
    assert registry.config.version == v
    assert registry.config.router.strategy == "session"


@pytest.mark.asyncio
async def test_udalenie_fayla_ne_ronyaet_reestr(registry, cfg_dir):
    """Файл может исчезнуть на мгновение при перезаписи редактором или
    выкладке. Это не повод терять конфигурацию."""
    v = registry.config.version
    upstreams_before = len(registry.config.upstreams)
    (cfg_dir / "ratelimit.yaml").unlink()
    await asyncio.sleep(0.2)
    assert len(registry.config.upstreams) == upstreams_before


# --------------------------------------------------------------------------
# Подстановка окружения
# --------------------------------------------------------------------------


def test_podstanovka_peremennyh_okruzheniya(monkeypatch):
    """Один файл работает и локально, и в контейнере, где апстримы
    доступны под именами сервисов. Две копии конфига разъехались бы."""
    monkeypatch.setenv("TEST_UPSTREAM", "http://mock-a:9000")
    assert expand_env("url: ${TEST_UPSTREAM}") == "url: http://mock-a:9000"


def test_znachenie_po_umolchaniyu_obyazatelno(monkeypatch):
    """Конфиг обязан запускаться без единой переменной окружения, иначе
    локальная разработка превращается в обряд."""
    monkeypatch.delenv("NET_TAKOI", raising=False)
    assert expand_env("url: ${NET_TAKOI:-http://127.0.0.1:9001}") == \
        "url: http://127.0.0.1:9001"


def test_konfig_repozitoriya_rabotaet_bez_okruzheniya(monkeypatch):
    """Проверка рабочих файлов, а не выдуманных: после `git clone`
    и `uv sync` всё обязано подняться без настройки."""
    for var in ("MOCK_A_URL", "MOCK_B_URL", "MOCK_C_URL", "MOCK_SLOW_URL"):
        monkeypatch.delenv(var, raising=False)
    cfg, issues = load_config("config")
    assert not issues, f"конфигурация репозитория не проходит валидацию: {issues}"
    assert cfg.upstreams_for("main"), "модель main осталась без апстримов"
    for u in cfg.upstreams.values():
        assert u.base_url.startswith("http"), f"адрес не подставлен: {u.base_url}"
        assert "${" not in u.base_url


def test_konfig_repozitoriya_rabotaet_v_konteinere(monkeypatch):
    """Те же файлы с окружением compose."""
    monkeypatch.setenv("MOCK_A_URL", "http://mock-a:9000")
    cfg, issues = load_config("config")
    assert cfg.upstreams["mock-a"].base_url == "http://mock-a:9000"
    assert cfg.upstreams["mock-a"].state_url == "http://mock-a:9000/state"
    assert not issues
