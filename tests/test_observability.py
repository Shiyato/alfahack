"""Наблюдаемость: метрики и их контракт с дашбордом (§3.3.10).

Зачем отдельный файл. Метрика и дашборд связаны **строкой имени**, и связь
эта ничем не проверяется: переименовал метрику — панель молча опустела.
Узнаёшь об этом на показе, когда график пустой, а объяснить нечем.

Здесь дашборд разбирается как данные и каждый его запрос сверяется с тем,
что гейтвей действительно отдаёт. Тест дешёвый и ловит целый класс
поломок, который не виден ни в одном другом.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from gateway.telemetry import metrics as m
from tests.conftest import chat_body, collect

DASHBOARD = Path("deploy/grafana/dashboards/gateway.json")
AGENT = "Bearer sk-agent-demo"


def exported_metric_names() -> set[str]:
    """Имена, которые гейтвей реально отдаёт в /metrics."""
    names: set[str] = set()
    for line in m.render().decode().splitlines():
        if line.startswith("# TYPE "):
            names.add(line.split()[2])
    return names


def declared_metric_names() -> set[str]:
    """Имена, объявленные в модуле метрик.

    Prometheus добавляет суффиксы `_total`, `_bucket`, `_sum`, `_count`,
    поэтому сверяем по базовым именам.
    """
    names: set[str] = set()
    for value in vars(m).values():
        name = getattr(value, "_name", None)
        if isinstance(name, str) and name.startswith("gateway_"):
            names.add(name)
    return names


def dashboard_queries() -> list[tuple[str, str]]:
    """(заголовок панели, запрос) для всех панелей дашборда."""
    dash = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    out = []
    for panel in dash["panels"]:
        if panel.get("type") == "row":
            continue
        for target in panel.get("targets", []):
            out.append((panel["title"], target["expr"]))
    return out


METRIC_RE = re.compile(r"\b((?:gateway|mock)_[a-z0-9_]+)\b")


# --------------------------------------------------------------------------
# Контракт «метрика ↔ дашборд»
# --------------------------------------------------------------------------


def test_dashboard_ssylaetsya_tolko_na_sushchestvuyushchie_metriki():
    """Главный тест файла.

    Панель, ссылающаяся на несуществующую метрику, не даёт ошибки — она
    просто пустая. Поэтому проверяется статически: каждое имя из запроса
    обязано существовать либо у гейтвея, либо у мок-апстрима.
    """
    gateway_names = declared_metric_names()
    # Суффиксы, которые Prometheus добавляет сам.
    expanded = set()
    for n in gateway_names:
        expanded.update({n, f"{n}_total", f"{n}_bucket", f"{n}_sum", f"{n}_count"})
    # Метрики мок-апстрима объявлены в mock-upstream/main.go.
    mock_names = {
        "mock_running_requests", "mock_queued_requests", "mock_pending_prefill_tokens",
        "mock_requests_total", "mock_rejected_total", "mock_canceled_total",
        "mock_tokens_in_total", "mock_tokens_out_total", "mock_cache_blocks",
        "mock_cache_block_hits_total", "mock_cache_block_misses_total",
    }
    known = expanded | mock_names

    missing: list[tuple[str, str]] = []
    for title, expr in dashboard_queries():
        for name in METRIC_RE.findall(expr):
            if name not in known:
                missing.append((title, name))

    assert not missing, (
        "панели дашборда ссылаются на несуществующие метрики "
        f"(они будут молча пустыми): {missing}"
    )


def test_mock_metriki_sootvetstvuyut_realnosti():
    """Список метрик мока в предыдущем тесте — копия, которая может
    разъехаться с самим моком. Сверяем с исходником."""
    source = Path("mock-upstream/main.go").read_text(encoding="utf-8")
    declared = set(re.findall(r'm\("(mock_[a-z_]+)"', source))
    used_in_dashboard = {
        name for _, expr in dashboard_queries()
        for name in METRIC_RE.findall(expr) if name.startswith("mock_")
    }
    assert used_in_dashboard <= declared, (
        f"дашборд ссылается на метрики мока, которых нет: "
        f"{used_in_dashboard - declared}"
    )


def test_kazhdaya_metrika_imeet_opisanie():
    """Метрика без описания через полгода означает «какое-то число».
    Prometheus отдаёт описание в HELP, и это единственное место, где
    объясняется, что величина значит."""
    text = m.render().decode()
    helps = {line.split()[2]: " ".join(line.split()[3:])
             for line in text.splitlines() if line.startswith("# HELP ")}
    empty = [n for n, h in helps.items() if not h.strip()]
    assert not empty, f"метрики без описания: {empty}"


def test_granitsy_gistogramm_pokryvayut_diapazon_llm():
    """Стандартные границы Prometheus заканчиваются на 10 секундах —
    для LLM это бесполезно: длинный префилл занимает десятки секунд, а
    попадание в кэш укладывается в единицы миллисекунд (§2.1)."""
    assert min(m.TTFT_BUCKETS) <= 0.01, "нижняя граница не ловит попадание в кэш"
    assert max(m.TTFT_BUCKETS) >= 60.0, "верхняя граница не ловит длинный префилл"
    assert max(m.OVERHEAD_BUCKETS) <= 1.0, (
        "границы накладных расходов слишком широкие: бюджет 5-20 мс, "
        "и разрешение должно быть соответствующим"
    )


# --------------------------------------------------------------------------
# Метрики наполняются на сквозном проходе
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_uspeshnyi_zapros_zapolnyaet_klyuchevye_metriki(gateway, fake_client):
    """Объявленная, но никогда не заполняемая метрика хуже отсутствующей:
    на дашборде она выглядит как «ноль», а не как «нет данных»."""
    before = m.render().decode()
    resp = await gateway.handle(chat_body(), AGENT)
    await collect(resp)
    after = m.render().decode()

    def count(text: str, needle: str) -> float:
        for line in text.splitlines():
            if line.startswith(needle):
                return float(line.split()[-1])
        return 0.0

    assert count(after, "gateway_ttft_seconds_count") > count(before, "gateway_ttft_seconds_count")
    assert count(after, "gateway_requests_total") > count(before, "gateway_requests_total")
    assert count(after, "gateway_tokens_total") > count(before, "gateway_tokens_total")


@pytest.mark.asyncio
async def test_nakladnye_raskhody_izmeryayutsya(gateway, fake_client):
    """Единственная метрика, за которую отвечаем мы (§2.1). Если она не
    заполняется, доказательства SLA не существует."""
    fake_client.default.ttft_s = 0.02
    resp = await gateway.handle(chat_body(), AGENT)
    await collect(resp)

    assert resp.result.upstream_ttft_ms > 0, "время апстрима не измерено"
    assert resp.result.ttft_ms >= resp.result.upstream_ttft_ms, (
        "наше время меньше времени апстрима — отсечка поставлена неверно"
    )
    assert resp.result.overhead_ms >= 0


@pytest.mark.asyncio
async def test_otklonennyi_zapros_popadaet_v_metriki(gateway):
    """Панель «заблокировано квотами» отвечает на вопрос, не задушили ли
    мы легитимный трафик. Пустая панель на этот вопрос не отвечает."""
    from gateway.core.gateway import AdmissionRejected

    gateway.load._pressure.update(5.0)
    with pytest.raises(AdmissionRejected):
        await gateway.handle(chat_body(), "Bearer sk-bg-demo")

    text = m.render().decode()
    assert "gateway_rejected_total" in text
    assert any("overload" in line for line in text.splitlines()
               if line.startswith("gateway_rejected_total"))


@pytest.mark.asyncio
async def test_razryv_klientom_uchityvaetsya_otdelno(gateway, fake_client):
    """Разрыв клиентом — не отказ апстрима. Смешивать их нельзя: иначе
    закрытая вкладка портит репутацию исправному апстриму."""
    import asyncio

    fake_client.default.tokens = 100
    resp = await gateway.handle(chat_body(), AGENT)
    gen = resp.stream
    await gen.__anext__()
    await gen.aclose()
    for _ in range(10):
        await asyncio.sleep(0)

    text = m.render().decode()
    assert any(line.startswith("gateway_client_disconnects_total")
               for line in text.splitlines()), "разрыв клиентом не учтён"


# --------------------------------------------------------------------------
# Служебные эндпоинты
# --------------------------------------------------------------------------


def test_probа_zhivosti_ne_zavisit_ot_apstrimov():
    """Гейтвей жив, даже когда все апстримы мертвы. Если проба живости
    зависит от них, балансировщик снимет исправный экземпляр, и отказ
    апстрима превратится в отказ сервиса."""
    source = Path("gateway/app.py").read_text(encoding="utf-8")
    healthz = source[source.index("async def healthz"):source.index("async def readyz")]
    for forbidden in ("breakers", "upstreams_for", "load."):
        assert forbidden not in healthz, (
            f"проба живости обращается к {forbidden}: отказ апстрима снимет "
            "исправный экземпляр с балансировки"
        )
