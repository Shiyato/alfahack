"""Проверка слоистости пакетов.

Зачем отдельный тест. Слоистость — свойство, которое теряется незаметно:
один импорт «по-быстрому» в четыре часа ночи, и подсистему больше нельзя
протестировать, не подняв половину системы. Заметить это при чтении
диффа почти невозможно, а последствия проявляются недели спустя.

Правило простое: **подсистемы импортируют только `core`** (домен и
конфигурацию), а `core` собирает из них цепочку. Одно осознанное
исключение перечислено явно.

Проверка статическая, по исходникам — не по графу CodeGraph: связи
`calls` там разрешаются по имени символа, и обычный `dict.get()`
привязывается к методу `get` любого класса. Первая версия выгрузки графа
из-за этого показывала зависимость `admission → resilience`, которой
в коде нет.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

GATEWAY = Path("gateway")

# Подсистемам разрешено импортировать только это.
ALLOWED_FOR_SUBSYSTEM = {"core"}

# Осознанные исключения: пакет → что ему дополнительно можно и почему.
EXCEPTIONS = {
    # Конвейеру нужен тип события потока. Зависимость от структуры
    # данных, а не от поведения: подменить адаптер он не может и не хочет.
    "stream": {"upstream"},
}

# `core` собирает цепочку и потому импортирует всех — это его работа.
ASSEMBLER = "core"

IMPORT_RE = re.compile(r"^\s*from\s+\.\.([a-z_]+)", re.MULTILINE)


def subsystems() -> list[str]:
    return sorted(
        p.name for p in GATEWAY.iterdir()
        if p.is_dir() and not p.name.startswith("_") and (p / "__init__.py").exists()
    )


def imports_of(package: str) -> set[str]:
    """Какие соседние пакеты импортирует этот."""
    found: set[str] = set()
    for f in (GATEWAY / package).glob("*.py"):
        found.update(IMPORT_RE.findall(f.read_text(encoding="utf-8")))
    return found


@pytest.mark.parametrize("package", subsystems())
def test_podsistema_importiruet_tolko_razreshennoe(package):
    if package == ASSEMBLER:
        pytest.skip("core собирает цепочку и импортирует всех по назначению")

    allowed = ALLOWED_FOR_SUBSYSTEM | EXCEPTIONS.get(package, set())
    actual = imports_of(package)
    extra = actual - allowed

    assert not extra, (
        f"пакет {package!r} импортирует {sorted(extra)}, а ему разрешено "
        f"только {sorted(allowed)}. Слой протёк: теперь {package} нельзя "
        "протестировать, не поднимая эти пакеты. Если зависимость нужна "
        "по существу — добавьте её в EXCEPTIONS с объяснением почему."
    )


def test_nikto_ne_importiruet_sborku():
    """Обратная зависимость на `core.gateway` означала бы круговой импорт
    и невозможность собрать подсистему отдельно."""
    offenders = []
    for package in subsystems():
        if package == ASSEMBLER:
            continue
        for f in (GATEWAY / package).glob("*.py"):
            text = f.read_text(encoding="utf-8")
            if re.search(r"from\s+\.\.core\.gateway\s+import", text):
                offenders.append(str(f))
    assert not offenders, (
        f"подсистемы импортируют сборку горячего пути: {offenders}"
    )


def test_kazhdyi_paket_opisan():
    """Пустой `__init__.py` — первое, что видит человек, зашедший в
    каталог. Он должен отвечать, за что пакет отвечает и что в нём легко
    сломать."""
    empty = [
        str(p) for p in GATEWAY.rglob("__init__.py")
        if len(p.read_text(encoding="utf-8").strip()) < 50
    ]
    assert not empty, f"пакеты без описания: {empty}"


def test_kazhdyi_modul_ssylaetsya_na_razdel_dokumenta():
    """Каждый модуль должен быть трассируем до раздела architecture.md:
    без этого через неделю непонятно, какую задачу он решает."""
    missing = []
    for f in GATEWAY.rglob("*.py"):
        if f.name == "__init__.py":
            continue
        head = f.read_text(encoding="utf-8")[:4000]
        if "§" not in head:
            missing.append(str(f))
    assert not missing, (
        f"модули без ссылки на раздел architecture.md: {missing}"
    )
