"""Выгрузка графа кода в версионируемый вид.

Зачем это нужно. CodeGraph держит индекс в SQLite (`.codegraph/codegraph.db`)
и сам помечает его как локальный для каждой машины: база стареет при
первой же правке кода, весит мегабайты и конфликтует при слиянии. Класть
её в git бессмысленно.

Но сам граф полезен и в репозитории: он отвечает на вопросы «кто вызывает
эту функцию» и «что сломается, если её тронуть» — без запуска
инструмента, из любого просмотрщика диффов, и меняется вместе с кодом,
поэтому его расхождение с реальностью видно в ревью.

Поэтому база остаётся локальной, а из неё выгружается текстовая карта.

Запуск:
    codegraph index              # обновить индекс
    uv run python tools/export_codegraph.py
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path

DB = Path(".codegraph/codegraph.db")
OUT = Path("docs/codegraph.md")

# Только наш код: тесты и стенд в карте зависимостей только шумят.
CORE_PREFIX = "gateway/"

# Карта зависимостей строится ТОЛЬКО по связям `imports`.
#
# Причина конкретная: связи `calls` в индексе разрешаются по имени
# символа, и обычный `dict.get()` привязывается к методу `get` любого
# класса в проекте. В первой версии выгрузки это дало ребро
# «admission → resilience», которого в коде нет: настоящие импорты
# пакетов слоисты и в resilience не ходит никто.
#
# Вызовы остаются полезны для другого вопроса — «кто дёргает этот
# символ», — но там ложное срабатывание стоит дешевле и видно глазом.
STRUCTURAL_EDGES = ("imports", "extends")

# Имена, по которым разрешение вызовов заведомо ненадёжно: они есть
# у словарей, множеств и половины классов сразу.
AMBIGUOUS = {
    "get", "set", "put", "add", "update", "clear", "remove", "pop",
    "keys", "values", "items", "stats", "close", "start", "stop",
    "append", "extend", "record", "observe", "select", "load",
}


def load(con: sqlite3.Connection):
    nodes = {
        r["id"]: dict(r)
        for r in con.execute(
            "SELECT id, kind, name, qualified_name, file_path, language, "
            "start_line, end_line, signature, is_async FROM nodes"
        )
    }
    edges = [dict(r) for r in con.execute(
        "SELECT source, target, kind, line FROM edges"
    )]
    return nodes, edges


def module_of(path: str) -> str:
    """Имя пакета: gateway/router/prefix.py → gateway/router"""
    p = Path(path)
    return str(p.parent) if p.parent != Path(".") else path


def main() -> None:
    if not DB.exists():
        raise SystemExit(
            "индекс не найден. Запустите `codegraph init` (или `codegraph index`)"
        )

    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    nodes, edges = load(con)

    core = {i: n for i, n in nodes.items()
            if (n["file_path"] or "").startswith(CORE_PREFIX)}

    # --- Зависимости между пакетами: только структурные связи ---
    pkg_edges: dict[tuple[str, str], int] = defaultdict(int)
    for e in edges:
        if e["kind"] not in STRUCTURAL_EDGES:
            continue
        src, tgt = core.get(e["source"]), core.get(e["target"])
        if not src or not tgt:
            continue
        a, b = module_of(src["file_path"]), module_of(tgt["file_path"])
        if a != b:
            pkg_edges[(a, b)] += 1

    # --- Кто кого вызывает: входящие связи по символам ---
    callers: dict[int, set[int]] = defaultdict(set)
    for e in edges:
        if e["kind"] != "calls":
            continue
        src, tgt = core.get(e["source"]), core.get(e["target"])
        if not src or not tgt:
            continue
        if tgt["name"] in AMBIGUOUS:
            continue
        callers[e["target"]].add(e["source"])

    lines: list[str] = []
    add = lines.append

    add("# Граф кода\n")
    add("Выгружается из индекса CodeGraph командой")
    add("`uv run python tools/export_codegraph.py`.\n")
    add("Сама база (`.codegraph/codegraph.db`) в репозиторий не кладётся:")
    add("она локальна для машины, стареет при первой правке и конфликтует")
    add("при слиянии. Здесь — её версионируемая выжимка, которая меняется")
    add("вместе с кодом, поэтому расхождение с реальностью видно в ревью.\n")

    add(f"Индекс: **{len(nodes)} узлов**, **{len(edges)} связей**, "
        f"из них в `gateway/` — {len(core)} узлов.\n")

    # --- Зависимости пакетов ---
    add("---\n")
    add("## Зависимости между пакетами\n")
    add("Построено по связям `imports`: они разрешаются однозначно.")
    add("Стрелка означает «импортирует». Число — количество связей.\n")
    add("```mermaid")
    add("graph LR")
    seen_nodes = set()
    for (a, b), n in sorted(pkg_edges.items(), key=lambda kv: -kv[1]):
        for x in (a, b):
            if x not in seen_nodes:
                seen_nodes.add(x)
                add(f'  {x.replace("/", "_")}["{x}"]')
        add(f'  {a.replace("/", "_")} -->|{n}| {b.replace("/", "_")}')
    add("```\n")

    add("**Направление связей — проверка архитектуры на месте.** `core`")
    add("собирает цепочку и потому импортирует всех. Подсистемы")
    add("импортируют только `core` (домен и конфигурацию) — это и есть")
    add("слоистость: любую из них можно тестировать отдельно, не поднимая")
    add("остальные. Появление стрелки между двумя подсистемами означало бы,")
    add("что слой протёк.\n")
    add("Единственное исключение — `stream → upstream`: конвейеру нужен тип")
    add("события потока. Зависимость от типа данных, а не от поведения.\n")

    # --- Самые связанные символы ---
    add("---\n")
    add("## Что тронуть опаснее всего\n")
    add("Символы с наибольшим числом входящих вызовов: правка любого из")
    add("них задевает много мест, поэтому изменения здесь требуют")
    add("отдельного внимания.\n")
    add("Оговорка о точности: вызовы разрешаются по имени символа, поэтому")
    add("одноимённые методы разных классов могут склеиваться. Имена, у")
    add("которых это заведомо так (`get`, `put`, `stats` и подобные),")
    add("исключены, но список эвристический — считайте это подсказкой,")
    add("а не доказательством.\n")
    add("| Символ | Где | Вызывающих |")
    add("|---|---|---:|")
    top = sorted(callers.items(), key=lambda kv: -len(kv[1]))[:20]
    for nid, srcs in top:
        n = core[nid]
        add(f"| `{n['name']}` | `{n['file_path']}:{n['start_line']}` | {len(srcs)} |")
    add("")

    # --- Карта символов по модулям ---
    add("---\n")
    add("## Символы по модулям\n")
    by_file: dict[str, list[dict]] = defaultdict(list)
    for n in core.values():
        if n["kind"] in ("class", "function", "method"):
            by_file[n["file_path"]].append(n)

    for path in sorted(by_file):
        items = sorted(by_file[path], key=lambda n: n["start_line"])
        classes = [n for n in items if n["kind"] == "class"]
        funcs = [n for n in items if n["kind"] == "function"]
        add(f"### `{path}`\n")
        if classes:
            add("Классы: " + ", ".join(
                f"`{c['name']}`" for c in classes) + "  ")
        if funcs:
            add("Функции: " + ", ".join(
                f"`{f['name']}`" for f in funcs))
        add("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"записано в {OUT}: {len(lines)} строк, "
          f"{len(pkg_edges)} межпакетных связей")


if __name__ == "__main__":
    main()
