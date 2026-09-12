"""Реестр конфигурации с hot-reload (§3.1).

Горячий путь не должен ходить в БД или на диск за конфигом на каждом
запросе — это и латентность, и точка отказа. Поэтому data-plane читает
конфиг из памяти, а control-plane его туда кладёт.

Из этого следуют два свойства, которые и требовались в §3.1:
падение control-plane не останавливает трафик (data-plane работает на
последнем известном конфиге), а модель добавляется правкой файла без
передеплоя.

Отдельное правило: **сломанный конфиг не применяется.** Если новая версия
не читается или валидатор нашёл в ней грубую ошибку, остаётся предыдущая,
и об этом громко сообщается. Гейтвей, упавший из-за опечатки в YAML в
прямом эфире, — худший исход из возможных.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable

from .config import GatewayConfig, load_config

log = logging.getLogger(__name__)

# Типы изменений, о которых уведомляются подписчики.
ReloadCallback = Callable[[GatewayConfig, GatewayConfig], None]


class ConfigRegistry:
    """Держит актуальную конфигурацию и следит за файлами.

    Чтение конфига из горячего пути — это обращение к одному атрибуту,
    без блокировок: ссылка подменяется целиком и атомарно. Запрос,
    начавшийся на старой версии, доработает на ней — это правильно,
    иначе конфиг менялся бы посреди обработки.
    """

    def __init__(self, config_dir: str | Path, *, poll_interval_s: float = 1.0) -> None:
        self._dir = Path(config_dir)
        self._poll_interval = poll_interval_s
        self._config = GatewayConfig()
        self._issues: list[str] = []
        self._mtimes: dict[str, float] = {}
        self._subscribers: list[ReloadCallback] = []
        self._task: asyncio.Task | None = None
        self._reload_count = 0
        self._failed_reloads = 0
        self._last_error: str | None = None

    # --- Чтение из горячего пути ---

    @property
    def config(self) -> GatewayConfig:
        return self._config

    @property
    def issues(self) -> list[str]:
        return list(self._issues)

    def stats(self) -> dict[str, object]:
        return {
            "version": self._config.version,
            "reloads": self._reload_count,
            "failed_reloads": self._failed_reloads,
            "issues": len(self._issues),
            "last_error": self._last_error,
        }

    # --- Управление ---

    def subscribe(self, cb: ReloadCallback) -> None:
        """Подписка на смену конфига.

        Нужна компонентам с состоянием: например, роутеру, которому при
        добавлении апстрима надо расширить хеш-кольцо, а не перестроить
        его с нуля (§3.3.6.5) — иначе ремаппинг разрушит cache affinity.
        """
        self._subscribers.append(cb)

    def load_now(self) -> list[str]:
        """Синхронная загрузка. Бросает исключение только на старте:
        стартовать с заведомо нечитаемым конфигом бессмысленно."""
        cfg, issues = load_config(self._dir)
        cfg.version = self._config.version + 1
        old = self._config
        self._config = cfg
        self._issues = issues
        self._snapshot_mtimes()
        self._notify(old, cfg)
        return issues

    async def start(self) -> None:
        self.load_now()
        if self._issues:
            for i in self._issues:
                log.warning("конфигурация: %s", i)
        self._task = asyncio.create_task(self._watch(), name="config-watch")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    # --- Внутреннее ---

    def _snapshot_mtimes(self) -> None:
        self._mtimes = {
            p.name: p.stat().st_mtime for p in self._dir.glob("*.yaml") if p.is_file()
        }

    def _changed(self) -> bool:
        try:
            current = {
                p.name: p.stat().st_mtime for p in self._dir.glob("*.yaml") if p.is_file()
            }
        except OSError:
            return False
        return current != self._mtimes

    async def _watch(self) -> None:
        """Опрос mtime. Осознанно проще inotify/watchfiles: конфиг меняется
        раз в минуты, а не тысячи раз в секунду, и одна зависимость здесь
        дороже, чем цикл на секундном таймере."""
        while True:
            try:
                await asyncio.sleep(self._poll_interval)
                if not self._changed():
                    continue
                await asyncio.to_thread(self._reload)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("сбой цикла наблюдения за конфигурацией")

    def _reload(self) -> None:
        try:
            cfg, issues = load_config(self._dir)
        except Exception as exc:
            # Сломанный конфиг не применяется: остаётся предыдущая версия.
            self._failed_reloads += 1
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._snapshot_mtimes()  # не долбиться в тот же битый файл
            log.error("конфигурация не перечитана, работаем на версии %d: %s",
                      self._config.version, self._last_error)
            return

        blocking = [i for i in issues if _is_blocking(i)]
        if blocking:
            self._failed_reloads += 1
            self._last_error = "; ".join(blocking)
            self._snapshot_mtimes()
            log.error("конфигурация отвергнута валидатором, работаем на версии %d: %s",
                      self._config.version, self._last_error)
            return

        old = self._config
        cfg.version = old.version + 1
        self._config = cfg
        self._issues = issues
        self._reload_count += 1
        self._last_error = None
        self._snapshot_mtimes()
        for i in issues:
            log.warning("конфигурация v%d: %s", cfg.version, i)
        log.info("конфигурация перечитана: версия %d, апстримов %d, моделей %d",
                 cfg.version, len(cfg.upstreams), len(cfg.model_aliases))
        self._notify(old, cfg)

    def _notify(self, old: GatewayConfig, new: GatewayConfig) -> None:
        for cb in self._subscribers:
            try:
                cb(old, new)
            except Exception:
                log.exception("подписчик на смену конфигурации упал")


# Замечания, при которых конфиг применять нельзя: они означают, что система
# будет вести себя не так, как написано в файле. Остальные — предупреждения.
_BLOCKING_MARKERS = (
    "гистерезис отсутствует",
    "неизвестная стратегия роутинга",
    "не имеет ни одного апстрима",
    "ewma_alpha",
)


def _is_blocking(issue: str) -> bool:
    return any(m in issue for m in _BLOCKING_MARKERS)
