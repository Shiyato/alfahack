"""Блочное цепочечное хеширование префикса (§3.3.6.2).

Схема из Mooncake: вход режется на блоки, и хеш каждого блока считается
**от токенов блока, сцепленных с хешем предыдущего блока**. Получается
цепочка, в которой каждый блок идентифицируется вместе со всей своей
предысторией.

Почему цепочка, а не независимые хеши блоков: одинаковый блок в разных
контекстах — это разные KV-состояния. Хеш от «блок + предыстория» решает
это автоматически и заодно даёт дедупликацию.

Сравнение ключей идёт последовательно **до первого несовпадения** — это и
даёт `prefix_len`. Разрыв в середине нельзя «перепрыгнуть»: KV-состояние
блока k осмысленно, только если посчитаны все блоки до него.

Эта же структура нужна и роутеру (§3.3.6), и кэшу (§3.3.5) — поэтому
живёт отдельно от обоих.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict

# Грубая, но детерминированная оценка: один токен ≈ 4 байта текста.
# Точный токенизатор здесь не нужен — нужна монотонность по длине и
# воспроизводимость. Настоящий подсчёт токенов для квот делает §3.3.10.
BYTES_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return len(text) // BYTES_PER_TOKEN


def block_hashes(text: str, block_tokens: int, *, salt: str = "") -> list[bytes]:
    """Цепочечные хеши полных блоков.

    `salt` — идентификатор тенанта. Он обязан входить в ключ (§3.3.6.10):
    общий префикс-кэш между тенантами есть боковой канал, по которому
    чужой промпт угадывается по времени ответа. Цена — ниже cache hit rate
    при множестве мелких тенантов; это осознанный размен.

    Хвост короче блока в ключ не попадает: частично посчитанный блок не
    даёт переиспользуемого KV-состояния.
    """
    block_bytes = block_tokens * BYTES_PER_TOKEN
    if block_bytes <= 0:
        return []
    data = text.encode("utf-8", errors="ignore")
    n = len(data) // block_bytes
    out: list[bytes] = []
    prev = salt.encode() if salt else b""
    for i in range(n):
        h = hashlib.blake2b(digest_size=16)
        h.update(prev)
        h.update(data[i * block_bytes : (i + 1) * block_bytes])
        prev = h.digest()
        out.append(prev)
    return out


def common_prefix_len(a: list[bytes], b: list[bytes]) -> int:
    """Длина общего префикса в блоках — до первого несовпадения."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class PrefixTable:
    """Таблица «префикс → апстрим» с TTL (§3.3.6.8, сценарий 2).

    Заменяет метаданные KV-кэша, которых у нас нет при непрозрачном
    апстриме. Мы не знаем, что лежит в кэше апстрима, но знаем, что сами
    туда отправляли, — и этого достаточно, чтобы предположить попадание.

    TTL порядка минут: по трассам агентов ~90% переиспользований
    происходит в пределах ~100 секунд, P99 около 841–911 секунд (§5.3).
    Хранить префиксы часами бессмысленно — TTL в 5–10 минут покрывает
    подавляющую часть выгоды при скромной памяти.

    Состояние мягкое: потеря таблицы при перезапуске экземпляра снижает
    долю попаданий, но не ломает корректность (§3.3.8.1).
    """

    def __init__(self, *, ttl_s: float = 600.0, max_entries: int = 200_000) -> None:
        self._ttl = ttl_s
        self._max = max_entries
        # ключ блока → (апстрим, время последнего касания)
        self._entries: OrderedDict[bytes, tuple[str, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def record(self, hashes: list[bytes], upstream_id: str, *, now: float | None = None) -> None:
        now = now if now is not None else time.monotonic()
        for h in hashes:
            self._entries[h] = (upstream_id, now)
            self._entries.move_to_end(h)
        self._evict(now)

    def hit_len(self, hashes: list[bytes], upstream_id: str, *, now: float | None = None) -> int:
        """Сколько блоков префикса, предположительно, лежит на апстриме.

        Считается последовательно до первого блока, которого там нет, —
        та же логика, что и в настоящем KV-кэше.
        """
        now = now if now is not None else time.monotonic()
        n = 0
        for h in hashes:
            entry = self._entries.get(h)
            if entry is None or entry[0] != upstream_id or (now - entry[1]) > self._ttl:
                break
            n += 1
        if n:
            self._hits += 1
        else:
            self._misses += 1
        return n

    def best_upstream(self, hashes: list[bytes], *, now: float | None = None) -> tuple[str | None, int]:
        """Апстрим с наибольшим ожидаемым попаданием и длина попадания."""
        now = now if now is not None else time.monotonic()
        # Кандидаты — те, кто владеет первым блоком: только они могут
        # иметь непрерывный префикс.
        if not hashes:
            return None, 0
        first = self._entries.get(hashes[0])
        if first is None or (now - first[1]) > self._ttl:
            return None, 0
        candidates = {first[0]}
        best_id, best_len = None, 0
        for uid in candidates:
            n = self.hit_len(hashes, uid, now=now)
            if n > best_len:
                best_id, best_len = uid, n
        return best_id, best_len

    def _evict(self, now: float) -> None:
        # Протухшие с головы (OrderedDict хранит в порядке касания).
        while self._entries:
            key = next(iter(self._entries))
            _, ts = self._entries[key]
            if (now - ts) <= self._ttl:
                break
            del self._entries[key]
        while len(self._entries) > self._max:
            self._entries.popitem(last=False)

    def stats(self) -> dict[str, int | float]:
        total = self._hits + self._misses
        return {
            "entries": len(self._entries),
            "lookups": total,
            "hit_rate": self._hits / total if total else 0.0,
        }
