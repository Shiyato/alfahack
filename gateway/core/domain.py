"""Внутренний доменный формат запроса (§3.3.1).

Зачем отдельный формат, а не «гоняем OpenAI-словарь насквозь»: как только
роутер, квоты или кэш начинают читать поля провайдерского формата, они к
нему прирастают, и требование «гибкость» (§1.2) умирает — новый провайдер
потребует правок во всей цепочке. Поэтому на входе нормализуем, на выходе
адаптер переводит обратно.

Формат намеренно узкий: сюда попадает только то, что реально читает
цепочка обработки. Всё остальное провайдерское едет в `passthrough`
нетронутым.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ServiceClass(str, Enum):
    """Классы обслуживания (§2.2).

    Класс `interactive` разделён на человеческий и агентский по §2.1.3:
    человек читает токены по мере поступления, поэтому для него критичен
    TTFT и стриминг; агент действует только по полному ответу, поэтому
    для него TTFT ослаблен, а важна пропускная способность.
    """

    INTERACTIVE = "interactive"   # чат с человеком: жёсткий TTFT, стриминг критичен
    AGENT = "agent"               # агент в IDE: мягкий TTFT, приоритет пропускной способности
    BATCH = "batch"               # массовая обработка: TTFT не нормируется
    BACKGROUND = "background"     # прогревы, аналитика: best effort

    @property
    def priority(self) -> int:
        """Меньше — важнее. Порядок обслуживания в очереди (§3.3.3.4)."""
        return _PRIORITY[self]


_PRIORITY = {
    ServiceClass.INTERACTIVE: 0,
    ServiceClass.AGENT: 1,
    ServiceClass.BATCH: 2,
    ServiceClass.BACKGROUND: 3,
}


@dataclass(slots=True)
class Message:
    role: str
    content: str

    def text(self) -> str:
        return self.content


@dataclass(slots=True)
class Tenant:
    """Результат резолвинга виртуального ключа (§3.3.2).

    Реальные ключи провайдеров не покидают гейтвей: клиент знает только
    виртуальный ключ, и его отзыв не требует ротации провайдерских.
    """

    id: str
    name: str
    team: str = "default"
    service_class: ServiceClass = ServiceClass.INTERACTIVE
    allowed_models: tuple[str, ...] = ()
    # Квоты: None означает «не ограничено этим измерением» (§3.3.3.5)
    tokens_per_minute: int | None = None
    requests_per_minute: int | None = None
    max_concurrent: int | None = None
    # Политики, включаемые на уровне тенанта
    semantic_cache_enabled: bool = False
    cross_tenant_cache: bool = False

    def may_use(self, model: str) -> bool:
        return not self.allowed_models or model in self.allowed_models


@dataclass(slots=True)
class ChatRequest:
    """Нормализованный запрос — то, с чем работает вся цепочка."""

    model: str
    messages: list[Message]
    stream: bool = False
    max_tokens: int | None = None
    temperature: float | None = None
    # Провайдерские поля, которые мы не интерпретируем, но обязаны донести.
    passthrough: dict[str, Any] = field(default_factory=dict)

    # --- Заполняется по ходу цепочки ---
    tenant: Tenant | None = None
    request_id: str = ""
    received_at: float = 0.0
    prompt_tokens_est: int = 0

    def prompt_text(self) -> str:
        """Плоский текст переписки — основа для хеширования префикса.

        Порядок и разделители фиксированы: от этого зависит устойчивость
        блочного цепочечного хеширования (§3.3.6.2), а значит и
        воспроизводимость попаданий в кэш между ходами одной сессии.
        """
        return "".join(f"{m.role}\n{m.content}\n" for m in self.messages)

    @property
    def turn(self) -> int:
        """Номер хода в сессии, выведенный из самого запроса (§3.3.6.3b).

        Ключевой приём: LLM-API устроен без состояния, каждый запрос несёт
        всю предыдущую переписку, поэтому номер хода определяется по
        количеству исторических сообщений. Роутер не хранит таблицу сессий,
        которая иначе росла бы бесконечно — сессия никогда не сообщает о
        своём завершении.

        Вырожденный случай безопасен: если агент отбросил историю, запрос
        будет обработан как первый в новой сессии. Это и есть правильное
        решение, потому что такой запрос всё равно почти не попадёт в кэш.
        """
        return max(0, sum(1 for m in self.messages if m.role == "assistant"))

    def session_key(self, tenant_id: str) -> str:
        """Идентификатор сессии: хеш истории без последнего хода.

        Последний ход исключается специально: именно он отличает текущий
        запрос от предыдущего, а нам нужно опознать, что это та же сессия.

        Идентификатор тенанта входит в ключ обязательно (§3.3.6.10):
        общий префикс-кэш между тенантами — это боковой канал, по которому
        чужой промпт угадывается по времени ответа.
        """
        history = self.messages[:-1] if len(self.messages) > 1 else self.messages
        h = hashlib.blake2b(digest_size=16)
        h.update(tenant_id.encode())
        h.update(b"\x00")
        for m in history:
            h.update(m.role.encode())
            h.update(b"\x00")
            h.update(m.content.encode())
            h.update(b"\x00")
        return h.hexdigest()


@dataclass(slots=True)
class Upstream:
    """Апстрим из реестра моделей (§3.3.7)."""

    id: str
    base_url: str
    model: str                      # имя модели на стороне провайдера
    adapter: str = "openai"
    api_key: str | None = None
    weight: float = 1.0
    # Эндпоинт внутреннего состояния, если апстрим его отдаёт (§3.3.6.8, сценарий 1).
    state_url: str | None = None
    timeout_connect_s: float = 5.0
    timeout_ttft_s: float = 30.0
    timeout_total_s: float = 600.0
    enabled: bool = True
