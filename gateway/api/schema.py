"""Внешний контракт: OpenAI-совместимый (§3.3.1).

Выбор формата — не вкусовщина. Это де-факто стандарт: любой существующий
клиент — плагин в IDE, Open WebUI, SDK — начинает работать без единой
строки правок. Нулевая стоимость интеграции и отсутствие вендор-лока.

Нормализация во внутренний формат происходит **сразу на входе**: иначе
логика роутинга прирастёт к формату конкретного провайдера, и требование
«гибкость» умрёт.
"""

from __future__ import annotations

from typing import Any

from ..core.domain import ChatRequest, Message

# Поля, которые мы интерпретируем сами. Всё остальное едет в passthrough
# нетронутым: провайдер может понимать больше, чем мы, и терять это
# нельзя.
_KNOWN = {"model", "messages", "stream", "max_tokens", "temperature", "stream_options"}


class ValidationError(Exception):
    pass


def parse_chat_request(body: dict[str, Any]) -> ChatRequest:
    if not isinstance(body, dict):
        raise ValidationError("тело запроса должно быть объектом")

    model = body.get("model")
    if not model or not isinstance(model, str):
        raise ValidationError("поле model обязательно и должно быть строкой")

    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValidationError("поле messages обязательно и должно быть непустым списком")

    messages: list[Message] = []
    for i, m in enumerate(raw_messages):
        if not isinstance(m, dict):
            raise ValidationError(f"messages[{i}] должен быть объектом")
        role = m.get("role")
        if not isinstance(role, str):
            raise ValidationError(f"messages[{i}].role обязателен")
        content = m.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            # Мультимодальный контент приходит списком частей. Для
            # хеширования префикса нам нужен текст, поэтому склеиваем
            # текстовые части, а исходную структуру сохраняем в passthrough.
            if isinstance(content, list):
                content = "".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            else:
                raise ValidationError(f"messages[{i}].content неподдерживаемого типа")
        messages.append(Message(role=role, content=content))

    max_tokens = body.get("max_tokens")
    if max_tokens is not None and (not isinstance(max_tokens, int) or max_tokens <= 0):
        raise ValidationError("max_tokens должен быть положительным целым")

    temperature = body.get("temperature")
    if temperature is not None and not isinstance(temperature, (int, float)):
        raise ValidationError("temperature должен быть числом")

    return ChatRequest(
        model=model,
        messages=messages,
        stream=bool(body.get("stream", False)),
        max_tokens=max_tokens,
        temperature=float(temperature) if temperature is not None else None,
        passthrough={k: v for k, v in body.items() if k not in _KNOWN},
    )


def error_body(message: str, *, kind: str = "invalid_request_error",
               code: str | None = None) -> dict[str, Any]:
    """Формат ошибки OpenAI: клиенты умеют его разбирать."""
    return {"error": {"message": message, "type": kind, "code": code}}
