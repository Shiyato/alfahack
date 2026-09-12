"""Резолвинг виртуальных ключей (§3.3.2).

Виртуальный ключ решает энтерпрайз-задачу: реальные ключи провайдеров не
покидают гейтвей. Отозвать доступ клиенту можно, не трогая провайдерские
ключи и не задевая остальных.

Резолвинг идёт из памяти, наполняемой control-plane: в горячем пути
**ноль сетевых вызовов**. Поход в базу за конфигом на каждом запросе —
это и латентность, и точка отказа (§3.1).
"""

from __future__ import annotations

import hmac

from ..core.config import GatewayConfig
from ..core.domain import Tenant


class AuthError(Exception):
    def __init__(self, message: str, *, status: int = 401) -> None:
        super().__init__(message)
        self.status = status


def resolve(cfg: GatewayConfig, authorization: str | None) -> Tenant:
    """Ключ → тенант. Бросает AuthError с осмысленным статусом."""
    if not authorization:
        raise AuthError("отсутствует заголовок Authorization")

    token = authorization
    if authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if not token:
        raise AuthError("пустой ключ")

    # Сравнение постоянного времени: обычное сравнение строк утекает
    # длину совпавшего префикса через время ответа. Ключей немного,
    # перебор дешёвый.
    #
    # Сравниваем байты, а не строки. `compare_digest` на строках требует
    # ASCII и бросает TypeError на чём угодно другом — то есть ключ с
    # кириллицей или эмодзи ронял бы запрос в 500 вместо честного 401.
    # Найдено тестом с ключом "sk-нет-такого".
    token_bytes = token.encode("utf-8", "surrogatepass")
    for known, tenant in cfg.tenants.items():
        if hmac.compare_digest(known.encode("utf-8", "surrogatepass"), token_bytes):
            return tenant
    raise AuthError("неизвестный ключ")


def authorize_model(tenant: Tenant, model: str, cfg: GatewayConfig) -> None:
    if model not in cfg.model_aliases:
        raise AuthError(f"модель {model!r} не зарегистрирована", status=404)
    if not tenant.may_use(model):
        raise AuthError(f"тенанту {tenant.id!r} не разрешена модель {model!r}", status=403)
