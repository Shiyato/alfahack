"""ASGI-приложение (§3.3.1).

Тонкий слой: разбор HTTP, коды ответов, SSE наружу. Вся логика живёт в
`Gateway`, чтобы её можно было тестировать без HTTP и без сети.

Отдельного внимания стоит контракт отказа (§3.3.3.6): 429 отдаётся с
заголовками, по которым клиент понимает, сколько осталось и когда
повторять. Смысл в том, чтобы клиент мог выстроить автоматическую логику
повтора, а не гадать. Дёшево в реализации и заметно улучшает поведение
системы под нагрузкой.
"""

from __future__ import annotations

import json
import logging
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from .api.schema import ValidationError, error_body
from .auth.resolver import AuthError
from .core.gateway import AdmissionRejected, Gateway
from .core.registry import ConfigRegistry
from .telemetry import metrics as m
from .upstream.adapter import UpstreamError

log = logging.getLogger(__name__)

CONFIG_DIR = os.environ.get("GATEWAY_CONFIG_DIR", "config")

SSE_HEADERS = {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # Без этого nginx буферизует стрим и превращает TTFT в полное время
    # ответа — ошибка, которую находят уже на демо.
    "X-Accel-Buffering": "no",
}


def create_app(config_dir: str | None = None) -> FastAPI:
    registry = ConfigRegistry(config_dir or CONFIG_DIR)
    registry.load_now()
    gateway = Gateway(registry)

    app = FastAPI(title="LLM Gateway", version="0.1.0", docs_url=None, redoc_url=None)
    app.state.registry = registry
    app.state.gateway = gateway

    @app.on_event("startup")
    async def _startup() -> None:
        await registry.start()
        await gateway.start()
        m.config_version.set(registry.config.version)
        for issue in registry.issues:
            log.warning("конфигурация: %s", issue)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await gateway.stop()
        await registry.stop()

    # ------------------------------------------------------------------
    # Горячий путь
    # ------------------------------------------------------------------

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content=error_body("тело не является JSON"))

        try:
            response = await gateway.handle(body, request.headers.get("authorization"))
        except ValidationError as exc:
            return JSONResponse(status_code=400, content=error_body(str(exc)))
        except AuthError as exc:
            return JSONResponse(status_code=exc.status,
                                content=error_body(str(exc), kind="authentication_error"))
        except AdmissionRejected as exc:
            return JSONResponse(
                status_code=429,
                content=error_body(str(exc), kind="rate_limit_error", code=exc.limit_kind),
                headers={
                    "Retry-After": str(max(1, int(exc.retry_after_s))),
                    "X-RateLimit-Remaining-Tokens": str(int(exc.remaining_tokens)),
                    "X-RateLimit-Limit-Kind": exc.limit_kind,
                },
            )
        except UpstreamError as exc:
            # 502 означает «апстрим отказал», а не «мы сломались» — для
            # клиента это разные ситуации с разной реакцией.
            status = 502 if exc.status is None else exc.status
            return JSONResponse(status_code=status,
                                content=error_body(str(exc), kind="upstream_error"))

        if not body.get("stream", False):
            # Нестриминговый ответ собираем из стрима: к апстриму мы в
            # любом случае идём стримом, поэтому второй ветки не нужно.
            return await _collect(response, body.get("model", ""))

        return StreamingResponse(
            response.stream, headers={**SSE_HEADERS, **response.headers}
        )

    async def _collect(response, model: str):
        parts: list[str] = []
        finish = "stop"
        async for chunk in response.stream:
            for line in chunk.split(b"\n"):
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:].strip()
                if payload == b"[DONE]" or not payload:
                    continue
                try:
                    obj = json.loads(payload)
                except ValueError:
                    continue
                for ch in obj.get("choices") or []:
                    if content := (ch.get("delta") or {}).get("content"):
                        parts.append(content)
                    if fr := ch.get("finish_reason"):
                        finish = fr
        r = response.result
        return JSONResponse(
            content={
                "id": f"chatcmpl-{r.upstream_id}",
                "object": "chat.completion",
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(parts)},
                    "finish_reason": finish,
                }],
                "usage": {
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "total_tokens": r.prompt_tokens + r.completion_tokens,
                },
            },
            headers=response.headers,
        )

    # ------------------------------------------------------------------
    # Служебные эндпоинты
    # ------------------------------------------------------------------

    @app.get("/v1/models")
    async def models():
        cfg = registry.config
        return {
            "object": "list",
            "data": [
                {"id": name, "object": "model", "owned_by": "gateway",
                 "upstreams": len(cfg.upstreams_for(name))}
                for name in cfg.model_aliases
            ],
        }

    @app.get("/healthz")
    async def healthz():
        """Проба живости. Намеренно не зависит от апстримов: гейтвей жив,
        даже когда все апстримы мертвы, и балансировщик не должен его
        снимать — иначе отказ апстрима превращается в отказ сервиса."""
        return {"status": "ok", "config_version": registry.config.version}

    @app.get("/readyz")
    async def readyz():
        cfg = registry.config
        live = [
            u.id for u in cfg.upstreams.values()
            if u.enabled and gateway.breakers.get(u.id).allows()
        ]
        ready = bool(live)
        return JSONResponse(
            status_code=200 if ready else 503,
            content={"ready": ready, "live_upstreams": live},
        )

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(m.render().decode(), media_type="text/plain; version=0.0.4")

    @app.get("/admin/state")
    async def admin_state():
        """Внутреннее состояние для дашборда и отладки на демо."""
        cfg = registry.config
        return {
            "config": registry.stats(),
            "issues": registry.issues,
            "strategy": gateway.strategy.name,
            "calibrated": gateway.ctx.calibrated,
            "queue": gateway.queue.snapshot(),
            "pressure": round(gateway.load.pressure, 3),
            "rejecting": gateway.load.should_reject(),
            "flips": gateway.load.flips,
            "breakers": gateway.breakers.snapshot(),
            "prefix_table": gateway.prefix.stats(),
            "dispatcher": gateway.dispatcher.stats(),
            "upstreams": [
                {
                    "id": u.upstream_id,
                    "inflight": u.inflight_requests,
                    "inflight_tokens": u.inflight_tokens,
                    "pending_prefill": u.pending_prefill_tokens,
                    "busy": u.busy(),
                    "observed_ttft_ms": round(u.observed_ttft_ms.value, 1),
                }
                for u in gateway.load.all_upstreams()
            ],
        }

    return app


app = create_app()
