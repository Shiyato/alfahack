# Гейтвей. Python выбран замером (docs/adr/001-language-runtime.md):
# накладные расходы 1-4 мс при бюджете 5-20 мс.
FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Зависимости отдельным слоем: код меняется чаще, чем они.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY gateway/ ./gateway/
COPY config/ ./config/

ENV PATH="/app/.venv/bin:$PATH" \
    GATEWAY_CONFIG_DIR=/app/config

EXPOSE 8080

# Число воркеров задаётся снаружи: потолок одного — около 200
# одновременных стримов (docs/benchmarks.md, Б-1).
ENV WORKERS=2

CMD uvicorn gateway.app:app --host 0.0.0.0 --port 8080 \
    --loop uvloop --http httptools --no-access-log --workers ${WORKERS}
