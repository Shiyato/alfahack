#!/usr/bin/env bash
# Свип для ADR-001: сколько стоит Python на горячем пути SSE.
#
# Порядок важен: сначала базовая линия напрямую в мок, затем те же запросы
# через прокси. Разница и есть gateway overhead — единственная метрика,
# за которую отвечаем мы (§2.1).
#
# Каждый прогон получает свою соль: KV-кэш мока переживает прогон, и без
# соли второй замер идёт по тёплому кэшу и выглядит быстрее первого.
set -euo pipefail

MOCK=${MOCK:-http://127.0.0.1:9001}
PROXY=${PROXY:-http://127.0.0.1:8000}
BENCH=${BENCH:-/tmp/bench}
OUT=${OUT:-/tmp/spike-results}
DUR=${DUR:-8s}
PROMPT=${PROMPT:-2048}
OUTTOK=${OUTTOK:-6}
LEVELS=${LEVELS:-"1 50 200 500"}

mkdir -p "$OUT"
ulimit -n 10240 || true

measure() { # name url concurrency
  local name=$1 url=$2 c=$3
  "$BENCH" -url "$url" -c "$c" -d "$DUR" \
    -prompt-tokens "$PROMPT" -output-tokens "$OUTTOK" -warmup 30 \
    -json "$OUT/${name}_c${c}.json" > "$OUT/${name}_c${c}.txt"
  sed -n '2p' "$OUT/${name}_c${c}.txt"
}

for c in $LEVELS; do
  echo "=== конкурентность $c ==="
  printf '  %-22s' "напрямую в мок:";      measure direct  "$MOCK/v1/chat/completions" "$c"
  printf '  %-22s' "httpx passthrough:";   measure httpx   "$PROXY/passthrough/v1/chat/completions" "$c"
  printf '  %-22s' "aiohttp passthrough:"; measure aiohttp "$PROXY/aiohttp/v1/chat/completions" "$c"
  printf '  %-22s' "aiohttp + разбор:";    measure parsed  "$PROXY/aiohttp-parsed/v1/chat/completions" "$c"
  printf '  %-22s' "ASGI без апстрима:";   measure asgi    "$PROXY/synthetic/v1/chat/completions" "$c"
done
echo "сырые результаты: $OUT"
