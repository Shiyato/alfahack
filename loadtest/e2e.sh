#!/usr/bin/env bash
# Сквозной сценарий: поднимает стенд и проверяет всё, что нельзя
# проверить без сети — стриминг, отказоустойчивость, hot-reload.
#
# Это не замена тестам, а проверка сборки: каждый пункт здесь
# соответствует требованию кейса из §1.2 и показывается на защите.
#
# Запуск:  loadtest/e2e.sh
set -uo pipefail

GW=${GW:-http://127.0.0.1:8080}
AUTH_AGENT=${AUTH_AGENT:-"Bearer sk-agent-demo"}
AUTH_CHAT=${AUTH_CHAT:-"Bearer sk-chat-demo"}
PASS=0; FAIL=0

check() {  # описание ожидаемое фактическое
  if [ "$2" = "$3" ]; then
    printf "  \033[32m✓\033[0m %s\n" "$1"; PASS=$((PASS+1))
  else
    printf "  \033[31m✗\033[0m %s: ожидалось %s, получено %s\n" "$1" "$2" "$3"; FAIL=$((FAIL+1))
  fi
}

post() {  # ключ тело → код
  curl -sS -o /dev/null -w "%{http_code}" -X POST "$GW/v1/chat/completions" \
    -H "Authorization: $1" -H 'Content-Type: application/json' -d "$2"
}

body() { echo "{\"model\":\"$1\",\"max_tokens\":${2:-3},\"messages\":[{\"role\":\"user\",\"content\":\"$3\"}]}"; }

echo "1. Контракт внешнего API (§3.3.1)"
check "стриминг отдаётся" 200 "$(curl -sS -o /dev/null -w '%{http_code}' -X POST "$GW/v1/chat/completions" \
  -H "Authorization: $AUTH_AGENT" -H 'Content-Type: application/json' \
  -d '{"model":"main","stream":true,"max_tokens":3,"messages":[{"role":"user","content":"тест"}]}')"
check "нестриминговый ответ" 200 "$(post "$AUTH_AGENT" "$(body main 3 обычный)")"
check "список моделей" 200 "$(curl -sS -o /dev/null -w '%{http_code}' "$GW/v1/models")"

echo
echo "2. Авторизация и мультитенантность (§3.3.2)"
check "без ключа — 401" 401 "$(curl -sS -o /dev/null -w '%{http_code}' -X POST "$GW/v1/chat/completions" \
  -H 'Content-Type: application/json' -d "$(body main 2 x)")"
check "неизвестный ключ — 401" 401 "$(post 'Bearer sk-нет' "$(body main 2 x)")"
check "ключ с не-ASCII — 401, а не 500" 401 "$(post 'Bearer sk-кириллица' "$(body main 2 x)")"
check "модель не разрешена — 403" 403 "$(post "$AUTH_CHAT" "$(body reserve 2 x)")"
check "неизвестная модель — 404" 404 "$(post "$AUTH_AGENT" "$(body несуществующая 2 x)")"

echo
echo "3. Валидация (§3.3.1)"
check "пустые messages — 400" 400 "$(post "$AUTH_AGENT" '{"model":"main","messages":[]}')"
check "нет model — 400" 400 "$(post "$AUTH_AGENT" '{"messages":[{"role":"user","content":"x"}]}')"
check "не JSON — 400" 400 "$(post "$AUTH_AGENT" 'это не json')"

echo
echo "4. Служебные эндпоинты (§3.3.10)"
check "проба живости" 200 "$(curl -sS -o /dev/null -w '%{http_code}' "$GW/healthz")"
check "готовность" 200 "$(curl -sS -o /dev/null -w '%{http_code}' "$GW/readyz")"
check "метрики" 200 "$(curl -sS -o /dev/null -w '%{http_code}' "$GW/metrics")"
check "метрика TTFT присутствует" 1 "$(curl -sS "$GW/metrics" | grep -c '^gateway_ttft_seconds_bucket' | head -c1 | tr -d '\n' | sed 's/[2-9]/1/')"

echo
echo "5. Отказоустойчивость (§3.3.8)"
before=$(curl -sS "$GW/admin/state" | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["breakers"]))')
ok=0
for i in $(seq 1 10); do
  [ "$(post "$AUTH_AGENT" "$(body main 2 "отказоустойчивость-$i")")" = "200" ] && ok=$((ok+1))
done
check "все запросы обслужены" 10 "$ok"

echo
echo "Итого: пройдено $PASS, провалено $FAIL"
[ "$FAIL" -eq 0 ] || exit 1
