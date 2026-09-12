// mock-upstream — управляемая заглушка LLM-апстрима для нагрузочного стенда
// (docs/architecture.md §5).
//
// Написан на Go сознательно: прибор обязан держать нагрузку с многократным
// запасом относительно измеряемой системы. Если заглушка захлебнётся раньше
// гейтвея, все графики будут измерять заглушку, а не решение.
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

type Server struct {
	cfg       *Config
	engine    *Engine
	rndMu     sync.Mutex
	rnd       *rand.Rand
	startedAt time.Time
}

func main() {
	cfg := parseFlags()
	s := &Server{
		cfg:       cfg,
		engine:    NewEngine(cfg),
		rnd:       rand.New(rand.NewSource(cfg.Seed)),
		startedAt: time.Now(),
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", s.handleChat)
	mux.HandleFunc("/v1/models", s.handleModels)
	mux.HandleFunc("/healthz", s.handleHealth)
	mux.HandleFunc("/state", s.handleState)
	mux.HandleFunc("/metrics", s.handleMetrics)
	srv := &http.Server{
		Addr:    cfg.Addr,
		Handler: mux,
		// Таймаут на запись не ставим: стрим длинного ответа легитимно живёт
		// минуты. Ограничение времени — задача гейтвея, а не апстрима.
		ReadHeaderTimeout: 10 * time.Second,
	}
	log.Printf("mock-upstream %s слушает %s (батч=%d, очередь=%d, itl=%s)",
		cfg.ID, cfg.Addr, cfg.MaxConcurrency, cfg.MaxQueue, cfg.ITLBase)
	if err := srv.ListenAndServe(); err != nil {
		log.Fatal(err)
	}
}
func (s *Server) chance(p float64) bool {
	if p <= 0 {
		return false
	}
	s.rndMu.Lock()
	defer s.rndMu.Unlock()
	return s.rnd.Float64() < p
}
func writeErr(w http.ResponseWriter, status int, msg, typ string) {
	w.Header().Set("Content-Type", "application/json")
	if status == http.StatusTooManyRequests {
		w.Header().Set("Retry-After", "1")
	}
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(APIError{APIErrorBody{Message: msg, Type: typ, Code: strconv.Itoa(status)}})
}
func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeErr(w, http.StatusMethodNotAllowed, "только POST", "invalid_request_error")
		return
	}
	var req ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, "не разобрано тело: "+err.Error(), "invalid_request_error")
		return
	}
	// --- Инъекция отказов: до всякой работы, как и положено раннему отказу ---
	if s.cfg.Overloaded {
		writeErr(w, http.StatusTooManyRequests, "апстрим перегружен", "rate_limit_error")
		return
	}
	if s.chance(s.cfg.ErrorRate) {
		writeErr(w, http.StatusServiceUnavailable, "инъекция отказа", "server_error")
		return
	}
	if s.chance(s.cfg.HangRate) {
		// Зависание без единого байта: проверяет раздельный таймаут на TTFT.
		// Один общий таймаут такое не ловит — см. §3.3.8.
		_ = sleepCtx(r.Context(), s.cfg.HangFor)
		return
	}
	prompt := flattenPrompt(req.Messages)
	adm, err := s.engine.Acquire(r.Context(), prompt)
	if err != nil {
		switch err {
		case ErrQueueFull:
			writeErr(w, http.StatusTooManyRequests, "очередь апстрима переполнена", "rate_limit_error")
		case ErrCanceled:
			// Клиент ушёл — отвечать некому, слот уже освобождён.
		default:
			writeErr(w, http.StatusInternalServerError, err.Error(), "server_error")
		}
		return
	}
	defer adm.Release()
	outTokens := s.cfg.DefaultOutputTokens
	if req.MockOutputTokens != nil && *req.MockOutputTokens > 0 {
		outTokens = *req.MockOutputTokens
	} else if req.MaxTokens != nil && *req.MaxTokens > 0 {
		outTokens = *req.MaxTokens
	}
	// Заголовки наблюдаемости: дают гейтвею и стенду сверить собственную
	// оценку TTFT с фактическим устройством апстрима. Это контрольный опыт
	// для калибровки (§3.3.6.7a), а не декорация.
	h := w.Header()
	h.Set("X-Mock-Instance", s.cfg.ID)
	h.Set("X-Mock-Prompt-Tokens", strconv.Itoa(adm.PromptTokens))
	h.Set("X-Mock-Cached-Blocks", strconv.Itoa(adm.CachedBlocks))
	h.Set("X-Mock-Cached-Tokens", strconv.Itoa(adm.CachedTokens))
	h.Set("X-Mock-Queue-Wait-Ms", strconv.FormatInt(adm.QueueWait.Milliseconds(), 10))
	h.Set("X-Mock-Prefill-Ms", strconv.FormatInt(adm.PrefillTime.Milliseconds(), 10))
	if req.Stream {
		s.streamResponse(w, r, &req, adm, outTokens)
		return
	}
	s.blockingResponse(w, r, &req, adm, outTokens)
}

// flattenPrompt склеивает переписку в одну строку. Порядок и разделители
// фиксированы: от этого зависит устойчивость блочного хеширования, а значит
// и воспроизводимость попаданий в кэш между ходами одной сессии.
func flattenPrompt(msgs []Message) string {
	var b strings.Builder
	for _, m := range msgs {
		b.WriteString(m.Role)
		b.WriteByte('\n')
		b.WriteString(m.Content)
		b.WriteByte('\n')
	}
	return b.String()
}
func (s *Server) streamResponse(w http.ResponseWriter, r *http.Request, req *ChatRequest, adm *Admission, outTokens int) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeErr(w, http.StatusInternalServerError, "стриминг не поддерживается", "server_error")
		return
	}
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no") // иначе nginx буферизует стрим и ломает TTFT
	w.WriteHeader(http.StatusOK)
	id := "chatcmpl-" + strconv.FormatInt(time.Now().UnixNano(), 36)
	created := time.Now().Unix()
	ctx := r.Context()
	enc := json.NewEncoder(w)
	send := func(c *ChatChunk) bool {
		if _, err := w.Write([]byte("data: ")); err != nil {
			return false
		}
		if err := enc.Encode(c); err != nil {
			return false
		}
		if _, err := w.Write([]byte("\n")); err != nil {
			return false
		}
		flusher.Flush()
		return true
	}
	// Первый кадр — роль, без содержимого: так делает OpenAI, и клиенты
	// на это рассчитывают.
	if !send(&ChatChunk{ID: id, Object: "chat.completion.chunk", Created: created, Model: req.Model,
		Choices: []StreamChoice{{Index: 0, Delta: Delta{Role: "assistant"}}}}) {
		return
	}
	emitted := 0
	for i := 0; i < outTokens; i++ {
		if err := sleepCtx(ctx, s.engine.NextTokenDelay()); err != nil {
			return // клиент отвалился — слот освободит defer Release
		}
		if !send(&ChatChunk{ID: id, Object: "chat.completion.chunk", Created: created, Model: req.Model,
			Choices: []StreamChoice{{Index: 0, Delta: Delta{Content: tokenText(i)}}}}) {
			return
		}
		s.engine.CountOutToken()
		emitted++
	}
	stop := "stop"
	if !send(&ChatChunk{ID: id, Object: "chat.completion.chunk", Created: created, Model: req.Model,
		Choices: []StreamChoice{{Index: 0, Delta: Delta{}, FinishReason: &stop}}}) {
		return
	}
	if req.StreamOptions != nil && req.StreamOptions.IncludeUsage {
		if !send(&ChatChunk{ID: id, Object: "chat.completion.chunk", Created: created, Model: req.Model,
			Choices: []StreamChoice{},
			Usage: &Usage{PromptTokens: adm.PromptTokens, CompletionTokens: emitted,
				TotalTokens: adm.PromptTokens + emitted}}) {
			return
		}
	}
	_, _ = w.Write([]byte("data: [DONE]\n\n"))
	flusher.Flush()
}
func (s *Server) blockingResponse(w http.ResponseWriter, r *http.Request, req *ChatRequest, adm *Admission, outTokens int) {
	ctx := r.Context()
	var b strings.Builder
	for i := 0; i < outTokens; i++ {
		if err := sleepCtx(ctx, s.engine.NextTokenDelay()); err != nil {
			return
		}
		b.WriteString(tokenText(i))
		s.engine.CountOutToken()
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(ChatResponse{
		ID:      "chatcmpl-" + strconv.FormatInt(time.Now().UnixNano(), 36),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []Choice{{Index: 0, Message: Message{Role: "assistant", Content: b.String()}, FinishReason: "stop"}},
		Usage: Usage{PromptTokens: adm.PromptTokens, CompletionTokens: outTokens,
			TotalTokens: adm.PromptTokens + outTokens},
	})
}

// tokenText — содержимое токена. Детерминированное и недлинное: нам важен
// ритм стрима, а не осмысленность текста.
func tokenText(i int) string {
	if i%12 == 11 {
		return "\n"
	}
	return mockWords[i%len(mockWords)] + " "
}

var mockWords = []string{"поток", "токен", "ответ", "модель", "контекст", "префикс",
	"кэш", "очередь", "батч", "префилл", "декод", "латентность"}

func (s *Server) handleModels(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"object": "list",
		"data": []map[string]any{
			{"id": "mock-llm", "object": "model", "owned_by": "stand"},
		},
	})
}
func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("ok\n"))
}

// handleState отдаёт то, что в реальности видно только у self-hosted движка:
// pending prefill-токены и наличие необслуженной очереди. Это сигналы для
// сценария 1 из §3.3.6.8 — с ними роутер может работать как DualMap.
// Держим их отдельным эндпоинтом, чтобы честно уметь их отключать и
// проверять деградацию до сценария 2.
func (s *Server) handleState(w http.ResponseWriter, _ *http.Request) {
	blocks, hits, misses := s.engine.cache.Stats()
	queued := s.engine.queued.Load()
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"instance":               s.cfg.ID,
		"running":                s.engine.running.Load(),
		"queued":                 queued,
		"has_pending_queue":      queued > 0, // бинарный признак занятости (§3.3.3.7)
		"pending_prefill_tokens": s.engine.pendingPrefill.Load(),
		"max_concurrency":        s.cfg.MaxConcurrency,
		"cache_blocks":           blocks,
		"cache_block_hits":       hits,
		"cache_block_misses":     misses,
		"uptime_sec":             int(time.Since(s.startedAt).Seconds()),
	})
}
func (s *Server) handleMetrics(w http.ResponseWriter, _ *http.Request) {
	blocks, hits, misses := s.engine.cache.Stats()
	lbl := fmt.Sprintf("{instance=%q}", s.cfg.ID)
	var b strings.Builder
	m := func(name, typ, help string, v any) {
		fmt.Fprintf(&b, "# HELP %s %s\n# TYPE %s %s\n%s%s %v\n", name, help, name, typ, name, lbl, v)
	}
	m("mock_running_requests", "gauge", "запросов в батче", s.engine.running.Load())
	m("mock_queued_requests", "gauge", "запросов в очереди перед батчем", s.engine.queued.Load())
	m("mock_pending_prefill_tokens", "gauge", "токенов, ожидающих префилла", s.engine.pendingPrefill.Load())
	m("mock_requests_total", "counter", "принятых запросов", s.engine.totalRequests.Load())
	m("mock_rejected_total", "counter", "отклонённых по переполнению очереди", s.engine.totalRejected.Load())
	m("mock_canceled_total", "counter", "отменённых клиентом", s.engine.totalCanceled.Load())
	m("mock_tokens_in_total", "counter", "входных токенов", s.engine.totalTokensIn.Load())
	m("mock_tokens_out_total", "counter", "выходных токенов", s.engine.totalTokensOut.Load())
	m("mock_cache_blocks", "gauge", "блоков в KV-кэше", blocks)
	m("mock_cache_block_hits_total", "counter", "попаданий блоков", hits)
	m("mock_cache_block_misses_total", "counter", "промахов блоков", misses)
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	_, _ = w.Write([]byte(b.String()))
}
