package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// Контрольные опыты для прибора. Заглушка задаёт всю систему координат
// стенда: если она не реализует заявленную модель, каждый последующий
// замер гейтвея измеряет выдумку. Поэтому здесь проверяется не «работает
// ли ручка», а совпадают ли наблюдаемые величины с формулой.

func testConfig() *Config {
	return &Config{
		Addr:                ":0",
		ID:                  "test",
		PrefillA:            10 * time.Millisecond,
		PrefillB:            40 * time.Microsecond,
		PrefillC:            1500 * time.Microsecond,
		ITLBase:             2 * time.Millisecond,
		BatchPenalty:        0.06,
		MaxConcurrency:      4,
		MaxQueue:            64,
		BlockTokens:         256,
		CacheTTL:            time.Minute,
		CacheBlocks:         10000,
		DefaultOutputTokens: 4,
		Seed:                1,
	}
}

func newTestServer(cfg *Config) (*Server, *httptest.Server) {
	s := &Server{cfg: cfg, engine: NewEngine(cfg), rnd: rand.New(rand.NewSource(cfg.Seed)), startedAt: time.Now()}
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", s.handleChat)
	mux.HandleFunc("/state", s.handleState)
	return s, httptest.NewServer(mux)
}

// prompt строит текст ровно на n токенов по мере самого мока (4 байта = токен).
func prompt(n int, salt string) string {
	body := strings.Repeat("a", n*BytesPerToken)
	if salt != "" {
		return salt + body[len(salt):]
	}
	return body
}

type ttftResult struct {
	ttft    time.Duration
	headers http.Header
	tokens  int
}

func streamOnce(t *testing.T, url, content string, outTokens int) ttftResult {
	t.Helper()
	body, _ := json.Marshal(ChatRequest{
		Model:            "mock-llm",
		Messages:         []Message{{Role: "user", Content: content}},
		Stream:           true,
		MockOutputTokens: &outTokens,
	})
	start := time.Now()
	resp, err := http.Post(url+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("запрос не прошёл: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("статус %d", resp.StatusCode)
	}
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	var ttft time.Duration
	seen := 0
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") || line == "data: [DONE]" {
			continue
		}
		var ch ChatChunk
		if err := json.Unmarshal([]byte(line[6:]), &ch); err != nil {
			continue
		}
		if len(ch.Choices) > 0 && ch.Choices[0].Delta.Content != "" {
			if seen == 0 {
				ttft = time.Since(start)
			}
			seen++
		}
	}
	return ttftResult{ttft: ttft, headers: resp.Header, tokens: seen}
}

// 1. Модель префилла: наблюдаемый TTFT обязан совпасть с формулой.
// Это и есть проверка того, что прибор откалиброван, а не «примерно медленный».
func TestPrefillModelMatchesFormula(t *testing.T) {
	cfg := testConfig()
	cfg.ITLBase = time.Millisecond
	_, srv := newTestServer(cfg)
	defer srv.Close()

	for _, n := range []int{512, 4096, 16384} {
		want := cfg.prefillDuration(n)
		got := streamOnce(t, srv.URL, prompt(n, fmt.Sprintf("s%d", n)), 2)
		diff := got.ttft - want
		if diff < 0 {
			diff = -diff
		}
		// Допуск: планировщик Go и HTTP-стек добавляют единицы миллисекунд.
		if diff > 25*time.Millisecond {
			t.Errorf("n=%d: TTFT=%s, модель даёт %s, расхождение %s", n, got.ttft, want, diff)
		}
	}
}

// 2. Суперлинейность: удвоение длины входа должно давать более чем
// двукратный рост префилла. Если это не так, квадратичный член не работает,
// и все выводы про длинные контексты недействительны.
func TestPrefillIsSuperlinear(t *testing.T) {
	cfg := testConfig()
	t1 := cfg.prefillDuration(32000) - cfg.PrefillA
	t2 := cfg.prefillDuration(64000) - cfg.PrefillA
	if float64(t2) <= 2.0*float64(t1) {
		t.Fatalf("префилл не суперлинеен: 32k=%s, 64k=%s", t1, t2)
	}
}

// 3. Кэш: повтор того же промпта обязан дать попадание всех полных блоков
// и резкое падение TTFT. Без этого cache-aware роутинг нечем мерить.
func TestPrefixCacheHitReducesTTFT(t *testing.T) {
	cfg := testConfig()
	cfg.ITLBase = time.Millisecond
	_, srv := newTestServer(cfg)
	defer srv.Close()

	p := prompt(8192, "cache-test")
	cold := streamOnce(t, srv.URL, p, 2)
	warm := streamOnce(t, srv.URL, p, 2)

	if cold.headers.Get("X-Mock-Cached-Blocks") != "0" {
		t.Fatalf("холодный запрос сообщил попадание: %s", cold.headers.Get("X-Mock-Cached-Blocks"))
	}
	wantBlocks := 8192 / cfg.BlockTokens
	if got := warm.headers.Get("X-Mock-Cached-Blocks"); got != strconv.Itoa(wantBlocks) {
		t.Fatalf("повтор дал %s блоков, ожидалось %d", got, wantBlocks)
	}
	if warm.ttft >= cold.ttft {
		t.Fatalf("попадание в кэш не ускорило префилл: холодный %s, тёплый %s", cold.ttft, warm.ttft)
	}
}

// 4. Частичное совпадение: следующий ход сессии дописывает историю,
// поэтому обязан попасть в кэш ровно на длину общего префикса.
// Это тот самый механизм, ради которого строится сессионный роутинг (§3.3.6.3a).
func TestPartialPrefixMatch(t *testing.T) {
	cfg := testConfig()
	cfg.ITLBase = time.Millisecond
	_, srv := newTestServer(cfg)
	defer srv.Close()

	base := prompt(4096, "session-1")
	streamOnce(t, srv.URL, base, 1)
	// Ход 2: та же история плюс продолжение.
	next := base + prompt(1024, "")
	got := streamOnce(t, srv.URL, next, 1)

	wantBlocks := 4096 / cfg.BlockTokens
	if g := got.headers.Get("X-Mock-Cached-Blocks"); g != strconv.Itoa(wantBlocks) {
		t.Fatalf("частичное совпадение дало %s блоков, ожидалось %d", g, wantBlocks)
	}
}

// 5. Расхождение истории обязано ломать попадание с точки расхождения.
// Проверяет, что хеширование действительно цепочечное: если бы блоки
// хешировались независимо, совпадение «перепрыгнуло» бы разрыв.
func TestChainedHashingBreaksOnDivergence(t *testing.T) {
	cfg := testConfig()
	_, srv := newTestServer(cfg)
	defer srv.Close()

	a := prompt(2048, "chain-a")
	streamOnce(t, srv.URL, a, 1)
	// Меняем ровно второй блок, остальное совпадает.
	b := []byte(a)
	b[cfg.BlockTokens*BytesPerToken+5] = 'Z'
	got := streamOnce(t, srv.URL, string(b), 1)
	if g := got.headers.Get("X-Mock-Cached-Blocks"); g != "1" {
		t.Fatalf("после расхождения во втором блоке совпало %s блоков, ожидался 1", g)
	}
}

// 6. Ёмкость батча и очередь: сверх MaxConcurrency запросы ждут, а не
// обрабатываются параллельно. Иначе перегрузки на стенде не воспроизвести.
func TestConcurrencyLimitQueues(t *testing.T) {
	cfg := testConfig()
	cfg.MaxConcurrency = 2
	cfg.PrefillA = 5 * time.Millisecond
	cfg.ITLBase = 20 * time.Millisecond
	s, srv := newTestServer(cfg)
	defer srv.Close()

	var wg sync.WaitGroup
	for i := 0; i < 6; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			streamOnce(t, srv.URL, prompt(256, fmt.Sprintf("c%d", i)), 10)
		}(i)
	}
	time.Sleep(120 * time.Millisecond)
	if r := s.engine.running.Load(); r > int64(cfg.MaxConcurrency) {
		t.Fatalf("в батче %d запросов при лимите %d", r, cfg.MaxConcurrency)
	}
	if q := s.engine.queued.Load(); q == 0 {
		t.Fatal("очередь пуста, хотя запросов больше ёмкости батча")
	}
	wg.Wait()
}

// 7. Переполнение очереди отвечает 429 с Retry-After — контракт отказа (§3.3.3.6).
// Фоновые запросы обязаны *дочитывать* стрим: закрытие тела сразу после
// заголовков отменяет запрос и освобождает слот, и тогда очередь не набьётся.
func TestQueueOverflowReturns429(t *testing.T) {
	cfg := testConfig()
	cfg.MaxConcurrency = 1
	cfg.MaxQueue = 1
	cfg.ITLBase = 100 * time.Millisecond
	cfg.DefaultOutputTokens = 20
	s, srv := newTestServer(cfg)
	defer srv.Close()

	ctx, cancelHolders := context.WithCancel(context.Background())
	defer cancelHolders()
	var wg sync.WaitGroup
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			body, _ := json.Marshal(ChatRequest{Model: "m", Stream: true,
				Messages: []Message{{Role: "user", Content: prompt(256, fmt.Sprintf("q%d", i))}}})
			req, _ := http.NewRequestWithContext(ctx, http.MethodPost,
				srv.URL+"/v1/chat/completions", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				return
			}
			defer resp.Body.Close()
			_, _ = io.Copy(io.Discard, resp.Body) // держим стрим открытым
		}(i)
	}

	// Ждём, пока очередь действительно набьётся, вместо слепого sleep.
	deadline := time.Now().Add(3 * time.Second)
	for s.engine.queued.Load() < int64(cfg.MaxQueue) {
		if time.Now().After(deadline) {
			t.Fatalf("очередь не набилась: running=%d queued=%d",
				s.engine.running.Load(), s.engine.queued.Load())
		}
		time.Sleep(5 * time.Millisecond)
	}

	body, _ := json.Marshal(ChatRequest{Model: "m", Stream: true,
		Messages: []Message{{Role: "user", Content: prompt(256, "overflow")}}})
	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("запрос не прошёл: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("ожидался 429, получен %d", resp.StatusCode)
	}
	if resp.Header.Get("Retry-After") == "" {
		t.Error("429 без Retry-After: клиент не знает, когда повторять")
	}
	cancelHolders()
	wg.Wait()
}

// 8. Отмена клиентом освобождает слот. Это проверка на зомби-запросы (§3.3.9):
// без неё брошенные генерации продолжают жечь ёмкость, и утечка проявится
// только под нагрузкой, когда искать будет некогда.
func TestClientDisconnectReleasesSlot(t *testing.T) {
	cfg := testConfig()
	cfg.MaxConcurrency = 1
	cfg.ITLBase = 50 * time.Millisecond
	s, srv := newTestServer(cfg)
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	body, _ := json.Marshal(ChatRequest{Model: "m", Stream: true,
		Messages: []Message{{Role: "user", Content: prompt(256, "cancel")}}})
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, srv.URL+"/v1/chat/completions",
		bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("запрос не прошёл: %v", err)
	}
	buf := make([]byte, 128)
	_, _ = resp.Body.Read(buf)
	if r := s.engine.running.Load(); r != 1 {
		t.Fatalf("ожидался 1 запрос в батче, получено %d", r)
	}
	cancel()
	resp.Body.Close()

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if s.engine.running.Load() == 0 {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("слот не освобождён после разрыва: running=%d", s.engine.running.Load())
}

// 9. TTL: протухший блок не считается попаданием. Обоснование TTL в минутах
// (§5.3) имеет смысл только если вытеснение действительно работает.
func TestCacheTTLExpiry(t *testing.T) {
	c := NewKVCache(1000, 40*time.Millisecond)
	h := BlockHashes(prompt(1024, "ttl"), 256)
	c.Insert(h)
	if got := c.MatchPrefix(h); got != len(h) {
		t.Fatalf("сразу после вставки совпало %d из %d", got, len(h))
	}
	time.Sleep(80 * time.Millisecond)
	if got := c.MatchPrefix(h); got != 0 {
		t.Fatalf("после истечения TTL совпало %d блоков", got)
	}
}

// 10. Вытеснение по ёмкости.
func TestCacheCapacityEviction(t *testing.T) {
	c := NewKVCache(4, time.Minute)
	first := BlockHashes(prompt(1024, "evict-a"), 256) // 4 блока
	c.Insert(first)
	second := BlockHashes(prompt(1024, "evict-b"), 256)
	c.Insert(second)
	if got := c.MatchPrefix(first); got != 0 {
		t.Fatalf("старые блоки не вытеснены: совпало %d", got)
	}
	if got := c.MatchPrefix(second); got != len(second) {
		t.Fatalf("свежие блоки потеряны: совпало %d из %d", got, len(second))
	}
}
