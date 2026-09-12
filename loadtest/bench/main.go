// bench — измеритель для стенда: гонит SSE-запросы и считает TTFT/ITL/goodput.
//
// На Go по той же причине, что и мок: измеритель обязан иметь запас по
// производительности относительно измеряемого. Клиент на Python при тысяче
// одновременных стримов мерил бы сам себя.
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type opts struct {
	URL          string
	Concurrency  int
	Requests     int
	Duration     time.Duration
	PromptTokens int
	OutputTokens int
	SharedPrefix float64
	Warmup       int
	Model        string
	JSONOut      string
	Salt         string
}

func main() {
	o := &opts{}
	flag.StringVar(&o.URL, "url", "http://127.0.0.1:9001/v1/chat/completions", "эндпоинт")
	flag.IntVar(&o.Concurrency, "c", 32, "число одновременных запросов")
	flag.IntVar(&o.Requests, "n", 0, "сколько запросов сделать (0 — по времени)")
	flag.DurationVar(&o.Duration, "d", 15*time.Second, "длительность, если -n не задан")
	flag.IntVar(&o.PromptTokens, "prompt-tokens", 2048, "длина входа в токенах")
	flag.IntVar(&o.OutputTokens, "output-tokens", 64, "длина ответа в токенах")
	flag.Float64Var(&o.SharedPrefix, "shared-prefix", 0, "доля входа, общая у всех запросов (0..1)")
	flag.IntVar(&o.Warmup, "warmup", 0, "сколько первых запросов исключить из статистики (§5.1)")
	flag.StringVar(&o.Model, "model", "mock-llm", "имя модели")
	flag.StringVar(&o.JSONOut, "json", "", "файл для машинночитаемого отчёта")
	// Соль обязательна при сравнении прогонов: KV-кэш апстрима переживает
	// прогон, и повтор тех же промптов измеряет тёплый кэш, а не систему.
	// Пустая соль означает «взять текущее время», то есть всегда холодный кэш.
	flag.StringVar(&o.Salt, "salt", "", "префикс, разделяющий прогоны по кэшу апстрима")
	flag.Parse()
	if o.Salt == "" {
		o.Salt = fmt.Sprintf("r%d", time.Now().UnixNano())
	}

	res := run(o)
	res.report(o)
}

type sample struct {
	ttft    time.Duration
	total   time.Duration
	tokens  int
	status  int
	err     bool
	started time.Time
}

type results struct {
	samples []sample
	mu      sync.Mutex

	sent     atomic.Int64
	errs     atomic.Int64
	statuses sync.Map // int -> *atomic.Int64

	wall time.Duration
}

func (r *results) add(s sample) {
	r.mu.Lock()
	r.samples = append(r.samples, s)
	r.mu.Unlock()
	v, _ := r.statuses.LoadOrStore(s.status, &atomic.Int64{})
	v.(*atomic.Int64).Add(1)
}

func run(o *opts) *results {
	tr := &http.Transport{
		MaxIdleConns:        o.Concurrency * 2,
		MaxIdleConnsPerHost: o.Concurrency * 2,
		MaxConnsPerHost:     0,
		IdleConnTimeout:     90 * time.Second,
		DisableCompression:  true, // сжатие исказило бы замер межтокенных задержек
	}
	client := &http.Client{Transport: tr, Timeout: 0}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if o.Requests == 0 {
		var stop context.CancelFunc
		ctx, stop = context.WithTimeout(ctx, o.Duration)
		defer stop()
	}

	res := &results{}
	var counter atomic.Int64
	start := time.Now()

	var wg sync.WaitGroup
	for w := 0; w < o.Concurrency; w++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			for {
				if ctx.Err() != nil {
					return
				}
				i := counter.Add(1) - 1
				if o.Requests > 0 && i >= int64(o.Requests) {
					return
				}
				res.add(one(ctx, client, o, i))
				res.sent.Add(1)
			}
		}(w)
	}
	wg.Wait()
	res.wall = time.Since(start)
	return res
}

// buildPrompt формирует вход с заданной долей общего префикса. Это рычаг,
// которым на стенде задаётся переиспользуемость кэша: 0 — все запросы
// уникальны (профиль «нулевого кэша» из §5.2), 0.8 — агентский профиль.
func buildPrompt(o *opts, i int64) string {
	total := o.PromptTokens * 4
	shared := int(float64(total) * o.SharedPrefix)
	if shared > total {
		shared = total
	}
	var b strings.Builder
	b.Grow(total)
	if shared > 0 {
		head := "|" + o.Salt + "-shared|"
		if len(head) > shared {
			head = head[:shared]
		}
		b.WriteString(head)
		b.WriteString(strings.Repeat("S", shared-len(head)))
	}
	tail := total - shared
	if tail > 0 {
		seed := fmt.Sprintf("|%s-req-%d|", o.Salt, i)
		b.WriteString(seed)
		if n := tail - len(seed); n > 0 {
			b.WriteString(strings.Repeat("u", n))
		}
	}
	s := b.String()
	if len(s) > total {
		s = s[:total]
	}
	return s
}

func one(ctx context.Context, client *http.Client, o *opts, i int64) sample {
	payload := map[string]any{
		"model":              o.Model,
		"stream":             true,
		"mock_output_tokens": o.OutputTokens,
		"messages": []map[string]string{
			{"role": "user", "content": buildPrompt(o, i)},
		},
	}
	body, _ := json.Marshal(payload)

	s := sample{started: time.Now()}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.URL, bytes.NewReader(body))
	if err != nil {
		s.err = true
		return s
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		s.err = true
		s.total = time.Since(s.started)
		return s
	}
	defer resp.Body.Close()
	s.status = resp.StatusCode
	if resp.StatusCode != http.StatusOK {
		s.total = time.Since(s.started)
		return s
	}

	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 1<<16), 1<<20)
	for sc.Scan() {
		line := sc.Bytes()
		if !bytes.HasPrefix(line, []byte("data: ")) {
			continue
		}
		payload := line[6:]
		if bytes.Equal(payload, []byte("[DONE]")) {
			break
		}
		// Считаем только кадры с содержимым: кадр с ролью — не токен.
		if !bytes.Contains(payload, []byte(`"content":"`)) {
			continue
		}
		if s.tokens == 0 {
			s.ttft = time.Since(s.started)
		}
		s.tokens++
	}
	s.total = time.Since(s.started)
	return s
}

func pct(sorted []time.Duration, p float64) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(math.Ceil(p/100*float64(len(sorted)))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

type report struct {
	Requests   int              `json:"requests"`
	OK         int              `json:"ok"`
	Errors     int              `json:"errors"`
	WallSec    float64          `json:"wall_sec"`
	RPS        float64          `json:"rps"`
	TokensOut  int              `json:"tokens_out"`
	TPS        float64          `json:"tokens_per_sec"`
	TTFTp50Ms  float64          `json:"ttft_p50_ms"`
	TTFTp95Ms  float64          `json:"ttft_p95_ms"`
	TTFTp99Ms  float64          `json:"ttft_p99_ms"`
	TTFTMaxMs  float64          `json:"ttft_max_ms"`
	ITLMeanMs  float64          `json:"itl_mean_ms"`
	TotalP95Ms float64          `json:"total_p95_ms"`
	Statuses   map[string]int64 `json:"statuses"`
}

func (r *results) report(o *opts) {
	var ok []sample
	for i, s := range r.samples {
		if i < o.Warmup {
			continue
		}
		if !s.err && s.status == http.StatusOK && s.tokens > 0 {
			ok = append(ok, s)
		}
	}
	ttfts := make([]time.Duration, 0, len(ok))
	totals := make([]time.Duration, 0, len(ok))
	var itlSum float64
	var tokens int
	for _, s := range ok {
		ttfts = append(ttfts, s.ttft)
		totals = append(totals, s.total)
		tokens += s.tokens
		if s.tokens > 1 {
			itlSum += float64(s.total-s.ttft) / float64(s.tokens-1) / float64(time.Millisecond)
		}
	}
	sort.Slice(ttfts, func(i, j int) bool { return ttfts[i] < ttfts[j] })
	sort.Slice(totals, func(i, j int) bool { return totals[i] < totals[j] })

	statuses := map[string]int64{}
	r.statuses.Range(func(k, v any) bool {
		statuses[fmt.Sprintf("%d", k.(int))] = v.(*atomic.Int64).Load()
		return true
	})

	ms := func(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }
	rep := report{
		Requests:   len(r.samples),
		OK:         len(ok),
		Errors:     int(r.errs.Load()),
		WallSec:    r.wall.Seconds(),
		RPS:        float64(len(ok)) / r.wall.Seconds(),
		TokensOut:  tokens,
		TPS:        float64(tokens) / r.wall.Seconds(),
		TTFTp50Ms:  ms(pct(ttfts, 50)),
		TTFTp95Ms:  ms(pct(ttfts, 95)),
		TTFTp99Ms:  ms(pct(ttfts, 99)),
		TTFTMaxMs:  ms(pct(ttfts, 100)),
		TotalP95Ms: ms(pct(totals, 95)),
		Statuses:   statuses,
	}
	if len(ok) > 0 {
		rep.ITLMeanMs = itlSum / float64(len(ok))
	}

	fmt.Printf("запросов: %d (успешных %d)  за %.1f с   RPS=%.1f  TPS=%.0f\n",
		rep.Requests, rep.OK, rep.WallSec, rep.RPS, rep.TPS)
	fmt.Printf("TTFT  p50=%.1f  p95=%.1f  p99=%.1f  max=%.1f мс\n",
		rep.TTFTp50Ms, rep.TTFTp95Ms, rep.TTFTp99Ms, rep.TTFTMaxMs)
	fmt.Printf("ITL   среднее=%.2f мс      полное время p95=%.1f мс\n", rep.ITLMeanMs, rep.TotalP95Ms)
	fmt.Printf("коды: %v\n", statuses)

	if o.JSONOut != "" {
		f, err := os.Create(o.JSONOut)
		if err == nil {
			defer f.Close()
			enc := json.NewEncoder(f)
			enc.SetIndent("", "  ")
			_ = enc.Encode(rep)
		}
	}
}
