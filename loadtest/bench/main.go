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
	"math/rand"
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
	Auth         string

	// --- Сессионный режим (§5.3) ---
	Sessions int     // сколько сессий держать живыми одновременно
	Rate     float64 // интенсивность, запросов/с; 0 — «как получится»
	Poisson  bool    // пуассоновский поток вместо равномерного
	Mix      string  // смесь классов: "agent=Bearer k1:0.7,batch=Bearer k2:0.3"
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
	flag.StringVar(&o.Auth, "auth", "", "значение заголовка Authorization (для гейтвея)")
	flag.IntVar(&o.Sessions, "sessions", 0, "число одновременно живых агентских сессий (§5.3); 0 — простой профиль")
	flag.Float64Var(&o.Rate, "rate", 0, "целевая интенсивность, запросов/с (0 — без ограничения)")
	flag.BoolVar(&o.Poisson, "poisson", true, "пуассоновский поток вместо равномерного")
	flag.StringVar(&o.Mix, "mix", "", "смесь классов: \"ключ1:доля,ключ2:доля\" — для графика приоритизации")
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
	label   string // класс обслуживания, если задана смесь
	turn    int
}

type results struct {
	samples []sample
	mu      sync.Mutex
	pool    *SessionPool

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

// mixEntry — один класс обслуживания в смеси.
type mixEntry struct {
	auth  string
	label string
	share float64
}

// parseMix разбирает "Bearer k1@agent:0.7,Bearer k2@batch:0.3".
// Смесь нужна для третьего графика защиты (§3.3.10): под нагрузкой
// interactive обязан держать SLO, а batch — растягиваться. На
// однородном трафике этого не увидеть.
func parseMix(spec string) []mixEntry {
	if spec == "" {
		return nil
	}
	var out []mixEntry
	total := 0.0
	for _, part := range strings.Split(spec, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		colon := strings.LastIndex(part, ":")
		if colon < 0 {
			continue
		}
		var share float64
		fmt.Sscanf(part[colon+1:], "%f", &share)
		head := part[:colon]
		label := head
		if at := strings.LastIndex(head, "@"); at >= 0 {
			label = head[at+1:]
			head = head[:at]
		}
		out = append(out, mixEntry{auth: head, label: label, share: share})
		total += share
	}
	// Нормируем доли: пусть пользователь пишет хоть проценты, хоть веса.
	if total > 0 {
		for i := range out {
			out[i].share /= total
		}
	}
	return out
}

func pickMix(mix []mixEntry, rnd *rand.Rand) mixEntry {
	u := rnd.Float64()
	acc := 0.0
	for _, e := range mix {
		acc += e.share
		if u <= acc {
			return e
		}
	}
	return mix[len(mix)-1]
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

	mix := parseMix(o.Mix)
	var pool *SessionPool
	if o.Sessions > 0 {
		pool = NewSessionPool(o.Sessions, DefaultSessionProfile(), 1, o.Salt)
		res.pool = pool
	}

	// Ограничитель интенсивности. Без него измеряется не система под
	// заданной нагрузкой, а то, сколько получилось выжать, — и кривая
	// «метрика против RPS» не строится (§5.2).
	var ticket chan struct{}
	if o.Rate > 0 {
		ticket = make(chan struct{}, o.Concurrency)
		go func() {
			rnd := rand.New(rand.NewSource(7))
			for {
				var d time.Duration
				if o.Poisson {
					d = PoissonInterval(rnd, o.Rate)
				} else {
					d = time.Duration(float64(time.Second) / o.Rate)
				}
				t := time.NewTimer(d)
				select {
				case <-t.C:
				case <-ctx.Done():
					t.Stop()
					close(ticket)
					return
				}
				select {
				case ticket <- struct{}{}:
				default: // воркеры не успевают — пропускаем такт
				}
			}
		}()
	}

	var wg sync.WaitGroup
	for w := 0; w < o.Concurrency; w++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			rnd := rand.New(rand.NewSource(int64(worker) + 1))
			for {
				if ctx.Err() != nil {
					return
				}
				if ticket != nil {
					select {
					case _, ok := <-ticket:
						if !ok {
							return
						}
					case <-ctx.Done():
						return
					}
				}
				i := counter.Add(1) - 1
				if o.Requests > 0 && i >= int64(o.Requests) {
					return
				}

				auth, label := o.Auth, ""
				if len(mix) > 0 {
					e := pickMix(mix, rnd)
					auth, label = e.auth, e.label
				}

				var s sample
				if pool != nil {
					sess, msgs := pool.Take()
					s = oneSession(ctx, client, o, sess, msgs, auth)
				} else {
					s = one(ctx, client, o, i, auth)
				}
				s.label = label
				res.add(s)
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

func one(ctx context.Context, client *http.Client, o *opts, i int64, auth string) sample {
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
	return doRequest(ctx, client, o, body, auth, s)
}

func doRequest(ctx context.Context, client *http.Client, o *opts,
	body []byte, auth string, s sample) sample {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.URL, bytes.NewReader(body))
	if err != nil {
		s.err = true
		return s
	}
	req.Header.Set("Content-Type", "application/json")
	if auth != "" {
		req.Header.Set("Authorization", auth)
	}

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

// oneSession отправляет очередной ход живой сессии.
func oneSession(ctx context.Context, client *http.Client, o *opts,
	sess *Session, msgs []Msg, auth string) sample {
	payload := map[string]any{
		"model":  o.Model,
		"stream": true,
		// Длина ответа фиксируется принудительно: если она плавает,
		// история следующего хода перестаёт совпадать с закэшированной
		// и паттерн переиспользования ломается (§5.2).
		"mock_output_tokens": o.OutputTokens,
		"max_tokens":         o.OutputTokens,
		"messages":           msgs,
	}
	body, _ := json.Marshal(payload)
	s := sample{started: time.Now(), turn: sess.Turn()}
	return doRequest(ctx, client, o, body, auth, s)
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
	Requests   int                    `json:"requests"`
	OK         int                    `json:"ok"`
	Errors     int                    `json:"errors"`
	WallSec    float64                `json:"wall_sec"`
	RPS        float64                `json:"rps"`
	TokensOut  int                    `json:"tokens_out"`
	TPS        float64                `json:"tokens_per_sec"`
	TTFTp50Ms  float64                `json:"ttft_p50_ms"`
	TTFTp95Ms  float64                `json:"ttft_p95_ms"`
	TTFTp99Ms  float64                `json:"ttft_p99_ms"`
	TTFTMaxMs  float64                `json:"ttft_max_ms"`
	ITLMeanMs  float64                `json:"itl_mean_ms"`
	TotalP95Ms float64                `json:"total_p95_ms"`
	Statuses   map[string]int64       `json:"statuses"`
	ByClass    map[string]classReport `json:"by_class,omitempty"`
}

type classReport struct {
	Requests  int     `json:"requests"`
	TTFTp50Ms float64 `json:"ttft_p50_ms"`
	TTFTp95Ms float64 `json:"ttft_p95_ms"`
	TTFTp99Ms float64 `json:"ttft_p99_ms"`
}

func groupByLabel(samples []sample) map[string][]sample {
	out := map[string][]sample{}
	for _, s := range samples {
		if s.label != "" {
			out[s.label] = append(out[s.label], s)
		}
	}
	return out
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
		ByClass:    map[string]classReport{},
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

	// Разбивка по классам обслуживания. Ради неё и существует режим
	// смеси: доказательство осмысленности архитектуры состоит в том,
	// что под одной и той же нагрузкой interactive держит SLO, а batch
	// растягивается (§2.2, §3.3.10).
	if byLabel := groupByLabel(ok); len(byLabel) > 1 {
		fmt.Println("\nпо классам обслуживания:")
		names := make([]string, 0, len(byLabel))
		for n := range byLabel {
			names = append(names, n)
		}
		sort.Strings(names)
		fmt.Printf("  %-14s %8s %10s %10s %10s\n", "класс", "запросов", "TTFT p50", "TTFT p95", "TTFT p99")
		for _, n := range names {
			g := byLabel[n]
			sort.Slice(g, func(i, j int) bool { return g[i].ttft < g[j].ttft })
			ttfts := make([]time.Duration, len(g))
			for i, x := range g {
				ttfts[i] = x.ttft
			}
			fmt.Printf("  %-14s %8d %9.1f %9.1f %9.1f\n", n, len(g),
				ms(pct(ttfts, 50)), ms(pct(ttfts, 95)), ms(pct(ttfts, 99)))
			rep.ByClass[n] = classReport{
				Requests: len(g), TTFTp50Ms: ms(pct(ttfts, 50)),
				TTFTp95Ms: ms(pct(ttfts, 95)), TTFTp99Ms: ms(pct(ttfts, 99)),
			}
		}
	}

	if r.pool != nil {
		fmt.Printf("\nсессий завершено: %d\n", r.pool.Finished())
	}

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
