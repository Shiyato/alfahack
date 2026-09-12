package main

import (
	"context"
	"errors"
	"sync/atomic"
	"time"
)

var (
	ErrQueueFull = errors.New("очередь переполнена")
	ErrCanceled  = errors.New("запрос отменён клиентом")
)

// Engine моделирует движок инференса: ограниченный батч, очередь перед ним,
// префилл с учётом попадания в KV-кэш и декодирование с замедлением от батча.
//
// Смысл именно такой модели — воспроизвести два свойства, на которых стоит
// весь роутинг из §3.3.6: (1) префилл суперлинеен по длине входа и резко
// дешевеет при попадании в кэш; (2) запрос, не попавший в батч, ждёт —
// и само наличие таких запросов есть признак занятости апстрима (§3.3.3.7).
type Engine struct {
	cfg   *Config
	cache *KVCache

	slots chan struct{}

	queued         atomic.Int64 // ждут места в батче = pending-запросы
	running        atomic.Int64 // в батче
	pendingPrefill atomic.Int64 // токены, ожидающие префилла: метрика нагрузки из §3.3.6.3

	totalRequests  atomic.Uint64
	totalRejected  atomic.Uint64
	totalCanceled  atomic.Uint64
	totalTokensIn  atomic.Uint64
	totalTokensOut atomic.Uint64
}

func NewEngine(cfg *Config) *Engine {
	return &Engine{
		cfg:   cfg,
		cache: NewKVCache(cfg.CacheBlocks, cfg.CacheTTL),
		slots: make(chan struct{}, cfg.MaxConcurrency),
	}
}

// Admission — результат постановки запроса в работу.
type Admission struct {
	PromptTokens  int
	CachedBlocks  int
	CachedTokens  int
	ComputeTokens int
	QueueWait     time.Duration
	PrefillTime   time.Duration
	release       func()
}

// Acquire ставит запрос в очередь, дожидается места в батче и выполняет
// префилл. Возвращает управление в момент, когда должен пойти первый токен.
// Всё время до этого момента — и есть TTFT апстрима.
func (e *Engine) Acquire(ctx context.Context, prompt string) (*Admission, error) {
	e.totalRequests.Add(1)

	promptTokens := CountTokens(prompt)
	hashes := BlockHashes(prompt, e.cfg.BlockTokens)
	cachedBlocks := e.cache.MatchPrefix(hashes)
	cachedTokens := cachedBlocks * e.cfg.BlockTokens
	computeTokens := promptTokens - cachedTokens
	if computeTokens < 0 {
		computeTokens = 0
	}

	if e.queued.Load() >= int64(e.cfg.MaxQueue) {
		e.totalRejected.Add(1)
		return nil, ErrQueueFull
	}

	e.queued.Add(1)
	e.pendingPrefill.Add(int64(computeTokens))
	queueStart := time.Now()

	select {
	case e.slots <- struct{}{}:
	case <-ctx.Done():
		e.queued.Add(-1)
		e.pendingPrefill.Add(-int64(computeTokens))
		e.totalCanceled.Add(1)
		return nil, ErrCanceled
	}

	e.queued.Add(-1)
	e.running.Add(1)
	queueWait := time.Since(queueStart)

	prefill := e.cfg.prefillDuration(computeTokens)
	if err := sleepCtx(ctx, prefill); err != nil {
		e.pendingPrefill.Add(-int64(computeTokens))
		e.running.Add(-1)
		<-e.slots
		e.totalCanceled.Add(1)
		return nil, ErrCanceled
	}
	e.pendingPrefill.Add(-int64(computeTokens))
	e.cache.Insert(hashes)
	e.totalTokensIn.Add(uint64(promptTokens))

	var released atomic.Bool
	adm := &Admission{
		PromptTokens:  promptTokens,
		CachedBlocks:  cachedBlocks,
		CachedTokens:  cachedTokens,
		ComputeTokens: computeTokens,
		QueueWait:     queueWait,
		PrefillTime:   prefill,
		release: func() {
			if released.CompareAndSwap(false, true) {
				e.running.Add(-1)
				<-e.slots
			}
		},
	}
	return adm, nil
}

// Release обязателен: без него слот в батче течёт. Освобождение при разрыве
// соединения клиентом — это ровно то, что проверяет §3.3.9 (зомби-запросы).
func (a *Admission) Release() { a.release() }

// NextTokenDelay — межтокенная задержка с учётом текущего размера батча.
// Декодирование сублинейно по батчу: чем больше одновременных запросов,
// тем медленнее каждый, но суммарный TPS растёт.
func (e *Engine) NextTokenDelay() time.Duration {
	batch := e.running.Load()
	if batch < 1 {
		batch = 1
	}
	mult := 1 + e.cfg.BatchPenalty*float64(batch-1)
	return time.Duration(float64(e.cfg.ITLBase) * mult)
}

func (e *Engine) CountOutToken() { e.totalTokensOut.Add(1) }

func sleepCtx(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return nil
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
