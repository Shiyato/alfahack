package main

import (
	"flag"
	"time"
)

// Config описывает поведение мок-апстрима. Все параметры задаются флагами,
// чтобы один и тот же бинарь мог играть роль быстрого и медленного инстанса
// в одном стенде (см. docs/architecture.md §5).
type Config struct {
	Addr string
	ID   string

	// --- Модель времени префилла (§3.3.6.7a) ---
	// t_prefill(m) = PrefillA + PrefillB*m + PrefillC*(m/1000)^2, m — число токенов,
	// которые реально нужно посчитать (вход минус попадание в кэш).
	// Квадратичный член обязателен: внимание квадратично по длине входа,
	// поэтому линейная аппроксимация врёт на длинных контекстах.
	PrefillA time.Duration
	PrefillB time.Duration
	PrefillC time.Duration

	// --- Модель декодирования ---
	// Межтокенная задержка при батче размера 1. Декодирование сублинейно
	// по размеру батча: реальный ITL = ITLBase * (1 + BatchPenalty*(batch-1)).
	ITLBase      time.Duration
	BatchPenalty float64

	// --- Ёмкость ---
	// Сколько запросов инстанс обрабатывает одновременно. Всё сверх — в очередь.
	// Запрос в очереди считается pending: именно по наличию таких запросов
	// SkyLB определяет занятость апстрима (§3.3.3.7).
	MaxConcurrency int
	MaxQueue       int

	// --- KV-кэш ---
	BlockTokens int           // размер блока для цепочечного хеширования (§3.3.6.2)
	CacheTTL    time.Duration // по трассам агентов 5-10 минут покрывают ~90% переиспользований (§5.3)
	CacheBlocks int           // ёмкость кэша в блоках; 0 — без ограничения

	// --- Инъекция отказов (§5) ---
	ErrorRate  float64       // доля запросов, отвечающих 5xx
	Overloaded bool          // отвечать 429 на всё: имитация исчерпания квоты апстрима
	HangRate   float64       // доля запросов, зависающих без единого байта
	HangFor    time.Duration // на сколько зависать

	// --- Генерация ответа ---
	DefaultOutputTokens int // если клиент не задал max_tokens
	Seed                int64
}

func parseFlags() *Config {
	c := &Config{}
	flag.StringVar(&c.Addr, "addr", ":9001", "адрес прослушивания")
	flag.StringVar(&c.ID, "id", "", "идентификатор инстанса (по умолчанию — из адреса)")

	flag.DurationVar(&c.PrefillA, "prefill-a", 15*time.Millisecond, "константа модели префилла")
	flag.DurationVar(&c.PrefillB, "prefill-b", 40*time.Microsecond, "линейный коэффициент на токен")
	flag.DurationVar(&c.PrefillC, "prefill-c", 1500*time.Microsecond, "квадратичный коэффициент: добавка на (тыс. токенов)^2")

	flag.DurationVar(&c.ITLBase, "itl", 12*time.Millisecond, "межтокенная задержка при батче 1")
	flag.Float64Var(&c.BatchPenalty, "batch-penalty", 0.06, "рост ITL на каждый дополнительный запрос в батче")

	flag.IntVar(&c.MaxConcurrency, "concurrency", 8, "размер батча: сколько запросов обрабатывается одновременно")
	flag.IntVar(&c.MaxQueue, "queue", 256, "максимальная глубина очереди перед отказом 429")

	flag.IntVar(&c.BlockTokens, "block-tokens", 256, "размер блока KV-кэша в токенах")
	flag.DurationVar(&c.CacheTTL, "cache-ttl", 10*time.Minute, "время жизни блока в KV-кэше")
	flag.IntVar(&c.CacheBlocks, "cache-blocks", 20000, "ёмкость KV-кэша в блоках (0 — без ограничения)")

	flag.Float64Var(&c.ErrorRate, "error-rate", 0, "доля запросов с ответом 5xx")
	flag.BoolVar(&c.Overloaded, "overloaded", false, "отвечать 429 на все запросы")
	flag.Float64Var(&c.HangRate, "hang-rate", 0, "доля зависающих запросов")
	flag.DurationVar(&c.HangFor, "hang-for", 120*time.Second, "длительность зависания")

	flag.IntVar(&c.DefaultOutputTokens, "default-output", 182, "длина ответа по умолчанию (агентская трасса, §5.1)")
	flag.Int64Var(&c.Seed, "seed", 42, "зерно генератора случайных чисел")
	flag.Parse()

	if c.ID == "" {
		c.ID = "mock" + c.Addr
	}
	return c
}

// prefillDuration возвращает время префилла для m вычисляемых токенов.
// Квадратичный член нормирован на тысячи токенов, чтобы коэффициент
// оставался читаемым человеком.
func (c *Config) prefillDuration(m int) time.Duration {
	if m < 0 {
		m = 0
	}
	k := float64(m) / 1000.0
	return c.PrefillA +
		time.Duration(float64(c.PrefillB)*float64(m)) +
		time.Duration(float64(c.PrefillC)*k*k)
}
