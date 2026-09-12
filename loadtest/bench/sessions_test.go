package main

// Контрольные опыты для генератора нагрузки.
//
// Зачем. Все выводы о роутере (Б-3) построены на предположении, что
// профиль воспроизводит агентскую нагрузку из §5.3: высокая общность
// префикса, переиспользование внутри сессии, тяжёлый хвост по длине.
// Если генератор этого не даёт, измерялся не роутер, а собственная
// выдумка — и сравнение стратегий недействительно.
//
// Здесь заявленные характеристики проверяются численно.

import (
	"crypto/sha256"
	"math"
	"math/rand"
	"sort"
	"testing"
	"time"
)

const blockTokens = 256

// blockHashes повторяет схему цепочечного хеширования из гейтвея
// (gateway/router/prefix.py). Дублирование намеренное: тест обязан быть
// независимой проверкой, а не вызовом той же функции, которую проверяет.
func blockHashes(text string) [][32]byte {
	blockBytes := blockTokens * 4
	n := len(text) / blockBytes
	out := make([][32]byte, 0, n)
	var prev [32]byte
	for i := 0; i < n; i++ {
		h := sha256.New()
		h.Write(prev[:])
		h.Write([]byte(text[i*blockBytes : (i+1)*blockBytes]))
		copy(prev[:], h.Sum(nil))
		out = append(out, prev)
	}
	return out
}

func flatten(msgs []Msg) string {
	s := ""
	for _, m := range msgs {
		s += m.Role + "\n" + m.Content + "\n"
	}
	return s
}

// collectRequests прогоняет пул и возвращает все запросы в порядке выдачи.
type reqRecord struct {
	sessionID int
	turn      int
	hashes    [][32]byte
	tokens    int
}

func collectRequests(t *testing.T, sessions, total int) []reqRecord {
	t.Helper()
	pool := NewSessionPool(sessions, DefaultSessionProfile(), 42, "test")
	out := make([]reqRecord, 0, total)
	for i := 0; i < total; i++ {
		sess, msgs := pool.Take()
		if msgs == nil {
			t.Fatal("пул перестал выдавать запросы")
		}
		text := flatten(msgs)
		out = append(out, reqRecord{
			sessionID: sess.ID,
			turn:      sess.Turn(),
			hashes:    blockHashes(text),
			tokens:    len(text) / 4,
		})
	}
	return out
}

// 1. Общность префикса — главное свойство. Без неё весь роутер
// бессмыслен: least-load достигал бы 97% пропускной способности
// продвинутого решения (§3.3.6.3g).
func TestObshchnostPrefiksaVyshe80Procentov(t *testing.T) {
	reqs := collectRequests(t, 60, 1500)

	// Доля блоков, которые уже встречались раньше в потоке. Это прямая
	// оценка того, сколько префилла можно было бы переиспользовать при
	// идеальном кэше.
	seen := map[[32]byte]bool{}
	var total, reused int
	for _, r := range reqs {
		for _, h := range r.hashes {
			total++
			if seen[h] {
				reused++
			} else {
				seen[h] = true
			}
		}
	}
	if total == 0 {
		t.Fatal("генератор не произвёл ни одного полного блока")
	}
	ratio := float64(reused) / float64(total)
	t.Logf("переиспользуемых блоков: %.1f%% (%d из %d)", ratio*100, reused, total)
	if ratio < 0.80 {
		t.Errorf("общность префикса %.1f%% ниже заявленных 80%% (§5.3): "+
			"на таком профиле роутер тестируется вхолостую", ratio*100)
	}
}

// 2. Источник переиспользования. По §5.3 около 67% приходит из
// предыдущих ходов той же сессии, а не от общего системного промпта.
// Если весь выигрыш даёт системный промпт, сессионная маршрутизация
// проверяется впустую — её преимущество именно во внутрисессионном
// переиспользовании.
func TestPereispolzovanieVnutriSessii(t *testing.T) {
	reqs := collectRequests(t, 60, 1500)

	// Блоки, встречавшиеся ранее В ТОЙ ЖЕ сессии, против встречавшихся
	// только в других.
	perSession := map[int]map[[32]byte]bool{}
	global := map[[32]byte]bool{}
	var reusedInSession, reusedCross int

	for _, r := range reqs {
		own := perSession[r.sessionID]
		if own == nil {
			own = map[[32]byte]bool{}
			perSession[r.sessionID] = own
		}
		for _, h := range r.hashes {
			switch {
			case own[h]:
				reusedInSession++
			case global[h]:
				reusedCross++
			}
			own[h] = true
			global[h] = true
		}
	}

	totalReuse := reusedInSession + reusedCross
	if totalReuse == 0 {
		t.Fatal("переиспользования нет вовсе")
	}
	share := float64(reusedInSession) / float64(totalReuse)
	t.Logf("внутрисессионное переиспользование: %.1f%% (%d из %d)",
		share*100, reusedInSession, totalReuse)
	if share < 0.40 {
		t.Errorf("внутрисессионное переиспользование %.1f%% — слишком мало "+
			"относительно 67%% из §5.3; выигрыш даёт общий системный промпт, "+
			"а не сессии, и сессионная маршрутизация проверяется впустую", share*100)
	}
}

// 3. Вклад системного промпта: 18-20% переиспользований по §5.3.
// Проверяем, что он есть, но не доминирует.
func TestVkladSistemnogoPrompta(t *testing.T) {
	p := DefaultSessionProfile()
	sysBlocks := p.SystemTokens / blockTokens
	if sysBlocks < 1 {
		t.Fatalf("системный промпт короче одного блока: %d токенов", p.SystemTokens)
	}
	reqs := collectRequests(t, 60, 800)

	var totalBlocks int
	for _, r := range reqs {
		totalBlocks += len(r.hashes)
	}
	// Системный префикс присутствует в каждом запросе.
	share := float64(sysBlocks*len(reqs)) / float64(totalBlocks)
	t.Logf("доля блоков системного промпта: %.1f%%", share*100)
	if share > 0.60 {
		t.Errorf("системный промпт занимает %.1f%% всех блоков — профиль "+
			"вырожден, различия между сессиями теряются", share*100)
	}
}

// 4. Тяжёлый хвост: верхние 25% сессий дают более 80% токенов (§5.3).
// Без хвоста нагрузка однородна, и перекос, который должен устранять
// роутер, просто не возникает.
func TestTyazhelyiHvostPoDlineSessii(t *testing.T) {
	rnd := rand.New(rand.NewSource(7))
	p := DefaultSessionProfile()

	const n = 2000
	lengths := make([]int, 0, n)
	for i := 0; i < n; i++ {
		s := NewSession(i, p, rnd, "x")
		lengths = append(lengths, s.Turns)
	}
	sort.Sort(sort.Reverse(sort.IntSlice(lengths)))

	total := 0
	for _, l := range lengths {
		total += l
	}
	top := 0
	for _, l := range lengths[:n/4] {
		top += l
	}
	share := float64(top) / float64(total)
	t.Logf("верхние 25%% сессий дают %.1f%% ходов (медиана %d, максимум %d)",
		share*100, lengths[n/2], lengths[0])
	if share < 0.60 {
		t.Errorf("верхние 25%% дают %.1f%% ходов — хвост слишком лёгкий "+
			"относительно 80%% из §5.3, перекос нагрузки не возникнет", share*100)
	}
}

// 5. Ходы одной сессии обязаны наращивать историю, а не заменять её:
// на этом стоит внутрисессионное переиспользование.
func TestHodyNarashchivayutIstoriyu(t *testing.T) {
	rnd := rand.New(rand.NewSource(1))
	p := DefaultSessionProfile()
	s := NewSession(0, p, rnd, "x")
	s.Turns = 6

	var prev [][32]byte
	for turn := 0; turn < 6; turn++ {
		msgs := s.Next()
		if msgs == nil {
			t.Fatalf("сессия закончилась на ходе %d", turn)
		}
		h := blockHashes(flatten(msgs))
		if turn > 0 {
			if len(h) <= len(prev) {
				t.Errorf("ход %d не удлинил историю: было %d блоков, стало %d",
					turn, len(prev), len(h))
			}
			common := 0
			for i := range prev {
				if i < len(h) && h[i] == prev[i] {
					common++
				} else {
					break
				}
			}
			if common < len(prev) {
				t.Errorf("ход %d изменил историю: совпало %d блоков из %d — "+
					"переиспользование сломано", turn, common, len(prev))
			}
		}
		prev = h
	}
}

// 6. Разные сессии обязаны различаться. Если все одинаковы, роутер
// нечего балансировать.
func TestRaznyeSessiiRazlichayutsya(t *testing.T) {
	reqs := collectRequests(t, 40, 200)

	// Берём последний блок первого хода каждой сессии — он приходится
	// на пользовательскую часть, а не на общий системный промпт.
	firstTurnTails := map[[32]byte]int{}
	for _, r := range reqs {
		if r.turn == 0 && len(r.hashes) > 0 {
			firstTurnTails[r.hashes[len(r.hashes)-1]]++
		}
	}
	if len(firstTurnTails) < 10 {
		t.Errorf("первые ходы дали всего %d различных хвостов: сессии "+
			"неразличимы, балансировать нечего", len(firstTurnTails))
	}
}

// 7. Длина входа должна быть сопоставима с агентской трассой: средний
// вход 8596 токенов (§5.1). Порядок величины важен, потому что префилл
// суперлинеен и на коротких входах роутер не проявится.
func TestDlinaVhodaSopostavimaSTrassoi(t *testing.T) {
	reqs := collectRequests(t, 60, 600)
	total := 0
	for _, r := range reqs {
		total += r.tokens
	}
	avg := total / len(reqs)
	t.Logf("средняя длина входа: %d токенов (трасса: 8596, §5.1)", avg)
	// Допуск вдвое в обе стороны. Уже, чем кажется: первая версия
	// профиля давала 4004 токена и прошла бы порог «на порядок», хотя
	// систематически занижала нагрузку — префилл суперлинеен.
	if avg < 4300 || avg > 17200 {
		t.Errorf("средний вход %d токенов расходится с агентской трассой "+
			"(8596, §5.1) более чем вдвое: замеры роутера сместятся, "+
			"потому что время префилла растёт суперлинейно", avg)
	}
}

// 8. Пуассоновский поток: равномерные интервалы дают неестественно
// гладкую нагрузку и скрывают всплески, ради которых выбран token
// bucket (§3.3.3.5).
func TestPuassonovskiyPotokDaetVspleski(t *testing.T) {
	rnd := rand.New(rand.NewSource(3))
	const rate = 100.0
	const n = 20000

	var sum float64
	intervals := make([]float64, 0, n)
	for i := 0; i < n; i++ {
		d := PoissonInterval(rnd, rate).Seconds()
		intervals = append(intervals, d)
		sum += d
	}
	mean := sum / float64(n)

	var variance float64
	for _, d := range intervals {
		variance += (d - mean) * (d - mean)
	}
	cv := math.Sqrt(variance/float64(n)) / mean

	t.Logf("средний интервал %.5f с (ожидается %.5f), CV=%.3f", mean, 1/rate, cv)
	if math.Abs(mean-1/rate) > 0.1/rate {
		t.Errorf("средний интервал %.5f не соответствует заданной "+
			"интенсивности %.0f/с", mean, rate)
	}
	// У экспоненциального распределения CV равен 1 ровно.
	if cv < 0.8 || cv > 1.2 {
		t.Errorf("CV интервалов %.3f далёк от 1 — поток не пуассоновский, "+
			"всплесков не будет", cv)
	}
}

// 9. Пул держит постоянное число живых сессий: при малом числе
// одновременных запросов балансировка не спасает, и выводы
// нерепрезентативны (§5.3).
func TestPulDerzhitPostoyannoeChisloSessii(t *testing.T) {
	const size = 30
	pool := NewSessionPool(size, DefaultSessionProfile(), 5, "x")
	for i := 0; i < 2000; i++ {
		if _, msgs := pool.Take(); msgs == nil {
			t.Fatalf("пул иссяк на запросе %d", i)
		}
	}
	if pool.Finished() == 0 {
		t.Error("ни одна сессия не завершилась за 2000 запросов — " +
			"пул не обновляется, профиль вырождается в статичный")
	}
	t.Logf("завершено сессий: %d за 2000 запросов", pool.Finished())
}

// 10. Воспроизводимость: одно зерно — одна и та же последовательность.
// Без неё замеры несравнимы между прогонами.
func TestProfilVosproizvodim(t *testing.T) {
	run := func() []int {
		pool := NewSessionPool(20, DefaultSessionProfile(), 99, "s")
		out := make([]int, 0, 100)
		for i := 0; i < 100; i++ {
			sess, _ := pool.Take()
			out = append(out, sess.ID)
		}
		return out
	}
	a, b := run(), run()
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("прогоны разошлись на позиции %d: %d против %d", i, a[i], b[i])
		}
	}
}

var _ = time.Second
