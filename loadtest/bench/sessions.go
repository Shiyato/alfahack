package main

// Генератор агентской нагрузки (§5.3).
//
// Профиль воспроизводит форму реальных трасс кодового агента, а не
// «случайные независимые промпты». Разница принципиальна: на независимых
// промптах префикс-кэш не работает, роутер тестируется вхолостую, и все
// выводы о нём недействительны.
//
// Что воспроизводится:
//   • общий системный промпт у всех сессий (18-20% переиспользований);
//   • многоходовые сессии с дописыванием истории (~67% переиспользований
//     — внутрисессионные, из предыдущих ходов);
//   • тяжёлый хвост по длине сессий (верхние 25% дают >80% токенов);
//   • короткие паузы между ходами (92-94% запросов инициирует агент,
//     а не человек);
//   • высокое отношение числа одновременных сессий к числу апстримов —
//     при малом числе конкурентных запросов балансировка не спасает,
//     и выводы нерепрезентативны.

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"sync"
	"time"
)

// SystemPrompt общий у всех сессий: это он даёт 18-20% переиспользований.
const systemPromptUnit = "Ты — агент разработки, работающий с кодовой базой. "

type SessionProfile struct {
	SystemTokens  int // длина общего системного промпта
	FirstTurnMin  int // длина первого сообщения пользователя
	FirstTurnMax  int
	TurnTokens    int     // сколько дописывает каждый следующий ход
	OutputTokens  int     // длина ответа; фиксируется принудительно (см. ниже)
	LongTailRatio float64 // доля длинных сессий
	ShortTurns    int
	LongTurnsMin  int
	LongTurnsMax  int
	ThinkTimeMs   int // пауза между ходами
}

// DefaultSessionProfile подобран так, чтобы средняя длина входа совпала
// с агентской трассой: 8596 входных токенов (§5.1).
//
// Числа не назначены, а выведены: первая версия профиля давала средний
// вход 4004 токена — вдвое меньше трассы. Поскольку время префилла растёт
// суперлинейно, такой профиль систематически занижал нагрузку и смещал
// колено насыщения. Проверяется тестом TestDlinaVhodaSopostavimaSTrassoi.
func DefaultSessionProfile() SessionProfile {
	return SessionProfile{
		SystemTokens:  2500,
		FirstTurnMin:  1500,
		FirstTurnMax:  4000,
		TurnTokens:    300,
		OutputTokens:  182, // средняя длина выхода агентской трассы, §5.1
		LongTailRatio: 0.25,
		ShortTurns:    2,
		LongTurnsMin:  8,
		LongTurnsMax:  30,
		ThinkTimeMs:   200,
	}
}

type Msg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Session хранит растущую историю. Ответ модели фиксируется
// принудительно: если он плавает, история следующего хода перестаёт
// совпадать с закэшированной и паттерн переиспользования ломается (§5.2).
type Session struct {
	ID       int
	Turns    int
	turn     int
	messages []Msg
	profile  SessionProfile
	rnd      *rand.Rand
	salt     string
}

func filler(unit string, tokens int) string {
	// Один токен ≈ 4 байта — та же мера, что у мока. Точность
	// токенизатора здесь не нужна, нужна воспроизводимость.
	target := tokens * 4
	var b strings.Builder
	b.Grow(target + len(unit))
	for b.Len() < target {
		b.WriteString(unit)
	}
	return b.String()[:target]
}

func NewSession(id int, p SessionProfile, rnd *rand.Rand, salt string) *Session {
	turns := p.ShortTurns
	if rnd.Float64() < p.LongTailRatio {
		turns = p.LongTurnsMin + rnd.Intn(p.LongTurnsMax-p.LongTurnsMin+1)
	}
	return &Session{ID: id, Turns: turns, profile: p, rnd: rnd, salt: salt}
}

// Next возвращает сообщения для следующего хода или nil, если сессия
// закончилась.
func (s *Session) Next() []Msg {
	if s.turn >= s.Turns {
		return nil
	}
	if s.turn == 0 {
		first := s.profile.FirstTurnMin +
			s.rnd.Intn(s.profile.FirstTurnMax-s.profile.FirstTurnMin+1)
		s.messages = []Msg{
			{Role: "system", Content: filler(systemPromptUnit, s.profile.SystemTokens)},
			{Role: "user", Content: fmt.Sprintf("|%s-задача-%d| ", s.salt, s.ID) +
				filler("проанализируй модуль и предложи правку. ", first)},
		}
	} else {
		s.messages = append(s.messages,
			Msg{Role: "assistant", Content: filler("предлагаю следующее изменение. ",
				s.profile.OutputTokens)},
			Msg{Role: "user", Content: filler("продолжай со следующим файлом. ",
				s.profile.TurnTokens)},
		)
	}
	s.turn++
	out := make([]Msg, len(s.messages))
	copy(out, s.messages)
	return out
}

func (s *Session) Turn() int { return s.turn - 1 }

// SessionPool раздаёт сессии воркерам. Закончившиеся заменяются новыми,
// чтобы число одновременно живых сессий держалось постоянным.
type SessionPool struct {
	mu       sync.Mutex
	active   []*Session
	profile  SessionProfile
	rnd      *rand.Rand
	nextID   int
	salt     string
	finished int
}

func NewSessionPool(size int, p SessionProfile, seed int64, salt string) *SessionPool {
	pool := &SessionPool{profile: p, rnd: rand.New(rand.NewSource(seed)), salt: salt}
	for i := 0; i < size; i++ {
		pool.active = append(pool.active, NewSession(pool.nextID, p, pool.rnd, salt))
		pool.nextID++
	}
	return pool
}

// Take возвращает сессию и её следующий ход.
func (p *SessionPool) Take() (*Session, []Msg) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.active) == 0 {
		return nil, nil
	}
	idx := p.rnd.Intn(len(p.active))
	s := p.active[idx]
	msgs := s.Next()
	if msgs == nil {
		p.finished++
		p.active[idx] = NewSession(p.nextID, p.profile, p.rnd, p.salt)
		p.nextID++
		s = p.active[idx]
		msgs = s.Next()
	}
	return s, msgs
}

func (p *SessionPool) Finished() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.finished
}

// PoissonInterval — интервал между запросами при пуассоновском потоке.
// Равномерные интервалы дают неестественно гладкую нагрузку и скрывают
// как раз те всплески, ради которых выбран token bucket (§3.3.3.5).
func PoissonInterval(rnd *rand.Rand, ratePerSec float64) time.Duration {
	if ratePerSec <= 0 {
		return 0
	}
	u := rnd.Float64()
	if u <= 0 {
		u = 1e-9
	}
	return time.Duration(-math.Log(u) / ratePerSec * float64(time.Second))
}
