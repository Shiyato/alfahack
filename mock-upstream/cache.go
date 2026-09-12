package main

import (
	"container/list"
	"encoding/binary"
	"hash/fnv"
	"sync"
	"time"
)

// BlockHashes считает цепочечные хеши блоков по схеме Mooncake (§3.3.6.2):
// хеш блока считается от токенов блока, сцепленных с хешем предыдущего блока.
// Благодаря цепочке одинаковый блок в разных контекстах даёт разные ключи —
// это и есть корректная идентификация KV-состояния, а не просто текста.
//
// Мок работает с текстом, а не с настоящими токенами: один токен ≈ 4 байта.
// Для стенда этого достаточно — важна не абсолютная точность токенизатора,
// а воспроизводимость и монотонность по длине.
func BlockHashes(text string, blockTokens int) []uint64 {
	blockBytes := blockTokens * BytesPerToken
	if blockBytes <= 0 {
		return nil
	}
	n := len(text) / blockBytes // хвост короче блока в кэш не попадает
	out := make([]uint64, 0, n)
	var prev uint64
	buf := make([]byte, 8)
	for i := 0; i < n; i++ {
		h := fnv.New64a()
		binary.LittleEndian.PutUint64(buf, prev)
		h.Write(buf)
		h.Write([]byte(text[i*blockBytes : (i+1)*blockBytes]))
		prev = h.Sum64()
		out = append(out, prev)
	}
	return out
}

// BytesPerToken — грубая, но детерминированная оценка. Мок самосогласован:
// он считает токены только собственной мерой и ни с кем её не сверяет.
const BytesPerToken = 4

func CountTokens(text string) int { return len(text) / BytesPerToken }

type cacheEntry struct {
	key     uint64
	expires time.Time
	lruElem *list.Element
}

// KVCache моделирует префикс-кэш инстанса: LRU с TTL по блокам.
// Мок обязан иметь настоящий кэш, иначе cache-aware роутинг нечем измерить —
// мы бы мерили собственную выдумку, а не поведение системы.
type KVCache struct {
	mu       sync.Mutex
	blocks   map[uint64]*cacheEntry
	lru      *list.List // front — самый свежий
	capacity int
	ttl      time.Duration

	hits   uint64 // попавших блоков накопительно
	misses uint64
}

func NewKVCache(capacity int, ttl time.Duration) *KVCache {
	return &KVCache{
		blocks:   make(map[uint64]*cacheEntry),
		lru:      list.New(),
		capacity: capacity,
		ttl:      ttl,
	}
}

// MatchPrefix возвращает число блоков совпавшего префикса. Сравнение идёт
// последовательно до первого несовпадения — ровно так определяется prefix_len
// в §3.3.6.2. Разрыв в середине цепочки не может «перепрыгнуться»:
// KV-состояние блока k осмысленно только если посчитаны все блоки до него.
func (c *KVCache) MatchPrefix(hashes []uint64) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	now := time.Now()
	matched := 0
	for _, h := range hashes {
		e, ok := c.blocks[h]
		if !ok || now.After(e.expires) {
			break
		}
		c.lru.MoveToFront(e.lruElem)
		e.expires = now.Add(c.ttl)
		matched++
	}
	c.hits += uint64(matched)
	c.misses += uint64(len(hashes) - matched)
	return matched
}

// Insert кладёт в кэш все блоки запроса — после префилла они посчитаны.
func (c *KVCache) Insert(hashes []uint64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	now := time.Now()
	for _, h := range hashes {
		if e, ok := c.blocks[h]; ok {
			e.expires = now.Add(c.ttl)
			c.lru.MoveToFront(e.lruElem)
			continue
		}
		e := &cacheEntry{key: h, expires: now.Add(c.ttl)}
		e.lruElem = c.lru.PushFront(e)
		c.blocks[h] = e
	}
	c.evictLocked(now)
}

func (c *KVCache) evictLocked(now time.Time) {
	// Сначала протухшие с хвоста, затем — по ёмкости.
	for c.lru.Len() > 0 {
		back := c.lru.Back()
		e := back.Value.(*cacheEntry)
		if !now.After(e.expires) {
			break
		}
		c.lru.Remove(back)
		delete(c.blocks, e.key)
	}
	if c.capacity <= 0 {
		return
	}
	for c.lru.Len() > c.capacity {
		back := c.lru.Back()
		e := back.Value.(*cacheEntry)
		c.lru.Remove(back)
		delete(c.blocks, e.key)
	}
}

func (c *KVCache) Stats() (blocks int, hits, misses uint64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.lru.Len(), c.hits, c.misses
}
