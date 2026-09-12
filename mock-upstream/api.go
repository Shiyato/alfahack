package main

// Минимальный OpenAI-совместимый контракт. Расширяем только тем, что реально
// используется стендом: контракт должен быть узнаваем клиентами, а не полон.

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type StreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type ChatRequest struct {
	Model         string         `json:"model"`
	Messages      []Message      `json:"messages"`
	Stream        bool           `json:"stream"`
	MaxTokens     *int           `json:"max_tokens"`
	Temperature   *float64       `json:"temperature"`
	StreamOptions *StreamOptions `json:"stream_options"`

	// Нестандартное расширение стенда: при воспроизведении трасс длину
	// ответа фиксируют принудительно, иначе паттерн переиспользования
	// кэша ломается — история следующего хода перестаёт совпадать (§5.2).
	MockOutputTokens *int `json:"mock_output_tokens"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type Delta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

type StreamChoice struct {
	Index        int     `json:"index"`
	Delta        Delta   `json:"delta"`
	FinishReason *string `json:"finish_reason"`
}

type ChatChunk struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []StreamChoice `json:"choices"`
	Usage   *Usage         `json:"usage,omitempty"`
}

type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

type ChatResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

type APIError struct {
	Error APIErrorBody `json:"error"`
}

type APIErrorBody struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}
