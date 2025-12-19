package tooladapter

import (
	"encoding/json"
	"io"
	"net/http"
)

// SSEChunk represents a parsed Server-Sent Events chunk.
// This is a provider-agnostic representation of a streaming chat completion chunk.
// It preserves non-standard fields from various providers (vLLM, LiteLLM, etc.)
// such as "reasoning", "reasoning_content", "system_fingerprint", etc.
type SSEChunk struct {
	// ID is the unique identifier for this completion.
	ID string `json:"id,omitempty"`

	// Object is the object type (typically "chat.completion.chunk").
	Object string `json:"object,omitempty"`

	// Created is the Unix timestamp of when this was created.
	Created int64 `json:"created,omitempty"`

	// Model is the model that generated this completion.
	Model string `json:"model,omitempty"`

	// Choices contains the completion choices.
	Choices []SSEChoice `json:"choices,omitempty"`

	// Usage contains token usage information (typically only in final chunk).
	Usage *SSEUsage `json:"usage,omitempty"`

	// ExtraFields contains any non-standard fields from the provider.
	// Examples: "reasoning", "reasoning_content", "system_fingerprint", etc.
	ExtraFields map[string]json.RawMessage `json:"-"`
}

// knownChunkFields lists the standard SSEChunk field names for filtering.
var knownChunkFields = map[string]bool{
	"id": true, "object": true, "created": true, "model": true,
	"choices": true, "usage": true,
}

// UnmarshalJSON implements custom unmarshaling to preserve extra fields.
func (c *SSEChunk) UnmarshalJSON(data []byte) error {
	// First unmarshal into a map to capture all fields
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}

	// Extract known fields using a temporary struct
	type Alias SSEChunk
	var alias Alias
	if err := json.Unmarshal(data, &alias); err != nil {
		return err
	}
	*c = SSEChunk(alias)

	// Capture extra fields
	c.ExtraFields = make(map[string]json.RawMessage)
	for key, value := range raw {
		if !knownChunkFields[key] {
			c.ExtraFields[key] = value
		}
	}
	if len(c.ExtraFields) == 0 {
		c.ExtraFields = nil
	}

	return nil
}

// MarshalJSON implements custom marshaling to include extra fields.
func (c SSEChunk) MarshalJSON() ([]byte, error) {
	// Create a map with all standard fields
	result := make(map[string]interface{})

	if c.ID != "" {
		result["id"] = c.ID
	}
	if c.Object != "" {
		result["object"] = c.Object
	}
	if c.Created != 0 {
		result["created"] = c.Created
	}
	if c.Model != "" {
		result["model"] = c.Model
	}
	if len(c.Choices) > 0 {
		result["choices"] = c.Choices
	}
	if c.Usage != nil {
		result["usage"] = c.Usage
	}

	// Add extra fields
	for key, value := range c.ExtraFields {
		result[key] = value
	}

	return json.Marshal(result)
}

// SSEChoice represents a single choice in a streaming response.
// It preserves non-standard fields like "logprobs", "stop_reason", etc.
type SSEChoice struct {
	// Index is the index of this choice.
	Index int `json:"index"`

	// Delta contains the incremental content for this chunk.
	Delta SSEDelta `json:"delta"`

	// FinishReason indicates why generation stopped (null until complete).
	FinishReason string `json:"finish_reason,omitempty"`

	// ExtraFields contains any non-standard fields from the provider.
	// Examples: "logprobs", "stop_reason", etc.
	ExtraFields map[string]json.RawMessage `json:"-"`
}

// knownChoiceFields lists the standard SSEChoice field names for filtering.
var knownChoiceFields = map[string]bool{
	"index": true, "delta": true, "finish_reason": true,
}

// UnmarshalJSON implements custom unmarshaling to preserve extra fields.
func (c *SSEChoice) UnmarshalJSON(data []byte) error {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}

	type Alias SSEChoice
	var alias Alias
	if err := json.Unmarshal(data, &alias); err != nil {
		return err
	}
	*c = SSEChoice(alias)

	c.ExtraFields = make(map[string]json.RawMessage)
	for key, value := range raw {
		if !knownChoiceFields[key] {
			c.ExtraFields[key] = value
		}
	}
	if len(c.ExtraFields) == 0 {
		c.ExtraFields = nil
	}

	return nil
}

// MarshalJSON implements custom marshaling to include extra fields.
func (c SSEChoice) MarshalJSON() ([]byte, error) {
	result := make(map[string]interface{})

	result["index"] = c.Index
	result["delta"] = c.Delta
	if c.FinishReason != "" {
		result["finish_reason"] = c.FinishReason
	}

	for key, value := range c.ExtraFields {
		result[key] = value
	}

	return json.Marshal(result)
}

// SSEDelta represents the incremental content in a streaming chunk.
// It preserves non-standard fields like "reasoning_content", "reasoning_signature", "audio", etc.
type SSEDelta struct {
	// Role is the role of the message author (typically only in first chunk).
	Role string `json:"role,omitempty"`

	// Content is the text content of this chunk.
	Content string `json:"content,omitempty"`

	// ToolCalls contains any tool calls in this chunk.
	ToolCalls []SSEToolCall `json:"tool_calls,omitempty"`

	// ExtraFields contains any non-standard fields from the provider.
	// Examples: "reasoning_content", "reasoning_signature", "audio", etc.
	ExtraFields map[string]json.RawMessage `json:"-"`
}

// knownDeltaFields lists the standard SSEDelta field names for filtering.
var knownDeltaFields = map[string]bool{
	"role": true, "content": true, "tool_calls": true,
}

// UnmarshalJSON implements custom unmarshaling to preserve extra fields.
func (d *SSEDelta) UnmarshalJSON(data []byte) error {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}

	type Alias SSEDelta
	var alias Alias
	if err := json.Unmarshal(data, &alias); err != nil {
		return err
	}
	*d = SSEDelta(alias)

	d.ExtraFields = make(map[string]json.RawMessage)
	for key, value := range raw {
		if !knownDeltaFields[key] {
			d.ExtraFields[key] = value
		}
	}
	if len(d.ExtraFields) == 0 {
		d.ExtraFields = nil
	}

	return nil
}

// MarshalJSON implements custom marshaling to include extra fields.
func (d SSEDelta) MarshalJSON() ([]byte, error) {
	result := make(map[string]interface{})

	if d.Role != "" {
		result["role"] = d.Role
	}
	if d.Content != "" {
		result["content"] = d.Content
	}
	if len(d.ToolCalls) > 0 {
		result["tool_calls"] = d.ToolCalls
	}

	for key, value := range d.ExtraFields {
		result[key] = value
	}

	return json.Marshal(result)
}

// SSEToolCall represents a tool call in a streaming chunk.
type SSEToolCall struct {
	// Index is the index of this tool call within the message.
	Index int `json:"index"`

	// ID is the unique identifier for this tool call.
	ID string `json:"id,omitempty"`

	// Type is the type of tool (typically "function").
	Type string `json:"type,omitempty"`

	// Function contains the function call details.
	Function SSEFunctionCall `json:"function,omitempty"`
}

// SSEFunctionCall represents a function call within a tool call.
type SSEFunctionCall struct {
	// Name is the name of the function to call.
	Name string `json:"name,omitempty"`

	// Arguments is the JSON-encoded arguments for the function.
	Arguments string `json:"arguments,omitempty"`
}

// SSEUsage contains token usage information.
// It preserves non-standard fields like "prompt_tokens_details", "completion_tokens_details", etc.
type SSEUsage struct {
	PromptTokens     int `json:"prompt_tokens,omitempty"`
	CompletionTokens int `json:"completion_tokens,omitempty"`
	TotalTokens      int `json:"total_tokens,omitempty"`

	// ExtraFields contains any non-standard fields from the provider.
	// Examples: "prompt_tokens_details", "completion_tokens_details", etc.
	ExtraFields map[string]json.RawMessage `json:"-"`
}

// knownUsageFields lists the standard SSEUsage field names for filtering.
var knownUsageFields = map[string]bool{
	"prompt_tokens": true, "completion_tokens": true, "total_tokens": true,
}

// UnmarshalJSON implements custom unmarshaling to preserve extra fields.
func (u *SSEUsage) UnmarshalJSON(data []byte) error {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}

	type Alias SSEUsage
	var alias Alias
	if err := json.Unmarshal(data, &alias); err != nil {
		return err
	}
	*u = SSEUsage(alias)

	u.ExtraFields = make(map[string]json.RawMessage)
	for key, value := range raw {
		if !knownUsageFields[key] {
			u.ExtraFields[key] = value
		}
	}
	if len(u.ExtraFields) == 0 {
		u.ExtraFields = nil
	}

	return nil
}

// MarshalJSON implements custom marshaling to include extra fields.
func (u SSEUsage) MarshalJSON() ([]byte, error) {
	result := make(map[string]interface{})

	if u.PromptTokens != 0 {
		result["prompt_tokens"] = u.PromptTokens
	}
	if u.CompletionTokens != 0 {
		result["completion_tokens"] = u.CompletionTokens
	}
	if u.TotalTokens != 0 {
		result["total_tokens"] = u.TotalTokens
	}

	for key, value := range u.ExtraFields {
		result[key] = value
	}

	return json.Marshal(result)
}

// RawFunctionCall represents a parsed function call from model output.
// This is the public equivalent of the internal functionCall type.
type RawFunctionCall struct {
	// Name is the name of the function to call.
	Name string `json:"name"`

	// Parameters contains the function arguments as raw JSON.
	Parameters json.RawMessage `json:"parameters,omitempty"`
}

// SSEStreamReader provides a simple interface for reading SSE streams.
// Implementations can wrap various sources (http.Response, io.Reader, etc.).
type SSEStreamReader interface {
	// Next advances to the next SSE event, returning false when done.
	Next() bool

	// Data returns the data payload of the current event (without "data: " prefix).
	Data() string

	// Err returns any error encountered during reading.
	Err() error

	// Close releases resources associated with the reader.
	Close() error
}

// SSEStreamWriter provides an interface for writing SSE responses.
type SSEStreamWriter interface {
	// WriteChunk writes an SSE chunk to the response.
	WriteChunk(chunk *SSEChunk) error

	// WriteRaw writes raw SSE data (already formatted as "data: {...}\n\n").
	WriteRaw(data []byte) error

	// WriteDone writes the "[DONE]" marker.
	WriteDone() error

	// Flush ensures buffered data is sent to the client.
	Flush()
}

// httpSSEWriter implements SSEStreamWriter for http.ResponseWriter.
type httpSSEWriter struct {
	w       http.ResponseWriter
	flusher http.Flusher
}

// NewHTTPSSEWriter creates a new SSE writer for an HTTP response.
// It sets appropriate headers for SSE streaming.
func NewHTTPSSEWriter(w http.ResponseWriter) SSEStreamWriter {
	// Set SSE headers
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	flusher, _ := w.(http.Flusher)

	return &httpSSEWriter{
		w:       w,
		flusher: flusher,
	}
}

// WriteChunk writes an SSE chunk as JSON.
func (h *httpSSEWriter) WriteChunk(chunk *SSEChunk) error {
	data, err := json.Marshal(chunk)
	if err != nil {
		return err
	}

	_, err = h.w.Write([]byte("data: "))
	if err != nil {
		return err
	}
	_, err = h.w.Write(data)
	if err != nil {
		return err
	}
	_, err = h.w.Write([]byte("\n\n"))
	if err != nil {
		return err
	}

	h.Flush()
	return nil
}

// WriteRaw writes raw SSE data.
func (h *httpSSEWriter) WriteRaw(data []byte) error {
	_, err := h.w.Write(data)
	if err != nil {
		return err
	}
	h.Flush()
	return nil
}

// WriteDone writes the [DONE] marker.
func (h *httpSSEWriter) WriteDone() error {
	_, err := h.w.Write([]byte("data: [DONE]\n\n"))
	if err != nil {
		return err
	}
	h.Flush()
	return nil
}

// Flush flushes buffered data.
func (h *httpSSEWriter) Flush() {
	if h.flusher != nil {
		h.flusher.Flush()
	}
}

// httpSSEReader implements SSEStreamReader for http.Response bodies.
type httpSSEReader struct {
	body    io.ReadCloser
	scanner *sseScanner
	current string
	err     error
	done    bool
}

// sseScanner is a simple line-based scanner for SSE events.
type sseScanner struct {
	reader io.Reader
	buf    []byte
	start  int
	end    int
}

func newSSEScanner(r io.Reader) *sseScanner {
	return &sseScanner{
		reader: r,
		buf:    make([]byte, 32*1024), // 32KB buffer
	}
}

func (s *sseScanner) readLine() (string, error) {
	for {
		// Look for newline in existing buffer
		for i := s.start; i < s.end; i++ {
			if s.buf[i] == '\n' {
				line := string(s.buf[s.start:i])
				s.start = i + 1
				// Trim carriage return if present
				if len(line) > 0 && line[len(line)-1] == '\r' {
					line = line[:len(line)-1]
				}
				return line, nil
			}
		}

		// Need more data
		if s.start > 0 {
			// Shift remaining data to start
			copy(s.buf, s.buf[s.start:s.end])
			s.end -= s.start
			s.start = 0
		}

		// Check if buffer is full
		if s.end == len(s.buf) {
			// Grow buffer
			newBuf := make([]byte, len(s.buf)*2)
			copy(newBuf, s.buf[:s.end])
			s.buf = newBuf
		}

		// Read more data
		n, err := s.reader.Read(s.buf[s.end:])
		if n > 0 {
			s.end += n
		}
		if err != nil {
			if s.start < s.end {
				// Return remaining data
				line := string(s.buf[s.start:s.end])
				s.start = s.end
				return line, nil
			}
			return "", err
		}
	}
}

// NewHTTPSSEReader creates a new SSE reader from an HTTP response.
func NewHTTPSSEReader(resp *http.Response) SSEStreamReader {
	return &httpSSEReader{
		body:    resp.Body,
		scanner: newSSEScanner(resp.Body),
	}
}

// NewSSEReaderFromReadCloser creates a new SSE reader from an io.ReadCloser.
func NewSSEReaderFromReadCloser(rc io.ReadCloser) SSEStreamReader {
	return &httpSSEReader{
		body:    rc,
		scanner: newSSEScanner(rc),
	}
}

// Next advances to the next SSE data event.
func (h *httpSSEReader) Next() bool {
	if h.done || h.err != nil {
		return false
	}

	for {
		line, err := h.scanner.readLine()
		if err != nil {
			if err != io.EOF {
				h.err = err
			}
			h.done = true
			return false
		}

		// Skip empty lines and comments
		if line == "" || line[0] == ':' {
			continue
		}

		// Parse SSE data line
		if len(line) > 6 && line[:6] == "data: " {
			data := line[6:]

			// Check for stream end
			if data == "[DONE]" {
				h.done = true
				return false
			}

			h.current = data
			return true
		}

		// Skip other SSE fields (event:, id:, retry:)
	}
}

// Data returns the current event's data payload.
func (h *httpSSEReader) Data() string {
	return h.current
}

// Err returns any error encountered.
func (h *httpSSEReader) Err() error {
	return h.err
}

// Close closes the underlying reader.
func (h *httpSSEReader) Close() error {
	if h.body != nil {
		return h.body.Close()
	}
	return nil
}

// ParseSSEChunk parses a JSON string into an SSEChunk.
func ParseSSEChunk(data string) (*SSEChunk, error) {
	var chunk SSEChunk
	if err := json.Unmarshal([]byte(data), &chunk); err != nil {
		return nil, err
	}
	return &chunk, nil
}
