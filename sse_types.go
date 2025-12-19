package tooladapter

import (
	"encoding/json"
	"io"
	"net/http"
)

// SSEChunk represents a parsed Server-Sent Events chunk.
// This is a provider-agnostic representation of a streaming chat completion chunk.
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
}

// SSEChoice represents a single choice in a streaming response.
type SSEChoice struct {
	// Index is the index of this choice.
	Index int `json:"index"`

	// Delta contains the incremental content for this chunk.
	Delta SSEDelta `json:"delta"`

	// FinishReason indicates why generation stopped (null until complete).
	FinishReason string `json:"finish_reason,omitempty"`
}

// SSEDelta represents the incremental content in a streaming chunk.
type SSEDelta struct {
	// Role is the role of the message author (typically only in first chunk).
	Role string `json:"role,omitempty"`

	// Content is the text content of this chunk.
	Content string `json:"content,omitempty"`

	// ToolCalls contains any tool calls in this chunk.
	ToolCalls []SSEToolCall `json:"tool_calls,omitempty"`
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
type SSEUsage struct {
	PromptTokens     int `json:"prompt_tokens,omitempty"`
	CompletionTokens int `json:"completion_tokens,omitempty"`
	TotalTokens      int `json:"total_tokens,omitempty"`
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
