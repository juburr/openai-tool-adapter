package tooladapter

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ============================================================================
// SSEChunk Tests
// ============================================================================

func TestSSEChunk_JSONMarshal(t *testing.T) {
	chunk := SSEChunk{
		ID:      "chatcmpl-123",
		Object:  "chat.completion.chunk",
		Created: 1234567890,
		Model:   "gpt-4",
		Choices: []SSEChoice{
			{
				Index: 0,
				Delta: SSEDelta{
					Role:    "assistant",
					Content: "Hello",
				},
			},
		},
	}

	data, err := json.Marshal(chunk)
	require.NoError(t, err)

	assert.Contains(t, string(data), `"id":"chatcmpl-123"`)
	assert.Contains(t, string(data), `"model":"gpt-4"`)
	assert.Contains(t, string(data), `"content":"Hello"`)
}

func TestSSEChunk_JSONUnmarshal(t *testing.T) {
	jsonData := `{
		"id": "chatcmpl-456",
		"object": "chat.completion.chunk",
		"created": 1234567890,
		"model": "gpt-4",
		"choices": [{
			"index": 0,
			"delta": {
				"role": "assistant",
				"content": "World"
			}
		}]
	}`

	var chunk SSEChunk
	err := json.Unmarshal([]byte(jsonData), &chunk)
	require.NoError(t, err)

	assert.Equal(t, "chatcmpl-456", chunk.ID)
	assert.Equal(t, "gpt-4", chunk.Model)
	assert.Len(t, chunk.Choices, 1)
	assert.Equal(t, "World", chunk.Choices[0].Delta.Content)
}

func TestSSEChunk_WithToolCalls(t *testing.T) {
	chunk := SSEChunk{
		ID:      "chatcmpl-789",
		Object:  "chat.completion.chunk",
		Created: 1234567890,
		Model:   "gpt-4",
		Choices: []SSEChoice{
			{
				Index: 0,
				Delta: SSEDelta{
					Role: "assistant",
					ToolCalls: []SSEToolCall{
						{
							Index: 0,
							ID:    "call_123",
							Type:  "function",
							Function: SSEFunctionCall{
								Name:      "get_weather",
								Arguments: `{"city": "Seattle"}`,
							},
						},
					},
				},
				FinishReason: "tool_calls",
			},
		},
	}

	data, err := json.Marshal(chunk)
	require.NoError(t, err)

	assert.Contains(t, string(data), `"tool_calls"`)
	assert.Contains(t, string(data), `"get_weather"`)
	assert.Contains(t, string(data), `"finish_reason":"tool_calls"`)
}

func TestSSEChunk_WithUsage(t *testing.T) {
	chunk := SSEChunk{
		ID:     "chatcmpl-usage",
		Object: "chat.completion.chunk",
		Usage: &SSEUsage{
			PromptTokens:     100,
			CompletionTokens: 50,
			TotalTokens:      150,
		},
	}

	data, err := json.Marshal(chunk)
	require.NoError(t, err)

	assert.Contains(t, string(data), `"prompt_tokens":100`)
	assert.Contains(t, string(data), `"completion_tokens":50`)
	assert.Contains(t, string(data), `"total_tokens":150`)
}

// ============================================================================
// ParseSSEChunk Tests
// ============================================================================

func TestParseSSEChunk_Valid(t *testing.T) {
	data := `{"id":"test","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hi"}}]}`

	chunk, err := ParseSSEChunk(data)
	require.NoError(t, err)
	require.NotNil(t, chunk)

	assert.Equal(t, "test", chunk.ID)
	assert.Len(t, chunk.Choices, 1)
	assert.Equal(t, "Hi", chunk.Choices[0].Delta.Content)
}

func TestParseSSEChunk_Invalid(t *testing.T) {
	data := `not valid json`

	chunk, err := ParseSSEChunk(data)
	assert.Error(t, err)
	assert.Nil(t, chunk)
}

func TestParseSSEChunk_Empty(t *testing.T) {
	data := `{}`

	chunk, err := ParseSSEChunk(data)
	require.NoError(t, err)
	require.NotNil(t, chunk)

	assert.Empty(t, chunk.ID)
	assert.Empty(t, chunk.Choices)
}

// ============================================================================
// sseScanner Tests
// ============================================================================

func TestSSEScanner_BasicReading(t *testing.T) {
	input := "line1\nline2\nline3\n"
	scanner := newSSEScanner(strings.NewReader(input))

	line1, err := scanner.readLine()
	require.NoError(t, err)
	assert.Equal(t, "line1", line1)

	line2, err := scanner.readLine()
	require.NoError(t, err)
	assert.Equal(t, "line2", line2)

	line3, err := scanner.readLine()
	require.NoError(t, err)
	assert.Equal(t, "line3", line3)
}

func TestSSEScanner_CarriageReturn(t *testing.T) {
	input := "line1\r\nline2\r\n"
	scanner := newSSEScanner(strings.NewReader(input))

	line1, err := scanner.readLine()
	require.NoError(t, err)
	assert.Equal(t, "line1", line1)

	line2, err := scanner.readLine()
	require.NoError(t, err)
	assert.Equal(t, "line2", line2)
}

func TestSSEScanner_LongLine(t *testing.T) {
	// Create a line longer than initial buffer
	longLine := strings.Repeat("x", 50000)
	input := longLine + "\n"
	scanner := newSSEScanner(strings.NewReader(input))

	line, err := scanner.readLine()
	require.NoError(t, err)
	assert.Equal(t, longLine, line)
}

func TestSSEScanner_NoTrailingNewline(t *testing.T) {
	input := "final line without newline"
	scanner := newSSEScanner(strings.NewReader(input))

	line, err := scanner.readLine()
	require.NoError(t, err)
	assert.Equal(t, "final line without newline", line)
}

func TestSSEScanner_EmptyLines(t *testing.T) {
	input := "\n\ndata\n\n"
	scanner := newSSEScanner(strings.NewReader(input))

	// Empty lines
	line1, err := scanner.readLine()
	require.NoError(t, err)
	assert.Equal(t, "", line1)

	line2, err := scanner.readLine()
	require.NoError(t, err)
	assert.Equal(t, "", line2)

	// Data line
	line3, err := scanner.readLine()
	require.NoError(t, err)
	assert.Equal(t, "data", line3)
}

// ============================================================================
// httpSSEReader Tests
// ============================================================================

func TestHTTPSSEReader_CompleteStream(t *testing.T) {
	sseData := strings.Join([]string{
		"data: {\"id\": \"1\", \"choices\": [{\"delta\": {\"content\": \"Hello\"}}]}",
		"",
		"data: {\"id\": \"1\", \"choices\": [{\"delta\": {\"content\": \" World\"}}]}",
		"",
		"data: {\"id\": \"1\", \"choices\": [{\"delta\": {}, \"finish_reason\": \"stop\"}]}",
		"",
		"data: [DONE]",
		"",
	}, "\n")

	rc := io.NopCloser(strings.NewReader(sseData))
	reader := NewSSEReaderFromReadCloser(rc)
	defer func() { _ = reader.Close() }()

	var events []string
	for reader.Next() {
		events = append(events, reader.Data())
	}

	require.NoError(t, reader.Err())
	assert.Len(t, events, 3) // 3 data events, not the [DONE]
}

func TestHTTPSSEReader_SkipsNonDataLines(t *testing.T) {
	sseData := strings.Join([]string{
		": this is a comment",
		"event: message",
		"id: 123",
		"retry: 5000",
		"data: {\"test\": true}",
		"",
		"data: [DONE]",
		"",
	}, "\n")

	rc := io.NopCloser(strings.NewReader(sseData))
	reader := NewSSEReaderFromReadCloser(rc)
	defer func() { _ = reader.Close() }()

	var events []string
	for reader.Next() {
		events = append(events, reader.Data())
	}

	require.NoError(t, reader.Err())
	assert.Len(t, events, 1)
	assert.Equal(t, `{"test": true}`, events[0])
}

func TestHTTPSSEReader_EmptyStream(t *testing.T) {
	rc := io.NopCloser(strings.NewReader(""))
	reader := NewSSEReaderFromReadCloser(rc)
	defer func() { _ = reader.Close() }()

	hasNext := reader.Next()
	assert.False(t, hasNext)
	assert.NoError(t, reader.Err())
}

func TestHTTPSSEReader_OnlyDone(t *testing.T) {
	sseData := "data: [DONE]\n\n"
	rc := io.NopCloser(strings.NewReader(sseData))
	reader := NewSSEReaderFromReadCloser(rc)
	defer func() { _ = reader.Close() }()

	hasNext := reader.Next()
	assert.False(t, hasNext)
	assert.NoError(t, reader.Err())
}

func TestHTTPSSEReader_MultipleDataBeforeDone(t *testing.T) {
	sseData := strings.Join([]string{
		"data: {\"n\": 1}",
		"",
		"data: {\"n\": 2}",
		"",
		"data: {\"n\": 3}",
		"",
		"data: [DONE]",
		"",
	}, "\n")

	rc := io.NopCloser(strings.NewReader(sseData))
	reader := NewSSEReaderFromReadCloser(rc)
	defer func() { _ = reader.Close() }()

	count := 0
	for reader.Next() {
		count++
	}

	assert.Equal(t, 3, count)
	assert.NoError(t, reader.Err())
}

// ============================================================================
// httpSSEWriter Tests
// ============================================================================

func TestHTTPSSEWriter_WriteChunk(t *testing.T) {
	var buf bytes.Buffer
	w := &testHTTPResponseWriter{Buffer: &buf}
	writer := NewHTTPSSEWriter(w)

	chunk := &SSEChunk{
		ID:     "test-chunk",
		Object: "chat.completion.chunk",
		Model:  "gpt-4",
		Choices: []SSEChoice{
			{
				Index: 0,
				Delta: SSEDelta{
					Content: "Test content",
				},
			},
		},
	}

	err := writer.WriteChunk(chunk)
	require.NoError(t, err)

	output := buf.String()
	assert.True(t, strings.HasPrefix(output, "data: "))
	assert.True(t, strings.HasSuffix(output, "\n\n"))
	assert.Contains(t, output, "test-chunk")
	assert.Contains(t, output, "Test content")
}

func TestHTTPSSEWriter_WriteRaw(t *testing.T) {
	var buf bytes.Buffer
	w := &testHTTPResponseWriter{Buffer: &buf}
	writer := NewHTTPSSEWriter(w)

	rawData := []byte("data: {\"raw\": true}\n\n")
	err := writer.WriteRaw(rawData)
	require.NoError(t, err)

	assert.Equal(t, string(rawData), buf.String())
}

func TestHTTPSSEWriter_WriteDone(t *testing.T) {
	var buf bytes.Buffer
	w := &testHTTPResponseWriter{Buffer: &buf}
	writer := NewHTTPSSEWriter(w)

	err := writer.WriteDone()
	require.NoError(t, err)

	assert.Equal(t, "data: [DONE]\n\n", buf.String())
}

func TestHTTPSSEWriter_SetsHeaders(t *testing.T) {
	var buf bytes.Buffer
	w := &testHTTPResponseWriter{Buffer: &buf}

	_ = NewHTTPSSEWriter(w)

	assert.Equal(t, "text/event-stream", w.Header().Get("Content-Type"))
	assert.Equal(t, "no-cache", w.Header().Get("Cache-Control"))
	assert.Equal(t, "keep-alive", w.Header().Get("Connection"))
	assert.Equal(t, "no", w.Header().Get("X-Accel-Buffering"))
}

func TestHTTPSSEWriter_MultipleChunks(t *testing.T) {
	var buf bytes.Buffer
	w := &testHTTPResponseWriter{Buffer: &buf}
	writer := NewHTTPSSEWriter(w)

	// Write multiple chunks
	for i := 0; i < 3; i++ {
		chunk := &SSEChunk{
			ID: "chunk",
			Choices: []SSEChoice{
				{
					Index: 0,
					Delta: SSEDelta{
						Content: "content",
					},
				},
			},
		}
		err := writer.WriteChunk(chunk)
		require.NoError(t, err)
	}

	err := writer.WriteDone()
	require.NoError(t, err)

	output := buf.String()
	// Should have 3 data lines + 1 done
	dataCount := strings.Count(output, "data: ")
	assert.Equal(t, 4, dataCount)
}

// testHTTPResponseWriter implements http.ResponseWriter for testing
type testHTTPResponseWriter struct {
	Buffer  *bytes.Buffer
	headers http.Header
}

func (w *testHTTPResponseWriter) Header() http.Header {
	if w.headers == nil {
		w.headers = make(http.Header)
	}
	return w.headers
}

func (w *testHTTPResponseWriter) Write(data []byte) (int, error) {
	return w.Buffer.Write(data)
}

func (w *testHTTPResponseWriter) WriteHeader(statusCode int) {}

func (w *testHTTPResponseWriter) Flush() {}

// ============================================================================
// RawFunctionCall Tests
// ============================================================================

func TestRawFunctionCall_Marshal(t *testing.T) {
	call := RawFunctionCall{
		Name:       "test_function",
		Parameters: json.RawMessage(`{"key": "value"}`),
	}

	data, err := json.Marshal(call)
	require.NoError(t, err)

	assert.Contains(t, string(data), `"name":"test_function"`)
	assert.Contains(t, string(data), `"parameters":{"key":"value"}`)
}

func TestRawFunctionCall_Unmarshal(t *testing.T) {
	jsonData := `{"name": "my_func", "parameters": {"arg1": 123}}`

	var call RawFunctionCall
	err := json.Unmarshal([]byte(jsonData), &call)
	require.NoError(t, err)

	assert.Equal(t, "my_func", call.Name)
	assert.Equal(t, `{"arg1": 123}`, string(call.Parameters))
}

func TestRawFunctionCall_EmptyParameters(t *testing.T) {
	jsonData := `{"name": "simple_func"}`

	var call RawFunctionCall
	err := json.Unmarshal([]byte(jsonData), &call)
	require.NoError(t, err)

	assert.Equal(t, "simple_func", call.Name)
	assert.Nil(t, call.Parameters)
}

// ============================================================================
// Integration-style Tests
// ============================================================================

func TestSSETypes_RoundTrip(t *testing.T) {
	// Create original chunk
	original := SSEChunk{
		ID:      "roundtrip-test",
		Object:  "chat.completion.chunk",
		Created: 1234567890,
		Model:   "test-model",
		Choices: []SSEChoice{
			{
				Index: 0,
				Delta: SSEDelta{
					Role:    "assistant",
					Content: "Round trip content",
				},
			},
		},
	}

	// Marshal to JSON
	jsonData, err := json.Marshal(original)
	require.NoError(t, err)

	// Parse back
	parsed, err := ParseSSEChunk(string(jsonData))
	require.NoError(t, err)

	// Verify equality
	assert.Equal(t, original.ID, parsed.ID)
	assert.Equal(t, original.Object, parsed.Object)
	assert.Equal(t, original.Created, parsed.Created)
	assert.Equal(t, original.Model, parsed.Model)
	require.Len(t, parsed.Choices, 1)
	assert.Equal(t, original.Choices[0].Delta.Content, parsed.Choices[0].Delta.Content)
}

func TestSSETypes_FullStreamSimulation(t *testing.T) {
	// Simulate a full SSE stream
	var buf bytes.Buffer
	w := &testHTTPResponseWriter{Buffer: &buf}
	writer := NewHTTPSSEWriter(w)

	// Write initial chunk with role
	chunk1 := &SSEChunk{
		ID:     "sim-1",
		Object: "chat.completion.chunk",
		Model:  "gpt-4",
		Choices: []SSEChoice{
			{
				Index: 0,
				Delta: SSEDelta{
					Role: "assistant",
				},
			},
		},
	}
	require.NoError(t, writer.WriteChunk(chunk1))

	// Write content chunks
	for _, content := range []string{"Hello", " ", "World", "!"} {
		chunk := &SSEChunk{
			ID:     "sim-1",
			Object: "chat.completion.chunk",
			Choices: []SSEChoice{
				{
					Index: 0,
					Delta: SSEDelta{
						Content: content,
					},
				},
			},
		}
		require.NoError(t, writer.WriteChunk(chunk))
	}

	// Write finish chunk
	finishChunk := &SSEChunk{
		ID:     "sim-1",
		Object: "chat.completion.chunk",
		Choices: []SSEChoice{
			{
				Index:        0,
				Delta:        SSEDelta{},
				FinishReason: "stop",
			},
		},
	}
	require.NoError(t, writer.WriteChunk(finishChunk))
	require.NoError(t, writer.WriteDone())

	// Now read it back
	output := buf.String()
	rc := io.NopCloser(strings.NewReader(output))
	reader := NewSSEReaderFromReadCloser(rc)
	defer func() { _ = reader.Close() }()

	var chunks []*SSEChunk
	for reader.Next() {
		chunk, err := ParseSSEChunk(reader.Data())
		require.NoError(t, err)
		chunks = append(chunks, chunk)
	}

	require.NoError(t, reader.Err())
	assert.Len(t, chunks, 6) // role + 4 content + finish

	// Reconstruct content
	var content strings.Builder
	for _, chunk := range chunks {
		if len(chunk.Choices) > 0 {
			content.WriteString(chunk.Choices[0].Delta.Content)
		}
	}
	assert.Equal(t, "Hello World!", content.String())
}
