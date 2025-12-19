package tooladapter

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockSSEReader implements SSEStreamReader for testing.
type mockSSEReader struct {
	events []string
	index  int
	err    error
	closed bool
}

func newMockSSEReader(events []string) *mockSSEReader {
	return &mockSSEReader{
		events: events,
		index:  -1,
	}
}

func newMockSSEReaderWithError(events []string, err error) *mockSSEReader {
	return &mockSSEReader{
		events: events,
		index:  -1,
		err:    err,
	}
}

func (m *mockSSEReader) Next() bool {
	if m.closed {
		return false
	}
	m.index++
	// If we have an error and consumed all events, stop and return error via Err()
	if m.err != nil && m.index >= len(m.events) {
		return false
	}
	return m.index < len(m.events)
}

func (m *mockSSEReader) Data() string {
	if m.index < 0 || m.index >= len(m.events) {
		return ""
	}
	return m.events[m.index]
}

func (m *mockSSEReader) Err() error {
	// Return error after all events have been consumed
	if m.err != nil && m.index >= len(m.events) {
		return m.err
	}
	return nil
}

func (m *mockSSEReader) Close() error {
	m.closed = true
	return nil
}

// mockSSEWriter implements SSEStreamWriter for testing.
type mockSSEWriter struct {
	chunks      []*SSEChunk
	rawWrites   [][]byte
	doneWritten bool
}

func newMockSSEWriter() *mockSSEWriter {
	return &mockSSEWriter{
		chunks:    make([]*SSEChunk, 0),
		rawWrites: make([][]byte, 0),
	}
}

func (m *mockSSEWriter) WriteChunk(chunk *SSEChunk) error {
	m.chunks = append(m.chunks, chunk)
	return nil
}

func (m *mockSSEWriter) WriteRaw(data []byte) error {
	m.rawWrites = append(m.rawWrites, append([]byte(nil), data...))
	return nil
}

func (m *mockSSEWriter) WriteDone() error {
	m.doneWritten = true
	return nil
}

func (m *mockSSEWriter) Flush() {}

// Helper to create an SSE chunk JSON string
func createSSEChunkJSON(id, model, content string, finishReason string) string {
	chunk := SSEChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: 1234567890,
		Model:   model,
		Choices: []SSEChoice{
			{
				Index: 0,
				Delta: SSEDelta{
					Content: content,
					Role:    "assistant",
				},
				FinishReason: finishReason,
			},
		},
	}
	data, _ := json.Marshal(chunk)
	return string(data)
}

// ============================================================================
// Basic Functionality Tests
// ============================================================================

func TestSSEStreamAdapter_BasicPassthrough(t *testing.T) {
	// Test that non-tool content is passed through unchanged
	events := []string{
		createSSEChunkJSON("chatcmpl-123", "gpt-4", "Hello ", ""),
		createSSEChunkJSON("chatcmpl-123", "gpt-4", "world!", ""),
		createSSEChunkJSON("chatcmpl-123", "gpt-4", "", "stop"),
	}

	reader := newMockSSEReader(events)
	writer := newMockSSEWriter()

	adapter := New(WithLogLevel(slog.LevelError))
	sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)

	err := sseAdapter.Process(context.Background())
	require.NoError(t, err)

	// Should pass through all chunks
	assert.Equal(t, len(events), len(writer.rawWrites))
	assert.True(t, writer.doneWritten)
}

func TestSSEStreamAdapter_ToolCallDetection(t *testing.T) {
	// Test that tool calls are detected and transformed
	toolJSON := `[{"name": "get_weather", "parameters": {"city": "Seattle"}}]`
	events := []string{
		createSSEChunkJSON("chatcmpl-123", "gpt-4", toolJSON, ""),
		createSSEChunkJSON("chatcmpl-123", "gpt-4", "", "stop"),
	}

	reader := newMockSSEReader(events)
	writer := newMockSSEWriter()

	adapter := New(WithLogLevel(slog.LevelError))
	sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)

	err := sseAdapter.Process(context.Background())
	require.NoError(t, err)

	// Should emit transformed chunks (tool call + finish), not raw passthrough
	assert.Equal(t, 2, len(writer.chunks), "Expected 2 chunks: tool call and finish")
	assert.Empty(t, writer.rawWrites, "Should not have raw writes when tools detected")

	// Verify tool call chunk
	toolChunk := writer.chunks[0]
	require.Len(t, toolChunk.Choices, 1)
	require.Len(t, toolChunk.Choices[0].Delta.ToolCalls, 1)
	assert.Equal(t, "get_weather", toolChunk.Choices[0].Delta.ToolCalls[0].Function.Name)
	assert.Contains(t, toolChunk.Choices[0].Delta.ToolCalls[0].Function.Arguments, "Seattle")

	// Verify finish chunk
	finishChunk := writer.chunks[1]
	assert.Equal(t, "tool_calls", finishChunk.Choices[0].FinishReason)
}

func TestSSEStreamAdapter_MultipleToolCalls(t *testing.T) {
	// Test detecting multiple tool calls in array
	toolJSON := `[{"name": "get_weather", "parameters": {"city": "Seattle"}}, {"name": "get_time", "parameters": {"timezone": "PST"}}]`
	events := []string{
		createSSEChunkJSON("chatcmpl-123", "gpt-4", toolJSON, ""),
		createSSEChunkJSON("chatcmpl-123", "gpt-4", "", "stop"),
	}

	reader := newMockSSEReader(events)
	writer := newMockSSEWriter()

	adapter := New(
		WithLogLevel(slog.LevelError),
		WithToolPolicy(ToolDrainAll), // Allow multiple tools
	)
	sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)

	err := sseAdapter.Process(context.Background())
	require.NoError(t, err)

	// Should have 2 chunks
	require.Len(t, writer.chunks, 2)

	// Tool chunk should have 2 tool calls
	toolChunk := writer.chunks[0]
	require.Len(t, toolChunk.Choices[0].Delta.ToolCalls, 2)
	assert.Equal(t, "get_weather", toolChunk.Choices[0].Delta.ToolCalls[0].Function.Name)
	assert.Equal(t, "get_time", toolChunk.Choices[0].Delta.ToolCalls[1].Function.Name)
}

func TestSSEStreamAdapter_ToolInMarkdownCodeBlock(t *testing.T) {
	// Test detecting tool calls in markdown code blocks
	content := "Here's the weather data:\n```json\n" +
		`[{"name": "get_weather", "parameters": {"city": "NYC"}}]` +
		"\n```"
	events := []string{
		createSSEChunkJSON("chatcmpl-123", "gpt-4", content, ""),
		createSSEChunkJSON("chatcmpl-123", "gpt-4", "", "stop"),
	}

	reader := newMockSSEReader(events)
	writer := newMockSSEWriter()

	adapter := New(WithLogLevel(slog.LevelError))
	sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)

	err := sseAdapter.Process(context.Background())
	require.NoError(t, err)

	// Should detect and transform the tool call
	require.Len(t, writer.chunks, 2)
	toolChunk := writer.chunks[0]
	require.Len(t, toolChunk.Choices[0].Delta.ToolCalls, 1)
	assert.Equal(t, "get_weather", toolChunk.Choices[0].Delta.ToolCalls[0].Function.Name)
}

// ============================================================================
// Tool Policy Tests
// ============================================================================

func TestSSEStreamAdapter_PolicyStopOnFirst(t *testing.T) {
	// Test ToolStopOnFirst policy limits to single tool
	toolJSON := `[{"name": "tool1", "parameters": {}}, {"name": "tool2", "parameters": {}}]`
	events := []string{
		createSSEChunkJSON("chatcmpl-123", "gpt-4", toolJSON, ""),
	}

	reader := newMockSSEReader(events)
	writer := newMockSSEWriter()

	adapter := New(
		WithLogLevel(slog.LevelError),
		WithToolPolicy(ToolStopOnFirst),
	)
	sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)

	err := sseAdapter.Process(context.Background())
	require.NoError(t, err)

	require.Len(t, writer.chunks, 2)
	toolChunk := writer.chunks[0]
	assert.Len(t, toolChunk.Choices[0].Delta.ToolCalls, 1)
	assert.Equal(t, "tool1", toolChunk.Choices[0].Delta.ToolCalls[0].Function.Name)
}

func TestSSEStreamAdapter_PolicyDrainAll(t *testing.T) {
	// Test ToolDrainAll policy allows all tools
	toolJSON := `[{"name": "tool1", "parameters": {}}, {"name": "tool2", "parameters": {}}, {"name": "tool3", "parameters": {}}]`
	events := []string{
		createSSEChunkJSON("chatcmpl-123", "gpt-4", toolJSON, ""),
	}

	reader := newMockSSEReader(events)
	writer := newMockSSEWriter()

	adapter := New(
		WithLogLevel(slog.LevelError),
		WithToolPolicy(ToolDrainAll),
	)
	sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)

	err := sseAdapter.Process(context.Background())
	require.NoError(t, err)

	require.Len(t, writer.chunks, 2)
	toolChunk := writer.chunks[0]
	assert.Len(t, toolChunk.Choices[0].Delta.ToolCalls, 3)
}

func TestSSEStreamAdapter_PolicyMaxCalls(t *testing.T) {
	// Test that WithToolMaxCalls limits tool count
	toolJSON := `[{"name": "t1", "parameters": {}}, {"name": "t2", "parameters": {}}, {"name": "t3", "parameters": {}}, {"name": "t4", "parameters": {}}]`
	events := []string{
		createSSEChunkJSON("chatcmpl-123", "gpt-4", toolJSON, ""),
	}

	reader := newMockSSEReader(events)
	writer := newMockSSEWriter()

	adapter := New(
		WithLogLevel(slog.LevelError),
		WithToolPolicy(ToolDrainAll),
		WithToolMaxCalls(2),
	)
	sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)

	err := sseAdapter.Process(context.Background())
	require.NoError(t, err)

	require.Len(t, writer.chunks, 2)
	toolChunk := writer.chunks[0]
	assert.Len(t, toolChunk.Choices[0].Delta.ToolCalls, 2)
}

// ============================================================================
// Early Detection Tests
// ============================================================================

func TestSSEStreamAdapter_EarlyDetectionWithToolPattern(t *testing.T) {
	// Test early detection when tool pattern is detected
	events := []string{
		createSSEChunkJSON("chatcmpl-123", "gpt-4", `[{"name":`, ""),
		createSSEChunkJSON("chatcmpl-123", "gpt-4", ` "test_func", "parameters": {}}]`, ""),
		createSSEChunkJSON("chatcmpl-123", "gpt-4", "", "stop"),
	}

	reader := newMockSSEReader(events)
	writer := newMockSSEWriter()

	adapter := New(WithLogLevel(slog.LevelError))
	sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)

	err := sseAdapter.ProcessWithPassthrough(context.Background(), 50)
	require.NoError(t, err)

	// Should detect tools and transform
	require.Len(t, writer.chunks, 2)
	assert.Equal(t, "test_func", writer.chunks[0].Choices[0].Delta.ToolCalls[0].Function.Name)
}

func TestSSEStreamAdapter_EarlyDetectionNoToolPattern(t *testing.T) {
	// Test early detection passes through when no tool pattern in early content
	events := []string{
		createSSEChunkJSON("chatcmpl-123", "gpt-4", "Hello, this is a ", ""),
		createSSEChunkJSON("chatcmpl-123", "gpt-4", "normal response without ", ""),
		createSSEChunkJSON("chatcmpl-123", "gpt-4", "any tool calls in it.", ""),
		createSSEChunkJSON("chatcmpl-123", "gpt-4", "", "stop"),
	}

	reader := newMockSSEReader(events)
	writer := newMockSSEWriter()

	adapter := New(WithLogLevel(slog.LevelError))
	sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)

	// Early detection with small window
	err := sseAdapter.ProcessWithPassthrough(context.Background(), 20)
	require.NoError(t, err)

	// Should pass through since no tool pattern in first 20 chars
	assert.NotEmpty(t, writer.rawWrites)
	assert.True(t, writer.doneWritten)
}

func TestSSEStreamAdapter_EarlyDetectionDisabled(t *testing.T) {
	// Test that early detection with 0 falls back to Process
	events := []string{
		createSSEChunkJSON("chatcmpl-123", "gpt-4", "Just text", ""),
	}

	reader := newMockSSEReader(events)
	writer := newMockSSEWriter()

	adapter := New(WithLogLevel(slog.LevelError))
	sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)

	err := sseAdapter.ProcessWithPassthrough(context.Background(), 0)
	require.NoError(t, err)

	// Should still work normally
	assert.NotEmpty(t, writer.rawWrites)
}

// ============================================================================
// ProcessToResult Tests
// ============================================================================

func TestSSEStreamAdapter_ProcessToResult_NoTools(t *testing.T) {
	events := []string{
		createSSEChunkJSON("chatcmpl-123", "gpt-4", "Regular content", ""),
	}

	reader := newMockSSEReader(events)
	writer := newMockSSEWriter()

	adapter := New(WithLogLevel(slog.LevelError))
	sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)

	result, chunks, err := sseAdapter.ProcessToResult(context.Background())
	require.NoError(t, err)

	assert.False(t, result.HasToolCalls)
	assert.True(t, result.Passthrough)
	assert.Equal(t, "Regular content", result.Content)
	assert.Len(t, chunks, 1)
}

func TestSSEStreamAdapter_ProcessToResult_WithTools(t *testing.T) {
	toolJSON := `[{"name": "search", "parameters": {"query": "test"}}]`
	events := []string{
		createSSEChunkJSON("chatcmpl-123", "gpt-4", toolJSON, ""),
	}

	reader := newMockSSEReader(events)
	writer := newMockSSEWriter()

	adapter := New(WithLogLevel(slog.LevelError))
	sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)

	result, chunks, err := sseAdapter.ProcessToResult(context.Background())
	require.NoError(t, err)

	assert.True(t, result.HasToolCalls)
	assert.False(t, result.Passthrough)
	require.Len(t, result.ToolCalls, 1)
	assert.Equal(t, "search", result.ToolCalls[0].Name)
	assert.Len(t, chunks, 1)
}

func TestSSEStreamAdapter_WriteToolCallsFromResult(t *testing.T) {
	toolJSON := `[{"name": "execute", "parameters": {"cmd": "ls"}}]`
	events := []string{
		createSSEChunkJSON("chatcmpl-123", "gpt-4", toolJSON, ""),
	}

	reader := newMockSSEReader(events)
	writer := newMockSSEWriter()

	adapter := New(WithLogLevel(slog.LevelError))
	sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)

	result, _, err := sseAdapter.ProcessToResult(context.Background())
	require.NoError(t, err)
	require.True(t, result.HasToolCalls)

	// Now write the result
	err = sseAdapter.WriteToolCallsFromResult(result)
	require.NoError(t, err)

	assert.Len(t, writer.chunks, 2) // tool chunk + finish chunk
	assert.Equal(t, "execute", writer.chunks[0].Choices[0].Delta.ToolCalls[0].Function.Name)
	assert.True(t, writer.doneWritten)
}

func TestSSEStreamAdapter_WritePassthrough(t *testing.T) {
	events := []string{
		createSSEChunkJSON("chatcmpl-123", "gpt-4", "Hello", ""),
		createSSEChunkJSON("chatcmpl-123", "gpt-4", " World", ""),
	}

	reader := newMockSSEReader(events)
	writer := newMockSSEWriter()

	adapter := New(WithLogLevel(slog.LevelError))
	sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)

	result, chunks, err := sseAdapter.ProcessToResult(context.Background())
	require.NoError(t, err)
	require.True(t, result.Passthrough)

	// Write passthrough
	err = sseAdapter.WritePassthrough(chunks)
	require.NoError(t, err)

	assert.Len(t, writer.chunks, 2)
	assert.True(t, writer.doneWritten)
}

// ============================================================================
// Error Handling Tests
// ============================================================================

func TestSSEStreamAdapter_ContextCancellation(t *testing.T) {
	events := []string{
		createSSEChunkJSON("chatcmpl-123", "gpt-4", "Hello", ""),
		createSSEChunkJSON("chatcmpl-123", "gpt-4", " World", ""),
	}

	reader := newMockSSEReader(events)
	writer := newMockSSEWriter()

	adapter := New(WithLogLevel(slog.LevelError))
	sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	err := sseAdapter.Process(ctx)
	assert.ErrorIs(t, err, context.Canceled)
}

func TestSSEStreamAdapter_ReaderError(t *testing.T) {
	expectedErr := errors.New("reader error")
	events := []string{
		createSSEChunkJSON("chatcmpl-123", "gpt-4", "Hello", ""),
	}

	reader := newMockSSEReaderWithError(events, expectedErr)
	writer := newMockSSEWriter()

	adapter := New(WithLogLevel(slog.LevelError))
	sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)

	err := sseAdapter.Process(context.Background())
	assert.ErrorIs(t, err, expectedErr)
}

func TestSSEStreamAdapter_MalformedJSONChunk(t *testing.T) {
	// Malformed JSON should be passed through
	events := []string{
		"not valid json",
		createSSEChunkJSON("chatcmpl-123", "gpt-4", "Valid content", ""),
	}

	reader := newMockSSEReader(events)
	writer := newMockSSEWriter()

	adapter := New(WithLogLevel(slog.LevelError))
	sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)

	err := sseAdapter.Process(context.Background())
	require.NoError(t, err)

	// Should have raw writes for both (malformed passed through, valid parsed and passed through)
	assert.NotEmpty(t, writer.rawWrites)
}

func TestSSEStreamAdapter_EmptyContent(t *testing.T) {
	events := []string{
		createSSEChunkJSON("chatcmpl-123", "gpt-4", "", ""),
		createSSEChunkJSON("chatcmpl-123", "gpt-4", "", "stop"),
	}

	reader := newMockSSEReader(events)
	writer := newMockSSEWriter()

	adapter := New(WithLogLevel(slog.LevelError))
	sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)

	err := sseAdapter.Process(context.Background())
	require.NoError(t, err)

	// Should pass through
	assert.NotEmpty(t, writer.rawWrites)
	assert.True(t, writer.doneWritten)
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

func TestSSEStreamAdapter_SingleToolObject(t *testing.T) {
	// Test single tool object (not array)
	toolJSON := `{"name": "single_tool", "parameters": {"key": "value"}}`
	events := []string{
		createSSEChunkJSON("chatcmpl-123", "gpt-4", toolJSON, ""),
	}

	reader := newMockSSEReader(events)
	writer := newMockSSEWriter()

	adapter := New(WithLogLevel(slog.LevelError))
	sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)

	err := sseAdapter.Process(context.Background())
	require.NoError(t, err)

	require.Len(t, writer.chunks, 2)
	assert.Equal(t, "single_tool", writer.chunks[0].Choices[0].Delta.ToolCalls[0].Function.Name)
}

func TestSSEStreamAdapter_InvalidFunctionName(t *testing.T) {
	// Invalid function names should cause passthrough
	toolJSON := `[{"name": "", "parameters": {}}]`
	events := []string{
		createSSEChunkJSON("chatcmpl-123", "gpt-4", toolJSON, ""),
	}

	reader := newMockSSEReader(events)
	writer := newMockSSEWriter()

	adapter := New(WithLogLevel(slog.LevelError))
	sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)

	err := sseAdapter.Process(context.Background())
	require.NoError(t, err)

	// Should pass through since function name is invalid
	assert.NotEmpty(t, writer.rawWrites)
}

func TestSSEStreamAdapter_MixedValidInvalidTools(t *testing.T) {
	// Array with some invalid function names
	toolJSON := `[{"name": "", "parameters": {}}, {"name": "valid_tool", "parameters": {}}]`
	events := []string{
		createSSEChunkJSON("chatcmpl-123", "gpt-4", toolJSON, ""),
	}

	reader := newMockSSEReader(events)
	writer := newMockSSEWriter()

	adapter := New(WithLogLevel(slog.LevelError))
	sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)

	err := sseAdapter.Process(context.Background())
	require.NoError(t, err)

	// Should pass through since validation fails on the whole array
	assert.NotEmpty(t, writer.rawWrites)
}

func TestSSEStreamAdapter_ToolWithComplexParameters(t *testing.T) {
	toolJSON := `[{"name": "complex_func", "parameters": {"nested": {"deep": {"value": 123}}, "array": [1, 2, 3], "unicode": "🎉"}}]`
	events := []string{
		createSSEChunkJSON("chatcmpl-123", "gpt-4", toolJSON, ""),
	}

	reader := newMockSSEReader(events)
	writer := newMockSSEWriter()

	adapter := New(WithLogLevel(slog.LevelError))
	sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)

	err := sseAdapter.Process(context.Background())
	require.NoError(t, err)

	require.Len(t, writer.chunks, 2)
	args := writer.chunks[0].Choices[0].Delta.ToolCalls[0].Function.Arguments
	assert.Contains(t, args, "nested")
	assert.Contains(t, args, "🎉")
}

func TestSSEStreamAdapter_MetadataCapture(t *testing.T) {
	// Verify metadata is captured from first chunk
	events := []string{
		createSSEChunkJSON("chatcmpl-unique-id", "test-model", `[{"name": "test", "parameters": {}}]`, ""),
	}

	reader := newMockSSEReader(events)
	writer := newMockSSEWriter()

	adapter := New(WithLogLevel(slog.LevelError))
	sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)

	err := sseAdapter.Process(context.Background())
	require.NoError(t, err)

	// Check metadata is propagated to output
	require.Len(t, writer.chunks, 2)
	assert.Equal(t, "chatcmpl-unique-id", writer.chunks[0].ID)
	assert.Equal(t, "test-model", writer.chunks[0].Model)
}

// ============================================================================
// SSEStreamReader/Writer Implementation Tests
// ============================================================================

func TestHTTPSSEReader_Basic(t *testing.T) {
	// Test the httpSSEReader implementation
	sseData := strings.Join([]string{
		"data: {\"id\": \"1\", \"choices\": [{\"delta\": {\"content\": \"Hello\"}}]}",
		"",
		"data: {\"id\": \"1\", \"choices\": [{\"delta\": {\"content\": \" World\"}}]}",
		"",
		"data: [DONE]",
		"",
	}, "\n")

	rc := io.NopCloser(strings.NewReader(sseData))
	reader := NewSSEReaderFromReadCloser(rc)
	defer func() { _ = reader.Close() }()

	var contents []string
	for reader.Next() {
		data := reader.Data()
		chunk, err := ParseSSEChunk(data)
		require.NoError(t, err)
		if len(chunk.Choices) > 0 {
			contents = append(contents, chunk.Choices[0].Delta.Content)
		}
	}

	assert.NoError(t, reader.Err())
	assert.Equal(t, []string{"Hello", " World"}, contents)
}

func TestHTTPSSEReader_SkipsComments(t *testing.T) {
	sseData := strings.Join([]string{
		": this is a comment",
		"data: {\"id\": \"1\", \"choices\": [{\"delta\": {\"content\": \"content\"}}]}",
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

	assert.NoError(t, reader.Err())
	assert.Equal(t, 1, count) // Only the data line, not the comment
}

func TestHTTPSSEReader_HandlesEmptyLines(t *testing.T) {
	sseData := strings.Join([]string{
		"",
		"",
		"data: {\"id\": \"1\", \"choices\": [{\"delta\": {\"content\": \"test\"}}]}",
		"",
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

	assert.NoError(t, reader.Err())
	assert.Equal(t, 1, count)
}

// ============================================================================
// hasToolPattern Tests
// ============================================================================

func TestSSEStreamAdapter_HasToolPattern(t *testing.T) {
	adapter := New(WithLogLevel(slog.LevelError))
	sseAdapter := adapter.NewSSEStreamAdapter(nil, nil)

	tests := []struct {
		name     string
		content  string
		expected bool
	}{
		{
			name:     "JSON array start",
			content:  `[{"name": "test"`,
			expected: true,
		},
		{
			name:     "JSON object start",
			content:  `{"name": "test"`,
			expected: true,
		},
		{
			name:     "Markdown code block",
			content:  "```json\n{\"name\": \"test\"}",
			expected: true,
		},
		{
			name:     "Backtick enclosed",
			content:  "Here is: `{\"name\": \"func\"}",
			expected: true,
		},
		{
			name:     "Regular text",
			content:  "Hello, how can I help you today?",
			expected: false,
		},
		{
			name:     "Empty string",
			content:  "",
			expected: false,
		},
		{
			name:     "Whitespace only",
			content:  "   \n\t  ",
			expected: false,
		},
		{
			name:     "JSON-like but no name",
			content:  `{"key": "value"}`,
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := sseAdapter.hasToolPattern(tt.content)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// ============================================================================
// Benchmark Tests
// ============================================================================

func BenchmarkSSEStreamAdapter_Process_NoTools(b *testing.B) {
	adapter := New(WithLogLevel(slog.LevelError))

	events := []string{
		createSSEChunkJSON("chatcmpl-123", "gpt-4", "Hello ", ""),
		createSSEChunkJSON("chatcmpl-123", "gpt-4", "world! ", ""),
		createSSEChunkJSON("chatcmpl-123", "gpt-4", "This is a test.", ""),
		createSSEChunkJSON("chatcmpl-123", "gpt-4", "", "stop"),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reader := newMockSSEReader(events)
		writer := newMockSSEWriter()
		sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)
		_ = sseAdapter.Process(context.Background())
	}
}

func BenchmarkSSEStreamAdapter_Process_WithTools(b *testing.B) {
	adapter := New(WithLogLevel(slog.LevelError))

	toolJSON := `[{"name": "get_weather", "parameters": {"city": "Seattle", "units": "fahrenheit"}}]`
	events := []string{
		createSSEChunkJSON("chatcmpl-123", "gpt-4", toolJSON, ""),
		createSSEChunkJSON("chatcmpl-123", "gpt-4", "", "stop"),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reader := newMockSSEReader(events)
		writer := newMockSSEWriter()
		sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)
		_ = sseAdapter.Process(context.Background())
	}
}

func BenchmarkSSEStreamAdapter_ProcessWithPassthrough(b *testing.B) {
	adapter := New(WithLogLevel(slog.LevelError))

	events := []string{
		createSSEChunkJSON("chatcmpl-123", "gpt-4", "Hello world this is regular content", ""),
		createSSEChunkJSON("chatcmpl-123", "gpt-4", " without any tool calls.", ""),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reader := newMockSSEReader(events)
		writer := newMockSSEWriter()
		sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)
		_ = sseAdapter.ProcessWithPassthrough(context.Background(), 50)
	}
}

// ============================================================================
// Integration-style Tests
// ============================================================================

func TestSSEStreamAdapter_ProgressiveJSONBuildup(t *testing.T) {
	// Simulate tool JSON arriving in multiple chunks
	events := []string{
		createSSEChunkJSON("chatcmpl-123", "gpt-4", "[", ""),
		createSSEChunkJSON("chatcmpl-123", "gpt-4", `{"name": "test_func",`, ""),
		createSSEChunkJSON("chatcmpl-123", "gpt-4", ` "parameters": {"arg": "val"}`, ""),
		createSSEChunkJSON("chatcmpl-123", "gpt-4", `}]`, ""),
	}

	reader := newMockSSEReader(events)
	writer := newMockSSEWriter()

	adapter := New(WithLogLevel(slog.LevelError))
	sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)

	err := sseAdapter.Process(context.Background())
	require.NoError(t, err)

	// Should assemble and detect the complete tool call
	require.Len(t, writer.chunks, 2)
	assert.Equal(t, "test_func", writer.chunks[0].Choices[0].Delta.ToolCalls[0].Function.Name)
}

func TestSSEStreamAdapter_LongRunningStream(t *testing.T) {
	// Test with many chunks
	events := make([]string, 0, 102)
	for i := 0; i < 100; i++ {
		events = append(events, createSSEChunkJSON("chatcmpl-123", "gpt-4", "word ", ""))
	}
	events = append(events, createSSEChunkJSON("chatcmpl-123", "gpt-4", `[{"name": "final_tool", "parameters": {}}]`, ""))
	events = append(events, createSSEChunkJSON("chatcmpl-123", "gpt-4", "", "stop"))

	reader := newMockSSEReader(events)
	writer := newMockSSEWriter()

	adapter := New(WithLogLevel(slog.LevelError))
	sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)

	err := sseAdapter.Process(context.Background())
	require.NoError(t, err)

	// Should detect the tool at the end
	require.Len(t, writer.chunks, 2)
	assert.Equal(t, "final_tool", writer.chunks[0].Choices[0].Delta.ToolCalls[0].Function.Name)
}

func TestSSEStreamAdapter_TimeoutContext(t *testing.T) {
	events := make([]string, 1000)
	for i := range events {
		events[i] = createSSEChunkJSON("chatcmpl-123", "gpt-4", "content ", "")
	}

	// Create a slow reader that will trigger timeout
	reader := &slowMockSSEReader{
		mockSSEReader: newMockSSEReader(events),
		delay:         10 * time.Millisecond,
	}
	writer := newMockSSEWriter()

	adapter := New(WithLogLevel(slog.LevelError))
	sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	err := sseAdapter.Process(ctx)
	// Should timeout before processing all chunks
	assert.ErrorIs(t, err, context.DeadlineExceeded)
}

// slowMockSSEReader adds delays for timeout testing
type slowMockSSEReader struct {
	*mockSSEReader
	delay time.Duration
}

func (s *slowMockSSEReader) Next() bool {
	time.Sleep(s.delay)
	return s.mockSSEReader.Next()
}
