package tooladapter

import (
	"fmt"

	"github.com/openai/openai-go/v2"
)

// MockStream implements ChatCompletionStreamInterface for testing
type MockStream struct {
	chunks    []string
	index     int
	err       error
	closed    bool
	hasFinish bool
}

// NewMockStream creates a new mock stream with the given content chunks
func NewMockStream(chunks []string) *MockStream {
	return &MockStream{
		chunks: chunks,
		index:  -1, // Start before first element
	}
}

// NewMockStreamWithError creates a mock stream that returns an error
func NewMockStreamWithError(err error) *MockStream {
	return &MockStream{
		chunks: nil,
		index:  -1,
		err:    err,
	}
}

// Next advances to the next chunk
func (m *MockStream) Next() bool {
	if m.closed || m.err != nil {
		return false
	}

	m.index++

	// If we've processed all chunks and haven't sent a finish chunk yet
	if m.index >= len(m.chunks) {
		if !m.hasFinish {
			m.hasFinish = true
			return true // Return true to emit the finish chunk
		}
		return false // No more chunks
	}

	return true
}

// Current returns the current chunk
func (m *MockStream) Current() openai.ChatCompletionChunk {
	if m.index < 0 {
		return openai.ChatCompletionChunk{}
	}

	// If we're past the content chunks, return a finish chunk
	if m.index >= len(m.chunks) {
		return openai.ChatCompletionChunk{
			Choices: []openai.ChatCompletionChunkChoice{
				{
					FinishReason: "stop",
					Delta: openai.ChatCompletionChunkChoiceDelta{
						Role: "assistant",
					},
				},
			},
		}
	}

	content := m.chunks[m.index]
	return openai.ChatCompletionChunk{
		Choices: []openai.ChatCompletionChunkChoice{
			{
				Delta: openai.ChatCompletionChunkChoiceDelta{
					Content: content,
					Role:    "assistant",
				},
			},
		},
	}
}

// Err returns any error
func (m *MockStream) Err() error {
	return m.err
}

// Close closes the stream
func (m *MockStream) Close() error {
	m.closed = true
	return nil
}

// MockStreamWithFinishReason creates a stream that ends with a specific finish reason
func NewMockStreamWithFinishReason(chunks []string, finishReason string) *MockStreamWithFinishReason {
	return &MockStreamWithFinishReason{
		MockStream:   NewMockStream(chunks),
		finishReason: finishReason,
	}
}

// MockStreamWithFinishReason extends MockStream with custom finish reasons
type MockStreamWithFinishReason struct {
	*MockStream
	finishReason string
}

func (m *MockStreamWithFinishReason) Current() openai.ChatCompletionChunk {
	if m.index < 0 {
		return openai.ChatCompletionChunk{}
	}

	// If we're past the content chunks, return a finish chunk with custom reason
	if m.index >= len(m.chunks) {
		return openai.ChatCompletionChunk{
			Choices: []openai.ChatCompletionChunkChoice{
				{
					FinishReason: m.finishReason,
					Delta: openai.ChatCompletionChunkChoiceDelta{
						Role: "assistant",
					},
				},
			},
		}
	}

	// Return regular content chunk
	return m.MockStream.Current()
}

// MockStreamWithDelay simulates streaming delays (for timeout testing)
type MockStreamWithDelay struct {
	*MockStream
	delays []int // Millisecond delays for each chunk
}

func NewMockStreamWithDelay(chunks []string, delays []int) *MockStreamWithDelay {
	return &MockStreamWithDelay{
		MockStream: NewMockStream(chunks),
		delays:     delays,
	}
}

// MockStreamWithBufferLimitTest creates a stream for testing buffer limits
func NewMockStreamForBufferLimit(contentSize int) *MockStream {
	// Create a large content chunk
	largeContent := ""
	for i := 0; i < contentSize; i++ {
		largeContent += "A"
	}

	chunks := []string{
		largeContent,
		`[{"name": "test_tool", "parameters": {"test": "value"}}]`,
	}

	return NewMockStream(chunks)
}

// MockStreamWithVariousChunkTypes creates a stream with different chunk types for comprehensive testing
func NewMockStreamWithVariousChunkTypes() *MockStream {
	return &MockStream{
		chunks: []string{
			"",                             // Empty chunk
			"   ",                          // Whitespace only
			"Regular content before tools", // Regular content
			"Here comes a tool call:",      // Prefatory content
			`[{"name": "get_weather", "parameters": {"city": "Seattle"}}]`, // Tool call
			"And some content after", // Content after tool
		},
		index: -1,
	}
}

// MockStreamWithMalformedJSON creates a stream with malformed JSON for error testing
func NewMockStreamWithMalformedJSON() *MockStream {
	return &MockStream{
		chunks: []string{
			`[{"name": "broken_tool", "parameters":`, // Incomplete JSON
			`{"missing": "closing"}`,                 // No closing brackets
			`invalid json content`,                   // Not JSON at all
		},
		index: -1,
	}
}

// MockStreamWithMultipleToolCalls creates a stream with multiple sequential tool calls
func NewMockStreamWithMultipleToolCalls() *MockStream {
	return &MockStream{
		chunks: []string{
			`[{"name": "tool1", "parameters": {"a": 1}}]`,
			" Some content between tools",
			`[{"name": "tool2", "parameters": {"b": 2}}]`,
			" More content",
			`[{"name": "tool3", "parameters": {"c": 3}}]`,
			" Final content",
		},
		index: -1,
	}
}

// MockStreamWithProgressiveJSON creates a stream that builds JSON progressively
func NewMockStreamWithProgressiveJSON() *MockStream {
	return &MockStream{
		chunks: []string{
			`[`,
			`{"name": "progressive_tool",`,
			` "parameters": {`,
			`"city": "Seattle",`,
			` "units": "fahrenheit"`,
			`}}`,
			`]`,
			" Content after progressive JSON",
		},
		index: -1,
	}
}

// Helper function to create a mock stream that simulates real-world chunking patterns
func NewRealisticMockStream(toolName string, parameters string, additionalContent string) *MockStream {
	chunks := []string{
		"I'll help you with that. ", // Natural language start
		"Let me ",                   // Partial sentence
		"use the appropriate tool.", // Complete sentence
		fmt.Sprintf(`[{"name": "%s", "parameters": %s}]`, toolName, parameters), // Tool call
	}

	if additionalContent != "" {
		chunks = append(chunks, additionalContent)
	}

	return NewMockStream(chunks)
}
