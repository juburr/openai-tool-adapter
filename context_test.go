package tooladapter

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type contextKey int

const (
	ctxKeyRequestID contextKey = iota
)

// Test helper functions specific to context tests
func createMockTool(name, description string) openai.ChatCompletionToolParam {
	tool := openai.ChatCompletionToolParam{
		Type: "function",
		Function: openai.FunctionDefinitionParam{
			Name: name,
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]interface{}{
					"param1": map[string]interface{}{
						"type":        "string",
						"description": "A parameter",
					},
				},
			},
		},
	}

	// Only set description if it's not empty
	if description != "" {
		tool.Function.Description = openai.String(description)
	}

	return tool
}

func createMockRequest(tools []openai.ChatCompletionToolParam) openai.ChatCompletionNewParams {
	return openai.ChatCompletionNewParams{
		Model: openai.ChatModelGPT4o,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("Hello, please help me."),
		},
		Tools: tools,
	}
}

func createMockCompletion(content string) openai.ChatCompletion {
	return openai.ChatCompletion{
		Choices: []openai.ChatCompletionChoice{
			{
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: content,
				},
			},
		},
	}
}

// ============================================================================
// CONTEXT CANCELLATION TESTS
// ============================================================================

func TestTransformCompletionsRequestWithContext_ImmediateCancellation(t *testing.T) {
	adapter := New()

	// Pre-cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	req := createMockRequest([]openai.ChatCompletionToolParam{
		createMockTool("test_func", "Test function"),
	})

	_, err := adapter.TransformCompletionsRequestWithContext(ctx, req)
	assert.Error(t, err)
	assert.True(t, errors.Is(err, context.Canceled), "Expected context.Canceled, got: %v", err)
}

func TestTransformCompletionsRequestWithContext_CancellationDuringProcessing(t *testing.T) {
	adapter := New()

	// Create many tools to increase processing time
	tools := make([]openai.ChatCompletionToolParam, 50)
	for i := 0; i < 50; i++ {
		tools[i] = createMockTool("test_func_"+string(rune('A'+i)), "Test function")
	}

	req := createMockRequest(tools)

	ctx, cancel := context.WithCancel(context.Background())

	// Cancel after a very short time
	go func() {
		time.Sleep(1 * time.Millisecond)
		cancel()
	}()

	_, err := adapter.TransformCompletionsRequestWithContext(ctx, req)
	// Should either succeed (if very fast) or be cancelled
	if err != nil {
		assert.True(t, errors.Is(err, context.Canceled), "Expected context.Canceled, got: %v", err)
	}
}

func TestTransformCompletionsResponseWithContext_ImmediateCancellation(t *testing.T) {
	adapter := New()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	resp := createMockCompletion(`[{"name": "test_func", "parameters": {"param1": "value1"}}]`)

	_, err := adapter.TransformCompletionsResponseWithContext(ctx, resp)
	assert.Error(t, err)
	assert.True(t, errors.Is(err, context.Canceled), "Expected context.Canceled, got: %v", err)
}

func TestTransformCompletionsResponseWithContext_CancellationDuringJSONParsing(t *testing.T) {
	adapter := New()

	// Create a large, complex JSON response to increase parsing time
	largeJSON := `[`
	for i := 0; i < 100; i++ {
		if i > 0 {
			largeJSON += ","
		}
		largeJSON += `{"name": "func_` + string(rune('A'+i%26)) + `", "parameters": {"data": {"nested": {"deep": {"value": ` + string(rune('0'+i%10)) + `}}}}}`
	}
	largeJSON += `]`

	resp := createMockCompletion(largeJSON)

	ctx, cancel := context.WithCancel(context.Background())

	// Cancel after a short time
	go func() {
		time.Sleep(1 * time.Millisecond)
		cancel()
	}()

	_, err := adapter.TransformCompletionsResponseWithContext(ctx, resp)
	// Should either succeed (if very fast) or be cancelled
	if err != nil {
		assert.True(t, errors.Is(err, context.Canceled), "Expected context.Canceled, got: %v", err)
	}
}

// ============================================================================
// TIMEOUT TESTS
// ============================================================================

func TestTransformCompletionsRequestWithContext_ShortTimeout(t *testing.T) {
	adapter := New()

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Nanosecond)
	defer cancel()

	// Wait for timeout to definitely trigger
	time.Sleep(1 * time.Millisecond)

	req := createMockRequest([]openai.ChatCompletionToolParam{
		createMockTool("test_func", "Test function"),
	})

	_, err := adapter.TransformCompletionsRequestWithContext(ctx, req)
	assert.Error(t, err)
	assert.True(t, errors.Is(err, context.DeadlineExceeded), "Expected context.DeadlineExceeded, got: %v", err)
}

func TestTransformCompletionsRequestWithContext_ReasonableTimeout(t *testing.T) {
	adapter := New()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req := createMockRequest([]openai.ChatCompletionToolParam{
		createMockTool("test_func", "Test function"),
	})

	result, err := adapter.TransformCompletionsRequestWithContext(ctx, req)
	require.NoError(t, err)
	assert.NotNil(t, result)
}

func TestTransformCompletionsResponseWithContext_ShortTimeout(t *testing.T) {
	adapter := New()

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Nanosecond)
	defer cancel()

	time.Sleep(1 * time.Millisecond)

	resp := createMockCompletion(`[{"name": "test_func", "parameters": null}]`)

	_, err := adapter.TransformCompletionsResponseWithContext(ctx, resp)
	assert.Error(t, err)
	assert.True(t, errors.Is(err, context.DeadlineExceeded), "Expected context.DeadlineExceeded, got: %v", err)
}

func TestTransformCompletionsResponseWithContext_ReasonableTimeout(t *testing.T) {
	adapter := New()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	resp := createMockCompletion(`[{"name": "test_func", "parameters": {"param1": "value1"}}]`)

	result, err := adapter.TransformCompletionsResponseWithContext(ctx, resp)
	require.NoError(t, err)
	assert.NotNil(t, result)
	assert.Len(t, result.Choices[0].Message.ToolCalls, 1)
}

// ============================================================================
// STREAMING CONTEXT TESTS
// ============================================================================

// MockControlledStream allows precise control over streaming behavior for testing
type MockControlledStream struct {
	chunks      []openai.ChatCompletionChunk
	index       int
	closed      bool
	nextDelay   time.Duration
	blockOnNext chan struct{}
	shouldError bool
	mu          sync.Mutex
}

func NewMockControlledStream(chunks []openai.ChatCompletionChunk) *MockControlledStream {
	return &MockControlledStream{
		chunks: chunks,
		index:  0,
	}
}

func (m *MockControlledStream) SetNextDelay(delay time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.nextDelay = delay
}

func (m *MockControlledStream) BlockOnNext() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.blockOnNext = make(chan struct{})
}

func (m *MockControlledStream) UnblockNext() {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.blockOnNext != nil {
		close(m.blockOnNext)
		m.blockOnNext = nil
	}
}

func (m *MockControlledStream) Next() bool {
	m.mu.Lock()
	blockCh := m.blockOnNext
	delay := m.nextDelay
	m.mu.Unlock()

	// Block if requested
	if blockCh != nil {
		<-blockCh
	}

	// Add delay if requested
	if delay > 0 {
		time.Sleep(delay)
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if m.closed || m.index >= len(m.chunks) {
		return false
	}
	m.index++
	return m.index <= len(m.chunks)
}

func (m *MockControlledStream) Current() openai.ChatCompletionChunk {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.index == 0 || m.index > len(m.chunks) {
		return openai.ChatCompletionChunk{}
	}
	return m.chunks[m.index-1]
}

func (m *MockControlledStream) Err() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.shouldError {
		return assert.AnError
	}
	return nil
}

func (m *MockControlledStream) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.closed = true
	return nil
}

func TestStreamingWithContext_ImmediateCancellation(t *testing.T) {
	adapter := New()

	mockStream := NewMockControlledStream([]openai.ChatCompletionChunk{
		{
			Choices: []openai.ChatCompletionChunkChoice{
				{
					Delta: openai.ChatCompletionChunkChoiceDelta{
						Content: `[{"name": "test_func", "parameters": null}]`,
						Role:    "assistant",
					},
				},
			},
		},
	})

	// Pre-cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	streamAdapter := adapter.TransformStreamingResponseWithContext(ctx, mockStream)
	defer func() {
		if err := streamAdapter.Close(); err != nil {
			t.Logf("Failed to close stream adapter in test: %v", err)
		}
	}()

	// Should return false immediately due to cancellation
	hasNext := streamAdapter.Next()
	assert.False(t, hasNext)

	err := streamAdapter.Err()
	assert.True(t, errors.Is(err, context.Canceled), "Expected context.Canceled, got: %v", err)
}

func TestStreamingWithContext_CancellationDuringStreaming(t *testing.T) {
	adapter := New()

	mockStream := NewMockControlledStream([]openai.ChatCompletionChunk{
		{
			Choices: []openai.ChatCompletionChunkChoice{
				{
					Delta: openai.ChatCompletionChunkChoiceDelta{
						Content: `[{"name": "test_func1"`,
						Role:    "assistant",
					},
				},
			},
		},
		{
			Choices: []openai.ChatCompletionChunkChoice{
				{
					Delta: openai.ChatCompletionChunkChoiceDelta{
						Content: `, "parameters": null}]`,
						Role:    "assistant",
					},
				},
			},
		},
	})

	ctx, cancel := context.WithCancel(context.Background())

	streamAdapter := adapter.TransformStreamingResponseWithContext(ctx, mockStream)
	defer func() {
		if err := streamAdapter.Close(); err != nil {
			t.Logf("Failed to close stream adapter in test: %v", err)
		}
	}()

	// Start reading
	hasNext := streamAdapter.Next()
	if hasNext {
		_ = streamAdapter.Current()
	}

	// Cancel the context
	cancel()

	// Next call should detect cancellation
	hasNext = streamAdapter.Next()
	assert.False(t, hasNext)

	err := streamAdapter.Err()
	assert.True(t, errors.Is(err, context.Canceled), "Expected context.Canceled, got: %v", err)
}

func TestStreamingWithContext_TimeoutDuringStreaming(t *testing.T) {
	adapter := New()

	mockStream := NewMockControlledStream([]openai.ChatCompletionChunk{
		{
			Choices: []openai.ChatCompletionChunkChoice{
				{
					Delta: openai.ChatCompletionChunkChoiceDelta{
						Content: `[{"name": "test_func", "parameters": null}]`,
						Role:    "assistant",
					},
				},
			},
		},
	})

	// Block the stream indefinitely to trigger timeout
	mockStream.BlockOnNext()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()

	streamAdapter := adapter.TransformStreamingResponseWithContext(ctx, mockStream)
	defer func() {
		if err := streamAdapter.Close(); err != nil {
			t.Logf("Failed to close stream adapter in test: %v", err)
		}
	}()

	// Start a goroutine to unblock after timeout should have occurred
	go func() {
		time.Sleep(50 * time.Millisecond)
		mockStream.UnblockNext()
	}()

	// This should timeout
	hasNext := streamAdapter.Next()
	assert.False(t, hasNext)

	err := streamAdapter.Err()
	assert.True(t, errors.Is(err, context.DeadlineExceeded), "Expected context.DeadlineExceeded, got: %v", err)
}

func TestStreamingWithContext_ProperCleanup(t *testing.T) {
	adapter := New()

	mockStream := NewMockControlledStream([]openai.ChatCompletionChunk{
		{
			Choices: []openai.ChatCompletionChunkChoice{
				{
					Delta: openai.ChatCompletionChunkChoiceDelta{
						Content: `[{"name": "test_func", "parameters": null}]`,
						Role:    "assistant",
					},
				},
			},
		},
	})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	streamAdapter := adapter.TransformStreamingResponseWithContext(ctx, mockStream)

	// Verify context is properly set up
	assert.NotNil(t, streamAdapter.ctx)
	assert.NotNil(t, streamAdapter.cancel)

	// Close should clean up context
	err := streamAdapter.Close()
	assert.NoError(t, err)

	// Verify underlying stream was closed
	assert.True(t, mockStream.closed)
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

func TestContextWithValue_Propagation(t *testing.T) {
	adapter := New()

	// Create context with value
	ctx := context.WithValue(context.Background(), ctxKeyRequestID, "test_123")

	req := createMockRequest([]openai.ChatCompletionToolParam{
		createMockTool("test_func", "Test function"),
	})

	result, err := adapter.TransformCompletionsRequestWithContext(ctx, req)
	require.NoError(t, err)
	assert.NotNil(t, result)

	// Context values should be accessible (though we don't use them in adapter)
	// This verifies the context chain isn't broken
	assert.Equal(t, "test_123", ctx.Value(ctxKeyRequestID))
}

func TestNilContext_Panics(t *testing.T) {
	adapter := New()

	req := createMockRequest([]openai.ChatCompletionToolParam{
		createMockTool("test_func", "Test function"),
	})

	// This should panic as per Go context conventions
	assert.Panics(t, func() {
		//nolint:errcheck,staticcheck // Testing panic behavior with nil context
		adapter.TransformCompletionsRequestWithContext(nil, req)
	})
}

func TestEmptyTools_WithContext(t *testing.T) {
	adapter := New()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Request with no tools
	req := createMockRequest(nil)

	result, err := adapter.TransformCompletionsRequestWithContext(ctx, req)
	require.NoError(t, err)
	assert.Equal(t, req, result) // Should pass through unchanged
}

func TestEmptyResponse_WithContext(t *testing.T) {
	adapter := New()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Empty response
	resp := openai.ChatCompletion{}

	result, err := adapter.TransformCompletionsResponseWithContext(ctx, resp)
	require.NoError(t, err)
	assert.Equal(t, resp, result) // Should pass through unchanged
}

// ============================================================================
// CONCURRENT CONTEXT OPERATIONS
// ============================================================================

func TestConcurrentContextOperations(t *testing.T) {
	adapter := New()

	const numGoroutines = 10
	const numOperations = 100

	var wg sync.WaitGroup
	var successCount int64
	var cancelledCount int64
	var timeoutCount int64

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(goroutineID int) {
			defer wg.Done()

			for j := 0; j < numOperations; j++ {
				// Randomly choose timeout or cancellation
				var ctx context.Context
				var cancel context.CancelFunc

				switch j % 3 {
				case 0:
					// Pre-cancelled context (deterministic cancellation)
					ctx, cancel = context.WithCancel(context.Background())
					cancel() // Cancel immediately - guaranteed to be cancelled
				case 1:
					// Another pre-cancelled context for variety
					ctx, cancel = context.WithCancel(context.Background())
					cancel() // Cancel immediately - guaranteed to be cancelled
				default:
					// Normal operation (never cancelled)
					ctx, cancel = context.WithTimeout(context.Background(), 1*time.Second)
				}

				req := createMockRequest([]openai.ChatCompletionToolParam{
					createMockTool("test_func", "Test function"),
				})

				_, err := adapter.TransformCompletionsRequestWithContext(ctx, req)

				if err == nil {
					atomic.AddInt64(&successCount, 1)
				} else if errors.Is(err, context.Canceled) {
					atomic.AddInt64(&cancelledCount, 1)
				} else if errors.Is(err, context.DeadlineExceeded) {
					atomic.AddInt64(&timeoutCount, 1)
				} else {
					// Count other errors as part of total for accounting
					atomic.AddInt64(&successCount, 1)
				}

				cancel()
			}
		}(i)
	}

	wg.Wait()

	total := successCount + cancelledCount + timeoutCount
	expectedTotal := int64(numGoroutines * numOperations)

	assert.Equal(t, expectedTotal, total, "All operations should be accounted for")
	assert.Greater(t, successCount, int64(0), "Should have some successful operations")

	t.Logf("Results: %d successful, %d cancelled, %d timed out",
		successCount, cancelledCount, timeoutCount)
}

func TestConcurrentStreamingContextOperations(t *testing.T) {
	adapter := New()

	const numStreams = 5
	const numChunks = 10

	var wg sync.WaitGroup

	for i := 0; i < numStreams; i++ {
		wg.Add(1)
		go func(streamID int) {
			defer wg.Done()

			// Create chunks for this stream
			chunks := make([]openai.ChatCompletionChunk, numChunks)
			for j := 0; j < numChunks; j++ {
				chunks[j] = openai.ChatCompletionChunk{
					Choices: []openai.ChatCompletionChunkChoice{
						{
							Delta: openai.ChatCompletionChunkChoiceDelta{
								Content: "chunk content",
								Role:    "assistant",
							},
						},
					},
				}
			}

			mockStream := NewMockControlledStream(chunks)
			ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
			defer cancel()

			streamAdapter := adapter.TransformStreamingResponseWithContext(ctx, mockStream)
			defer func() {
				if err := streamAdapter.Close(); err != nil {
					t.Logf("Failed to close stream adapter in test: %v", err)
				}
			}()

			// Read all chunks or until cancelled
			chunkCount := 0
			for streamAdapter.Next() {
				_ = streamAdapter.Current()
				chunkCount++
			}

			// Should either read all chunks or be cancelled/timed out
			err := streamAdapter.Err()
			if err != nil {
				assert.True(t, errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded),
					"Error should be context-related: %v", err)
			}

			t.Logf("Stream %d processed %d chunks", streamID, chunkCount)
		}(i)
	}

	wg.Wait()
}

// ============================================================================
// BACKWARD COMPATIBILITY TESTS
// ============================================================================

func TestBackwardCompatibility_OriginalMethodsUnchanged(t *testing.T) {
	adapter := New()

	// Test original TransformCompletionsRequest
	req := createMockRequest([]openai.ChatCompletionToolParam{
		createMockTool("test_func", "Test function"),
	})

	result1, err1 := adapter.TransformCompletionsRequest(req)
	require.NoError(t, err1)

	// Test context version with background context
	result2, err2 := adapter.TransformCompletionsRequestWithContext(context.Background(), req)
	require.NoError(t, err2)

	// Results should be identical
	assert.Equal(t, result1, result2)

	// Test original TransformCompletionsResponse
	resp := createMockCompletion(`[{"name": "test_func", "parameters": null}]`)

	result3, err3 := adapter.TransformCompletionsResponse(resp)
	require.NoError(t, err3)

	// Test context version with background context
	result4, err4 := adapter.TransformCompletionsResponseWithContext(context.Background(), resp)
	require.NoError(t, err4)

	// Both should have tool calls
	require.Len(t, result3.Choices, 1)
	require.Len(t, result4.Choices, 1)
	require.Len(t, result3.Choices[0].Message.ToolCalls, 1)
	require.Len(t, result4.Choices[0].Message.ToolCalls, 1)

	// Compare everything except the randomly generated IDs
	assert.Equal(t, result3.Choices[0].Message.Role, result4.Choices[0].Message.Role)
	assert.Equal(t, result3.Choices[0].Message.Content, result4.Choices[0].Message.Content)
	assert.Equal(t, result3.Choices[0].Message.ToolCalls[0].Function.Name, result4.Choices[0].Message.ToolCalls[0].Function.Name)
	assert.Equal(t, result3.Choices[0].Message.ToolCalls[0].Function.Arguments, result4.Choices[0].Message.ToolCalls[0].Function.Arguments)
	assert.Equal(t, result3.Choices[0].FinishReason, result4.Choices[0].FinishReason)
}

func TestBackwardCompatibility_StreamingUnchanged(t *testing.T) {
	adapter := New()

	mockStream := NewMockControlledStream([]openai.ChatCompletionChunk{
		{
			Choices: []openai.ChatCompletionChunkChoice{
				{
					Delta: openai.ChatCompletionChunkChoiceDelta{
						Content: `[{"name": "test_func", "parameters": null}]`,
						Role:    "assistant",
					},
				},
			},
		},
	})

	// Original method should still work
	streamAdapter := adapter.TransformStreamingResponse(mockStream)
	assert.NotNil(t, streamAdapter)

	// Should have context support internally (uses background context)
	assert.NotNil(t, streamAdapter.ctx)
	assert.NotNil(t, streamAdapter.cancel)

	assert.NoError(t, streamAdapter.Close())
}
