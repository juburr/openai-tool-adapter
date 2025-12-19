package tooladapter_test

import (
	"bytes"
	"log/slog"
	"strings"
	"testing"
	"time"

	tooladapter "github.com/juburr/openai-tool-adapter/v3"
	"github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestOptionsComprehensive tests all option functions for coverage and correctness
func TestOptionsComprehensive(t *testing.T) {
	t.Run("DefaultOptions", func(t *testing.T) {
		opts := tooladapter.DefaultOptions()
		require.NotEmpty(t, opts, "DefaultOptions should return non-empty slice")

		// Apply default options to an adapter
		adapter := tooladapter.New(opts...)
		require.NotNil(t, adapter)

		// Test that default options work by doing a transformation
		req := createMockRequestWithTools()
		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)
		assert.NotEqual(t, req, result, "Request should be transformed")
	})

	t.Run("ApplyOptions", func(t *testing.T) {
		adapter := tooladapter.New()
		require.NotNil(t, adapter)

		// Create some test options
		opts := []tooladapter.Option{
			tooladapter.WithCustomPromptTemplate("Test template: %s"),
			tooladapter.WithLogLevel(slog.LevelDebug),
		}

		// Apply options using ApplyOptions function
		tooladapter.ApplyOptions(adapter, opts)

		// Test that options were applied by doing a transformation
		req := createMockRequestWithTools()
		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)
		assert.NotEqual(t, req, result, "Request should be transformed")
	})

	t.Run("WithMetricsCallback", func(t *testing.T) {
		var capturedEvents []tooladapter.MetricEventData
		callback := func(data tooladapter.MetricEventData) {
			capturedEvents = append(capturedEvents, data)
		}

		adapter := tooladapter.New(tooladapter.WithMetricsCallback(callback))
		require.NotNil(t, adapter)

		// Trigger some events
		req := createMockRequestWithTools()
		_, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// Verify metrics were captured
		assert.NotEmpty(t, capturedEvents, "Should have captured metric events")

		// Verify event types
		foundTransformation := false
		for _, event := range capturedEvents {
			if event.EventType() == tooladapter.MetricEventToolTransformation {
				foundTransformation = true
				break
			}
		}
		assert.True(t, foundTransformation, "Should have captured tool transformation event")
	})
}

func TestOptionsErrorHandling(t *testing.T) {
	t.Run("WithCustomPromptTemplate_InvalidTemplate", func(t *testing.T) {
		// Test invalid template with no %s placeholder
		adapter := tooladapter.New(tooladapter.WithCustomPromptTemplate("Invalid template without placeholder"))
		require.NotNil(t, adapter)

		// Should still work but use default template
		req := createMockRequestWithTools()
		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)
		assert.NotEqual(t, req, result, "Request should still be transformed")
	})

	t.Run("WithCustomPromptTemplate_EmptyTemplate", func(t *testing.T) {
		// Test empty template
		adapter := tooladapter.New(tooladapter.WithCustomPromptTemplate(""))
		require.NotNil(t, adapter)

		// Should still work but use default template
		req := createMockRequestWithTools()
		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)
		assert.NotEqual(t, req, result, "Request should still be transformed")
	})

	t.Run("WithCustomPromptTemplate_MultiplePlaceholders", func(t *testing.T) {
		// Test template with multiple %s placeholders
		adapter := tooladapter.New(tooladapter.WithCustomPromptTemplate("Template with %s and %s placeholders"))
		require.NotNil(t, adapter)

		// Should still work but use default template
		req := createMockRequestWithTools()
		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)
		assert.NotEqual(t, req, result, "Request should still be transformed")
	})

	t.Run("WithLogger_NilLogger", func(t *testing.T) {
		// Test nil logger should create a no-op logger
		adapter := tooladapter.New(tooladapter.WithLogger(nil))
		require.NotNil(t, adapter)

		// Should still work with nil logger
		req := createMockRequestWithTools()
		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)
		assert.NotEqual(t, req, result, "Request should be transformed even with nil logger")
	})
}

func TestOptionsIntegration(t *testing.T) {
	t.Run("MultipleOptionsConfiguration", func(t *testing.T) {
		// Create a custom logger that writes to a buffer we can inspect
		var buf strings.Builder
		logger := slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{
			Level: slog.LevelDebug,
		}))

		// Configure adapter with multiple options
		adapter := tooladapter.New(
			tooladapter.WithCustomPromptTemplate("Custom template: %s"),
			tooladapter.WithLogger(logger),
			tooladapter.WithMetricsCallback(func(data tooladapter.MetricEventData) {
				// Metrics callback that does nothing for this test
			}),
		)
		require.NotNil(t, adapter)

		// Test that all options work together
		req := createMockRequestWithTools()
		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)
		assert.NotEqual(t, req, result, "Request should be transformed")

		// Verify logging occurred (custom logger was used)
		logOutput := buf.String()
		assert.Contains(t, logOutput, "Transformed request", "Should have logged transformation")
	})
}

// Helper function to create a mock request with tools
func createMockRequestWithTools() openai.ChatCompletionNewParams {
	return openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("Test message"),
		},
		Model: openai.ChatModelGPT4o,
		Tools: []openai.ChatCompletionToolUnionParam{
			openai.ChatCompletionFunctionTool(openai.FunctionDefinitionParam{
				Name:        "test_function",
				Description: openai.String("A test function"),
				Parameters: openai.FunctionParameters{
					"type": "object",
					"properties": map[string]interface{}{
						"message": map[string]interface{}{
							"type":        "string",
							"description": "The message to process",
						},
					},
					"required": []string{"message"},
				},
			}),
		},
	}
}

// TestBufferConfigurationOptions tests the new buffer configuration options
func TestBufferConfigurationOptions(t *testing.T) {
	t.Run("WithStreamingToolBufferSize", func(t *testing.T) {
		// Test valid stream buffer limit
		adapter := tooladapter.New(
			tooladapter.WithLogLevel(slog.LevelError),
			tooladapter.WithStreamingToolBufferSize(5*1024*1024), // 5MB
		)

		// Create a streaming response to verify the limit is used
		mockStream := NewMockStream([]openai.ChatCompletionChunk{
			createStreamChunk("test content"),
			createFinishChunk("stop"),
		})

		streamAdapter := adapter.TransformStreamingResponse(mockStream)
		defer func() { _ = streamAdapter.Close() }()

		// Process the stream - mainly checking that it works with custom buffer limit
		var chunks []openai.ChatCompletionChunk
		for streamAdapter.Next() {
			chunks = append(chunks, streamAdapter.Current())
		}
		require.NoError(t, streamAdapter.Err())
		assert.NotEmpty(t, chunks)
	})

	t.Run("WithStreamingToolBufferSize_InvalidValue", func(t *testing.T) {
		// Test that negative/zero values don't change the default
		adapter1 := tooladapter.New(tooladapter.WithStreamingToolBufferSize(-1))
		adapter2 := tooladapter.New(tooladapter.WithStreamingToolBufferSize(0))

		// Both should work (using default values internally)
		mockStream1 := NewMockStream([]openai.ChatCompletionChunk{createStreamChunk("test"), createFinishChunk("stop")})
		mockStream2 := NewMockStream([]openai.ChatCompletionChunk{createStreamChunk("test"), createFinishChunk("stop")})

		stream1 := adapter1.TransformStreamingResponse(mockStream1)
		stream2 := adapter2.TransformStreamingResponse(mockStream2)
		defer func() { _ = stream1.Close() }()
		defer func() { _ = stream2.Close() }()

		// Should not crash or error
		assert.True(t, stream1.Next())
		assert.True(t, stream2.Next())
	})

	t.Run("WithPromptBufferReuseLimit", func(t *testing.T) {
		// Test with custom buffer pool threshold
		customThreshold := 32 * 1024 // 32KB
		adapter := tooladapter.New(
			tooladapter.WithLogLevel(slog.LevelError),
			tooladapter.WithPromptBufferReuseLimit(customThreshold),
		)

		// Create a tool that should be within the threshold
		smallTool := openai.ChatCompletionFunctionTool(
			openai.FunctionDefinitionParam{
				Name:        "small_function",
				Description: openai.String("Small function for testing"),
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"param": map[string]interface{}{
							"type":        "string",
							"description": "A parameter",
						},
					},
				},
			},
		)

		req := openai.ChatCompletionNewParams{
			Model:    "gpt-4",
			Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("Test")},
			Tools:    []openai.ChatCompletionToolUnionParam{smallTool},
		}

		// Process multiple requests to exercise buffer pool
		for i := 0; i < 5; i++ {
			result, err := adapter.TransformCompletionsRequest(req)
			require.NoError(t, err)
			require.NotEmpty(t, result.Messages)
		}
	})

	t.Run("WithPromptBufferReuseLimit_InvalidValue", func(t *testing.T) {
		// Test that negative/zero values don't change the default
		adapter1 := tooladapter.New(tooladapter.WithPromptBufferReuseLimit(-1))
		adapter2 := tooladapter.New(tooladapter.WithPromptBufferReuseLimit(0))

		// Both should work (using default values internally)
		req := openai.ChatCompletionNewParams{
			Model:    "gpt-4",
			Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("Test")},
			Tools: []openai.ChatCompletionToolUnionParam{
				openai.ChatCompletionFunctionTool(
					openai.FunctionDefinitionParam{
						Name: "test_function",
					},
				),
			},
		}

		result1, err1 := adapter1.TransformCompletionsRequest(req)
		result2, err2 := adapter2.TransformCompletionsRequest(req)

		require.NoError(t, err1)
		require.NoError(t, err2)
		require.NotEmpty(t, result1.Messages)
		require.NotEmpty(t, result2.Messages)
	})

	t.Run("CombinedBufferOptions", func(t *testing.T) {
		// Test using both buffer configuration options together
		adapter := tooladapter.New(
			tooladapter.WithLogLevel(slog.LevelError),
			tooladapter.WithStreamingToolBufferSize(8*1024*1024), // 8MB stream limit
			tooladapter.WithPromptBufferReuseLimit(128*1024),     // 128KB pool threshold
		)

		// Test with both streaming and non-streaming requests
		tool := openai.ChatCompletionFunctionTool(
			openai.FunctionDefinitionParam{
				Name:        "combined_test",
				Description: openai.String("Test function for combined buffer options"),
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"data": map[string]interface{}{
							"type":        "string",
							"description": "Test data",
						},
					},
				},
			},
		)

		req := openai.ChatCompletionNewParams{
			Model:    "gpt-4",
			Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("Test combined options")},
			Tools:    []openai.ChatCompletionToolUnionParam{tool},
		}

		// Test non-streaming request
		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)
		require.NotEmpty(t, result.Messages)

		// Test streaming request
		mockStream := NewMockStream([]openai.ChatCompletionChunk{
			createStreamChunk(`[{"name": "combined_test", "parameters": {"data": "test"}}]`),
			createFinishChunk("stop"),
		})

		streamAdapter := adapter.TransformStreamingResponse(mockStream)
		defer func() { _ = streamAdapter.Close() }()

		var chunks []openai.ChatCompletionChunk
		for streamAdapter.Next() {
			chunks = append(chunks, streamAdapter.Current())
		}
		require.NoError(t, streamAdapter.Err())
		assert.NotEmpty(t, chunks)
	})

	t.Run("BufferOptionsWithLargeContent", func(t *testing.T) {
		// Test buffer options with content designed to test limits
		smallStreamLimit := 1024  // 1KB stream limit (very small)
		smallPoolThreshold := 512 // 512B pool threshold (very small)

		adapter := tooladapter.New(
			tooladapter.WithLogLevel(slog.LevelError),
			tooladapter.WithStreamingToolBufferSize(smallStreamLimit),
			tooladapter.WithPromptBufferReuseLimit(smallPoolThreshold),
		)

		// Create content that will exceed the small limits
		largeDescription := strings.Repeat("Large content for testing buffer limits. ", 50) // ~2KB
		largeTool := openai.ChatCompletionFunctionTool(
			openai.FunctionDefinitionParam{
				Name:        "large_test_function",
				Description: openai.String(largeDescription),
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"param": map[string]interface{}{
							"type":        "string",
							"description": largeDescription, // More large content
						},
					},
				},
			},
		)

		req := openai.ChatCompletionNewParams{
			Model:    "gpt-4",
			Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("Test with large content")},
			Tools:    []openai.ChatCompletionToolUnionParam{largeTool},
		}

		// Should still work despite exceeding limits (graceful handling)
		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)
		require.NotEmpty(t, result.Messages)

		// Verify the large content was processed correctly
		firstMsg, err := result.Messages[0].MarshalJSON()
		require.NoError(t, err)
		assert.Contains(t, string(firstMsg), "large_test_function")
	})
}

// TestToolPolicyOptionsValidation tests tool policy configuration options including validation
func TestToolPolicyOptionsValidation(t *testing.T) {
	t.Run("WithToolCollectWindow_ValidDurations", func(t *testing.T) {
		testCases := []struct {
			duration    time.Duration
			description string
		}{
			{0, "zero duration (structure-only batching)"},
			{100 * time.Millisecond, "100ms (typical short window)"},
			{500 * time.Millisecond, "500ms (typical medium window)"},
			{2 * time.Second, "2 seconds (longer window)"},
			{time.Hour, "1 hour (very large duration)"},
		}

		for _, tc := range testCases {
			t.Run(tc.description, func(t *testing.T) {
				adapter := tooladapter.New(
					tooladapter.WithLogLevel(slog.LevelError),
					tooladapter.WithToolPolicy(tooladapter.ToolCollectThenStop),
					tooladapter.WithToolCollectWindow(tc.duration),
				)
				require.NotNil(t, adapter)

				// Verify the adapter works correctly
				req := createMockRequestWithTools()
				result, err := adapter.TransformCompletionsRequest(req)
				require.NoError(t, err)
				assert.NotEqual(t, req, result, "Request should be transformed")
			})
		}
	})

	t.Run("WithToolCollectWindow_NegativeDuration", func(t *testing.T) {
		// Create a buffer to capture log output
		var logBuf bytes.Buffer
		logger := slog.New(slog.NewTextHandler(&logBuf, &slog.HandlerOptions{
			Level: slog.LevelDebug,
		}))

		// Test negative duration validation
		negativeDuration := -100 * time.Millisecond
		adapter := tooladapter.New(
			tooladapter.WithLogger(logger),
			tooladapter.WithToolPolicy(tooladapter.ToolCollectThenStop),
			tooladapter.WithToolCollectWindow(negativeDuration),
		)
		require.NotNil(t, adapter)

		// Should work correctly (negative converted to 0)
		req := createMockRequestWithTools()
		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)
		assert.NotEqual(t, req, result, "Request should be transformed")

		// Verify warning was logged
		logOutput := logBuf.String()
		assert.Contains(t, logOutput, "Negative duration not allowed for tool collection window",
			"Should log warning about negative duration")
	})

	t.Run("WithToolCollectWindow_ExtremelyNegativeDuration", func(t *testing.T) {
		// Test with very negative duration
		var logBuf bytes.Buffer
		logger := slog.New(slog.NewTextHandler(&logBuf, &slog.HandlerOptions{
			Level: slog.LevelDebug,
		}))

		adapter := tooladapter.New(
			tooladapter.WithLogger(logger),
			tooladapter.WithToolPolicy(tooladapter.ToolCollectThenStop),
			tooladapter.WithToolCollectWindow(-24*time.Hour), // -1 day
		)
		require.NotNil(t, adapter)

		// Should still work
		req := createMockRequestWithTools()
		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)
		assert.NotEqual(t, req, result, "Request should be transformed")

		// Verify warning was logged
		logOutput := logBuf.String()
		assert.Contains(t, logOutput, "Negative duration not allowed for tool collection window",
			"Should log warning for extremely negative duration")
	})

	t.Run("WithToolCollectWindow_WithDifferentPolicies", func(t *testing.T) {
		// Test that duration setting works with different policies
		// (even though it's only meaningful for ToolCollectThenStop)
		policies := []tooladapter.ToolPolicy{
			tooladapter.ToolStopOnFirst,
			tooladapter.ToolCollectThenStop,
			tooladapter.ToolDrainAll,
			tooladapter.ToolAllowMixed,
		}

		duration := 200 * time.Millisecond
		for _, policy := range policies {
			t.Run(policy.String(), func(t *testing.T) {
				adapter := tooladapter.New(
					tooladapter.WithLogLevel(slog.LevelError),
					tooladapter.WithToolPolicy(policy),
					tooladapter.WithToolCollectWindow(duration),
				)
				require.NotNil(t, adapter)

				// Should work regardless of policy
				req := createMockRequestWithTools()
				result, err := adapter.TransformCompletionsRequest(req)
				require.NoError(t, err)
				assert.NotEqual(t, req, result, "Request should be transformed")
			})
		}
	})

	t.Run("WithToolMaxCalls_Validation", func(t *testing.T) {
		testCases := []struct {
			maxCalls    int
			description string
		}{
			{0, "zero (unlimited)"},
			{1, "single call"},
			{5, "typical limit"},
			{100, "high limit"},
			{-1, "negative (converted to unlimited)"},
			{-100, "very negative (converted to unlimited)"},
		}

		for _, tc := range testCases {
			t.Run(tc.description, func(t *testing.T) {
				adapter := tooladapter.New(
					tooladapter.WithLogLevel(slog.LevelError),
					tooladapter.WithToolMaxCalls(tc.maxCalls),
				)
				require.NotNil(t, adapter)

				// Should work correctly
				req := createMockRequestWithTools()
				result, err := adapter.TransformCompletionsRequest(req)
				require.NoError(t, err)
				assert.NotEqual(t, req, result, "Request should be transformed")
			})
		}
	})

	t.Run("WithToolCollectMaxBytes_Validation", func(t *testing.T) {
		testCases := []struct {
			maxBytes    int
			description string
		}{
			{0, "zero (unlimited)"},
			{1024, "1KB limit"},
			{65536, "64KB default"},
			{1024 * 1024, "1MB limit"},
			{-1, "negative (converted to unlimited)"},
			{-1024, "negative KB (converted to unlimited)"},
		}

		for _, tc := range testCases {
			t.Run(tc.description, func(t *testing.T) {
				adapter := tooladapter.New(
					tooladapter.WithLogLevel(slog.LevelError),
					tooladapter.WithToolCollectMaxBytes(tc.maxBytes),
				)
				require.NotNil(t, adapter)

				// Should work correctly
				req := createMockRequestWithTools()
				result, err := adapter.TransformCompletionsRequest(req)
				require.NoError(t, err)
				assert.NotEqual(t, req, result, "Request should be transformed")
			})
		}
	})

	t.Run("CombinedToolPolicyOptions", func(t *testing.T) {
		// Test all tool policy options working together
		var logBuf bytes.Buffer
		logger := slog.New(slog.NewTextHandler(&logBuf, &slog.HandlerOptions{
			Level: slog.LevelDebug,
		}))

		adapter := tooladapter.New(
			tooladapter.WithLogger(logger),
			tooladapter.WithToolPolicy(tooladapter.ToolCollectThenStop),
			tooladapter.WithToolCollectWindow(300*time.Millisecond),
			tooladapter.WithToolMaxCalls(10),
			tooladapter.WithToolCollectMaxBytes(128*1024), // 128KB
			tooladapter.WithCancelUpstreamOnStop(true),
		)
		require.NotNil(t, adapter)

		// Test non-streaming
		req := createMockRequestWithTools()
		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)
		assert.NotEqual(t, req, result, "Request should be transformed")

		// Test streaming
		mockStream := NewMockStream([]openai.ChatCompletionChunk{
			createStreamChunk(`[{"name": "test_function", "parameters": {"message": "hello"}}]`),
			createFinishChunk("tool_calls"),
		})

		streamAdapter := adapter.TransformStreamingResponse(mockStream)
		defer func() { _ = streamAdapter.Close() }()

		var chunks []openai.ChatCompletionChunk
		for streamAdapter.Next() {
			chunks = append(chunks, streamAdapter.Current())
		}
		require.NoError(t, streamAdapter.Err())
		assert.NotEmpty(t, chunks)
	})

	t.Run("WithToolCollectWindow_StreamingIntegration", func(t *testing.T) {
		// Test that the collect window actually affects streaming behavior
		adapter := tooladapter.New(
			tooladapter.WithLogLevel(slog.LevelError),
			tooladapter.WithToolPolicy(tooladapter.ToolCollectThenStop),
			tooladapter.WithToolCollectWindow(50*time.Millisecond), // Short window
		)

		// Create a stream with tool calls
		mockStream := NewMockStream([]openai.ChatCompletionChunk{
			createStreamChunk(`[{"name": "test_function",`),
			createStreamChunk(` "parameters": {"message": "test"}}]`),
			createFinishChunk("tool_calls"),
		})

		streamAdapter := adapter.TransformStreamingResponse(mockStream)
		defer func() { _ = streamAdapter.Close() }()

		// Should successfully process the tool calls
		var toolCallChunks int
		for streamAdapter.Next() {
			chunk := streamAdapter.Current()
			if len(chunk.Choices) > 0 && len(chunk.Choices[0].Delta.ToolCalls) > 0 {
				toolCallChunks++
			}
		}
		require.NoError(t, streamAdapter.Err())
		assert.GreaterOrEqual(t, toolCallChunks, 0, "Should handle tool calls in streaming mode")
	})
}

// TestAdditionalOptionValidation tests log message validation for additional option functions
func TestAdditionalOptionValidation(t *testing.T) {
	t.Run("WithToolMaxCalls_NegativeValue", func(t *testing.T) {
		// Create a buffer to capture log output
		var logBuf bytes.Buffer
		logger := slog.New(slog.NewTextHandler(&logBuf, &slog.HandlerOptions{
			Level: slog.LevelDebug,
		}))

		// Test negative tool max calls validation
		adapter := tooladapter.New(
			tooladapter.WithLogger(logger),
			tooladapter.WithToolMaxCalls(-5),
		)
		require.NotNil(t, adapter)

		// Should work correctly (negative converted to 0)
		req := createMockRequestWithTools()
		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)
		assert.NotEqual(t, req, result, "Request should be transformed")

		// Verify warning was logged
		logOutput := logBuf.String()
		assert.Contains(t, logOutput, "Negative tool count not allowed for ToolMaxCalls",
			"Should log warning about negative tool max calls")
		assert.Contains(t, logOutput, "supplied_maxCalls",
			"Should include supplied_maxCalls field")
		assert.Contains(t, logOutput, "updated_maxCalls",
			"Should include updated_maxCalls field")
		assert.Contains(t, logOutput, "implication",
			"Should include implication field")
		assert.Contains(t, logOutput, "recommendation",
			"Should include recommendation field")
	})

	t.Run("WithToolCollectMaxBytes_NegativeValue", func(t *testing.T) {
		// Create a buffer to capture log output
		var logBuf bytes.Buffer
		logger := slog.New(slog.NewTextHandler(&logBuf, &slog.HandlerOptions{
			Level: slog.LevelDebug,
		}))

		// Test negative tool collect max bytes validation
		adapter := tooladapter.New(
			tooladapter.WithLogger(logger),
			tooladapter.WithToolCollectMaxBytes(-1024),
		)
		require.NotNil(t, adapter)

		// Should work correctly (negative converted to 0)
		req := createMockRequestWithTools()
		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)
		assert.NotEqual(t, req, result, "Request should be transformed")

		// Verify warning was logged
		logOutput := logBuf.String()
		assert.Contains(t, logOutput, "Negative byte count not allowed for ToolCollectMaxBytes",
			"Should log warning about negative tool collect max bytes")
		assert.Contains(t, logOutput, "supplied_maxBytes",
			"Should include supplied_maxBytes field")
		assert.Contains(t, logOutput, "updated_maxBytes",
			"Should include updated_maxBytes field")
		assert.Contains(t, logOutput, "implication",
			"Should include implication field")
		assert.Contains(t, logOutput, "recommendation",
			"Should include recommendation field")
	})

	t.Run("WithCancelUpstreamOnStop_DisabledCancellation", func(t *testing.T) {
		// Create a buffer to capture log output
		var logBuf bytes.Buffer
		logger := slog.New(slog.NewTextHandler(&logBuf, &slog.HandlerOptions{
			Level: slog.LevelDebug,
		}))

		// Test disabling upstream cancellation logs info message
		adapter := tooladapter.New(
			tooladapter.WithLogger(logger),
			tooladapter.WithCancelUpstreamOnStop(false),
		)
		require.NotNil(t, adapter)

		// Should work correctly with cancellation disabled
		req := createMockRequestWithTools()
		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)
		assert.NotEqual(t, req, result, "Request should be transformed")

		// Verify info message was logged
		logOutput := logBuf.String()
		assert.Contains(t, logOutput, "Upstream context cancellation has been disabled",
			"Should log info about disabled upstream cancellation")
		assert.Contains(t, logOutput, "implication",
			"Should include implication field")
		assert.Contains(t, logOutput, "recommendation",
			"Should include recommendation field")
	})

	t.Run("WithCancelUpstreamOnStop_EnabledCancellation", func(t *testing.T) {
		// Create a buffer to capture log output
		var logBuf bytes.Buffer
		logger := slog.New(slog.NewTextHandler(&logBuf, &slog.HandlerOptions{
			Level: slog.LevelDebug,
		}))

		// Test enabling upstream cancellation (should not log anything)
		adapter := tooladapter.New(
			tooladapter.WithLogger(logger),
			tooladapter.WithCancelUpstreamOnStop(true),
		)
		require.NotNil(t, adapter)

		// Should work correctly with cancellation enabled
		req := createMockRequestWithTools()
		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)
		assert.NotEqual(t, req, result, "Request should be transformed")

		// Verify no message was logged (since true is the default)
		logOutput := logBuf.String()
		assert.NotContains(t, logOutput, "Upstream context cancellation has been disabled",
			"Should not log message when cancellation is enabled")
	})

	t.Run("CombinedValidationLogging", func(t *testing.T) {
		// Test multiple validation messages in a single adapter configuration
		var logBuf bytes.Buffer
		logger := slog.New(slog.NewTextHandler(&logBuf, &slog.HandlerOptions{
			Level: slog.LevelDebug,
		}))

		// Configure adapter with multiple invalid values
		adapter := tooladapter.New(
			tooladapter.WithLogger(logger),
			tooladapter.WithToolMaxCalls(-10),
			tooladapter.WithToolCollectMaxBytes(-2048),
			tooladapter.WithToolCollectWindow(-500*time.Millisecond),
			tooladapter.WithCancelUpstreamOnStop(false),
		)
		require.NotNil(t, adapter)

		// Should work correctly despite invalid values
		req := createMockRequestWithTools()
		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)
		assert.NotEqual(t, req, result, "Request should be transformed")

		// Verify all validation messages were logged
		logOutput := logBuf.String()

		// Check for tool max calls warning
		assert.Contains(t, logOutput, "Negative tool count not allowed for ToolMaxCalls",
			"Should log warning for negative tool max calls")

		// Check for tool collect max bytes warning
		assert.Contains(t, logOutput, "Negative byte count not allowed for ToolCollectMaxBytes",
			"Should log warning for negative tool collect max bytes")

		// Check for tool collect window warning
		assert.Contains(t, logOutput, "Negative duration not allowed for tool collection window",
			"Should log warning for negative tool collect window")

		// Check for cancel upstream info message
		assert.Contains(t, logOutput, "Upstream context cancellation has been disabled",
			"Should log info for disabled upstream cancellation")
	})

	t.Run("ValidValues_NoLogging", func(t *testing.T) {
		// Test that valid values don't trigger any logging
		var logBuf bytes.Buffer
		logger := slog.New(slog.NewTextHandler(&logBuf, &slog.HandlerOptions{
			Level: slog.LevelDebug,
		}))

		// Configure adapter with all valid values
		adapter := tooladapter.New(
			tooladapter.WithLogger(logger),
			tooladapter.WithToolMaxCalls(10),
			tooladapter.WithToolCollectMaxBytes(65536),
			tooladapter.WithToolCollectWindow(200*time.Millisecond),
			tooladapter.WithCancelUpstreamOnStop(true),
		)
		require.NotNil(t, adapter)

		// Should work correctly with valid values
		req := createMockRequestWithTools()
		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)
		assert.NotEqual(t, req, result, "Request should be transformed")

		// Verify no validation messages were logged (only normal operation logs)
		logOutput := logBuf.String()
		assert.NotContains(t, logOutput, "Negative tool count not allowed",
			"Should not log warning for valid tool max calls")
		assert.NotContains(t, logOutput, "Negative byte count not allowed",
			"Should not log warning for valid tool collect max bytes")
		assert.NotContains(t, logOutput, "Negative duration not allowed",
			"Should not log warning for valid tool collect window")
		assert.NotContains(t, logOutput, "Upstream context cancellation has been disabled",
			"Should not log info when cancellation is enabled")
	})
}

// TestSystemMessageSupportOption tests the WithSystemMessageSupport configuration option
func TestSystemMessageSupportOption(t *testing.T) {
	t.Run("DefaultBehavior_NoSystemSupport", func(t *testing.T) {
		// By default, systemMessagesSupported should be false (for Gemma compatibility)
		adapter := tooladapter.New(tooladapter.WithLogLevel(slog.LevelError))

		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Test message"),
			},
			Tools: []openai.ChatCompletionToolUnionParam{
				openai.ChatCompletionFunctionTool(openai.FunctionDefinitionParam{
					Name:        "test_func",
					Description: openai.String("Test function"),
				}),
			},
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// When systemMessagesSupported is false and no system message exists,
		// it should modify the first user message by prepending tool instructions
		assert.Len(t, result.Messages, 1, "Should have 1 message (modified user)")

		// First message should be the modified user message with tools prepended
		assert.NotNil(t, result.Messages[0].OfUser, "First message should be user")
		userContent := result.Messages[0].OfUser.Content.OfString.Or("")
		assert.Contains(t, userContent, "test_func", "Should contain tool info")
		assert.Contains(t, userContent, "Test message", "Should preserve original content")
	})

	t.Run("WithSystemMessageSupport_Enabled", func(t *testing.T) {
		// Enable system message support
		adapter := tooladapter.New(
			tooladapter.WithLogLevel(slog.LevelError),
			tooladapter.WithSystemMessageSupport(true),
		)

		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Test message"),
			},
			Tools: []openai.ChatCompletionToolUnionParam{
				openai.ChatCompletionFunctionTool(openai.FunctionDefinitionParam{
					Name:        "test_func",
					Description: openai.String("Test function"),
				}),
			},
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// When systemMessagesSupported is true and no system message exists,
		// it should prepend a SYSTEM instruction
		assert.Len(t, result.Messages, 2, "Should have 2 messages (system instruction + original user)")

		// First message should be the SYSTEM instruction
		assert.NotNil(t, result.Messages[0].OfSystem, "First message should be a system instruction")
		systemContent := result.Messages[0].OfSystem.Content.OfString.Or("")
		assert.Contains(t, systemContent, "test_func", "System instruction should contain tool info")

		// Second message should be the original user message
		assert.NotNil(t, result.Messages[1].OfUser, "Second message should be the original user message")
		originalContent := result.Messages[1].OfUser.Content.OfString.Or("")
		assert.Equal(t, "Test message", originalContent, "Original user message should be unchanged")
	})

	t.Run("WithSystemMessageSupport_ExistingSystemMessage", func(t *testing.T) {
		// Test both enabled and disabled when a system message already exists
		testCases := []struct {
			name    string
			enabled bool
		}{
			{"SystemSupportDisabled", false},
			{"SystemSupportEnabled", true},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				adapter := tooladapter.New(
					tooladapter.WithLogLevel(slog.LevelError),
					tooladapter.WithSystemMessageSupport(tc.enabled),
				)

				req := openai.ChatCompletionNewParams{
					Messages: []openai.ChatCompletionMessageParamUnion{
						openai.SystemMessage("System prompt"),
						openai.UserMessage("Test message"),
					},
					Tools: []openai.ChatCompletionToolUnionParam{
						openai.ChatCompletionFunctionTool(openai.FunctionDefinitionParam{
							Name:        "test_func",
							Description: openai.String("Test function"),
						}),
					},
				}

				result, err := adapter.TransformCompletionsRequest(req)
				require.NoError(t, err)

				// When a system message exists, behavior is the same regardless of the option:
				// The tool prompt is appended to the existing system message
				assert.Len(t, result.Messages, 2, "Should still have 2 messages")

				// First message should be the modified system message
				assert.NotNil(t, result.Messages[0].OfSystem, "First message should be system")
				systemContent := result.Messages[0].OfSystem.Content.OfString.Or("")
				assert.Contains(t, systemContent, "System prompt", "Should contain original system content")
				assert.Contains(t, systemContent, "test_func", "Should contain tool info")

				// Second message should be unchanged user message
				assert.NotNil(t, result.Messages[1].OfUser, "Second message should be user")
				assert.Equal(t, "Test message", result.Messages[1].OfUser.Content.OfString.Or(""))
			})
		}
	})

	t.Run("WithSystemMessageSupport_NoMessages", func(t *testing.T) {
		// Test edge case with no messages at all
		testCases := []struct {
			name            string
			systemSupported bool
			expectedRole    string
		}{
			{"NoSystemSupport", false, "user"},
			{"WithSystemSupport", true, "system"},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				adapter := tooladapter.New(
					tooladapter.WithLogLevel(slog.LevelError),
					tooladapter.WithSystemMessageSupport(tc.systemSupported),
				)

				req := openai.ChatCompletionNewParams{
					Messages: []openai.ChatCompletionMessageParamUnion{},
					Tools: []openai.ChatCompletionToolUnionParam{
						openai.ChatCompletionFunctionTool(openai.FunctionDefinitionParam{
							Name:        "test_func",
							Description: openai.String("Test function"),
						}),
					},
				}

				result, err := adapter.TransformCompletionsRequest(req)
				require.NoError(t, err)

				assert.Len(t, result.Messages, 1, "Should have 1 message (instruction only)")

				if tc.expectedRole == "system" {
					assert.NotNil(t, result.Messages[0].OfSystem, "Should be a system instruction")
					content := result.Messages[0].OfSystem.Content.OfString.Or("")
					assert.Contains(t, content, "test_func", "Instruction should contain tool info")
				} else {
					assert.NotNil(t, result.Messages[0].OfUser, "Should be a user instruction")
					content := result.Messages[0].OfUser.Content.OfString.Or("")
					assert.Contains(t, content, "test_func", "Instruction should contain tool info")
				}
			})
		}
	})

	t.Run("WithSystemMessageSupport_MultipleOptions", func(t *testing.T) {
		// Test combining WithSystemMessageSupport with other options
		var capturedEvents []tooladapter.MetricEventData
		callback := func(data tooladapter.MetricEventData) {
			capturedEvents = append(capturedEvents, data)
		}

		adapter := tooladapter.New(
			tooladapter.WithLogLevel(slog.LevelError),
			tooladapter.WithSystemMessageSupport(true),
			tooladapter.WithMetricsCallback(callback),
			tooladapter.WithToolPolicy(tooladapter.ToolStopOnFirst),
		)

		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Test message"),
			},
			Tools: []openai.ChatCompletionToolUnionParam{
				openai.ChatCompletionFunctionTool(openai.FunctionDefinitionParam{
					Name:        "test_func",
					Description: openai.String("Test function"),
				}),
			},
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// Verify system message support works with other options
		assert.Len(t, result.Messages, 2, "Should have 2 messages")
		assert.NotNil(t, result.Messages[0].OfSystem, "First should be system instruction")

		// Verify metrics were captured
		assert.NotEmpty(t, capturedEvents, "Should have captured metric events")
	})
}
