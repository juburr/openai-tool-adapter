package tooladapter_test

import (
	"bytes"
	"log/slog"
	"strings"
	"testing"

	tooladapter "github.com/juburr/openai-tool-adapter/v3"
	"github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestMetricsCallbackPanicRecovery verifies that panics in metrics callbacks are recovered
func TestMetricsCallbackPanicRecovery(t *testing.T) {
	t.Run("PanicInCallback_TransformCompletionsRequest", func(t *testing.T) {
		// Capture logs to verify panic was logged
		var logBuffer bytes.Buffer
		logger := slog.New(slog.NewTextHandler(&logBuffer, &slog.HandlerOptions{
			Level: slog.LevelDebug,
		}))

		panickingCallback := func(data tooladapter.MetricEventData) {
			panic("intentional test panic")
		}

		adapter := tooladapter.New(
			tooladapter.WithLogger(logger),
			tooladapter.WithMetricsCallback(panickingCallback),
		)

		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Hello"),
			},
			Tools: []openai.ChatCompletionToolUnionParam{
				openai.ChatCompletionFunctionTool(openai.FunctionDefinitionParam{
					Name:        "test_function",
					Description: openai.String("Test function"),
				}),
			},
		}

		// Should not panic - the panic should be recovered
		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err, "Transform should succeed despite callback panic")
		assert.NotNil(t, result)

		// Verify the request was transformed correctly
		// With no system, we modify the first user message
		assert.Len(t, result.Messages, 1, "Should modify existing message, not add new one")
		assert.Empty(t, result.Tools, "Tools should be removed")

		// Verify the panic was logged
		logOutput := logBuffer.String()
		assert.Contains(t, logOutput, "Metrics callback panicked")
		assert.Contains(t, logOutput, "intentional test panic")
		assert.Contains(t, logOutput, "tool_transformation")
	})

	t.Run("PanicInCallback_TransformCompletionsResponse", func(t *testing.T) {
		// Capture logs to verify panic was logged
		var logBuffer bytes.Buffer
		logger := slog.New(slog.NewTextHandler(&logBuffer, &slog.HandlerOptions{
			Level: slog.LevelDebug,
		}))

		panickingCallback := func(data tooladapter.MetricEventData) {
			// Only panic for function call detection events
			if data.EventType() == "function_call_detection" {
				panic("response processing panic")
			}
		}

		adapter := tooladapter.New(
			tooladapter.WithLogger(logger),
			tooladapter.WithMetricsCallback(panickingCallback),
		)

		resp := openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Content: `[{"name": "get_weather", "parameters": {"city": "London"}}]`,
					},
				},
			},
		}

		// Should not panic - the panic should be recovered
		result, err := adapter.TransformCompletionsResponse(resp)
		require.NoError(t, err, "Transform should succeed despite callback panic")
		assert.NotNil(t, result)

		// Verify the response was transformed correctly
		assert.Len(t, result.Choices, 1)
		assert.Len(t, result.Choices[0].Message.ToolCalls, 1)
		assert.Equal(t, "get_weather", result.Choices[0].Message.ToolCalls[0].Function.Name)

		// Verify the panic was logged
		logOutput := logBuffer.String()
		assert.Contains(t, logOutput, "Metrics callback panicked")
		assert.Contains(t, logOutput, "response processing panic")
		assert.Contains(t, logOutput, "function_call_detection")
	})

	t.Run("MultipleCallbacks_OnePanics", func(t *testing.T) {
		// Test that one callback panicking doesn't affect subsequent operations
		var logBuffer bytes.Buffer
		logger := slog.New(slog.NewTextHandler(&logBuffer, &slog.HandlerOptions{
			Level: slog.LevelDebug,
		}))

		callCount := 0
		sometimesPanickingCallback := func(data tooladapter.MetricEventData) {
			callCount++
			if callCount == 1 {
				panic("first call panic")
			}
			// Subsequent calls should work fine
		}

		adapter := tooladapter.New(
			tooladapter.WithLogger(logger),
			tooladapter.WithMetricsCallback(sometimesPanickingCallback),
		)

		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Hello"),
			},
			Tools: []openai.ChatCompletionToolUnionParam{
				openai.ChatCompletionFunctionTool(openai.FunctionDefinitionParam{
					Name:        "test_function",
					Description: openai.String("Test function"),
				}),
			},
		}

		// First call - will panic and recover
		result1, err1 := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err1)
		assert.NotNil(t, result1)

		// Second call - should work normally
		result2, err2 := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err2)
		assert.NotNil(t, result2)

		// Verify both calls completed
		assert.Equal(t, 2, callCount, "Both callbacks should have been attempted")

		// Verify only the first panic was logged
		logOutput := logBuffer.String()
		panicCount := strings.Count(logOutput, "Metrics callback panicked")
		assert.Equal(t, 1, panicCount, "Should have logged exactly one panic")
	})

	t.Run("NilDataPanic", func(t *testing.T) {
		// Test handling of panic when callback tries to access nil data
		var logBuffer bytes.Buffer
		logger := slog.New(slog.NewTextHandler(&logBuffer, &slog.HandlerOptions{
			Level: slog.LevelDebug,
		}))

		nilAccessCallback := func(data tooladapter.MetricEventData) {
			// This would panic if data methods are called incorrectly
			var nilData *tooladapter.ToolTransformationData
			_ = nilData.ToolCount // This will panic
		}

		adapter := tooladapter.New(
			tooladapter.WithLogger(logger),
			tooladapter.WithMetricsCallback(nilAccessCallback),
		)

		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Test"),
			},
			Tools: []openai.ChatCompletionToolUnionParam{
				openai.ChatCompletionFunctionTool(openai.FunctionDefinitionParam{
					Name: "test",
				}),
			},
		}

		// Should handle the nil pointer panic
		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)
		assert.NotNil(t, result)

		// Verify panic was caught and logged
		logOutput := logBuffer.String()
		assert.Contains(t, logOutput, "Metrics callback panicked")
	})
}

// TestMetricsCallbackNoRegression ensures existing functionality still works
func TestMetricsCallbackNoRegression(t *testing.T) {
	t.Run("NormalCallback_StillWorks", func(t *testing.T) {
		var capturedData []tooladapter.MetricEventData

		callback := func(data tooladapter.MetricEventData) {
			capturedData = append(capturedData, data)
		}

		opts := append([]tooladapter.Option{tooladapter.WithLogLevel(slog.LevelError)},
			tooladapter.WithMetricsCallback(callback))
		adapter := tooladapter.New(opts...)

		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Hello"),
			},
			Tools: []openai.ChatCompletionToolUnionParam{
				openai.ChatCompletionFunctionTool(openai.FunctionDefinitionParam{
					Name:        "test_function",
					Description: openai.String("Test function"),
				}),
			},
		}

		// Normal operation should work exactly as before
		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)
		assert.NotNil(t, result)

		// Verify metrics were captured
		assert.Len(t, capturedData, 1, "Should have captured one metric event")
		assert.Equal(t, tooladapter.MetricEvent("tool_transformation"), capturedData[0].EventType())
	})

	t.Run("NilCallback_NoIssues", func(t *testing.T) {
		// Test with no callback set
		adapter := tooladapter.New(tooladapter.WithLogLevel(slog.LevelError))

		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Hello"),
			},
			Tools: []openai.ChatCompletionToolUnionParam{
				openai.ChatCompletionFunctionTool(openai.FunctionDefinitionParam{
					Name: "test",
				}),
			},
		}

		// Should work fine without callback
		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)
		assert.NotNil(t, result)
	})
}
