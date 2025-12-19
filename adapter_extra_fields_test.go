package tooladapter

import (
	"context"
	"log/slog"
	"testing"

	"github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ============================================================================
// Extra Fields Preservation Tests for Non-SSE (OpenAI SDK) Mode
//
// These tests verify that non-standard OpenAI fields are preserved when using
// the adapter with standard OpenAI SDK types (not raw SSE streaming).
//
// The OpenAI SDK captures extra fields in the JSON.ExtraFields map, which
// should be preserved during request/response transformation.
// ============================================================================

func TestAdapter_PreservesResponseExtraFields_NoToolCalls(t *testing.T) {
	// When no tool calls are found, the original response should be returned unchanged
	adapter := New(WithLogLevel(slog.LevelError))

	// Create a response with regular content (no tool calls)
	resp := openai.ChatCompletion{
		ID:    "chatcmpl-123",
		Model: "deepseek-r1",
		Choices: []openai.ChatCompletionChoice{
			{
				Index:        0,
				FinishReason: "stop",
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: "Hello, how can I help you today?",
				},
			},
		},
	}

	result, err := adapter.TransformCompletionsResponse(resp)
	require.NoError(t, err)

	// Since no tool calls were detected, the original response should be returned
	assert.Equal(t, resp.ID, result.ID)
	assert.Equal(t, resp.Model, result.Model)
	assert.Equal(t, "Hello, how can I help you today?", result.Choices[0].Message.Content)
	assert.Empty(t, result.Choices[0].Message.ToolCalls)
}

func TestAdapter_PreservesResponseMetadata_WithToolCalls(t *testing.T) {
	// When tool calls are detected, response metadata should be preserved
	adapter := New(WithLogLevel(slog.LevelError))

	// Create a response with tool call content
	resp := openai.ChatCompletion{
		ID:                "chatcmpl-456",
		Model:             "deepseek-r1",
		SystemFingerprint: "fp_abc123", //nolint:staticcheck // Testing deprecated field preservation
		Choices: []openai.ChatCompletionChoice{
			{
				Index:        0,
				FinishReason: "stop",
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: `[{"name": "get_weather", "parameters": {"city": "Seattle"}}]`,
				},
			},
		},
	}

	result, err := adapter.TransformCompletionsResponse(resp)
	require.NoError(t, err)

	// Metadata should be preserved
	assert.Equal(t, "chatcmpl-456", result.ID)
	assert.Equal(t, "deepseek-r1", result.Model)
	assert.Equal(t, "fp_abc123", result.SystemFingerprint) //nolint:staticcheck // Testing deprecated field preservation

	// Tool calls should be extracted
	require.Len(t, result.Choices, 1)
	require.NotEmpty(t, result.Choices[0].Message.ToolCalls)
	assert.Equal(t, "get_weather", result.Choices[0].Message.ToolCalls[0].Function.Name)
}

func TestAdapter_PreservesChoiceMetadata_WithToolCalls(t *testing.T) {
	// When tool calls are detected, choice-level metadata should be preserved
	adapter := New(
		WithLogLevel(slog.LevelError),
		WithToolPolicy(ToolAllowMixed), // Allow mixed mode to preserve content
	)

	resp := openai.ChatCompletion{
		ID:    "chatcmpl-789",
		Model: "gpt-4",
		Choices: []openai.ChatCompletionChoice{
			{
				Index:        0,
				FinishReason: "stop",
				Logprobs: openai.ChatCompletionChoiceLogprobs{
					Content: []openai.ChatCompletionTokenLogprob{
						{Token: "test"},
					},
				},
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: `[{"name": "search", "parameters": {"query": "test"}}]`,
				},
			},
		},
	}

	result, err := adapter.TransformCompletionsResponse(resp)
	require.NoError(t, err)

	// Choice metadata should be preserved
	assert.Equal(t, int64(0), result.Choices[0].Index)
	// Logprobs should be preserved (shallow copy)
	assert.Len(t, result.Choices[0].Logprobs.Content, 1)
}

func TestAdapter_PreservesUsage_WithToolCalls(t *testing.T) {
	// Usage information should be preserved during transformation
	adapter := New(WithLogLevel(slog.LevelError))

	resp := openai.ChatCompletion{
		ID:    "chatcmpl-usage",
		Model: "gpt-4",
		Choices: []openai.ChatCompletionChoice{
			{
				Index:        0,
				FinishReason: "stop",
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: `[{"name": "func1", "parameters": {}}]`,
				},
			},
		},
		Usage: openai.CompletionUsage{
			PromptTokens:     100,
			CompletionTokens: 50,
			TotalTokens:      150,
		},
	}

	result, err := adapter.TransformCompletionsResponse(resp)
	require.NoError(t, err)

	// Usage should be preserved
	assert.Equal(t, int64(100), result.Usage.PromptTokens)
	assert.Equal(t, int64(50), result.Usage.CompletionTokens)
	assert.Equal(t, int64(150), result.Usage.TotalTokens)
}

func TestAdapter_MultipleChoices_PreservesMetadata(t *testing.T) {
	// Multiple choices should each preserve their metadata
	adapter := New(
		WithLogLevel(slog.LevelError),
		WithToolPolicy(ToolDrainAll),
	)

	resp := openai.ChatCompletion{
		ID:    "chatcmpl-multi",
		Model: "gpt-4",
		Choices: []openai.ChatCompletionChoice{
			{
				Index:        0,
				FinishReason: "stop",
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: `[{"name": "tool1", "parameters": {}}]`,
				},
			},
			{
				Index:        1,
				FinishReason: "stop",
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: "Regular text response",
				},
			},
			{
				Index:        2,
				FinishReason: "length",
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: `[{"name": "tool2", "parameters": {}}]`,
				},
			},
		},
	}

	result, err := adapter.TransformCompletionsResponse(resp)
	require.NoError(t, err)

	require.Len(t, result.Choices, 3)

	// First choice: has tool calls, finish_reason changed to tool_calls
	assert.NotEmpty(t, result.Choices[0].Message.ToolCalls)
	assert.Equal(t, "tool_calls", result.Choices[0].FinishReason)
	assert.Equal(t, int64(0), result.Choices[0].Index)

	// Second choice: no tool calls, unchanged
	assert.Empty(t, result.Choices[1].Message.ToolCalls)
	assert.Equal(t, "Regular text response", result.Choices[1].Message.Content)
	assert.Equal(t, int64(1), result.Choices[1].Index)

	// Third choice: has tool calls
	assert.NotEmpty(t, result.Choices[2].Message.ToolCalls)
	assert.Equal(t, int64(2), result.Choices[2].Index)
}

func TestAdapter_PreservesServiceTier(t *testing.T) {
	// Service tier should be preserved during transformation
	adapter := New(WithLogLevel(slog.LevelError))

	resp := openai.ChatCompletion{
		ID:          "chatcmpl-tier",
		Model:       "gpt-4",
		ServiceTier: "scale",
		Choices: []openai.ChatCompletionChoice{
			{
				Index:        0,
				FinishReason: "stop",
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: `[{"name": "test_func", "parameters": {}}]`,
				},
			},
		},
	}

	result, err := adapter.TransformCompletionsResponse(resp)
	require.NoError(t, err)

	// Service tier should be preserved
	assert.Equal(t, openai.ChatCompletionServiceTier("scale"), result.ServiceTier)
}

func TestAdapter_SDKExtraFields_Accessible(t *testing.T) {
	// Verify that the SDK's JSON.ExtraFields mechanism works with our adapter
	// This test documents how to access extra fields from transformed responses

	adapter := New(WithLogLevel(slog.LevelError))

	// Note: In real usage, the response would come from the API with extra fields
	// populated in the JSON struct. Here we test that the transformation
	// doesn't break access to these fields.
	resp := openai.ChatCompletion{
		ID:    "chatcmpl-extra",
		Model: "gpt-4",
		Choices: []openai.ChatCompletionChoice{
			{
				Index:        0,
				FinishReason: "stop",
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: "Hello!",
				},
			},
		},
	}

	result, err := adapter.TransformCompletionsResponse(resp)
	require.NoError(t, err)

	// The JSON struct should be accessible (even if empty for test data)
	// In production, responses from the API would have this populated
	_ = result.JSON.ExtraFields // Should compile and not panic

	// Similarly for choices
	if len(result.Choices) > 0 {
		_ = result.Choices[0].JSON.ExtraFields
		_ = result.Choices[0].Message.JSON.ExtraFields
	}
}

func TestAdapter_ContextCancellation_PreservesResponse(t *testing.T) {
	// Even with context cancellation checks, responses should be preserved correctly
	adapter := New(WithLogLevel(slog.LevelError))

	ctx := context.Background()

	resp := openai.ChatCompletion{
		ID:                "chatcmpl-ctx",
		Model:             "gpt-4",
		SystemFingerprint: "fp_test", //nolint:staticcheck // Testing deprecated field preservation
		Choices: []openai.ChatCompletionChoice{
			{
				Index:        0,
				FinishReason: "stop",
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: `[{"name": "ctx_func", "parameters": {"key": "value"}}]`,
				},
			},
		},
	}

	result, err := adapter.TransformCompletionsResponseWithContext(ctx, resp)
	require.NoError(t, err)

	// All metadata should be preserved
	assert.Equal(t, "chatcmpl-ctx", result.ID)
	assert.Equal(t, "gpt-4", result.Model)
	assert.Equal(t, "fp_test", result.SystemFingerprint) //nolint:staticcheck // Testing deprecated field preservation

	// Tool call should be extracted
	require.NotEmpty(t, result.Choices[0].Message.ToolCalls)
	assert.Equal(t, "ctx_func", result.Choices[0].Message.ToolCalls[0].Function.Name)
}

// ============================================================================
// Request Extra Fields Tests
// ============================================================================

func TestAdapter_PreservesRequestExtraFields(t *testing.T) {
	// Test that extra fields set on request params are preserved after transformation
	adapter := New(WithLogLevel(slog.LevelError))

	req := openai.ChatCompletionNewParams{
		Model: "gpt-4",
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("Hello"),
		},
		Tools: []openai.ChatCompletionToolUnionParam{
			openai.ChatCompletionFunctionTool(openai.FunctionDefinitionParam{
				Name:        "test_func",
				Description: openai.String("A test function"),
				Parameters: openai.FunctionParameters{
					"type": "object",
					"properties": map[string]interface{}{
						"arg": map[string]interface{}{"type": "string"},
					},
				},
			}),
		},
	}

	// Set extra fields on the request (e.g., custom provider fields)
	req.SetExtraFields(map[string]any{
		"custom_field":      "custom_value",
		"provider_specific": 123,
		"reasoning_effort":  "high",
	})

	result, err := adapter.TransformCompletionsRequest(req)
	require.NoError(t, err)

	// Extra fields should be preserved
	extraFields := result.ExtraFields()
	assert.NotNil(t, extraFields, "Extra fields should be preserved")
	assert.Equal(t, "custom_value", extraFields["custom_field"])
	assert.Equal(t, 123, extraFields["provider_specific"])
	assert.Equal(t, "high", extraFields["reasoning_effort"])

	// Tools should be removed (transformed into prompt)
	assert.Nil(t, result.Tools)
}

func TestAdapter_PreservesRequestMetadata(t *testing.T) {
	// Test that standard request params are preserved
	adapter := New(WithLogLevel(slog.LevelError))

	req := openai.ChatCompletionNewParams{
		Model: "gpt-4",
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("Hello"),
		},
		Tools: []openai.ChatCompletionToolUnionParam{
			openai.ChatCompletionFunctionTool(openai.FunctionDefinitionParam{
				Name: "func1",
			}),
		},
		Temperature: openai.Float(0.7),
		MaxTokens:   openai.Int(1000),
		TopP:        openai.Float(0.9),
		N:           openai.Int(1),
		Seed:        openai.Int(42),
		Store:       openai.Bool(true),
	}

	result, err := adapter.TransformCompletionsRequest(req)
	require.NoError(t, err)

	// All standard params should be preserved
	assert.Equal(t, "gpt-4", string(result.Model))
	assert.Equal(t, float64(0.7), result.Temperature.Value)
	assert.Equal(t, int64(1000), result.MaxTokens.Value)
	assert.Equal(t, float64(0.9), result.TopP.Value)
	assert.Equal(t, int64(1), result.N.Value)
	assert.Equal(t, int64(42), result.Seed.Value)
	assert.True(t, result.Store.Value)
}

func TestAdapter_RequestWithNoTools_Unchanged(t *testing.T) {
	// When no tools are present, request should pass through with all fields preserved
	adapter := New(WithLogLevel(slog.LevelError))

	req := openai.ChatCompletionNewParams{
		Model: "gpt-4",
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("Hello"),
		},
		Temperature: openai.Float(0.5),
	}

	// Set extra fields
	req.SetExtraFields(map[string]any{
		"custom": "value",
	})

	result, err := adapter.TransformCompletionsRequest(req)
	require.NoError(t, err)

	// Extra fields should still be preserved (even though no transformation happened)
	extraFields := result.ExtraFields()
	assert.Equal(t, "value", extraFields["custom"])
}

// Benchmark to ensure extra field preservation doesn't add overhead
func BenchmarkAdapter_ResponseTransformation_WithMetadata(b *testing.B) {
	adapter := New(WithLogLevel(slog.LevelError))

	resp := openai.ChatCompletion{
		ID:                "chatcmpl-bench",
		Model:             "gpt-4",
		SystemFingerprint: "fp_bench",
		Choices: []openai.ChatCompletionChoice{
			{
				Index:        0,
				FinishReason: "stop",
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: `[{"name": "bench_func", "parameters": {"arg": "value"}}]`,
				},
			},
		},
		Usage: openai.CompletionUsage{
			PromptTokens:     100,
			CompletionTokens: 50,
			TotalTokens:      150,
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = adapter.TransformCompletionsResponse(resp)
	}
}
