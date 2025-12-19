package tooladapter

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ============================================================================
// Extra Fields Preservation Tests
//
// These tests verify that non-standard OpenAI fields (like "reasoning",
// "reasoning_content", "reasoning_signature") are preserved during
// request/response transformation.
//
// Many providers (vLLM, LiteLLM, etc.) add custom fields that must not be
// dropped when the adapter processes requests and responses.
// ============================================================================

func TestSSEChunk_PreservesExtraFields(t *testing.T) {
	// JSON with non-standard fields like "reasoning" from vLLM/DeepSeek
	jsonWithExtras := `{
		"id": "chatcmpl-123",
		"object": "chat.completion.chunk",
		"created": 1234567890,
		"model": "deepseek-r1",
		"reasoning": "Let me think about this step by step...",
		"reasoning_content": "This is my internal reasoning process",
		"custom_provider_field": {"nested": "value"},
		"choices": [{
			"index": 0,
			"delta": {
				"role": "assistant",
				"content": "Hello",
				"reasoning_signature": "sig123"
			},
			"finish_reason": null,
			"logprobs": {"tokens": ["Hello"]}
		}],
		"system_fingerprint": "fp_abc123"
	}`

	chunk, err := ParseSSEChunk(jsonWithExtras)
	require.NoError(t, err)

	// Verify standard fields were parsed
	assert.Equal(t, "chatcmpl-123", chunk.ID)
	assert.Equal(t, "deepseek-r1", chunk.Model)
	assert.Equal(t, "Hello", chunk.Choices[0].Delta.Content)

	// Verify extra fields are preserved at chunk level
	assert.NotNil(t, chunk.ExtraFields, "ExtraFields should not be nil")
	assert.Contains(t, chunk.ExtraFields, "reasoning")
	assert.Contains(t, chunk.ExtraFields, "reasoning_content")
	assert.Contains(t, chunk.ExtraFields, "custom_provider_field")
	assert.Contains(t, chunk.ExtraFields, "system_fingerprint")

	// Verify reasoning field content
	var reasoning string
	err = json.Unmarshal(chunk.ExtraFields["reasoning"], &reasoning)
	require.NoError(t, err)
	assert.Equal(t, "Let me think about this step by step...", reasoning)

	// Verify extra fields are preserved at delta level
	assert.NotNil(t, chunk.Choices[0].Delta.ExtraFields)
	assert.Contains(t, chunk.Choices[0].Delta.ExtraFields, "reasoning_signature")

	// Verify extra fields are preserved at choice level
	assert.NotNil(t, chunk.Choices[0].ExtraFields)
	assert.Contains(t, chunk.Choices[0].ExtraFields, "logprobs")

	// Marshal back and verify extra fields are included
	marshaled, err := json.Marshal(chunk)
	require.NoError(t, err)

	marshaledStr := string(marshaled)
	assert.Contains(t, marshaledStr, `"reasoning"`)
	assert.Contains(t, marshaledStr, `"reasoning_content"`)
	assert.Contains(t, marshaledStr, `"custom_provider_field"`)
	assert.Contains(t, marshaledStr, `"system_fingerprint"`)
	assert.Contains(t, marshaledStr, `"reasoning_signature"`)
	assert.Contains(t, marshaledStr, `"logprobs"`)
}

func TestSSEStreamAdapter_PreservesExtraFieldsOnPassthrough(t *testing.T) {
	// SSE stream with extra fields - no tool calls, should passthrough
	sseData := strings.Join([]string{
		`data: {"id":"1","object":"chat.completion.chunk","model":"deepseek-r1","reasoning":"thinking...","choices":[{"index":0,"delta":{"content":"Hello","reasoning_signature":"sig1"},"logprobs":null}]}`,
		"",
		`data: {"id":"1","choices":[{"index":0,"delta":{"content":" World"},"finish_reason":"stop"}],"system_fingerprint":"fp_123"}`,
		"",
		"data: [DONE]",
		"",
	}, "\n")

	rc := io.NopCloser(strings.NewReader(sseData))
	reader := NewSSEReaderFromReadCloser(rc)
	defer func() { _ = reader.Close() }()

	writer := newMockSSEWriter()

	adapter := New(WithLogLevel(slog.LevelError))
	sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)

	err := sseAdapter.Process(context.Background())
	require.NoError(t, err)

	// Should passthrough since no tool calls
	assert.NotEmpty(t, writer.rawWrites, "Expected raw passthrough writes")

	// Verify extra fields are present in passthrough
	allOutput := ""
	for _, raw := range writer.rawWrites {
		allOutput += string(raw)
	}

	assert.Contains(t, allOutput, `"reasoning"`)
	assert.Contains(t, allOutput, `"reasoning_signature"`)
	assert.Contains(t, allOutput, `"system_fingerprint"`)
}

func TestSSEStreamAdapter_PreservesExtraFieldsOnToolTransformation(t *testing.T) {
	// SSE stream with extra fields AND tool calls
	// The tool call should be transformed but extra fields should be preserved
	// Note: The tool JSON content needs to be properly escaped within the JSON string
	sseData := strings.Join([]string{
		`data: {"id":"chatcmpl-456","object":"chat.completion.chunk","model":"deepseek-r1","reasoning":"I need to check the weather","system_fingerprint":"fp_abc","choices":[{"index":0,"delta":{"content":"[{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Seattle\"}}]","reasoning_signature":"sig456"},"logprobs":{"tokens":["test"]}}]}`,
		"",
		"data: [DONE]",
		"",
	}, "\n")

	rc := io.NopCloser(strings.NewReader(sseData))
	reader := NewSSEReaderFromReadCloser(rc)
	defer func() { _ = reader.Close() }()

	writer := newMockSSEWriter()

	adapter := New(WithLogLevel(slog.LevelError))
	sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)

	err := sseAdapter.Process(context.Background())
	require.NoError(t, err)

	// Should have transformed tool call chunks
	require.NotEmpty(t, writer.chunks, "Expected transformed chunks")

	// First chunk should be the tool call chunk
	toolChunk := writer.chunks[0]

	// Verify tool call was detected
	require.NotEmpty(t, toolChunk.Choices)
	require.NotEmpty(t, toolChunk.Choices[0].Delta.ToolCalls)
	assert.Equal(t, "get_weather", toolChunk.Choices[0].Delta.ToolCalls[0].Function.Name)

	// Verify chunk-level extra fields are preserved
	assert.NotNil(t, toolChunk.ExtraFields, "Chunk ExtraFields should be preserved")
	assert.Contains(t, toolChunk.ExtraFields, "reasoning", "reasoning field should be preserved")
	assert.Contains(t, toolChunk.ExtraFields, "system_fingerprint", "system_fingerprint should be preserved")

	// Marshal the output chunk and verify extra fields
	marshaled, err := json.Marshal(toolChunk)
	require.NoError(t, err)

	marshaledStr := string(marshaled)
	assert.Contains(t, marshaledStr, `"reasoning"`)
	assert.Contains(t, marshaledStr, `"system_fingerprint"`)

	t.Logf("Transformed chunk JSON: %s", marshaledStr)
}

func TestSSEDelta_PreservesExtraFields(t *testing.T) {
	jsonWithExtras := `{
		"role": "assistant",
		"content": "Hello",
		"reasoning_signature": "signature123",
		"audio": {"data": "base64audio"},
		"custom_field": 42
	}`

	var delta SSEDelta
	err := json.Unmarshal([]byte(jsonWithExtras), &delta)
	require.NoError(t, err)

	// Verify standard fields
	assert.Equal(t, "assistant", delta.Role)
	assert.Equal(t, "Hello", delta.Content)

	// Verify extra fields preserved
	assert.NotNil(t, delta.ExtraFields)
	assert.Contains(t, delta.ExtraFields, "reasoning_signature")
	assert.Contains(t, delta.ExtraFields, "audio")
	assert.Contains(t, delta.ExtraFields, "custom_field")

	// Marshal back and verify
	marshaled, err := json.Marshal(delta)
	require.NoError(t, err)

	marshaledStr := string(marshaled)
	assert.Contains(t, marshaledStr, `"reasoning_signature"`)
	assert.Contains(t, marshaledStr, `"audio"`)
	assert.Contains(t, marshaledStr, `"custom_field"`)
}

func TestSSEChoice_PreservesExtraFields(t *testing.T) {
	jsonWithExtras := `{
		"index": 0,
		"delta": {"content": "test"},
		"finish_reason": "stop",
		"logprobs": {"tokens": ["test"]},
		"custom_choice_field": "value"
	}`

	var choice SSEChoice
	err := json.Unmarshal([]byte(jsonWithExtras), &choice)
	require.NoError(t, err)

	// Verify standard fields
	assert.Equal(t, 0, choice.Index)
	assert.Equal(t, "test", choice.Delta.Content)
	assert.Equal(t, "stop", choice.FinishReason)

	// Verify extra fields preserved
	assert.NotNil(t, choice.ExtraFields)
	assert.Contains(t, choice.ExtraFields, "logprobs")
	assert.Contains(t, choice.ExtraFields, "custom_choice_field")

	// Marshal back and verify
	marshaled, err := json.Marshal(choice)
	require.NoError(t, err)

	marshaledStr := string(marshaled)
	assert.Contains(t, marshaledStr, `"logprobs"`)
	assert.Contains(t, marshaledStr, `"custom_choice_field"`)
}

func TestSSEUsage_PreservesExtraFields(t *testing.T) {
	jsonWithExtras := `{
		"prompt_tokens": 100,
		"completion_tokens": 50,
		"total_tokens": 150,
		"prompt_tokens_details": {"cached_tokens": 20},
		"completion_tokens_details": {"reasoning_tokens": 30}
	}`

	var usage SSEUsage
	err := json.Unmarshal([]byte(jsonWithExtras), &usage)
	require.NoError(t, err)

	// Verify standard fields
	assert.Equal(t, 100, usage.PromptTokens)
	assert.Equal(t, 50, usage.CompletionTokens)
	assert.Equal(t, 150, usage.TotalTokens)

	// Verify extra fields preserved
	assert.NotNil(t, usage.ExtraFields)
	assert.Contains(t, usage.ExtraFields, "prompt_tokens_details")
	assert.Contains(t, usage.ExtraFields, "completion_tokens_details")

	// Marshal back and verify
	marshaled, err := json.Marshal(usage)
	require.NoError(t, err)

	marshaledStr := string(marshaled)
	assert.Contains(t, marshaledStr, `"prompt_tokens_details"`)
	assert.Contains(t, marshaledStr, `"completion_tokens_details"`)
}

// Test real-world vLLM response format
func TestSSEChunk_vLLMFormat(t *testing.T) {
	// Actual vLLM response format with extra fields
	vllmJSON := `{
		"id": "cmpl-abc123",
		"object": "chat.completion.chunk",
		"created": 1700000000,
		"model": "meta-llama/Llama-3.1-70B-Instruct",
		"choices": [{
			"index": 0,
			"delta": {
				"role": "assistant",
				"content": "Hello!"
			},
			"logprobs": null,
			"finish_reason": null,
			"stop_reason": null
		}],
		"usage": null
	}`

	chunk, err := ParseSSEChunk(vllmJSON)
	require.NoError(t, err)

	assert.Equal(t, "cmpl-abc123", chunk.ID)
	assert.Equal(t, "Hello!", chunk.Choices[0].Delta.Content)

	// vLLM adds stop_reason at choice level
	if chunk.Choices[0].ExtraFields != nil {
		_, hasStopReason := chunk.Choices[0].ExtraFields["stop_reason"]
		assert.True(t, hasStopReason, "stop_reason should be preserved")
	}
}

// Test DeepSeek reasoning format
func TestSSEChunk_DeepSeekReasoningFormat(t *testing.T) {
	// DeepSeek R1 response with reasoning fields
	deepseekJSON := `{
		"id": "chatcmpl-deepseek",
		"object": "chat.completion.chunk",
		"created": 1700000000,
		"model": "deepseek-reasoner",
		"choices": [{
			"index": 0,
			"delta": {
				"role": "assistant",
				"content": "The answer is 42.",
				"reasoning_content": "Let me work through this problem step by step..."
			},
			"finish_reason": "stop"
		}],
		"reasoning": "Internal model reasoning trace",
		"reasoning_tokens": 150
	}`

	chunk, err := ParseSSEChunk(deepseekJSON)
	require.NoError(t, err)

	assert.Equal(t, "chatcmpl-deepseek", chunk.ID)
	assert.Equal(t, "The answer is 42.", chunk.Choices[0].Delta.Content)

	// Verify reasoning fields at chunk level
	assert.NotNil(t, chunk.ExtraFields)
	assert.Contains(t, chunk.ExtraFields, "reasoning")
	assert.Contains(t, chunk.ExtraFields, "reasoning_tokens")

	// Verify reasoning_content at delta level
	assert.NotNil(t, chunk.Choices[0].Delta.ExtraFields)
	assert.Contains(t, chunk.Choices[0].Delta.ExtraFields, "reasoning_content")

	// Marshal and verify preservation
	marshaled, err := json.Marshal(chunk)
	require.NoError(t, err)

	marshaledStr := string(marshaled)
	assert.Contains(t, marshaledStr, `"reasoning"`)
	assert.Contains(t, marshaledStr, `"reasoning_tokens"`)
	assert.Contains(t, marshaledStr, `"reasoning_content"`)
}

// Benchmark to ensure extra fields handling doesn't add significant overhead
func BenchmarkSSEChunk_WithExtraFields(b *testing.B) {
	jsonWithExtras := `{
		"id": "chatcmpl-123",
		"object": "chat.completion.chunk",
		"created": 1234567890,
		"model": "gpt-4",
		"reasoning": "thinking...",
		"system_fingerprint": "fp_123",
		"choices": [{
			"index": 0,
			"delta": {"content": "Hello", "reasoning_signature": "sig"},
			"logprobs": {"tokens": ["Hello"]}
		}]
	}`

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		chunk, _ := ParseSSEChunk(jsonWithExtras)
		_, _ = json.Marshal(chunk)
	}
}

func BenchmarkSSEChunk_WithoutExtraFields(b *testing.B) {
	jsonWithoutExtras := `{
		"id": "chatcmpl-123",
		"object": "chat.completion.chunk",
		"created": 1234567890,
		"model": "gpt-4",
		"choices": [{
			"index": 0,
			"delta": {"content": "Hello"}
		}]
	}`

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		chunk, _ := ParseSSEChunk(jsonWithoutExtras)
		_, _ = json.Marshal(chunk)
	}
}
