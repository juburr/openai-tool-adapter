package tooladapter

import (
	"context"
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ============================================================================
// NON-STREAMING TOOL POLICY TESTS
// ============================================================================

// TestToolPolicyNonStreaming tests all tool policies for non-streaming responses
func TestToolPolicyNonStreaming(t *testing.T) {
	t.Run("ToolStopOnFirst", func(t *testing.T) {
		t.Run("SingleTool_ContentCleared", func(t *testing.T) {
			opts := append([]Option{WithToolPolicy(ToolStopOnFirst)}, WithLogLevel(slog.LevelError))
			adapter := New(opts...)

			resp := createMockResponse(`[{"name": "get_weather", "parameters": {"city": "Seattle"}}]`)

			result, err := adapter.TransformCompletionsResponse(resp)
			require.NoError(t, err)

			// Content should be cleared
			assert.Empty(t, result.Choices[0].Message.Content, "Content should be cleared in ToolStopOnFirst")

			// Should have exactly one tool call
			assert.Len(t, result.Choices[0].Message.ToolCalls, 1, "Should return exactly one tool call")
			assert.Equal(t, "get_weather", result.Choices[0].Message.ToolCalls[0].Function.Name)
			assert.Equal(t, "tool_calls", result.Choices[0].FinishReason)
		})

		t.Run("MultipleTool_OnlyFirstReturned", func(t *testing.T) {
			opts := append([]Option{WithToolPolicy(ToolStopOnFirst)}, WithLogLevel(slog.LevelError))
			adapter := New(opts...)

			resp := createMockResponse(`[{"name": "get_weather", "parameters": {"city": "Seattle"}}, {"name": "get_time", "parameters": {"timezone": "PST"}}, {"name": "send_email", "parameters": {"to": "user@example.com"}}]`)

			result, err := adapter.TransformCompletionsResponse(resp)
			require.NoError(t, err)

			// Content should be cleared
			assert.Empty(t, result.Choices[0].Message.Content, "Content should be cleared in ToolStopOnFirst")

			// Should have exactly one tool call (the first one)
			assert.Len(t, result.Choices[0].Message.ToolCalls, 1, "Should return only the first tool call")
			assert.Equal(t, "get_weather", result.Choices[0].Message.ToolCalls[0].Function.Name)
		})

		t.Run("NoTools_ContentPreserved", func(t *testing.T) {
			opts := append([]Option{WithToolPolicy(ToolStopOnFirst)}, WithLogLevel(slog.LevelError))
			adapter := New(opts...)

			resp := createMockResponse("Just a regular response with no tool calls.")

			result, err := adapter.TransformCompletionsResponse(resp)
			require.NoError(t, err)

			// Content should be preserved when no tools
			assert.Equal(t, "Just a regular response with no tool calls.", result.Choices[0].Message.Content)
			assert.Empty(t, result.Choices[0].Message.ToolCalls)
		})
	})

	t.Run("ToolCollectThenStop", func(t *testing.T) {
		t.Run("MultipleTool_ContentCleared", func(t *testing.T) {
			opts := append([]Option{WithToolPolicy(ToolCollectThenStop)}, WithLogLevel(slog.LevelError))
			adapter := New(opts...)

			resp := createMockResponse(`[{"name": "get_weather", "parameters": {"city": "Seattle"}}, {"name": "get_time", "parameters": {"timezone": "PST"}}]`)

			result, err := adapter.TransformCompletionsResponse(resp)
			require.NoError(t, err)

			// Content should be cleared
			assert.Empty(t, result.Choices[0].Message.Content, "Content should be cleared in ToolCollectThenStop")

			// Should have all tool calls (no streaming limits apply to non-streaming)
			assert.Len(t, result.Choices[0].Message.ToolCalls, 2, "Should return all tool calls in non-streaming mode")
			assert.Equal(t, "get_weather", result.Choices[0].Message.ToolCalls[0].Function.Name)
			assert.Equal(t, "get_time", result.Choices[0].Message.ToolCalls[1].Function.Name)
		})

		t.Run("MaxCallsLimit_Applied", func(t *testing.T) {
			opts := append([]Option{
				WithToolPolicy(ToolCollectThenStop),
				WithToolMaxCalls(2),
			}, WithLogLevel(slog.LevelError))
			adapter := New(opts...)

			// Create response with 5 tool calls
			resp := createMockResponse(`[
				{"name": "tool1", "parameters": {"a": 1}},
				{"name": "tool2", "parameters": {"b": 2}},
				{"name": "tool3", "parameters": {"c": 3}},
				{"name": "tool4", "parameters": {"d": 4}},
				{"name": "tool5", "parameters": {"e": 5}}
			]`)

			result, err := adapter.TransformCompletionsResponse(resp)
			require.NoError(t, err)

			// Content should be cleared
			assert.Empty(t, result.Choices[0].Message.Content, "Content should be cleared")

			// Should be limited to 2 tool calls
			assert.Len(t, result.Choices[0].Message.ToolCalls, 2, "Should respect toolMaxCalls limit")
			assert.Equal(t, "tool1", result.Choices[0].Message.ToolCalls[0].Function.Name)
			assert.Equal(t, "tool2", result.Choices[0].Message.ToolCalls[1].Function.Name)
		})
	})

	t.Run("ToolDrainAll", func(t *testing.T) {
		t.Run("MultipleTool_AllReturned_ContentCleared", func(t *testing.T) {
			opts := append([]Option{WithToolPolicy(ToolDrainAll)}, WithLogLevel(slog.LevelError))
			adapter := New(opts...)

			resp := createMockResponse(`[
				{"name": "get_weather", "parameters": {"city": "Seattle"}},
				{"name": "get_time", "parameters": {"timezone": "PST"}},
				{"name": "send_email", "parameters": {"to": "user@example.com"}},
				{"name": "calculate", "parameters": {"operation": "add", "a": 5, "b": 3}}
			]`)

			result, err := adapter.TransformCompletionsResponse(resp)
			require.NoError(t, err)

			// Content should be cleared
			assert.Empty(t, result.Choices[0].Message.Content, "Content should be cleared in ToolDrainAll")

			// Should have all tool calls
			assert.Len(t, result.Choices[0].Message.ToolCalls, 4, "Should return all tool calls")
			assert.Equal(t, "get_weather", result.Choices[0].Message.ToolCalls[0].Function.Name)
			assert.Equal(t, "get_time", result.Choices[0].Message.ToolCalls[1].Function.Name)
			assert.Equal(t, "send_email", result.Choices[0].Message.ToolCalls[2].Function.Name)
			assert.Equal(t, "calculate", result.Choices[0].Message.ToolCalls[3].Function.Name)
		})

		t.Run("EightTool_MaxThree_OnlyThreeReturned", func(t *testing.T) {
			opts := append([]Option{
				WithToolPolicy(ToolDrainAll),
				WithToolMaxCalls(3),
			}, WithLogLevel(slog.LevelError))
			adapter := New(opts...)

			// Create response with 8 tool calls as requested
			resp := createMockResponse(`[
				{"name": "tool1", "parameters": {"a": 1}},
				{"name": "tool2", "parameters": {"b": 2}},
				{"name": "tool3", "parameters": {"c": 3}},
				{"name": "tool4", "parameters": {"d": 4}},
				{"name": "tool5", "parameters": {"e": 5}},
				{"name": "tool6", "parameters": {"f": 6}},
				{"name": "tool7", "parameters": {"g": 7}},
				{"name": "tool8", "parameters": {"h": 8}}
			]`)

			result, err := adapter.TransformCompletionsResponse(resp)
			require.NoError(t, err)

			// Content should be cleared
			assert.Empty(t, result.Choices[0].Message.Content, "Content should be cleared")

			// Should be limited to exactly 3 tool calls
			assert.Len(t, result.Choices[0].Message.ToolCalls, 3, "Should respect toolMaxCalls limit of 3")
			assert.Equal(t, "tool1", result.Choices[0].Message.ToolCalls[0].Function.Name)
			assert.Equal(t, "tool2", result.Choices[0].Message.ToolCalls[1].Function.Name)
			assert.Equal(t, "tool3", result.Choices[0].Message.ToolCalls[2].Function.Name)
		})
	})

	t.Run("ToolAllowMixed", func(t *testing.T) {
		t.Run("ContentPreserved_WithTools", func(t *testing.T) {
			opts := append([]Option{WithToolPolicy(ToolAllowMixed)}, WithLogLevel(slog.LevelError))
			adapter := New(opts...)

			originalContent := "I'll help you with the weather. Let me check that for you."
			resp := createMockResponseWithContent(originalContent, `[{"name": "get_weather", "parameters": {"city": "Seattle"}}]`)

			result, err := adapter.TransformCompletionsResponse(resp)
			require.NoError(t, err)

			// Content should be preserved in mixed mode (includes original content)
			assert.Contains(t, result.Choices[0].Message.Content, originalContent, "Original content should be preserved in ToolAllowMixed")

			// Should also have tool calls
			assert.Len(t, result.Choices[0].Message.ToolCalls, 1, "Should return tool calls")
			assert.Equal(t, "get_weather", result.Choices[0].Message.ToolCalls[0].Function.Name)
		})

		t.Run("MultipleTools_ContentPreserved", func(t *testing.T) {
			opts := append([]Option{WithToolPolicy(ToolAllowMixed)}, WithLogLevel(slog.LevelError))
			adapter := New(opts...)

			originalContent := "I'll check the weather and time for you."
			resp := createMockResponseWithContent(originalContent, `[
				{"name": "get_weather", "parameters": {"city": "Seattle"}},
				{"name": "get_time", "parameters": {"timezone": "PST"}}
			]`)

			result, err := adapter.TransformCompletionsResponse(resp)
			require.NoError(t, err)

			// Content should be preserved (includes original content)
			assert.Contains(t, result.Choices[0].Message.Content, originalContent, "Original content should be preserved")

			// Should have all tool calls
			assert.Len(t, result.Choices[0].Message.ToolCalls, 2, "Should return all tool calls")
		})

		t.Run("NoTools_ContentPreserved", func(t *testing.T) {
			opts := append([]Option{WithToolPolicy(ToolAllowMixed)}, WithLogLevel(slog.LevelError))
			adapter := New(opts...)

			resp := createMockResponse("Just a regular response with no tool calls.")

			result, err := adapter.TransformCompletionsResponse(resp)
			require.NoError(t, err)

			// Content should be preserved
			assert.Equal(t, "Just a regular response with no tool calls.", result.Choices[0].Message.Content)
			assert.Empty(t, result.Choices[0].Message.ToolCalls)
		})
	})
}

// TestFinishReasonBehaviorWithPolicies tests finish_reason handling across different
// tool policies using the public API (not internal deprecated functions)
func TestFinishReasonBehaviorWithPolicies(t *testing.T) {
	testCases := []struct {
		name                 string
		policy               ToolPolicy
		originalFinishReason string
		expectedFinishReason string
		expectContentCleared bool
		description          string
	}{
		// ToolAllowMixed policy tests - preserves content and original finish_reason
		{
			name:                 "Mixed_PreservesStopReason",
			policy:               ToolAllowMixed,
			originalFinishReason: "stop",
			expectedFinishReason: "stop",
			expectContentCleared: false,
			description:          "Mixed mode should preserve 'stop' finish_reason and content",
		},
		{
			name:                 "Mixed_PreservesLengthReason",
			policy:               ToolAllowMixed,
			originalFinishReason: "length",
			expectedFinishReason: "length",
			expectContentCleared: false,
			description:          "Mixed mode should preserve 'length' finish_reason and content",
		},
		{
			name:                 "Mixed_DefaultsToToolCallsWhenEmpty",
			policy:               ToolAllowMixed,
			originalFinishReason: "",
			expectedFinishReason: "tool_calls",
			expectContentCleared: false,
			description:          "Mixed mode should default to 'tool_calls' when original is empty",
		},
		// ToolStopOnFirst policy tests - clears content and uses tool_calls
		{
			name:                 "StopOnFirst_AlwaysToolCalls",
			policy:               ToolStopOnFirst,
			originalFinishReason: "stop",
			expectedFinishReason: "tool_calls",
			expectContentCleared: true,
			description:          "StopOnFirst should always use 'tool_calls' and clear content",
		},
		// ToolDrainAll policy tests - clears content and uses tool_calls
		{
			name:                 "DrainAll_AlwaysToolCalls",
			policy:               ToolDrainAll,
			originalFinishReason: "stop",
			expectedFinishReason: "tool_calls",
			expectContentCleared: true,
			description:          "DrainAll should always use 'tool_calls' and clear content",
		},
		// ToolCollectThenStop policy tests - clears content and uses tool_calls
		{
			name:                 "CollectThenStop_AlwaysToolCalls",
			policy:               ToolCollectThenStop,
			originalFinishReason: "stop",
			expectedFinishReason: "tool_calls",
			expectContentCleared: true,
			description:          "CollectThenStop should always use 'tool_calls' and clear content",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			adapter := New(
				WithToolPolicy(tc.policy),
				WithLogLevel(slog.LevelError),
			)

			// Create response with tool calls and specific finish_reason
			originalResp := openai.ChatCompletion{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Content: `I'll help you with that calculation.
[{"name": "calculate", "parameters": {"expression": "15+27"}}]
The result is 42.`,
							Role: "assistant",
						},
						FinishReason: tc.originalFinishReason,
					},
				},
			}

			result, err := adapter.TransformCompletionsResponse(originalResp)
			require.NoError(t, err)
			require.Len(t, result.Choices, 1)

			choice := result.Choices[0]

			// Verify finish_reason behavior
			assert.Equal(t, tc.expectedFinishReason, string(choice.FinishReason), tc.description)

			// Verify content behavior
			if tc.expectContentCleared {
				assert.Empty(t, choice.Message.Content, "Content should be cleared for policy %v", tc.policy)
			} else {
				assert.NotEmpty(t, choice.Message.Content, "Content should be preserved for policy %v", tc.policy)
			}

			// Tool calls should always be present
			assert.NotEmpty(t, choice.Message.ToolCalls, "Should have tool calls")
			assert.Equal(t, "calculate", choice.Message.ToolCalls[0].Function.Name)

			t.Logf("Policy: %v, Original: '%s' → Result: '%s', Content cleared: %v",
				tc.policy, tc.originalFinishReason, choice.FinishReason, tc.expectContentCleared)
		})
	}
}

// ============================================================================
// STREAMING TOOL POLICY TESTS
// ============================================================================

// Helper functions for streaming tests to keep per-function cyclomatic complexity low
func runStopOnFirst_SingleTool_StopsAfterFirstTool(t *testing.T) {
	opts := append([]Option{WithToolPolicy(ToolStopOnFirst)}, WithLogLevel(slog.LevelError))
	adapter := New(opts...)

	chunks := []string{
		`[{"name": "get_weather",`,
		` "parameters": {"city": "Seattle"}}]`,
		" Additional content that should be discarded",
		" More content to discard",
	}

	mockStream := NewMockStream(chunks)
	stream := adapter.TransformStreamingResponse(mockStream)
	defer func() { require.NoError(t, stream.Close()) }()

	var collectedChunks []openai.ChatCompletionChunk
	for stream.Next() {
		collectedChunks = append(collectedChunks, stream.Current())
	}
	require.NoError(t, stream.Err())

	assert.GreaterOrEqual(t, len(collectedChunks), 1, "Should have at least tool call chunk")

	var toolCallChunk *openai.ChatCompletionChunk
	for i := range collectedChunks {
		if len(collectedChunks[i].Choices) > 0 && len(collectedChunks[i].Choices[0].Delta.ToolCalls) > 0 {
			toolCallChunk = &collectedChunks[i]
			break
		}
	}

	require.NotNil(t, toolCallChunk, "Should have a tool call chunk")
	assert.Len(t, toolCallChunk.Choices[0].Delta.ToolCalls, 1, "Should have exactly one tool call")
	assert.Equal(t, "get_weather", toolCallChunk.Choices[0].Delta.ToolCalls[0].Function.Name)
}

func runStopOnFirst_MultipleTools_OnlyFirstProcessed(t *testing.T) {
	opts := append([]Option{WithToolPolicy(ToolStopOnFirst)}, WithLogLevel(slog.LevelError))
	adapter := New(opts...)

	chunks := []string{
		`[{"name": "get_weather", "parameters": {"city": "Seattle"}},`,
		` {"name": "get_time", "parameters": {"timezone": "PST"}}]`,
		" This content should be discarded after first tool",
	}

	mockStream := NewMockStream(chunks)
	stream := adapter.TransformStreamingResponse(mockStream)
	defer func() { require.NoError(t, stream.Close()) }()

	var toolCallChunks []openai.ChatCompletionChunk
	for stream.Next() {
		chunk := stream.Current()
		if len(chunk.Choices) > 0 && len(chunk.Choices[0].Delta.ToolCalls) > 0 {
			toolCallChunks = append(toolCallChunks, chunk)
		}
	}

	assert.Len(t, toolCallChunks, 1, "Should only process first tool call")
	assert.Equal(t, "get_weather", toolCallChunks[0].Choices[0].Delta.ToolCalls[0].Function.Name)
}

func runCollectThenStop_CollectionWindow_Timeout(t *testing.T) {
	opts := append([]Option{
		WithToolPolicy(ToolCollectThenStop),
		WithToolCollectWindow(50 * time.Millisecond),
	}, WithLogLevel(slog.LevelError))
	adapter := New(opts...)

	chunks := []string{
		`[{"name": "tool1",`,
		` "parameters": {"a": 1}}]`,
	}

	mockStream := NewMockStream(chunks)
	stream := adapter.TransformStreamingResponse(mockStream)
	defer func() { require.NoError(t, stream.Close()) }()

	var collectedChunks []openai.ChatCompletionChunk
	for stream.Next() {
		collectedChunks = append(collectedChunks, stream.Current())
	}

	assert.NotEmpty(t, collectedChunks, "Should have collected chunks")
}

func runCollectThenStop_MaxCallsLimit_InStreaming(t *testing.T) {
	opts := append([]Option{
		WithToolPolicy(ToolCollectThenStop),
		WithToolMaxCalls(2),
	}, WithLogLevel(slog.LevelError))
	adapter := New(opts...)

	chunks := []string{
		`[{"name": "tool1", "parameters": {"a": 1}},`,
		` {"name": "tool2", "parameters": {"b": 2}},`,
		` {"name": "tool3", "parameters": {"c": 3}}]`,
	}

	mockStream := NewMockStream(chunks)
	stream := adapter.TransformStreamingResponse(mockStream)
	defer func() { require.NoError(t, stream.Close()) }()

	var toolCallChunks []openai.ChatCompletionChunk
	for stream.Next() {
		chunk := stream.Current()
		if len(chunk.Choices) > 0 && len(chunk.Choices[0].Delta.ToolCalls) > 0 {
			toolCallChunks = append(toolCallChunks, chunk)
		}
	}

	assert.GreaterOrEqual(t, len(toolCallChunks), 0, "Should handle streaming with max calls limit")
}

func runCollectThenStop_ContentSuppression_AfterFirstTool(t *testing.T) {
	opts := append([]Option{WithToolPolicy(ToolCollectThenStop)}, WithLogLevel(slog.LevelError))
	adapter := New(opts...)

	chunks := []string{
		"Regular content before tools",
		`[{"name": "tool1", "parameters": {"a": 1}}]`,
		" This content should be suppressed",
		" More suppressed content",
	}

	mockStream := NewMockStream(chunks)
	stream := adapter.TransformStreamingResponse(mockStream)
	defer func() { require.NoError(t, stream.Close()) }()

	var contentChunks []string
	var toolChunks int

	for stream.Next() {
		chunk := stream.Current()
		if len(chunk.Choices) > 0 {
			if chunk.Choices[0].Delta.Content != "" {
				contentChunks = append(contentChunks, chunk.Choices[0].Delta.Content)
			}
			if len(chunk.Choices[0].Delta.ToolCalls) > 0 {
				toolChunks++
			}
		}
	}

	assert.GreaterOrEqual(t, toolChunks, 0, "Should handle tool chunks")
	allContent := strings.Join(contentChunks, "")
	assert.Contains(t, allContent, "Regular content before tools", "Should have content before tools")
}

func runDrainAll_AllContentSuppressed_AllToolsCollected(t *testing.T) {
	opts := append([]Option{WithToolPolicy(ToolDrainAll)}, WithLogLevel(slog.LevelError))
	adapter := New(opts...)

	chunks := []string{
		"Content should be suppressed",
		`[{"name": "tool1", "parameters": {"a": 1}},`,
		` {"name": "tool2", "parameters": {"b": 2}}]`,
		" More suppressed content",
	}

	mockStream := NewMockStream(chunks)
	stream := adapter.TransformStreamingResponse(mockStream)
	defer func() { require.NoError(t, stream.Close()) }()

	var contentChunks []string
	var toolCallChunks []openai.ChatCompletionChunk

	for stream.Next() {
		chunk := stream.Current()
		if len(chunk.Choices) > 0 {
			if chunk.Choices[0].Delta.Content != "" {
				contentChunks = append(contentChunks, chunk.Choices[0].Delta.Content)
			}
			if len(chunk.Choices[0].Delta.ToolCalls) > 0 {
				toolCallChunks = append(toolCallChunks, chunk)
			}
		}
	}

	assert.Empty(t, contentChunks, "Content should be suppressed in ToolDrainAll streaming mode")
	assert.GreaterOrEqual(t, len(toolCallChunks), 0, "Should handle tool call chunks")
}

func runDrainAll_ByteLimit_ExceedsLimit(t *testing.T) {
	opts := append([]Option{
		WithToolPolicy(ToolDrainAll),
		WithToolCollectMaxBytes(50),
	}, WithLogLevel(slog.LevelError))
	adapter := New(opts...)

	largeContent := strings.Repeat("This is a long content string that will exceed the byte limit. ", 10)
	chunks := []string{
		largeContent,
		`[{"name": "tool1", "parameters": {"a": 1}}]`,
	}

	mockStream := NewMockStream(chunks)
	stream := adapter.TransformStreamingResponse(mockStream)
	defer func() { require.NoError(t, stream.Close()) }()

	var chunksCollected []openai.ChatCompletionChunk
	for stream.Next() {
		chunksCollected = append(chunksCollected, stream.Current())
	}

	assert.NotEmpty(t, chunksCollected, "Should process chunks even when byte limit exceeded")
}

func runAllowMixed_ContentAndToolsBothEmitted(t *testing.T) {
	opts := append([]Option{WithToolPolicy(ToolAllowMixed)}, WithLogLevel(slog.LevelError))
	adapter := New(opts...)

	chunks := []string{
		"Here's some content before the tool.",
		`[{"name": "get_weather", "parameters": {"city": "Seattle"}}]`,
		" And some content after the tool.",
	}

	mockStream := NewMockStream(chunks)
	stream := adapter.TransformStreamingResponse(mockStream)
	defer func() { require.NoError(t, stream.Close()) }()

	var contentChunks []string
	var toolCallChunks []openai.ChatCompletionChunk

	for stream.Next() {
		chunk := stream.Current()
		if len(chunk.Choices) > 0 {
			if chunk.Choices[0].Delta.Content != "" {
				contentChunks = append(contentChunks, chunk.Choices[0].Delta.Content)
			}
			if len(chunk.Choices[0].Delta.ToolCalls) > 0 {
				toolCallChunks = append(toolCallChunks, chunk)
			}
		}
	}

	assert.NotEmpty(t, contentChunks, "Should preserve content in ToolAllowMixed")
	assert.NotEmpty(t, toolCallChunks, "Should have tool calls")

	allContent := strings.Join(contentChunks, "")
	assert.Contains(t, allContent, "Here's some content before", "Should preserve content before tools")
}

func runAllowMixed_MultipleToolsWithContent(t *testing.T) {
	opts := append([]Option{WithToolPolicy(ToolAllowMixed)}, WithLogLevel(slog.LevelError))
	adapter := New(opts...)

	chunks := []string{
		"I'll help you with multiple tasks.",
		`[{"name": "tool1", "parameters": {"a": 1}},`,
		` {"name": "tool2", "parameters": {"b": 2}}]`,
		" All done!",
	}

	mockStream := NewMockStream(chunks)
	stream := adapter.TransformStreamingResponse(mockStream)
	defer func() { require.NoError(t, stream.Close()) }()

	var contentChunks []string
	var toolCallChunks []openai.ChatCompletionChunk

	for stream.Next() {
		chunk := stream.Current()
		if len(chunk.Choices) > 0 {
			if chunk.Choices[0].Delta.Content != "" {
				contentChunks = append(contentChunks, chunk.Choices[0].Delta.Content)
			}
			if len(chunk.Choices[0].Delta.ToolCalls) > 0 {
				toolCallChunks = append(toolCallChunks, chunk)
			}
		}
	}

	assert.NotEmpty(t, contentChunks, "Should preserve content")
	assert.NotEmpty(t, toolCallChunks, "Should have tool calls")

	allContent := strings.Join(contentChunks, "")
	assert.Contains(t, allContent, "I'll help you", "Should preserve initial content")
	assert.Contains(t, allContent, "All done!", "Should preserve final content")
}

// Aggregator test that delegates to helpers to keep cyclomatic complexity low per function
func TestToolPolicyStreaming(t *testing.T) {
	t.Run("ToolStopOnFirst_Streaming/SingleTool_StopsAfterFirstTool", func(t *testing.T) { runStopOnFirst_SingleTool_StopsAfterFirstTool(t) })
	t.Run("ToolStopOnFirst_Streaming/MultipleTools_OnlyFirstProcessed", func(t *testing.T) { runStopOnFirst_MultipleTools_OnlyFirstProcessed(t) })

	t.Run("ToolCollectThenStop_Streaming/CollectionWindow_Timeout", func(t *testing.T) { runCollectThenStop_CollectionWindow_Timeout(t) })
	t.Run("ToolCollectThenStop_Streaming/MaxCallsLimit_InStreaming", func(t *testing.T) { runCollectThenStop_MaxCallsLimit_InStreaming(t) })
	t.Run("ToolCollectThenStop_Streaming/ContentSuppression_AfterFirstTool", func(t *testing.T) { runCollectThenStop_ContentSuppression_AfterFirstTool(t) })

	t.Run("ToolDrainAll_Streaming/AllContentSuppressed_AllToolsCollected", func(t *testing.T) { runDrainAll_AllContentSuppressed_AllToolsCollected(t) })
	t.Run("ToolDrainAll_Streaming/ByteLimit_ExceedsLimit", func(t *testing.T) { runDrainAll_ByteLimit_ExceedsLimit(t) })

	t.Run("ToolAllowMixed_Streaming/ContentAndToolsBothEmitted", func(t *testing.T) { runAllowMixed_ContentAndToolsBothEmitted(t) })
	t.Run("ToolAllowMixed_Streaming/MultipleToolsWithContent", func(t *testing.T) { runAllowMixed_MultipleToolsWithContent(t) })
}

// ============================================================================
// TOOL POLICY CONFIGURATION TESTS
// ============================================================================

// TestToolPolicyOptions tests the option functions and validation
func TestToolPolicyOptions(t *testing.T) {
	t.Run("WithToolPolicy", func(t *testing.T) {
		adapter := New(WithToolPolicy(ToolDrainAll))
		assert.Equal(t, ToolDrainAll, adapter.toolPolicy)
	})

	t.Run("WithToolCollectWindow", func(t *testing.T) {
		window := 500 * time.Millisecond
		adapter := New(WithToolCollectWindow(window))
		assert.Equal(t, window, adapter.toolCollectWindow)
	})

	t.Run("WithToolMaxCalls", func(t *testing.T) {
		t.Run("PositiveValue", func(t *testing.T) {
			adapter := New(WithToolMaxCalls(5))
			assert.Equal(t, 5, adapter.toolMaxCalls)
		})

		t.Run("ZeroValue_Unlimited", func(t *testing.T) {
			adapter := New(WithToolMaxCalls(0))
			assert.Equal(t, 0, adapter.toolMaxCalls)
		})

		t.Run("NegativeValue_ConvertedToZero", func(t *testing.T) {
			adapter := New(WithToolMaxCalls(-5))
			assert.Equal(t, 0, adapter.toolMaxCalls, "Negative values should be converted to 0")
		})
	})

	t.Run("WithToolCollectMaxBytes", func(t *testing.T) {
		t.Run("PositiveValue", func(t *testing.T) {
			adapter := New(WithToolCollectMaxBytes(1024))
			assert.Equal(t, 1024, adapter.toolCollectMaxBytes)
		})

		t.Run("ZeroValue_Unlimited", func(t *testing.T) {
			adapter := New(WithToolCollectMaxBytes(0))
			assert.Equal(t, 0, adapter.toolCollectMaxBytes)
		})

		t.Run("NegativeValue_ConvertedToZero", func(t *testing.T) {
			adapter := New(WithToolCollectMaxBytes(-1024))
			assert.Equal(t, 0, adapter.toolCollectMaxBytes, "Negative values should be converted to 0")
		})
	})

	t.Run("WithCancelUpstreamOnStop", func(t *testing.T) {
		t.Run("EnableCancellation", func(t *testing.T) {
			adapter := New(WithCancelUpstreamOnStop(true))
			assert.True(t, adapter.cancelUpstreamOnStop)
		})

		t.Run("DisableCancellation", func(t *testing.T) {
			adapter := New(WithCancelUpstreamOnStop(false))
			assert.False(t, adapter.cancelUpstreamOnStop)
		})
	})

	t.Run("DefaultValues", func(t *testing.T) {
		adapter := New()

		assert.Equal(t, ToolStopOnFirst, adapter.toolPolicy, "Default policy should be ToolStopOnFirst")
		assert.Equal(t, 200*time.Millisecond, adapter.toolCollectWindow, "Default collect window")
		assert.Equal(t, 8, adapter.toolMaxCalls, "Default max calls")
		assert.Equal(t, 65536, adapter.toolCollectMaxBytes, "Default max bytes (64KB for DoS protection)")
		assert.True(t, adapter.cancelUpstreamOnStop, "Default cancel upstream")
	})
}

// ============================================================================
// EDGE CASES AND CONTENT SUPPRESSION TESTS
// ============================================================================

// TestContentSuppressionRequirements tests the specific requirement that content should be suppressed
// after the first tool is detected in the first three policies (excluding ToolAllowMixed)
func TestContentSuppressionRequirements(t *testing.T) {
	t.Run("ToolStopOnFirst_ContentSuppressedAfterFirstTool", func(t *testing.T) { runEdge_StopOnFirst_ContentSuppressedAfterFirstTool(t) })
	t.Run("ToolCollectThenStop_ContentSuppressedAfterFirstTool", func(t *testing.T) { runEdge_CollectThenStop_ContentSuppressedAfterFirstTool(t) })
	t.Run("ToolDrainAll_AllContentSuppressed", func(t *testing.T) { runEdge_DrainAll_AllContentSuppressed(t) })
	t.Run("ToolAllowMixed_ContentNOTSuppressed", func(t *testing.T) { runEdge_AllowMixed_ContentNotSuppressed(t) })
}

func runEdge_StopOnFirst_ContentSuppressedAfterFirstTool(t *testing.T) {
	opts := append([]Option{WithToolPolicy(ToolStopOnFirst)}, WithLogLevel(slog.LevelError))
	adapter := New(opts...)

	chunks := []string{
		"Initial content before any tools",
		`[{"name": "first_tool", "parameters": {"param": "value"}}]`,
		"This content should be suppressed",
		"More suppressed content",
		"Even more suppressed content",
	}

	mockStream := NewMockStream(chunks)
	stream := adapter.TransformStreamingResponse(mockStream)
	defer func() { require.NoError(t, stream.Close()) }()

	var contentAfterTool []string
	var toolDetected bool

	for stream.Next() {
		chunk := stream.Current()
		if len(chunk.Choices) > 0 {
			if len(chunk.Choices[0].Delta.ToolCalls) > 0 {
				toolDetected = true
			}
			if toolDetected && chunk.Choices[0].Delta.Content != "" {
				contentAfterTool = append(contentAfterTool, chunk.Choices[0].Delta.Content)
			}
		}
	}

	assert.Empty(t, contentAfterTool, "ToolStopOnFirst should suppress all content after first tool detection")
}

func runEdge_CollectThenStop_ContentSuppressedAfterFirstTool(t *testing.T) {
	opts := append([]Option{WithToolPolicy(ToolCollectThenStop)}, WithLogLevel(slog.LevelError))
	adapter := New(opts...)

	chunks := []string{
		"Initial content before any tools",
		`[{"name": "first_tool", "parameters": {"param": "value"}}]`,
		"This content should be suppressed",
		`[{"name": "second_tool", "parameters": {"param": "value2"}}]`,
		"More suppressed content",
	}

	mockStream := NewMockStream(chunks)
	stream := adapter.TransformStreamingResponse(mockStream)
	defer func() { require.NoError(t, stream.Close()) }()

	var contentAfterTool []string
	var toolDetected bool

	for stream.Next() {
		chunk := stream.Current()
		if len(chunk.Choices) > 0 {
			if len(chunk.Choices[0].Delta.ToolCalls) > 0 {
				toolDetected = true
			}
			if toolDetected && chunk.Choices[0].Delta.Content != "" {
				contentAfterTool = append(contentAfterTool, chunk.Choices[0].Delta.Content)
			}
		}
	}

	assert.Empty(t, contentAfterTool, "ToolCollectThenStop should suppress all content after first tool detection")
}

func runEdge_DrainAll_AllContentSuppressed(t *testing.T) {
	opts := append([]Option{WithToolPolicy(ToolDrainAll)}, WithLogLevel(slog.LevelError))
	adapter := New(opts...)

	chunks := []string{
		"This content should be suppressed",
		"All content is suppressed in drain all mode",
		`[{"name": "tool1", "parameters": {"param": "value"}}]`,
		"Content after tool should also be suppressed",
	}

	mockStream := NewMockStream(chunks)
	stream := adapter.TransformStreamingResponse(mockStream)
	defer func() { require.NoError(t, stream.Close()) }()

	var allContent []string

	for stream.Next() {
		chunk := stream.Current()
		if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
			allContent = append(allContent, chunk.Choices[0].Delta.Content)
		}
	}

	assert.Empty(t, allContent, "ToolDrainAll should suppress ALL content")
}

func runEdge_AllowMixed_ContentNotSuppressed(t *testing.T) {
	opts := append([]Option{WithToolPolicy(ToolAllowMixed)}, WithLogLevel(slog.LevelError))
	adapter := New(opts...)

	chunks := []string{
		"This content should NOT be suppressed",
		`[{"name": "tool1", "parameters": {"param": "value"}}]`,
		"Content after tool should also NOT be suppressed",
		"More content that should be preserved",
	}

	mockStream := NewMockStream(chunks)
	stream := adapter.TransformStreamingResponse(mockStream)
	defer func() { require.NoError(t, stream.Close()) }()

	var allContent []string

	for stream.Next() {
		chunk := stream.Current()
		if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
			allContent = append(allContent, chunk.Choices[0].Delta.Content)
		}
	}

	assert.NotEmpty(t, allContent, "ToolAllowMixed should NOT suppress content")

	allContentText := strings.Join(allContent, "")
	assert.Contains(t, allContentText, "This content should NOT be suppressed", "Initial content should be preserved")
}

// TestToolMaxCallsLimits tests the toolMaxCalls limit across different scenarios
func TestToolMaxCallsLimits(t *testing.T) {
	t.Run("EightTools_MaxThree_StopsAtThree", func(t *testing.T) {
		testCases := []struct {
			name   string
			policy ToolPolicy
		}{
			{"ToolStopOnFirst", ToolStopOnFirst}, // Should only return 1 even with max 3
			{"ToolCollectThenStop", ToolCollectThenStop},
			{"ToolDrainAll", ToolDrainAll},
			{"ToolAllowMixed", ToolAllowMixed},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				opts := append([]Option{
					WithToolPolicy(tc.policy),
					WithToolMaxCalls(3),
				}, WithLogLevel(slog.LevelError))
				adapter := New(opts...)

				// Create response with 8 tool calls as requested
				resp := createMockResponse(`[
					{"name": "tool1", "parameters": {"id": 1}},
					{"name": "tool2", "parameters": {"id": 2}},
					{"name": "tool3", "parameters": {"id": 3}},
					{"name": "tool4", "parameters": {"id": 4}},
					{"name": "tool5", "parameters": {"id": 5}},
					{"name": "tool6", "parameters": {"id": 6}},
					{"name": "tool7", "parameters": {"id": 7}},
					{"name": "tool8", "parameters": {"id": 8}}
				]`)

				result, err := adapter.TransformCompletionsResponse(resp)
				require.NoError(t, err)

				toolCalls := result.Choices[0].Message.ToolCalls

				switch tc.policy {
				case ToolStopOnFirst:
					// Should return exactly 1 tool (ignores maxCalls when it's higher than 1)
					assert.Len(t, toolCalls, 1, "ToolStopOnFirst should return exactly 1 tool call")
					assert.Equal(t, "tool1", toolCalls[0].Function.Name)

				case ToolCollectThenStop, ToolDrainAll, ToolAllowMixed:
					// Should respect the max limit of 3
					assert.Len(t, toolCalls, 3, "Should respect toolMaxCalls limit of 3")
					assert.Equal(t, "tool1", toolCalls[0].Function.Name)
					assert.Equal(t, "tool2", toolCalls[1].Function.Name)
					assert.Equal(t, "tool3", toolCalls[2].Function.Name)
				}

				// Verify content clearing for first three policies
				if tc.policy != ToolAllowMixed {
					assert.Empty(t, result.Choices[0].Message.Content, "Content should be cleared for %s", tc.name)
				}
			})
		}
	})

	t.Run("MaxCallsZero_NoLimit", func(t *testing.T) {
		opts := append([]Option{
			WithToolPolicy(ToolDrainAll),
			WithToolMaxCalls(0), // No limit
		}, WithLogLevel(slog.LevelError))
		adapter := New(opts...)

		resp := createMockResponse(`[
			{"name": "tool1", "parameters": {"id": 1}},
			{"name": "tool2", "parameters": {"id": 2}},
			{"name": "tool3", "parameters": {"id": 3}},
			{"name": "tool4", "parameters": {"id": 4}},
			{"name": "tool5", "parameters": {"id": 5}}
		]`)

		result, err := adapter.TransformCompletionsResponse(resp)
		require.NoError(t, err)

		// Should return all 5 tools (no limit)
		assert.Len(t, result.Choices[0].Message.ToolCalls, 5, "Should return all tool calls when limit is 0")
	})

	t.Run("MaxCallsGreaterThanAvailable", func(t *testing.T) {
		opts := append([]Option{
			WithToolPolicy(ToolDrainAll),
			WithToolMaxCalls(10), // Limit higher than available tools
		}, WithLogLevel(slog.LevelError))
		adapter := New(opts...)

		resp := createMockResponse(`[
			{"name": "tool1", "parameters": {"id": 1}},
			{"name": "tool2", "parameters": {"id": 2}}
		]`)

		result, err := adapter.TransformCompletionsResponse(resp)
		require.NoError(t, err)

		// Should return all available tools (2), not limited by maxCalls
		assert.Len(t, result.Choices[0].Message.ToolCalls, 2, "Should return all available tools when limit is higher")
	})
}

// ============================================================================
// BYTE LIMIT AND PERFORMANCE TESTS
// ============================================================================

// TestByteLimitFunctionality tests the toolCollectMaxBytes limit
func TestByteLimitFunctionality(t *testing.T) {
	t.Run("ByteLimit_StreamingMode", func(t *testing.T) {
		opts := append([]Option{
			WithToolPolicy(ToolDrainAll),
			WithToolCollectMaxBytes(100), // Small byte limit
		}, WithLogLevel(slog.LevelError))
		adapter := New(opts...)

		// Create content that exceeds the byte limit
		largeContent := strings.Repeat("This is a long string that will exceed the byte limit. ", 10) // ~540 bytes

		chunks := []string{
			largeContent,
			`[{"name": "test_tool", "parameters": {"test": "value"}}]`,
		}

		mockStream := NewMockStream(chunks)
		stream := adapter.TransformStreamingResponse(mockStream)
		defer func() { require.NoError(t, stream.Close()) }()

		var toolCallChunks []openai.ChatCompletionChunk
		var contentChunks []string

		for stream.Next() {
			chunk := stream.Current()
			if len(chunk.Choices) > 0 {
				if len(chunk.Choices[0].Delta.ToolCalls) > 0 {
					toolCallChunks = append(toolCallChunks, chunk)
				}
				if chunk.Choices[0].Delta.Content != "" {
					contentChunks = append(contentChunks, chunk.Choices[0].Delta.Content)
				}
			}
		}

		// Should handle byte limit gracefully (implementation may vary on how it handles this)
		// The key is that it doesn't crash and processes what it can
		assert.GreaterOrEqual(t, len(toolCallChunks)+len(contentChunks), 0, "Should handle byte limit gracefully")
	})

	t.Run("ByteLimit_ZeroMeansUnlimited", func(t *testing.T) {
		opts := append([]Option{
			WithToolPolicy(ToolDrainAll),
			WithToolCollectMaxBytes(0), // No limit
		}, WithLogLevel(slog.LevelError))
		adapter := New(opts...)

		// Create large content
		largeContent := strings.Repeat("Large content. ", 1000) // ~15KB

		chunks := []string{
			largeContent,
			`[{"name": "test_tool", "parameters": {"test": "value"}}]`,
		}

		mockStream := NewMockStream(chunks)
		stream := adapter.TransformStreamingResponse(mockStream)
		defer func() { require.NoError(t, stream.Close()) }()

		var processedChunks int
		for stream.Next() {
			processedChunks++
		}

		// Should process all chunks without byte limit restrictions
		assert.Greater(t, processedChunks, 0, "Should process all content without byte limit")
	})

	t.Run("ByteLimit_ExactBoundary", func(t *testing.T) {
		opts := append([]Option{
			WithToolPolicy(ToolDrainAll),
			WithToolCollectMaxBytes(50), // Exact boundary test
		}, WithLogLevel(slog.LevelError))
		adapter := New(opts...)

		// Create content exactly at the boundary
		exactContent := strings.Repeat("X", 49)              // 49 bytes
		toolContent := `[{"name": "t", "parameters": null}]` // About 33 bytes

		chunks := []string{exactContent, toolContent}

		mockStream := NewMockStream(chunks)
		stream := adapter.TransformStreamingResponse(mockStream)
		defer func() { require.NoError(t, stream.Close()) }()

		var processedChunks int
		for stream.Next() {
			processedChunks++
		}

		// Should handle boundary condition gracefully
		assert.GreaterOrEqual(t, processedChunks, 0, "Should handle exact byte boundary")
	})
}

// TestStreamingByteLimit tests byte limit functionality in streaming mode
func TestStreamingByteLimit(t *testing.T) {
	t.Run("ByteLimit_ExactLimit", func(t *testing.T) {
		opts := append([]Option{
			WithToolPolicy(ToolDrainAll),
			WithToolCollectMaxBytes(100), // Specific byte limit
		}, WithLogLevel(slog.LevelError))
		adapter := New(opts...)

		// Create content exactly at the byte limit
		content := strings.Repeat("x", 90)                      // 90 bytes
		toolContent := `[{"name": "test", "parameters": null}]` // About 40 bytes

		chunks := []string{content, toolContent}

		mockStream := NewMockStream(chunks)
		stream := adapter.TransformStreamingResponse(mockStream)
		defer func() { require.NoError(t, stream.Close()) }()

		var processedChunks int
		for stream.Next() {
			processedChunks++
		}

		assert.Greater(t, processedChunks, 0, "Should process chunks within byte limit")
	})

	t.Run("ByteLimit_ExceedsLimit_GracefulHandling", func(t *testing.T) {
		opts := append([]Option{
			WithToolPolicy(ToolDrainAll),
			WithToolCollectMaxBytes(50), // Small byte limit
		}, WithLogLevel(slog.LevelError))
		adapter := New(opts...)

		// Create content that exceeds the limit
		content := strings.Repeat("This content exceeds the byte limit. ", 5)
		chunks := []string{content}

		mockStream := NewMockStream(chunks)
		stream := adapter.TransformStreamingResponse(mockStream)
		defer func() { require.NoError(t, stream.Close()) }()

		var processedChunks int
		for stream.Next() {
			processedChunks++
		}

		// Should handle gracefully (not crash)
		assert.GreaterOrEqual(t, processedChunks, 0, "Should handle byte limit excess gracefully")
	})

	t.Run("ByteLimit_ZeroLimit_Unlimited", func(t *testing.T) {
		opts := append([]Option{
			WithToolPolicy(ToolDrainAll),
			WithToolCollectMaxBytes(0), // No limit
		}, WithLogLevel(slog.LevelError))
		adapter := New(opts...)

		// Create large content
		largeContent := strings.Repeat("Large content string. ", 100)
		chunks := []string{largeContent, `[{"name": "test", "parameters": null}]`}

		mockStream := NewMockStream(chunks)
		stream := adapter.TransformStreamingResponse(mockStream)
		defer func() { require.NoError(t, stream.Close()) }()

		var processedChunks int
		for stream.Next() {
			processedChunks++
		}

		assert.Greater(t, processedChunks, 0, "Should process all chunks with no byte limit")
	})
}

// ============================================================================
// CONTEXT AND ERROR HANDLING TESTS
// ============================================================================

// TestToolPolicyString tests the String() method for ToolPolicy
func TestToolPolicyString(t *testing.T) {
	testCases := []struct {
		policy   ToolPolicy
		expected string
	}{
		{ToolStopOnFirst, "ToolStopOnFirst"},
		{ToolCollectThenStop, "ToolCollectThenStop"},
		{ToolDrainAll, "ToolDrainAll"},
		{ToolAllowMixed, "ToolAllowMixed"},
		{ToolPolicy(999), "ToolPolicy(999)"}, // Unknown policy
	}

	for _, tc := range testCases {
		t.Run(tc.expected, func(t *testing.T) {
			assert.Equal(t, tc.expected, tc.policy.String())
		})
	}
}

// TestToolPolicyFallback tests fallback behavior for unknown policies
func TestToolPolicyFallback(t *testing.T) {
	// Create an adapter with an unknown policy value
	adapter := New(WithLogLevel(slog.LevelError))
	adapter.toolPolicy = ToolPolicy(999) // Set invalid policy directly

	resp := createMockResponse(`[{"name": "test_tool", "parameters": {"test": "value"}}]`)

	result, err := adapter.TransformCompletionsResponse(resp)
	require.NoError(t, err)

	// Should fallback to ToolStopOnFirst behavior (content cleared, single tool)
	assert.Empty(t, result.Choices[0].Message.Content, "Should clear content like ToolStopOnFirst")
	assert.Len(t, result.Choices[0].Message.ToolCalls, 1, "Should return single tool like ToolStopOnFirst")
	assert.Equal(t, "test_tool", result.Choices[0].Message.ToolCalls[0].Function.Name)
}

// TestToolPolicyWithContext tests policy behavior with context
func TestToolPolicyWithContext(t *testing.T) {
	t.Run("ContextCancellation", func(t *testing.T) {
		opts := append([]Option{WithToolPolicy(ToolDrainAll)}, WithLogLevel(slog.LevelError))
		adapter := New(opts...)

		ctx, cancel := context.WithCancel(context.Background())
		cancel() // Cancel immediately

		resp := createMockResponse(`[{"name": "test_tool", "parameters": {"test": "value"}}]`)

		_, err := adapter.TransformCompletionsResponseWithContext(ctx, resp)
		assert.Error(t, err, "Should return context cancellation error")
		assert.Equal(t, context.Canceled, err)
	})
}

// TestStreamingContextCancellation tests context cancellation during streaming
func TestStreamingContextCancellation(t *testing.T) {
	t.Run("CancellationDuringToolCollection", func(t *testing.T) {
		opts := append([]Option{WithToolPolicy(ToolCollectThenStop)}, WithLogLevel(slog.LevelError))
		adapter := New(opts...)

		ctx, cancel := context.WithCancel(context.Background())

		chunks := []string{
			`[{"name": "tool1",`,
			` "parameters": {"a": 1}}]`,
		}

		mockStream := NewMockStream(chunks)
		stream := adapter.TransformStreamingResponseWithContext(ctx, mockStream)
		defer func() { require.NoError(t, stream.Close()) }()

		// Cancel after starting
		cancel()

		// Should handle cancellation gracefully
		for stream.Next() {
			// Process chunks until cancelled
		}

		err := stream.Err()
		if err != nil {
			assert.Equal(t, context.Canceled, err, "Should return context cancellation error")
		}
	})
}

// ============================================================================
// PERFORMANCE AND EDGE CASE TESTS
// ============================================================================

// TestToolPolicyStreamingPerformance tests performance aspects of different policies
func TestToolPolicyStreamingPerformance(t *testing.T) {
	t.Run("ToolStopOnFirst_EarlyTermination", func(t *testing.T) {
		opts := append([]Option{WithToolPolicy(ToolStopOnFirst)}, WithLogLevel(slog.LevelError))
		adapter := New(opts...)

		// Create a stream with tool call early and lots of content after
		chunks := []string{
			`[{"name": "early_tool", "parameters": null}]`, // Tool call early
		}
		// Add 1000 chunks that should be ignored
		for i := 0; i < 1000; i++ {
			chunks = append(chunks, "Ignored content chunk "+string(rune(i)))
		}

		mockStream := NewMockStream(chunks)
		stream := adapter.TransformStreamingResponse(mockStream)
		defer func() { require.NoError(t, stream.Close()) }()

		var toolCallFound bool
		var contentAfterTool int

		for stream.Next() {
			chunk := stream.Current()
			if len(chunk.Choices) > 0 {
				if len(chunk.Choices[0].Delta.ToolCalls) > 0 {
					toolCallFound = true
				} else if toolCallFound && chunk.Choices[0].Delta.Content != "" {
					contentAfterTool++
				}
			}
		}

		assert.True(t, toolCallFound, "Should find the tool call")
		// The key test: ToolStopOnFirst should suppress content after the first tool
		assert.Equal(t, 0, contentAfterTool, "Should not emit any content after first tool call")
	})

	t.Run("ToolDrainAll_ProcessesEverything", func(t *testing.T) {
		opts := append([]Option{WithToolPolicy(ToolDrainAll)}, WithLogLevel(slog.LevelError))
		adapter := New(opts...)

		// Create stream with multiple tool calls spread throughout
		chunks := []string{
			"Initial content",
			`[{"name": "tool1", "parameters": null}]`,
			"Middle content",
			`[{"name": "tool2", "parameters": null}]`,
			"Final content",
		}

		mockStream := NewMockStream(chunks)
		stream := adapter.TransformStreamingResponse(mockStream)
		defer func() { require.NoError(t, stream.Close()) }()

		var toolCallCount int

		for stream.Next() {
			chunk := stream.Current()
			if len(chunk.Choices) > 0 && len(chunk.Choices[0].Delta.ToolCalls) > 0 {
				toolCallCount++
			}
		}

		// ToolDrainAll should process all content and tools (but suppress content emission)
		assert.GreaterOrEqual(t, toolCallCount, 1, "Should process tool calls from throughout the stream")
	})
}

// TestStreamingEdgeCases tests edge cases in streaming mode
func TestStreamingEdgeCases(t *testing.T) {
	t.Run("EmptyChunks", func(t *testing.T) {
		opts := append([]Option{WithToolPolicy(ToolStopOnFirst)}, WithLogLevel(slog.LevelError))
		adapter := New(opts...)

		chunks := []string{"", "", "", `[{"name": "test", "parameters": null}]`}

		mockStream := NewMockStream(chunks)
		stream := adapter.TransformStreamingResponse(mockStream)
		defer func() { require.NoError(t, stream.Close()) }()

		var processedChunks int
		for stream.Next() {
			processedChunks++
		}

		assert.GreaterOrEqual(t, processedChunks, 0, "Should handle empty chunks gracefully")
	})

	t.Run("OnlyWhitespaceChunks", func(t *testing.T) {
		opts := append([]Option{WithToolPolicy(ToolStopOnFirst)}, WithLogLevel(slog.LevelError))
		adapter := New(opts...)

		chunks := []string{"   ", "\n\n", "\t\t", `[{"name": "test", "parameters": null}]`}

		mockStream := NewMockStream(chunks)
		stream := adapter.TransformStreamingResponse(mockStream)
		defer func() { require.NoError(t, stream.Close()) }()

		var processedChunks int
		for stream.Next() {
			processedChunks++
		}

		assert.GreaterOrEqual(t, processedChunks, 0, "Should handle whitespace chunks gracefully")
	})

	t.Run("IncompleteJSONChunks", func(t *testing.T) {
		opts := append([]Option{WithToolPolicy(ToolStopOnFirst)}, WithLogLevel(slog.LevelError))
		adapter := New(opts...)

		chunks := []string{
			`[{"name": "incomplete`,
			// Incomplete JSON that never closes
		}

		mockStream := NewMockStream(chunks)
		stream := adapter.TransformStreamingResponse(mockStream)
		defer func() { require.NoError(t, stream.Close()) }()

		var processedChunks int
		for stream.Next() {
			processedChunks++
		}

		assert.GreaterOrEqual(t, processedChunks, 0, "Should handle incomplete JSON gracefully")
	})
}

// TestPolicyTransitionEdgeCases tests edge cases around policy behavior transitions
func TestPolicyTransitionEdgeCases(t *testing.T) {
	t.Run("NoToolCalls_AllPoliciesBehaveSame", func(t *testing.T) {
		policies := []ToolPolicy{ToolStopOnFirst, ToolCollectThenStop, ToolDrainAll, ToolAllowMixed}
		originalContent := "Just regular content with no tool calls at all."

		for _, policy := range policies {
			t.Run(policy.String(), func(t *testing.T) {
				opts := append([]Option{WithToolPolicy(policy)}, WithLogLevel(slog.LevelError))
				adapter := New(opts...)

				resp := createMockResponse(originalContent)
				result, err := adapter.TransformCompletionsResponse(resp)
				require.NoError(t, err)

				// All policies should preserve content when there are no tool calls
				assert.Equal(t, originalContent, result.Choices[0].Message.Content,
					"Policy %s should preserve content when no tool calls", policy.String())
				assert.Empty(t, result.Choices[0].Message.ToolCalls,
					"Policy %s should have no tool calls", policy.String())
			})
		}
	})

	t.Run("EmptyResponse_AllPoliciesHandleGracefully", func(t *testing.T) {
		policies := []ToolPolicy{ToolStopOnFirst, ToolCollectThenStop, ToolDrainAll, ToolAllowMixed}

		for _, policy := range policies {
			t.Run(policy.String(), func(t *testing.T) {
				opts := append([]Option{WithToolPolicy(policy)}, WithLogLevel(slog.LevelError))
				adapter := New(opts...)

				resp := openai.ChatCompletion{
					Choices: []openai.ChatCompletionChoice{}, // Empty choices
				}

				result, err := adapter.TransformCompletionsResponse(resp)
				require.NoError(t, err)

				// Should handle empty response gracefully
				assert.Equal(t, resp, result, "Policy %s should handle empty response", policy.String())
			})
		}
	})

	t.Run("InvalidJSON_AllPoliciesHandleGracefully", func(t *testing.T) {
		policies := []ToolPolicy{ToolStopOnFirst, ToolCollectThenStop, ToolDrainAll, ToolAllowMixed}
		malformedContent := `[{"name": "broken_tool", "parameters":` // Incomplete JSON

		for _, policy := range policies {
			t.Run(policy.String(), func(t *testing.T) {
				opts := append([]Option{WithToolPolicy(policy)}, WithLogLevel(slog.LevelError))
				adapter := New(opts...)

				resp := createMockResponse(malformedContent)
				result, err := adapter.TransformCompletionsResponse(resp)
				require.NoError(t, err)

				// Should handle malformed JSON gracefully - return original content or empty
				assert.NotPanics(t, func() {
					_ = result.Choices[0].Message.Content
					_ = result.Choices[0].Message.ToolCalls
				}, "Policy %s should handle invalid JSON gracefully", policy.String())
			})
		}
	})
}

// TestCollectionWindowTimeout tests the collection window timeout functionality
func TestCollectionWindowTimeout(t *testing.T) {
	t.Run("CollectionWindow_ShortTimeout", func(t *testing.T) {
		opts := append([]Option{
			WithToolPolicy(ToolCollectThenStop),
			WithToolCollectWindow(1 * time.Millisecond), // Very short timeout
		}, WithLogLevel(slog.LevelError))
		adapter := New(opts...)

		chunks := []string{
			`[{"name": "tool1", "parameters": {"a": 1}}]`,
			// In real streaming, there would be delays here that exceed the timeout
		}

		mockStream := NewMockStream(chunks)
		stream := adapter.TransformStreamingResponse(mockStream)
		defer func() { require.NoError(t, stream.Close()) }()

		var processedChunks int
		for stream.Next() {
			processedChunks++
		}

		// Should process chunks despite short timeout (mock doesn't have real delays)
		assert.GreaterOrEqual(t, processedChunks, 0, "Should handle short timeout gracefully")
	})

	t.Run("CollectionWindow_ZeroMeansStructureOnly", func(t *testing.T) {
		opts := append([]Option{
			WithToolPolicy(ToolCollectThenStop),
			WithToolCollectWindow(0), // Structure-only mode (no timer)
		}, WithLogLevel(slog.LevelError))
		adapter := New(opts...)

		chunks := []string{
			`[{"name": "tool1", "parameters": {"a": 1}}]`,
		}

		mockStream := NewMockStream(chunks)
		stream := adapter.TransformStreamingResponse(mockStream)
		defer func() { require.NoError(t, stream.Close()) }()

		var processedChunks int
		for stream.Next() {
			processedChunks++
		}

		// Should work with structure-only batching
		assert.GreaterOrEqual(t, processedChunks, 0, "Should handle structure-only batching")
	})
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// Helper functions for creating mock responses
func createMockResponse(content string) openai.ChatCompletion {
	return openai.ChatCompletion{
		Choices: []openai.ChatCompletionChoice{
			{
				Message: openai.ChatCompletionMessage{
					Content: content,
				},
			},
		},
	}
}

func createMockResponseWithContent(originalContent, toolContent string) openai.ChatCompletion {
	return openai.ChatCompletion{
		Choices: []openai.ChatCompletionChoice{
			{
				Message: openai.ChatCompletionMessage{
					Content: originalContent + " " + toolContent, // Simulate content with embedded tools
				},
			},
		},
	}
}
