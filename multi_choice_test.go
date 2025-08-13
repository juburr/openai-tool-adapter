package tooladapter

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"testing"
	"time"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestMultiChoiceTransformation tests that the adapter correctly processes
// responses with multiple choices (n > 1)
func TestMultiChoiceTransformation(t *testing.T) {
	t.Run("SingleChoice_BackwardCompatibility", func(t *testing.T) {
		// Ensure single-choice responses work exactly as before
		adapter := New(WithLogLevel(slog.LevelError))

		response := openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Content: `I'll help you. [{"name": "get_weather", "parameters": {"location": "NYC"}}]`,
					},
				},
			},
		}

		result, err := adapter.TransformCompletionsResponse(response)
		require.NoError(t, err)
		require.Len(t, result.Choices, 1)
		assert.Len(t, result.Choices[0].Message.ToolCalls, 1)
		assert.Equal(t, "get_weather", result.Choices[0].Message.ToolCalls[0].Function.Name)
	})

	t.Run("MultiChoice_ToolsInFirstChoice", func(t *testing.T) {
		// n=3, tools in choices[0]
		adapter := New(WithLogLevel(slog.LevelError))

		response := openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Content: `[{"name": "tool1", "parameters": {"x": 1}}]`,
					},
				},
				{
					Message: openai.ChatCompletionMessage{
						Content: `Just regular text in second choice`,
					},
				},
				{
					Message: openai.ChatCompletionMessage{
						Content: `More regular text in third choice`,
					},
				},
			},
		}

		result, err := adapter.TransformCompletionsResponse(response)
		require.NoError(t, err)
		require.Len(t, result.Choices, 3)

		// First choice should have tool calls
		assert.Len(t, result.Choices[0].Message.ToolCalls, 1)
		assert.Equal(t, "tool1", result.Choices[0].Message.ToolCalls[0].Function.Name)

		// Other choices should remain unchanged
		assert.Empty(t, result.Choices[1].Message.ToolCalls)
		assert.Equal(t, "Just regular text in second choice", result.Choices[1].Message.Content)
		assert.Empty(t, result.Choices[2].Message.ToolCalls)
		assert.Equal(t, "More regular text in third choice", result.Choices[2].Message.Content)
	})

	t.Run("MultiChoice_ToolsInMiddleChoice", func(t *testing.T) {
		// n=3, tools in choices[1]
		adapter := New(WithLogLevel(slog.LevelError))

		response := openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Content: `Just regular text in first choice`,
					},
				},
				{
					Message: openai.ChatCompletionMessage{
						Content: `I'll call a function: {"name": "calculate", "parameters": {"expression": "2+2"}}`,
					},
				},
				{
					Message: openai.ChatCompletionMessage{
						Content: `Regular text in third choice`,
					},
				},
			},
		}

		result, err := adapter.TransformCompletionsResponse(response)
		require.NoError(t, err)
		require.Len(t, result.Choices, 3)

		// First choice should remain unchanged
		assert.Empty(t, result.Choices[0].Message.ToolCalls)
		assert.Equal(t, "Just regular text in first choice", result.Choices[0].Message.Content)

		// Second choice should have tool calls
		assert.Len(t, result.Choices[1].Message.ToolCalls, 1)
		assert.Equal(t, "calculate", result.Choices[1].Message.ToolCalls[0].Function.Name)

		// Third choice should remain unchanged
		assert.Empty(t, result.Choices[2].Message.ToolCalls)
		assert.Equal(t, "Regular text in third choice", result.Choices[2].Message.Content)
	})

	t.Run("MultiChoice_ToolsInMultipleChoices", func(t *testing.T) {
		// n=3, tools in choices[0] and choices[2]
		adapter := New(WithLogLevel(slog.LevelError))

		response := openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Content: `[{"name": "weather", "parameters": {"location": "NYC"}}]`,
					},
				},
				{
					Message: openai.ChatCompletionMessage{
						Content: `No tools here, just text`,
					},
				},
				{
					Message: openai.ChatCompletionMessage{
						Content: `Another tool call: [{"name": "search", "parameters": {"query": "news"}}]`,
					},
				},
			},
		}

		result, err := adapter.TransformCompletionsResponse(response)
		require.NoError(t, err)
		require.Len(t, result.Choices, 3)

		// First choice should have tool calls
		assert.Len(t, result.Choices[0].Message.ToolCalls, 1)
		assert.Equal(t, "weather", result.Choices[0].Message.ToolCalls[0].Function.Name)

		// Second choice should remain unchanged
		assert.Empty(t, result.Choices[1].Message.ToolCalls)
		assert.Equal(t, "No tools here, just text", result.Choices[1].Message.Content)

		// Third choice should have tool calls
		assert.Len(t, result.Choices[2].Message.ToolCalls, 1)
		assert.Equal(t, "search", result.Choices[2].Message.ToolCalls[0].Function.Name)
	})

	t.Run("MultiChoice_NoToolsAnywhere", func(t *testing.T) {
		// n=3, no tools in any choice
		adapter := New(WithLogLevel(slog.LevelError))

		response := openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Content: `Just regular conversation text`,
					},
				},
				{
					Message: openai.ChatCompletionMessage{
						Content: `More regular text`,
					},
				},
				{
					Message: openai.ChatCompletionMessage{
						Content: `Even more regular text`,
					},
				},
			},
		}

		result, err := adapter.TransformCompletionsResponse(response)
		require.NoError(t, err)
		require.Len(t, result.Choices, 3)

		// All choices should remain unchanged
		for i, choice := range result.Choices {
			assert.Empty(t, choice.Message.ToolCalls, "Choice %d should have no tool calls", i)
			assert.NotEmpty(t, choice.Message.Content, "Choice %d should retain content", i)
		}
	})

	t.Run("MultiChoice_EmptyContentChoices", func(t *testing.T) {
		// Some choices have empty content
		adapter := New(WithLogLevel(slog.LevelError))

		response := openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Content: `[{"name": "tool1", "parameters": null}]`,
					},
				},
				{
					Message: openai.ChatCompletionMessage{
						Content: ``, // Empty content
					},
				},
				{
					Message: openai.ChatCompletionMessage{
						Content: `Some text`,
					},
				},
			},
		}

		result, err := adapter.TransformCompletionsResponse(response)
		require.NoError(t, err)
		require.Len(t, result.Choices, 3)

		// First choice should have tool calls
		assert.Len(t, result.Choices[0].Message.ToolCalls, 1)

		// Second choice should remain unchanged (empty)
		assert.Empty(t, result.Choices[1].Message.ToolCalls)
		assert.Empty(t, result.Choices[1].Message.Content)

		// Third choice should remain unchanged
		assert.Empty(t, result.Choices[2].Message.ToolCalls)
		assert.Equal(t, "Some text", result.Choices[2].Message.Content)
	})

	t.Run("MultiChoice_MultipleToolsPerChoice", func(t *testing.T) {
		// Multiple tool calls within a single choice
		// Default policy is ToolStopOnFirst, so only first tool should be kept
		adapter := New(WithLogLevel(slog.LevelError))

		response := openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Content: `[
							{"name": "weather", "parameters": {"location": "NYC"}},
							{"name": "search", "parameters": {"query": "news"}}
						]`,
					},
				},
				{
					Message: openai.ChatCompletionMessage{
						Content: `[{"name": "calculate", "parameters": {"expression": "2+2"}}]`,
					},
				},
			},
		}

		result, err := adapter.TransformCompletionsResponse(response)
		require.NoError(t, err)
		require.Len(t, result.Choices, 2)

		// First choice should have only 1 tool call (ToolStopOnFirst policy)
		assert.Len(t, result.Choices[0].Message.ToolCalls, 1)
		assert.Equal(t, "weather", result.Choices[0].Message.ToolCalls[0].Function.Name)

		// Second choice should have 1 tool call
		assert.Len(t, result.Choices[1].Message.ToolCalls, 1)
		assert.Equal(t, "calculate", result.Choices[1].Message.ToolCalls[0].Function.Name)
	})

	t.Run("MultiChoice_MultipleToolsPerChoice_AllowMixed", func(t *testing.T) {
		// Multiple tool calls within a single choice with ToolAllowMixed policy
		adapter := New(
			WithLogLevel(slog.LevelError),
			WithToolPolicy(ToolAllowMixed), // This allows all tools to be preserved
		)

		response := openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Content: `[
							{"name": "weather", "parameters": {"location": "NYC"}},
							{"name": "search", "parameters": {"query": "news"}}
						]`,
					},
				},
				{
					Message: openai.ChatCompletionMessage{
						Content: `[{"name": "calculate", "parameters": {"expression": "2+2"}}]`,
					},
				},
			},
		}

		result, err := adapter.TransformCompletionsResponse(response)
		require.NoError(t, err)
		require.Len(t, result.Choices, 2)

		// First choice should have 2 tool calls (ToolAllowMixed preserves all)
		assert.Len(t, result.Choices[0].Message.ToolCalls, 2)
		assert.Equal(t, "weather", result.Choices[0].Message.ToolCalls[0].Function.Name)
		assert.Equal(t, "search", result.Choices[0].Message.ToolCalls[1].Function.Name)

		// Second choice should have 1 tool call
		assert.Len(t, result.Choices[1].Message.ToolCalls, 1)
		assert.Equal(t, "calculate", result.Choices[1].Message.ToolCalls[0].Function.Name)
	})
}

// TestMultiChoiceWithToolPolicies tests that tool policies are correctly applied
// to each choice independently
func TestMultiChoiceWithToolPolicies(t *testing.T) {
	createMultiToolResponse := func() openai.ChatCompletion {
		return openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Content: `Here are the tools: [
							{"name": "tool1", "parameters": {"x": 1}},
							{"name": "tool2", "parameters": {"x": 2}},
							{"name": "tool3", "parameters": {"x": 3}}
						]`,
					},
				},
				{
					Message: openai.ChatCompletionMessage{
						Content: `Alternative: [
							{"name": "toolA", "parameters": {"y": 1}},
							{"name": "toolB", "parameters": {"y": 2}}
						]`,
					},
				},
				{
					Message: openai.ChatCompletionMessage{
						Content: `No tools, just text`,
					},
				},
			},
		}
	}

	t.Run("ToolStopOnFirst_PerChoice", func(t *testing.T) {
		// Each choice should only have the first tool call
		adapter := New(
			WithLogLevel(slog.LevelError),
			WithToolPolicy(ToolStopOnFirst),
		)

		response := createMultiToolResponse()
		result, err := adapter.TransformCompletionsResponse(response)
		require.NoError(t, err)
		require.Len(t, result.Choices, 3)

		// First choice: only first tool (tool1)
		assert.Len(t, result.Choices[0].Message.ToolCalls, 1)
		assert.Equal(t, "tool1", result.Choices[0].Message.ToolCalls[0].Function.Name)
		assert.Empty(t, result.Choices[0].Message.Content, "Content should be cleared")

		// Second choice: only first tool (toolA)
		assert.Len(t, result.Choices[1].Message.ToolCalls, 1)
		assert.Equal(t, "toolA", result.Choices[1].Message.ToolCalls[0].Function.Name)
		assert.Empty(t, result.Choices[1].Message.Content, "Content should be cleared")

		// Third choice: no tools, content preserved
		assert.Empty(t, result.Choices[2].Message.ToolCalls)
		assert.Equal(t, "No tools, just text", result.Choices[2].Message.Content)
	})

	t.Run("ToolCollectThenStop_PerChoice", func(t *testing.T) {
		// Each choice should collect up to the limit
		adapter := New(
			WithLogLevel(slog.LevelError),
			WithToolPolicy(ToolCollectThenStop),
			WithToolMaxCalls(2), // Limit to 2 per choice
		)

		response := createMultiToolResponse()
		result, err := adapter.TransformCompletionsResponse(response)
		require.NoError(t, err)
		require.Len(t, result.Choices, 3)

		// First choice: first 2 tools (tool1, tool2)
		assert.Len(t, result.Choices[0].Message.ToolCalls, 2)
		assert.Equal(t, "tool1", result.Choices[0].Message.ToolCalls[0].Function.Name)
		assert.Equal(t, "tool2", result.Choices[0].Message.ToolCalls[1].Function.Name)
		assert.Empty(t, result.Choices[0].Message.Content, "Content should be cleared")

		// Second choice: both tools (toolA, toolB)
		assert.Len(t, result.Choices[1].Message.ToolCalls, 2)
		assert.Equal(t, "toolA", result.Choices[1].Message.ToolCalls[0].Function.Name)
		assert.Equal(t, "toolB", result.Choices[1].Message.ToolCalls[1].Function.Name)
		assert.Empty(t, result.Choices[1].Message.Content, "Content should be cleared")

		// Third choice: no tools, content preserved
		assert.Empty(t, result.Choices[2].Message.ToolCalls)
		assert.Equal(t, "No tools, just text", result.Choices[2].Message.Content)
	})

	t.Run("ToolDrainAll_PerChoice", func(t *testing.T) {
		// Each choice should have all tools
		adapter := New(
			WithLogLevel(slog.LevelError),
			WithToolPolicy(ToolDrainAll),
		)

		response := createMultiToolResponse()
		result, err := adapter.TransformCompletionsResponse(response)
		require.NoError(t, err)
		require.Len(t, result.Choices, 3)

		// First choice: all 3 tools
		assert.Len(t, result.Choices[0].Message.ToolCalls, 3)
		assert.Equal(t, "tool1", result.Choices[0].Message.ToolCalls[0].Function.Name)
		assert.Equal(t, "tool2", result.Choices[0].Message.ToolCalls[1].Function.Name)
		assert.Equal(t, "tool3", result.Choices[0].Message.ToolCalls[2].Function.Name)
		assert.Empty(t, result.Choices[0].Message.Content, "Content should be cleared")

		// Second choice: both tools
		assert.Len(t, result.Choices[1].Message.ToolCalls, 2)
		assert.Equal(t, "toolA", result.Choices[1].Message.ToolCalls[0].Function.Name)
		assert.Equal(t, "toolB", result.Choices[1].Message.ToolCalls[1].Function.Name)
		assert.Empty(t, result.Choices[1].Message.Content, "Content should be cleared")

		// Third choice: no tools, content preserved
		assert.Empty(t, result.Choices[2].Message.ToolCalls)
		assert.Equal(t, "No tools, just text", result.Choices[2].Message.Content)
	})

	t.Run("ToolAllowMixed_PerChoice", func(t *testing.T) {
		// Each choice should have tools AND content preserved
		adapter := New(
			WithLogLevel(slog.LevelError),
			WithToolPolicy(ToolAllowMixed),
		)

		response := createMultiToolResponse()
		result, err := adapter.TransformCompletionsResponse(response)
		require.NoError(t, err)
		require.Len(t, result.Choices, 3)

		// First choice: all tools with content preserved
		assert.Len(t, result.Choices[0].Message.ToolCalls, 3)
		assert.Contains(t, result.Choices[0].Message.Content, "Here are the tools")

		// Second choice: both tools with content preserved
		assert.Len(t, result.Choices[1].Message.ToolCalls, 2)
		assert.Contains(t, result.Choices[1].Message.Content, "Alternative")

		// Third choice: no tools, content preserved
		assert.Empty(t, result.Choices[2].Message.ToolCalls)
		assert.Equal(t, "No tools, just text", result.Choices[2].Message.Content)
	})

	t.Run("ToolMaxCalls_AppliedPerChoice", func(t *testing.T) {
		// Verify that tool max calls is applied per choice, not globally
		adapter := New(
			WithLogLevel(slog.LevelError),
			WithToolPolicy(ToolDrainAll),
			WithToolMaxCalls(1), // Limit to 1 per choice
		)

		response := createMultiToolResponse()
		result, err := adapter.TransformCompletionsResponse(response)
		require.NoError(t, err)
		require.Len(t, result.Choices, 3)

		// First choice: only 1 tool (due to limit)
		assert.Len(t, result.Choices[0].Message.ToolCalls, 1)
		assert.Equal(t, "tool1", result.Choices[0].Message.ToolCalls[0].Function.Name)

		// Second choice: only 1 tool (due to limit)
		assert.Len(t, result.Choices[1].Message.ToolCalls, 1)
		assert.Equal(t, "toolA", result.Choices[1].Message.ToolCalls[0].Function.Name)

		// Third choice: no tools
		assert.Empty(t, result.Choices[2].Message.ToolCalls)
	})
}

// TestMultiChoiceEdgeCases tests edge cases for multi-choice processing
func TestMultiChoiceEdgeCases(t *testing.T) {
	t.Run("EmptyChoicesArray", func(t *testing.T) {
		adapter := New(WithLogLevel(slog.LevelError))

		response := openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{},
		}

		result, err := adapter.TransformCompletionsResponse(response)
		require.NoError(t, err)
		assert.Empty(t, result.Choices)
	})

	t.Run("LargeNumberOfChoices", func(t *testing.T) {
		// Test with n=10 choices
		adapter := New(WithLogLevel(slog.LevelError))

		choices := make([]openai.ChatCompletionChoice, 10)
		for i := range choices {
			if i%2 == 0 {
				// Even indices get tool calls
				choices[i] = openai.ChatCompletionChoice{
					Message: openai.ChatCompletionMessage{
						Content: `[{"name": "tool` + string(rune('A'+i)) + `", "parameters": {}}]`,
					},
				}
			} else {
				// Odd indices get regular text
				choices[i] = openai.ChatCompletionChoice{
					Message: openai.ChatCompletionMessage{
						Content: `Text for choice ` + string(rune('0'+i)),
					},
				}
			}
		}

		response := openai.ChatCompletion{Choices: choices}
		result, err := adapter.TransformCompletionsResponse(response)
		require.NoError(t, err)
		require.Len(t, result.Choices, 10)

		// Verify even indices have tool calls
		for i := 0; i < 10; i++ {
			if i%2 == 0 {
				assert.NotEmpty(t, result.Choices[i].Message.ToolCalls, "Choice %d should have tool calls", i)
			} else {
				assert.Empty(t, result.Choices[i].Message.ToolCalls, "Choice %d should not have tool calls", i)
				assert.NotEmpty(t, result.Choices[i].Message.Content, "Choice %d should have content", i)
			}
		}
	})

	t.Run("MalformedJSONInSomeChoices", func(t *testing.T) {
		adapter := New(WithLogLevel(slog.LevelError))

		response := openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Content: `[{"name": "valid_tool", "parameters": {}}]`,
					},
				},
				{
					Message: openai.ChatCompletionMessage{
						Content: `[{"name": "broken_json" // missing closing`,
					},
				},
				{
					Message: openai.ChatCompletionMessage{
						Content: `[{"name": "another_valid", "parameters": null}]`,
					},
				},
			},
		}

		result, err := adapter.TransformCompletionsResponse(response)
		require.NoError(t, err)
		require.Len(t, result.Choices, 3)

		// First choice should have valid tool call
		assert.Len(t, result.Choices[0].Message.ToolCalls, 1)
		assert.Equal(t, "valid_tool", result.Choices[0].Message.ToolCalls[0].Function.Name)

		// Second choice should remain unchanged (malformed JSON)
		assert.Empty(t, result.Choices[1].Message.ToolCalls)
		assert.Contains(t, result.Choices[1].Message.Content, "broken_json")

		// Third choice should have valid tool call
		assert.Len(t, result.Choices[2].Message.ToolCalls, 1)
		assert.Equal(t, "another_valid", result.Choices[2].Message.ToolCalls[0].Function.Name)
	})

	t.Run("ContextCancellation", func(t *testing.T) {
		// Test that context cancellation works with multi-choice
		adapter := New(WithLogLevel(slog.LevelError))

		ctx, cancel := context.WithCancel(context.Background())
		cancel() // Cancel immediately

		response := openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Content: `[{"name": "tool1", "parameters": {}}]`,
					},
				},
			},
		}

		_, err := adapter.TransformCompletionsResponseWithContext(ctx, response)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "context canceled")
	})

	t.Run("UniqueToolIDsAcrossChoices", func(t *testing.T) {
		// Verify that tool IDs are unique across all choices
		// Use ToolDrainAll to get all tools from each choice
		adapter := New(
			WithLogLevel(slog.LevelError),
			WithToolPolicy(ToolDrainAll), // Get all tools, not just first
		)

		response := openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Content: `[{"name": "tool1", "parameters": {}}, {"name": "tool2", "parameters": {}}]`,
					},
				},
				{
					Message: openai.ChatCompletionMessage{
						Content: `[{"name": "tool3", "parameters": {}}, {"name": "tool4", "parameters": {}}]`,
					},
				},
			},
		}

		result, err := adapter.TransformCompletionsResponse(response)
		require.NoError(t, err)

		// Collect all tool IDs
		seenIDs := make(map[string]bool)
		totalTools := 0
		for _, choice := range result.Choices {
			for _, toolCall := range choice.Message.ToolCalls {
				assert.NotEmpty(t, toolCall.ID, "Tool call should have an ID")
				assert.False(t, seenIDs[toolCall.ID], "Tool ID %s should be unique", toolCall.ID)
				seenIDs[toolCall.ID] = true
				totalTools++
			}
		}

		// Should have 4 unique IDs
		assert.Equal(t, 4, totalTools, "Should have 4 total tools")
		assert.Len(t, seenIDs, 4, "Should have 4 unique IDs")
	})
}

// TestMultiChoiceIntegration tests the complete flow with multi-choice responses
func TestMultiChoiceIntegration(t *testing.T) {
	t.Run("CallerSelectionStrategies", func(t *testing.T) {
		adapter := New(WithLogLevel(slog.LevelError))

		response := openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Content: `Regular text response`,
					},
				},
				{
					Message: openai.ChatCompletionMessage{
						Content: `[{"name": "weather", "parameters": {"location": "NYC"}}]`,
					},
				},
				{
					Message: openai.ChatCompletionMessage{
						Content: `Another regular response`,
					},
				},
			},
		}

		result, err := adapter.TransformCompletionsResponse(response)
		require.NoError(t, err)
		require.Len(t, result.Choices, 3)

		// Strategy 1: Caller uses first choice (backward compatible)
		firstChoice := result.Choices[0]
		assert.Empty(t, firstChoice.Message.ToolCalls)
		assert.Equal(t, "Regular text response", firstChoice.Message.Content)

		// Strategy 2: Caller finds first choice with tool calls
		var choiceWithTools *openai.ChatCompletionChoice
		for i := range result.Choices {
			if len(result.Choices[i].Message.ToolCalls) > 0 {
				choiceWithTools = &result.Choices[i]
				break
			}
		}
		require.NotNil(t, choiceWithTools, "Should find a choice with tool calls")
		assert.Equal(t, "weather", choiceWithTools.Message.ToolCalls[0].Function.Name)

		// Strategy 3: Caller uses all choices
		toolCount := 0
		textCount := 0
		for _, choice := range result.Choices {
			if len(choice.Message.ToolCalls) > 0 {
				toolCount++
			} else if choice.Message.Content != "" {
				textCount++
			}
		}
		assert.Equal(t, 1, toolCount, "Should have 1 choice with tools")
		assert.Equal(t, 2, textCount, "Should have 2 choices with text")
	})

	t.Run("RealWorldScenario", func(t *testing.T) {
		// Simulate a real scenario where user requests n=3 for diversity
		adapter := New(
			WithLogLevel(slog.LevelError),
			WithToolPolicy(ToolStopOnFirst), // Common production setting
		)

		// Model returns 3 different approaches to the same request
		response := openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Content: `Let me check the weather for you. [{"name": "get_weather", "parameters": {"location": "San Francisco", "unit": "fahrenheit"}}]`,
					},
				},
				{
					Message: openai.ChatCompletionMessage{
						Content: `I'll get that information. [{"name": "get_weather", "parameters": {"location": "San Francisco", "unit": "celsius"}}]`,
					},
				},
				{
					Message: openai.ChatCompletionMessage{
						Content: `Based on typical weather patterns, San Francisco is usually mild with temperatures around 60-70°F year-round.`,
					},
				},
			},
		}

		result, err := adapter.TransformCompletionsResponse(response)
		require.NoError(t, err)
		require.Len(t, result.Choices, 3)

		// Choices 0 and 1 should have tool calls (content cleared due to ToolStopOnFirst)
		assert.Len(t, result.Choices[0].Message.ToolCalls, 1)
		assert.Empty(t, result.Choices[0].Message.Content)
		assert.Equal(t, "fahrenheit", extractParam(t, result.Choices[0].Message.ToolCalls[0].Function.Arguments, "unit"))

		assert.Len(t, result.Choices[1].Message.ToolCalls, 1)
		assert.Empty(t, result.Choices[1].Message.Content)
		assert.Equal(t, "celsius", extractParam(t, result.Choices[1].Message.ToolCalls[0].Function.Arguments, "unit"))

		// Choice 2 should have no tool calls (regular response)
		assert.Empty(t, result.Choices[2].Message.ToolCalls)
		assert.Contains(t, result.Choices[2].Message.Content, "60-70°F")
	})
}

// Helper function to extract a parameter from tool call arguments
func extractParam(t *testing.T, arguments string, key string) string {
	var params map[string]interface{}
	err := json.Unmarshal([]byte(arguments), &params)
	require.NoError(t, err)
	value, ok := params[key].(string)
	require.True(t, ok, "Parameter %s should exist and be a string", key)
	return value
}

// verifyPolicyApplication is a helper function to verify that a tool policy
// was correctly applied to a choice, reducing cyclomatic complexity
func verifyPolicyApplication(t *testing.T, policy ToolPolicy, choice openai.ChatCompletionChoice, choiceIndex int) {
	switch policy {
	case ToolStopOnFirst:
		assert.Len(t, choice.Message.ToolCalls, 1, "Choice %d should have 1 tool", choiceIndex)
		assert.Empty(t, choice.Message.Content, "Choice %d content should be cleared", choiceIndex)
	case ToolCollectThenStop:
		assert.LessOrEqual(t, len(choice.Message.ToolCalls), 2, "Choice %d should respect limit", choiceIndex)
		assert.Empty(t, choice.Message.Content, "Choice %d content should be cleared", choiceIndex)
	case ToolDrainAll:
		// DrainAll respects the global limit
		assert.LessOrEqual(t, len(choice.Message.ToolCalls), 2, "Choice %d should respect limit", choiceIndex)
		assert.Empty(t, choice.Message.Content, "Choice %d content should be cleared", choiceIndex)
	case ToolAllowMixed:
		assert.LessOrEqual(t, len(choice.Message.ToolCalls), 2, "Choice %d should respect limit", choiceIndex)
		assert.NotEmpty(t, choice.Message.Content, "Choice %d content should be preserved", choiceIndex)
	}
}

// TestMultiChoiceEdgeCasesExtended tests additional edge cases for robustness
func TestMultiChoiceEdgeCasesExtended(t *testing.T) {
	t.Run("NilParametersInToolCalls", func(t *testing.T) {
		adapter := New(WithLogLevel(slog.LevelError))

		response := openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Content: `[{"name": "tool1", "parameters": null}, {"name": "tool2"}]`,
					},
				},
				{
					Message: openai.ChatCompletionMessage{
						Content: `{"name": "tool3", "parameters": null}`,
					},
				},
			},
		}

		result, err := adapter.TransformCompletionsResponse(response)
		require.NoError(t, err)
		require.Len(t, result.Choices, 2)

		// Check first choice has tools with null parameters handled correctly
		if len(result.Choices[0].Message.ToolCalls) > 0 {
			for _, toolCall := range result.Choices[0].Message.ToolCalls {
				assert.NotEmpty(t, toolCall.Function.Arguments, "Arguments should not be empty")
				// Verify it's valid JSON (either null or an object)
				var args interface{}
				err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args)
				assert.NoError(t, err, "Arguments should be valid JSON")
			}
		}
	})

	t.Run("VeryLongFunctionNames", func(t *testing.T) {
		adapter := New(WithLogLevel(slog.LevelError))

		// Create a function name at the 64-character limit
		longName := "a_very_long_function_name_that_is_exactly_sixty_four_charactersX"[:64]
		response := openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Content: `[{"name": "` + longName + `", "parameters": {}}]`,
					},
				},
			},
		}

		result, err := adapter.TransformCompletionsResponse(response)
		require.NoError(t, err)

		if len(result.Choices[0].Message.ToolCalls) > 0 {
			assert.Equal(t, longName, result.Choices[0].Message.ToolCalls[0].Function.Name)
		}
	})

	t.Run("InvalidJSONRecovery", func(t *testing.T) {
		adapter := New(WithLogLevel(slog.LevelError))

		response := openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Content: `[{"name": "valid_tool", "parameters": {}}]`,
					},
				},
				{
					Message: openai.ChatCompletionMessage{
						Content: `[{"name": "broken_json", "parameters": {`, // Incomplete JSON
					},
				},
				{
					Message: openai.ChatCompletionMessage{
						Content: `Not JSON at all, just text`,
					},
				},
			},
		}

		result, err := adapter.TransformCompletionsResponse(response)
		require.NoError(t, err)
		require.Len(t, result.Choices, 3)

		// First choice should have valid tool
		assert.NotEmpty(t, result.Choices[0].Message.ToolCalls)

		// Second and third choices should remain unchanged
		assert.Empty(t, result.Choices[1].Message.ToolCalls)
		assert.Empty(t, result.Choices[2].Message.ToolCalls)
	})

	t.Run("ConcurrentMultiChoiceProcessing", func(t *testing.T) {
		// Test thread safety with concurrent processing
		adapter := New(WithLogLevel(slog.LevelError))

		response := openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Content: `[{"name": "tool1", "parameters": {"x": 1}}]`,
					},
				},
				{
					Message: openai.ChatCompletionMessage{
						Content: `[{"name": "tool2", "parameters": {"x": 2}}]`,
					},
				},
			},
		}

		// Run concurrent transformations
		done := make(chan bool, 10)
		for i := 0; i < 10; i++ {
			go func() {
				_, err := adapter.TransformCompletionsResponse(response)
				assert.NoError(t, err)
				done <- true
			}()
		}

		// Wait for all goroutines
		for i := 0; i < 10; i++ {
			<-done
		}
	})

	t.Run("ChoiceIndexBoundaries", func(t *testing.T) {
		adapter := New(WithLogLevel(slog.LevelError))

		// Test with maximum reasonable number of choices
		choices := make([]openai.ChatCompletionChoice, 100)
		for i := range choices {
			if i%10 == 0 {
				// Every 10th choice has a tool call
				choices[i] = openai.ChatCompletionChoice{
					Message: openai.ChatCompletionMessage{
						Content: `[{"name": "tool_` + fmt.Sprintf("%d", i) + `", "parameters": {}}]`,
					},
				}
			} else {
				choices[i] = openai.ChatCompletionChoice{
					Message: openai.ChatCompletionMessage{
						Content: `Text for choice ` + fmt.Sprintf("%d", i),
					},
				}
			}
		}

		response := openai.ChatCompletion{Choices: choices}
		result, err := adapter.TransformCompletionsResponse(response)
		require.NoError(t, err)
		assert.Len(t, result.Choices, 100)

		// Verify each choice was processed correctly
		toolCount := 0
		for i, choice := range result.Choices {
			if i%10 == 0 && len(choice.Message.ToolCalls) > 0 {
				toolCount++
			}
		}
		assert.Equal(t, 10, toolCount, "Should have tools in every 10th choice")
	})

	t.Run("PolicyConsistencyAcrossChoices", func(t *testing.T) {
		// Ensure tool policy is applied consistently to all choices
		policies := []ToolPolicy{
			ToolStopOnFirst,
			ToolCollectThenStop,
			ToolDrainAll,
			ToolAllowMixed,
		}

		for _, policy := range policies {
			t.Run(policy.String(), func(t *testing.T) {
				adapter := New(
					WithLogLevel(slog.LevelError),
					WithToolPolicy(policy),
					WithToolMaxCalls(2), // Limit for testing
				)

				response := openai.ChatCompletion{
					Choices: []openai.ChatCompletionChoice{
						{
							Message: openai.ChatCompletionMessage{
								Content: `[{"name": "t1", "parameters": {}}, {"name": "t2", "parameters": {}}, {"name": "t3", "parameters": {}}]`,
							},
						},
						{
							Message: openai.ChatCompletionMessage{
								Content: `[{"name": "t4", "parameters": {}}, {"name": "t5", "parameters": {}}, {"name": "t6", "parameters": {}}]`,
							},
						},
					},
				}

				result, err := adapter.TransformCompletionsResponse(response)
				require.NoError(t, err)

				// Verify policy is applied to each choice
				for i, choice := range result.Choices {
					verifyPolicyApplication(t, policy, choice, i)
				}
			})
		}
	})

	t.Run("TimeoutDuringMultiChoice", func(t *testing.T) {
		adapter := New(WithLogLevel(slog.LevelError))

		// Create context that times out immediately
		ctx, cancel := context.WithTimeout(context.Background(), 1*time.Nanosecond)
		defer cancel()

		time.Sleep(1 * time.Millisecond) // Ensure timeout

		response := openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Content: `[{"name": "tool1", "parameters": {}}]`,
					},
				},
			},
		}

		_, err := adapter.TransformCompletionsResponseWithContext(ctx, response)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "context")
	})
}
