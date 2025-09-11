//go:build e2e

package e2e

import (
	"context"
	"encoding/json"
	"log/slog"
	"strings"
	"testing"
	"time"

	tooladapter "github.com/juburr/openai-tool-adapter"
	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestStreamingBasicRequests(t *testing.T) {
	client := NewTestClient()
	ctx, cancel := client.CreateTimeoutContext()
	defer cancel()

	t.Run("StreamingWithoutTools", func(t *testing.T) {
		request := client.CreateBasicRequest("Hello, please tell me a short story.")

		streamAdapter, err := client.SendStreamingRequest(ctx, request)
		require.NoError(t, err, "Streaming request should not fail")
		require.NotNil(t, streamAdapter, "Stream adapter should not be nil")
		defer streamAdapter.Close()

		var contentBuilder strings.Builder
		var toolCallsFromStream []openai.ChatCompletionChunkChoiceDeltaToolCall
		chunkCount := 0

		for streamAdapter.Next() {
			chunkCount++
			chunk := streamAdapter.Current()

			require.Len(t, chunk.Choices, 1, "Each chunk should have exactly one choice")

			choice := chunk.Choices[0]

			// Collect content
			if choice.Delta.Content != "" {
				contentBuilder.WriteString(choice.Delta.Content)
			}

			// Collect tool calls (should be none for this test)
			toolCallsFromStream = append(toolCallsFromStream, choice.Delta.ToolCalls...)

			// Verify no tool calls in basic request
			assert.Empty(t, choice.Delta.ToolCalls, "Basic streaming should not have tool calls")
		}

		require.NoError(t, streamAdapter.Err(), "Stream should not have errors")

		content := contentBuilder.String()
		assert.NotEmpty(t, content, "Should receive content from stream")
		assert.Empty(t, toolCallsFromStream, "Should have no tool calls")
		assert.Greater(t, chunkCount, 0, "Should receive at least one chunk")

		t.Logf("Received %d chunks, total content length: %d", chunkCount, len(content))
		t.Logf("Content preview: %s", content[:min(100, len(content))])
	})

	t.Run("StreamingEmptyResponse", func(t *testing.T) {
		// Test with a request that might produce minimal response
		request := client.CreateBasicRequest("Respond with just 'OK'")

		streamAdapter, err := client.SendStreamingRequest(ctx, request)
		require.NoError(t, err, "Streaming request should not fail")
		require.NotNil(t, streamAdapter, "Stream adapter should not be nil")
		defer streamAdapter.Close()

		var contentBuilder strings.Builder
		chunkCount := 0

		for streamAdapter.Next() {
			chunkCount++
			chunk := streamAdapter.Current()

			if len(chunk.Choices) > 0 {
				choice := chunk.Choices[0]
				if choice.Delta.Content != "" {
					contentBuilder.WriteString(choice.Delta.Content)
				}
			}
		}

		require.NoError(t, streamAdapter.Err(), "Stream should not have errors")

		content := contentBuilder.String()
		// Content may be empty or contain minimal response
		t.Logf("Minimal response test - chunks: %d, content: '%s'", chunkCount, content)
	})
}

func TestStreamingToolCalling(t *testing.T) {
	client := NewTestClient()
	ctx, cancel := client.CreateTimeoutContext()
	defer cancel()

	t.Run("StreamingWeatherToolCall", func(t *testing.T) {
		weatherTool := CreateWeatherTool()
		request := client.CreateToolRequest("What's the weather in New York?", []openai.ChatCompletionToolUnionParam{weatherTool})

		streamAdapter, err := client.SendStreamingRequest(ctx, request)
		require.NoError(t, err, "Streaming tool request should not fail")
		require.NotNil(t, streamAdapter, "Stream adapter should not be nil")
		defer streamAdapter.Close()

		var contentBuilder strings.Builder
		var toolCallsFromStream []openai.ChatCompletionChunkChoiceDeltaToolCall
		chunkCount := 0
		var finalFinishReason string

		for streamAdapter.Next() {
			chunkCount++
			chunk := streamAdapter.Current()

			if len(chunk.Choices) > 0 {
				choice := chunk.Choices[0]

				// Collect content
				if choice.Delta.Content != "" {
					contentBuilder.WriteString(choice.Delta.Content)
				}

				// Collect tool calls
				toolCallsFromStream = append(toolCallsFromStream, choice.Delta.ToolCalls...)

				// Track finish reason
				if choice.FinishReason != "" {
					finalFinishReason = choice.FinishReason
				}
			}
		}

		require.NoError(t, streamAdapter.Err(), "Stream should not have errors")

		assert.Greater(t, chunkCount, 0, "Should receive at least one chunk")

		// Check if we got tool calls - if not, the model might not have called tools
		if len(toolCallsFromStream) == 0 {
			t.Logf("Model did not call tools in streaming mode (finish reason: %s)", finalFinishReason)
			// This is acceptable behavior - streaming tool calling can be inconsistent
			// Some models may not follow tool calling instructions as reliably in streaming mode
			t.Logf("Total chunks received: %d", chunkCount)
			return
		}

		assert.Equal(t, "tool_calls", finalFinishReason, "Should finish with tool_calls")

		// Verify the tool call
		toolCall := toolCallsFromStream[0]
		assert.NotEmpty(t, toolCall.ID, "Tool call should have an ID")
		assert.Equal(t, "function", string(toolCall.Type), "Tool call type should be function")
		assert.Equal(t, "get_weather", toolCall.Function.Name, "Function name should be get_weather")
		assert.NotEmpty(t, toolCall.Function.Arguments, "Should have arguments")

		// Validate arguments
		var args map[string]interface{}
		err = json.Unmarshal([]byte(toolCall.Function.Arguments), &args)
		require.NoError(t, err, "Arguments should be valid JSON")

		location := args["location"].(string)
		assert.Contains(t, strings.ToLower(location), "new york", "Location should be New York")

		t.Logf("Streaming tool call - chunks: %d, tool calls: %d", chunkCount, len(toolCallsFromStream))
		t.Logf("Tool call arguments: %s", toolCall.Function.Arguments)
	})

	t.Run("StreamingWithMultipleToolsAvailable", func(t *testing.T) {
		weatherTool := CreateWeatherTool()
		calcTool := CreateCalculatorTool()
		tools := []openai.ChatCompletionToolUnionParam{weatherTool, calcTool}

		request := client.CreateToolRequest("Calculate 15 plus 27", tools)

		streamAdapter, err := client.SendStreamingRequest(ctx, request)
		require.NoError(t, err, "Streaming multiple tools request should not fail")
		require.NotNil(t, streamAdapter, "Stream adapter should not be nil")
		defer streamAdapter.Close()

		var toolCallsFromStream []openai.ChatCompletionChunkChoiceDeltaToolCall
		var finalFinishReason string

		for streamAdapter.Next() {
			chunk := streamAdapter.Current()

			if len(chunk.Choices) > 0 {
				choice := chunk.Choices[0]
				toolCallsFromStream = append(toolCallsFromStream, choice.Delta.ToolCalls...)

				if choice.FinishReason != "" {
					finalFinishReason = choice.FinishReason
				}
			}
		}

		require.NoError(t, streamAdapter.Err(), "Stream should not have errors")

		// Check if we got tool calls - if not, the model might not have called tools
		if len(toolCallsFromStream) == 0 {
			t.Logf("Model did not call tools for math question (finish reason: %s)", finalFinishReason)
			return
		}

		assert.Equal(t, "tool_calls", finalFinishReason, "Should finish with tool_calls")

		// Should call the calculator tool, not weather
		toolCall := toolCallsFromStream[0]
		assert.Equal(t, "calculate", toolCall.Function.Name, "Should call calculate function for math question")

		var args map[string]interface{}
		err = json.Unmarshal([]byte(toolCall.Function.Arguments), &args)
		require.NoError(t, err, "Arguments should be valid JSON")

		expression := args["expression"].(string)
		assert.True(t,
			(strings.Contains(expression, "15") && strings.Contains(expression, "27")) ||
				strings.Contains(expression, "42"), // Model might compute directly
			"Expression should involve 15 and 27 or result 42")

		t.Logf("Streaming selected tool: %s with expression: %s", toolCall.Function.Name, expression)
	})
}

func TestStreamingWithDifferentPolicies(t *testing.T) {
	config := LoadTestConfig()

	t.Run("StreamingWithStopOnFirst", func(t *testing.T) {
		// Create adapter with ToolStopOnFirst policy (default)
		adapter := tooladapter.New(
			tooladapter.WithLogLevel(slog.LevelError),
			tooladapter.WithToolPolicy(tooladapter.ToolStopOnFirst),
		)

		client := openai.NewClient(
			option.WithBaseURL(config.BaseURL),
			option.WithAPIKey(config.APIKey),
		)

		ctx, cancel := context.WithTimeout(context.Background(), time.Duration(config.Timeout)*time.Second)
		defer cancel()

		weatherTool := CreateWeatherTool()
		request := openai.ChatCompletionNewParams{
			Model: config.Model,
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("What's the weather in Paris?"),
			},
			Tools: []openai.ChatCompletionToolUnionParam{weatherTool},
		}

		transformedRequest, err := adapter.TransformCompletionsRequest(request)
		require.NoError(t, err, "Request transformation should not fail")

		stream := client.Chat.Completions.NewStreaming(ctx, transformedRequest)
		streamAdapter := adapter.TransformStreamingResponseWithContext(ctx, stream)
		defer streamAdapter.Close()

		var toolCallsFromStream []openai.ChatCompletionChunkChoiceDeltaToolCall
		var content strings.Builder
		chunkCount := 0

		for streamAdapter.Next() {
			chunkCount++
			chunk := streamAdapter.Current()

			if len(chunk.Choices) > 0 {
				choice := chunk.Choices[0]

				if choice.Delta.Content != "" {
					content.WriteString(choice.Delta.Content)
				}

				toolCallsFromStream = append(toolCallsFromStream, choice.Delta.ToolCalls...)
			}
		}

		require.NoError(t, streamAdapter.Err(), "Stream should not have errors")

		if len(toolCallsFromStream) > 0 {
			// With ToolStopOnFirst, we should get the first tool call
			assert.Equal(t, "get_weather", toolCallsFromStream[0].Function.Name, "Should get weather tool call")
			t.Logf("ToolStopOnFirst policy - tool calls: %d, chunks: %d", len(toolCallsFromStream), chunkCount)
		} else {
			t.Logf("Model chose not to call tools - content: %s", content.String())
		}
	})

	t.Run("StreamingWithAllowMixed", func(t *testing.T) {
		// Create adapter with ToolAllowMixed policy
		adapter := tooladapter.New(
			tooladapter.WithLogLevel(slog.LevelError),
			tooladapter.WithToolPolicy(tooladapter.ToolAllowMixed),
		)

		client := openai.NewClient(
			option.WithBaseURL(config.BaseURL),
			option.WithAPIKey(config.APIKey),
		)

		ctx, cancel := context.WithTimeout(context.Background(), time.Duration(config.Timeout)*time.Second)
		defer cancel()

		weatherTool := CreateWeatherTool()
		request := openai.ChatCompletionNewParams{
			Model: config.Model,
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("What's the weather in Berlin? Also tell me about the city."),
			},
			Tools: []openai.ChatCompletionToolUnionParam{weatherTool},
		}

		transformedRequest, err := adapter.TransformCompletionsRequest(request)
		require.NoError(t, err, "Request transformation should not fail")

		stream := client.Chat.Completions.NewStreaming(ctx, transformedRequest)
		streamAdapter := adapter.TransformStreamingResponseWithContext(ctx, stream)
		defer streamAdapter.Close()

		var toolCallsFromStream []openai.ChatCompletionChunkChoiceDeltaToolCall
		var content strings.Builder
		chunkCount := 0

		for streamAdapter.Next() {
			chunkCount++
			chunk := streamAdapter.Current()

			if len(chunk.Choices) > 0 {
				choice := chunk.Choices[0]

				if choice.Delta.Content != "" {
					content.WriteString(choice.Delta.Content)
				}

				toolCallsFromStream = append(toolCallsFromStream, choice.Delta.ToolCalls...)
			}
		}

		require.NoError(t, streamAdapter.Err(), "Stream should not have errors")

		// With ToolAllowMixed, we might get both tools and content
		t.Logf("ToolAllowMixed policy - tool calls: %d, content length: %d, chunks: %d",
			len(toolCallsFromStream), content.Len(), chunkCount)

		if len(toolCallsFromStream) > 0 {
			assert.Equal(t, "get_weather", toolCallsFromStream[0].Function.Name, "Should get weather tool call")
		}

		// Content is preserved with AllowMixed policy
		// (though the specific behavior depends on the model)
		t.Logf("Content preview: %s", content.String()[:min(100, content.Len())])
	})
}
