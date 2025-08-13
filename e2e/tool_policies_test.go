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
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestToolPolicyStopOnFirst(t *testing.T) {
	config := LoadTestConfig()

	t.Run("NonStreamingStopOnFirst", func(t *testing.T) {
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

		// Create multiple tools that could all be relevant
		weatherTool := CreateWeatherTool()
		timeTool := CreateTimeTool()
		searchTool := CreateSearchTool()
		calcTool := CreateCalculatorTool()

		tools := []openai.ChatCompletionToolParam{weatherTool, timeTool, searchTool, calcTool}

		// Ask a question that could reasonably trigger multiple tools
		request := openai.ChatCompletionNewParams{
			Model: config.Model,
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("What's the weather in New York, what time is it there, and calculate 15+27? Also search for 'New York tourism'."),
			},
			Tools: tools,
		}

		transformedRequest, err := adapter.TransformCompletionsRequest(request)
		require.NoError(t, err, "Request transformation should not fail")

		response, err := client.Chat.Completions.New(ctx, transformedRequest)
		require.NoError(t, err, "Chat completion should not fail")

		transformedResponse, err := adapter.TransformCompletionsResponse(*response)
		require.NoError(t, err, "Response transformation should not fail")

		if len(transformedResponse.Choices) == 0 {
			t.Skip("No choices in response")
		}

		choice := transformedResponse.Choices[0]

		// With ToolStopOnFirst, we should get at most one tool call
		toolCallCount := len(choice.Message.ToolCalls)
		if toolCallCount == 0 {
			t.Logf("Model chose not to call any tools - finish reason: %s", string(choice.FinishReason))
			t.Logf("Content: %s", choice.Message.Content)
			return
		}

		assert.LessOrEqual(t, toolCallCount, 1, "ToolStopOnFirst should return at most one tool call")
		assert.Equal(t, "tool_calls", string(choice.FinishReason), "Should finish with tool_calls")

		// Verify the tool call is well-formed
		toolCall := choice.Message.ToolCalls[0]
		assert.NotEmpty(t, toolCall.ID, "Tool call should have an ID")
		assert.Equal(t, "function", string(toolCall.Type), "Tool call type should be function")
		assert.NotEmpty(t, toolCall.Function.Name, "Function name should not be empty")
		assert.NotEmpty(t, toolCall.Function.Arguments, "Arguments should not be empty")

		// Validate that arguments are valid JSON
		var args map[string]interface{}
		err = json.Unmarshal([]byte(toolCall.Function.Arguments), &args)
		require.NoError(t, err, "Arguments should be valid JSON")

		t.Logf("ToolStopOnFirst - tool calls: %d, function: %s", toolCallCount, toolCall.Function.Name)
		t.Logf("Arguments: %s", toolCall.Function.Arguments)
	})

	t.Run("StreamingStopOnFirst", func(t *testing.T) {
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
		calcTool := CreateCalculatorTool()
		timeTool := CreateTimeTool()

		tools := []openai.ChatCompletionToolParam{weatherTool, calcTool, timeTool}

		request := openai.ChatCompletionNewParams{
			Model: config.Model,
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("What's the weather in Paris, what time is it there, and calculate 10*5?"),
			},
			Tools: tools,
		}

		transformedRequest, err := adapter.TransformCompletionsRequest(request)
		require.NoError(t, err, "Request transformation should not fail")

		stream := client.Chat.Completions.NewStreaming(ctx, transformedRequest)
		streamAdapter := adapter.TransformStreamingResponseWithContext(ctx, stream)
		defer streamAdapter.Close()

		var toolCallsFromStream []openai.ChatCompletionChunkChoiceDeltaToolCall
		var finalFinishReason string
		chunkCount := 0

		for streamAdapter.Next() {
			chunkCount++
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

		if len(toolCallsFromStream) == 0 {
			t.Logf("Model chose not to call tools in streaming mode (finish reason: %s)", finalFinishReason)
			return
		}

		// With ToolStopOnFirst in streaming, we should get only the first tool call
		// Count unique tool calls by ID
		toolCallIDs := make(map[string]bool)
		for _, toolCall := range toolCallsFromStream {
			if toolCall.ID != "" {
				toolCallIDs[toolCall.ID] = true
			}
		}

		uniqueToolCalls := len(toolCallIDs)
		assert.LessOrEqual(t, uniqueToolCalls, 1, "ToolStopOnFirst streaming should return at most one unique tool call")

		t.Logf("ToolStopOnFirst streaming - unique tool calls: %d, total chunks: %d", uniqueToolCalls, chunkCount)
	})
}

func TestToolPolicyCollectThenStop(t *testing.T) {
	config := LoadTestConfig()

	t.Run("NonStreamingCollectThenStop", func(t *testing.T) {
		adapter := tooladapter.New(
			tooladapter.WithLogLevel(slog.LevelError),
			tooladapter.WithToolPolicy(tooladapter.ToolCollectThenStop),
		)

		client := openai.NewClient(
			option.WithBaseURL(config.BaseURL),
			option.WithAPIKey(config.APIKey),
		)

		ctx, cancel := context.WithTimeout(context.Background(), time.Duration(config.Timeout)*time.Second)
		defer cancel()

		weatherTool := CreateWeatherTool()
		calcTool := CreateCalculatorTool()
		translationTool := CreateTranslationTool()

		tools := []openai.ChatCompletionToolParam{weatherTool, calcTool, translationTool}

		request := openai.ChatCompletionNewParams{
			Model: config.Model,
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("What's the weather in London? Calculate 25*4. Translate 'hello' from English to Spanish."),
			},
			Tools: tools,
		}

		transformedRequest, err := adapter.TransformCompletionsRequest(request)
		require.NoError(t, err, "Request transformation should not fail")

		response, err := client.Chat.Completions.New(ctx, transformedRequest)
		require.NoError(t, err, "Chat completion should not fail")

		transformedResponse, err := adapter.TransformCompletionsResponse(*response)
		require.NoError(t, err, "Response transformation should not fail")

		if len(transformedResponse.Choices) == 0 {
			t.Skip("No choices in response")
		}

		choice := transformedResponse.Choices[0]
		toolCallCount := len(choice.Message.ToolCalls)

		if toolCallCount == 0 {
			t.Logf("Model chose not to call any tools - finish reason: %s", string(choice.FinishReason))
			return
		}

		// ToolCollectThenStop should collect tools until array closes
		// This might result in multiple tool calls, but fewer than ToolDrainAll
		t.Logf("ToolCollectThenStop - tool calls: %d, finish reason: %s", toolCallCount, string(choice.FinishReason))

		for i, toolCall := range choice.Message.ToolCalls {
			t.Logf("Tool Call %d: %s", i, toolCall.Function.Name)
		}
	})
}

func TestToolPolicyDrainAll(t *testing.T) {
	config := LoadTestConfig()

	t.Run("NonStreamingDrainAll", func(t *testing.T) {
		adapter := tooladapter.New(
			tooladapter.WithLogLevel(slog.LevelError),
			tooladapter.WithToolPolicy(tooladapter.ToolDrainAll),
		)

		client := openai.NewClient(
			option.WithBaseURL(config.BaseURL),
			option.WithAPIKey(config.APIKey),
		)

		ctx, cancel := context.WithTimeout(context.Background(), time.Duration(config.Timeout)*time.Second)
		defer cancel()

		weatherTool := CreateWeatherTool()
		calcTool := CreateCalculatorTool()
		timeTool := CreateTimeTool()
		searchTool := CreateSearchTool()

		tools := []openai.ChatCompletionToolParam{weatherTool, calcTool, timeTool, searchTool}

		// Ask a complex question that should trigger multiple tools
		request := openai.ChatCompletionNewParams{
			Model: config.Model,
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("I need to know: the weather in Tokyo, the current time in Japan timezone, calculate 50*3, and search for 'Tokyo travel guide'. Please help with all of these."),
			},
			Tools: tools,
		}

		transformedRequest, err := adapter.TransformCompletionsRequest(request)
		require.NoError(t, err, "Request transformation should not fail")

		response, err := client.Chat.Completions.New(ctx, transformedRequest)
		require.NoError(t, err, "Chat completion should not fail")

		transformedResponse, err := adapter.TransformCompletionsResponse(*response)
		require.NoError(t, err, "Response transformation should not fail")

		if len(transformedResponse.Choices) == 0 {
			t.Skip("No choices in response")
		}

		choice := transformedResponse.Choices[0]
		toolCallCount := len(choice.Message.ToolCalls)

		if toolCallCount == 0 {
			t.Logf("Model chose not to call any tools - finish reason: %s", string(choice.FinishReason))
			return
		}

		// ToolDrainAll should collect ALL tool calls from the response
		t.Logf("ToolDrainAll - tool calls: %d, finish reason: %s", toolCallCount, string(choice.FinishReason))

		// Verify each tool call is well-formed
		for i, toolCall := range choice.Message.ToolCalls {
			assert.NotEmpty(t, toolCall.ID, "Tool call %d should have an ID", i)
			assert.Equal(t, "function", string(toolCall.Type), "Tool call %d type should be function", i)
			assert.NotEmpty(t, toolCall.Function.Name, "Tool call %d function name should not be empty", i)
			assert.NotEmpty(t, toolCall.Function.Arguments, "Tool call %d arguments should not be empty", i)

			var args map[string]interface{}
			err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args)
			require.NoError(t, err, "Tool call %d arguments should be valid JSON", i)

			t.Logf("Tool Call %d: %s with args: %s", i, toolCall.Function.Name, toolCall.Function.Arguments)
		}

		// ToolDrainAll should generally produce more tool calls than ToolStopOnFirst
		// (Though this depends on the model's behavior)
		assert.Greater(t, toolCallCount, 0, "ToolDrainAll should have at least one tool call")
	})

	t.Run("StreamingDrainAll", func(t *testing.T) {
		adapter := tooladapter.New(
			tooladapter.WithLogLevel(slog.LevelError),
			tooladapter.WithToolPolicy(tooladapter.ToolDrainAll),
		)

		client := openai.NewClient(
			option.WithBaseURL(config.BaseURL),
			option.WithAPIKey(config.APIKey),
		)

		ctx, cancel := context.WithTimeout(context.Background(), time.Duration(config.Timeout)*time.Second)
		defer cancel()

		weatherTool := CreateWeatherTool()
		calcTool := CreateCalculatorTool()
		timeTool := CreateTimeTool()

		tools := []openai.ChatCompletionToolParam{weatherTool, calcTool, timeTool}

		request := openai.ChatCompletionNewParams{
			Model: config.Model,
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("What's the weather in Berlin? What time is it there? Calculate 7*8."),
			},
			Tools: tools,
		}

		transformedRequest, err := adapter.TransformCompletionsRequest(request)
		require.NoError(t, err, "Request transformation should not fail")

		stream := client.Chat.Completions.NewStreaming(ctx, transformedRequest)
		streamAdapter := adapter.TransformStreamingResponseWithContext(ctx, stream)
		defer streamAdapter.Close()

		var toolCallsFromStream []openai.ChatCompletionChunkChoiceDeltaToolCall
		var finalFinishReason string
		chunkCount := 0

		for streamAdapter.Next() {
			chunkCount++
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

		if len(toolCallsFromStream) == 0 {
			t.Logf("Model chose not to call tools in streaming mode (finish reason: %s)", finalFinishReason)
			return
		}

		// Count unique tool calls by ID
		toolCallIDs := make(map[string]bool)
		for _, toolCall := range toolCallsFromStream {
			if toolCall.ID != "" {
				toolCallIDs[toolCall.ID] = true
			}
		}

		uniqueToolCalls := len(toolCallIDs)

		t.Logf("ToolDrainAll streaming - unique tool calls: %d, total deltas: %d, chunks: %d",
			uniqueToolCalls, len(toolCallsFromStream), chunkCount)

		// ToolDrainAll should read to end of stream and collect all tools
		assert.Greater(t, uniqueToolCalls, 0, "ToolDrainAll should have at least one unique tool call")
	})
}

func TestToolPolicyAllowMixed(t *testing.T) {
	config := LoadTestConfig()

	t.Run("NonStreamingAllowMixed", func(t *testing.T) {
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
		calcTool := CreateCalculatorTool()

		tools := []openai.ChatCompletionToolParam{weatherTool, calcTool}

		// Ask a question that encourages both tool use AND explanatory text
		request := openai.ChatCompletionNewParams{
			Model: config.Model,
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Please check the weather in Miami and calculate 12*15. Also explain what you're doing."),
			},
			Tools: tools,
		}

		transformedRequest, err := adapter.TransformCompletionsRequest(request)
		require.NoError(t, err, "Request transformation should not fail")

		response, err := client.Chat.Completions.New(ctx, transformedRequest)
		require.NoError(t, err, "Chat completion should not fail")

		transformedResponse, err := adapter.TransformCompletionsResponse(*response)
		require.NoError(t, err, "Response transformation should not fail")

		if len(transformedResponse.Choices) == 0 {
			t.Skip("No choices in response")
		}

		choice := transformedResponse.Choices[0]
		toolCallCount := len(choice.Message.ToolCalls)
		contentLength := len(strings.TrimSpace(choice.Message.Content))

		t.Logf("ToolAllowMixed - tool calls: %d, content length: %d, finish reason: %s",
			toolCallCount, contentLength, string(choice.FinishReason))

		// ToolAllowMixed should allow both tools AND content
		// The key difference is that content is not suppressed when tools are present
		if toolCallCount > 0 {
			t.Logf("Tool calls detected:")
			for i, toolCall := range choice.Message.ToolCalls {
				t.Logf("  Tool %d: %s", i, toolCall.Function.Name)
			}
		}

		if contentLength > 0 {
			t.Logf("Content preview: %s", choice.Message.Content[:min(200, len(choice.Message.Content))])
		}

		// With ToolAllowMixed, we might have tools, content, or both
		// The important thing is that content is not suppressed when tools are present
		hasAnyOutput := toolCallCount > 0 || contentLength > 0
		assert.True(t, hasAnyOutput, "ToolAllowMixed should have either tools or content or both")
	})

	t.Run("StreamingAllowMixed", func(t *testing.T) {
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
		timeTool := CreateTimeTool()

		tools := []openai.ChatCompletionToolParam{weatherTool, timeTool}

		request := openai.ChatCompletionNewParams{
			Model: config.Model,
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Tell me about Seattle's weather and the current time there. Please explain what you find."),
			},
			Tools: tools,
		}

		transformedRequest, err := adapter.TransformCompletionsRequest(request)
		require.NoError(t, err, "Request transformation should not fail")

		stream := client.Chat.Completions.NewStreaming(ctx, transformedRequest)
		streamAdapter := adapter.TransformStreamingResponseWithContext(ctx, stream)
		defer streamAdapter.Close()

		var toolCallsFromStream []openai.ChatCompletionChunkChoiceDeltaToolCall
		var content strings.Builder
		var finalFinishReason string
		chunkCount := 0

		for streamAdapter.Next() {
			chunkCount++
			chunk := streamAdapter.Current()

			if len(chunk.Choices) > 0 {
				choice := chunk.Choices[0]

				// Collect content
				if choice.Delta.Content != "" {
					content.WriteString(choice.Delta.Content)
				}

				// Collect tool calls
				toolCallsFromStream = append(toolCallsFromStream, choice.Delta.ToolCalls...)

				if choice.FinishReason != "" {
					finalFinishReason = choice.FinishReason
				}
			}
		}

		require.NoError(t, streamAdapter.Err(), "Stream should not have errors")

		// Count unique tool calls by ID
		toolCallIDs := make(map[string]bool)
		for _, toolCall := range toolCallsFromStream {
			if toolCall.ID != "" {
				toolCallIDs[toolCall.ID] = true
			}
		}

		uniqueToolCalls := len(toolCallIDs)
		contentLength := content.Len()

		t.Logf("ToolAllowMixed streaming - tool calls: %d, content length: %d, chunks: %d, finish reason: %s",
			uniqueToolCalls, contentLength, chunkCount, finalFinishReason)

		if contentLength > 0 {
			contentStr := content.String()
			t.Logf("Content preview: %s", contentStr[:min(200, len(contentStr))])
		}

		// ToolAllowMixed should preserve both tools and content in streaming
		hasAnyOutput := uniqueToolCalls > 0 || contentLength > 0
		assert.True(t, hasAnyOutput, "ToolAllowMixed streaming should have either tools or content or both")
	})
}

// Helper function for min (Go < 1.21 compatibility)
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
