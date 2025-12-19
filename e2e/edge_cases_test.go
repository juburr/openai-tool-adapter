//go:build e2e

package e2e

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEdgeCasesAndErrorHandling(t *testing.T) {
	client := NewTestClient()

	t.Run("RequestTimeout", func(t *testing.T) {
		// Create a very short timeout context
		ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
		defer cancel()

		request := client.CreateBasicRequest("Tell me a very long story with lots of details...")

		_, err := client.SendRequest(ctx, request)
		// This should timeout or complete quickly
		if err != nil {
			assert.Contains(t, err.Error(), "context", "Timeout error should mention context")
			t.Logf("Request properly timed out: %v", err)
		} else {
			t.Logf("Request completed within timeout (model was fast)")
		}
	})

	t.Run("InvalidToolDefinition", func(t *testing.T) {
		ctx, cancel := client.CreateTimeoutContext()
		defer cancel()

		// Create tool with invalid/malformed definition
		invalidTool := openai.ChatCompletionFunctionTool(openai.FunctionDefinitionParam{
			Name:        "invalid-tool-name-with-hyphens-and-special$chars!",
			Description: openai.String("An invalid tool for testing"),
			Parameters: openai.FunctionParameters{
				"type": "object",
				// Missing properties and other required fields
			},
		})

		request := client.CreateToolRequest("Use the invalid tool", []openai.ChatCompletionToolUnionParam{invalidTool})

		// The adapter should handle this gracefully
		response, err := client.SendRequest(ctx, request)
		if err != nil {
			t.Logf("Request failed as expected with invalid tool: %v", err)
		} else {
			require.NotNil(t, response, "Response should not be nil")
			// Model might ignore the invalid tool or handle it gracefully
			t.Logf("Model handled invalid tool gracefully")
		}
	})

	t.Run("VeryLongToolArguments", func(t *testing.T) {
		// Create longer timeout for this specific test
		ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
		defer cancel()

		// Create a tool that might receive very long arguments
		longParamTool := openai.ChatCompletionFunctionTool(openai.FunctionDefinitionParam{
			Name:        "process_text",
			Description: openai.String("Process a long text input"),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]interface{}{
					"text": map[string]interface{}{
						"type":        "string",
						"description": "Long text to process",
					},
				},
				"required": []string{"text"},
			},
		})

		// Create a moderately long message (reduced from 100 to 20 repetitions)
		longText := strings.Repeat("This is a sentence that should be processed. ", 20)
		request := client.CreateToolRequest("Process this text: "+longText, []openai.ChatCompletionToolUnionParam{longParamTool})

		response, err := client.SendRequest(ctx, request)
		if err != nil {
			t.Logf("Long arguments request failed (acceptable for slow models): %v", err)
			return
		}
		require.NotNil(t, response, "Response should not be nil")

		if len(response.Choices) > 0 && len(response.Choices[0].Message.ToolCalls) > 0 {
			toolCall := response.Choices[0].Message.ToolCalls[0]
			t.Logf("Tool call with long arguments - length: %d", len(toolCall.Function.Arguments))

			// Verify arguments are still valid JSON
			var args map[string]interface{}
			err = json.Unmarshal([]byte(toolCall.Function.Arguments), &args)
			assert.NoError(t, err, "Long arguments should still be valid JSON")
		} else {
			t.Logf("Model chose not to call tools for long text processing")
		}
	})

	t.Run("MultipleConsecutiveRequests", func(t *testing.T) {
		ctx, cancel := client.CreateTimeoutContext()
		defer cancel()

		weatherTool := CreateWeatherTool()

		cities := []string{"London", "Paris", "Tokyo", "New York", "Sydney"}

		for _, city := range cities {
			request := client.CreateToolRequest("What's the weather in "+city+"?", []openai.ChatCompletionToolUnionParam{weatherTool})

			response, err := client.SendRequest(ctx, request)
			require.NoError(t, err, "Consecutive request for %s should not fail", city)
			require.NotNil(t, response, "Response should not be nil")

			if len(response.Choices) > 0 && len(response.Choices[0].Message.ToolCalls) > 0 {
				toolCall := response.Choices[0].Message.ToolCalls[0]
				assert.Equal(t, "get_weather", toolCall.Function.Name, "Should call weather function")

				var args map[string]interface{}
				err = json.Unmarshal([]byte(toolCall.Function.Arguments), &args)
				require.NoError(t, err, "Arguments should be valid JSON")

				location := strings.ToLower(args["location"].(string))
				assert.Contains(t, location, strings.ToLower(city), "Location should contain the requested city")

				t.Logf("Request for %s: location=%s", city, args["location"])
			}
		}
	})

	t.Run("UnicodeAndSpecialCharacters", func(t *testing.T) {
		ctx, cancel := client.CreateTimeoutContext()
		defer cancel()

		weatherTool := CreateWeatherTool()

		// Test with unicode cities and special characters
		unicodeCities := []string{
			"北京",        // Beijing in Chinese
			"Москва",    // Moscow in Russian
			"São Paulo", // São Paulo with accent
			"München",   // München with umlaut
		}

		for _, city := range unicodeCities {
			request := client.CreateToolRequest("What's the weather in "+city+"?", []openai.ChatCompletionToolUnionParam{weatherTool})

			response, err := client.SendRequest(ctx, request)
			require.NoError(t, err, "Unicode request for %s should not fail", city)
			require.NotNil(t, response, "Response should not be nil")

			if len(response.Choices) > 0 && len(response.Choices[0].Message.ToolCalls) > 0 {
				toolCall := response.Choices[0].Message.ToolCalls[0]

				// Verify arguments are still valid JSON with unicode
				var args map[string]interface{}
				err = json.Unmarshal([]byte(toolCall.Function.Arguments), &args)
				assert.NoError(t, err, "Unicode arguments should still be valid JSON")

				location := args["location"].(string)
				t.Logf("Unicode request for %s: extracted location=%s", city, location)
			}
		}
	})
}

func TestStreamingEdgeCases(t *testing.T) {
	client := NewTestClient()

	t.Run("StreamingTimeout", func(t *testing.T) {
		// Create a very short timeout for streaming
		ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
		defer cancel()

		request := client.CreateBasicRequest("Tell me a very detailed story about space exploration...")

		streamAdapter, err := client.SendStreamingRequest(ctx, request)
		if err != nil {
			t.Logf("Streaming request failed as expected: %v", err)
			return
		}

		require.NotNil(t, streamAdapter, "Stream adapter should not be nil")
		defer streamAdapter.Close()

		chunkCount := 0
		for streamAdapter.Next() {
			chunkCount++
			_ = streamAdapter.Current()

			// Check if context is done
			select {
			case <-ctx.Done():
				t.Logf("Context cancelled during streaming after %d chunks", chunkCount)
				break
			default:
				continue
			}
		}

		// Stream error might indicate timeout
		if err := streamAdapter.Err(); err != nil {
			assert.Contains(t, strings.ToLower(err.Error()), "context", "Stream error should mention context")
			t.Logf("Stream properly handled timeout: %v", err)
		}
	})

	t.Run("StreamingWithMalformedResponse", func(t *testing.T) {
		ctx, cancel := client.CreateTimeoutContext()
		defer cancel()

		// Use a request that might result in unusual formatting
		request := client.CreateBasicRequest("Respond with JSON-like text but not actually JSON: {hello: world}")

		streamAdapter, err := client.SendStreamingRequest(ctx, request)
		require.NoError(t, err, "Streaming request should not fail")
		require.NotNil(t, streamAdapter, "Stream adapter should not be nil")
		defer streamAdapter.Close()

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

				// Should not have tool calls for malformed JSON-like content
				assert.Empty(t, choice.Delta.ToolCalls, "Malformed JSON-like content should not trigger tool calls")
			}
		}

		require.NoError(t, streamAdapter.Err(), "Stream should handle malformed content gracefully")

		finalContent := content.String()
		t.Logf("Malformed response test - chunks: %d, content length: %d", chunkCount, len(finalContent))
		t.Logf("Content preview: %s", finalContent[:min(200, len(finalContent))])
	})

	t.Run("StreamingEarlyClose", func(t *testing.T) {
		ctx, cancel := client.CreateTimeoutContext()
		defer cancel()

		weatherTool := CreateWeatherTool()
		request := client.CreateToolRequest("What's the weather in Miami?", []openai.ChatCompletionToolUnionParam{weatherTool})

		streamAdapter, err := client.SendStreamingRequest(ctx, request)
		require.NoError(t, err, "Streaming request should not fail")
		require.NotNil(t, streamAdapter, "Stream adapter should not be nil")

		chunkCount := 0
		for streamAdapter.Next() {
			chunkCount++
			_ = streamAdapter.Current()

			// Close stream early after receiving some chunks
			if chunkCount >= 2 {
				break
			}
		}

		// Close the stream early
		err = streamAdapter.Close()
		assert.NoError(t, err, "Early close should not cause errors")

		t.Logf("Stream closed early after %d chunks", chunkCount)
	})
}

func TestResourceLimitsAndSafety(t *testing.T) {
	client := NewTestClient()

	t.Run("VeryLargeRequest", func(t *testing.T) {
		ctx, cancel := client.CreateTimeoutContext()
		defer cancel()

		// Create a very large message
		largeMessage := strings.Repeat("This is a test sentence. ", 10000) // ~250KB
		request := client.CreateBasicRequest(largeMessage)

		response, err := client.SendRequest(ctx, request)
		if err != nil {
			// Large requests might fail due to model limits
			t.Logf("Large request failed as expected: %v", err)
		} else {
			require.NotNil(t, response, "Response should not be nil")
			t.Logf("Large request handled successfully, response length: %d",
				len(response.Choices[0].Message.Content))
		}
	})

	t.Run("ManyToolsInRequest", func(t *testing.T) {
		ctx, cancel := client.CreateTimeoutContext()
		defer cancel()

		// Create many tools
		var tools []openai.ChatCompletionToolUnionParam
		for i := 0; i < 20; i++ {
			tool := openai.ChatCompletionFunctionTool(openai.FunctionDefinitionParam{
				Name:        fmt.Sprintf("test_function_%d", i),
				Description: openai.String(fmt.Sprintf("Test function number %d", i)),
				Parameters: openai.FunctionParameters{
					"type": "object",
					"properties": map[string]interface{}{
						"param": map[string]interface{}{
							"type":        "string",
							"description": "A parameter",
						},
					},
				},
			})
			tools = append(tools, tool)
		}

		request := client.CreateToolRequest("Use any appropriate function from the available tools", tools)

		response, err := client.SendRequest(ctx, request)
		if err != nil {
			// Many tools might exceed model context limits
			t.Logf("Many tools request failed: %v", err)
		} else {
			require.NotNil(t, response, "Response should not be nil")

			if len(response.Choices) > 0 && len(response.Choices[0].Message.ToolCalls) > 0 {
				toolCall := response.Choices[0].Message.ToolCalls[0]
				assert.Contains(t, toolCall.Function.Name, "test_function_", "Should call one of the test functions")
				t.Logf("Selected function from many tools: %s", toolCall.Function.Name)
			}
		}
	})
}
