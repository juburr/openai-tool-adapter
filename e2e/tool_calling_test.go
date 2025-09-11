//go:build e2e

package e2e

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/openai/openai-go/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestToolCallingNonStreaming(t *testing.T) {
	client := NewTestClient()
	ctx, cancel := client.CreateTimeoutContext()
	defer cancel()

	t.Run("WeatherToolCall", func(t *testing.T) {
		weatherTool := CreateWeatherTool()
		request := client.CreateToolRequest("What's the weather like in San Francisco?", []openai.ChatCompletionToolUnionParam{weatherTool})

		response, err := client.SendRequest(ctx, request)
		require.NoError(t, err, "Weather tool request should not fail")
		require.NotNil(t, response, "Response should not be nil")
		require.Len(t, response.Choices, 1, "Should have exactly one choice")

		choice := response.Choices[0]
		assert.Equal(t, "tool_calls", choice.FinishReason, "Should finish with tool_calls")
		assert.NotEmpty(t, choice.Message.ToolCalls, "Should have tool calls")

		// Verify the tool call structure
		toolCall := choice.Message.ToolCalls[0]
		assert.NotEmpty(t, toolCall.ID, "Tool call should have an ID")
		assert.Equal(t, "function", string(toolCall.Type), "Tool call type should be function")
		assert.Equal(t, "get_weather", toolCall.Function.Name, "Function name should be get_weather")
		assert.NotEmpty(t, toolCall.Function.Arguments, "Should have arguments")

		// Parse and validate arguments
		var args map[string]interface{}
		err = json.Unmarshal([]byte(toolCall.Function.Arguments), &args)
		require.NoError(t, err, "Arguments should be valid JSON")

		location, exists := args["location"]
		require.True(t, exists, "Arguments should contain location")
		assert.Contains(t, strings.ToLower(location.(string)), "san francisco", "Location should be San Francisco")

		t.Logf("Tool call ID: %s", toolCall.ID)
		t.Logf("Arguments: %s", toolCall.Function.Arguments)
	})

	t.Run("CalculatorToolCall", func(t *testing.T) {
		calcTool := CreateCalculatorTool()
		request := client.CreateToolRequest("What is 25 multiplied by 4?", []openai.ChatCompletionToolUnionParam{calcTool})

		response, err := client.SendRequest(ctx, request)
		require.NoError(t, err, "Calculator tool request should not fail")
		require.NotNil(t, response, "Response should not be nil")
		require.Len(t, response.Choices, 1, "Should have exactly one choice")

		choice := response.Choices[0]
		assert.Equal(t, "tool_calls", choice.FinishReason, "Should finish with tool_calls")
		assert.NotEmpty(t, choice.Message.ToolCalls, "Should have tool calls")

		toolCall := choice.Message.ToolCalls[0]
		assert.Equal(t, "calculate", toolCall.Function.Name, "Function name should be calculate")

		var args map[string]interface{}
		err = json.Unmarshal([]byte(toolCall.Function.Arguments), &args)
		require.NoError(t, err, "Arguments should be valid JSON")

		expression, exists := args["expression"]
		require.True(t, exists, "Arguments should contain expression")

		// The expression should contain multiplication of 25 and 4
		exprStr := strings.ToLower(expression.(string))
		assert.True(t,
			(strings.Contains(exprStr, "25") && strings.Contains(exprStr, "4")) ||
				(strings.Contains(exprStr, "100")), // Model might directly compute it
			"Expression should involve 25 and 4 or the result 100")

		t.Logf("Expression: %s", expression)
	})

	t.Run("MultipleToolsAvailable", func(t *testing.T) {
		weatherTool := CreateWeatherTool()
		calcTool := CreateCalculatorTool()
		tools := []openai.ChatCompletionToolUnionParam{weatherTool, calcTool}

		request := client.CreateToolRequest("What's the weather in Tokyo?", tools)

		response, err := client.SendRequest(ctx, request)
		require.NoError(t, err, "Multiple tools request should not fail")
		require.NotNil(t, response, "Response should not be nil")
		require.Len(t, response.Choices, 1, "Should have exactly one choice")

		choice := response.Choices[0]
		assert.Equal(t, "tool_calls", choice.FinishReason, "Should finish with tool_calls")
		assert.NotEmpty(t, choice.Message.ToolCalls, "Should have tool calls")

		// Should call the weather tool, not the calculator
		toolCall := choice.Message.ToolCalls[0]
		assert.Equal(t, "get_weather", toolCall.Function.Name, "Should call the weather function for weather question")

		var args map[string]interface{}
		err = json.Unmarshal([]byte(toolCall.Function.Arguments), &args)
		require.NoError(t, err, "Arguments should be valid JSON")

		location := args["location"].(string)
		assert.Contains(t, strings.ToLower(location), "tokyo", "Location should be Tokyo")

		t.Logf("Selected tool: %s with args: %s", toolCall.Function.Name, toolCall.Function.Arguments)
	})

	t.Run("AmbiguousRequestWithTools", func(t *testing.T) {
		weatherTool := CreateWeatherTool()
		request := client.CreateToolRequest("Tell me about your capabilities", []openai.ChatCompletionToolUnionParam{weatherTool})

		response, err := client.SendRequest(ctx, request)
		require.NoError(t, err, "Ambiguous request should not fail")
		require.NotNil(t, response, "Response should not be nil")
		require.Len(t, response.Choices, 1, "Should have exactly one choice")

		choice := response.Choices[0]
		// The model might or might not call tools for this ambiguous request
		// We just verify it responds appropriately
		if len(choice.Message.ToolCalls) > 0 {
			t.Logf("Model decided to call tools: %d calls", len(choice.Message.ToolCalls))
			assert.Equal(t, "tool_calls", choice.FinishReason, "Should finish with tool_calls if tools were called")
		} else {
			t.Logf("Model decided not to call tools, content: %s", choice.Message.Content)
			assert.NotEmpty(t, choice.Message.Content, "Should have content if no tools were called")
			assert.NotEqual(t, "tool_calls", choice.FinishReason, "Should not finish with tool_calls if no tools were called")
		}
	})

	t.Run("ToolCallWithSpecificUnits", func(t *testing.T) {
		weatherTool := CreateWeatherTool()
		request := client.CreateToolRequest("What's the weather in London in Fahrenheit?", []openai.ChatCompletionToolUnionParam{weatherTool})

		response, err := client.SendRequest(ctx, request)
		require.NoError(t, err, "Specific units request should not fail")
		require.NotNil(t, response, "Response should not be nil")
		require.Len(t, response.Choices, 1, "Should have exactly one choice")

		choice := response.Choices[0]
		if len(choice.Message.ToolCalls) > 0 {
			toolCall := choice.Message.ToolCalls[0]
			assert.Equal(t, "get_weather", toolCall.Function.Name, "Should call weather function")

			var args map[string]interface{}
			err = json.Unmarshal([]byte(toolCall.Function.Arguments), &args)
			require.NoError(t, err, "Arguments should be valid JSON")

			location := args["location"].(string)
			assert.Contains(t, strings.ToLower(location), "london", "Location should be London")

			// Check if unit was properly extracted
			if unit, exists := args["unit"]; exists {
				assert.Equal(t, "fahrenheit", strings.ToLower(unit.(string)), "Unit should be fahrenheit")
				t.Logf("Model correctly extracted unit: %s", unit)
			} else {
				t.Logf("Model did not extract unit (optional parameter)")
			}
		}
	})
}
