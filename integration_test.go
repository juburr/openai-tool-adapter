package tooladapter_test

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"sync"
	"testing"
	"time"

	"github.com/juburr/openai-tool-adapter"
	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestIntegrationScenarios tests full end-to-end workflows and real-world usage patterns
func TestIntegrationScenarios(t *testing.T) {
	t.Run("FullWorkflow_RequestToResponse", func(t *testing.T) {
		// Test complete workflow from request transformation to response processing
		adapter := tooladapter.New(tooladapter.WithLogLevel(slog.LevelDebug))

		// 1. Transform request with tools
		originalReq := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Calculate the tax for someone earning $50000 in California"),
			},
			Model: openai.ChatModelGPT4o,
			Tools: []openai.ChatCompletionToolParam{
				{
					Type: "function",
					Function: openai.FunctionDefinitionParam{
						Name:        "calculate_tax",
						Description: openai.String("Calculate income tax based on salary and state"),
						Parameters: openai.FunctionParameters{
							"type": "object",
							"properties": map[string]interface{}{
								"income": map[string]interface{}{
									"type":        "number",
									"description": "Annual income in USD",
								},
								"state": map[string]interface{}{
									"type":        "string",
									"description": "State abbreviation (e.g., CA, NY)",
								},
							},
							"required": []string{"income", "state"},
						},
					},
				},
			},
		}

		transformedReq, err := adapter.TransformCompletionsRequest(originalReq)
		require.NoError(t, err)
		assert.Empty(t, transformedReq.Tools, "Tools should be removed from transformed request")
		// With no system message, we modify the first user message
		assert.Len(t, transformedReq.Messages, 1, "Should modify existing message, not add new one")
		// Verify tools were removed
		assert.Empty(t, transformedReq.Tools, "Tools should be removed from transformed request")

		// 2. Simulate LLM response with function call
		mockLLMResponse := openai.ChatCompletion{
			ID:    "test-completion",
			Model: "gpt-4",
			Choices: []openai.ChatCompletionChoice{
				{
					Index: 0,
					Message: openai.ChatCompletionMessage{
						Role:    "assistant",
						Content: `I'll calculate the tax for you. [{"name": "calculate_tax", "parameters": {"income": 50000, "state": "CA"}}]`,
					},
					FinishReason: "stop",
				},
			},
		}

		// 3. Transform response to extract tool calls
		finalResponse, err := adapter.TransformCompletionsResponse(mockLLMResponse)
		require.NoError(t, err)
		assert.Empty(t, finalResponse.Choices[0].Message.Content, "Content should be cleared when tool calls present")
		assert.Len(t, finalResponse.Choices[0].Message.ToolCalls, 1, "Should extract one tool call")

		toolCall := finalResponse.Choices[0].Message.ToolCalls[0]
		assert.Equal(t, "calculate_tax", toolCall.Function.Name)
		assert.Contains(t, toolCall.ID, "call_")
		assert.Equal(t, "tool_calls", finalResponse.Choices[0].FinishReason)

		// Verify parameters are correct JSON
		var params map[string]interface{}
		err = json.Unmarshal([]byte(toolCall.Function.Arguments), &params)
		require.NoError(t, err)
		assert.Equal(t, float64(50000), params["income"])
		assert.Equal(t, "CA", params["state"])
	})

	t.Run("MultipleToolCallsWorkflow", func(t *testing.T) {
		adapter := tooladapter.New(tooladapter.WithToolPolicy(tooladapter.ToolDrainAll))

		// Request with multiple tools
		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Get weather for Seattle and calculate travel time from there to Portland"),
			},
			Model: openai.ChatModelGPT4o,
			Tools: []openai.ChatCompletionToolParam{
				{
					Type: "function",
					Function: openai.FunctionDefinitionParam{
						Name:        "get_weather",
						Description: openai.String("Get current weather for a city"),
						Parameters: openai.FunctionParameters{
							"type": "object",
							"properties": map[string]interface{}{
								"city": map[string]interface{}{
									"type":        "string",
									"description": "City name",
								},
							},
						},
					},
				},
				{
					Type: "function",
					Function: openai.FunctionDefinitionParam{
						Name:        "calculate_travel_time",
						Description: openai.String("Calculate travel time between two cities"),
						Parameters: openai.FunctionParameters{
							"type": "object",
							"properties": map[string]interface{}{
								"from_city": map[string]interface{}{
									"type": "string",
								},
								"to_city": map[string]interface{}{
									"type": "string",
								},
							},
						},
					},
				},
			},
		}

		transformedReq, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)
		// Verify transformation occurred
		// Should modify existing message, not add new one
		assert.Len(t, transformedReq.Messages, 1, "Should modify existing message, not add new one")
		assert.Empty(t, transformedReq.Tools, "Tools should be removed from transformed request")

		// Response with multiple function calls
		mockResp := openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Content: `[{"name": "get_weather", "parameters": {"city": "Seattle"}}, {"name": "calculate_travel_time", "parameters": {"from_city": "Seattle", "to_city": "Portland"}}]`,
					},
				},
			},
		}

		finalResp, err := adapter.TransformCompletionsResponse(mockResp)
		require.NoError(t, err)
		assert.Len(t, finalResp.Choices[0].Message.ToolCalls, 2, "Should extract both tool calls")

		// Verify both tool calls
		toolCalls := finalResp.Choices[0].Message.ToolCalls
		assert.Equal(t, "get_weather", toolCalls[0].Function.Name)
		assert.Equal(t, "calculate_travel_time", toolCalls[1].Function.Name)
	})

	t.Run("MetricsIntegration", func(t *testing.T) {
		// Test that metrics are properly emitted throughout the workflow
		var capturedMetrics []tooladapter.MetricEventData
		var metricsMux sync.Mutex

		metricsCallback := func(data tooladapter.MetricEventData) {
			metricsMux.Lock()
			defer metricsMux.Unlock()
			capturedMetrics = append(capturedMetrics, data)
		}

		adapter := tooladapter.New(
			tooladapter.WithMetricsCallback(metricsCallback),
			tooladapter.WithLogLevel(slog.LevelDebug),
		)

		// Perform request transformation
		req := createIntegrationTestRequest()
		_, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// Perform response transformation
		mockResp := openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Content: `[{"name": "test_func", "parameters": {"test": "value"}}]`,
					},
				},
			},
		}
		_, err = adapter.TransformCompletionsResponse(mockResp)
		require.NoError(t, err)

		// Verify metrics were captured
		metricsMux.Lock()
		defer metricsMux.Unlock()

		assert.GreaterOrEqual(t, len(capturedMetrics), 2, "Should have at least transformation and detection metrics")

		// Check for specific metric types
		hasTransformation := false
		hasDetection := false

		for _, metric := range capturedMetrics {
			switch metric.EventType() {
			case tooladapter.MetricEventToolTransformation:
				hasTransformation = true
				if transformData, ok := metric.(tooladapter.ToolTransformationData); ok {
					assert.Greater(t, transformData.ToolCount, 0)
					assert.NotEmpty(t, transformData.ToolNames)
					assert.Greater(t, transformData.PromptLength, 0)
				}
			case tooladapter.MetricEventFunctionCallDetection:
				hasDetection = true
				if detectionData, ok := metric.(tooladapter.FunctionCallDetectionData); ok {
					assert.Greater(t, detectionData.FunctionCount, 0)
					assert.NotEmpty(t, detectionData.FunctionNames)
					assert.Greater(t, detectionData.ContentLength, 0)
				}
			}
		}

		assert.True(t, hasTransformation, "Should have transformation metrics")
		assert.True(t, hasDetection, "Should have detection metrics")
	})

	t.Run("ErrorRecoveryWorkflow", func(t *testing.T) {
		adapter := tooladapter.New()

		// Test graceful handling of various error conditions in sequence
		errorScenarios := []struct {
			name            string
			response        openai.ChatCompletion
			expectToolCalls bool
		}{
			{
				name: "ValidThenInvalid",
				response: openai.ChatCompletion{
					Choices: []openai.ChatCompletionChoice{
						{
							Message: openai.ChatCompletionMessage{
								Content: `[{"name": "valid_func", "parameters": {"test": "value"}}] followed by invalid JSON {`,
							},
						},
					},
				},
				expectToolCalls: true, // Should extract the valid part
			},
			{
				name: "InvalidThenValid",
				response: openai.ChatCompletion{
					Choices: []openai.ChatCompletionChoice{
						{
							Message: openai.ChatCompletionMessage{
								Content: `Let me help you with that. [{"name": "valid_func", "parameters": {"test": "value"}}]`,
							},
						},
					},
				},
				expectToolCalls: true,
			},
			{
				name: "CompletelyInvalid",
				response: openai.ChatCompletion{
					Choices: []openai.ChatCompletionChoice{
						{
							Message: openai.ChatCompletionMessage{
								Content: `This is just plain text with no JSON at all`,
							},
						},
					},
				},
				expectToolCalls: false,
			},
		}

		for _, scenario := range errorScenarios {
			t.Run(scenario.name, func(t *testing.T) {
				result, err := adapter.TransformCompletionsResponse(scenario.response)
				require.NoError(t, err, "Should handle error scenario gracefully")

				if scenario.expectToolCalls {
					assert.NotEmpty(t, result.Choices[0].Message.ToolCalls, "Should extract tool calls when possible")
				} else {
					assert.Empty(t, result.Choices[0].Message.ToolCalls, "Should not extract invalid tool calls")
					assert.Equal(t, scenario.response.Choices[0].Message.Content,
						result.Choices[0].Message.Content, "Should preserve original content when no valid JSON")
				}
			})
		}
	})

	t.Run("ContextPropagation", func(t *testing.T) {
		// Test that context is properly propagated through all operations
		adapter := tooladapter.New()

		// Create context with timeout that should not be triggered
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		req := createIntegrationTestRequest()

		// Test request transformation with context
		transformedReq, err := adapter.TransformCompletionsRequestWithContext(ctx, req)
		require.NoError(t, err)
		assert.NotEqual(t, req, transformedReq)

		// Test response transformation with context
		mockResp := openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Content: `[{"name": "test_func", "parameters": {"test": "value"}}]`,
					},
				},
			},
		}

		finalResp, err := adapter.TransformCompletionsResponseWithContext(ctx, mockResp)
		require.NoError(t, err)
		assert.NotEmpty(t, finalResp.Choices[0].Message.ToolCalls)
	})

	t.Run("BackwardCompatibilityMethods", func(t *testing.T) {
		// Ensure backward compatibility methods still work
		adapter := tooladapter.New()

		req := createIntegrationTestRequest()

		// Test non-context methods
		result1, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		result2, err := adapter.TransformCompletionsRequestWithContext(context.Background(), req)
		require.NoError(t, err)

		// Results should be equivalent
		assert.Equal(t, result1.Messages, result2.Messages)
		assert.Equal(t, result1.Tools, result2.Tools)

		// Test response methods
		mockResp := openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Content: `[{"name": "test_func", "parameters": {}}]`,
					},
				},
			},
		}

		resp1, err := adapter.TransformCompletionsResponse(mockResp)
		require.NoError(t, err)

		resp2, err := adapter.TransformCompletionsResponseWithContext(context.Background(), mockResp)
		require.NoError(t, err)

		// Results should be equivalent
		assert.Equal(t, len(resp1.Choices[0].Message.ToolCalls),
			len(resp2.Choices[0].Message.ToolCalls))
	})
}

// TestRealWorldUsagePatterns tests patterns commonly seen in production applications
func TestRealWorldUsagePatterns(t *testing.T) {
	t.Run("ChatbotWithMultipleCapabilities", func(t *testing.T) {
		// Simulate a chatbot with weather, calendar, and email capabilities
		adapter := tooladapter.New(
			tooladapter.WithLogLevel(slog.LevelInfo),
			tooladapter.WithToolPolicy(tooladapter.ToolDrainAll), // Use ToolDrainAll to get all tool calls
		)

		tools := []openai.ChatCompletionToolParam{
			createToolParam("get_weather", "Get weather information", map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"location": map[string]interface{}{
						"type": "string", "description": "City or location",
					},
				},
			}),
			createToolParam("schedule_meeting", "Schedule a meeting", map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"title":     map[string]interface{}{"type": "string"},
					"datetime":  map[string]interface{}{"type": "string"},
					"attendees": map[string]interface{}{"type": "array"},
				},
			}),
			createToolParam("send_email", "Send an email", map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"to":      map[string]interface{}{"type": "string"},
					"subject": map[string]interface{}{"type": "string"},
					"body":    map[string]interface{}{"type": "string"},
				},
			}),
		}

		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Check the weather in New York and schedule a meeting for tomorrow at 2pm"),
			},
			Model: openai.ChatModelGPT4o,
			Tools: tools,
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// Verify transformation modified the first message
		// With no system message, first user message is modified
		assert.Len(t, result.Messages, 1, "Should modify existing message, not add new one")
		assert.Empty(t, result.Tools, "Tools should be removed from request")

		// Simulate LLM calling multiple tools
		mockResp := openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Content: `I'll help you with that. Let me check the weather and schedule your meeting.

[{"name": "get_weather", "parameters": {"location": "New York"}}, {"name": "schedule_meeting", "parameters": {"title": "Meeting", "datetime": "tomorrow 2pm", "attendees": []}}]`,
					},
				},
			},
		}

		finalResp, err := adapter.TransformCompletionsResponse(mockResp)
		require.NoError(t, err)
		assert.Len(t, finalResp.Choices[0].Message.ToolCalls, 2)
	})

	t.Run("HighVolumeProcessing", func(t *testing.T) {
		// Simulate high-volume processing scenario
		adapter := tooladapter.New()
		const iterations = 100

		// Process many requests quickly
		for i := 0; i < iterations; i++ {
			req := openai.ChatCompletionNewParams{
				Messages: []openai.ChatCompletionMessageParamUnion{
					openai.UserMessage(fmt.Sprintf("Process item %d", i)),
				},
				Model: openai.ChatModelGPT4o,
				Tools: []openai.ChatCompletionToolParam{
					createToolParam("process_item", "Process an item", map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"id": map[string]interface{}{"type": "number"},
						},
					}),
				},
			}

			result, err := adapter.TransformCompletionsRequest(req)
			require.NoError(t, err)
			assert.NotEmpty(t, result.Messages)

			// Simulate response processing
			mockResp := openai.ChatCompletion{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Content: fmt.Sprintf(`[{"name": "process_item", "parameters": {"id": %d}}]`, i),
						},
					},
				},
			}

			finalResp, err := adapter.TransformCompletionsResponse(mockResp)
			require.NoError(t, err)
			assert.Len(t, finalResp.Choices[0].Message.ToolCalls, 1)
		}
	})

	t.Run("DynamicToolConfiguration", func(t *testing.T) {
		// Test scenario where different requests use different tool sets
		toolSets := [][]openai.ChatCompletionToolParam{
			// Basic tools
			{createToolParam("basic_func", "Basic function", nil)},
			// Extended tools
			{
				createToolParam("func1", "Function 1", nil),
				createToolParam("func2", "Function 2", nil),
				createToolParam("func3", "Function 3", nil),
			},
			// No tools
			{},
		}

		adapter := tooladapter.New()

		for i, tools := range toolSets {
			req := openai.ChatCompletionNewParams{
				Messages: []openai.ChatCompletionMessageParamUnion{
					openai.UserMessage(fmt.Sprintf("Request %d", i)),
				},
				Model: openai.ChatModelGPT4o,
				Tools: tools,
			}

			result, err := adapter.TransformCompletionsRequest(req)
			require.NoError(t, err)

			if len(tools) == 0 {
				// No tools means no transformation
				assert.Equal(t, req, result)
			} else {
				assert.NotEqual(t, req, result)
				assert.Empty(t, result.Tools)
			}
		}
	})
}

// Helper functions for integration tests
func createIntegrationTestRequest() openai.ChatCompletionNewParams {
	return openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("Test message for integration"),
		},
		Model: openai.ChatModelGPT4o,
		Tools: []openai.ChatCompletionToolParam{
			{
				Type: "function",
				Function: openai.FunctionDefinitionParam{
					Name:        "test_func",
					Description: openai.String("Test function for integration"),
					Parameters: openai.FunctionParameters{
						"type": "object",
						"properties": map[string]interface{}{
							"test": map[string]interface{}{
								"type":        "string",
								"description": "Test parameter",
							},
						},
					},
				},
			},
		},
	}
}

func createToolParam(name, description string, parameters map[string]interface{}) openai.ChatCompletionToolParam {
	return openai.ChatCompletionToolParam{
		Type: "function",
		Function: openai.FunctionDefinitionParam{
			Name:        name,
			Description: openai.String(description),
			Parameters:  parameters,
		},
	}
}
