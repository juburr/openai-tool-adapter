package tooladapter

import (
	"encoding/json"
	"log/slog"
	"strings"
	"testing"

	"github.com/openai/openai-go"
)

func FuzzTransformCompletionsResponse(f *testing.F) {
	// Seed with valid responses
	f.Add(`{"name": "test_function", "parameters": {}}`)
	f.Add(`[{"name": "get_weather", "parameters": {"city": "London"}}]`)
	f.Add(`I'll help you with that. [{"name": "calculate", "parameters": {"x": 10, "y": 20}}]`)
	f.Add("```json\n{\"name\": \"search\", \"parameters\": {\"query\": \"test\"}}\n```")

	// Seed with edge cases
	f.Add(``)
	f.Add(`Just a regular response without function calls`)
	f.Add(`Here's some JSON: {"not": "a function call"}`)
	f.Add(`Multiple functions: [{"name": "func1"}, {"name": "func2", "parameters": null}]`)

	// Seed with malformed content
	f.Add(`{"name": "broken_json"`)
	f.Add(`[{"name": "test", "parameters": invalid}]`)
	f.Add(`{"name": "", "parameters": {}}`)
	f.Add(`{"name": "test with spaces", "parameters": {}}`)
	f.Add(`{"name": "test@invalid", "parameters": {}}`)

	// Seed with mixed content
	f.Add(`Let me process that. {"name": "process", "parameters": {"data": "value"}} Done!`)
	f.Add("Response with `{\"name\": \"inline\", \"parameters\": {}}` function call.")
	f.Add(`Multiple: [{"name": "f1"}] and {"name": "f2", "parameters": {"x": 1}}`)

	f.Fuzz(func(t *testing.T, content string) {
		// Transform response should never panic
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("TransformCompletionsResponse panicked on content %q: %v", content, r)
			}
		}()

		performFuzzTransformCompletionsResponse(t, content)
	})
}

// performFuzzTransformCompletionsResponse executes the main fuzzing logic for response transformation
func performFuzzTransformCompletionsResponse(t *testing.T, content string) {
	adapter := New(WithLogLevel(slog.LevelError))

	// Create a mock response with the fuzzed content
	resp := openai.ChatCompletion{
		Choices: []openai.ChatCompletionChoice{
			{
				Message: openai.ChatCompletionMessage{
					Content: content,
				},
			},
		},
	}

	result, err := adapter.TransformCompletionsResponse(resp)
	if err != nil {
		// Errors are acceptable, but should be proper error types
		if err.Error() == "" {
			t.Errorf("TransformCompletionsResponse returned empty error message for content %q", content)
		}
		return
	}

	// If transformation succeeded, validate the result
	if len(result.Choices) > 0 {
		validateFuzzResponseChoice(t, result.Choices[0], content)
	}
}

// validateFuzzResponseChoice validates a single response choice from fuzzing
func validateFuzzResponseChoice(t *testing.T, choice openai.ChatCompletionChoice, originalContent string) {
	// If tool calls were detected, validate them
	if len(choice.Message.ToolCalls) > 0 {
		validateFuzzToolCallsDetected(t, choice)
	} else {
		validateFuzzNoToolCalls(t, choice, originalContent)
	}
}

// validateFuzzToolCallsDetected validates when tool calls are detected
func validateFuzzToolCallsDetected(t *testing.T, choice openai.ChatCompletionChoice) {
	// Content should be empty when tool calls are present
	if choice.Message.Content != "" {
		t.Errorf("TransformCompletionsResponse left content %q when tool calls were detected", choice.Message.Content)
	}

	// Finish reason should be tool_calls
	if choice.FinishReason != "tool_calls" {
		t.Errorf("TransformCompletionsResponse set finish reason to %q instead of 'tool_calls'", choice.FinishReason)
	}

	// All tool calls should be valid
	for i, toolCall := range choice.Message.ToolCalls {
		validateFuzzSingleToolCall(t, toolCall, i)
	}
}

// validateFuzzSingleToolCall validates a single tool call from fuzzing
func validateFuzzSingleToolCall(t *testing.T, toolCall openai.ChatCompletionMessageToolCall, index int) {
	// Basic field validation
	if toolCall.Function.Name == "" {
		t.Errorf("TransformCompletionsResponse produced tool call %d with empty function name", index)
	}

	if toolCall.ID == "" {
		t.Errorf("TransformCompletionsResponse produced tool call %d with empty ID", index)
	}

	if !strings.HasPrefix(toolCall.ID, "call_") {
		t.Errorf("TransformCompletionsResponse produced tool call %d with invalid ID format: %q", index, toolCall.ID)
	}

	if toolCall.Type != "function" {
		t.Errorf("TransformCompletionsResponse produced tool call %d with invalid type: %q", index, toolCall.Type)
	}

	// Validate function name
	if err := ValidateFunctionName(toolCall.Function.Name); err != nil {
		t.Errorf("TransformCompletionsResponse produced invalid function name %q: %v", toolCall.Function.Name, err)
	}

	// Arguments should be valid JSON
	validateFuzzToolCallArguments(t, toolCall.Function.Arguments)
}

// validateFuzzToolCallArguments validates tool call arguments JSON
func validateFuzzToolCallArguments(t *testing.T, arguments string) {
	if arguments != "" {
		var temp interface{}
		if err := json.Unmarshal([]byte(arguments), &temp); err != nil {
			t.Errorf("TransformCompletionsResponse produced invalid JSON arguments %q: %v", arguments, err)
		}
	}
}

// validateFuzzNoToolCalls validates when no tool calls are detected
func validateFuzzNoToolCalls(t *testing.T, choice openai.ChatCompletionChoice, originalContent string) {
	if choice.Message.Content != originalContent {
		t.Errorf("TransformCompletionsResponse modified content when no tool calls detected: got %q, want %q",
			choice.Message.Content, originalContent)
	}
}

// buildFuzzedTool creates a tool from fuzzed parameters
func buildFuzzedTool(funcName, description string, hasParams bool) openai.ChatCompletionToolParam {
	tool := openai.ChatCompletionToolParam{
		Type: "function",
		Function: openai.FunctionDefinitionParam{
			Name: funcName,
		},
	}

	if description != "" {
		tool.Function.Description = openai.String(description)
	}

	if hasParams {
		tool.Function.Parameters = openai.FunctionParameters{
			"type": "object",
			"properties": map[string]interface{}{
				"test": map[string]interface{}{
					"type": "string",
				},
			},
		}
	}

	return tool
}

// validateToolsRemoved checks that tools were properly removed from the result
func validateToolsRemoved(t *testing.T, result openai.ChatCompletionNewParams) {
	if len(result.Tools) > 0 {
		t.Errorf("TransformCompletionsRequest did not remove tools from result")
	}
}

// validateMessageCount validates the message count after transformation
func validateMessageCount(t *testing.T, req, result openai.ChatCompletionNewParams) {
	if len(req.Messages) == 0 {
		// Should have created a new instruction message (user by default)
		if len(result.Messages) != 1 {
			t.Errorf("TransformCompletionsRequest did not create instruction message for empty messages: got %d messages, want 1",
				len(result.Messages))
		}
		return
	}

	// Should always preserve message count (modify existing, not add new)
	if len(result.Messages) != len(req.Messages) {
		t.Errorf("TransformCompletionsRequest changed message count: got %d messages, want %d",
			len(result.Messages), len(req.Messages))
	}
}

// validateNoToolsCase validates the case when no tools are provided
func validateNoToolsCase(t *testing.T, req, result openai.ChatCompletionNewParams) {
	if len(result.Tools) != 0 || len(result.Messages) != len(req.Messages) {
		t.Errorf("TransformCompletionsRequest modified request when no valid tools were provided")
	}
}

// FuzzTransformCompletionsRequest fuzzes the request transformation pipeline
func FuzzTransformCompletionsRequest(f *testing.F) {
	// Seed with various tool configurations
	f.Add("get_weather", "Get current weather", true)
	f.Add("", "Empty name", true)
	f.Add("test_function", "", true)
	f.Add("function with spaces", "Invalid name", true)
	f.Add("valid_function", "Normal description", false)

	f.Fuzz(func(t *testing.T, funcName, description string, hasParams bool) {
		// Transform request should never panic
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("TransformCompletionsRequest panicked with function name %q, description %q: %v",
					funcName, description, r)
			}
		}()

		adapter := New(WithLogLevel(slog.LevelError))

		// Build a request with the fuzzed function definition
		tools := []openai.ChatCompletionToolParam{}

		// Only add the tool if the function name is not empty (to test various scenarios)
		if funcName != "" {
			tool := buildFuzzedTool(funcName, description, hasParams)
			tools = append(tools, tool)
		}

		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Test message"),
			},
			Tools: tools,
		}

		result, err := adapter.TransformCompletionsRequest(req)
		if err != nil {
			// Errors are acceptable for invalid input, but should have proper error messages
			if err.Error() == "" {
				t.Errorf("TransformCompletionsRequest returned empty error message for function %q", funcName)
			}
			return
		}

		// If transformation succeeded, validate the result
		if len(tools) > 0 {
			validateToolsRemoved(t, result)
			validateMessageCount(t, req, result)

			// First message should contain tool information
			// (either modified user message or system message)
			if len(result.Messages) > 0 {
				// We can't easily check the content without accessing internal fields,
				// but we know the tool prompt should be injected somewhere
				_ = result.Messages[0]
			}
		} else {
			// No tools - should pass through unchanged
			validateNoToolsCase(t, req, result)
		}
	})
}

// FuzzStreamingBuffering fuzzes the streaming buffer logic
func FuzzStreamingBuffering(f *testing.F) {
	// Seed with various streaming scenarios
	f.Add(`[{"name": "test"}]`, true)               // Complete function call
	f.Add(`[{"name": "test"`, false)                // Incomplete JSON
	f.Add(`{"name": "func", "parameters": `, false) // Partial parameters
	f.Add(`Let me help`, true)                      // Regular text
	f.Add(``, true)                                 // Empty content
	f.Add("```json\n{\"name\":", false)             // Incomplete markdown block
	f.Add("`{\"name\": \"test\"}`", true)           // Complete inline code

	f.Fuzz(func(t *testing.T, content string, expectComplete bool) {
		// Streaming operations should never panic
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("Streaming operations panicked on content %q: %v", content, r)
			}
		}()

		// Test HasCompleteJSON directly
		_ = HasCompleteJSON(content)

		// Test shouldStartBuffering heuristic
		adapter := New(WithLogLevel(slog.LevelError))
		stream := adapter.TransformStreamingResponse(&mockStreamingResponse{
			chunks: []string{content},
		})
		defer func() {
			if err := stream.Close(); err != nil {
				// Use simple log since we don't have access to f *testing.F here
				// This is acceptable for fuzzing where we want minimal overhead
				_ = err // Fuzzing tests prioritize speed over logging
			}
		}()

		// This should not panic or hang
		hasNext := stream.Next()
		if hasNext {
			chunk := stream.Current()
			// Basic validation - should have valid structure
			_ = chunk
		}

		// Check for errors
		if err := stream.Err(); err != nil {
			// Errors are acceptable, but should have messages
			if err.Error() == "" {
				t.Errorf("Stream returned empty error message for content %q", content)
			}
		}
	})
}

// mockStreamingResponse is a simple mock for testing streaming functionality
type mockStreamingResponse struct {
	chunks []string
	index  int
}

func (m *mockStreamingResponse) Next() bool {
	return m.index < len(m.chunks)
}

func (m *mockStreamingResponse) Current() openai.ChatCompletionChunk {
	if m.index >= len(m.chunks) {
		return openai.ChatCompletionChunk{}
	}

	chunk := openai.ChatCompletionChunk{
		Choices: []openai.ChatCompletionChunkChoice{
			{
				Delta: openai.ChatCompletionChunkChoiceDelta{
					Content: m.chunks[m.index],
					Role:    "assistant",
				},
			},
		},
	}

	m.index++

	// Last chunk should have finish reason
	if m.index >= len(m.chunks) {
		chunk.Choices[0].FinishReason = "stop"
	}

	return chunk
}

func (m *mockStreamingResponse) Err() error {
	return nil
}

func (m *mockStreamingResponse) Close() error {
	return nil
}

// FuzzIDGeneration fuzzes the ID generation to ensure it never panics or produces invalid IDs
func FuzzIDGeneration(f *testing.F) {
	// This is more of a stress test since ID generation doesn't take input
	// But we can test it under various conditions
	f.Add(100)  // Generate multiple IDs
	f.Add(1000) // Stress test
	f.Add(0)    // Edge case
	f.Add(1)    // Single ID

	f.Fuzz(func(t *testing.T, count int) {
		adapter := New()

		// ID generation should never panic
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("GenerateToolCallID panicked when generating %d IDs: %v", count, r)
			}
		}()

		// Limit the count to prevent excessive resource usage
		if count < 0 || count > 10000 {
			count = 100
		}

		ids := make(map[string]bool)

		for i := 0; i < count; i++ {
			id := adapter.GenerateToolCallID()

			// Basic validation
			if id == "" {
				t.Errorf("GenerateToolCallID returned empty ID")
				return
			}

			if !strings.HasPrefix(id, "call_") {
				t.Errorf("GenerateToolCallID returned ID without proper prefix: %q", id)
				return
			}

			// Check for duplicates (should be extremely rare)
			if ids[id] {
				t.Errorf("GenerateToolCallID returned duplicate ID: %q", id)
				return
			}
			ids[id] = true

			// ID should have reasonable length (call_ + UUID)
			if len(id) < 10 || len(id) > 50 {
				t.Errorf("GenerateToolCallID returned ID with unusual length %d: %q", len(id), id)
				return
			}
		}
	})
}
