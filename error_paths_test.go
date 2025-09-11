package tooladapter_test

import (
	"context"
	"fmt"
	"strings"
	"testing"

	tooladapter "github.com/juburr/openai-tool-adapter"
	"github.com/openai/openai-go/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestErrorPaths focuses on error conditions and failure modes
func TestErrorPaths(t *testing.T) {
	t.Run("UUIDGeneration", func(t *testing.T) {
		adapter := tooladapter.New()

		t.Run("GenerateToolCallID_NormalOperation", func(t *testing.T) {
			// Test normal operation
			id1 := adapter.GenerateToolCallID()
			id2 := adapter.GenerateToolCallID()

			assert.NotEmpty(t, id1)
			assert.NotEmpty(t, id2)
			assert.NotEqual(t, id1, id2, "Should generate unique IDs")
			assert.True(t, strings.HasPrefix(id1, "call_"), "Should have correct prefix")
			assert.True(t, strings.HasPrefix(id2, "call_"), "Should have correct prefix")
		})

		t.Run("GenerateToolCallID_MultipleCallsUnique", func(t *testing.T) {
			// Test that we get unique IDs across many calls
			ids := make(map[string]bool)
			for i := 0; i < 1000; i++ {
				id := adapter.GenerateToolCallID()
				assert.False(t, ids[id], "ID %s should be unique", id)
				ids[id] = true
			}
		})
	})

	t.Run("TemplateValidation", func(t *testing.T) {
		t.Run("ValidTemplateFormats", func(t *testing.T) {
			validTemplates := []string{
				"Simple template with %s placeholder",
				"Template at start: %s",
				"%s at the beginning",
				"Multiple words before %s and after",
				"   %s   ", // With whitespace
			}

			for _, template := range validTemplates {
				adapter := tooladapter.New(tooladapter.WithCustomPromptTemplate(template))
				require.NotNil(t, adapter)

				// Test that valid template works
				req := createMockRequestForErrorTests()
				result, err := adapter.TransformCompletionsRequest(req)
				require.NoError(t, err)
				assert.NotEqual(t, req, result, "Valid template should work")
			}
		})

		t.Run("InvalidTemplateFormats", func(t *testing.T) {
			invalidTemplates := []string{
				"No placeholder at all",
				"Multiple %s placeholders %s here",
				"Wrong placeholder type %d here",
				"",
				"   ", // Only whitespace
			}

			for _, template := range invalidTemplates {
				// Should create adapter without error but use default template
				adapter := tooladapter.New(tooladapter.WithCustomPromptTemplate(template))
				require.NotNil(t, adapter, "Should create adapter even with invalid template")

				// Should still work by falling back to default
				req := createMockRequestForErrorTests()
				result, err := adapter.TransformCompletionsRequest(req)
				require.NoError(t, err, "Should fallback to default template for: %s", template)
				assert.NotEqual(t, req, result)
			}
		})
	})

	t.Run("JSONProcessingErrors", func(t *testing.T) {
		t.Run("UnmarshalableParameters", func(t *testing.T) {
			adapter := tooladapter.New()

			// Create tool with parameters that might cause JSON issues
			circularRef := map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"data": "this should be an object, not a string",
				},
			}

			req := openai.ChatCompletionNewParams{
				Messages: []openai.ChatCompletionMessageParamUnion{
					openai.UserMessage("Test message"),
				},
				Model: openai.ChatModelGPT4o,
				Tools: []openai.ChatCompletionToolUnionParam{
					openai.ChatCompletionFunctionTool(openai.FunctionDefinitionParam{
						Name:        "test_func",
						Description: openai.String("Test function"),
						Parameters:  circularRef,
					}),
				},
			}

			// Should handle potentially problematic parameters gracefully
			result, err := adapter.TransformCompletionsRequest(req)
			require.NoError(t, err)
			assert.NotEqual(t, req, result)
		})

		t.Run("ResponseWithInvalidJSON", func(t *testing.T) {
			adapter := tooladapter.New()

			invalidJSONResponses := []string{
				`{"name": "func" "parameters": {}}`,                // Missing comma
				`[{"name": "func", "parameters": undefined}]`,      // JavaScript undefined
				`[{"name": "func", "parameters": null,}]`,          // Trailing comma
				`{"name": "func", "parameters": {'key': 'value'}}`, // Single quotes
				`[{"name": "func", parameters: {}}]`,               // Unquoted key
				`{"name": "func", "parameters": {key: value}}`,     // No quotes on key/value
			}

			for _, invalidJSON := range invalidJSONResponses {
				mockResp := openai.ChatCompletion{
					Choices: []openai.ChatCompletionChoice{
						{
							Message: openai.ChatCompletionMessage{
								Content: invalidJSON,
							},
						},
					},
				}

				// Should handle invalid JSON gracefully
				result, err := adapter.TransformCompletionsResponse(mockResp)
				require.NoError(t, err, "Should handle invalid JSON: %s", invalidJSON)
				assert.Equal(t, invalidJSON, result.Choices[0].Message.Content,
					"Should preserve original content when JSON is invalid")
			}
		})
	})

	t.Run("BoundaryConditions", func(t *testing.T) {
		t.Run("EmptyInputs", func(t *testing.T) {
			adapter := tooladapter.New()

			// Test with empty request
			emptyReq := openai.ChatCompletionNewParams{
				Messages: []openai.ChatCompletionMessageParamUnion{},
				Model:    openai.ChatModelGPT4o,
				Tools:    []openai.ChatCompletionToolUnionParam{},
			}

			result, err := adapter.TransformCompletionsRequest(emptyReq)
			require.NoError(t, err)
			assert.Equal(t, emptyReq, result, "Empty tools should return unchanged request")

			// Test with empty response
			emptyResp := openai.ChatCompletion{
				Choices: []openai.ChatCompletionChoice{},
			}

			respResult, err := adapter.TransformCompletionsResponse(emptyResp)
			require.NoError(t, err)
			assert.Equal(t, emptyResp, respResult, "Empty response should return unchanged")
		})

		t.Run("NilInputHandling", func(t *testing.T) {
			adapter := tooladapter.New()

			// Test response with nil/empty content
			nilContentResp := openai.ChatCompletion{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Content: "",
						},
					},
				},
			}

			result, err := adapter.TransformCompletionsResponse(nilContentResp)
			require.NoError(t, err)
			assert.Equal(t, nilContentResp, result, "Empty content should return unchanged")
		})

		t.Run("ExtremeParameterValues", func(t *testing.T) {
			adapter := tooladapter.New()

			// Test with tool that has nil parameters
			req := openai.ChatCompletionNewParams{
				Messages: []openai.ChatCompletionMessageParamUnion{
					openai.UserMessage("Test message"),
				},
				Model: openai.ChatModelGPT4o,
				Tools: []openai.ChatCompletionToolUnionParam{
					openai.ChatCompletionFunctionTool(openai.FunctionDefinitionParam{
						Name:        "nil_params_func",
						Description: openai.String("Function with nil parameters"),
						Parameters:  nil, // Nil parameters
					}),
				},
			}

			result, err := adapter.TransformCompletionsRequest(req)
			require.NoError(t, err)
			// Verify transformation occurred and first message was modified
			assert.Len(t, result.Messages, 1, "Should modify existing message, not add new one")
			assert.Empty(t, result.Tools, "Tools should be removed from result")

			// Test response with null parameters
			nullParamResp := openai.ChatCompletion{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Content: `[{"name": "func", "parameters": null}]`,
						},
					},
				},
			}

			respResult, err := adapter.TransformCompletionsResponse(nullParamResp)
			require.NoError(t, err)
			assert.Len(t, respResult.Choices[0].Message.ToolCalls, 1)
			assert.Equal(t, "null", respResult.Choices[0].Message.ToolCalls[0].Function.Arguments)
		})
	})

	t.Run("ContextCancellationScenarios", func(t *testing.T) {
		t.Run("CancellationDuringToolPromptBuild", func(t *testing.T) {
			adapter := tooladapter.New()

			// Create context that's already cancelled
			ctx, cancel := context.WithCancel(context.Background())
			cancel()

			req := createMockRequestForErrorTests()
			_, err := adapter.TransformCompletionsRequestWithContext(ctx, req)
			assert.Equal(t, context.Canceled, err)
		})

		t.Run("CancellationDuringResponseProcessing", func(t *testing.T) {
			adapter := tooladapter.New()

			// Create context that's already cancelled
			ctx, cancel := context.WithCancel(context.Background())
			cancel()

			mockResp := openai.ChatCompletion{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Content: `[{"name": "func", "parameters": {}}]`,
						},
					},
				},
			}

			_, err := adapter.TransformCompletionsResponseWithContext(ctx, mockResp)
			assert.Equal(t, context.Canceled, err)
		})
	})

	t.Run("FunctionNameValidationEdgeCases", func(t *testing.T) {
		t.Run("ValidFunctionNames", func(t *testing.T) {
			validNames := []string{
				"simple",
				"with_underscore",
				"with-dash",
				"withNumbers123",
				"a",                     // Single character
				strings.Repeat("a", 64), // Max length
				"server.function",       // MCP format
				"a1b2c3.test_func-123",  // Complex MCP format
			}

			for _, name := range validNames {
				err := tooladapter.ValidateFunctionName(name)
				assert.NoError(t, err, "Name should be valid: %s", name)
			}
		})

		t.Run("InvalidFunctionNames", func(t *testing.T) {
			invalidNames := []string{
				"",                      // Empty
				strings.Repeat("a", 65), // Too long
				"with spaces",           // Spaces not allowed
				"with@symbol",           // Special characters
				"with.multiple.periods", // Multiple periods
				".function",             // Empty prefix
				"server.",               // Empty function part
				"with#hash",             // Hash symbol not allowed
			}

			for _, name := range invalidNames {
				err := tooladapter.ValidateFunctionName(name)
				assert.Error(t, err, "Name should be invalid: %s", name)
				assert.Contains(t, err.Error(), "function name validation failed")
			}
		})
	})

	t.Run("ResourceLimits", func(t *testing.T) {
		t.Run("VeryLongDescriptions", func(t *testing.T) {
			adapter := tooladapter.New()

			// Create tool with very long description
			longDescription := strings.Repeat("This is a very long description. ", 1000) // ~34KB description

			req := openai.ChatCompletionNewParams{
				Messages: []openai.ChatCompletionMessageParamUnion{
					openai.UserMessage("Test with long description"),
				},
				Model: openai.ChatModelGPT4o,
				Tools: []openai.ChatCompletionToolUnionParam{
					openai.ChatCompletionFunctionTool(openai.FunctionDefinitionParam{
						Name:        "long_desc_func",
						Description: openai.String(longDescription),
					}),
				},
			}

			result, err := adapter.TransformCompletionsRequest(req)
			require.NoError(t, err)
			// Verify transformation occurred and first message was modified
			assert.Len(t, result.Messages, 1, "Should modify existing message, not add new one")
			assert.Empty(t, result.Tools, "Tools should be removed from result")
		})

		t.Run("ManyMessagesInRequest", func(t *testing.T) {
			adapter := tooladapter.New()

			// Create request with many existing messages
			messages := make([]openai.ChatCompletionMessageParamUnion, 100)
			for i := 0; i < 100; i++ {
				messages[i] = openai.UserMessage(fmt.Sprintf("Message %d", i))
			}

			req := openai.ChatCompletionNewParams{
				Messages: messages,
				Model:    openai.ChatModelGPT4o,
				Tools: []openai.ChatCompletionToolUnionParam{
					openai.ChatCompletionFunctionTool(openai.FunctionDefinitionParam{
						Name:        "test_func",
						Description: openai.String("Test function"),
					}),
				},
			}

			result, err := adapter.TransformCompletionsRequest(req)
			require.NoError(t, err)
			// Modifies first message when no system exists
			assert.Len(t, result.Messages, 100, "Should modify existing messages, not add new one")
		})
	})
}

// Helper function to create a mock request with tools (for error path tests)
func createMockRequestForErrorTests() openai.ChatCompletionNewParams {
	return openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("Test message"),
		},
		Model: openai.ChatModelGPT4o,
		Tools: []openai.ChatCompletionToolUnionParam{
			openai.ChatCompletionFunctionTool(openai.FunctionDefinitionParam{
				Name:        "test_function",
				Description: openai.String("A test function"),
				Parameters: openai.FunctionParameters{
					"type": "object",
					"properties": map[string]interface{}{
						"message": map[string]interface{}{
							"type":        "string",
							"description": "The message to process",
						},
					},
					"required": []string{"message"},
				},
			}),
		},
	}
}
