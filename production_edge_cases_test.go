package tooladapter_test

import (
	"context"
	"fmt"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/juburr/openai-tool-adapter"
	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestProductionEdgeCases covers critical edge cases that could occur in production
func TestProductionEdgeCases(t *testing.T) {
	t.Run("ResourceExhaustion", func(t *testing.T) {
		t.Run("ExtremelyLargeToolCount", func(t *testing.T) {
			adapter := tooladapter.New()

			// Create request with many tools to test memory usage
			const toolCount = 1000
			tools := make([]openai.ChatCompletionToolParam, toolCount)
			for i := 0; i < toolCount; i++ {
				tools[i] = openai.ChatCompletionToolParam{
					Type: "function",
					Function: openai.FunctionDefinitionParam{
						Name:        fmt.Sprintf("function_%d", i),
						Description: openai.String(fmt.Sprintf("Function number %d", i)),
						Parameters: map[string]interface{}{
							"type": "object",
							"properties": map[string]interface{}{
								"param": map[string]interface{}{
									"type":        "string",
									"description": "Parameter",
								},
							},
						},
					},
				}
			}

			req := openai.ChatCompletionNewParams{
				Messages: []openai.ChatCompletionMessageParamUnion{
					openai.UserMessage("Test with many tools"),
				},
				Model: openai.ChatModelGPT4o,
				Tools: tools,
			}

			// Should handle large number of tools gracefully
			result, err := adapter.TransformCompletionsRequest(req)
			require.NoError(t, err)
			// With no system message, a system instruction is prepended
			assert.Len(t, result.Messages, 1, "Should modify existing message, not add new one")
			assert.Empty(t, result.Tools, "Tools should be removed from result")
		})

		t.Run("ExtremelyLongToolNames", func(t *testing.T) {
			adapter := tooladapter.New()

			// Create tool with very long name (but within limits)
			longName := strings.Repeat("a", 64) // Max allowed length
			req := openai.ChatCompletionNewParams{
				Messages: []openai.ChatCompletionMessageParamUnion{
					openai.UserMessage("Test with long tool name"),
				},
				Model: openai.ChatModelGPT4o,
				Tools: []openai.ChatCompletionToolParam{
					{
						Type: "function",
						Function: openai.FunctionDefinitionParam{
							Name:        longName,
							Description: openai.String("Test function"),
						},
					},
				},
			}

			result, err := adapter.TransformCompletionsRequest(req)
			require.NoError(t, err)
			// System instruction is prepended when no system exists
			assert.Len(t, result.Messages, 1, "Should modify existing message, not add new one")
			assert.Empty(t, result.Tools, "Tools should be removed from result")
		})

		t.Run("VeryLargeParameterSchemas", func(t *testing.T) {
			adapter := tooladapter.New()

			// Create tool with very large parameter schema
			largeSchema := map[string]interface{}{
				"type":       "object",
				"properties": map[string]interface{}{},
			}

			// Add many properties
			properties := largeSchema["properties"].(map[string]interface{})
			for i := 0; i < 100; i++ {
				properties[fmt.Sprintf("param_%d", i)] = map[string]interface{}{
					"type":        "string",
					"description": fmt.Sprintf("Parameter %d with a very long description that could potentially cause issues with memory or parsing", i),
				}
			}

			req := openai.ChatCompletionNewParams{
				Messages: []openai.ChatCompletionMessageParamUnion{
					openai.UserMessage("Test with large schema"),
				},
				Model: openai.ChatModelGPT4o,
				Tools: []openai.ChatCompletionToolParam{
					{
						Type: "function",
						Function: openai.FunctionDefinitionParam{
							Name:        "large_schema_function",
							Description: openai.String("Function with large schema"),
							Parameters:  largeSchema,
						},
					},
				},
			}

			result, err := adapter.TransformCompletionsRequest(req)
			require.NoError(t, err)
			// System instruction is prepended when no system exists
			assert.Len(t, result.Messages, 1, "Should modify existing message, not add new one")
			assert.Empty(t, result.Tools, "Tools should be removed from result")
		})
	})

	t.Run("MaliciousInputHandling", func(t *testing.T) {
		t.Run("DeeplyNestedJSON", func(t *testing.T) {
			adapter := tooladapter.New()

			// Create deeply nested JSON response that could cause parser issues
			depth := 50
			nestedJSON := "["
			for i := 0; i < depth; i++ {
				nestedJSON += fmt.Sprintf(`{"name": "func_%d", "parameters": {"nested": `, i)
			}
			nestedJSON += `"value"`
			for i := 0; i < depth; i++ {
				nestedJSON += "}}"
			}
			nestedJSON += "]"

			mockResp := openai.ChatCompletion{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Content: nestedJSON,
						},
					},
				},
			}

			// Should handle deeply nested JSON gracefully
			result, err := adapter.TransformCompletionsResponse(mockResp)
			require.NoError(t, err)

			// Either it parses successfully or falls back to original content
			assert.NotNil(t, result)
		})

		t.Run("ExtremelyLongJSONStrings", func(t *testing.T) {
			adapter := tooladapter.New()

			// Create JSON with extremely long string values
			longString := strings.Repeat("a", 10000)
			jsonContent := fmt.Sprintf(`[{"name": "test_func", "parameters": {"long_param": "%s"}}]`, longString)

			mockResp := openai.ChatCompletion{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Content: jsonContent,
						},
					},
				},
			}

			result, err := adapter.TransformCompletionsResponse(mockResp)
			require.NoError(t, err)
			assert.NotNil(t, result)
		})

		t.Run("MalformedJSONWithGoodStructure", func(t *testing.T) {
			adapter := tooladapter.New()

			// JSON that looks like function calls but has subtle errors
			malformedCases := []string{
				`[{"name": "func", "parameters": {"key": "value"}}`,   // Missing closing bracket
				`{"name": "func", "parameters": {"key": "value"]`,     // Wrong closing bracket
				`[{"name": "func", "parameters": {"key": value}}]`,    // Unquoted value
				`[{"name": "func", "parameters": {"key": "value",}}]`, // Trailing comma
			}

			for _, malformed := range malformedCases {
				mockResp := openai.ChatCompletion{
					Choices: []openai.ChatCompletionChoice{
						{
							Message: openai.ChatCompletionMessage{
								Content: malformed,
							},
						},
					},
				}

				// Should handle malformed JSON gracefully without crashing
				result, err := adapter.TransformCompletionsResponse(mockResp)
				require.NoError(t, err, "Should not error on malformed JSON: %s", malformed)
				assert.NotNil(t, result)

				// Should either parse what it can or return original
				assert.NotEmpty(t, result.Choices)
			}
		})
	})

	t.Run("ConcurrencyStress", func(t *testing.T) {
		t.Run("HighConcurrentRequestTransforms", func(t *testing.T) {
			adapter := tooladapter.New()
			const goroutineCount = 100
			const operationsPerGoroutine = 10

			var wg sync.WaitGroup
			errors := make(chan error, goroutineCount*operationsPerGoroutine)

			for i := 0; i < goroutineCount; i++ {
				wg.Add(1)
				go func(id int) {
					defer wg.Done()

					for j := 0; j < operationsPerGoroutine; j++ {
						req := openai.ChatCompletionNewParams{
							Messages: []openai.ChatCompletionMessageParamUnion{
								openai.UserMessage(fmt.Sprintf("Message from goroutine %d, operation %d", id, j)),
							},
							Model: openai.ChatModelGPT4o,
							Tools: []openai.ChatCompletionToolParam{
								{
									Type: "function",
									Function: openai.FunctionDefinitionParam{
										Name:        fmt.Sprintf("func_%d_%d", id, j),
										Description: openai.String("Test function"),
									},
								},
							},
						}

						_, err := adapter.TransformCompletionsRequest(req)
						if err != nil {
							errors <- err
						}
					}
				}(i)
			}

			wg.Wait()
			close(errors)

			// Check for any errors
			for err := range errors {
				t.Errorf("Concurrent operation failed: %v", err)
			}
		})

		t.Run("ConcurrentResponseTransforms", func(t *testing.T) {
			adapter := tooladapter.New()
			const goroutineCount = 50

			var wg sync.WaitGroup
			errors := make(chan error, goroutineCount)

			for i := 0; i < goroutineCount; i++ {
				wg.Add(1)
				go func(id int) {
					defer wg.Done()

					mockResp := openai.ChatCompletion{
						Choices: []openai.ChatCompletionChoice{
							{
								Message: openai.ChatCompletionMessage{
									Content: fmt.Sprintf(`[{"name": "func_%d", "parameters": {"id": %d}}]`, id, id),
								},
							},
						},
					}

					_, err := adapter.TransformCompletionsResponse(mockResp)
					if err != nil {
						errors <- err
					}
				}(i)
			}

			wg.Wait()
			close(errors)

			// Check for any errors
			for err := range errors {
				t.Errorf("Concurrent response transform failed: %v", err)
			}
		})
	})

	t.Run("ContextHandling", func(t *testing.T) {
		t.Run("RequestTransformWithTimeout", func(t *testing.T) {
			adapter := tooladapter.New()

			// Create a context with a very short timeout
			ctx, cancel := context.WithTimeout(context.Background(), 1*time.Millisecond)
			defer cancel()

			// Create a large request that might take time to process
			tools := make([]openai.ChatCompletionToolParam, 100)
			for i := 0; i < 100; i++ {
				tools[i] = openai.ChatCompletionToolParam{
					Type: "function",
					Function: openai.FunctionDefinitionParam{
						Name:        fmt.Sprintf("func_%d", i),
						Description: openai.String("Test function"),
					},
				}
			}

			req := openai.ChatCompletionNewParams{
				Messages: []openai.ChatCompletionMessageParamUnion{
					openai.UserMessage("Test with timeout"),
				},
				Model: openai.ChatModelGPT4o,
				Tools: tools,
			}

			// Add a small delay to make timeout more likely
			time.Sleep(2 * time.Millisecond)

			_, err := adapter.TransformCompletionsRequestWithContext(ctx, req)
			// Should either complete successfully or return context error
			if err != nil {
				assert.Equal(t, context.DeadlineExceeded, err)
			}
		})

		t.Run("ResponseTransformWithCancellation", func(t *testing.T) {
			adapter := tooladapter.New()

			ctx, cancel := context.WithCancel(context.Background())

			// Cancel immediately
			cancel()

			mockResp := openai.ChatCompletion{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Content: `[{"name": "test_func", "parameters": {}}]`,
						},
					},
				},
			}

			_, err := adapter.TransformCompletionsResponseWithContext(ctx, mockResp)
			assert.Equal(t, context.Canceled, err)
		})
	})
}

// TestMemoryUsagePatterns ensures the adapter doesn't leak memory under various conditions
func TestMemoryUsagePatterns(t *testing.T) {
	t.Run("BufferPoolReuse", func(t *testing.T) {
		adapter := tooladapter.New()

		// Force garbage collection before test
		runtime.GC()
		var m1, m2 runtime.MemStats
		runtime.ReadMemStats(&m1)

		// Perform many operations that should reuse buffers
		for i := 0; i < 1000; i++ {
			req := openai.ChatCompletionNewParams{
				Messages: []openai.ChatCompletionMessageParamUnion{
					openai.UserMessage("Test message"),
				},
				Model: openai.ChatModelGPT4o,
				Tools: []openai.ChatCompletionToolParam{
					{
						Type: "function",
						Function: openai.FunctionDefinitionParam{
							Name:        "test_func",
							Description: openai.String("Test function"),
						},
					},
				},
			}

			_, err := adapter.TransformCompletionsRequest(req)
			require.NoError(t, err)
		}

		// Check memory usage hasn't grown excessively
		runtime.GC()
		runtime.ReadMemStats(&m2)

		// Check memory usage is reasonable (handle potential underflow)
		var memoryGrowth uint64
		if m2.Alloc > m1.Alloc {
			memoryGrowth = m2.Alloc - m1.Alloc
		}
		// Memory growth should be reasonable (less than 50MB for 1000 operations)
		assert.Less(t, memoryGrowth, uint64(50*1024*1024),
			"Memory growth should be reasonable: %d bytes", memoryGrowth)
	})

	t.Run("JSONCandidatePoolReuse", func(t *testing.T) {
		adapter := tooladapter.New()

		// Force garbage collection before test
		runtime.GC()
		var m1, m2 runtime.MemStats
		runtime.ReadMemStats(&m1)

		// Perform many response transformations to test candidate pool
		for i := 0; i < 500; i++ {
			mockResp := openai.ChatCompletion{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Content: fmt.Sprintf(`[{"name": "func_%d", "parameters": {"test": "value"}}]`, i),
						},
					},
				},
			}

			_, err := adapter.TransformCompletionsResponse(mockResp)
			require.NoError(t, err)
		}

		// Check memory usage
		runtime.GC()
		runtime.ReadMemStats(&m2)

		// Check memory usage is reasonable (handle potential underflow)
		var memoryGrowth uint64
		if m2.Alloc > m1.Alloc {
			memoryGrowth = m2.Alloc - m1.Alloc
		}
		// Memory growth should be reasonable (less than 25MB for 500 operations)
		assert.Less(t, memoryGrowth, uint64(25*1024*1024),
			"Memory growth from JSON parsing should be reasonable: %d bytes", memoryGrowth)
	})
}
