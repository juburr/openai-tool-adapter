package tooladapter

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"testing"

	"github.com/openai/openai-go/v3"
)

// Benchmark sink variables to prevent compiler optimizations
var (
	benchRequestResult  openai.ChatCompletionNewParams
	benchResponseResult openai.ChatCompletion
	benchError          error
)

// BenchmarkTransformCompletionsRequest_EndToEnd benchmarks the complete request transformation workflow
func BenchmarkTransformCompletionsRequest_EndToEnd(b *testing.B) {
	// Use quiet options to reduce logging noise during benchmarks
	adapter := New(WithLogLevel(slog.LevelError))

	testCases := []struct {
		name  string
		tools []openai.ChatCompletionToolUnionParam
	}{
		{
			name:  "Tiny_1Tool",
			tools: createBenchmarkTools(1),
		},
		{
			name:  "Small_5Tools",
			tools: createBenchmarkTools(5),
		},
		{
			name:  "Medium_20Tools",
			tools: createBenchmarkTools(20),
		},
		{
			name:  "Large_50Tools",
			tools: createBenchmarkTools(50),
		},
	}

	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			req := openai.ChatCompletionNewParams{
				Model: openai.ChatModelGPT4o,
				Messages: []openai.ChatCompletionMessageParamUnion{
					openai.UserMessage("Hello, please help me with these tools."),
				},
				Tools: tc.tools,
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchRequestResult, benchError = adapter.TransformCompletionsRequest(req)
			}
		})
	}
}

// BenchmarkTransformCompletionsResponse_EndToEnd benchmarks the complete response transformation workflow
func BenchmarkTransformCompletionsResponse_EndToEnd(b *testing.B) {
	adapter := New(WithLogLevel(slog.LevelError))

	testCases := []struct {
		name     string
		response openai.ChatCompletion
	}{
		{
			name:     "Tiny_NoToolCalls",
			response: createBenchmarkResponse("Just a simple text response with no tool calls."),
		},
		{
			name:     "Small_SingleToolCall",
			response: createBenchmarkResponse(`[{"name": "get_weather", "parameters": {"location": "Boston", "unit": "celsius"}}]`),
		},
		{
			name:     "Medium_MultipleToolCalls",
			response: createBenchmarkResponse(createMultipleToolCallJSON(5)),
		},
		{
			name:     "Large_ManyToolCalls",
			response: createBenchmarkResponse(createMultipleToolCallJSON(20)),
		},
		{
			name:     "Large_ComplexToolCall",
			response: createBenchmarkResponse(createComplexToolCallJSON()),
		},
		{
			name:     "Large_MixedContent",
			response: createBenchmarkResponse(createMixedContentResponse()),
		},
	}

	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchResponseResult, benchError = adapter.TransformCompletionsResponse(tc.response)
			}
		})
	}
}

// BenchmarkFullWorkflow_EndToEnd benchmarks a complete request -> response cycle
func BenchmarkFullWorkflow_EndToEnd(b *testing.B) {
	adapter := New(WithLogLevel(slog.LevelError))

	testCases := []struct {
		name     string
		tools    []openai.ChatCompletionToolUnionParam
		response string
	}{
		{
			name:     "Tiny_Simple",
			tools:    createBenchmarkTools(2),
			response: `[{"name": "get_weather", "parameters": {"location": "NYC"}}]`,
		},
		{
			name:     "Small_Typical",
			tools:    createBenchmarkTools(8),
			response: createMultipleToolCallJSON(3),
		},
		{
			name:     "Medium_Complex",
			tools:    createBenchmarkTools(25),
			response: createMultipleToolCallJSON(8),
		},
		{
			name:     "Large_Heavy",
			tools:    createBenchmarkTools(50),
			response: createComplexToolCallJSON(),
		},
	}

	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			req := openai.ChatCompletionNewParams{
				Model: openai.ChatModelGPT4o,
				Messages: []openai.ChatCompletionMessageParamUnion{
					openai.UserMessage("Please help me."),
				},
				Tools: tc.tools,
			}
			resp := createBenchmarkResponse(tc.response)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				// Simulate full workflow: transform request, make call, transform response
				benchRequestResult, benchError = adapter.TransformCompletionsRequest(req)
				if benchError == nil {
					benchResponseResult, benchError = adapter.TransformCompletionsResponse(resp)
				}
			}
		})
	}
}

// Helper functions for creating benchmark data

func createBenchmarkTools(count int) []openai.ChatCompletionToolUnionParam {
	tools := make([]openai.ChatCompletionToolUnionParam, count)

	for i := 0; i < count; i++ {
		tools[i] = openai.ChatCompletionFunctionTool(
			openai.FunctionDefinitionParam{
				Name:        formatToolName(i),
				Description: openai.String(formatToolDescription(i)),
				Parameters:  createToolParameters(i),
			},
		)
	}

	return tools
}

func formatToolName(i int) string {
	names := []string{
		"get_weather", "search_web", "send_email", "create_file", "analyze_data",
		"generate_image", "translate_text", "calculate_math", "get_time", "format_json",
		"validate_input", "process_payment", "send_notification", "backup_data", "compress_file",
		"encrypt_data", "decode_base64", "parse_csv", "generate_uuid", "hash_password",
	}
	return names[i%len(names)] + formatToolSuffix(i)
}

func formatToolSuffix(i int) string {
	if i < 20 {
		return ""
	}
	return "_" + string(rune('a'+i%26)) + string(rune('0'+i%10))
}

func formatToolDescription(i int) string {
	descriptions := []string{
		"Gets weather information for a location",
		"Searches the web for information",
		"Sends an email to specified recipients",
		"Creates a new file with given content",
		"Analyzes data using statistical methods",
		"Generates images based on text descriptions",
		"Translates text between languages",
		"Performs mathematical calculations",
		"Gets the current time and date",
		"Formats JSON data for display",
	}
	return descriptions[i%len(descriptions)]
}

func createToolParameters(i int) openai.FunctionParameters {
	// Create varied parameter schemas of different complexity
	switch i % 4 {
	case 0: // Simple
		return openai.FunctionParameters{
			"type": "object",
			"properties": map[string]interface{}{
				"query": map[string]interface{}{
					"type":        "string",
					"description": "Search query",
				},
			},
		}
	case 1: // Medium
		return openai.FunctionParameters{
			"type": "object",
			"properties": map[string]interface{}{
				"location": map[string]interface{}{
					"type":        "string",
					"description": "Location name",
				},
				"unit": map[string]interface{}{
					"type": "string",
					"enum": []string{"celsius", "fahrenheit"},
				},
				"format": map[string]interface{}{
					"type":    "string",
					"default": "json",
				},
			},
			"required": []string{"location"},
		}
	case 2: // Complex
		return openai.FunctionParameters{
			"type": "object",
			"properties": map[string]interface{}{
				"data": map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"values": map[string]interface{}{
							"type": "array",
							"items": map[string]interface{}{
								"type": "number",
							},
						},
						"metadata": map[string]interface{}{
							"type": "object",
							"properties": map[string]interface{}{
								"source": map[string]interface{}{
									"type": "string",
								},
								"timestamp": map[string]interface{}{
									"type":   "string",
									"format": "date-time",
								},
							},
						},
					},
				},
				"options": map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"algorithm": map[string]interface{}{
							"type": "string",
							"enum": []string{"linear", "polynomial", "exponential"},
						},
						"precision": map[string]interface{}{
							"type":    "integer",
							"minimum": 1,
							"maximum": 10,
						},
					},
				},
			},
			"required": []string{"data"},
		}
	default: // Very simple
		return openai.FunctionParameters{
			"type": "object",
		}
	}
}

func createBenchmarkResponse(content string) openai.ChatCompletion {
	return openai.ChatCompletion{
		Choices: []openai.ChatCompletionChoice{
			{
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: content,
				},
			},
		},
	}
}

func createMultipleToolCallJSON(count int) string {
	var calls []map[string]interface{}

	for i := 0; i < count; i++ {
		call := map[string]interface{}{
			"name":       formatToolName(i),
			"parameters": createSampleParameters(i),
		}
		calls = append(calls, call)
	}

	jsonBytes, _ := json.Marshal(calls)
	return string(jsonBytes)
}

func createSampleParameters(i int) interface{} {
	switch i % 4 {
	case 0:
		return map[string]interface{}{
			"query": "test search query " + string(rune('A'+i%26)),
		}
	case 1:
		return map[string]interface{}{
			"location": "City" + string(rune('A'+i%26)),
			"unit":     "celsius",
		}
	case 2:
		return map[string]interface{}{
			"data": map[string]interface{}{
				"values": []float64{1.0, 2.0, 3.0, float64(i)},
				"metadata": map[string]interface{}{
					"source":    "benchmark",
					"timestamp": "2023-01-01T00:00:00Z",
				},
			},
			"options": map[string]interface{}{
				"algorithm": "linear",
				"precision": 5,
			},
		}
	default:
		return nil
	}
}

func createComplexToolCallJSON() string {
	complexCall := map[string]interface{}{
		"name": "process_complex_data",
		"parameters": map[string]interface{}{
			"dataset": map[string]interface{}{
				"records": []map[string]interface{}{
					{"id": 1, "value": 42.5, "tags": []string{"important", "verified"}},
					{"id": 2, "value": 38.2, "tags": []string{"pending", "review"}},
					{"id": 3, "value": 91.7, "tags": []string{"approved", "urgent"}},
				},
				"metadata": map[string]interface{}{
					"source":    "production_database",
					"timestamp": "2023-12-01T15:30:00Z",
					"version":   "2.1",
					"schema": map[string]interface{}{
						"fields": []map[string]interface{}{
							{"name": "id", "type": "integer", "required": true},
							{"name": "value", "type": "float", "required": true},
							{"name": "tags", "type": "array", "required": false},
						},
					},
				},
			},
			"processing": map[string]interface{}{
				"algorithm": "advanced_analytics",
				"parameters": map[string]interface{}{
					"window_size": 100,
					"threshold":   0.85,
					"normalize":   true,
					"filters":     []string{"outliers", "duplicates"},
				},
				"output_format": "structured_json",
			},
		},
	}

	jsonBytes, _ := json.Marshal([]interface{}{complexCall})
	return string(jsonBytes)
}

func createMixedContentResponse() string {
	return `Here's a comprehensive analysis of your request:

## Data Processing Results

I've processed your data using advanced algorithms and found several interesting patterns:

1. **Trend Analysis**: The data shows a clear upward trend over the past quarter
2. **Anomaly Detection**: Found 3 potential outliers that need review
3. **Correlation Insights**: Strong positive correlation (r=0.84) between variables A and B

Now let me execute the data processing function:

` + "```json" + `
[{"name": "process_analytics", "parameters": {"dataset_id": "Q4_2023", "analysis_type": "comprehensive", "filters": {"remove_outliers": true, "min_confidence": 0.85}, "output": {"format": "detailed_report", "include_charts": true}}}]
` + "```" + `

The processing will include:
- Statistical analysis
- Visualization generation  
- Report compilation
- Quality assurance checks

This should give you a complete picture of your data patterns and insights.`
}

// BenchmarkBufferPoolWithMemoryGrowthProtection benchmarks buffer pool performance
// with memory growth protection enabled to ensure pooling benefits are preserved
func BenchmarkBufferPoolWithMemoryGrowthProtection(b *testing.B) {
	adapter := New(WithLogLevel(slog.LevelError))

	// Test with different tool sizes to verify pooling works across various scenarios
	testCases := []struct {
		name      string
		toolCount int
		descSize  int // Size of description in characters
	}{
		{"SmallTools_5x100chars", 5, 100},
		{"MediumTools_10x500chars", 10, 500},
		{"LargeTools_20x1000chars", 20, 1000},
		{"VeryLargeTools_5x5000chars", 5, 5000}, // Should trigger buffer discard
	}

	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			// Create tools of specified size
			var tools []openai.ChatCompletionToolUnionParam
			for i := 0; i < tc.toolCount; i++ {
				description := strings.Repeat("x", tc.descSize)
				tools = append(tools, openai.ChatCompletionFunctionTool(
					openai.FunctionDefinitionParam{
						Name:        fmt.Sprintf("function_%d", i),
						Description: openai.String(description),
						Parameters: map[string]interface{}{
							"type": "object",
							"properties": map[string]interface{}{
								"param": map[string]interface{}{
									"type":        "string",
									"description": description,
								},
							},
						},
					},
				))
			}

			req := openai.ChatCompletionNewParams{
				Model:    "gpt-4",
				Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("Test message")},
				Tools:    tools,
			}

			// Reset timer to exclude setup time
			b.ResetTimer()

			// Run the benchmark
			for i := 0; i < b.N; i++ {
				benchRequestResult, benchError = adapter.TransformCompletionsRequest(req)
				if benchError != nil {
					b.Fatal(benchError)
				}
			}

			// Report memory allocations
			b.ReportAllocs()
		})
	}
}

// BenchmarkMultiChoiceTransformation benchmarks the performance of multi-choice
// response transformation to ensure no performance regression
func BenchmarkMultiChoiceTransformation(b *testing.B) {
	adapter := New(WithLogLevel(slog.LevelError))

	// Test cases with varying number of choices
	testCases := []struct {
		name        string
		choiceCount int
		hasTools    []bool // Which choices have tool calls
	}{
		{
			name:        "SingleChoice_WithTools",
			choiceCount: 1,
			hasTools:    []bool{true},
		},
		{
			name:        "ThreeChoices_AllWithTools",
			choiceCount: 3,
			hasTools:    []bool{true, true, true},
		},
		{
			name:        "ThreeChoices_OneWithTools",
			choiceCount: 3,
			hasTools:    []bool{true, false, false},
		},
		{
			name:        "FiveChoices_MixedTools",
			choiceCount: 5,
			hasTools:    []bool{true, false, true, false, true},
		},
		{
			name:        "TenChoices_AllWithTools",
			choiceCount: 10,
			hasTools:    []bool{true, true, true, true, true, true, true, true, true, true},
		},
		{
			name:        "TenChoices_NoTools",
			choiceCount: 10,
			hasTools:    []bool{false, false, false, false, false, false, false, false, false, false},
		},
	}

	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			// Create response with specified number of choices
			choices := make([]openai.ChatCompletionChoice, tc.choiceCount)
			for i := 0; i < tc.choiceCount; i++ {
				if i < len(tc.hasTools) && tc.hasTools[i] {
					choices[i] = openai.ChatCompletionChoice{
						Message: openai.ChatCompletionMessage{
							Content: `Here's the result: [{"name": "tool` + string(rune('A'+i)) + `", "parameters": {"x": ` + string(rune('0'+i)) + `}}]`,
							Role:    "assistant",
						},
						FinishReason: "stop",
					}
				} else {
					choices[i] = openai.ChatCompletionChoice{
						Message: openai.ChatCompletionMessage{
							Content: `Just regular text response for choice ` + string(rune('0'+i)),
							Role:    "assistant",
						},
						FinishReason: "stop",
					}
				}
			}

			response := openai.ChatCompletion{Choices: choices}

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				_, err := adapter.TransformCompletionsResponse(response)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkMultiChoiceWithPolicies benchmarks different tool policies
// to ensure they perform consistently across multi-choice scenarios
func BenchmarkMultiChoiceWithPolicies(b *testing.B) {
	policies := []struct {
		name   string
		policy ToolPolicy
	}{
		{"StopOnFirst", ToolStopOnFirst},
		{"CollectThenStop", ToolCollectThenStop},
		{"DrainAll", ToolDrainAll},
		{"AllowMixed", ToolAllowMixed},
	}

	// Create a 3-choice response with multiple tools per choice
	response := openai.ChatCompletion{
		Choices: []openai.ChatCompletionChoice{
			{
				Message: openai.ChatCompletionMessage{
					Content: `[{"name": "weather", "parameters": {"location": "NYC"}}, {"name": "news", "parameters": {"topic": "tech"}}]`,
					Role:    "assistant",
				},
			},
			{
				Message: openai.ChatCompletionMessage{
					Content: `[{"name": "calculate", "parameters": {"expression": "2+2"}}, {"name": "search", "parameters": {"query": "golang"}}]`,
					Role:    "assistant",
				},
			},
			{
				Message: openai.ChatCompletionMessage{
					Content: `Regular text without any tool calls`,
					Role:    "assistant",
				},
			},
		},
	}

	for _, p := range policies {
		b.Run(p.name, func(b *testing.B) {
			adapter := New(
				WithLogLevel(slog.LevelError),
				WithToolPolicy(p.policy),
			)

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				_, err := adapter.TransformCompletionsResponse(response)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
