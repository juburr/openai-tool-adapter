package tooladapter_test

import (
	"encoding/json"
	"fmt"
	"sync"
	"testing"
	"time"

	tooladapter "github.com/juburr/openai-tool-adapter/v3"
	"github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// MetricsCollector captures metrics events for testing
type MetricsCollector struct {
	mu     sync.Mutex
	events []tooladapter.MetricEventData
}

func NewMetricsCollector() *MetricsCollector {
	return &MetricsCollector{
		events: make([]tooladapter.MetricEventData, 0),
	}
}

func (mc *MetricsCollector) Callback(data tooladapter.MetricEventData) {
	mc.mu.Lock()
	defer mc.mu.Unlock()
	mc.events = append(mc.events, data)
}

func (mc *MetricsCollector) GetEvents() []tooladapter.MetricEventData {
	mc.mu.Lock()
	defer mc.mu.Unlock()
	// Return a copy to prevent race conditions
	events := make([]tooladapter.MetricEventData, len(mc.events))
	copy(events, mc.events)
	return events
}

func (mc *MetricsCollector) Clear() {
	mc.mu.Lock()
	defer mc.mu.Unlock()
	mc.events = mc.events[:0]
}

func (mc *MetricsCollector) EventCount() int {
	mc.mu.Lock()
	defer mc.mu.Unlock()
	return len(mc.events)
}

// ============================================================================
// TOOL TRANSFORMATION METRICS TESTS
// ============================================================================

func TestMetrics_ToolTransformation_BasicEvents(t *testing.T) {
	collector := NewMetricsCollector()
	adapter := tooladapter.New(
		tooladapter.WithMetricsCallback(collector.Callback),
	)

	t.Run("SingleToolTransformation", func(t *testing.T) {
		collector.Clear()

		tools := []openai.ChatCompletionToolUnionParam{
			createMockToolForMetrics("get_weather", "Get weather information"),
		}
		req := createMockRequestForMetrics(tools)

		startTime := time.Now()
		_, err := adapter.TransformCompletionsRequest(req)
		endTime := time.Now()

		require.NoError(t, err)
		assert.Equal(t, 1, collector.EventCount(), "Should emit exactly one metrics event")

		events := collector.GetEvents()
		event, ok := events[0].(tooladapter.ToolTransformationData)
		require.True(t, ok, "Event should be ToolTransformationData type")

		// Verify event type
		assert.Equal(t, tooladapter.MetricEventToolTransformation, event.EventType())

		// Verify basic data
		assert.Equal(t, 1, event.ToolCount, "Should have correct tool count")
		assert.Equal(t, []string{"get_weather"}, event.ToolNames, "Should have correct tool names")
		assert.Greater(t, event.PromptLength, 0, "Should have non-zero prompt length")

		// Verify performance data
		assert.Greater(t, event.Performance.ProcessingDuration, time.Duration(0), "Should have positive processing duration")
		assert.Less(t, event.Performance.ProcessingDuration, endTime.Sub(startTime), "Processing duration should be reasonable")
	})

	t.Run("MultipleToolsTransformation", func(t *testing.T) {
		collector.Clear()

		tools := []openai.ChatCompletionToolUnionParam{
			createMockToolForMetrics("get_weather", "Get weather information"),
			createMockToolForMetrics("calculate_tip", "Calculate tip amount"),
			createMockToolForMetrics("convert_currency", "Convert between currencies"),
		}
		req := createMockRequestForMetrics(tools)

		_, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		events := collector.GetEvents()
		require.Len(t, events, 1, "Should emit exactly one event for multiple tools")

		event, ok := events[0].(tooladapter.ToolTransformationData)
		require.True(t, ok, "Event should be ToolTransformationData type")

		assert.Equal(t, 3, event.ToolCount, "Should have correct tool count")
		expectedNames := []string{"get_weather", "calculate_tip", "convert_currency"}
		assert.Equal(t, expectedNames, event.ToolNames, "Should have correct tool names in order")
		assert.Greater(t, event.PromptLength, 100, "Should have substantial prompt length for multiple tools")
	})

	t.Run("NoToolsRequest", func(t *testing.T) {
		collector.Clear()

		req := createMockRequestForMetrics(nil) // No tools

		_, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		assert.Equal(t, 0, collector.EventCount(), "Should not emit metrics for requests with no tools")
	})
}

func TestMetrics_ToolTransformation_PerformanceAccuracy(t *testing.T) {
	collector := NewMetricsCollector()
	adapter := tooladapter.New(
		tooladapter.WithMetricsCallback(collector.Callback),
	)

	t.Run("PerformanceTimingPrecision", func(t *testing.T) {
		collector.Clear()

		// Create a request with many tools to ensure measurable processing time
		tools := make([]openai.ChatCompletionToolUnionParam, 10)
		for i := 0; i < 10; i++ {
			tools[i] = createMockToolForMetrics(
				fmt.Sprintf("tool_%d", i),
				fmt.Sprintf("Tool number %d with a longer description to increase processing time", i),
			)
		}
		req := createMockRequestForMetrics(tools)

		_, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		events := collector.GetEvents()
		require.Len(t, events, 1)

		event := events[0].(tooladapter.ToolTransformationData)

		// Verify timing precision
		assert.Greater(t, event.Performance.ProcessingDuration, time.Duration(0), "Should have positive duration")
		assert.Less(t, event.Performance.ProcessingDuration, 100*time.Millisecond, "Should complete quickly")

		// Verify nanosecond precision is available
		nanoseconds := event.Performance.ProcessingDuration.Nanoseconds()
		assert.Greater(t, nanoseconds, int64(0), "Should have nanosecond precision")
	})

	t.Run("ConsistentPerformanceMetrics", func(t *testing.T) {
		// Run the same transformation multiple times and verify metrics consistency
		tools := []openai.ChatCompletionToolUnionParam{
			createMockToolForMetrics("consistent_tool", "Consistent tool for testing"),
		}
		req := createMockRequestForMetrics(tools)

		var durations []time.Duration
		for i := 0; i < 5; i++ {
			collector.Clear()

			_, err := adapter.TransformCompletionsRequest(req)
			require.NoError(t, err)

			events := collector.GetEvents()
			require.Len(t, events, 1)

			event := events[0].(tooladapter.ToolTransformationData)
			durations = append(durations, event.Performance.ProcessingDuration)

			// Verify basic consistency
			assert.Equal(t, 1, event.ToolCount)
			assert.Equal(t, []string{"consistent_tool"}, event.ToolNames)
		}

		// All durations should be positive and reasonable
		for i, duration := range durations {
			assert.Greater(t, duration, time.Duration(0), "Duration %d should be positive", i)
			assert.Less(t, duration, 50*time.Millisecond, "Duration %d should be reasonable", i)
		}
	})
}

// ============================================================================
// FUNCTION CALL DETECTION METRICS TESTS
// ============================================================================

func TestMetrics_FunctionCallDetection_BasicEvents(t *testing.T) {
	collector := NewMetricsCollector()
	adapter := tooladapter.New(
		tooladapter.WithMetricsCallback(collector.Callback),
	)

	t.Run("SingleFunctionCallDetection", func(t *testing.T) {
		collector.Clear()

		response := createMockCompletionForMetrics(`[{"name": "get_weather", "parameters": {"location": "Boston"}}]`)

		startTime := time.Now()
		_, err := adapter.TransformCompletionsResponse(response)
		endTime := time.Now()

		require.NoError(t, err)
		assert.Equal(t, 1, collector.EventCount(), "Should emit exactly one metrics event")

		events := collector.GetEvents()
		event, ok := events[0].(tooladapter.FunctionCallDetectionData)
		require.True(t, ok, "Event should be FunctionCallDetectionData type")

		// Verify event type
		assert.Equal(t, tooladapter.MetricEventFunctionCallDetection, event.EventType())

		// Verify basic data
		assert.Equal(t, 1, event.FunctionCount, "Should have correct function count")
		assert.Equal(t, []string{"get_weather"}, event.FunctionNames, "Should have correct function names")
		assert.Greater(t, event.ContentLength, 0, "Should have non-zero content length")
		assert.Greater(t, event.JSONCandidates, 0, "Should have found JSON candidates")
		assert.False(t, event.Streaming, "Should indicate non-streaming mode")

		// Verify performance data
		assert.Greater(t, event.Performance.ProcessingDuration, time.Duration(0), "Should have positive processing duration")
		assert.Less(t, event.Performance.ProcessingDuration, endTime.Sub(startTime), "Processing duration should be reasonable")

		// Verify sub-operations
		assert.Contains(t, event.Performance.SubOperations, "json_parsing", "Should include json_parsing timing")
		assert.Contains(t, event.Performance.SubOperations, "call_extraction", "Should include call_extraction timing")
		assert.Greater(t, event.Performance.SubOperations["json_parsing"], time.Duration(0), "JSON parsing should have positive duration")
		assert.Greater(t, event.Performance.SubOperations["call_extraction"], time.Duration(0), "Call extraction should have positive duration")
	})

	t.Run("MultipleFunctionCallsDetection", func(t *testing.T) {
		collector.Clear()

		response := createMockCompletionForMetrics(`[
			{"name": "get_weather", "parameters": {"location": "Boston"}},
			{"name": "calculate_tip", "parameters": {"amount": 50, "percentage": 20}},
			{"name": "get_time", "parameters": null}
		]`)

		_, err := adapter.TransformCompletionsResponse(response)
		require.NoError(t, err)

		events := collector.GetEvents()
		require.Len(t, events, 1, "Should emit exactly one event for multiple function calls")

		event, ok := events[0].(tooladapter.FunctionCallDetectionData)
		require.True(t, ok, "Event should be FunctionCallDetectionData type")

		assert.Equal(t, 3, event.FunctionCount, "Should have correct function count")
		expectedNames := []string{"get_weather", "calculate_tip", "get_time"}
		assert.Equal(t, expectedNames, event.FunctionNames, "Should have correct function names in order")
	})

	t.Run("NoFunctionCallsInResponse", func(t *testing.T) {
		collector.Clear()

		response := createMockCompletionForMetrics("This is just a regular text response with no function calls.")

		_, err := adapter.TransformCompletionsResponse(response)
		require.NoError(t, err)

		assert.Equal(t, 0, collector.EventCount(), "Should not emit metrics for responses with no function calls")
	})

	t.Run("InvalidJSONInResponse", func(t *testing.T) {
		collector.Clear()

		response := createMockCompletionForMetrics(`[{"name": "broken_func", "parameters": invalid_json}]`)

		_, err := adapter.TransformCompletionsResponse(response)
		require.NoError(t, err)

		assert.Equal(t, 0, collector.EventCount(), "Should not emit metrics for responses with invalid JSON")
	})
}

func TestMetrics_FunctionCallDetection_PerformanceBreakdown(t *testing.T) {
	collector := NewMetricsCollector()
	adapter := tooladapter.New(
		tooladapter.WithMetricsCallback(collector.Callback),
	)

	t.Run("SubOperationTimingBreakdown", func(t *testing.T) {
		collector.Clear()

		// Create a complex response with nested JSON to ensure measurable sub-operation times
		response := createMockCompletionForMetrics(`[{
			"name": "complex_function",
			"parameters": {
				"nested": {
					"deep": {
						"structure": {
							"with": ["many", "values", "to", "parse"]
						}
					}
				},
				"array": [1, 2, 3, 4, 5],
				"string": "Some text with \"escaped quotes\" and special chars"
			}
		}]`)

		_, err := adapter.TransformCompletionsResponse(response)
		require.NoError(t, err)

		events := collector.GetEvents()
		require.Len(t, events, 1)

		event := events[0].(tooladapter.FunctionCallDetectionData)

		// Verify performance breakdown
		totalDuration := event.Performance.ProcessingDuration
		jsonParsing := event.Performance.SubOperations["json_parsing"]
		callExtraction := event.Performance.SubOperations["call_extraction"]

		// All durations should be positive
		assert.Greater(t, totalDuration, time.Duration(0), "Total duration should be positive")
		assert.Greater(t, jsonParsing, time.Duration(0), "JSON parsing duration should be positive")
		assert.Greater(t, callExtraction, time.Duration(0), "Call extraction duration should be positive")

		// Sub-operations should sum to less than or equal to total (allowing for measurement overhead)
		subOperationSum := jsonParsing + callExtraction
		assert.LessOrEqual(t, subOperationSum, totalDuration, "Sub-operations should not exceed total duration")

		// Verify precision - should be able to measure sub-microsecond differences
		assert.Greater(t, jsonParsing.Nanoseconds(), int64(0), "Should have nanosecond precision for JSON parsing")
		assert.Greater(t, callExtraction.Nanoseconds(), int64(0), "Should have nanosecond precision for call extraction")
	})

	t.Run("ContentLengthAccuracy", func(t *testing.T) {
		testCases := []struct {
			name     string
			content  string
			expected int
		}{
			{
				name:     "SimpleFunction",
				content:  `{"name": "test", "parameters": {}}`,
				expected: len(`{"name": "test", "parameters": {}}`),
			},
			{
				name:     "ComplexFunction",
				content:  `[{"name": "complex", "parameters": {"key": "value", "number": 42}}]`,
				expected: len(`[{"name": "complex", "parameters": {"key": "value", "number": 42}}]`),
			},
			{
				name:     "UnicodeContent",
				content:  `{"name": "unicode_test", "parameters": {"message": "Hello 世界 🌍"}}`,
				expected: len(`{"name": "unicode_test", "parameters": {"message": "Hello 世界 🌍"}}`),
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				collector.Clear()

				response := createMockCompletionForMetrics(tc.content)
				_, err := adapter.TransformCompletionsResponse(response)
				require.NoError(t, err)

				events := collector.GetEvents()
				require.Len(t, events, 1)

				event := events[0].(tooladapter.FunctionCallDetectionData)
				assert.Equal(t, tc.expected, event.ContentLength, "Content length should match exactly")
			})
		}
	})
}

// ============================================================================
// STREAMING METRICS TESTS
// ============================================================================

func TestMetrics_StreamingFunctionCallDetection(t *testing.T) {
	collector := NewMetricsCollector()
	adapter := tooladapter.New(
		tooladapter.WithMetricsCallback(collector.Callback),
	)

	t.Run("StreamingModeDetection", func(t *testing.T) {
		collector.Clear()

		// Create a mock streaming scenario
		mockStream := NewMockStreamForMetrics([]openai.ChatCompletionChunk{
			createStreamChunkForMetrics(`[{"name": "stream_func", "parameters": {"test": "value"}}]`),
		})

		adaptedStream := adapter.TransformStreamingResponse(mockStream)
		defer func() {
			if err := adaptedStream.Close(); err != nil {
				t.Logf("Failed to close stream in metrics test: %v", err)
			}
		}()

		// Process the stream
		for adaptedStream.Next() {
			_ = adaptedStream.Current()
		}
		require.NoError(t, adaptedStream.Err())

		events := collector.GetEvents()
		require.Len(t, events, 1, "Should emit one metrics event for streaming function call")

		event, ok := events[0].(tooladapter.FunctionCallDetectionData)
		require.True(t, ok, "Event should be FunctionCallDetectionData type")

		// Verify streaming mode is correctly identified
		assert.True(t, event.Streaming, "Should indicate streaming mode")
		assert.Equal(t, 1, event.FunctionCount, "Should have correct function count")
		assert.Equal(t, []string{"stream_func"}, event.FunctionNames, "Should have correct function names")

		// Verify performance data is captured for streaming
		assert.Greater(t, event.Performance.ProcessingDuration, time.Duration(0), "Should have positive processing duration")
		assert.Contains(t, event.Performance.SubOperations, "json_parsing", "Should include json_parsing timing")
		assert.Contains(t, event.Performance.SubOperations, "call_extraction", "Should include call_extraction timing")
	})

	t.Run("StreamingVsNonStreamingComparison", func(t *testing.T) {
		functionCallJSON := `[{"name": "comparison_func", "parameters": {"mode": "test"}}]`

		// Test non-streaming
		collector.Clear()
		response := createMockCompletionForMetrics(functionCallJSON)
		_, err := adapter.TransformCompletionsResponse(response)
		require.NoError(t, err)

		nonStreamingEvents := collector.GetEvents()
		require.Len(t, nonStreamingEvents, 1)
		nonStreamingEvent := nonStreamingEvents[0].(tooladapter.FunctionCallDetectionData)

		// Test streaming
		collector.Clear()
		mockStream := NewMockStreamForMetrics([]openai.ChatCompletionChunk{
			createStreamChunkForMetrics(functionCallJSON),
		})
		adaptedStream := adapter.TransformStreamingResponse(mockStream)
		for adaptedStream.Next() {
			_ = adaptedStream.Current()
		}
		if err := adaptedStream.Close(); err != nil {
			t.Logf("Failed to close stream in metrics test: %v", err)
		}

		streamingEvents := collector.GetEvents()
		require.Len(t, streamingEvents, 1)
		streamingEvent := streamingEvents[0].(tooladapter.FunctionCallDetectionData)

		// Compare the two modes
		assert.False(t, nonStreamingEvent.Streaming, "Non-streaming should be marked as such")
		assert.True(t, streamingEvent.Streaming, "Streaming should be marked as such")

		// Both should have the same basic function call data
		assert.Equal(t, nonStreamingEvent.FunctionCount, streamingEvent.FunctionCount)
		assert.Equal(t, nonStreamingEvent.FunctionNames, streamingEvent.FunctionNames)

		// Both should have performance data
		assert.Greater(t, nonStreamingEvent.Performance.ProcessingDuration, time.Duration(0))
		assert.Greater(t, streamingEvent.Performance.ProcessingDuration, time.Duration(0))
	})
}

// ============================================================================
// EDGE CASES AND ERROR SCENARIOS
// ============================================================================

func TestMetrics_EdgeCases(t *testing.T) {
	t.Run("NoMetricsCallback", func(t *testing.T) {
		// Create adapter without metrics callback - should not panic
		adapter := tooladapter.New()

		tools := []openai.ChatCompletionToolUnionParam{
			createMockToolForMetrics("test_tool", "Test tool"),
		}
		req := createMockRequestForMetrics(tools)

		// Should not panic even without metrics callback
		_, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		response := createMockCompletionForMetrics(`[{"name": "test_func", "parameters": {}}]`)
		_, err = adapter.TransformCompletionsResponse(response)
		require.NoError(t, err)
	})

	t.Run("NilMetricsCallback", func(t *testing.T) {
		// Explicitly set nil callback - should not panic
		adapter := tooladapter.New(
			tooladapter.WithMetricsCallback(nil),
		)

		tools := []openai.ChatCompletionToolUnionParam{
			createMockToolForMetrics("test_tool", "Test tool"),
		}
		req := createMockRequestForMetrics(tools)

		_, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)
	})

	t.Run("EmptyResponseContent", func(t *testing.T) {
		collector := NewMetricsCollector()
		adapter := tooladapter.New(
			tooladapter.WithMetricsCallback(collector.Callback),
		)

		response := createMockCompletionForMetrics("") // Empty content
		_, err := adapter.TransformCompletionsResponse(response)
		require.NoError(t, err)

		assert.Equal(t, 0, collector.EventCount(), "Should not emit metrics for empty content")
	})

	t.Run("LargeNumberOfTools", func(t *testing.T) {
		collector := NewMetricsCollector()
		adapter := tooladapter.New(
			tooladapter.WithMetricsCallback(collector.Callback),
		)

		// Create 50 tools to test performance with large numbers
		tools := make([]openai.ChatCompletionToolUnionParam, 50)
		expectedNames := make([]string, 50)
		for i := 0; i < 50; i++ {
			name := fmt.Sprintf("tool_%02d", i)
			tools[i] = createMockToolForMetrics(name, fmt.Sprintf("Tool number %d", i))
			expectedNames[i] = name
		}

		req := createMockRequestForMetrics(tools)
		_, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		events := collector.GetEvents()
		require.Len(t, events, 1)

		event := events[0].(tooladapter.ToolTransformationData)
		assert.Equal(t, 50, event.ToolCount, "Should handle large number of tools")
		assert.Equal(t, expectedNames, event.ToolNames, "Should preserve tool order")
		assert.Greater(t, event.PromptLength, 1000, "Should have substantial prompt length")

		// Should still complete in reasonable time even with many tools
		assert.Less(t, event.Performance.ProcessingDuration, 100*time.Millisecond, "Should handle large numbers efficiently")
	})
}

// ============================================================================
// METRICS DATA INTEGRITY TESTS
// ============================================================================

func TestMetrics_DataIntegrity(t *testing.T) {
	collector := NewMetricsCollector()
	adapter := tooladapter.New(
		tooladapter.WithMetricsCallback(collector.Callback),
	)

	t.Run("EventTypeConsistency", func(t *testing.T) {
		collector.Clear()

		// Test tool transformation
		tools := []openai.ChatCompletionToolUnionParam{
			createMockToolForMetrics("test_tool", "Test tool"),
		}
		req := createMockRequestForMetrics(tools)
		_, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// Test function call detection
		response := createMockCompletionForMetrics(`[{"name": "test_func", "parameters": {}}]`)
		_, err = adapter.TransformCompletionsResponse(response)
		require.NoError(t, err)

		events := collector.GetEvents()
		require.Len(t, events, 2, "Should have two events")

		// Verify first event is tool transformation
		toolEvent, ok := events[0].(tooladapter.ToolTransformationData)
		require.True(t, ok, "First event should be ToolTransformationData")
		assert.Equal(t, tooladapter.MetricEventToolTransformation, toolEvent.EventType())

		// Verify second event is function call detection
		funcEvent, ok := events[1].(tooladapter.FunctionCallDetectionData)
		require.True(t, ok, "Second event should be FunctionCallDetectionData")
		assert.Equal(t, tooladapter.MetricEventFunctionCallDetection, funcEvent.EventType())
	})

	t.Run("JSONSerializability", func(t *testing.T) {
		collector := NewMetricsCollector()
		adapter := tooladapter.New(
			tooladapter.WithMetricsCallback(collector.Callback),
		)

		// Generate some metrics events
		tools := []openai.ChatCompletionToolUnionParam{
			createMockToolForMetrics("serialization_test", "Test JSON serialization"),
		}
		req := createMockRequestForMetrics(tools)
		_, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		events := collector.GetEvents()
		require.Len(t, events, 1)

		// Verify the event can be serialized to JSON
		eventData := events[0].(tooladapter.ToolTransformationData)
		jsonData, err := json.Marshal(eventData)
		require.NoError(t, err, "Event should be JSON serializable")

		// Verify it can be deserialized
		var deserializedEvent tooladapter.ToolTransformationData
		err = json.Unmarshal(jsonData, &deserializedEvent)
		require.NoError(t, err, "Event should be JSON deserializable")

		// Verify data integrity after round-trip
		assert.Equal(t, eventData.ToolCount, deserializedEvent.ToolCount)
		assert.Equal(t, eventData.ToolNames, deserializedEvent.ToolNames)
		assert.Equal(t, eventData.PromptLength, deserializedEvent.PromptLength)
		// Note: time.Duration JSON serialization is implementation-specific, so we just verify it's present
		assert.NotZero(t, deserializedEvent.Performance.ProcessingDuration)
	})

	t.Run("ThreadSafety", func(t *testing.T) {
		collector := NewMetricsCollector()
		adapter := tooladapter.New(
			tooladapter.WithMetricsCallback(collector.Callback),
		)

		// Run multiple operations concurrently
		const numGoroutines = 10
		const operationsPerGoroutine = 5

		var wg sync.WaitGroup
		wg.Add(numGoroutines)

		for i := 0; i < numGoroutines; i++ {
			go func(goroutineID int) {
				defer wg.Done()

				for j := 0; j < operationsPerGoroutine; j++ {
					// Alternate between tool transformations and function call detections
					if j%2 == 0 {
						tools := []openai.ChatCompletionToolUnionParam{
							createMockToolForMetrics(fmt.Sprintf("tool_g%d_op%d", goroutineID, j), "Concurrent test tool"),
						}
						req := createMockRequestForMetrics(tools)
						_, err := adapter.TransformCompletionsRequest(req)
						require.NoError(t, err)
					} else {
						response := createMockCompletionForMetrics(fmt.Sprintf(`[{"name": "func_g%d_op%d", "parameters": {}}]`, goroutineID, j))
						_, err := adapter.TransformCompletionsResponse(response)
						require.NoError(t, err)
					}
				}
			}(i)
		}

		wg.Wait()

		// Verify we got all expected events
		expectedEvents := numGoroutines * operationsPerGoroutine
		assert.Equal(t, expectedEvents, collector.EventCount(), "Should have received all metrics events from concurrent operations")

		// Verify no data corruption
		events := collector.GetEvents()
		for i, event := range events {
			switch e := event.(type) {
			case tooladapter.ToolTransformationData:
				assert.Greater(t, e.ToolCount, 0, "Event %d should have positive tool count", i)
				assert.NotEmpty(t, e.ToolNames, "Event %d should have tool names", i)
				assert.Greater(t, e.Performance.ProcessingDuration, time.Duration(0), "Event %d should have positive duration", i)
			case tooladapter.FunctionCallDetectionData:
				assert.Greater(t, e.FunctionCount, 0, "Event %d should have positive function count", i)
				assert.NotEmpty(t, e.FunctionNames, "Event %d should have function names", i)
				assert.Greater(t, e.Performance.ProcessingDuration, time.Duration(0), "Event %d should have positive duration", i)
			default:
				t.Errorf("Event %d has unexpected type: %T", i, e)
			}
		}
	})
}

// ============================================================================
// HELPER FUNCTIONS FOR METRICS TESTS
// ============================================================================

func createMockToolForMetrics(name, description string) openai.ChatCompletionToolUnionParam {
	return openai.ChatCompletionFunctionTool(
		openai.FunctionDefinitionParam{
			Name:        name,
			Description: openai.String(description),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]interface{}{
					"param1": map[string]interface{}{
						"type":        "string",
						"description": "A test parameter",
					},
				},
			},
		},
	)
}

func createMockRequestForMetrics(tools []openai.ChatCompletionToolUnionParam) openai.ChatCompletionNewParams {
	return openai.ChatCompletionNewParams{
		Model: openai.ChatModelGPT4o,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("Test message for metrics"),
		},
		Tools: tools,
	}
}

func createMockCompletionForMetrics(content string) openai.ChatCompletion {
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

// MockChatCompletionStreamForMetrics implements the streaming interface for metrics testing
type MockChatCompletionStreamForMetrics struct {
	chunks  []openai.ChatCompletionChunk
	current int
	err     error
}

func NewMockStreamForMetrics(chunks []openai.ChatCompletionChunk) *MockChatCompletionStreamForMetrics {
	return &MockChatCompletionStreamForMetrics{
		chunks:  chunks,
		current: -1,
	}
}

func (m *MockChatCompletionStreamForMetrics) Next() bool {
	m.current++
	return m.current < len(m.chunks)
}

func (m *MockChatCompletionStreamForMetrics) Current() openai.ChatCompletionChunk {
	if m.current >= 0 && m.current < len(m.chunks) {
		return m.chunks[m.current]
	}
	return openai.ChatCompletionChunk{}
}

func (m *MockChatCompletionStreamForMetrics) Err() error {
	return m.err
}

func (m *MockChatCompletionStreamForMetrics) Close() error {
	return nil
}

func createStreamChunkForMetrics(content string) openai.ChatCompletionChunk {
	return openai.ChatCompletionChunk{
		Choices: []openai.ChatCompletionChunkChoice{
			{
				Delta: openai.ChatCompletionChunkChoiceDelta{
					Content: content,
					Role:    "assistant",
				},
			},
		},
	}
}
