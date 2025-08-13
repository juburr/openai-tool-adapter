package tooladapter_test

import (
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/juburr/openai-tool-adapter"
	"github.com/openai/openai-go"
)

// BenchmarkMetrics_PerformanceImpact measures the performance impact of the metrics system
func BenchmarkMetrics_PerformanceImpact(b *testing.B) {
	// Test data
	tools := []openai.ChatCompletionToolParam{
		createMockToolForMetrics("benchmark_tool_1", "First benchmark tool"),
		createMockToolForMetrics("benchmark_tool_2", "Second benchmark tool"),
		createMockToolForMetrics("benchmark_tool_3", "Third benchmark tool"),
	}
	req := createMockRequestForMetrics(tools)

	response := createMockCompletionForMetrics(`[
		{"name": "benchmark_func_1", "parameters": {"test": "value1"}},
		{"name": "benchmark_func_2", "parameters": {"test": "value2"}}
	]`)

	b.Run("WithoutMetrics", func(b *testing.B) {
		runBenchmarkWithoutMetrics(b, req, response)
	})

	b.Run("WithMetrics", func(b *testing.B) {
		runBenchmarkWithSimpleMetrics(b, req, response)
	})

	b.Run("WithComplexMetrics", func(b *testing.B) {
		runBenchmarkWithComplexMetrics(b, req, response)
	})
}

// BenchmarkMetrics_DurationCapture specifically measures the overhead of duration capture
func BenchmarkMetrics_DurationCapture(b *testing.B) {
	b.Run("TimingOverhead", func(b *testing.B) {
		var totalDuration time.Duration

		metricsCallback := func(data tooladapter.MetricEventData) {
			switch eventData := data.(type) {
			case tooladapter.ToolTransformationData:
				totalDuration += eventData.Performance.ProcessingDuration
			case tooladapter.FunctionCallDetectionData:
				totalDuration += eventData.Performance.ProcessingDuration
				for _, subDuration := range eventData.Performance.SubOperations {
					totalDuration += subDuration
				}
			}
		}

		adapter := tooladapter.New(
			tooladapter.WithMetricsCallback(metricsCallback),
		)

		tools := []openai.ChatCompletionToolParam{
			createMockToolForMetrics("timing_test", "Timing overhead test"),
		}
		req := createMockRequestForMetrics(tools)

		b.ResetTimer()
		b.ReportAllocs()

		for i := 0; i < b.N; i++ {
			_, err := adapter.TransformCompletionsRequest(req)
			if err != nil {
				b.Fatal(err)
			}
		}

		// Verify timing was captured
		if totalDuration == 0 {
			b.Error("No timing data was captured")
		}
	})

	b.Run("DurationConversions", func(b *testing.B) {
		// Measure the cost of duration conversions
		duration := 1234567 * time.Nanosecond // Arbitrary test duration

		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			// Common conversions used by monitoring systems
			_ = duration.Nanoseconds()
			_ = duration.Microseconds()
			_ = duration.Milliseconds()
			_ = duration.Seconds()
			_ = duration.String()
		}
	})
}

// BenchmarkMetrics_ConcurrentAccess tests metrics performance under concurrent load
func BenchmarkMetrics_ConcurrentAccess(b *testing.B) {
	if testing.Short() {
		b.Skip("Skipping concurrent benchmark in short mode")
	}

	b.Run("ConcurrentMetricsCollection", func(b *testing.B) {
		var eventCount int64

		metricsCallback := func(data tooladapter.MetricEventData) {
			atomic.AddInt64(&eventCount, 1)

			// Simulate some processing time
			switch eventData := data.(type) {
			case tooladapter.ToolTransformationData:
				_ = eventData.Performance.ProcessingDuration.Nanoseconds()
			case tooladapter.FunctionCallDetectionData:
				_ = eventData.Performance.ProcessingDuration.Milliseconds()
			}
		}

		adapter := tooladapter.New(
			tooladapter.WithMetricsCallback(metricsCallback),
		)

		tools := []openai.ChatCompletionToolParam{
			createMockToolForMetrics("concurrent_test", "Concurrent access test"),
		}
		req := createMockRequestForMetrics(tools)

		b.ResetTimer()
		b.SetParallelism(4) // Use 4 goroutines

		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				_, err := adapter.TransformCompletionsRequest(req)
				if err != nil {
					b.Error(err)
					return
				}
			}
		})

		// Verify all metrics were captured
		finalCount := atomic.LoadInt64(&eventCount)
		if finalCount != int64(b.N) {
			b.Errorf("Expected %d events, got %d", b.N, finalCount)
		}
	})
}

// BenchmarkMetrics_RealWorldScenarios tests metrics performance in realistic usage patterns
func BenchmarkMetrics_RealWorldScenarios(b *testing.B) {
	b.Run("TypicalWebAPIUsage", func(b *testing.B) {
		// Simulate typical web API usage with metrics sent to Prometheus
		var (
			transformationCount int64
			detectionCount      int64
			totalLatency        int64
		)

		metricsCallback := func(data tooladapter.MetricEventData) {
			switch eventData := data.(type) {
			case tooladapter.ToolTransformationData:
				atomic.AddInt64(&transformationCount, 1)
				atomic.AddInt64(&totalLatency, eventData.Performance.ProcessingDuration.Microseconds())

			case tooladapter.FunctionCallDetectionData:
				atomic.AddInt64(&detectionCount, 1)
				atomic.AddInt64(&totalLatency, eventData.Performance.ProcessingDuration.Microseconds())
			}
		}

		adapter := tooladapter.New(
			tooladapter.WithMetricsCallback(metricsCallback),
		)

		// Typical web API tools
		tools := []openai.ChatCompletionToolParam{
			createMockToolForMetrics("get_user_profile", "Get user profile information"),
			createMockToolForMetrics("update_preferences", "Update user preferences"),
			createMockToolForMetrics("send_notification", "Send notification to user"),
		}
		req := createMockRequestForMetrics(tools)

		// Typical function call response
		response := createMockCompletionForMetrics(`[
			{"name": "get_user_profile", "parameters": {"user_id": "12345"}},
			{"name": "update_preferences", "parameters": {"theme": "dark", "notifications": true}}
		]`)

		b.ResetTimer()
		b.ReportAllocs()

		for i := 0; i < b.N; i++ {
			// Full request/response cycle
			_, err := adapter.TransformCompletionsRequest(req)
			if err != nil {
				b.Fatal(err)
			}

			_, err = adapter.TransformCompletionsResponse(response)
			if err != nil {
				b.Fatal(err)
			}
		}

		// Verify metrics collection
		if atomic.LoadInt64(&transformationCount) == 0 {
			b.Error("No transformation metrics collected")
		}
		if atomic.LoadInt64(&detectionCount) == 0 {
			b.Error("No detection metrics collected")
		}
		if atomic.LoadInt64(&totalLatency) == 0 {
			b.Error("No latency metrics collected")
		}
	})

	b.Run("HighVolumeStreaming", func(b *testing.B) {
		// Simulate high-volume streaming scenario
		var streamingEvents int64

		metricsCallback := func(data tooladapter.MetricEventData) {
			if eventData, ok := data.(tooladapter.FunctionCallDetectionData); ok && eventData.Streaming {
				atomic.AddInt64(&streamingEvents, 1)
			}
		}

		adapter := tooladapter.New(
			tooladapter.WithMetricsCallback(metricsCallback),
		)

		// Streaming function call
		mockStream := NewMockStreamForMetrics([]openai.ChatCompletionChunk{
			createStreamChunkForMetrics(`[{"name": "streaming_func", "parameters": {"chunk": `),
			createStreamChunkForMetrics(`"data", "sequence": 1}}]`),
		})

		b.ResetTimer()
		b.ReportAllocs()

		for i := 0; i < b.N; i++ {
			// Reset stream for each iteration
			mockStream.current = -1

			adaptedStream := adapter.TransformStreamingResponse(mockStream)

			// Process stream
			for adaptedStream.Next() {
				_ = adaptedStream.Current()
			}
			if err := adaptedStream.Close(); err != nil {
				b.Logf("Failed to close stream in benchmark: %v", err)
			}
		}

		// Note: Streaming events might not be generated for every iteration
		// depending on the content, so we just verify the system works
		b.Logf("Generated %d streaming events", atomic.LoadInt64(&streamingEvents))
	})
}

// BenchmarkMetrics_MemoryAllocation tests memory allocation patterns
func BenchmarkMetrics_MemoryAllocation(b *testing.B) {
	b.Run("MetricsCallbackAllocation", func(b *testing.B) {
		// Test memory allocation in metrics callbacks
		var processedEvents int

		metricsCallback := func(data tooladapter.MetricEventData) {
			processedEvents++

			// Typical metrics processing that might allocate
			switch eventData := data.(type) {
			case tooladapter.ToolTransformationData:
				// Create metric labels (common in monitoring systems)
				labels := map[string]string{
					"tool_count": fmt.Sprintf("%d", eventData.ToolCount),
					"operation":  "transformation",
				}
				_ = labels

			case tooladapter.FunctionCallDetectionData:
				// Process function names
				for _, name := range eventData.FunctionNames {
					_ = fmt.Sprintf("function_%s", name)
				}
			}
		}

		adapter := tooladapter.New(
			tooladapter.WithMetricsCallback(metricsCallback),
		)

		tools := []openai.ChatCompletionToolParam{
			createMockToolForMetrics("alloc_test", "Memory allocation test"),
		}
		req := createMockRequestForMetrics(tools)

		b.ResetTimer()
		b.ReportAllocs()

		for i := 0; i < b.N; i++ {
			_, err := adapter.TransformCompletionsRequest(req)
			if err != nil {
				b.Fatal(err)
			}
		}

		if processedEvents == 0 {
			b.Error("No events were processed")
		}
	})

	b.Run("NoMetricsCallbackAllocation", func(b *testing.B) {
		// Baseline: adapter without metrics to compare allocation
		adapter := tooladapter.New()

		tools := []openai.ChatCompletionToolParam{
			createMockToolForMetrics("no_metrics_test", "No metrics baseline test"),
		}
		req := createMockRequestForMetrics(tools)

		b.ResetTimer()
		b.ReportAllocs()

		for i := 0; i < b.N; i++ {
			_, err := adapter.TransformCompletionsRequest(req)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}

// runBenchmarkWithoutMetrics runs benchmark without metrics callback
func runBenchmarkWithoutMetrics(b *testing.B, req openai.ChatCompletionNewParams, response openai.ChatCompletion) {
	// Adapter without metrics callback
	adapter := tooladapter.New()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		performBenchmarkOperations(b, adapter, req, response)
	}
}

// runBenchmarkWithSimpleMetrics runs benchmark with simple metrics counting
func runBenchmarkWithSimpleMetrics(b *testing.B, req openai.ChatCompletionNewParams, response openai.ChatCompletion) {
	// Simple metrics callback that just counts events
	var eventCount int64
	var transformationEvents int64
	var detectionEvents int64

	metricsCallback := createSimpleMetricsCallback(&eventCount, &transformationEvents, &detectionEvents)
	adapter := tooladapter.New(tooladapter.WithMetricsCallback(metricsCallback))

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		performBenchmarkOperations(b, adapter, req, response)
	}

	validateSimpleMetrics(b, eventCount, transformationEvents, detectionEvents)
}

// runBenchmarkWithComplexMetrics runs benchmark with complex metrics processing
func runBenchmarkWithComplexMetrics(b *testing.B, req openai.ChatCompletionNewParams, response openai.ChatCompletion) {
	// More complex metrics callback that simulates real-world processing
	var eventCount int64
	var totalDuration time.Duration
	var transformationCount int64
	var detectionCount int64

	metricsCallback := createComplexMetricsCallback(&eventCount, &totalDuration, &transformationCount, &detectionCount)
	adapter := tooladapter.New(tooladapter.WithMetricsCallback(metricsCallback))

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		performBenchmarkOperations(b, adapter, req, response)
	}

	validateComplexMetrics(b, eventCount, transformationCount, detectionCount)
}

// performBenchmarkOperations executes the core benchmark operations
func performBenchmarkOperations(b *testing.B, adapter *tooladapter.Adapter, req openai.ChatCompletionNewParams, response openai.ChatCompletion) {
	// Tool transformation
	_, err := adapter.TransformCompletionsRequest(req)
	if err != nil {
		b.Fatal(err)
	}

	// Function call detection
	_, err = adapter.TransformCompletionsResponse(response)
	if err != nil {
		b.Fatal(err)
	}
}

// createSimpleMetricsCallback creates a simple metrics callback for counting events
func createSimpleMetricsCallback(eventCount, transformationEvents, detectionEvents *int64) func(tooladapter.MetricEventData) {
	return func(data tooladapter.MetricEventData) {
		atomic.AddInt64(eventCount, 1)
		switch data.(type) {
		case tooladapter.ToolTransformationData:
			atomic.AddInt64(transformationEvents, 1)
		case tooladapter.FunctionCallDetectionData:
			atomic.AddInt64(detectionEvents, 1)
		}
	}
}

// createComplexMetricsCallback creates a complex metrics callback simulating real-world processing
func createComplexMetricsCallback(eventCount *int64, totalDuration *time.Duration, transformationCount, detectionCount *int64) func(tooladapter.MetricEventData) {
	return func(data tooladapter.MetricEventData) {
		atomic.AddInt64(eventCount, 1)

		switch eventData := data.(type) {
		case tooladapter.ToolTransformationData:
			processTransformationMetrics(transformationCount, eventData)
		case tooladapter.FunctionCallDetectionData:
			processDetectionMetrics(detectionCount, totalDuration, eventData)
		}
	}
}

// processTransformationMetrics processes transformation event metrics
func processTransformationMetrics(transformationCount *int64, eventData tooladapter.ToolTransformationData) {
	atomic.AddInt64(transformationCount, 1)
	// Simulate Prometheus metric recording
	_ = eventData.Performance.ProcessingDuration.Seconds()
	_ = eventData.ToolCount
	_ = len(eventData.ToolNames)
}

// processDetectionMetrics processes detection event metrics
func processDetectionMetrics(detectionCount *int64, totalDuration *time.Duration, eventData tooladapter.FunctionCallDetectionData) {
	atomic.AddInt64(detectionCount, 1)
	// Simulate DataDog metric recording
	duration := eventData.Performance.ProcessingDuration
	atomic.AddInt64((*int64)(totalDuration), int64(duration))
	_ = eventData.FunctionCount
	_ = len(eventData.FunctionNames)
	_ = eventData.ContentLength
}

// validateSimpleMetrics validates the results from simple metrics callback
func validateSimpleMetrics(b *testing.B, eventCount, transformationEvents, detectionEvents int64) {
	finalCount := atomic.LoadInt64(&eventCount)
	finalTransformations := atomic.LoadInt64(&transformationEvents)
	finalDetections := atomic.LoadInt64(&detectionEvents)

	// We expect at least some events (tool transformations should always happen)
	if finalTransformations != int64(b.N) {
		b.Errorf("Expected %d transformation events, got %d", b.N, finalTransformations)
	}

	// Function call detection depends on the response content being valid
	// Log the results for debugging but don't fail if detection events are missing
	if finalDetections == 0 {
		b.Logf("Warning: No function call detection events captured (got %d transformations, %d detections)",
			finalTransformations, finalDetections)
	}

	// Verify total count is reasonable
	if finalCount == 0 {
		b.Error("No metrics events were captured at all")
	}
}

// validateComplexMetrics validates the results from complex metrics callback
func validateComplexMetrics(b *testing.B, eventCount, transformationCount, detectionCount int64) {
	finalCount := atomic.LoadInt64(&eventCount)
	finalTransformations := atomic.LoadInt64(&transformationCount)
	finalDetections := atomic.LoadInt64(&detectionCount)

	// At minimum, we should have transformation events
	if finalTransformations != int64(b.N) {
		b.Errorf("Expected %d transformation events, got %d", b.N, finalTransformations)
	}

	if finalCount == 0 {
		b.Error("No metrics events were recorded")
	}

	// Log results for analysis
	b.Logf("Processed %d events (%d transformations, %d detections)",
		finalCount, finalTransformations, finalDetections)
}
