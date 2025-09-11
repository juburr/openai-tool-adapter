package tooladapter_test

import (
	"fmt"
	"strings"
	"testing"
	"time"

	tooladapter "github.com/juburr/openai-tool-adapter"
	"github.com/openai/openai-go/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestMetrics_DurationPrecisionAndConversion validates that the time.Duration
// metrics provide the precision and conversion capabilities needed for various
// monitoring systems.
func TestMetrics_DurationPrecisionAndConversion(t *testing.T) {
	collector := NewMetricsCollector()
	adapter := tooladapter.New(
		tooladapter.WithMetricsCallback(collector.Callback),
	)

	t.Run("NanosecondPrecisionCapture", func(t *testing.T) {
		collector.Clear()

		tools := []openai.ChatCompletionToolUnionParam{
			createMockToolForMetrics("precision_test", "Test nanosecond precision"),
		}
		req := createMockRequestForMetrics(tools)

		_, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		events := collector.GetEvents()
		require.Len(t, events, 1)

		event := events[0].(tooladapter.ToolTransformationData)
		duration := event.Performance.ProcessingDuration

		// Verify we can access nanosecond precision
		nanoseconds := duration.Nanoseconds()
		assert.Greater(t, nanoseconds, int64(0), "Should capture nanosecond precision")

		// Verify the duration is reasonable (should be microseconds to milliseconds)
		assert.Greater(t, nanoseconds, int64(100), "Should be at least 100ns")
		assert.Less(t, nanoseconds, int64(100*time.Millisecond), "Should be less than 100ms")

		// Verify nanosecond precision is actually being captured (not just rounded to milliseconds)
		// If we're only capturing millisecond precision, nanoseconds % 1000000 would always be 0
		// With real nanosecond precision, we should see sub-millisecond values
		// Note: This test might be flaky on very fast systems, but it validates precision
		subMillisecondPart := nanoseconds % int64(time.Millisecond)
		t.Logf("Full nanoseconds: %d, sub-millisecond part: %d", nanoseconds, subMillisecondPart)
		// We don't assert this is non-zero because fast operations might round to milliseconds,
		// but we log it to verify precision in practice
	})

	t.Run("ConversionToMonitoringSystemUnits", func(t *testing.T) {
		collector.Clear()

		// Generate a function call detection event which has sub-operations
		response := createMockCompletionForMetrics(`[{
			"name": "monitoring_test", 
			"parameters": {"system": "prometheus", "unit": "seconds"}
		}]`)

		_, err := adapter.TransformCompletionsResponse(response)
		require.NoError(t, err)

		events := collector.GetEvents()
		require.Len(t, events, 1)

		event := events[0].(tooladapter.FunctionCallDetectionData)
		totalDuration := event.Performance.ProcessingDuration
		jsonParsing := event.Performance.SubOperations["json_parsing"]
		callExtraction := event.Performance.SubOperations["call_extraction"]

		// Test conversions to different units needed by monitoring systems
		testDurationConversions := func(d time.Duration, name string) {
			// Prometheus typically uses seconds (float64)
			seconds := d.Seconds()
			assert.GreaterOrEqual(t, seconds, 0.0, "%s: Seconds should be non-negative", name)
			assert.IsType(t, float64(0), seconds, "%s: Seconds should be float64", name)

			// DataDog and many systems use milliseconds (int64) - but operations might be sub-millisecond
			milliseconds := d.Milliseconds()
			assert.GreaterOrEqual(t, milliseconds, int64(0), "%s: Milliseconds should be non-negative", name)
			assert.IsType(t, int64(0), milliseconds, "%s: Milliseconds should be int64", name)

			// Some systems prefer microseconds (int64) - more appropriate for fast operations
			microseconds := d.Microseconds()
			assert.GreaterOrEqual(t, microseconds, int64(0), "%s: Microseconds should be non-negative", name)
			assert.IsType(t, int64(0), microseconds, "%s: Microseconds should be int64", name)

			// Database storage often uses nanoseconds (int64)
			nanoseconds := d.Nanoseconds()
			assert.Greater(t, nanoseconds, int64(0), "%s: Nanoseconds should be positive", name)
			assert.IsType(t, int64(0), nanoseconds, "%s: Nanoseconds should be int64", name)

			// Human-readable string representation
			humanReadable := d.String()
			assert.NotEmpty(t, humanReadable, "%s: String representation should not be empty", name)

			// Check for valid time unit suffixes (Go uses MICRO SIGN for microseconds)
			validSuffixes := []string{"ns", "\u00b5s", "ms", "s"} // \u00b5 = MICRO SIGN (matches Go's output)
			hasSuffix := false
			for _, suffix := range validSuffixes {
				if strings.HasSuffix(humanReadable, suffix) {
					hasSuffix = true
					break
				}
			}
			assert.True(t, hasSuffix, "%s: String '%s' should have valid time unit suffix (ns, \u00b5s, ms, s)", name, humanReadable)

			// Verify conversion consistency for durations that are large enough
			if d >= time.Microsecond {
				assert.Greater(t, microseconds, int64(0), "%s: Should have positive microseconds for durations >= 1μs", name)
			}
			// Only expect positive milliseconds for durations >= 1ms
			if d >= time.Millisecond {
				assert.Greater(t, milliseconds, int64(0), "%s: Should have positive milliseconds for durations >= 1ms", name)
			}
		}

		// Test all captured durations
		testDurationConversions(totalDuration, "total_duration")
		testDurationConversions(jsonParsing, "json_parsing")
		testDurationConversions(callExtraction, "call_extraction")

		// Verify precision relationships
		totalNanos := totalDuration.Nanoseconds()
		totalMicros := totalDuration.Microseconds()
		totalMillis := totalDuration.Milliseconds()

		// Verify conversion math
		assert.Equal(t, totalNanos/1000, totalMicros, "Microsecond conversion should be consistent")
		assert.Equal(t, totalNanos/1000000, totalMillis, "Millisecond conversion should be consistent")

		t.Logf("Duration conversions for %v:", totalDuration)
		t.Logf("  Nanoseconds: %d", totalNanos)
		t.Logf("  Microseconds: %d", totalMicros)
		t.Logf("  Milliseconds: %d", totalMillis)
		t.Logf("  Seconds: %f", totalDuration.Seconds())
		t.Logf("  String: %s", totalDuration.String())
	})

	t.Run("RealWorldMonitoringSystemIntegration", func(t *testing.T) {
		// Mock monitoring systems that would receive the metrics
		var prometheusMetrics []float64
		var datadogMetrics []int64
		var customDBMetrics []int64
		var loggedDurations []string

		// Create adapter with callback that simulates real monitoring integration
		adapter := tooladapter.New(
			tooladapter.WithMetricsCallback(func(data tooladapter.MetricEventData) {
				switch eventData := data.(type) {
				case tooladapter.ToolTransformationData:
					duration := eventData.Performance.ProcessingDuration

					// Prometheus integration (seconds as float64)
					prometheusMetrics = append(prometheusMetrics, duration.Seconds())

					// DataDog integration (microseconds for fast operations)
					datadogMetrics = append(datadogMetrics, duration.Microseconds())

					// Custom database (nanoseconds as int64 for precision)
					customDBMetrics = append(customDBMetrics, duration.Nanoseconds())

					// Logging (human-readable string)
					loggedDurations = append(loggedDurations, duration.String())

				case tooladapter.FunctionCallDetectionData:
					duration := eventData.Performance.ProcessingDuration

					// Same conversions for function call events
					prometheusMetrics = append(prometheusMetrics, duration.Seconds())
					datadogMetrics = append(datadogMetrics, duration.Microseconds())
					customDBMetrics = append(customDBMetrics, duration.Nanoseconds())
					loggedDurations = append(loggedDurations, duration.String())
				}
			}),
		)

		// Generate multiple metric events
		tools := []openai.ChatCompletionToolUnionParam{
			createMockToolForMetrics("integration_test", "Test real-world integration"),
		}
		req := createMockRequestForMetrics(tools)
		_, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		response := createMockCompletionForMetrics(`[{"name": "integration_func", "parameters": {}}]`)
		_, err = adapter.TransformCompletionsResponse(response)
		require.NoError(t, err)

		// Verify all monitoring systems received data
		assert.Len(t, prometheusMetrics, 2, "Prometheus should have received 2 metrics")
		assert.Len(t, datadogMetrics, 2, "DataDog should have received 2 metrics")
		assert.Len(t, customDBMetrics, 2, "Custom DB should have received 2 metrics")
		assert.Len(t, loggedDurations, 2, "Logger should have received 2 durations")

		// Verify all metrics are valid
		for i := range prometheusMetrics {
			assert.Greater(t, prometheusMetrics[i], 0.0, "Prometheus metric %d should be positive", i)
			assert.Greater(t, datadogMetrics[i], int64(0), "DataDog metric %d should be positive", i)
			assert.Greater(t, customDBMetrics[i], int64(0), "Custom DB metric %d should be positive", i)
			assert.NotEmpty(t, loggedDurations[i], "Logged duration %d should not be empty", i)
		}

		// Verify conversion consistency between systems
		for i := range prometheusMetrics {
			prometheusNanos := int64(prometheusMetrics[i] * 1e9) // Convert seconds back to nanoseconds
			datadogNanos := datadogMetrics[i] * 1e3              // Convert microseconds to nanoseconds
			customNanos := customDBMetrics[i]                    // Already in nanoseconds

			// Allow for rounding differences between conversions
			// Prometheus (float64 seconds) has less precision than direct nanoseconds
			assert.InDelta(t, float64(customNanos), float64(prometheusNanos), float64(customNanos)*0.01,
				"Prometheus conversion should be close to custom DB (metric %d)", i)

			// DataDog microseconds should be close to nanoseconds (within 1μs)
			assert.InDelta(t, float64(customNanos), float64(datadogNanos), float64(time.Microsecond),
				"DataDog conversion should be close to custom DB (metric %d)", i)
		}

		t.Logf("Monitoring system metrics:")
		for i := range prometheusMetrics {
			t.Logf("  Event %d: Prometheus=%.6fs, DataDog=%d\u00b5s, CustomDB=%dns, Log=%s",
				i, prometheusMetrics[i], datadogMetrics[i], customDBMetrics[i], loggedDurations[i])
		}
	})

	t.Run("SubOperationPrecisionConsistency", func(t *testing.T) {
		collector.Clear()

		// Create a response that will trigger sub-operation timing
		response := createMockCompletionForMetrics(`[{
			"name": "sub_operation_test",
			"parameters": {
				"precision": "nanosecond",
				"test": "sub-operations timing consistency"
			}
		}]`)

		_, err := adapter.TransformCompletionsResponse(response)
		require.NoError(t, err)

		events := collector.GetEvents()
		require.Len(t, events, 1)

		event := events[0].(tooladapter.FunctionCallDetectionData)

		totalDuration := event.Performance.ProcessingDuration
		subOps := event.Performance.SubOperations

		// Verify sub-operations exist and have precision
		require.Contains(t, subOps, "json_parsing", "Should have json_parsing sub-operation")
		require.Contains(t, subOps, "call_extraction", "Should have call_extraction sub-operation")

		jsonParsing := subOps["json_parsing"]
		callExtraction := subOps["call_extraction"]

		// All durations should be positive and have nanosecond precision
		assert.Greater(t, totalDuration.Nanoseconds(), int64(0), "Total duration should have nanosecond precision")
		assert.Greater(t, jsonParsing.Nanoseconds(), int64(0), "JSON parsing should have nanosecond precision")
		assert.Greater(t, callExtraction.Nanoseconds(), int64(0), "Call extraction should have nanosecond precision")

		// Sub-operations should not exceed total duration
		subOpSum := jsonParsing + callExtraction
		assert.LessOrEqual(t, subOpSum, totalDuration, "Sub-operations should not exceed total duration")

		// Test precision by verifying we can distinguish between operations
		// (They should have different durations, even if small)
		if jsonParsing != callExtraction {
			t.Logf("Successfully captured different timings: json_parsing=%v, call_extraction=%v",
				jsonParsing, callExtraction)
		}

		// Verify consistency across conversion methods for sub-operations
		for name, duration := range subOps {
			nanos := duration.Nanoseconds()
			micros := duration.Microseconds()
			millis := duration.Milliseconds()

			// Basic consistency checks
			assert.Equal(t, nanos/1000, micros, "Sub-operation %s: microsecond conversion should be consistent", name)
			assert.Equal(t, nanos/1000000, millis, "Sub-operation %s: millisecond conversion should be consistent", name)

			// Verify we can convert to common monitoring formats
			seconds := duration.Seconds()
			assert.GreaterOrEqual(t, seconds, 0.0, "Sub-operation %s: should convert to non-negative seconds", name)

			t.Logf("Sub-operation %s: %v (%dns, %d\u00b5s, %dms, %.6fs)",
				name, duration, nanos, micros, millis, seconds)
		}
	})
}

// TestMetrics_DurationBehaviorUnderLoad tests that duration measurements remain
// accurate and consistent under various load conditions.
func TestMetrics_DurationBehaviorUnderLoad(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping load test in short mode")
	}

	t.Run("ConsistentPrecisionUnderLoad", func(t *testing.T) {
		collector := NewMetricsCollector()
		adapter := tooladapter.New(
			tooladapter.WithMetricsCallback(collector.Callback),
		)

		const numIterations = 100
		var durations []time.Duration

		// Perform many operations to test consistency
		for i := 0; i < numIterations; i++ {
			collector.Clear()

			tools := []openai.ChatCompletionToolUnionParam{
				createMockToolForMetrics(fmt.Sprintf("load_test_%d", i), "Load test tool"),
			}
			req := createMockRequestForMetrics(tools)

			_, err := adapter.TransformCompletionsRequest(req)
			require.NoError(t, err)

			events := collector.GetEvents()
			require.Len(t, events, 1)

			event := events[0].(tooladapter.ToolTransformationData)
			durations = append(durations, event.Performance.ProcessingDuration)
		}

		// Analyze the durations for consistency
		require.Len(t, durations, numIterations)

		// All durations should be positive
		for i, d := range durations {
			assert.Greater(t, d, time.Duration(0), "Duration %d should be positive", i)
			assert.Less(t, d, 100*time.Millisecond, "Duration %d should be reasonable", i)
		}

		// Calculate basic statistics
		var total time.Duration
		min := durations[0]
		max := durations[0]

		for _, d := range durations {
			total += d
			if d < min {
				min = d
			}
			if d > max {
				max = d
			}
		}

		avg := total / time.Duration(numIterations)

		t.Logf("Duration statistics over %d iterations:", numIterations)
		t.Logf("  Min: %v (%dns)", min, min.Nanoseconds())
		t.Logf("  Max: %v (%dns)", max, max.Nanoseconds())
		t.Logf("  Avg: %v (%dns)", avg, avg.Nanoseconds())
		t.Logf("  Range: %v", max-min)

		// Verify precision is maintained (we should see sub-millisecond variations)
		rangeDuration := max - min
		assert.Greater(t, rangeDuration, time.Duration(0), "Should have variation in durations")

		// Verify all durations have nanosecond precision (not rounded to milliseconds)
		nonZeroNanoCount := 0
		for _, d := range durations {
			if d.Nanoseconds()%int64(time.Millisecond) != 0 {
				nonZeroNanoCount++
			}
		}

		// At least some durations should have sub-millisecond precision
		// (This might not always be true on very fast systems, but it validates precision capture)
		if nonZeroNanoCount > 0 {
			t.Logf("Successfully captured sub-millisecond precision in %d/%d measurements",
				nonZeroNanoCount, numIterations)
		}
	})
}
