# 📈 Tool Adapter Metrics Guide

## Overview

The Tool Adapter provides comprehensive metrics through a type-safe callback system that enables integration with monitoring platforms like Prometheus, DataDog, New Relic, or custom metrics collection systems.

The metrics system emits structured events at key business operations, providing visibility into:
- Tool transformation performance and processing times
- Function call detection accuracy and parsing metrics  
- Detailed performance breakdowns for optimization

## Quick Start

```go
import (
	"github.com/juburr/openai-tool-adapter"
    "log/slog"
    "time"
)

// Create adapter with metrics callback
adapter := tooladapter.New(
    tooladapter.WithMetricsCallback(func(data tooladapter.MetricEventData) {
        switch eventData := data.(type) {
        case tooladapter.ToolTransformationData:
            // Handle tool transformation metrics
            fmt.Printf("Transformed %d tools, processing time: %v\n", 
                eventData.ToolCount, 
                eventData.Performance.ProcessingDuration)
                
        case tooladapter.FunctionCallDetectionData:
            // Handle function call detection metrics
            fmt.Printf("Detected %d function calls in %v\n",
                eventData.FunctionCount, 
                eventData.Performance.ProcessingDuration)
        }
    }),
    tooladapter.WithLogger(slog.Default()),
)
```

## Event Types

### MetricEventToolTransformation

**When:** Tools are converted from OpenAI format to system prompt format  
**Frequency:** Once per request with tools  
**Data Structure:** `ToolTransformationData`

```go
type ToolTransformationData struct {
    ToolCount    int      `json:"tool_count"`     // Number of tools transformed
    ToolNames    []string `json:"tool_names"`     // Names of tools
    PromptLength int      `json:"prompt_length"`  // Generated prompt length
    Performance  PerformanceMetrics `json:"performance"`
}
```

**Key Metrics:**
- Transformation frequency and tool usage patterns
- Prompt generation performance
- Tool combination analysis

### MetricEventFunctionCallDetection

**When:** Function calls are parsed from LLM responses  
**Frequency:** Once per response containing function calls  
**Data Structure:** `FunctionCallDetectionData`

```go
type FunctionCallDetectionData struct {
    FunctionCount   int      `json:"function_count"`   // Number of calls detected
    FunctionNames   []string `json:"function_names"`   // Function names called
    ContentLength   int      `json:"content_length"`   // Response content length
    JSONCandidates  int      `json:"json_candidates"`  // JSON blocks found
    Streaming       bool     `json:"streaming"`        // Streaming vs batch mode
    Performance     PerformanceMetrics `json:"performance"`
}
```

**Key Metrics:**
- Function call success rates and patterns
- JSON parsing efficiency 
- Response processing performance
- Streaming vs batch performance comparison

### Performance Metrics

All events include detailed performance data with nanosecond precision:

```go
type PerformanceMetrics struct {
    ProcessingDuration   time.Duration            `json:"processing_duration"`
    MemoryAllocatedBytes int64                    `json:"memory_allocated_bytes,omitempty"`
    SubOperations        map[string]time.Duration `json:"sub_operations,omitempty"`
}
```

**Benefits of time.Duration:**
- **Nanosecond precision** for accurate performance measurement
- **Flexible conversion** - callers can convert to any unit (ms, μs, etc.)
- **Standard Go type** with built-in string formatting and arithmetic

**SubOperations** may include:
- `json_parsing` - Time spent extracting JSON from responses
- `call_extraction` - Time spent parsing function calls
- `prompt_generation` - Time spent building tool prompts

## Integration Examples

### Prometheus Integration

```go
import (
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promauto"
)

// Define Prometheus metrics
var (
    toolTransformations = promauto.NewCounter(prometheus.CounterOpts{
        Name: "tool_adapter_transformations_total",
        Help: "Total number of tool transformations",
    })
    
    processingDuration = promauto.NewHistogram(prometheus.HistogramOpts{
        Name: "tool_adapter_processing_duration_seconds",
        Help: "Processing duration in seconds",
        Buckets: prometheus.DefBuckets,
    })
    
    functionCalls = promauto.NewCounterVec(prometheus.CounterOpts{
        Name: "tool_adapter_function_calls_total", 
        Help: "Total function calls by name",
    }, []string{"function_name", "streaming"})
)

// Create adapter with Prometheus integration
adapter := tooladapter.New(
    tooladapter.WithMetricsCallback(func(data tooladapter.MetricEventData) {
        switch eventData := data.(type) {
        case tooladapter.ToolTransformationData:
            toolTransformations.Inc()
            
            // Convert time.Duration to seconds for Prometheus
            processingDuration.Observe(eventData.Performance.ProcessingDuration.Seconds())
            
        case tooladapter.FunctionCallDetectionData:
            streamingLabel := "false"
            if eventData.Streaming {
                streamingLabel = "true"
            }
            
            for _, funcName := range eventData.FunctionNames {
                functionCalls.WithLabelValues(funcName, streamingLabel).Inc()
            }
            
            // Convert time.Duration to seconds for Prometheus
            processingDuration.Observe(eventData.Performance.ProcessingDuration.Seconds())
        }
    }),
)
```

### DataDog Integration

```go
import "github.com/DataDog/datadog-go/v5/statsd"

// Create DataDog client
client, _ := statsd.New("127.0.0.1:8125")

adapter := tooladapter.New(
    tooladapter.WithMetricsCallback(func(data tooladapter.MetricEventData) {
        switch eventData := data.(type) {
        case tooladapter.ToolTransformationData:
            client.Incr("tool_adapter.transformations", nil, 1)
            
            // time.Duration can be used directly with DataDog
            client.Timing("tool_adapter.processing_time", 
                eventData.Performance.ProcessingDuration, nil, 1)
            
        case tooladapter.FunctionCallDetectionData:
            client.Incr("tool_adapter.function_calls", nil, float64(eventData.FunctionCount))
            client.Timing("tool_adapter.detection_time",
                eventData.Performance.ProcessingDuration, nil, 1)
            
            // Tag by streaming mode
            tags := []string{fmt.Sprintf("streaming:%t", eventData.Streaming)}
            client.Incr("tool_adapter.detections.by_mode", tags, 1)
        }
    }),
)
```

### Custom Database Logging

```go
// For detailed analysis in TimescaleDB or similar
type MetricsDB struct {
    db *sql.DB
}

func (m *MetricsDB) recordMetrics(data tooladapter.MetricEventData) {
    switch eventData := data.(type) {
    case tooladapter.ToolTransformationData:
        // Insert detailed transformation data
        query := `
            INSERT INTO tool_transformations 
            (timestamp, tool_count, tool_names, prompt_length, duration_ns)
            VALUES ($1, $2, $3, $4, $5)`
        
        m.db.Exec(query,
            time.Now(),
            eventData.ToolCount,
            pq.Array(eventData.ToolNames),
            eventData.PromptLength,
            eventData.Performance.ProcessingDuration.Nanoseconds(),
        )
        
    case tooladapter.FunctionCallDetectionData:
        // Insert function call analysis data
        query := `
            INSERT INTO function_detections
            (timestamp, function_count, function_names, content_length, candidates, streaming, duration_ns)
            VALUES ($1, $2, $3, $4, $5, $6, $7)`
            
        m.db.Exec(query,
            time.Now(),
            eventData.FunctionCount,
            pq.Array(eventData.FunctionNames),
            eventData.ContentLength,
            eventData.JSONCandidates,
            eventData.Streaming,
            eventData.Performance.ProcessingDuration.Nanoseconds(),
        )
    }
}
```

### Conversion Examples

```go
// time.Duration provides flexible conversion options
adapter := tooladapter.New(
    tooladapter.WithMetricsCallback(func(data tooladapter.MetricEventData) {
        // Extract performance data from the concrete event type
        var duration time.Duration
        switch eventData := data.(type) {
        case tooladapter.ToolTransformationData:
            duration = eventData.Performance.ProcessingDuration
        case tooladapter.FunctionCallDetectionData:
            duration = eventData.Performance.ProcessingDuration
        default:
            return
        }
        
        // Convert to different units as needed
        nanoseconds := duration.Nanoseconds()   // int64 nanoseconds
        microseconds := duration.Microseconds() // int64 microseconds
        milliseconds := duration.Milliseconds() // int64 milliseconds
        seconds := duration.Seconds()           // float64 seconds
        
        // Use formatted string representation
        humanReadable := duration.String()      // "1.23ms", "45.67μs", etc.
        
        // Send to your preferred monitoring system
        myMonitoring.RecordDuration("tool_processing", milliseconds)
        myLogger.Info("Processing completed", "duration", humanReadable)
    }),
)
```

## Monitoring and Alerting

### Key Performance Indicators (KPIs)

1. **Processing Performance**
   - Tool transformation: <50ms p95
   - Function detection: <100ms p95
   - Alert on p95 >200ms for 2 minutes

2. **Success Rates**
   - Function call detection success rate
   - JSON parsing success rate
   - Alert on success rate <95%

3. **System Health**
   - Request processing rate
   - Error rate monitoring
   - Memory usage patterns

### Sample Prometheus Alerting Rules

```yaml
groups:
- name: tool_adapter
  rules:
  - alert: ToolAdapterHighLatency
    expr: histogram_quantile(0.95, rate(tool_adapter_processing_duration_seconds_bucket[5m])) > 0.2
    for: 2m
    annotations:
      summary: "Tool adapter processing latency is high"
      description: "95th percentile processing time is {{ $value }}s"

  - alert: ToolAdapterErrorRate
    expr: rate(tool_adapter_errors_total[5m]) > 0.1
    for: 1m
    annotations:
      summary: "High error rate in tool adapter"
      description: "Error rate is {{ $value }} errors/second"
```

### Integration with Logging

The tool adapter provides both metrics and logging for comprehensive observability:

**Metrics System (this document):**
- Quantitative performance data for monitoring and alerting
- Integration with Prometheus, DataDog, custom systems
- Focus on measurement and trend analysis

**Logging System ([LOGGING.md](LOGGING.md)):**
- Structured operational events for debugging and analysis
- Request tracing and error investigation
- Performance troubleshooting and audit trails
- Focus on operational visibility and debugging

**Combined Usage:**
```go
// Complete observability setup
adapter := tooladapter.New(
    // Structured logging for operational visibility
    tooladapter.WithLogger(slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
        Level: slog.LevelInfo,
    }))),
    
    // Metrics for monitoring and alerting
    tooladapter.WithMetricsCallback(func(data tooladapter.MetricEventData) {
        // Send to your monitoring system with precise timing
        switch eventData := data.(type) {
        case tooladapter.ToolTransformationData:
            prometheus.RecordTransformation(eventData)
        case tooladapter.FunctionCallDetectionData:
            prometheus.RecordDetection(eventData)
        }
    }),
)
```

## Performance Considerations

### Callback Performance

Metrics callbacks run synchronously during request processing. Keep callbacks fast:

```go
// ✅ Good - Fast operations
adapter := tooladapter.New(
    tooladapter.WithMetricsCallback(func(data tooladapter.MetricEventData) {
        // In-memory counters, gauge updates
        myCounters.increment("transformations")
        myGauges.set("processing_time", data.Performance.ProcessingDuration)
    }),
)

// ❌ Avoid - Slow operations that block request processing
adapter := tooladapter.New(
    tooladapter.WithMetricsCallback(func(data tooladapter.MetricEventData) {
        // Database writes, HTTP calls, file I/O
        database.Insert(data) // This blocks the request!
    }),
)
```

### Asynchronous Processing

For expensive operations, use background processing:

```go
// Create a buffered channel for metrics
metricsChan := make(chan tooladapter.MetricEventData, 1000)

// Background goroutine for expensive processing
go func() {
    for data := range metricsChan {
        // Expensive operations here
        database.Insert(data)
        externalAPI.Send(data)
    }
}()

// Fast callback that queues metrics
adapter := tooladapter.New(
    tooladapter.WithMetricsCallback(func(data tooladapter.MetricEventData) {
        select {
        case metricsChan <- data:
            // Queued successfully
        default:
            // Channel full, drop metric or handle overflow
            log.Warn("Metrics channel full, dropping event")
        }
    }),
)
```

## Precision and Performance Benefits

### Nanosecond Precision

Using `time.Duration` provides several advantages:

```go
// Measure very fast operations accurately
start := time.Now()
lightweightOperation()
duration := time.Since(start) // Could be 100ns, 1μs, etc.

// Perfect for micro-benchmarking and optimization
if duration < 100*time.Microsecond {
    metrics.FastOperations.Inc()
} else {
    metrics.SlowOperations.Inc()
}
```

### Flexible Unit Conversion

```go
// Convert to any unit your monitoring system needs
func (m *MyMetrics) Record(duration time.Duration) {
    // Prometheus (seconds)
    m.prometheus.Observe(duration.Seconds())
    
    // DataDog (milliseconds) 
    m.datadog.Timing("operation", duration)
    
    // Custom logging (microseconds)
    m.logger.Info("timing", "duration_us", duration.Microseconds())
    
    // Database storage (nanoseconds for precision)
    m.db.Insert(duration.Nanoseconds())
}
```

This approach ensures maximum precision and flexibility for all monitoring and observability needs.
