# 📝 Tool Adapter Logging Strategy & Best Practices

## Overview

The enhanced tool adapter includes comprehensive structured logging using Go's standard `log/slog` package. This provides production-ready observability for monitoring, debugging, and operational insights.

**Note:** This document focuses on logging events and log analysis. For metrics collection, performance monitoring, and alerting, see [METRICS.md](METRICS.md).

## Key Logging Events

### 🔄 Request Transformation (Outbound)
**When:** Tools are injected into system prompt  
**Level:** `INFO`  
**Message:** `"Transformed request: moved tools into system prompt"`  
**Key Fields:**
- `tool_count`: Number of tools being transformed
- `tool_names`: Array of tool names 
- `prompt_length`: Length of generated system prompt

```json
{
  "time": "2024-01-30T10:30:15Z",
  "level": "INFO", 
  "msg": "Transformed request: moved tools into system prompt",
  "tool_count": 2,
  "tool_names": ["get_weather", "calculate_tip"],
  "prompt_length": 245
}
```

### 🔍 Response Transformation (Inbound)
**When:** Function calls are detected and converted  
**Level:** `INFO`  
**Message:** `"Transformed response: detected and converted function calls"`  
**Key Fields:**
- `function_count`: Number of function calls detected
- `function_names`: Array of function names called
- `content_length`: Length of original response content
- `json_candidates`: Number of JSON blocks found
- `function_arguments`: Function arguments (DEBUG level only)

```json
{
  "time": "2024-01-30T10:30:15Z",
  "level": "INFO",
  "msg": "Transformed response: detected and converted function calls", 
  "function_count": 2,
  "function_names": ["get_weather", "calculate_tip"],
  "content_length": 134,
  "json_candidates": 1
}
```

### 📊 Performance Monitoring
**Level:** `DEBUG`  
**Events:**
- Tool prompt generation timing (using `time.Duration` for precision)
- JSON parsing performance with nanosecond accuracy
- Function call extraction duration

**Example DEBUG log:**
```json
{
  "time": "2024-01-30T10:30:15Z",
  "level": "DEBUG",
  "msg": "Built tool prompt",
  "tool_count": 3,
  "prompt_length": 387,
  "build_duration": "2.3ms"
}
```

### ⚠️ Error Handling
**Level:** `ERROR`  
**Events:**
- Tool prompt generation failures
- JSON parsing errors
- Function call validation errors

## Configuration Options

### WithLogger(logger *slog.Logger)
Provides a custom logger instance for complete control over formatting, output destination, and levels.

```go
// Production: JSON to stdout
prodLogger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
    Level: slog.LevelInfo,
}))

adapter := tooladapter.New(
    tooladapter.WithLogger(prodLogger),
)
```

### WithLogLevel(level slog.Level)
Convenience option for controlling log level with default text formatting.

```go
adapter := tooladapter.New(
    tooladapter.WithLogLevel(slog.LevelDebug),
)
```

### Pre-configured Options
Use explicit configuration:
 - WithLogger(...) for custom handlers (JSON/text)
 - WithLogLevel(...) for simple level-only control

## Production Best Practices

### 1. Use Structured JSON Logging
```go
logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
    Level: slog.LevelInfo,
})).With(
    "service", "my-api",
    "version", "1.2.3",
    "environment", "production",
)
```

### 2. Set Appropriate Log Levels
- **Production:** `INFO` level for operational events
- **Development:** `DEBUG` level for detailed troubleshooting
- **Testing:** `ERROR` level to minimize noise

### 3. Add Request Context
```go
// Add request-specific context
logger := slog.With(
    "request_id", requestID,
    "user_id", userID,
    "correlation_id", correlationID,
)

adapter := tooladapter.New(tooladapter.WithLogger(logger))
```

### 4. Secure Sensitive Data
The adapter automatically excludes function arguments from INFO level logs. They only appear at DEBUG level, making it safe for production use.

## Development Workflow

### Local Development
```go
adapter := tooladapter.New(tooladapter.WithLogLevel(slog.LevelDebug))
```
- Text formatting for readability
- DEBUG level for detailed information
- Source code locations included

### Testing
```go  
adapter := tooladapter.New(tooladapter.WithLogLevel(slog.LevelError))
```
- ERROR level only to reduce test noise
- Focus on functionality over logging

### CI/CD Integration
```go
// Use environment variable to control logging
level := slog.LevelInfo
if os.Getenv("LOG_LEVEL") == "debug" {
    level = slog.LevelDebug
}

adapter := tooladapter.New(tooladapter.WithLogLevel(level))
```

## Log Analysis Examples

### Finding Performance Issues
```bash
# Find slow transformations (durations are logged as human-readable strings)
cat logs.json | jq 'select(.msg | contains("Built tool prompt")) | select(.build_duration | test("^[0-9]+[0-9.]+(ms|s)"))'

# Performance analysis across operations
cat logs.json | jq 'select(.build_duration) | {tools: .tool_count, duration: .build_duration}'
```

### Monitoring Tool Usage
```bash
# Most popular tools
cat logs.json | jq -r '.function_names[]?' | sort | uniq -c | sort -nr

# Function call patterns
cat logs.json | jq 'select(.function_count > 1) | {count: .function_count, names: .function_names}'
```

### Streaming Analysis
```bash
# Streaming vs non-streaming detection
cat logs.json | jq 'select(.streaming == true) | {streaming: .streaming, functions: .function_names}'
cat logs.json | jq 'select(.streaming == false) | {streaming: .streaming, functions: .function_names}'
```

### Performance Trend Analysis
```bash
# Extract timing information from debug logs
cat logs.json | jq 'select(.build_duration) | {
    timestamp: .time,
    operation: .msg,
    duration: .build_duration,
    tool_count: .tool_count
}'

# Find operations taking longer than expected
cat logs.json | jq 'select(.build_duration | test("^[0-9]+(ms|μs|ns)")) | select(.build_duration | test("^[1-9][0-9]+(ms|s)"))'
```

## Integration with Metrics

Logging provides operational visibility through structured events, while metrics provide quantitative measurements for monitoring and alerting. For comprehensive production observability:

- **Use logging** for debugging, error tracking, and operational event analysis
- **Use metrics** for performance monitoring, alerting, and dashboard creation  
- **Combine both** for complete system observability

### Timing Data Comparison

**Logging** provides human-readable timing information in log events:
```json
{
  "msg": "Built tool prompt",
  "build_duration": "2.3ms",
  "tool_count": 3
}
```

**Metrics** provide precise `time.Duration` values for quantitative analysis:
```go
// Metrics callback receives precise timing
eventData.Performance.ProcessingDuration // time.Duration with nanosecond precision
```

See [METRICS.md](METRICS.md) for detailed metrics integration, including:
- Performance monitoring with Prometheus/DataDog using precise `time.Duration` measurements
- Function call success rates and timing distributions
- Processing time breakdowns and optimization insights

This logging strategy provides comprehensive observability for debugging and operational awareness, complementing the metrics system for complete production monitoring with both human-readable logs and precise quantitative measurements.
