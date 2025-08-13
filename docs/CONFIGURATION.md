# ⚙️ Configuration Guide

This guide provides comprehensive configuration options for the OpenAI Tool Adapter, including detailed usage examples and best practices for different deployment scenarios.

## Quick Start

```go
// Basic adapter with default settings
adapter := tooladapter.New()

// Adapter with custom configuration
adapter := tooladapter.New(
    tooladapter.WithCustomPromptTemplate(template),
    tooladapter.WithLogger(logger),
    tooladapter.WithMetricsCallback(callback),
)

// Error-only logging
// import "log/slog"
adapter := tooladapter.New(tooladapter.WithLogLevel(slog.LevelError))
```

## Configuration Options

### WithCustomPromptTemplate(template string)

Overrides the default system prompt template used for tool instruction injection.

**Parameters:**
- `template` - Custom template string with `%s` placeholder for tool definitions

**Default Template:**
```text
System/tooling instructions:

You have access to the following functions. When a function call is needed, respond immediately (starting at the first token) with a single JSON array of tool calls, and include no natural-language text before or after the JSON.

Available functions:
%s

Formatting requirements:
- Output must be valid JSON only (no code fences).
- Structure: [{"name": "function_name", "parameters": {…}}] (use null if there are no parameters).
- If multiple calls are required, include them all in the single JSON array.

Decision policy:
- Use tools when they are required to answer correctly or efficiently; otherwise reply in natural language without calling any tools.
```

**Usage:**
```go
customTemplate := `You are an AI assistant. When you need to use tools, respond with JSON in this format:

Tools available:
%s

Use JSON array format: [{"name": "tool_name", "parameters": {...}}]`

adapter := tooladapter.New(
    tooladapter.WithCustomPromptTemplate(customTemplate),
)
```

**Template Requirements:**
- Must contain exactly one `%s` placeholder for tool definitions
- Should provide clear instructions for JSON format
- Should specify when to use tools vs. natural language responses

**Template Validation:**
- Invalid templates (missing `%s` or multiple placeholders) fall back to default template
- Template validation happens at adapter creation time

### WithLogger(logger *slog.Logger)

Sets a custom structured logger for operational events and debugging.

**Usage:**
```go
// Production JSON logging
prodLogger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
    Level: slog.LevelInfo,
}))

// Development text logging
devLogger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
    Level: slog.LevelDebug,
    AddSource: true,
}))

adapter := tooladapter.New(tooladapter.WithLogger(prodLogger))
```

**Nil Logger Handling:**
- Passing `nil` creates a no-op logger that discards all log messages
- Safe for testing environments where logging is undesired

### WithLogLevel(level slog.Level)

Convenience option for setting log level with default text formatting.

**Available Levels:**
- `slog.LevelDebug` - Detailed operational information
- `slog.LevelInfo` - Standard operational events  
- `slog.LevelWarn` - Warning conditions
- `slog.LevelError` - Error conditions only

**Usage:**
```go
// Debug logging for development
adapter := tooladapter.New(tooladapter.WithLogLevel(slog.LevelDebug))

// Error-only logging for production
adapter := tooladapter.New(tooladapter.WithLogLevel(slog.LevelError))
```

**Note:** This creates a default text handler. For JSON logging or custom formatting, use `WithLogger()` instead.

### WithMetricsCallback(callback func(MetricEventData))

Enables metrics collection through a callback function for integration with monitoring systems.

**Callback Function:**
- Receives `MetricEventData` interface for type-safe event handling
- Called synchronously during request/response processing
- Should be fast to avoid blocking request processing

**Usage:**
```go
adapter := tooladapter.New(
    tooladapter.WithMetricsCallback(func(data tooladapter.MetricEventData) {
        switch eventData := data.(type) {
        case tooladapter.ToolTransformationData:
            metrics.TransformationCount.Inc()
            metrics.ProcessingTime.Observe(eventData.Performance.ProcessingDuration.Seconds())
            
        case tooladapter.FunctionCallDetectionData:
            metrics.FunctionCallCount.Add(float64(eventData.FunctionCount))
            metrics.DetectionTime.Observe(eventData.Performance.ProcessingDuration.Seconds())
        }
    }),
)
```

**Performance Considerations:**
- Keep callbacks fast - they run synchronously
- For expensive operations, use buffered channels or background goroutines
- Avoid database writes, HTTP calls, or file I/O in callbacks

### WithSystemMessageSupport(supported bool)

Configures whether the target model supports system messages, affecting how tool instructions are injected into the conversation.

**Parameters:**
- `supported` - Set to `true` if the model supports system messages, `false` otherwise

**Default:** `false` (optimized for models like Gemma that lack system role support)

**Behavior:**
The adapter uses an intelligent message injection strategy based on this setting:

1. **If system messages exist**: Tool instructions are always appended to the last system message (regardless of this setting)
2. **If no system messages exist:**
   - When `supported=false`: Tool instructions are prepended to the first user message content, preserving multimodal content and avoiding system role conflicts
   - When `supported=true`: Tool instructions are added as a new system message at the beginning
3. **If no messages at all**: Creates instruction message with role based on this setting

**Usage:**
```go
// For models without system message support (e.g., Gemma)
adapter := tooladapter.New(
    tooladapter.WithSystemMessageSupport(false), // Default
)

// For models with system message support (e.g., most OpenAI-compatible models)
adapter := tooladapter.New(
    tooladapter.WithSystemMessageSupport(true),
)
```

**Model Compatibility Examples:**
- **Set to `false` for:** Gemma, models with chat templates that ignore system role
- **Set to `true` for:** GPT-4, Claude, most OpenAI-compatible models with proper system role support

**Important Notes:**
- This setting preserves multimodal content (images, audio) when modifying user messages by using intelligent content merging
- The adapter automatically detects and handles existing system messages optimally
- Choose based on your model's actual capabilities, not the API endpoint being used

## Tool Processing Policies

### Policy quick reference

- ToolStopOnFirst (default)
    - Stops at first valid tool call; clears content.
    - Streaming: emits a single tool_calls delta, then ends; optionally closes upstream early with WithCancelUpstreamOnStop(true) while shielding consumers from context.Canceled.
    - Non-streaming: returns only the first tool; finish_reason="tool_calls".
    - Use for lowest latency and safest behavior.

- ToolCollectThenStop
    - Suppresses content once tools begin; collects tools until earliest of: closing bracket of the tool_calls array, ToolMaxCalls, ToolCollectWindow (if > 0), ToolCollectMaxBytes (if > 0), or upstream end.
    - Streaming: emits a single tool_calls delta when collection ends; optionally closes upstream early with WithCancelUpstreamOnStop(true).
    - Non-streaming: timing flags ignored; applies caps and returns collected tools; finish_reason="tool_calls".
    - Use for short batching of tools with predictable limits.

- ToolDrainAll
    - Suppresses all content; reads to EOS and returns all detected tools; finish_reason="tool_calls".
    - No early upstream close; prioritizes completeness over latency.
    - Use for full extraction or offline processing.

- ToolAllowMixed
    - Streams content and tools together; does not suppress post-tool content.
    - Emits tool_calls events as they arrive; finish_reason may be "tool_calls" if any tools detected.
    - No early upstream close.
    - Use for conversational UX that preserves assistant text.

Defaults: ToolPolicy=ToolStopOnFirst, ToolCollectWindow=200ms, ToolMaxCalls=8, ToolCollectMaxBytes=0, CancelUpstreamOnStop=true.

### Prompt injection and roles

- The adapter injects tool instructions into the conversation using the `WithSystemMessageSupport` setting to handle model-specific message role requirements.
- Role selection strategy:
    - If any existing system message is present, tool instructions are appended to the last system message (regardless of `WithSystemMessageSupport` setting).
    - If `supported=true` and no system messages exist, tool instructions are added as a new system message at the beginning.
    - If `supported=false` and no system messages exist, tool instructions are prepended to the first user message content to preserve multimodal content and avoid system role conflicts.
- This preserves compatibility with providers that lack a strict "system" role while keeping system semantics when available.

### WithToolPolicy(policy ToolPolicy)

Controls how tool calls are detected, processed, and emitted in both streaming and non-streaming modes.

**Available Policies:**
- `ToolStopOnFirst` - Stop processing on first tool call (default, lowest latency)
- `ToolCollectThenStop` - Collect tools until limits/timeout, then stop content emission
- `ToolDrainAll` - Process entire response, collect all tools, suppress content
- `ToolAllowMixed` - Allow both text content and tools to be emitted together

**Usage:**
```go
// Default: Stop on first tool call (lowest latency, safest)
adapter := tooladapter.New(tooladapter.WithToolPolicy(tooladapter.ToolStopOnFirst))

// Collect multiple tools within timeout window
adapter := tooladapter.New(tooladapter.WithToolPolicy(tooladapter.ToolCollectThenStop))

// Process entire response, collect all tools
adapter := tooladapter.New(tooladapter.WithToolPolicy(tooladapter.ToolDrainAll))

// Allow mixed content and tools
adapter := tooladapter.New(tooladapter.WithToolPolicy(tooladapter.ToolAllowMixed))
```

**Policy Behaviors:**

| Policy | Content Handling | Tool Processing | Best For |
|--------|------------------|-----------------|----------|
| `ToolStopOnFirst` | Cleared after first tool | First tool only | Low latency, simple workflows |
| `ToolCollectThenStop` | Cleared after collection | Multiple tools (limited) | Structured tool batching |
| `ToolDrainAll` | Cleared after first tool | All detected tools | Complete tool extraction |
| `ToolAllowMixed` | Preserved | All detected tools | Mixed content/tool responses |

### WithToolCollectWindow(duration time.Duration)

Sets maximum collection timeout for `ToolCollectThenStop` policy in streaming mode.

**Parameters:**
- `duration` - Maximum time to wait for additional tools (streaming only)
- `0` - Structure-only batching (no timer), waits for JSON structure completion

**Usage:**
```go
// 500ms collection window
adapter := tooladapter.New(
    tooladapter.WithToolPolicy(tooladapter.ToolCollectThenStop),
    tooladapter.WithToolCollectWindow(500 * time.Millisecond),
)

// Structure-only batching (no timeout)
adapter := tooladapter.New(
    tooladapter.WithToolPolicy(tooladapter.ToolCollectThenStop),
    tooladapter.WithToolCollectWindow(0),
)
```

**Default:** 200ms

### WithToolMaxCalls(maxCalls int)

Sets maximum number of tool calls to process across all policies.

**Parameters:**
- `maxCalls` - Maximum tool calls to process (0 = unlimited)
- Applies to both streaming and non-streaming modes
- Provides safety cap against excessive tool processing

**Usage:**
```go
// Limit to 3 tool calls maximum
adapter := tooladapter.New(
    tooladapter.WithToolPolicy(tooladapter.ToolDrainAll),
    tooladapter.WithToolMaxCalls(3),
)

// No limit (not recommended for production)
adapter := tooladapter.New(tooladapter.WithToolMaxCalls(0))
```

**Default:** 8

### WithToolCollectMaxBytes(maxBytes int)

Sets maximum bytes to collect during tool processing as safety limit.

**Parameters:**
- `maxBytes` - Maximum bytes to buffer during tool collection (0 = unlimited)
- Prevents memory exhaustion from malformed or malicious responses
- Applies primarily to streaming mode buffering

**Usage:**
```go
// 64KB safety limit
adapter := tooladapter.New(
    tooladapter.WithToolPolicy(tooladapter.ToolDrainAll),
    tooladapter.WithToolCollectMaxBytes(64 * 1024),
)

// No byte limit
adapter := tooladapter.New(tooladapter.WithToolCollectMaxBytes(0))
```

**Default:** 0 (unlimited)

### WithCancelUpstreamOnStop(cancel bool)

Controls whether upstream context is cancelled when tool processing stops in streaming mode.

**Parameters:**
- `cancel` - Whether to cancel upstream when stopping tool collection
- Applies only to `ToolStopOnFirst` and `ToolCollectThenStop` policies
- Helps conserve resources by stopping LLM generation early

**Behavior (Streaming):**
- When enabled, the adapter actively closes the upstream streaming transport as soon as:
    - `ToolStopOnFirst`: the first valid tool call is emitted, or
    - `ToolCollectThenStop`: collection completes due to array-close, timeout window, max-calls, max-bytes, or upstream end.
- Consumers are shielded from cancellation errors:
    - The adapter masks `context.Canceled` from the underlying stream. `stream.Err()` returns `nil` for this intentional shutdown.
    - The tool_calls delta is emitted, and the stream completes cleanly (finish_reason remains usable for downstream logic).
- No effect in non-streaming mode.
- Ignored for `ToolAllowMixed` and `ToolDrainAll` (these policies do not stop early).

**Usage:**
```go
// Cancel upstream after tool detection (default)
adapter := tooladapter.New(
    tooladapter.WithToolPolicy(tooladapter.ToolStopOnFirst),
    tooladapter.WithCancelUpstreamOnStop(true),
)

// Continue processing upstream even after tool detection
adapter := tooladapter.New(
    tooladapter.WithCancelUpstreamOnStop(false),
)
```

**Default:** true

**Notes:**
- This option prioritizes latency and cost savings by halting generation early while maintaining a graceful consumer experience.
- If upstream close fails, the adapter still shields `context.Canceled` and completes emission of the tool_calls event.

## Buffer Management Options

### WithStreamingToolBufferSize(limitBytes int)

Sets the maximum amount of content that can be buffered while parsing streaming responses for tool calls.

**Parameters:**
- `limitBytes` - Maximum bytes to buffer during streaming tool call parsing
- Prevents memory exhaustion during streaming by limiting buffered content
- When exceeded, buffered content is processed as regular text instead of searching for tool calls

**Usage:**
```go
// Default 10MB streaming buffer
adapter := tooladapter.New() // Uses 10MB default

// Memory-constrained environment (1MB)
adapter := tooladapter.New(
    tooladapter.WithStreamingToolBufferSize(1 * 1024 * 1024),
)

// High-throughput environment (50MB)
adapter := tooladapter.New(
    tooladapter.WithStreamingToolBufferSize(50 * 1024 * 1024),
)

// Testing with very small buffer (1KB)
adapter := tooladapter.New(
    tooladapter.WithStreamingToolBufferSize(1024),
)
```

**Use Cases:**
- **Increase** for models that generate very large tool calls with complex parameters
- **Decrease** for memory-constrained environments (IoT, edge devices)
- **Set very low** for testing buffer overflow behavior in development

**Default:** 10MB (10 * 1024 * 1024 bytes)

### WithPromptBufferReuseLimit(thresholdBytes int)

Sets the maximum size of prompt generation buffers that will be returned to the buffer pool for reuse.

**Parameters:**
- `thresholdBytes` - Maximum buffer size to return to pool for reuse
- Larger buffers are garbage collected instead of pooled
- Prevents buffer pool from growing unbounded with very large tool definitions

**Usage:**
```go
// Default 64KB threshold
adapter := tooladapter.New() // Uses 64KB default

// Memory-sensitive environment (8KB threshold)
adapter := tooladapter.New(
    tooladapter.WithPromptBufferReuseLimit(8 * 1024),
)

// Large tool schemas (256KB threshold)
adapter := tooladapter.New(
    tooladapter.WithPromptBufferReuseLimit(256 * 1024),
)

// Testing pool behavior (very small threshold)
adapter := tooladapter.New(
    tooladapter.WithPromptBufferReuseLimit(512),
)
```

**Use Cases:**
- **Increase** for applications with consistently large tool schemas and high throughput
- **Decrease** for memory-sensitive environments to limit buffer pool memory usage
- **Set very low** for testing buffer pool discard behavior in development

**Default:** 64KB (64 * 1024 bytes)

### WithStreamingEarlyDetection(lookAheadChars int)

Enables early tool call detection in streaming responses by looking ahead within the first N characters of content for tool call patterns to prevent mixed content/tool responses.

**Parameters:**
- `lookAheadChars` - Number of characters to search ahead for tool call patterns
- `0` - Disabled (default), uses only immediate JSON detection
- Recommended values: 80-120 characters for good balance of recall vs false positives

**How It Works:**
- Searches for tool call patterns like `{"name":` or `[{"name":` within the specified character limit
- When found, starts buffering immediately instead of emitting the preface text
- Without this: "Let me help you with that." gets emitted, then tool calls follow (mixed response)
- With this: The preface text is buffered and suppressed, only tool calls are emitted (clean response)
- Most beneficial for `ToolStopOnFirst` and `ToolCollectThenStop` policies that suppress content
- Causes a short buffering delay (~80 characters) before streaming to prevent mixed responses

**Usage:**
```go
// Disabled by default - only immediate JSON detection
adapter := tooladapter.New() // Uses 0 (disabled)

// Enable with 80-character lookahead (recommended)
adapter := tooladapter.New(
    tooladapter.WithStreamingEarlyDetection(80),
    tooladapter.WithToolPolicy(tooladapter.ToolStopOnFirst),
)

// More generous 120-character lookahead
adapter := tooladapter.New(
    tooladapter.WithStreamingEarlyDetection(120),
    tooladapter.WithToolPolicy(tooladapter.ToolCollectThenStop),
)

// Conservative 40-character lookahead
adapter := tooladapter.New(
    tooladapter.WithStreamingEarlyDetection(40),
)
```

**When to Enable:**
- Models frequently add explanatory text before tool calls
- Using `ToolStopOnFirst` or `ToolCollectThenStop` policies and want cleaner suppression
- Want to prevent mixed content/tool responses in streaming output

**When to Keep Disabled:**
- Maximum performance with minimal overhead is priority
- Models already emit immediate JSON without preface text
- Using `ToolAllowMixed` policy (streams content regardless of tool calls)

**Default:** 0 (disabled)

### Buffer Configuration Examples

```go
// Memory-constrained deployment (IoT/edge devices)
adapter := tooladapter.New(
    tooladapter.WithStreamingToolBufferSize(1 * 1024 * 1024), // 1MB streaming limit
    tooladapter.WithPromptBufferReuseLimit(8 * 1024),         // 8KB pool limit
    tooladapter.WithLogLevel(slog.LevelWarn),                 // Minimal logging
)

// High-throughput server deployment
adapter := tooladapter.New(
    tooladapter.WithStreamingToolBufferSize(50 * 1024 * 1024), // 50MB streaming limit
    tooladapter.WithPromptBufferReuseLimit(256 * 1024),        // 256KB pool limit
    tooladapter.WithLogger(prodLogger),
    tooladapter.WithMetricsCallback(metricsCallback),
)

// Security-focused deployment (minimal memory exposure)
adapter := tooladapter.New(
    tooladapter.WithStreamingToolBufferSize(512 * 1024), // 512KB streaming limit
    tooladapter.WithPromptBufferReuseLimit(4 * 1024),    // 4KB pool limit
    tooladapter.WithLogLevel(slog.LevelError),           // Error-only logging
)

// Testing/development with observable buffer behavior
adapter := tooladapter.New(
    tooladapter.WithStreamingToolBufferSize(1024),      // 1KB (triggers limits quickly)
    tooladapter.WithPromptBufferReuseLimit(512),        // 512B (forces pool discarding)
    tooladapter.WithLogLevel(slog.LevelDebug),          // Observe buffer behavior
)
```

## Pre-configured Option Sets

### Logging presets

Optimized configuration for production deployments with comprehensive observability.

**Configuration:**
- JSON structured logging at INFO level
- Metrics collection enabled (requires callback setup)
- Error handling optimized for production stability

**Usage:**
```go
adapter := tooladapter.New(tooladapter.WithLogLevel(slog.LevelInfo))

// Or combine with additional options
adapter := tooladapter.New(append(
    tooladapter.WithLogLevel(slog.LevelInfo),
    tooladapter.WithMetricsCallback(myMetricsCallback),
)...)
```

**Best For:**
- Production deployments
- Microservice environments
- Container deployments
- Cloud native applications

### Presets removed

The previous preset helpers (ProductionOptions, DevelopmentOptions, QuietOptions) have been removed in favor of explicit configuration using WithLogger or WithLogLevel.

**Best For:**
- Unit testing  
- CI/CD pipelines
- Embedded systems
- Batch processing

## Advanced Configuration Patterns

### Environment-Based Configuration

```go
func createAdapter() *tooladapter.Adapter {
    switch os.Getenv("ENVIRONMENT") {
    case "production":
        return tooladapter.New(append(
            tooladapter.WithLogger(prodLogger),
            tooladapter.WithMetricsCallback(prometheusCallback),
            tooladapter.WithToolPolicy(tooladapter.ToolStopOnFirst), // Safest for production
            tooladapter.WithToolMaxCalls(5), // Conservative limit
        )...)
        
    case "development":
        return tooladapter.New(append(
            tooladapter.WithLogger(devLogger),
            tooladapter.WithToolPolicy(tooladapter.ToolAllowMixed), // See content and tools
        )...)
        
    case "test":
        return tooladapter.New(append(
            tooladapter.WithLogLevel(slog.LevelError),
            tooladapter.WithToolPolicy(tooladapter.ToolDrainAll), // Complete testing
        )...)
        
    default:
        return tooladapter.New() // Default configuration (ToolStopOnFirst)
    }
}
```

### Service-Specific Configuration

```go
// API Gateway service - prioritizes speed and safety
func createAPIAdapter() *tooladapter.Adapter {
    logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
        Level: slog.LevelInfo,
    })).With(
        "service", "api-gateway",
        "version", version,
    )
    
    return tooladapter.New(
        tooladapter.WithLogger(logger),
        tooladapter.WithMetricsCallback(createMetricsCallback("api-gateway")),
        tooladapter.WithToolPolicy(tooladapter.ToolStopOnFirst), // Lowest latency
        tooladapter.WithToolMaxCalls(3), // Strict limit for API responses
        tooladapter.WithCancelUpstreamOnStop(true), // Save resources
    )
}

// Background processor service - processes complex workflows
func createProcessorAdapter() *tooladapter.Adapter {
    return tooladapter.New(
        tooladapter.WithLogLevel(slog.LevelWarn), // Quiet background processing
        tooladapter.WithMetricsCallback(batchMetricsCallback),
        tooladapter.WithToolPolicy(tooladapter.ToolDrainAll), // Complete processing
        tooladapter.WithToolMaxCalls(0), // No limits for batch processing
        tooladapter.WithToolCollectMaxBytes(1024*1024), // 1MB safety limit
    )
}

// Chat service - preserves conversational context
func createChatAdapter() *tooladapter.Adapter {
    return tooladapter.New(
        tooladapter.WithLogger(chatLogger),
        tooladapter.WithMetricsCallback(chatMetricsCallback),
        tooladapter.WithToolPolicy(tooladapter.ToolAllowMixed), // Preserve conversation
        tooladapter.WithToolMaxCalls(5), // Reasonable for chat
    )
}
```

### Multiple Adapter Instances

```go
// Different adapters for different use cases
type Services struct {
    UserAdapter    *tooladapter.Adapter // User-facing requests
    InternalAdapter *tooladapter.Adapter // Internal processing
    TestAdapter    *tooladapter.Adapter // Testing/validation
}

func NewServices() *Services {
    return &Services{
        UserAdapter: tooladapter.New(
            tooladapter.WithLogger(userLogger),
            tooladapter.WithMetricsCallback(userMetricsCallback),
        ),
        
        InternalAdapter: tooladapter.New(
            tooladapter.WithLogLevel(slog.LevelWarn),
            tooladapter.WithMetricsCallback(internalMetricsCallback),
        ),
        
    TestAdapter: tooladapter.New(tooladapter.WithLogLevel(slog.LevelError)),
    }
}
```

## Configuration Validation

### Template Validation

```go
// Valid templates
validTemplates := []string{
    "Custom instructions: %s",
    "Tools available:\n%s\nUse JSON format for responses.",
    "%s", // Minimal template
}

// Invalid templates (will use default)
invalidTemplates := []string{
    "No placeholder",           // Missing %s
    "Multiple %s and %s",       // Multiple placeholders
    "",                         // Empty template
}
```

### Logger Validation

```go
// Valid logger configurations
logger1 := slog.New(slog.NewJSONHandler(os.Stdout, nil))     // JSON to stdout
logger2 := slog.New(slog.NewTextHandler(os.Stderr, nil))     // Text to stderr
logger3 := slog.New(customHandler)                           // Custom handler

// Special case handling
adapter := tooladapter.New(tooladapter.WithLogger(nil))      // Creates no-op logger
```

## Best Practices

### Production Deployment

```go
// Production-ready configuration
func NewProductionAdapter(serviceName string) *tooladapter.Adapter {
    // Structured logger with service context
    logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
        Level: slog.LevelInfo,
    })).With(
        "service", serviceName,
        "version", os.Getenv("SERVICE_VERSION"),
        "environment", "production",
    )
    
    // Metrics callback with error handling
    metricsCallback := func(data tooladapter.MetricEventData) {
        defer func() {
            if r := recover(); r != nil {
                logger.Error("Metrics callback panic", "error", r)
            }
        }()
        
        // Send to monitoring system
        prometheus.RecordEvent(data)
    }
    
    return tooladapter.New(
        tooladapter.WithLogger(logger),
        tooladapter.WithMetricsCallback(metricsCallback),
    )
}
```

### Development Workflow

```go
// Development configuration with debugging
func NewDevelopmentAdapter() *tooladapter.Adapter {
    // Console-friendly logging
    logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
        Level: slog.LevelDebug,
        AddSource: true,
    }))
    
    // Simple metrics logging
    metricsCallback := func(data tooladapter.MetricEventData) {
        switch eventData := data.(type) {
        case tooladapter.ToolTransformationData:
            logger.Debug("Tool transformation",
                "tool_count", eventData.ToolCount,
                "duration", eventData.Performance.ProcessingDuration,
            )
        }
    }
    
    return tooladapter.New(
        tooladapter.WithLogger(logger),
        tooladapter.WithMetricsCallback(metricsCallback),
    )
}
```

### Testing Configuration

```go
// Test-friendly configuration
func NewTestAdapter() *tooladapter.Adapter {
    // Capture logs for test assertions
    var logBuffer bytes.Buffer
    logger := slog.New(slog.NewJSONHandler(&logBuffer, &slog.HandlerOptions{
        Level: slog.LevelDebug,
    }))
    
    // Capture metrics for test assertions
    var capturedMetrics []tooladapter.MetricEventData
    metricsCallback := func(data tooladapter.MetricEventData) {
        capturedMetrics = append(capturedMetrics, data)
    }
    
    return tooladapter.New(
        tooladapter.WithLogger(logger),
        tooladapter.WithMetricsCallback(metricsCallback),
    )
}
```

## Configuration Migration

### From Version 1.x

```go
// Old configuration (v1.x)
adapter := tooladapter.NewWithOptions(tooladapter.Options{
    Logger: logger,
    Template: template,
})

// New configuration (v2.x+)
adapter := tooladapter.New(
    tooladapter.WithLogger(logger),
    tooladapter.WithCustomPromptTemplate(template),
)
```

### Gradual Adoption

```go
// Start with minimal configuration
adapter := tooladapter.New()

// Add logging when needed
adapter = tooladapter.New(tooladapter.WithLogLevel(slog.LevelInfo))

// Add metrics when monitoring is ready
adapter = tooladapter.New(
    tooladapter.WithLogLevel(slog.LevelInfo),
    tooladapter.WithMetricsCallback(metricsCallback),
)

// Move to production configuration
adapter = tooladapter.New(tooladapter.WithLogger(prodLogger))
```

## Troubleshooting

### Common Issues

**Template Not Applied:**
- Verify template contains exactly one `%s` placeholder
- Check for syntax errors in template string
- Invalid templates fall back to default (check logs)

**Missing Logs:**
- Verify logger is not nil
- Check log level configuration
- Ensure handler destination is accessible (stdout, file, etc.)

**Metrics Not Firing:**
- Verify callback function is set
- Check for panics in callback function
- Ensure callback completes quickly

**Performance Issues:**
- Profile metrics callback performance
- Consider async processing for expensive metrics operations
- Check logging level and output destination performance

### Debug Configuration

```go
// Debug configuration issues
func debugAdapter() *tooladapter.Adapter {
    logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
        Level: slog.LevelDebug,
        AddSource: true,
    }))
    
    metricsCallback := func(data tooladapter.MetricEventData) {
        logger.Debug("Metrics event", "type", data.EventType())
    }
    
    adapter := tooladapter.New(
        tooladapter.WithLogger(logger),
        tooladapter.WithMetricsCallback(metricsCallback),
    )
    
    // Test configuration
    logger.Info("Adapter configured successfully")
    return adapter
}
```

This configuration guide provides comprehensive coverage of all configuration options with practical examples for different deployment scenarios and use cases.