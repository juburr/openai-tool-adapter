# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Go package that provides OpenAI tool compatibility for Large Language Models that lack native function calling support. It transforms OpenAI-style tool requests into prompt-based format and parses model responses back into structured tool calls using a finite state machine JSON parser.

## Development Commands

### Testing
```bash
# Run all tests
go test ./...

# Run tests with coverage
go test -race -coverprofile=coverage.out ./...

# View coverage report  
go tool cover -html=coverage.out

# Run with benchmarking
go test -bench=. ./...
```

### Building
```bash
# Standard build
go build

# Build with optimizations
go build -ldflags="-w -s"

# Run with race detection
go run -race main.go
```

### Linting/Formatting
```bash
# Format code
go fmt ./...

# Run go vet
go vet ./...

# Check for modules
go mod tidy
```

## Architecture

### Core Components

- **Adapter** (`adapter.go`): Main transformation engine that converts OpenAI requests/responses
- **State Machine Parser** (`parser.go`): Robust JSON extraction from LLM responses using finite state machine
- **Streaming Support** (`streaming.go`): Real-time tool call detection in streaming responses  
- **Metrics & Observability** (`metrics.go`): Type-safe metrics collection via observer pattern
- **Configuration** (`options.go`): Flexible configuration system with pre-built option sets
- **Multi-Choice Processing**: Support for OpenAI n > 1 parameter with independent choice processing
- **Tool Policies**: Configurable policies for tool call handling and content management

### Key Design Patterns

- **Observer Pattern**: Metrics system allows integration with any monitoring platform without vendor lock-in
- **Finite State Machine**: JSON parsing handles nested structures, escaped characters, edge cases more reliably than regex
- **Object Pool**: Buffer pooling for high-performance prompt generation
- **Functional Options Pattern**: Backward-compatible configuration system
- **Minimal Dependencies**: Uses only standard library and OpenAI SDK

### Request/Response Flow

1. **Request Transformation**: Tools are injected into system prompt, `tools`/`tool_choice` fields removed
2. **Tool Result Processing**: `ToolMessage` types extracted and converted to natural language context
3. **LLM Processing**: Model receives prompt-based instructions for tool calling with previous results
4. **Response Parsing**: State machine extracts JSON tool calls from varied response formats
5. **Structure Reconstruction**: Parsed content rebuilt into OpenAI-compatible format

### Multi-Choice Response Processing

The adapter supports OpenAI's multi-choice responses (n > 1 parameter) with independent processing:
- **Per-Choice Processing**: Each choice in the response is processed independently
- **Policy Application**: Tool policies are applied to each choice individually
- **Race Condition Safety**: Lazy allocation with copy-on-write prevents data races during concurrent processing
- **Performance Optimized**: Zero-overhead for single-choice responses and responses without tool calls
- **Backward Compatible**: Existing single-choice code continues to work unchanged

### Tool Policy System

Configurable policies control how tool calls are processed and content is managed:
- **ToolStopOnFirst**: Stop processing on first tool call detected (default behavior)
- **ToolCollectThenStop**: Collect tools within time/count limits, then stop
- **ToolDrainAll**: Read entire response and collect all detected tools
- **ToolAllowMixed**: Allow both explanatory text and tool calls in the final response

### Multi-turn Tool Conversation Support

The adapter handles tool results in multi-turn conversations by automatically:
- **Detecting ToolMessage types** in conversation history
- **Extracting tool results** and removing them from the message flow
- **Converting to natural language** with context like "Previous tool calls requested by you returned the following results..."
- **Combining with tool definitions** when both are present

This enables models like Gemma 3 (via vLLM) to understand tool results without native `ToolMessage` support.

## Important Files

### Core Implementation
- `adapter.go`: Core transformation logic and main API with helper functions:
  - `processChoiceForToolCalls()`: Per-choice tool extraction and processing
  - `logAndEmitFunctionCalls()`: Centralized logging and metrics emission
  - `applyToolPolicyToChoice()`: Policy application per choice
- `parser.go`: State machine-based JSON extraction (critical for reliability)
- `streaming.go`: Streaming response handling with real-time parsing
- `options.go`: Configuration system with production/development presets
- `metrics.go`: Observability interfaces and event data structures
- `idgen.go`: UUIDv7-based tool call ID generation

### Testing (89.6% coverage)
- `adapter_test.go`: Core adapter functionality tests
- `parser_test.go`: JSON parsing and state machine tests
- `streaming_test.go`: Streaming functionality tests
- `metrics_test.go`: Metrics system tests
- `context_test.go`: Context handling tests
- `options_test.go`: Configuration option tests
- `error_paths_test.go`: Error handling and edge case tests
- `production_edge_cases_test.go`: Production scenario testing
- `integration_test.go`: End-to-end workflow tests
- `multi_choice_test.go`: Multi-choice response processing and policy application
- `tool_policy_comprehensive_test.go`: Comprehensive tool policy system testing
- `streaming_early_detection_test.go`: Streaming early detection feature tests
- `adapter_fuzz_test.go`, `parser_fuzz_test.go`: Fuzz testing for robustness
- `mock_stream_test.go`: Mock streaming infrastructure for tests
- `metrics_panic_test.go`: Panic recovery and error handling in metrics

### Performance & Benchmarks
- `parser_benchmark_test.go`: JSON parsing performance benchmarks
- `metrics_benchmark_test.go`: Metrics system performance tests
- `metrics_duration_test.go`: High-precision timing validation
- `adapter_benchmark_test.go`: End-to-end adapter performance benchmarks

### End-to-End Testing
- `e2e/`: Comprehensive integration test suite with real LLM scenarios
  - `simple_test.go`: Basic tool calling workflows
  - `tool_calling_test.go`: Complex tool interaction patterns
  - `tool_policies_test.go`: Policy behavior validation
  - `config_test.go`: Configuration validation
  - `edge_cases_test.go`: Production edge case handling
  - `streaming_early_detection_test.go`: Streaming feature validation

### JSON Implementation
- All JSON operations use Go's standard library `encoding/json` package

## Configuration System

### Available Options

| Option | Description | Use Case |
|--------|-------------|----------|
| `WithCustomPromptTemplate(string)` | Override default tool prompt template | Custom instruction formatting |
| `WithLogger(*slog.Logger)` | Set custom structured logger | Production logging integration |
| `WithLogLevel(slog.Level)` | Set logging level with default handler | Simple log level control |
| `WithMetricsCallback(func)` | Enable metrics collection | Performance monitoring |
| `WithStreamingToolBufferSize(int)` | Set streaming tool buffer size limit (bytes) | Control streaming memory usage |
| `WithPromptBufferReuseLimit(int)` | Set prompt buffer pool reuse threshold (bytes) | Control buffer pool memory |
| `WithToolPolicy(ToolPolicy)` | Set tool processing policy | Control tool detection behavior |
| `WithToolCollectWindow(time.Duration)` | Set collection timeout window | Time-based tool collection limits |
| `WithToolMaxCalls(int)` | Set maximum tool calls per choice | Prevent excessive tool generation |
| `WithToolCollectMaxBytes(int)` | Set safety limit for JSON collection | Memory protection during collection |
| `WithCancelUpstreamOnStop(bool)` | Cancel upstream on tool detection | Control upstream cancellation |
| `WithStreamingEarlyDetection(int)` | Set early tool detection lookahead chars | Improve streaming tool detection |

### Pre-configured Option Sets

| Option Set | Configuration | Best For |
|------------|---------------|----------|
| `WithLogger()` | Custom logger (JSON or text) | Any environment |
| `WithLogLevel()` | Level-only using default handler | Simple setups |

## Testing Strategy

### Test Structure
- **Unit Tests**: Component-level testing with comprehensive edge cases
- **Integration Tests**: End-to-end workflow validation  
- **Production Edge Case Tests**: Resource exhaustion, malicious input handling
- **Concurrency Tests**: Race condition detection with high-concurrency stress testing
- **Benchmark Tests**: Performance regression detection
- **Error Path Tests**: Comprehensive error handling validation

### Coverage Areas
- Tool transformation and response parsing
- Streaming functionality with buffer management
- Metrics collection and observer pattern
- Configuration validation and option handling
- Context propagation and cancellation
- Memory management and resource limits
- JSON parsing state machine edge cases
- Error recovery and graceful degradation

## Performance Considerations

### Optimizations
- Zero-allocation core string processing operations
- Buffer pooling reduces memory allocations during prompt generation
- State machine parser avoids regex backtracking issues
- Object pools for frequently allocated structures
- High-precision timing metrics use `time.Duration` for nanosecond accuracy
- **Lazy Allocation**: Copy-on-write for multi-choice responses to avoid unnecessary copying
- **Race Condition Prevention**: Thread-safe choice processing with independent copying
- **Performance-Optimized Policies**: Zero overhead for responses without tool calls

### Benchmarks (AMD Ryzen 9 9950X3D)
- **Core rune operations**: 1.34 ns/op, 0 allocs/op
- **JSON extraction**: 3.39 μs/op, 32 allocs/op
- **Function call processing**: 1.03 μs/op, 16 allocs/op
- **Streaming detection**: 115.8 ns/op, 1 alloc/op
- **Single choice with tools**: 2.5 μs/op, 32 allocs/op
- **Ten choices without tools**: 1.6 μs/op, 10 allocs/op (lazy allocation benefit)
- **Multi-choice transformation**: 7.2-23.8 μs/op depending on tool density

### Memory Management
- **Streaming Tool Buffer**: Content buffered during streaming tool parsing is limited by `WithStreamingToolBufferSize()` (default: 10MB)
- **Prompt Buffer Pool**: Buffers exceeding the reuse threshold set by `WithPromptBufferReuseLimit()` (default: 64KB) are discarded
- **Automatic Cleanup**: Oversized buffers are garbage collected rather than pooled to prevent memory leaks
- **DoS Protection**: Configurable limits prevent memory exhaustion attacks through large streaming responses or tool definitions

## Dependencies

### Required
- `github.com/openai/openai-go/v3` (official OpenAI SDK)
- `github.com/google/uuid` (UUIDv7-based tool call ID generation)

### Development
- `github.com/stretchr/testify` (testing framework)

## OpenAI Go SDK Considerations

The official OpenAI Go SDK is used throughout the codebase:

- **Version**: Check `go.mod` for current version
- **Breaking Changes**: SDK was recently in BETA, frequent updates
- **Type Usage**: Always verify types against the current SDK version
- **Compatibility**: LLMs often use older OpenAI types that may no longer work

### Key SDK Types Used
- `openai.ChatCompletionNewParams`
- `openai.ChatCompletion`
- `openai.ChatCompletionMessageParamUnion`
- `openai.ChatCompletionToolParam`
- `openai.FunctionDefinitionParam`
- `openai.ChatCompletionMessageToolCall`

## Observability

### Structured Logging
- Uses Go's standard `log/slog` package
- JSON formatting for production deployments
- Configurable log levels (DEBUG, INFO, WARN, ERROR)
- Operational events with performance timing

### Metrics System
- Observer pattern for vendor-agnostic integration
- Type-safe metric events with performance data
- High-precision timing with `time.Duration`
- Works with Prometheus, DataDog, custom systems

### Key Metrics
- `MetricEventToolTransformation`: Tool processing performance
- `MetricEventFunctionCallDetection`: Function call parsing metrics
- `MetricEventStreamingToolDetection`: Streaming tool detection performance
- **Performance Timing**: All metrics include high-precision duration measurements
- **Memory Tracking**: Buffer usage and allocation patterns

## Recent Architecture Changes

### Cyclomatic Complexity Reduction
Recent refactoring in `adapter.go` reduced cyclomatic complexity by extracting helper functions:
- **Before**: `TransformCompletionsResponseWithContext` had complexity of 16
- **After**: Complexity reduced to <15 by extracting `processChoiceForToolCalls()` and `logAndEmitFunctionCalls()`
- **Benefit**: Improved maintainability and testability

### Race Condition Fixes
Implemented lazy allocation pattern to prevent data races:
- **Problem**: Concurrent access to shared response objects in multi-choice scenarios
- **Solution**: Copy-on-write with lazy allocation only when tool calls are detected
- **Performance**: Zero overhead for responses without tool calls

### Test Stability Improvements
Fixed flaky tests related to random ID generation:
- **Issue**: `TestBackwardCompatibility_OriginalMethodsUnchanged` comparing random UUIDs
- **Fix**: Compare response fields individually, excluding randomly generated tool call IDs
- **Result**: Stable test execution for continuous integration

## Tool Policy Configuration

### Policy Types
```go
type ToolPolicy int

const (
    ToolStopOnFirst      ToolPolicy = iota // Default: stop on first tool
    ToolCollectThenStop                     // Collect within limits, then stop
    ToolDrainAll                            // Read entire response, collect all
    ToolAllowMixed                          // Allow both text and tools
)
```

### Policy Configuration Examples
```go
// Conservative policy - stop immediately on first tool
adapter := tooladapter.New(
    tooladapter.WithToolPolicy(tooladapter.ToolStopOnFirst),
)

// Aggressive collection with limits
adapter := tooladapter.New(
    tooladapter.WithToolPolicy(tooladapter.ToolCollectThenStop),
    tooladapter.WithToolCollectWindow(5*time.Second),
    tooladapter.WithToolMaxCalls(10),
)

// Allow mixed responses (explanatory text + tools)
adapter := tooladapter.New(
    tooladapter.WithToolPolicy(tooladapter.ToolAllowMixed),
)
```

### Streaming Early Detection
```go
// Enable early detection with 100-character lookahead
adapter := tooladapter.New(
    tooladapter.WithStreamingEarlyDetection(100),
)
```

## API Compatibility

### Function Name Changes
Functions were renamed for clarity while maintaining backward compatibility:
- **Old**: `TransformRequest()`, `TransformResponse()`
- **New**: `TransformCompletionsRequest()`, `TransformCompletionsResponse()`
- **Context Support**: All functions have `WithContext` variants
- **Backward Compatibility**: Old names still work but are deprecated

## Common Development Tasks

### Adding New Configuration Options
```go
func WithNewOption(value Type) Option {
    return func(a *Adapter) {
        // Validate and apply configuration
        a.newField = value
    }
}
```

### Extending Metrics
```go
// Add new metric event type
type NewMetricData struct {
    Field1      string            `json:"field1"`
    Performance PerformanceMetrics `json:"performance"`
}

func (d NewMetricData) EventType() MetricEventType {
    return MetricEventNewMetric
}
```

### Testing New Functionality
```go
func TestNewFeature(t *testing.T) {
    adapter := tooladapter.New(tooladapter.WithLogLevel(slog.LevelError))
    
    // Test implementation
    result, err := adapter.NewMethod(input)
    require.NoError(t, err)
    assert.Equal(t, expected, result)
}
```

### Testing Multi-Choice Scenarios
```go
func TestMultiChoiceFeature(t *testing.T) {
    adapter := tooladapter.New(
        tooladapter.WithToolPolicy(tooladapter.ToolAllowMixed),
        tooladapter.WithLogLevel(slog.LevelError),
    )
    
    // Create response with multiple choices
    resp := openai.ChatCompletion{
        Choices: []openai.ChatCompletionChoice{
            {Message: openai.ChatCompletionMessage{Content: "[{\"name\": \"func1\"}]"}},
            {Message: openai.ChatCompletionMessage{Content: "Regular text response"}},
        },
    }
    
    result, err := adapter.TransformCompletionsResponse(resp)
    require.NoError(t, err)
    assert.Len(t, result.Choices, 2)
}
```

### Adding Tool Policy Tests
```go
func TestCustomPolicy(t *testing.T) {
    adapter := tooladapter.New(
        tooladapter.WithToolPolicy(tooladapter.ToolCollectThenStop),
        tooladapter.WithToolMaxCalls(5),
    )
    
    // Test policy behavior
    // ...
}
```

## Production Deployment

### Recommended Configuration
```go
// Production configuration with comprehensive settings
adapter := tooladapter.New(
    tooladapter.WithLogger(prodLogger),
    tooladapter.WithMetricsCallback(metricsCallback),
    tooladapter.WithToolPolicy(tooladapter.ToolStopOnFirst), // Conservative policy
    tooladapter.WithStreamingToolBufferSize(10*1024*1024),   // 10MB streaming buffer
    tooladapter.WithPromptBufferReuseLimit(64*1024),         // 64KB buffer reuse limit
)

// High-performance streaming configuration
streamingAdapter := tooladapter.New(
    tooladapter.WithStreamingEarlyDetection(120),           // 120 char lookahead
    tooladapter.WithToolPolicy(tooladapter.ToolDrainAll),   // Collect all tools
    tooladapter.WithToolCollectWindow(10*time.Second),      // 10s collection window
)
```

### Build Commands
```bash
# Production build with optimization
go build -ldflags="-w -s" -o app

# Docker build
docker build .
```

## Troubleshooting

### Common Issues

#### High Memory Usage
- **Check**: `WithStreamingToolBufferSize()` and `WithPromptBufferReuseLimit()` settings
- **Solution**: Reduce buffer sizes for memory-constrained environments

#### Tool Detection Issues
- **Check**: Tool policy configuration and early detection settings
- **Solution**: Use `ToolDrainAll` policy or increase `WithStreamingEarlyDetection()` lookahead

#### Test Failures
- **Random ID Comparison**: Avoid comparing `ToolCall.ID` fields directly in tests
- **Race Conditions**: Ensure proper choice copying in concurrent scenarios
- **Context Cancellation**: Use appropriate timeouts for context-based operations

#### Performance Issues
- **Multi-Choice Overhead**: Verify lazy allocation is working (check logs for "No tool calls found")
- **JSON Parsing**: Profile `ExtractJSONBlocks()` for large responses
- **Buffer Pool**: Monitor buffer reuse rates in metrics

This development guide provides comprehensive coverage of the codebase structure, testing approach, and key architectural decisions that inform effective development and maintenance of the OpenAI Tool Adapter package.