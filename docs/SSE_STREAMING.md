# 🔌 SSE Streaming Guide

This guide covers the OpenAI Tool Adapter's raw SSE (Server-Sent Events) streaming capabilities for processing HTTP response bodies directly without requiring OpenAI SDK stream wrappers.

## Overview

The SSE streaming adapter is designed for:

- **Proxy/Gateway implementations** that forward raw HTTP responses
- **Custom LLM providers** that return OpenAI-compatible SSE responses
- **Direct HTTP integrations** without SDK dependencies
- **Load balancers** and middleware that need to inspect/transform streams

## Quick Start

```go
import (
    "net/http"
    tooladapter "github.com/juburr/openai-tool-adapter/v3"
)

func handleStreamingProxy(w http.ResponseWriter, upstreamResp *http.Response) error {
    // Create adapter with desired configuration
    adapter := tooladapter.New(
        tooladapter.WithToolPolicy(tooladapter.ToolDrainAll),
        tooladapter.WithToolMaxCalls(8),
    )

    // Create SSE reader from upstream response
    reader := tooladapter.NewHTTPSSEReader(upstreamResp)
    defer reader.Close()

    // Create SSE writer for downstream client
    writer := tooladapter.NewHTTPSSEWriter(w)

    // Process stream with tool detection
    sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)
    return sseAdapter.Process(context.Background())
}
```

## Core Interfaces

### SSEStreamReader

Reads SSE events from any source:

```go
type SSEStreamReader interface {
    // Next advances to the next SSE event, returning false when done.
    Next() bool

    // Data returns the data payload of the current event (without "data: " prefix).
    Data() string

    // Err returns any error encountered during reading.
    Err() error

    // Close releases resources associated with the reader.
    Close() error
}
```

### SSEStreamWriter

Writes SSE events to any destination:

```go
type SSEStreamWriter interface {
    // WriteChunk writes an SSE chunk to the response.
    WriteChunk(chunk *SSEChunk) error

    // WriteRaw writes raw SSE data (already formatted as "data: {...}\n\n").
    WriteRaw(data []byte) error

    // WriteDone writes the "[DONE]" marker.
    WriteDone() error

    // Flush ensures buffered data is sent to the client.
    Flush()
}
```

### Built-in Implementations

```go
// Create reader from HTTP response
reader := tooladapter.NewHTTPSSEReader(resp)
defer reader.Close()

// Create reader from io.ReadCloser
reader := tooladapter.NewSSEReaderFromReadCloser(rc)
defer reader.Close()

// Create writer for HTTP response
writer := tooladapter.NewHTTPSSEWriter(w)
```

## Processing Modes

### Process() - Full Buffering

Buffers the entire stream, analyzes for tool calls, then writes the appropriate response:

```go
sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)
if err := sseAdapter.Process(ctx); err != nil {
    return err
}
```

**Behavior:**
1. Reads all SSE chunks from the stream
2. Accumulates content from delta messages
3. Extracts JSON tool calls using the finite state machine parser
4. Writes transformed response with tool_calls or passes through original

**Best for:** Complete tool detection, batch processing, small to medium responses

### ProcessWithPassthrough() - Early Detection

Optimizes for non-tool responses by detecting tool patterns early:

```go
// Early detection with 100-character lookahead
sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)
if err := sseAdapter.ProcessWithPassthrough(ctx, 100); err != nil {
    return err
}
```

**Behavior:**
1. Scans first N characters for tool patterns (e.g., `[{"name":`)
2. If no pattern detected, passes through chunks immediately
3. If pattern detected, falls back to full buffering

**Best for:** High-throughput scenarios where most responses don't contain tools

### ProcessToResult() - Inspection Before Writing

Returns the result for inspection before deciding how to handle it:

```go
sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)

result, chunks, err := sseAdapter.ProcessToResult(ctx)
if err != nil {
    return err
}

if result.HasToolCalls {
    // Custom handling for tool calls
    log.Printf("Detected %d tool calls", len(result.ToolCalls))
    return sseAdapter.WriteToolCallsFromResult(result)
} else {
    // Pass through original chunks
    return sseAdapter.WritePassthrough(chunks)
}
```

**Result structure:**
```go
type SSETransformResult struct {
    HasToolCalls bool              // Tool calls were detected
    ToolCalls    []RawFunctionCall // Extracted tool calls
    Content      string            // Original content
    Passthrough  bool              // Should pass through unchanged
}
```

## SSE Chunk Types

### SSEChunk

Provider-agnostic representation of a streaming chat completion chunk:

```go
type SSEChunk struct {
    ID      string      `json:"id,omitempty"`
    Object  string      `json:"object,omitempty"`
    Created int64       `json:"created,omitempty"`
    Model   string      `json:"model,omitempty"`
    Choices []SSEChoice `json:"choices,omitempty"`
    Usage   *SSEUsage   `json:"usage,omitempty"`
}

type SSEChoice struct {
    Index        int      `json:"index"`
    Delta        SSEDelta `json:"delta"`
    FinishReason string   `json:"finish_reason,omitempty"`
}

type SSEDelta struct {
    Role      string        `json:"role,omitempty"`
    Content   string        `json:"content,omitempty"`
    ToolCalls []SSEToolCall `json:"tool_calls,omitempty"`
}
```

## Tool Policy Support

All tool policies from the main adapter are supported:

### ToolStopOnFirst

```go
adapter := tooladapter.New(
    tooladapter.WithToolPolicy(tooladapter.ToolStopOnFirst),
)

// Returns only the first detected tool call
```

### ToolDrainAll

```go
adapter := tooladapter.New(
    tooladapter.WithToolPolicy(tooladapter.ToolDrainAll),
    tooladapter.WithToolMaxCalls(10),
)

// Collects all tool calls (up to max limit)
```

### ToolCollectThenStop

```go
adapter := tooladapter.New(
    tooladapter.WithToolPolicy(tooladapter.ToolCollectThenStop),
    tooladapter.WithToolCollectWindow(500 * time.Millisecond),
    tooladapter.WithToolMaxCalls(5),
)

// Collects within time/count limits
```

### ToolAllowMixed

```go
adapter := tooladapter.New(
    tooladapter.WithToolPolicy(tooladapter.ToolAllowMixed),
)

// Allows both content and tool calls
```

## Real-World Examples

### HTTP Proxy with Tool Detection

```go
func proxyHandler(w http.ResponseWriter, r *http.Request) {
    // Forward request to upstream LLM
    upstreamResp, err := forwardToLLM(r)
    if err != nil {
        http.Error(w, err.Error(), http.StatusBadGateway)
        return
    }
    defer upstreamResp.Body.Close()

    // Check if streaming
    if upstreamResp.Header.Get("Content-Type") != "text/event-stream" {
        // Non-streaming: use standard response transformation
        handleNonStreaming(w, upstreamResp)
        return
    }

    // Streaming: use SSE adapter
    adapter := tooladapter.New(
        tooladapter.WithLogLevel(slog.LevelInfo),
        tooladapter.WithToolPolicy(tooladapter.ToolDrainAll),
    )

    reader := tooladapter.NewHTTPSSEReader(upstreamResp)
    defer reader.Close()

    writer := tooladapter.NewHTTPSSEWriter(w)

    sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)
    if err := sseAdapter.ProcessWithPassthrough(r.Context(), 100); err != nil {
        log.Printf("SSE processing error: %v", err)
    }
}
```

### Load Balancer with Metrics

```go
func loadBalancerHandler(w http.ResponseWriter, r *http.Request) {
    startTime := time.Now()
    var toolsDetected int

    adapter := tooladapter.New(
        tooladapter.WithLogLevel(slog.LevelError),
        tooladapter.WithMetricsCallback(func(data tooladapter.MetricEventData) {
            switch d := data.(type) {
            case tooladapter.FunctionCallDetectionData:
                toolsDetected = d.FunctionCount
                metrics.RecordDetection(d.Performance.ProcessingDuration)
            }
        }),
    )

    // Select backend
    backend := selectBackend()
    resp, err := backend.Forward(r)
    if err != nil {
        http.Error(w, err.Error(), http.StatusBadGateway)
        return
    }
    defer resp.Body.Close()

    reader := tooladapter.NewHTTPSSEReader(resp)
    defer reader.Close()

    writer := tooladapter.NewHTTPSSEWriter(w)

    sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)
    err = sseAdapter.Process(r.Context())

    // Record metrics
    duration := time.Since(startTime)
    metrics.RecordRequest(duration, toolsDetected, err)
}
```

### Custom SSE Reader

Implement custom readers for non-standard sources:

```go
type WebSocketSSEReader struct {
    conn    *websocket.Conn
    current string
    err     error
    done    bool
}

func (r *WebSocketSSEReader) Next() bool {
    if r.done || r.err != nil {
        return false
    }

    _, message, err := r.conn.ReadMessage()
    if err != nil {
        r.err = err
        return false
    }

    // Parse SSE format
    if string(message) == "[DONE]" {
        r.done = true
        return false
    }

    r.current = string(message)
    return true
}

func (r *WebSocketSSEReader) Data() string {
    return r.current
}

func (r *WebSocketSSEReader) Err() error {
    return r.err
}

func (r *WebSocketSSEReader) Close() error {
    return r.conn.Close()
}
```

## Error Handling

### Context Cancellation

```go
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()

sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)
err := sseAdapter.Process(ctx)

if err == context.DeadlineExceeded {
    log.Println("Stream processing timed out")
} else if err == context.Canceled {
    log.Println("Stream processing was cancelled")
} else if err != nil {
    log.Printf("Stream error: %v", err)
}
```

### Reader Errors

```go
sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)
err := sseAdapter.Process(ctx)

if err != nil {
    // Check for specific error types
    if errors.Is(err, io.ErrUnexpectedEOF) {
        log.Println("Connection closed unexpectedly")
    } else if errors.Is(err, context.DeadlineExceeded) {
        log.Println("Request timed out")
    } else {
        log.Printf("Processing error: %v", err)
    }
}
```

## Performance

### Benchmark Results

On AMD Ryzen 9 9950X3D:

| Operation | Time | Allocations |
|-----------|------|-------------|
| Process (no tools) | ~6 μs/op | 82 allocs |
| Process (with tools) | ~5.3 μs/op | 69 allocs |
| ProcessWithPassthrough | ~3.2 μs/op | 42 allocs |

### Optimization Tips

1. **Use ProcessWithPassthrough** when most responses don't contain tools:
   ```go
   // 100-character lookahead is typically sufficient
   sseAdapter.ProcessWithPassthrough(ctx, 100)
   ```

2. **Set appropriate buffer limits** for memory-constrained environments:
   ```go
   adapter := tooladapter.New(
       tooladapter.WithStreamingToolBufferSize(5 * 1024 * 1024), // 5MB
   )
   ```

3. **Use ProcessToResult** for custom handling:
   ```go
   result, chunks, _ := sseAdapter.ProcessToResult(ctx)
   if result.Passthrough {
       // Skip tool processing entirely
   }
   ```

## Thread Safety

**Important:** `SSEStreamAdapter` instances are NOT thread-safe. Each instance should be used by a single goroutine only. Create a new adapter for each concurrent stream.

```go
// Correct: each goroutine gets its own adapter
func handleRequest(w http.ResponseWriter, r *http.Request) {
    adapter := tooladapter.New()
    sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)
    sseAdapter.Process(r.Context())
}

// Incorrect: shared adapter across goroutines
var sharedAdapter = tooladapter.New()
func handleRequest(w http.ResponseWriter, r *http.Request) {
    // UNSAFE: race condition!
    sseAdapter := sharedAdapter.NewSSEStreamAdapter(reader, writer)
}
```

## Comparison with SDK Streaming

| Feature | SDK Streaming | SSE Streaming |
|---------|---------------|---------------|
| Input Type | `openai.ChatCompletionStream` | Raw HTTP response |
| Dependencies | OpenAI SDK | None (stdlib only) |
| Use Case | Direct OpenAI API usage | Proxy/gateway implementations |
| Memory Model | SDK-managed | User-controlled |
| Type Safety | Full SDK types | SSE-specific types |

Use SSE streaming when:
- Building proxies or gateways
- Working with raw HTTP responses
- Need provider-agnostic streaming
- Avoiding SDK dependencies

Use SDK streaming when:
- Direct OpenAI API usage
- Need full type safety
- Using other SDK features

## Troubleshooting

### Tool Calls Not Detected

1. **Check content format**: The adapter expects JSON function calls in the content
2. **Enable debug logging**:
   ```go
   adapter := tooladapter.New(tooladapter.WithLogLevel(slog.LevelDebug))
   ```
3. **Verify SSE format**: Ensure the upstream sends proper `data: {...}\n\n` format

### Memory Usage Growing

1. **Set buffer limits**:
   ```go
   tooladapter.WithStreamingToolBufferSize(10 * 1024 * 1024)
   ```
2. **Use early detection**:
   ```go
   sseAdapter.ProcessWithPassthrough(ctx, 50)
   ```
3. **Check for unclosed readers**: Always close readers in defer blocks

### Slow Processing

1. **Profile the metrics callback**: Ensure it doesn't block
2. **Check network latency**: SSE processing is usually not the bottleneck
3. **Use ProcessWithPassthrough**: Reduces processing for non-tool responses
