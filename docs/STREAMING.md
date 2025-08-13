# 🌊 Streaming Guide

This guide covers the OpenAI Tool Adapter's streaming capabilities, which provide real-time tool call detection and processing for streaming chat completions. The streaming adapter uses the same robust finite state machine parser to detect tool calls as they arrive in the stream.

## Overview

The streaming adapter wraps OpenAI's native streaming responses and provides:

- **Real-time tool call detection** - Identifies function calls as they arrive in chunks
- **Policy-driven processing** - Four distinct tool processing policies for different use cases
- **Intelligent buffering** - Buffers potential JSON until complete structures are detected
- **Content suppression** - Configurable content handling based on tool policy
- **Seamless fallback** - Handles partial JSON gracefully with automatic fallback to content
- **Memory efficiency** - Uses configurable buffer limits to prevent memory exhaustion
- **Performance optimization** - Minimal overhead with sub-microsecond processing times

## Quick Start

```go
// Create streaming request (same as non-streaming)
transformedRequest, err := adapter.TransformCompletionsRequest(request)
if err != nil {
    return err
}

// Create streaming response
stream := client.Chat.Completions.NewStreaming(ctx, transformedRequest)

// Wrap with tool adapter
adaptedStream := adapter.TransformStreamingResponse(stream)
defer adaptedStream.Close()

// Process stream with real-time tool call detection
for adaptedStream.Next() {
    chunk := adaptedStream.Current()
    
    if len(chunk.Choices) > 0 {
        choice := chunk.Choices[0]
        
        // Handle tool calls (detected in real-time)
        if len(choice.Delta.ToolCalls) > 0 {
            for _, toolCall := range choice.Delta.ToolCalls {
                fmt.Printf("Tool call: %s\n", toolCall.Function.Name)
                fmt.Printf("Arguments: %s\n", toolCall.Function.Arguments)
            }
        }
        
        // Handle regular content
        if choice.Delta.Content != "" {
            fmt.Print(choice.Delta.Content)
        }
        
        // Handle completion
        if choice.FinishReason == "tool_calls" {
            fmt.Println("\nTool calls detected")
        }
    }
}

// Check for errors
if err := adaptedStream.Err(); err != nil {
    log.Printf("Stream error: %v", err)
}
```

## Tool Processing Policies

The streaming adapter supports four distinct tool processing policies that control how tool calls and content are handled:

### ToolStopOnFirst (Default)

**Best for:** Low-latency applications, simple workflows, production APIs

Stops processing after the first tool call is detected, providing the lowest latency.

```go
adapter := tooladapter.New(
    tooladapter.WithToolPolicy(tooladapter.ToolStopOnFirst),
    tooladapter.WithCancelUpstreamOnStop(true), // Stop LLM generation
)

// Behavior:
// 1. Streams content normally until first tool detected
// 2. Emits first tool call only
// 3. Suppresses all subsequent content
// 4. Optionally cancels upstream to save resources
```

### ToolCollectThenStop

**Best for:** Batched tool processing, structured workflows, time-bounded collection

Collects multiple tools within a configurable time window or until limits are reached.

```go
adapter := tooladapter.New(
    tooladapter.WithToolPolicy(tooladapter.ToolCollectThenStop),
    tooladapter.WithToolCollectWindow(500 * time.Millisecond), // Collection timeout
    tooladapter.WithToolMaxCalls(5), // Maximum tools to collect
)

// Behavior:
// 1. Streams content until first tool detected
// 2. Suppresses content after first tool
// 3. Collects tools for up to 500ms or 5 tools
// 4. Emits batch of collected tools
// 5. Stops processing further content
```

### ToolDrainAll

**Best for:** Complete processing, batch workflows, maximum tool extraction

Processes the entire stream and collects all detected tools while suppressing content.

```go
adapter := tooladapter.New(
    tooladapter.WithToolPolicy(tooladapter.ToolDrainAll),
    tooladapter.WithToolCollectMaxBytes(1024*1024), // 1MB safety limit
)

// Behavior:
// 1. Suppresses ALL content from the start
// 2. Buffers entire stream content
// 3. Detects and collects all tools
// 4. Emits all tools at stream completion
```

### ToolAllowMixed

**Best for:** Conversational interfaces, mixed content/tool responses, chat applications

Allows both content and tools to be emitted together without suppression.

```go
adapter := tooladapter.New(
    tooladapter.WithToolPolicy(tooladapter.ToolAllowMixed),
    tooladapter.WithToolMaxCalls(8), // Reasonable limit for chat
)

// Behavior:
// 1. Streams all content normally
// 2. Detects and emits tools as found
// 3. Preserves conversational flow
// 4. No content suppression
```

## Core Concepts

### Buffered Processing

The streaming adapter uses intelligent buffering to handle partial JSON:

```go
// Streaming content arrives in chunks:
// Chunk 1: "I'll help you with that. ["
// Chunk 2: "{\"name\": \"get_weather\", "
// Chunk 3: "\"parameters\": {\"location\": \"San Francisco\"}}"
// Chunk 4: "]"

// The adapter:
// 1. Detects potential JSON start in Chunk 1
// 2. Buffers content until complete JSON in Chunk 4
// 3. Parses complete JSON and emits tool call
// 4. Returns to normal streaming for subsequent content
```

### State Management

The adapter maintains internal state to track:

- **Buffer state** - Current buffering mode (none, potential, active)
- **JSON parsing state** - Depth tracking for nested structures
- **Completion tracking** - Whether the stream is complete
- **Context state** - Request context for cancellation handling

### Memory Protection

Built-in safeguards prevent memory exhaustion:

```go
// Default buffer limit: 10MB
const DefaultBufferLimitMB = 10

// Automatic fallback when limit exceeded
if bufferSize > limit {
    // Emit buffered content as regular content
    // Continue with normal streaming
}
```

## Policy Comparison

| Policy | Content Handling | Tool Processing | Latency | Use Case |
|--------|------------------|-----------------|---------|----------|
| **ToolStopOnFirst** | Cleared after first tool | First tool only | Lowest | APIs, simple workflows |
| **ToolCollectThenStop** | Cleared after collection | Multiple (time/count limited) | Low | Structured batching |
| **ToolDrainAll** | Always suppressed | All tools found | Higher | Complete extraction |
| **ToolAllowMixed** | Always preserved | All tools found | Variable | Chat, conversational |

### Choosing the Right Policy

**For production APIs:**
```go
// Prioritize speed and resource efficiency
adapter := tooladapter.New(
    tooladapter.WithToolPolicy(tooladapter.ToolStopOnFirst),
    tooladapter.WithToolMaxCalls(3),
    tooladapter.WithCancelUpstreamOnStop(true),
)
```

**For batch processing:**
```go
// Process everything, collect all tools
adapter := tooladapter.New(
    tooladapter.WithToolPolicy(tooladapter.ToolDrainAll),
    tooladapter.WithToolMaxCalls(0), // No limit
    tooladapter.WithToolCollectMaxBytes(10*1024*1024), // 10MB safety
)
```

**For conversational interfaces:**
```go
// Preserve natural flow
adapter := tooladapter.New(
    tooladapter.WithToolPolicy(tooladapter.ToolAllowMixed),
    tooladapter.WithToolMaxCalls(8),
)
```

## Safety Configuration

### Resource Limits

Protect against resource exhaustion with configurable limits:

```go
adapter := tooladapter.New(
    tooladapter.WithToolMaxCalls(10),              // Max 10 tools per response
    tooladapter.WithToolCollectMaxBytes(1024*1024), // 1MB buffer limit
    tooladapter.WithToolCollectWindow(2*time.Second), // 2s collection timeout
)
```

### Upstream Cancellation

Control resource usage by cancelling upstream processing:

```go
// Cancel LLM generation after tool detection (saves tokens/compute)
adapter := tooladapter.New(
    tooladapter.WithToolPolicy(tooladapter.ToolStopOnFirst),
    tooladapter.WithCancelUpstreamOnStop(true), // Default: true
)

// Continue LLM generation even after tools found
adapter := tooladapter.New(
    tooladapter.WithCancelUpstreamOnStop(false),
)
```

## Advanced Usage

### Context Support

```go
// Create context with timeout
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()

// Context is automatically propagated
adaptedStream := adapter.TransformStreamingResponseWithContext(ctx, stream)

// Stream respects context cancellation
for adaptedStream.Next() {
    select {
    case <-ctx.Done():
        return ctx.Err()
    default:
        chunk := adaptedStream.Current()
        // Process chunk
    }
}
```

### Error Handling

```go
adaptedStream := adapter.TransformStreamingResponse(stream)
defer adaptedStream.Close()

for adaptedStream.Next() {
    chunk := adaptedStream.Current()
    
    // Process chunk
    if err := processChunk(chunk); err != nil {
        log.Printf("Chunk processing error: %v", err)
        // Continue processing or break based on error severity
    }
}

// Always check for stream errors
if err := adaptedStream.Err(); err != nil {
    // Handle stream-level errors
    if isRetryable(err) {
        // Implement retry logic
    } else {
        return err
    }
}
```

### Multiple Tool Calls

The adapter handles multiple tool calls in a single response:

```go
for adaptedStream.Next() {
    chunk := adaptedStream.Current()
    
    if len(chunk.Choices) > 0 {
        choice := chunk.Choices[0]
        
        // Multiple tool calls can arrive together
        for i, toolCall := range choice.Delta.ToolCalls {
            fmt.Printf("Tool call %d: %s\n", i+1, toolCall.Function.Name)
            
            // Handle each tool call
            result, err := executeToolCall(toolCall)
            if err != nil {
                log.Printf("Tool execution error: %v", err)
                continue
            }
            
            fmt.Printf("Result: %v\n", result)
        }
    }
}
```

## Performance Optimization

### Buffer Configuration

```go
// For high-throughput applications
adapter := tooladapter.New(
    tooladapter.WithStreamingToolBufferSize(50 * 1024 * 1024), // 50MB buffer
)

// For memory-constrained environments
adapter := tooladapter.New(
    tooladapter.WithStreamingToolBufferSize(1 * 1024 * 1024), // 1MB buffer
)
```

### Metrics Integration

Monitor streaming performance:

```go
adapter := tooladapter.New(
    tooladapter.WithMetricsCallback(func(data tooladapter.MetricEventData) {
        switch eventData := data.(type) {
        case tooladapter.FunctionCallDetectionData:
            if eventData.Streaming {
                // Track streaming-specific metrics
                metrics.StreamingDetectionTime.Observe(
                    eventData.Performance.ProcessingDuration.Seconds())
                metrics.StreamingBufferUtilization.Set(
                    float64(eventData.ContentLength))
            }
        }
    }),
)
```

## Real-World Examples

### Chatbot with Tool Integration

```go
func handleChatStream(w http.ResponseWriter, r *http.Request) {
    // Set up SSE headers
    w.Header().Set("Content-Type", "text/event-stream")
    w.Header().Set("Cache-Control", "no-cache")
    w.Header().Set("Connection", "keep-alive")
    
    // Create streaming response
    stream := client.Chat.Completions.NewStreaming(ctx, transformedRequest)
    adaptedStream := adapter.TransformStreamingResponse(stream)
    defer adaptedStream.Close()
    
    for adaptedStream.Next() {
        chunk := adaptedStream.Current()
        
        if len(chunk.Choices) > 0 {
            choice := chunk.Choices[0]
            
            // Send tool calls to client
            if len(choice.Delta.ToolCalls) > 0 {
                for _, toolCall := range choice.Delta.ToolCalls {
                    event := map[string]interface{}{
                        "type": "tool_call",
                        "data": toolCall,
                    }
                    sendSSEEvent(w, event)
                    
                    // Execute tool call
                    result := executeToolCall(toolCall)
                    resultEvent := map[string]interface{}{
                        "type": "tool_result",
                        "data": result,
                    }
                    sendSSEEvent(w, resultEvent)
                }
            }
            
            // Send content to client
            if choice.Delta.Content != "" {
                event := map[string]interface{}{
                    "type": "content",
                    "data": choice.Delta.Content,
                }
                sendSSEEvent(w, event)
            }
        }
        
        // Flush immediately for real-time streaming
        if f, ok := w.(http.Flusher); ok {
            f.Flush()
        }
    }
    
    if err := adaptedStream.Err(); err != nil {
        log.Printf("Stream error: %v", err)
        errorEvent := map[string]interface{}{
            "type": "error",
            "data": err.Error(),
        }
        sendSSEEvent(w, errorEvent)
    }
}
```

### Batch Processing with Streaming

```go
func processBatchStreaming(requests []openai.ChatCompletionNewParams) error {
    results := make(chan ProcessResult, len(requests))
    
    var wg sync.WaitGroup
    
    for i, req := range requests {
        wg.Add(1)
        go func(index int, request openai.ChatCompletionNewParams) {
            defer wg.Done()
            
            transformedRequest, err := adapter.TransformCompletionsRequest(request)
            if err != nil {
                results <- ProcessResult{Index: index, Error: err}
                return
            }
            
            stream := client.Chat.Completions.NewStreaming(ctx, transformedRequest)
            adaptedStream := adapter.TransformStreamingResponse(stream)
            
            var toolCalls []openai.ChatCompletionMessageToolCall
            var content strings.Builder
            
            for adaptedStream.Next() {
                chunk := adaptedStream.Current()
                
                if len(chunk.Choices) > 0 {
                    choice := chunk.Choices[0]
                    
                    // Collect tool calls
                    toolCalls = append(toolCalls, choice.Delta.ToolCalls...)
                    
                    // Collect content
                    if choice.Delta.Content != "" {
                        content.WriteString(choice.Delta.Content)
                    }
                }
            }
            
            if err := adaptedStream.Err(); err != nil {
                results <- ProcessResult{Index: index, Error: err}
                return
            }
            
            results <- ProcessResult{
                Index:     index,
                ToolCalls: toolCalls,
                Content:   content.String(),
            }
            
            adaptedStream.Close()
        }(i, req)
    }
    
    // Wait for all processing to complete
    go func() {
        wg.Wait()
        close(results)
    }()
    
    // Process results as they arrive
    for result := range results {
        if result.Error != nil {
            log.Printf("Request %d failed: %v", result.Index, result.Error)
            continue
        }
        
        log.Printf("Request %d completed: %d tool calls, %d chars content",
            result.Index, len(result.ToolCalls), len(result.Content))
    }
    
    return nil
}
```

### WebSocket Integration

```go
func handleWebSocketStreaming(conn *websocket.Conn) {
    defer conn.Close()
    
    for {
        // Read request from WebSocket
        var request ChatRequest
        if err := conn.ReadJSON(&request); err != nil {
            break
        }
        
        // Transform and create stream
        transformedRequest, err := adapter.TransformCompletionsRequest(request.OpenAIRequest)
        if err != nil {
            conn.WriteJSON(ErrorResponse{Error: err.Error()})
            continue
        }
        
        stream := client.Chat.Completions.NewStreaming(ctx, transformedRequest)
        adaptedStream := adapter.TransformStreamingResponse(stream)
        
        // Stream response back to client
        for adaptedStream.Next() {
            chunk := adaptedStream.Current()
            
            // Send chunk to WebSocket client
            response := StreamResponse{
                Type: "chunk",
                Data: chunk,
            }
            
            if err := conn.WriteJSON(response); err != nil {
                break
            }
        }
        
        if err := adaptedStream.Err(); err != nil {
            conn.WriteJSON(ErrorResponse{Error: err.Error()})
        }
        
        adaptedStream.Close()
        
        // Send completion signal
        conn.WriteJSON(StreamResponse{Type: "complete"})
    }
}
```

## Best Practices

### Resource Management

```go
// Always close streams
adaptedStream := adapter.TransformStreamingResponse(stream)
defer func() {
    if err := adaptedStream.Close(); err != nil {
        log.Printf("Stream close error: %v", err)
    }
}()

// Handle context cancellation
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
defer cancel()

adaptedStream := adapter.TransformStreamingResponseWithContext(ctx, stream)
```

### Error Recovery

```go
func streamWithRetry(request openai.ChatCompletionNewParams, maxRetries int) error {
    for attempt := 0; attempt < maxRetries; attempt++ {
        stream := client.Chat.Completions.NewStreaming(ctx, request)
        adaptedStream := adapter.TransformStreamingResponse(stream)
        
        err := processStream(adaptedStream)
        adaptedStream.Close()
        
        if err == nil {
            return nil // Success
        }
        
        if !isRetryable(err) {
            return err // Non-retryable error
        }
        
        // Exponential backoff
        time.Sleep(time.Duration(attempt) * time.Second)
    }
    
    return fmt.Errorf("streaming failed after %d attempts", maxRetries)
}
```

### Performance Monitoring

```go
// Monitor key streaming metrics
func monitorStreamingPerformance() {
    adapter := tooladapter.New(
        tooladapter.WithMetricsCallback(func(data tooladapter.MetricEventData) {
            switch eventData := data.(type) {
            case tooladapter.FunctionCallDetectionData:
                if eventData.Streaming {
                    // Alert on slow detection
                    if eventData.Performance.ProcessingDuration > 100*time.Millisecond {
                        alert.SlowStreamingDetection(eventData)
                    }
                    
                    // Track buffer utilization
                    bufferUtilization := float64(eventData.ContentLength) / float64(maxBufferSize)
                    metrics.StreamingBufferUtilization.Set(bufferUtilization)
                    
                    if bufferUtilization > 0.8 {
                        alert.HighBufferUtilization(bufferUtilization)
                    }
                }
            }
        }),
    )
}
```

## Troubleshooting

### Common Issues

**Tool Calls Not Detected in Stream:**
- Verify the LLM is producing JSON in the expected format
- Check buffer limits aren't causing premature fallback
- Enable debug logging to see JSON detection process

**Memory Usage Growing:**
- Check buffer limits configuration
- Monitor for streams that don't properly close
- Verify context cancellation is working

**Slow Stream Processing:**
- Profile the metrics callback for performance bottlenecks
- Check if tool call execution is blocking the stream
- Consider async tool execution for long-running operations

### Debug Configuration

```go
// Enable detailed streaming debugging
debugAdapter := tooladapter.New(
    tooladapter.WithLogLevel(slog.LevelDebug),
    tooladapter.WithMetricsCallback(func(data tooladapter.MetricEventData) {
        fmt.Printf("Streaming event: %+v\n", data)
    }),
)
```

### Stream Validation

```go
func validateStream(adaptedStream *tooladapter.StreamAdapter) error {
    chunkCount := 0
    toolCallCount := 0
    
    for adaptedStream.Next() {
        chunkCount++
        chunk := adaptedStream.Current()
        
        if len(chunk.Choices) > 0 {
            toolCallCount += len(chunk.Choices[0].Delta.ToolCalls)
        }
        
        // Validate chunk structure
        if err := validateChunk(chunk); err != nil {
            return fmt.Errorf("invalid chunk at position %d: %v", chunkCount, err)
        }
    }
    
    log.Printf("Stream validation: %d chunks, %d tool calls", chunkCount, toolCallCount)
    return adaptedStream.Err()
}
```

The streaming adapter provides powerful real-time tool call detection while maintaining the simplicity and reliability of the core adapter. Use it for interactive applications, real-time processing, and any scenario where immediate tool call detection is valuable.