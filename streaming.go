package tooladapter

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/openai/openai-go/v2"
)

// ChatCompletionStreamInterface represents the streaming interface returned by OpenAI SDK
// This matches the interface returned by client.Chat.Completions.NewStreaming()
type ChatCompletionStreamInterface interface {
	Next() bool
	Current() openai.ChatCompletionChunk
	Err() error
	Close() error
}

// toolCollectionState tracks the current state of tool collection in streaming mode
type toolCollectionState int

const (
	toolStateIdle       toolCollectionState = iota // Not collecting tools
	toolStateCollecting                            // Actively collecting tools
	toolStateFinished                              // Tool collection finished
)

// StreamAdapter wraps an OpenAI streaming response to intercept and transform tool calls.
// It implements the same interface as the original stream while processing tool calls.
//
// THREAD SAFETY: StreamAdapter instances are NOT thread-safe and designed for single-consumer use.
// Each StreamAdapter should be used by only one goroutine, following the same pattern as the
// underlying OpenAI streaming response.
//
// Concurrency design:
//   - Internal mutex (mu) protects all mutable fields during method calls
//   - Methods are safe to call sequentially from a single goroutine
//   - Multiple goroutines should NOT call methods on the same StreamAdapter instance
//   - Context cancellation is thread-safe and can be triggered from any goroutine
//   - Close() can be safely called concurrently with other operations
//
// Usage pattern:
//
//	go func() {
//	    stream := adapter.TransformStreamingResponseWithContext(ctx, sourceStream)
//	    defer stream.Close()
//	    for stream.Next() {  // Single goroutine only
//	        chunk := stream.Current()
//	        // process chunk
//	    }
//	}()
type StreamAdapter struct {
	source          ChatCompletionStreamInterface
	adapter         *Adapter
	buffer          strings.Builder
	hasEmitted      bool
	currentChunk    openai.ChatCompletionChunk
	done            bool
	err             error
	mu              sync.Mutex
	bufferLimit     int                         // Prevent unlimited buffer growth
	pendingFinish   *openai.ChatCompletionChunk // Store finish chunk to emit after content
	processedChunks int                         // Track chunks processed for logging
	ctx             context.Context             // Context for cancellation support
	cancel          context.CancelFunc          // Cancel function for cleanup

	// Tool policy state tracking
	toolCallsEmitted    bool                // Track if we've emitted tool calls
	toolCollectionState toolCollectionState // Current collection state
	collectedTools      []functionCall      // Tools collected so far
	contentSuppressed   bool                // Whether content emission is suppressed
	collectionStartTime time.Time           // When tool collection started (for timeouts)
	bytesCollected      int                 // Bytes collected for safety limits
	stopProcessing      bool                // Flag to stop processing further chunks after tool emission

	// Collect-then-stop specific tracking - removed complex array detection

	// Upstream control
	upstreamClosed bool // true if we explicitly closed the upstream to stop generation
}

// TransformStreamingResponse creates a stream adapter that processes tool calls.
// This is the backward-compatible version that uses context.Background().
// For production use with timeouts and cancellation, use TransformStreamingResponseWithContext.
func (a *Adapter) TransformStreamingResponse(stream ChatCompletionStreamInterface) *StreamAdapter {
	return a.TransformStreamingResponseWithContext(context.Background(), stream)
}

// TransformStreamingResponseWithContext creates a stream adapter that processes tool calls
// with context support for cancellation and timeouts.
func (a *Adapter) TransformStreamingResponseWithContext(ctx context.Context, stream ChatCompletionStreamInterface) *StreamAdapter {
	// Create a cancellable context for this stream
	streamCtx, cancel := context.WithCancel(ctx)

	adapter := &StreamAdapter{
		source:      stream,
		adapter:     a,
		bufferLimit: a.streamBufferLimit, // Configurable buffer limit to prevent memory issues
		ctx:         streamCtx,
		cancel:      cancel,
	}

	a.logger.Debug("Created streaming adapter with context support", "buffer_limit_mb", adapter.bufferLimit/(1024*1024))
	return adapter
}

// Next advances the stream to the next chunk.
// It buffers content chunks until complete tool calls are detected.
// checkCancellation checks if the context is cancelled and sets appropriate state
func (s *StreamAdapter) checkCancellation() bool {
	select {
	case <-s.ctx.Done():
		s.err = s.ctx.Err()
		s.done = true
		return true
	default:
		return false
	}
}

// handlePendingFinish processes pending finish chunk if it exists
func (s *StreamAdapter) handlePendingFinish() bool {
	if s.pendingFinish != nil {
		s.currentChunk = *s.pendingFinish
		s.pendingFinish = nil
		s.done = true
		s.adapter.logger.Debug("Emitted pending finish chunk", "total_processed_chunks", s.processedChunks)
		return true
	}
	return false
}

// handleStreamEnd processes the end of the source stream
func (s *StreamAdapter) handleStreamEnd() bool {
	if s.buffer.Len() > 0 {
		s.adapter.logger.Debug("Stream ended with buffered content",
			"buffer_length", s.buffer.Len(),
			"total_processed_chunks", s.processedChunks)
		if s.adapter.toolPolicy == ToolCollectThenStop {
			s.processBufferedContentForCollectionPhase()
		} else {
			s.processBufferedContent()
		}
		s.done = true
		return true
	}

	// Check if we have collected tools that haven't been emitted yet
	if len(s.collectedTools) > 0 {
		s.adapter.logger.Debug("Stream ended with collected tools, processing them",
			"collected_tool_count", len(s.collectedTools))
		s.processCollectedTools()
		s.done = true
		return true
	}

	s.done = true
	s.err = s.source.Err()
	s.adapter.logger.Debug("Stream ended",
		"total_processed_chunks", s.processedChunks,
		"error", s.err)
	return false
}

// handleContentChunk processes content chunks according to the configured tool policy
func (s *StreamAdapter) handleContentChunk(chunk openai.ChatCompletionChunk) bool {
	// Defensive bounds check: this should not happen since isContentChunk validates,
	// but we add it as an additional safety measure
	if len(chunk.Choices) == 0 {
		s.adapter.logger.Error("handleContentChunk called with no choices")
		return false
	}

	content := chunk.Choices[0].Delta.Content

	// Handle different tool policies
	switch s.adapter.toolPolicy {
	case ToolAllowMixed:
		// Always emit content, tools are handled separately
		return s.handleMixedMode(chunk, content)

	case ToolStopOnFirst:
		return s.handleStopOnFirstMode(chunk, content)

	case ToolCollectThenStop:
		return s.handleCollectThenStopMode(chunk, content)

	case ToolDrainAll:
		return s.handleDrainAllMode(chunk, content)

	default:
		// Fallback to ToolStopOnFirst for unknown policies
		s.adapter.logger.Warn("Unknown tool policy, falling back to ToolStopOnFirst",
			"policy", s.adapter.toolPolicy)
		return s.handleStopOnFirstMode(chunk, content)
	}
}

// handleBufferedContent processes content when already buffering
func (s *StreamAdapter) handleBufferedContent(content string) bool {
	s.buffer.WriteString(content)

	// Check if we have a complete JSON structure
	if s.hasCompleteJSON() {
		s.adapter.logger.Debug("Complete JSON detected in buffer",
			"buffer_length", s.buffer.Len(),
			"chunk_index", s.processedChunks)
		s.processBufferedContent()
		return true
	}

	// Safety check: prevent unlimited buffering
	if s.buffer.Len() > s.bufferLimit {
		s.adapter.logger.Warn("Buffer limit exceeded, processing as regular content",
			"buffer_length", s.buffer.Len(),
			"limit", s.bufferLimit)
		s.processBufferedContentAsRegular()
		return true
	}

	return false // Continue buffering
}

// handleFinishChunk processes finish chunks with buffer handling
func (s *StreamAdapter) handleFinishChunk(chunk openai.ChatCompletionChunk) bool {
	// Process any remaining buffer before the finish chunk
	if s.buffer.Len() > 0 {
		s.adapter.logger.Debug("Processing remaining buffer before finish chunk",
			"buffer_length", s.buffer.Len())
		s.processBufferedContent()
		// Store the finish chunk to emit after the content
		s.pendingFinish = &chunk
		return true
	}
	// No buffer - pass through finish chunk directly
	s.currentChunk = chunk
	s.done = true
	return true
}

func (s *StreamAdapter) Next() bool {
	// Fast state checks under lock
	s.mu.Lock()
	if s.done {
		s.mu.Unlock()
		return false
	}
	if s.checkCancellation() {
		s.mu.Unlock()
		return false
	}
	if s.handlePendingFinish() {
		s.mu.Unlock()
		return true
	}
	stopProcessing := s.stopProcessing
	s.mu.Unlock()

	// If we've set the stop processing flag, drain the stream until a finish chunk arrives
	if stopProcessing {
		for s.source.Next() {
			chunk := s.source.Current()
			if s.isFinishChunk(chunk) {
				s.mu.Lock()
				s.currentChunk = chunk
				s.done = true
				s.mu.Unlock()
				return true
			}
		}
		s.mu.Lock()
		s.done = true
		s.err = s.source.Err()
		s.mu.Unlock()
		return false
	}

	// Main streaming loop
	for {
		// If context was cancelled, terminate cleanly
		if s.ctx.Err() != nil {
			s.mu.Lock()
			s.err = s.ctx.Err()
			s.done = true
			s.mu.Unlock()
			return false
		}

		// Block for next chunk WITHOUT holding the mutex to avoid deadlocks with Close()
		hasNext := s.source.Next()

		// Check for cancellation after unblocking
		if s.ctx.Err() != nil {
			s.mu.Lock()
			s.err = s.ctx.Err()
			s.done = true
			s.mu.Unlock()
			return false
		}

		// Handle stream end
		if !hasNext {
			s.mu.Lock()
			result := s.handleStreamEnd()
			s.mu.Unlock()
			return result
		}

		chunk := s.source.Current()

		// Process the chunk under lock
		s.mu.Lock()
		s.processedChunks++

		if s.isContentChunk(chunk) {
			if result := s.handleContentChunk(chunk); result {
				s.mu.Unlock()
				return true
			}
			// Continue to next iteration if handleContentChunk returned false
			s.mu.Unlock()
			continue
		}

		if s.isFinishChunk(chunk) {
			result := s.handleFinishChunk(chunk)
			s.mu.Unlock()
			return result
		}

		// Pass through non-content chunks (like role assignments, etc.)
		s.currentChunk = chunk
		s.mu.Unlock()
		return true
	}
}

// shouldStartBuffering decides if we should start buffering based on content
// This uses a fast heuristic to minimize unnecessary buffering while catching
// tool calls that may appear after explanatory text (when early detection is enabled)
func (s *StreamAdapter) shouldStartBuffering(content string) bool {
	trimmed := strings.TrimSpace(content)
	if trimmed == "" {
		return false
	}

	// Check for immediate tool call patterns
	if s.hasImmediateToolCallPattern(trimmed) {
		return true
	}

	// Check for markdown code blocks with tool calls
	if s.hasMarkdownToolCallPattern(trimmed) {
		return true
	}

	// Check for backtick-enclosed tool calls
	if s.hasBacktickToolCallPattern(trimmed) {
		return true
	}

	// Check for tool calls within early detection lookahead range
	if s.hasEarlyDetectionToolCall(trimmed) {
		return true
	}

	// Conservative default: don't buffer unless we're quite sure
	return false
}

// hasImmediateToolCallPattern checks for direct function call patterns at the start
func (s *StreamAdapter) hasImmediateToolCallPattern(trimmed string) bool {
	return strings.HasPrefix(trimmed, `[{"name":`) ||
		strings.HasPrefix(trimmed, `[{"name": `) ||
		strings.HasPrefix(trimmed, `{"name":`) ||
		strings.HasPrefix(trimmed, `{"name": `)
}

// hasMarkdownToolCallPattern checks for markdown code blocks with tool calls
func (s *StreamAdapter) hasMarkdownToolCallPattern(trimmed string) bool {
	if !strings.HasPrefix(trimmed, "```json") && !strings.HasPrefix(trimmed, "```") {
		return false
	}
	// Look for function call indicators in the first part
	return strings.Contains(trimmed, `"name"`) || strings.Contains(trimmed, `[{`)
}

// hasBacktickToolCallPattern checks for backtick-enclosed function calls
func (s *StreamAdapter) hasBacktickToolCallPattern(trimmed string) bool {
	return strings.Contains(trimmed, "`{\"name\"") || strings.Contains(trimmed, "`[{\"name\"")
}

// hasEarlyDetectionToolCall checks for tool calls within the early detection lookahead range
func (s *StreamAdapter) hasEarlyDetectionToolCall(trimmed string) bool {
	if s.adapter.streamLookAheadLimit <= 0 {
		return false
	}

	// Search within the configured lookahead limit for tool call patterns
	searchRange := len(trimmed)
	if s.adapter.streamLookAheadLimit < searchRange {
		searchRange = s.adapter.streamLookAheadLimit
	}

	searchText := trimmed[:searchRange]

	// Look for tool call JSON patterns within the search range
	return strings.Contains(searchText, `{"name":`) ||
		strings.Contains(searchText, `{"name": `) ||
		strings.Contains(searchText, `[{"name":`) ||
		strings.Contains(searchText, `[{"name": `)
}

// Current returns the current chunk in the stream.
func (s *StreamAdapter) Current() openai.ChatCompletionChunk {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.currentChunk
}

// Err returns any error from the stream.
func (s *StreamAdapter) Err() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.err != nil {
		return s.err
	}
	// Shield callers from context cancellation when we intentionally closed upstream
	if err := s.source.Err(); err != nil {
		if s.upstreamClosed && (errors.Is(err, context.Canceled)) {
			return nil
		}
		return err
	}
	return nil
}

// Close closes the underlying stream and cancels the context.
func (s *StreamAdapter) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Capture metrics while holding the lock to prevent races
	finalBufferLength := s.buffer.Len()
	totalProcessedChunks := s.processedChunks

	// Cancel the context to clean up any waiting operations
	if s.cancel != nil {
		s.cancel()
		s.cancel = nil // Prevent double cancellation
	}

	// Log while still holding the lock to ensure consistent state
	s.adapter.logger.Debug("Closing streaming adapter",
		"total_processed_chunks", totalProcessedChunks,
		"final_buffer_length", finalBufferLength)

	// Close the underlying source stream
	return s.source.Close()
}

// isContentChunk checks if a chunk contains message content.
func (s *StreamAdapter) isContentChunk(chunk openai.ChatCompletionChunk) bool {
	return len(chunk.Choices) > 0 &&
		chunk.Choices[0].Delta.Content != ""
}

// isFinishChunk checks if a chunk signals the end of generation.
func (s *StreamAdapter) isFinishChunk(chunk openai.ChatCompletionChunk) bool {
	return len(chunk.Choices) > 0 &&
		chunk.Choices[0].FinishReason != ""
}

// hasCompleteJSON checks if the buffer contains complete JSON using the state machine parser
func (s *StreamAdapter) hasCompleteJSON() bool {
	content := s.buffer.String()
	if content == "" {
		return false
	}

	// Use the state machine parser to check for complete JSON structures
	return HasCompleteJSON(content)
}

// processBufferedContent processes the buffered content to extract tool calls
func (s *StreamAdapter) processBufferedContent() {
	content := s.buffer.String()
	if content == "" {
		return
	}

	s.hasEmitted = true
	startTime := time.Now()

	// Use state machine parser to extract JSON blocks
	jsonStartTime := time.Now()
	extractor := NewJSONExtractor(content)
	candidates := extractor.ExtractJSONBlocks()
	jsonParsingTime := time.Since(jsonStartTime)

	// Extract function calls from candidates
	extractionStartTime := time.Now()
	calls, _ := ExtractFunctionCallsDetailed(candidates)
	extractionTime := time.Since(extractionStartTime)
	totalDuration := time.Since(startTime)

	// Emit tool calls if found, otherwise emit as content
	if len(calls) > 0 {
		// Enforce global max cap as a safety
		if s.adapter.toolMaxCalls > 0 && len(calls) > s.adapter.toolMaxCalls {
			calls = calls[:s.adapter.toolMaxCalls]
		}
		// Extract function names for logging and metrics
		functionNames := make([]string, len(calls))
		for i, call := range calls {
			functionNames[i] = call.Name
		}

		// Log the streaming detection and conversion
		logAttrs := []any{
			"function_count", len(calls),
			"function_names", functionNames,
			"buffer_length", len(content),
			"json_candidates", len(candidates),
			"streaming", true,
		}

		// In debug mode, also log the function arguments
		if s.adapter.logger.Enabled(s.ctx, slog.LevelDebug) {
			args := make([]string, len(calls))
			for i, call := range calls {
				if call.Parameters != nil {
					args[i] = string(call.Parameters)
				} else {
					args[i] = "null"
				}
			}
			logAttrs = append(logAttrs, "function_arguments", args)
		}

		s.adapter.logger.Info("Streaming: detected and converted function calls", logAttrs...)

		// Emit metrics event for streaming function call detection
		s.adapter.emitMetric(FunctionCallDetectionData{
			FunctionCount:  len(calls),
			FunctionNames:  functionNames,
			ContentLength:  len(content),
			JSONCandidates: len(candidates),
			Streaming:      true, // This is the streaming path
			Performance: PerformanceMetrics{
				ProcessingDuration: totalDuration,
				SubOperations: map[string]time.Duration{
					"json_parsing":    jsonParsingTime,
					"call_extraction": extractionTime,
				},
			},
		})

		s.emitToolCallChunk(calls)
	} else {
		s.adapter.logger.Debug("Buffered content did not contain valid function calls, emitting as regular content",
			"buffer_length", len(content),
			"candidate_count", len(candidates))
		s.emitContentChunk(content)
	}

	// Clear the buffer after processing
	s.buffer.Reset()
}

// processBufferedContentAsRegular emits buffered content as regular text (fallback)
func (s *StreamAdapter) processBufferedContentAsRegular() {
	content := s.buffer.String()
	if content != "" {
		s.hasEmitted = true
		s.adapter.logger.Debug("Processing buffered content as regular content (fallback)",
			"content_length", len(content))
		s.emitContentChunk(content)
		s.buffer.Reset()
	}
}

// emitContentChunk creates a content chunk.
func (s *StreamAdapter) emitContentChunk(content string) {
	s.currentChunk = openai.ChatCompletionChunk{
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

// emitToolCallChunk creates tool call chunks.
//
// IMPORTANT: StopOnFirst semantics
// When the policy is ToolStopOnFirst and a single parsed JSON emission contains
// multiple tool calls (e.g., an array of calls), we emit ALL of those calls in
// the same chunk. "StopOnFirst" means we stop streaming further content AFTER
// the first tool emission, not that we truncate the emitted set to a single call.
// This preserves correctness for models that batch multiple calls in one JSON array
// while still suppressing any subsequent content after the emission.
func (s *StreamAdapter) emitToolCallChunk(calls []functionCall) {
	// Validate input
	if len(calls) == 0 {
		s.adapter.logger.Warn("Attempted to emit tool call chunk with no calls")
		s.emitContentChunk("") // Emit empty content as fallback
		return
	}

	// Create tool calls with bounds checking
	toolCalls := make([]openai.ChatCompletionChunkChoiceDeltaToolCall, 0, len(calls))
	for i, call := range calls {
		// Skip invalid calls
		if call.Name == "" {
			s.adapter.logger.Warn("Skipping invalid function call with empty name", "call_index", i)
			continue
		}

		// Handle missing parameters
		parameters := call.Parameters
		if parameters == nil {
			parameters = json.RawMessage("null")
		}

		// Generate unique IDs for each tool call using our fast ID generator
		toolCall := openai.ChatCompletionChunkChoiceDeltaToolCall{
			Index: int64(i),
			ID:    s.adapter.GenerateToolCallID(),
			Type:  functionType,
			Function: openai.ChatCompletionChunkChoiceDeltaToolCallFunction{
				Name:      call.Name,
				Arguments: string(parameters),
			},
		}
		toolCalls = append(toolCalls, toolCall)
	}

	// Only emit if we have valid tool calls
	if len(toolCalls) > 0 {
		s.currentChunk = openai.ChatCompletionChunk{
			Choices: []openai.ChatCompletionChunkChoice{
				{
					Delta: openai.ChatCompletionChunkChoiceDelta{
						Role:      "assistant",
						ToolCalls: toolCalls,
					},
					FinishReason: "tool_calls",
				},
			},
		}

		// Mark that we've emitted tool calls - all subsequent content will be discarded
		s.toolCallsEmitted = true

		// Set flag to stop processing if configured and appropriate for the policy
		if s.adapter.cancelUpstreamOnStop &&
			(s.adapter.toolPolicy == ToolStopOnFirst ||
				(s.adapter.toolPolicy == ToolCollectThenStop && s.toolCollectionState == toolStateFinished)) {
			s.adapter.logger.Debug("Setting stop processing flag after emitting tool calls",
				"policy", s.adapter.toolPolicy.String(),
				"cancel_upstream_on_stop", s.adapter.cancelUpstreamOnStop)
			s.stopProcessing = true
			// Proactively stop upstream generation without surfacing context.Canceled
			if err := s.source.Close(); err == nil {
				s.upstreamClosed = true
			} else {
				// Even if close errors, continue shielding cancellation later
				s.upstreamClosed = true
			}
		}

		s.adapter.logger.Debug("Emitted streaming tool call chunk",
			"valid_tool_calls", len(toolCalls),
			"original_call_count", len(calls))
	} else {
		// Fallback to content chunk if no valid tool calls
		s.adapter.logger.Warn("No valid tool calls after processing, falling back to empty content")
		s.emitContentChunk("")
	}
}

// handleMixedMode handles ToolAllowMixed policy - streams both content and tools
func (s *StreamAdapter) handleMixedMode(chunk openai.ChatCompletionChunk, content string) bool {
	// In mixed mode, always emit content immediately and handle tools separately
	// This is the simplest policy - no content suppression
	if s.buffer.Len() > 0 {
		// Continue buffering for tool detection while emitting content
		return s.handleBufferedContent(content)
	}

	// Check if we should start buffering for tool detection
	if s.shouldStartBuffering(content) {
		s.buffer.WriteString(content)
		s.adapter.logger.Debug("Started buffering potential tool call (mixed mode)",
			"content_prefix", s.truncateForLog(content, 50),
			"chunk_index", s.processedChunks)
	}

	// Always emit content in mixed mode
	s.currentChunk = chunk
	return true
}

// handleStopOnFirstMode handles ToolStopOnFirst policy - stops on first tool call
func (s *StreamAdapter) handleStopOnFirstMode(chunk openai.ChatCompletionChunk, content string) bool {
	// If we've already emitted tool calls, discard all subsequent content
	if s.toolCallsEmitted {
		s.adapter.logger.Debug("Discarding content after tool calls emitted (stop on first)",
			"content_length", len(content),
			"content_prefix", s.truncateForLog(content, 50),
			"chunk_index", s.processedChunks)

		// The upstream cancellation happens in emitToolCallChunk when tool calls are emitted
		return false // Discard this chunk, continue to next
	}

	// If we're already buffering, continue buffering
	if s.buffer.Len() > 0 {
		return s.handleBufferedContent(content)
	}

	// Not buffering yet - decide if we should start
	if s.shouldStartBuffering(content) {
		s.buffer.WriteString(content)
		s.adapter.logger.Debug("Started buffering potential tool call (stop on first)",
			"content_prefix", s.truncateForLog(content, 50),
			"chunk_index", s.processedChunks)
		return false // Continue to next chunk
	}

	// Regular content - pass through immediately
	s.currentChunk = chunk
	return true
}

// handleCollectThenStopMode handles ToolCollectThenStop policy - collects tools until limits reached
func (s *StreamAdapter) handleCollectThenStopMode(chunk openai.ChatCompletionChunk, content string) bool {
	// If we're already buffering or collecting, handle that content
	if s.buffer.Len() > 0 || s.contentSuppressed {
		return s.handleBufferedContentForCollection(content)
	}

	// Not buffering yet - decide if we should start
	if s.shouldStartBuffering(content) {
		s.startToolCollection(content)
		return false // Continue to next chunk
	}

	// Regular content - pass through immediately (before any tool detection)
	s.currentChunk = chunk
	return true
}

// handleDrainAllMode handles ToolDrainAll policy - reads entire stream and collects all tools
func (s *StreamAdapter) handleDrainAllMode(_ openai.ChatCompletionChunk, content string) bool {
	// In drain all mode, never emit content until the very end
	s.contentSuppressed = true

	s.adapter.logger.Debug("Buffering content for drain all mode",
		"content_length", len(content),
		"buffer_length", s.buffer.Len(),
		"chunk_index", s.processedChunks)

	// Always buffer all content
	s.buffer.WriteString(content)
	s.bytesCollected += len(content)

	// Check byte limits
	if s.adapter.toolCollectMaxBytes > 0 && s.bytesCollected > s.adapter.toolCollectMaxBytes {
		s.adapter.logger.Warn("Byte limit exceeded in drain all mode, processing collected content",
			"bytes_collected", s.bytesCollected,
			"limit", s.adapter.toolCollectMaxBytes,
			"recommendation", "Consider increasing limit with WithToolCollectMaxBytes() if legitimate use case")
		s.processBufferedContent()
		return true
	}

	return false // Never emit content chunks in drain all mode
}

// handleBufferedContentForCollection handles buffered content during tool collection phases
func (s *StreamAdapter) handleBufferedContentForCollection(content string) bool {
	if !s.contentSuppressed {
		// Still emitting content - check if we should start collection
		if s.shouldStartBuffering(content) {
			s.startToolCollection(content)
			return false
		}
		// Continue with regular content emission
		s.buffer.WriteString(content)
		if s.hasCompleteJSON() {
			s.processBufferedContent()
			return true
		}
		if s.buffer.Len() > s.bufferLimit {
			s.processBufferedContentAsRegular()
			return true
		}
		return false
	}

	// Content suppressed - collecting tools
	s.buffer.WriteString(content)
	s.bytesCollected += len(content)

	// Check stopping conditions
	if s.shouldStopCollection() {
		s.processCollectedTools()
		return true
	}

	// Check for complete JSON structure
	if s.hasCompleteJSON() {
		s.adapter.logger.Debug("Complete JSON detected during collection",
			"buffer_length", s.buffer.Len(),
			"chunk_index", s.processedChunks)
		s.processBufferedContentForCollectionPhase()
		return true
	}

	// Safety check: prevent unlimited buffering
	if s.buffer.Len() > s.bufferLimit {
		s.adapter.logger.Warn("Buffer limit exceeded during collection, processing as regular content",
			"buffer_length", s.buffer.Len(),
			"limit", s.bufferLimit)
		s.processBufferedContentAsRegular()
		return true
	}

	return false // Continue buffering
}

// shouldStopCollection determines if tool collection should stop based on policy limits
func (s *StreamAdapter) shouldStopCollection() bool {
	// Check tool count limit
	if s.adapter.toolMaxCalls > 0 && len(s.collectedTools) >= s.adapter.toolMaxCalls {
		s.adapter.logger.Debug("Tool collection stopped: max calls reached",
			"collected_tools", len(s.collectedTools),
			"max_calls", s.adapter.toolMaxCalls)
		return true
	}

	// Check byte limit
	if s.adapter.toolCollectMaxBytes > 0 && s.bytesCollected > s.adapter.toolCollectMaxBytes {
		s.adapter.logger.Warn("Tool collection stopped: max bytes reached",
			"bytes_collected", s.bytesCollected,
			"max_bytes", s.adapter.toolCollectMaxBytes,
			"recommendation", "Consider increasing limit with WithToolCollectMaxBytes() if legitimate use case")
		return true
	}

	// Check timeout for CollectThenStop policy
	if s.adapter.toolPolicy == ToolCollectThenStop && s.adapter.toolCollectWindow > 0 {
		if time.Since(s.collectionStartTime) > s.adapter.toolCollectWindow {
			s.adapter.logger.Debug("Tool collection stopped: timeout reached",
				"elapsed", time.Since(s.collectionStartTime),
				"window", s.adapter.toolCollectWindow)
			return true
		}
	}

	return false
}

// startToolCollection initializes tool collection state
func (s *StreamAdapter) startToolCollection(content string) {
	s.buffer.WriteString(content)
	s.contentSuppressed = true
	s.toolCollectionState = toolStateCollecting
	s.collectionStartTime = time.Now()
	s.adapter.logger.Debug("Started tool collection, suppressing content",
		"content_prefix", s.truncateForLog(content, 50),
		"chunk_index", s.processedChunks,
		"policy", s.adapter.toolPolicy)
}

// addToolsToCollection adds tools to the collection with limit enforcement
func (s *StreamAdapter) addToolsToCollection(calls []functionCall) {
	// Apply tool limit enforcement
	remainingCapacity := len(calls)
	if s.adapter.toolMaxCalls > 0 {
		currentCount := len(s.collectedTools)
		maxNewTools := s.adapter.toolMaxCalls - currentCount
		if maxNewTools <= 0 {
			return // Already at capacity
		}
		if remainingCapacity > maxNewTools {
			remainingCapacity = maxNewTools
		}
	}

	// Add tools up to capacity
	if remainingCapacity > 0 {
		s.collectedTools = append(s.collectedTools, calls[:remainingCapacity]...)
	}
}

// processCollectedTools processes and emits all collected tools
func (s *StreamAdapter) processCollectedTools() {
	if len(s.collectedTools) > 0 {
		s.adapter.logger.Info("Processing collected tools",
			"tool_count", len(s.collectedTools),
			"collection_duration", time.Since(s.collectionStartTime))
		s.emitToolCallChunk(s.collectedTools)
	}
	s.toolCollectionState = toolStateFinished
}

// processBufferedContentForCollectionPhase processes buffered content during collection phase
func (s *StreamAdapter) processBufferedContentForCollectionPhase() {
	content := s.buffer.String()
	if content == "" {
		return
	}

	// Parse JSON candidates
	extractor := NewJSONExtractor(content)
	candidates := extractor.ExtractJSONBlocks()
	calls := ExtractFunctionCalls(candidates) // Simplified - no array detection
	if len(calls) == 0 {
		// Not a valid tool JSON; emit as regular content only if we haven't suppressed content
		if !s.contentSuppressed {
			s.emitContentChunk(content)
		}
		s.buffer.Reset()
		return
	}

	// Add tools to collection (with limit enforcement)
	s.addToolsToCollection(calls)

	// For CollectThenStop: continue collecting - don't immediately emit
	// Only emit when we hit explicit stop conditions (timeout, limits, etc.)
	// This allows multiple individual tool calls to be collected together

	s.buffer.Reset()
}

// truncateForLog safely truncates a string for logging purposes
func (s *StreamAdapter) truncateForLog(str string, maxLen int) string {
	if len(str) <= maxLen {
		return str
	}
	return str[:maxLen] + "..."
}
