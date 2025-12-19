package tooladapter

import (
	"context"
	"encoding/json"
	"strings"
)

// SSEStreamAdapter processes raw SSE streams to detect and transform tool calls.
// It buffers content to detect JSON function calls and transforms them into
// proper OpenAI-compatible tool_calls format.
//
// THREAD SAFETY: SSEStreamAdapter instances are NOT thread-safe.
// Each instance should be used by a single goroutine only.
//
// Usage:
//
//	adapter := tooladapter.New(tooladapter.WithToolPolicy(tooladapter.ToolStopOnFirst))
//	sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)
//	if err := sseAdapter.Process(ctx); err != nil {
//	    // handle error
//	}
type SSEStreamAdapter struct {
	adapter *Adapter
	reader  SSEStreamReader
	writer  SSEStreamWriter

	// Buffering state
	bufferLimit   int
	contentBuffer strings.Builder

	// Chunk metadata (captured from first chunk)
	completionID string
	model        string
	created      int64

	// Extra fields from provider (captured from first chunk)
	// These are preserved when emitting transformed tool call responses.
	// Examples: "reasoning", "system_fingerprint", etc.
	chunkExtraFields map[string]json.RawMessage

	// Context for cancellation
	ctx context.Context
}

// NewSSEStreamAdapter creates a new SSE stream adapter for processing raw SSE streams.
// The adapter will read from the provided reader and write transformed output to the writer.
func (a *Adapter) NewSSEStreamAdapter(reader SSEStreamReader, writer SSEStreamWriter) *SSEStreamAdapter {
	return &SSEStreamAdapter{
		adapter:     a,
		reader:      reader,
		writer:      writer,
		bufferLimit: a.streamBufferLimit,
	}
}

// captureMetadata captures metadata from the first chunk with a valid ID.
// This includes both standard fields and any provider-specific extra fields.
func (s *SSEStreamAdapter) captureMetadata(chunk *SSEChunk) {
	if s.completionID == "" && chunk.ID != "" {
		s.completionID = chunk.ID
		s.model = chunk.Model
		s.created = chunk.Created

		// Capture extra fields from the first chunk (e.g., "reasoning", "system_fingerprint")
		if len(chunk.ExtraFields) > 0 {
			s.chunkExtraFields = make(map[string]json.RawMessage)
			for k, v := range chunk.ExtraFields {
				s.chunkExtraFields[k] = v
			}
		}
	}
}

// Process reads the SSE stream, detects tool calls, and writes transformed output.
// It returns when the stream ends or an error occurs.
func (s *SSEStreamAdapter) Process(ctx context.Context) error {
	s.ctx = ctx

	// Collect all chunks for analysis
	var chunks []*SSEChunk
	var rawChunks []string

	for s.reader.Next() {
		// Check for cancellation
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		data := s.reader.Data()
		rawChunks = append(rawChunks, data)

		chunk, err := ParseSSEChunk(data)
		if err != nil {
			s.adapter.logger.Debug("Failed to parse SSE chunk, passing through",
				"error", err,
				"data_length", len(data))
			// Pass through unparseable chunks
			if err := s.writer.WriteRaw([]byte("data: " + data + "\n\n")); err != nil {
				return err
			}
			continue
		}

		chunks = append(chunks, chunk)

		// Capture metadata from first chunk (including extra fields)
		s.captureMetadata(chunk)

		// Accumulate content
		if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
			s.contentBuffer.WriteString(chunk.Choices[0].Delta.Content)
		}
	}

	// Check for reader errors
	if err := s.reader.Err(); err != nil {
		return err
	}

	// Analyze accumulated content for tool calls
	fullContent := s.contentBuffer.String()
	if fullContent == "" {
		// No content - pass through all chunks
		return s.passthrough(rawChunks)
	}

	// Try to extract tool calls from the content
	extractor := NewJSONExtractor(fullContent)
	candidates := extractor.ExtractJSONBlocks()

	if len(candidates) == 0 {
		// No JSON found - pass through all chunks
		return s.passthrough(rawChunks)
	}

	// Try to parse function calls
	calls := s.extractRawFunctionCalls(candidates)
	if len(calls) == 0 {
		// No valid function calls - pass through all chunks
		return s.passthrough(rawChunks)
	}

	// Apply tool policy and emit transformed response
	return s.emitToolCallResponse(calls, chunks)
}

// passthrough writes all chunks without modification.
func (s *SSEStreamAdapter) passthrough(rawChunks []string) error {
	for _, data := range rawChunks {
		if err := s.writer.WriteRaw([]byte("data: " + data + "\n\n")); err != nil {
			return err
		}
	}
	return s.writer.WriteDone()
}

// extractRawFunctionCalls extracts function calls from JSON candidates.
func (s *SSEStreamAdapter) extractRawFunctionCalls(candidates []string) []RawFunctionCall {
	for _, candidate := range candidates {
		// Try parsing as array first
		var arrayCalls []RawFunctionCall
		decoder := json.NewDecoder(strings.NewReader(candidate))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&arrayCalls); err == nil && len(arrayCalls) > 0 {
			if s.validateRawFunctionCalls(arrayCalls) {
				return arrayCalls
			}
		}

		// Try parsing as single object
		var singleCall RawFunctionCall
		decoder = json.NewDecoder(strings.NewReader(candidate))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&singleCall); err == nil {
			if ValidateFunctionName(singleCall.Name) == nil {
				return []RawFunctionCall{singleCall}
			}
		}
	}
	return nil
}

// validateRawFunctionCalls validates all function calls in a slice.
func (s *SSEStreamAdapter) validateRawFunctionCalls(calls []RawFunctionCall) bool {
	if len(calls) == 0 {
		return false
	}
	for _, call := range calls {
		if ValidateFunctionName(call.Name) != nil {
			return false
		}
	}
	return true
}

// emitToolCallResponse emits a transformed response with tool calls.
func (s *SSEStreamAdapter) emitToolCallResponse(calls []RawFunctionCall, originalChunks []*SSEChunk) error {
	// Apply tool policy limits
	calls = s.applyToolPolicy(calls)

	// Log detection
	functionNames := make([]string, len(calls))
	for i, call := range calls {
		functionNames[i] = call.Name
	}

	s.adapter.logger.Info("SSE streaming: detected and converted function calls",
		"function_count", len(calls),
		"function_names", functionNames,
		"content_length", s.contentBuffer.Len(),
		"streaming", true)

	// Emit metrics
	s.adapter.emitMetric(FunctionCallDetectionData{
		FunctionCount:  len(calls),
		FunctionNames:  functionNames,
		ContentLength:  s.contentBuffer.Len(),
		JSONCandidates: len(calls),
		Streaming:      true,
	})

	// Build tool calls
	toolCalls := make([]SSEToolCall, len(calls))
	for i, call := range calls {
		args := "{}"
		if len(call.Parameters) > 0 {
			args = string(call.Parameters)
		}

		toolCalls[i] = SSEToolCall{
			Index: i,
			ID:    s.adapter.GenerateToolCallID(),
			Type:  "function",
			Function: SSEFunctionCall{
				Name:      call.Name,
				Arguments: args,
			},
		}
	}

	// Emit tool call chunk with preserved extra fields
	toolChunk := &SSEChunk{
		ID:          s.completionID,
		Object:      "chat.completion.chunk",
		Created:     s.created,
		Model:       s.model,
		ExtraFields: s.chunkExtraFields, // Preserve provider-specific fields
		Choices: []SSEChoice{
			{
				Index: 0,
				Delta: SSEDelta{
					Role:      "assistant",
					ToolCalls: toolCalls,
				},
			},
		},
	}

	if err := s.writer.WriteChunk(toolChunk); err != nil {
		return err
	}

	// Emit finish chunk with preserved extra fields
	finishChunk := &SSEChunk{
		ID:          s.completionID,
		Object:      "chat.completion.chunk",
		Created:     s.created,
		Model:       s.model,
		ExtraFields: s.chunkExtraFields, // Preserve provider-specific fields
		Choices: []SSEChoice{
			{
				Index:        0,
				Delta:        SSEDelta{},
				FinishReason: "tool_calls",
			},
		},
	}

	if err := s.writer.WriteChunk(finishChunk); err != nil {
		return err
	}

	return s.writer.WriteDone()
}

// applyToolPolicy applies the configured tool policy to limit tool calls.
func (s *SSEStreamAdapter) applyToolPolicy(calls []RawFunctionCall) []RawFunctionCall {
	switch s.adapter.toolPolicy {
	case ToolStopOnFirst:
		// Return only the first tool call
		if len(calls) > 0 {
			s.adapter.logger.Debug("Applied ToolStopOnFirst policy",
				"original_count", len(calls),
				"result_count", 1)
			return calls[:1]
		}
		return calls

	case ToolCollectThenStop, ToolDrainAll:
		// Apply max calls limit
		if s.adapter.toolMaxCalls > 0 && len(calls) > s.adapter.toolMaxCalls {
			s.adapter.logger.Debug("Applied tool call limit",
				"policy", s.adapter.toolPolicy,
				"original_count", len(calls),
				"max_calls", s.adapter.toolMaxCalls)
			return calls[:s.adapter.toolMaxCalls]
		}
		return calls

	case ToolAllowMixed:
		// Allow all calls (with max limit)
		if s.adapter.toolMaxCalls > 0 && len(calls) > s.adapter.toolMaxCalls {
			return calls[:s.adapter.toolMaxCalls]
		}
		return calls

	default:
		return calls
	}
}

// passthroughState holds the state for passthrough processing.
type passthroughState struct {
	contentSeen         strings.Builder
	chunks              []*SSEChunk
	rawChunks           []string
	toolPatternDetected bool
}

// processChunk handles a single chunk during passthrough processing.
func (s *SSEStreamAdapter) processChunk(state *passthroughState, data string, earlyDetection int) {
	state.rawChunks = append(state.rawChunks, data)

	chunk, err := ParseSSEChunk(data)
	if err != nil {
		return
	}
	state.chunks = append(state.chunks, chunk)

	s.captureMetadata(chunk)
	s.accumulateContent(state, chunk, earlyDetection)
}

// accumulateContent adds content from a chunk and checks for tool patterns.
func (s *SSEStreamAdapter) accumulateContent(state *passthroughState, chunk *SSEChunk, earlyDetection int) {
	if len(chunk.Choices) == 0 || chunk.Choices[0].Delta.Content == "" {
		return
	}

	content := chunk.Choices[0].Delta.Content
	state.contentSeen.WriteString(content)

	if state.toolPatternDetected || state.contentSeen.Len() > earlyDetection {
		return
	}

	if s.hasToolPattern(state.contentSeen.String()) {
		state.toolPatternDetected = true
		s.adapter.logger.Debug("Tool pattern detected in early content",
			"content_length", state.contentSeen.Len(),
			"detection_limit", earlyDetection)
	}
}

// analyzeContentForTools performs full analysis for tool calls.
func (s *SSEStreamAdapter) analyzeContentForTools(state *passthroughState) ([]RawFunctionCall, bool) {
	s.contentBuffer.WriteString(state.contentSeen.String())

	fullContent := s.contentBuffer.String()
	if fullContent == "" {
		return nil, false
	}

	extractor := NewJSONExtractor(fullContent)
	candidates := extractor.ExtractJSONBlocks()
	if len(candidates) == 0 {
		return nil, false
	}

	calls := s.extractRawFunctionCalls(candidates)
	return calls, len(calls) > 0
}

// ProcessWithPassthrough processes an SSE stream with the option to pass through
// chunks that don't need transformation. This is more efficient when most responses
// don't contain tool calls.
//
// The earlyDetection parameter controls how many characters to scan before deciding
// whether to buffer for tool detection. Set to 0 to always buffer entire response.
func (s *SSEStreamAdapter) ProcessWithPassthrough(ctx context.Context, earlyDetection int) error {
	s.ctx = ctx

	if earlyDetection <= 0 {
		return s.Process(ctx)
	}

	state := &passthroughState{}

	for s.reader.Next() {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		s.processChunk(state, s.reader.Data(), earlyDetection)
	}

	if err := s.reader.Err(); err != nil {
		return err
	}

	// No tool pattern and sufficient content seen - pass through
	if !state.toolPatternDetected && state.contentSeen.Len() > earlyDetection {
		s.adapter.logger.Debug("No tool pattern in early content, passing through",
			"content_length", state.contentSeen.Len())
		return s.passthrough(state.rawChunks)
	}

	calls, hasCalls := s.analyzeContentForTools(state)
	if !hasCalls {
		return s.passthrough(state.rawChunks)
	}

	return s.emitToolCallResponse(calls, state.chunks)
}

// hasToolPattern checks if content contains patterns indicating a tool call.
func (s *SSEStreamAdapter) hasToolPattern(content string) bool {
	trimmed := strings.TrimSpace(content)
	if trimmed == "" {
		return false
	}

	// Check for immediate JSON patterns
	if strings.HasPrefix(trimmed, `{"name":`) ||
		strings.HasPrefix(trimmed, `{"name": `) ||
		strings.HasPrefix(trimmed, `[{"name":`) ||
		strings.HasPrefix(trimmed, `[{"name": `) {
		return true
	}

	// Check for markdown code blocks
	if strings.HasPrefix(trimmed, "```json") || strings.HasPrefix(trimmed, "```") {
		if strings.Contains(trimmed, `"name"`) {
			return true
		}
	}

	// Check for backtick-enclosed patterns
	if strings.Contains(trimmed, "`{\"name\"") || strings.Contains(trimmed, "`[{\"name\"") {
		return true
	}

	return false
}

// SSETransformResult represents the result of processing an SSE stream.
type SSETransformResult struct {
	// HasToolCalls indicates whether tool calls were detected.
	HasToolCalls bool

	// ToolCalls contains the extracted tool calls (if any).
	ToolCalls []RawFunctionCall

	// Content contains the original content (if no tool calls or mixed mode).
	Content string

	// Passthrough indicates whether the response was passed through unchanged.
	Passthrough bool
}

// ProcessToResult processes the stream and returns a result instead of writing.
// This is useful when you need to inspect the result before deciding how to handle it.
func (s *SSEStreamAdapter) ProcessToResult(ctx context.Context) (*SSETransformResult, []*SSEChunk, error) {
	s.ctx = ctx

	var chunks []*SSEChunk

	for s.reader.Next() {
		select {
		case <-ctx.Done():
			return nil, nil, ctx.Err()
		default:
		}

		data := s.reader.Data()
		chunk, err := ParseSSEChunk(data)
		if err != nil {
			continue
		}

		chunks = append(chunks, chunk)

		// Capture metadata
		if s.completionID == "" && chunk.ID != "" {
			s.completionID = chunk.ID
			s.model = chunk.Model
			s.created = chunk.Created
		}

		// Accumulate content
		if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
			s.contentBuffer.WriteString(chunk.Choices[0].Delta.Content)
		}
	}

	if err := s.reader.Err(); err != nil {
		return nil, chunks, err
	}

	fullContent := s.contentBuffer.String()
	result := &SSETransformResult{
		Content: fullContent,
	}

	if fullContent == "" {
		result.Passthrough = true
		return result, chunks, nil
	}

	// Extract tool calls
	extractor := NewJSONExtractor(fullContent)
	candidates := extractor.ExtractJSONBlocks()

	if len(candidates) == 0 {
		result.Passthrough = true
		return result, chunks, nil
	}

	calls := s.extractRawFunctionCalls(candidates)
	if len(calls) == 0 {
		result.Passthrough = true
		return result, chunks, nil
	}

	// Apply policy
	calls = s.applyToolPolicy(calls)

	result.HasToolCalls = true
	result.ToolCalls = calls
	result.Passthrough = false

	return result, chunks, nil
}

// WriteToolCallsFromResult writes tool calls from a ProcessToResult output.
// This allows you to inspect the result first and then decide whether to write it.
func (s *SSEStreamAdapter) WriteToolCallsFromResult(result *SSETransformResult) error {
	if !result.HasToolCalls {
		return nil
	}

	// Build tool calls
	toolCalls := make([]SSEToolCall, len(result.ToolCalls))
	for i, call := range result.ToolCalls {
		args := "{}"
		if len(call.Parameters) > 0 {
			args = string(call.Parameters)
		}

		toolCalls[i] = SSEToolCall{
			Index: i,
			ID:    s.adapter.GenerateToolCallID(),
			Type:  "function",
			Function: SSEFunctionCall{
				Name:      call.Name,
				Arguments: args,
			},
		}
	}

	// Emit tool call chunk with preserved extra fields
	toolChunk := &SSEChunk{
		ID:          s.completionID,
		Object:      "chat.completion.chunk",
		Created:     s.created,
		Model:       s.model,
		ExtraFields: s.chunkExtraFields, // Preserve provider-specific fields
		Choices: []SSEChoice{
			{
				Index: 0,
				Delta: SSEDelta{
					Role:      "assistant",
					ToolCalls: toolCalls,
				},
			},
		},
	}

	if err := s.writer.WriteChunk(toolChunk); err != nil {
		return err
	}

	// Emit finish chunk with preserved extra fields
	finishChunk := &SSEChunk{
		ID:          s.completionID,
		Object:      "chat.completion.chunk",
		Created:     s.created,
		Model:       s.model,
		ExtraFields: s.chunkExtraFields, // Preserve provider-specific fields
		Choices: []SSEChoice{
			{
				Index:        0,
				Delta:        SSEDelta{},
				FinishReason: "tool_calls",
			},
		},
	}

	if err := s.writer.WriteChunk(finishChunk); err != nil {
		return err
	}

	return s.writer.WriteDone()
}

// WritePassthrough writes chunks without modification.
func (s *SSEStreamAdapter) WritePassthrough(chunks []*SSEChunk) error {
	for _, chunk := range chunks {
		if err := s.writer.WriteChunk(chunk); err != nil {
			return err
		}
	}
	return s.writer.WriteDone()
}
