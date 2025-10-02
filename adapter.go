// Package tooladapter provides OpenAI tool compatibility for Large Language Models
// that lack native function calling support. It transforms OpenAI-style tool requests
// into prompt-based format and parses model responses back into structured tool calls.
//
// CONCURRENCY SUMMARY:
//   - Adapter: Thread-safe, can be shared across goroutines
//   - StreamAdapter: NOT thread-safe, single-consumer design
//   - Parser functions: Thread-safe, stateless operations
package tooladapter

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/openai/openai-go/v3"
)

// Define the constant value for "function" type
const functionType = "function"

// Adapter translates standard tool-call requests into a prompt-based format.
//
// THREAD SAFETY: Adapter instances are safe for concurrent use by multiple goroutines.
// All public methods can be called concurrently without external synchronization.
//
// Concurrency design:
//   - All fields are immutable after construction (set once during New())
//   - sync.Pool handles concurrent buffer access internally
//   - slog.Logger is thread-safe
//   - Metrics callbacks should be implemented as thread-safe by users
//   - No shared mutable state between method calls
//
// Usage patterns:
//   - Single adapter instance can handle requests from multiple goroutines
//   - Each method call is independent and stateless
//   - StreamAdapter instances are NOT thread-safe (single-consumer design)
type Adapter struct {
	bufferPool      sync.Pool
	promptTemplate  string
	logger          *slog.Logger
	metricsCallback func(MetricEventData)

	// Tool policy configuration
	toolPolicy           ToolPolicy
	toolCollectWindow    time.Duration // streaming only; 0 => structure-only (no timer)
	toolMaxCalls         int           // cap across streaming + non-streaming (e.g., 8)
	toolCollectMaxBytes  int           // safety cap for JSON collection (e.g., 64*1024)
	cancelUpstreamOnStop bool          // streaming only; default true

	// Buffer size configuration
	streamBufferLimit    int // streaming buffer limit (e.g., 10*1024*1024)
	bufferPoolThreshold  int // buffer pool size threshold (e.g., 64*1024)
	streamLookAheadLimit int // early tool detection lookahead limit in chars (e.g., 100)

	// Indicates whether the model and its chat template support system messages
	// We leave it up to the caller to determine versus building a giant model registry
	systemMessagesSupported bool
}

// Internal structs for JSON manipulation
type functionCall struct {
	Name       string          `json:"name"`
	Parameters json.RawMessage `json:"parameters"`
}

// New creates a new tool adapter with optional configurations
func New(opts ...Option) *Adapter {
	adapter := &Adapter{
		promptTemplate: DefaultPromptTemplate,
		// Initialize with a no-op logger to avoid nil pointer issues
		logger: slog.New(slog.NewTextHandler(io.Discard, &slog.HandlerOptions{
			Level: slog.LevelError + 1, // Effectively disable all logging by default
		})),

		// Set default tool policy values
		toolPolicy:           ToolStopOnFirst,
		toolCollectWindow:    200 * time.Millisecond, // ignored if non-streaming or ==0
		toolMaxCalls:         8,
		toolCollectMaxBytes:  65536, // 64KB default limit for security (prevents DoS via memory exhaustion)
		cancelUpstreamOnStop: true,

		// Set default buffer size values
		streamBufferLimit:       10 * 1024 * 1024, // 10MB default streaming buffer limit
		bufferPoolThreshold:     64 * 1024,        // 64KB buffer pool threshold
		streamLookAheadLimit:    0,                // 0 = disabled, early detection off by default
		systemMessagesSupported: false,            // gemma will be the top model used with this package
	}

	// Apply all provided options
	for _, opt := range opts {
		opt(adapter)
	}

	// Buffer pool for efficient string building with memory growth protection
	adapter.bufferPool = sync.Pool{
		New: func() interface{} {
			return bytes.NewBuffer(make([]byte, 0, 1024))
		},
	}

	return adapter
}

// putBufferToPool safely returns a buffer to the pool with memory growth protection.
// Buffers that have grown beyond the configured size threshold are discarded to prevent
// unbounded memory growth in the pool.
func (a *Adapter) putBufferToPool(buf *bytes.Buffer) {
	buf.Reset() // Clear contents but preserve capacity

	// If the buffer has grown too large, don't return it to the pool
	// This prevents memory leaks from oversized buffers that were used for large tool sets
	if buf.Cap() <= a.bufferPoolThreshold {
		a.bufferPool.Put(buf)
	}
	// If buffer exceeds threshold, let it be garbage collected instead of pooling
}

// TransformCompletionsRequest modifies a chat completion request to inject tool definitions.
// This is the backward-compatible version that uses context.Background().
// For production use with timeouts and cancellation, use TransformCompletionsRequestWithContext.
func (a *Adapter) TransformCompletionsRequest(req openai.ChatCompletionNewParams) (openai.ChatCompletionNewParams, error) {
	return a.TransformCompletionsRequestWithContext(context.Background(), req)
}

// TransformCompletionsRequestWithContext modifies a chat completion request to inject tool definitions
// and process tool results with context support for cancellation and timeouts.
func (a *Adapter) TransformCompletionsRequestWithContext(ctx context.Context, req openai.ChatCompletionNewParams) (openai.ChatCompletionNewParams, error) {
	startTime := time.Now()

	// Check for cancellation early
	select {
	case <-ctx.Done():
		return openai.ChatCompletionNewParams{}, ctx.Err()
	default:
	}

	// Extract tool results from messages and filter out ToolMessage types
	toolResults, cleanMessages, err := a.extractToolResults(req.Messages)
	if err != nil {
		a.logger.Error("Failed to extract tool results", "error", err)
		return openai.ChatCompletionNewParams{}, fmt.Errorf("failed to extract tool results: %w", err)
	}

	// Determine what we have: tools, tool results, both, or neither
	hasTools := len(req.Tools) > 0
	hasToolResults := len(toolResults) > 0

	// Case 1: Neither tools nor tool results - pass through unchanged
	if !hasTools && !hasToolResults {
		a.logger.Debug("No tools or tool results present, passing through unchanged")
		return req, nil
	}

	// Extract tool names for logging and metrics
	toolNames := make([]string, 0, len(req.Tools))
	for _, tool := range req.Tools {
		if function := tool.GetFunction(); function != nil {
			toolNames = append(toolNames, function.Name)
		}
	}

	// Check for cancellation before expensive operation
	select {
	case <-ctx.Done():
		return openai.ChatCompletionNewParams{}, ctx.Err()
	default:
	}

	// Build the combined prompt based on what we have
	var combinedPrompt string

	if hasTools && hasToolResults {
		// Case 2: Both tools and tool results
		toolPrompt, err := a.buildToolPromptWithContext(ctx, req.Tools)
		if err != nil {
			a.logger.Error("Failed to build tool prompt", "error", err, "tool_count", len(req.Tools))
			return openai.ChatCompletionNewParams{}, fmt.Errorf("failed to build tool prompt: %w", err)
		}
		toolResultsPrompt := a.buildToolResultsPrompt(toolResults)
		combinedPrompt = toolPrompt + "\n\n" + toolResultsPrompt

		a.logger.Info("Transformed request: tools and tool results present",
			"tool_count", len(req.Tools),
			"tool_names", toolNames,
			"tool_results_count", len(toolResults),
			"combined_prompt_length", len(combinedPrompt))

	} else if hasTools {
		// Case 3: Only tools (original behavior)
		combinedPrompt, err = a.buildToolPromptWithContext(ctx, req.Tools)
		if err != nil {
			a.logger.Error("Failed to build tool prompt", "error", err, "tool_count", len(req.Tools))
			return openai.ChatCompletionNewParams{}, fmt.Errorf("failed to build tool prompt: %w", err)
		}

		a.logger.Info("Transformed request: tools present",
			"tool_count", len(req.Tools),
			"tool_names", toolNames,
			"prompt_length", len(combinedPrompt))

	} else {
		// Case 4: Only tool results (no callable tools)
		combinedPrompt = a.buildToolResultsPrompt(toolResults)

		a.logger.Info("Transformed request: tool results present",
			"tool_results_count", len(toolResults),
			"prompt_length", len(combinedPrompt))
	}

	totalDuration := time.Since(startTime)

	// Emit metrics event
	a.emitMetric(ToolTransformationData{
		ToolCount:    len(req.Tools),
		ToolNames:    toolNames,
		PromptLength: len(combinedPrompt),
		Performance: PerformanceMetrics{
			ProcessingDuration: totalDuration,
		},
	})

	// Apply the combined prompt with cleaned messages (ToolMessages removed)
	modifiedReq := req
	modifiedReq.Messages = cleanMessages
	return a.applyToolPrompt(modifiedReq, combinedPrompt), nil
}

// TransformCompletionsResponse processes LLM responses to extract and format tool calls.
// This is the backward-compatible version that uses context.Background().
// For production use with timeouts and cancellation, use TransformCompletionsResponseWithContext.
func (a *Adapter) TransformCompletionsResponse(resp openai.ChatCompletion) (openai.ChatCompletion, error) {
	return a.TransformCompletionsResponseWithContext(context.Background(), resp)
}

// processChoiceForToolCalls extracts and processes tool calls from a single choice
// Returns the function calls, timing metrics, and whether processing should continue
func (a *Adapter) processChoiceForToolCalls(
	ctx context.Context,
	choice *openai.ChatCompletionChoice,
	choiceIndex int,
	startTime time.Time,
) ([]functionCall, time.Duration, time.Duration, bool) {
	// Skip choices without content
	if choice.Message.Content == "" {
		a.logger.Debug("No content in choice, skipping",
			"choice_index", choiceIndex)
		return nil, 0, 0, false
	}

	content := choice.Message.Content
	contentLength := len(content)

	// Check for cancellation before expensive parsing
	select {
	case <-ctx.Done():
		return nil, 0, 0, false
	default:
	}

	// Track timing for JSON extraction
	jsonStartTime := time.Now()

	// Use state machine parser to extract JSON blocks
	extractor := NewJSONExtractor(content)
	candidates := extractor.ExtractJSONBlocks()

	jsonParsingTime := time.Since(jsonStartTime)

	if len(candidates) == 0 {
		a.logger.Debug("No JSON candidates found in choice content",
			"choice_index", choiceIndex,
			"content_length", contentLength)
		return nil, jsonParsingTime, 0, false
	}

	// Track timing for function call extraction
	extractionStartTime := time.Now()

	// Extract function calls from candidates
	calls := ExtractFunctionCalls(candidates)

	extractionTime := time.Since(extractionStartTime)

	if len(calls) == 0 {
		a.logger.Debug("No valid function calls extracted from JSON candidates",
			"choice_index", choiceIndex,
			"candidate_count", len(candidates),
			"content_length", contentLength)
		return nil, jsonParsingTime, extractionTime, false
	}

	// Log and emit metrics for detected function calls
	a.logAndEmitFunctionCalls(ctx, calls, choiceIndex, contentLength, len(candidates), startTime, jsonParsingTime, extractionTime)

	return calls, jsonParsingTime, extractionTime, true
}

// logAndEmitFunctionCalls handles logging and metrics emission for detected function calls
func (a *Adapter) logAndEmitFunctionCalls(
	ctx context.Context,
	calls []functionCall,
	choiceIndex int,
	contentLength int,
	candidateCount int,
	startTime time.Time,
	jsonParsingTime time.Duration,
	extractionTime time.Duration,
) {
	// Extract function names for logging and metrics
	functionNames := make([]string, len(calls))
	for i, call := range calls {
		functionNames[i] = call.Name
	}

	// Log the detection and conversion for this choice
	logAttrs := []any{
		"choice_index", choiceIndex,
		"function_count", len(calls),
		"function_names", functionNames,
		"content_length", contentLength,
		"json_candidates", candidateCount,
	}

	// In debug mode, also log the function arguments
	if a.logger.Enabled(ctx, slog.LevelDebug) {
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

	a.logger.Info("Transformed choice: detected and converted function calls", logAttrs...)

	// Emit metrics for this specific choice
	a.emitMetric(FunctionCallDetectionData{
		FunctionCount:  len(calls),
		FunctionNames:  functionNames,
		ContentLength:  contentLength,
		JSONCandidates: candidateCount,
		Streaming:      false,
		Performance: PerformanceMetrics{
			ProcessingDuration: time.Since(startTime),
			SubOperations: map[string]time.Duration{
				"json_parsing":    jsonParsingTime,
				"call_extraction": extractionTime,
			},
		},
	})
}

// TransformCompletionsResponseWithContext processes LLM responses to extract and format tool calls
// with context support for cancellation and timeouts.
// This function now processes ALL choices in the response, not just the first one.
func (a *Adapter) TransformCompletionsResponseWithContext(ctx context.Context, resp openai.ChatCompletion) (openai.ChatCompletion, error) {
	startTime := time.Now()

	// Check for cancellation early
	select {
	case <-ctx.Done():
		return openai.ChatCompletion{}, ctx.Err()
	default:
	}

	// Guard clauses: return early if there's nothing to process
	if len(resp.Choices) == 0 {
		a.logger.Debug("No choices in response, passing through unchanged")
		return resp, nil
	}

	// Track whether we've modified anything to avoid unnecessary copying
	var modifiedResp openai.ChatCompletion
	var choicesCopied bool

	totalToolCallsAcrossChoices := 0
	choicesWithTools := 0

	// Process each choice independently
	for choiceIndex := range resp.Choices {
		choice := &resp.Choices[choiceIndex]

		// Process the choice for tool calls
		calls, _, _, shouldContinue := a.processChoiceForToolCalls(ctx, choice, choiceIndex, startTime)
		if !shouldContinue {
			// Check if context was cancelled
			select {
			case <-ctx.Done():
				return openai.ChatCompletion{}, ctx.Err()
			default:
				continue
			}
		}

		if len(calls) == 0 {
			continue
		}

		// Apply tool policy to this specific choice
		transformedChoice, err := a.applyToolPolicyToChoice(*choice, calls, choiceIndex)
		if err != nil {
			a.logger.Error("Failed to apply tool policy to choice",
				"choice_index", choiceIndex,
				"error", err)
			continue
		}

		// Only create a copy of the response if this is the first modification.
		// This lazy allocation avoids copying when no tool calls are found.
		if !choicesCopied {
			modifiedResp = resp
			// Always create a new slice to avoid race conditions when the same
			// response object is processed concurrently by multiple goroutines
			modifiedResp.Choices = make([]openai.ChatCompletionChoice, len(resp.Choices))
			copy(modifiedResp.Choices, resp.Choices)
			choicesCopied = true
		}

		// Update the choice in the response
		modifiedResp.Choices[choiceIndex] = transformedChoice

		// Track statistics
		totalToolCallsAcrossChoices += len(transformedChoice.Message.ToolCalls)
		if len(transformedChoice.Message.ToolCalls) > 0 {
			choicesWithTools++
		}
	}

	// If we never copied (no tool calls found), return the original response
	if !choicesCopied {
		a.logger.Debug("No tool calls found in any choice, returning original response",
			"total_choices", len(resp.Choices))
		return resp, nil
	}

	a.logger.Debug("Completed multi-choice transformation",
		"total_choices", len(resp.Choices),
		"choices_with_tools", choicesWithTools,
		"total_tool_calls", totalToolCallsAcrossChoices,
		"total_duration", time.Since(startTime))

	return modifiedResp, nil
}

// applyToolPolicyToChoice applies the configured tool policy to a single choice
// from the response. This allows each choice to be transformed independently
// according to the policy.
func (a *Adapter) applyToolPolicyToChoice(
	choice openai.ChatCompletionChoice,
	calls []functionCall,
	choiceIndex int,
) (openai.ChatCompletionChoice, error) {
	// If no tool calls detected, return the choice unchanged
	if len(calls) == 0 {
		return choice, nil
	}

	// Apply policy-specific transformations
	switch a.toolPolicy {
	case ToolAllowMixed:
		// In mixed mode, return both content and tool calls
		return a.buildMixedChoice(choice, calls, choiceIndex)

	case ToolStopOnFirst:
		// Return only the first tool call with empty content
		return a.buildStopOnFirstChoice(choice, calls, choiceIndex)

	case ToolCollectThenStop:
		// Apply collection limits and return tools with empty content
		return a.buildCollectThenStopChoice(choice, calls, choiceIndex)

	case ToolDrainAll:
		// Return all detected tools with empty content
		return a.buildDrainAllChoice(choice, calls, choiceIndex)

	default:
		// Fallback to ToolStopOnFirst for unknown policies
		a.logger.Warn("Unknown tool policy, falling back to ToolStopOnFirst",
			"policy", a.toolPolicy,
			"choice_index", choiceIndex)
		return a.buildStopOnFirstChoice(choice, calls, choiceIndex)
	}
}

// buildMixedChoice creates a choice with both content and tool calls
func (a *Adapter) buildMixedChoice(choice openai.ChatCompletionChoice, calls []functionCall, choiceIndex int) (openai.ChatCompletionChoice, error) {
	// Apply collection limits
	maxCalls := len(calls)
	if a.toolMaxCalls > 0 && a.toolMaxCalls < maxCalls {
		maxCalls = a.toolMaxCalls
		a.logger.Debug("Applied tool call limit in mixed mode",
			"choice_index", choiceIndex,
			"original_calls", len(calls),
			"limited_to", maxCalls)
	}

	// Create tool calls while preserving original content
	toolCalls := make([]openai.ChatCompletionMessageToolCallUnion, maxCalls)
	for i, call := range calls[:maxCalls] {
		parameters := call.Parameters
		if parameters == nil {
			parameters = json.RawMessage("null")
		}

		toolCalls[i] = openai.ChatCompletionMessageToolCallUnion{
			ID:   a.GenerateToolCallID(),
			Type: functionType,
			Function: openai.ChatCompletionMessageFunctionToolCallFunction{
				Name:      call.Name,
				Arguments: string(parameters),
			},
		}
	}

	// Keep original content and add tool calls
	modifiedChoice := choice
	modifiedChoice.Message.ToolCalls = toolCalls

	// Preserve original finish_reason since content is preserved in mixed mode
	originalFinishReason := choice.FinishReason
	// Only override if the original was empty/unset
	if originalFinishReason == "" {
		modifiedChoice.FinishReason = "tool_calls"
	}

	a.logger.Debug("Built mixed choice with content and tool calls",
		"choice_index", choiceIndex,
		"content_preserved", true,
		"collected_calls", len(toolCalls),
		"total_detected", len(calls))

	return modifiedChoice, nil
}

// buildStopOnFirstChoice creates a choice with only the first tool call
func (a *Adapter) buildStopOnFirstChoice(choice openai.ChatCompletionChoice, calls []functionCall, choiceIndex int) (openai.ChatCompletionChoice, error) {
	// Use only the first tool call
	firstCall := calls[0]
	parameters := firstCall.Parameters
	if parameters == nil {
		parameters = json.RawMessage("null")
	}

	toolCalls := []openai.ChatCompletionMessageToolCallUnion{
		{
			ID:   a.GenerateToolCallID(),
			Type: functionType,
			Function: openai.ChatCompletionMessageFunctionToolCallFunction{
				Name:      firstCall.Name,
				Arguments: string(parameters),
			},
		},
	}

	// Clear content and set tool calls
	modifiedChoice := choice
	modifiedChoice.Message.Content = ""
	modifiedChoice.Message.ToolCalls = toolCalls
	modifiedChoice.FinishReason = "tool_calls"

	a.logger.Debug("Built stop-on-first choice",
		"choice_index", choiceIndex,
		"content_cleared", true,
		"first_tool_call", firstCall.Name,
		"discarded_calls", len(calls)-1)

	return modifiedChoice, nil
}

// buildCollectThenStopChoice creates a choice with collected tools up to limits
func (a *Adapter) buildCollectThenStopChoice(choice openai.ChatCompletionChoice, calls []functionCall, choiceIndex int) (openai.ChatCompletionChoice, error) {
	// Apply collection limits
	maxCalls := len(calls)
	if a.toolMaxCalls > 0 && a.toolMaxCalls < maxCalls {
		maxCalls = a.toolMaxCalls
		a.logger.Debug("Applied tool call limit in collect-then-stop mode",
			"choice_index", choiceIndex,
			"original_calls", len(calls),
			"limited_to", maxCalls)
	}

	// Create tool calls up to the limit
	toolCalls := make([]openai.ChatCompletionMessageToolCallUnion, maxCalls)
	for i, call := range calls[:maxCalls] {
		parameters := call.Parameters
		if parameters == nil {
			parameters = json.RawMessage("null")
		}

		toolCalls[i] = openai.ChatCompletionMessageToolCallUnion{
			ID:   a.GenerateToolCallID(),
			Type: functionType,
			Function: openai.ChatCompletionMessageFunctionToolCallFunction{
				Name:      call.Name,
				Arguments: string(parameters),
			},
		}
	}

	// Clear content and set collected tool calls
	modifiedChoice := choice
	modifiedChoice.Message.Content = ""
	modifiedChoice.Message.ToolCalls = toolCalls
	modifiedChoice.FinishReason = "tool_calls"

	a.logger.Debug("Built collect-then-stop choice",
		"choice_index", choiceIndex,
		"content_cleared", true,
		"collected_calls", len(toolCalls),
		"total_detected", len(calls))

	return modifiedChoice, nil
}

// buildDrainAllChoice creates a choice with all detected tool calls
func (a *Adapter) buildDrainAllChoice(choice openai.ChatCompletionChoice, calls []functionCall, choiceIndex int) (openai.ChatCompletionChoice, error) {
	// Apply global max limit as safety
	maxCalls := len(calls)
	if a.toolMaxCalls > 0 && a.toolMaxCalls < maxCalls {
		maxCalls = a.toolMaxCalls
		a.logger.Debug("Applied tool call limit in drain-all mode",
			"choice_index", choiceIndex,
			"original_calls", len(calls),
			"limited_to", maxCalls)
	}

	// Create tool calls for all detected calls
	toolCalls := make([]openai.ChatCompletionMessageToolCallUnion, maxCalls)
	for i, call := range calls[:maxCalls] {
		parameters := call.Parameters
		if parameters == nil {
			parameters = json.RawMessage("null")
		}

		toolCalls[i] = openai.ChatCompletionMessageToolCallUnion{
			ID:   a.GenerateToolCallID(),
			Type: functionType,
			Function: openai.ChatCompletionMessageFunctionToolCallFunction{
				Name:      call.Name,
				Arguments: string(parameters),
			},
		}
	}

	// Clear content and set all tool calls
	modifiedChoice := choice
	modifiedChoice.Message.Content = ""
	modifiedChoice.Message.ToolCalls = toolCalls
	modifiedChoice.FinishReason = "tool_calls"

	a.logger.Debug("Built drain-all choice",
		"choice_index", choiceIndex,
		"content_cleared", true,
		"drained_calls", len(toolCalls))

	return modifiedChoice, nil
}

// applyToolPolicy applies the configured tool policy to transform the response
// for non-streaming responses based on the detected tool calls.
// DEPRECATED: This function is kept for backward compatibility but should not be used
// buildToolPromptWithContext constructs the system prompt with tool definitions
// with context support for cancellation and timeouts.
func (a *Adapter) buildToolPromptWithContext(ctx context.Context, tools []openai.ChatCompletionToolUnionParam) (string, error) {
	if len(tools) == 0 {
		return "", nil
	}

	// Check for cancellation early
	select {
	case <-ctx.Done():
		return "", ctx.Err()
	default:
	}

	startTime := time.Now()

	// Use buffer pool for efficient string building
	buf := a.bufferPool.Get().(*bytes.Buffer)
	defer func() {
		a.putBufferToPool(buf)
	}()

	// Build human-readable tool descriptions
	for i, tool := range tools {
		// Check for cancellation in tool processing loop
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		default:
		}

		// Get the function definition from the union type
		function := tool.GetFunction()
		if function == nil {
			continue // Skip if this isn't a function tool
		}

		// Start with name and description - the core information LLMs need
		fmt.Fprintf(buf, "- %s", function.Name)

		// Use param.Opt's Or() method for efficient access with fallback
		if desc := function.Description.Or(""); desc != "" {
			fmt.Fprintf(buf, ": %s", desc)
		}

		// Include parameter schema if available - use compact JSON (no indentation)
		if function.Parameters != nil {
			paramsJSON, err := json.Marshal(function.Parameters) // Compact JSON, no indent
			if err == nil {
				fmt.Fprintf(buf, "\n  Parameters: %s", string(paramsJSON))
			}
		}

		// Include strict mode flag if specified (OpenAI Structured Outputs)
		// Note: We pass this field through for compatibility but don't add verbose
		// prompt instructions since small LLMs may not reliably follow strict compliance
		if function.Strict.Or(false) {
			buf.WriteString("\n  Strict: true")
		}

		// Add spacing between tools for readability
		if i < len(tools)-1 {
			buf.WriteString("\n")
		}
	}

	// Format the complete prompt using our template
	prompt := fmt.Sprintf(a.promptTemplate, buf.String())

	duration := time.Since(startTime)
	a.logger.Debug("Built tool prompt",
		"tool_count", len(tools),
		"prompt_length", len(prompt),
		"build_duration", duration)

	return prompt, nil
}

// applyToolPrompt injects tool instructions while preserving existing message parts
// (e.g., images/audio) and aligning with provider/template requirements.
//
// Strategy:
//  1. If there is at least one system message: append the tool instructions to the
//     LAST system message. This keeps the message count stable and leverages the
//     "last system wins" heuristic many templates/models use.
//  2. Else (no system present): choose injection role based on model capabilities:
//     - If the model likely DOES NOT support a system role (e.g., Gemma 3): INSERT a new
//     USER instruction message immediately BEFORE the first user message to avoid
//     mutating multimodal content and to keep instructions authoritative.
//     - Otherwise: PREPEND a single SYSTEM instruction message at the start to satisfy
//     templates that expect a leading system with strict role alternation.
//  3. Else (no system and no user present): INSERT a new instruction message. Prefer
//     SYSTEM for generic compatibility; prefer USER for models without system support.
func (a *Adapter) applyToolPrompt(req openai.ChatCompletionNewParams, toolPrompt string) openai.ChatCompletionNewParams {
	modifiedReq := req

	// Remove tool-related fields since the target model doesn't support them
	modifiedReq.Tools = nil
	// Clear tool choice - use zero value of the union type
	modifiedReq.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{}

	// Handle empty messages case first
	if len(modifiedReq.Messages) == 0 {
		// No messages: create instruction message based on system support configuration
		if a.systemMessagesSupported {
			modifiedReq.Messages = []openai.ChatCompletionMessageParamUnion{
				openai.SystemMessage(toolPrompt),
			}
			a.logger.Debug("Created new system message with tool prompt",
				"system_prompt_length", len(toolPrompt))
		} else {
			modifiedReq.Messages = []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage(toolPrompt),
			}
			a.logger.Debug("Created new user instruction with tool prompt",
				"instruction_length", len(toolPrompt))
		}
		return modifiedReq
	}

	// Find LAST system message or first user message to anchor insertion point
	lastSystemIndex := -1
	firstUserIndex := -1
	for i, m := range modifiedReq.Messages {
		if m.OfSystem != nil {
			lastSystemIndex = i // Keep updating to find the LAST one
		}
		if m.OfUser != nil && firstUserIndex == -1 {
			firstUserIndex = i // Only need the first user message
		}
	}

	// Copy messages to avoid modifying the original
	newMessages := make([]openai.ChatCompletionMessageParamUnion, len(modifiedReq.Messages))
	copy(newMessages, modifiedReq.Messages)

	// Preferred strategy:
	// - If a system message exists: append tool instructions to the LAST system message (safe, text-only)
	// - Else decide based on model capabilities (heuristic):
	//   * If model lacks system support (e.g., Gemma 3): INSERT a USER instruction BEFORE the first user
	//     (or at start if no user found) to preserve multimodal content and avoid system role.
	//   * Otherwise: PREPEND a SYSTEM instruction at the start to satisfy templates.
	if lastSystemIndex != -1 {
		// System message exists: append tool prompt to the LAST one (keeps count unchanged)
		originalContent := extractSystemContent(newMessages[lastSystemIndex])
		combinedContent := originalContent + "\n\n" + toolPrompt
		newMessages[lastSystemIndex] = openai.SystemMessage(combinedContent)

		a.logger.Debug("Appended tool prompt to last system message",
			"system_index", lastSystemIndex,
			"original_length", len(originalContent),
			"tool_prompt_length", len(toolPrompt),
			"combined_length", len(combinedContent))
	} else if firstUserIndex != -1 {
		// No system message present.
		if !a.systemMessagesSupported {
			// Modify the first user message by prepending tool instructions
			// This maintains alternating roles (user/assistant/user/assistant)
			// Providers like vLLM w/ Gemma 3 will error out if you try to use
			// two user messages consecutively.
			newMessages[firstUserIndex] = prependToolPromptToUserMessage(newMessages[firstUserIndex], toolPrompt)

			a.logger.Debug("Prepended tool prompt to first user message",
				"user_index", firstUserIndex,
				"tool_prompt_length", len(toolPrompt))
		} else {
			// Prepend a SYSTEM instruction to satisfy templates that expect it
			newMessages = append([]openai.ChatCompletionMessageParamUnion{openai.SystemMessage(toolPrompt)}, newMessages...)
			a.logger.Debug("Prepended system instruction (configured RoleSystem)",
				"tool_prompt_length", len(toolPrompt),
				"new_message_count", len(newMessages))
		}
	} else {
		// No system or user messages (only assistant or empty). Use configured default.
		if !a.systemMessagesSupported {
			newMessages = append([]openai.ChatCompletionMessageParamUnion{openai.UserMessage(toolPrompt)}, newMessages...)
			a.logger.Debug("Prepended new user instruction (no system/user messages found, configured RoleUser)",
				"original_message_count", len(modifiedReq.Messages),
				"new_message_count", len(newMessages))
		} else {
			newMessages = append([]openai.ChatCompletionMessageParamUnion{openai.SystemMessage(toolPrompt)}, newMessages...)
			a.logger.Debug("Prepended new system message (no system/user messages found, configured RoleSystem)",
				"original_message_count", len(modifiedReq.Messages),
				"new_message_count", len(newMessages))
		}
	}

	modifiedReq.Messages = newMessages
	return modifiedReq
}

// prependToolPromptToUserMessage creates a new user message with tool prompt prepended
// while preserving any existing multimodal content (images, etc.)
func prependToolPromptToUserMessage(msg openai.ChatCompletionMessageParamUnion, toolPrompt string) openai.ChatCompletionMessageParamUnion {
	if msg.OfUser == nil {
		// Not a user message, return as-is
		return msg
	}

	content := msg.OfUser.Content

	// Handle simple text content
	if str := content.OfString.Or(""); str != "" {
		// Simple text message - combine with tool prompt
		combinedContent := toolPrompt + "\n\n" + str
		return openai.UserMessage(combinedContent)
	}

	// Handle multimodal content (array of parts)
	if parts := content.OfArrayOfContentParts; len(parts) > 0 {
		// Create new content parts with tool prompt as first text part
		newParts := make([]openai.ChatCompletionContentPartUnionParam, 0, len(parts)+1)

		// Extract existing text content
		var existingText strings.Builder
		var nonTextParts []openai.ChatCompletionContentPartUnionParam

		for _, part := range parts {
			if textPart := part.OfText; textPart != nil {
				// Collect text content
				if existingText.Len() > 0 {
					existingText.WriteString(" ")
				}
				existingText.WriteString(textPart.Text)
			} else {
				// Preserve non-text parts (images, etc.)
				nonTextParts = append(nonTextParts, part)
			}
		}

		// Create combined text content
		combinedText := toolPrompt
		if existingText.Len() > 0 {
			combinedText += "\n\n" + existingText.String()
		}

		// Add combined text as first part
		newParts = append(newParts, openai.ChatCompletionContentPartUnionParam{
			OfText: &openai.ChatCompletionContentPartTextParam{
				Type: "text",
				Text: combinedText,
			},
		})

		// Add all non-text parts (images, etc.)
		newParts = append(newParts, nonTextParts...)

		// Create new user message with multimodal content
		return openai.UserMessage(newParts)
	}

	// Fallback: empty content, just add tool prompt
	return openai.UserMessage(toolPrompt)
}

// extractSystemContent extracts content from a system message
func extractSystemContent(msg openai.ChatCompletionMessageParamUnion) string {
	if msg.OfSystem != nil {
		// System messages have a ContentUnion with OfString or OfArrayOfContentParts
		content := msg.OfSystem.Content

		// Try to get string content (most common case)
		// When created with openai.SystemMessage("text"), OfString is populated
		if str := content.OfString.Or(""); str != "" {
			return str
		}

		// Handle array of content parts (multimodal content)
		if parts := content.OfArrayOfContentParts; len(parts) > 0 {
			var result strings.Builder
			for _, part := range parts {
				// System content parts are text-only
				result.WriteString(part.Text)
			}
			return result.String()
		}
	}
	return ""
}

// toolResult represents a parsed tool execution result
type toolResult struct {
	CallID  string
	Content string
}

// extractToolResults extracts ToolMessage types from messages and returns them along with cleaned messages.
// This implementation uses the OpenAI SDK's union type fields directly instead of JSON marshaling
// for efficient message type detection and content extraction.
func (a *Adapter) extractToolResults(messages []openai.ChatCompletionMessageParamUnion) ([]toolResult, []openai.ChatCompletionMessageParamUnion, error) {
	var results []toolResult
	var cleanMessages []openai.ChatCompletionMessageParamUnion

	for _, msg := range messages {
		// Check if this is a ToolMessage using the OpenAI SDK's union type field
		if msg.OfTool != nil {
			// Extract tool message data directly from the union type
			toolMsg := msg.OfTool

			// Extract content using union accessors for robustness
			var content string
			contentUnion := toolMsg.Content
			if str := contentUnion.OfString.Or(""); str != "" {
				content = str
			} else if parts := contentUnion.OfArrayOfContentParts; len(parts) > 0 {
				// Tool message parts are text-only in this SDK version; mirror system handling
				var sb strings.Builder
				for _, part := range parts {
					sb.WriteString(part.Text)
				}
				content = sb.String()
			}

			// Get tool call ID directly from the required field
			callID := toolMsg.ToolCallID

			results = append(results, toolResult{
				CallID:  callID,
				Content: content,
			})

			a.logger.Debug("Extracted tool result", "tool_call_id", callID, "content_length", len(content))
		} else {
			// Not a tool message, keep it in clean messages
			cleanMessages = append(cleanMessages, msg)
		}
	}

	return results, cleanMessages, nil
}

// buildToolResultsPrompt creates a natural language prompt section for tool results
func (a *Adapter) buildToolResultsPrompt(results []toolResult) string {
	if len(results) == 0 {
		return ""
	}

	var promptBuilder strings.Builder
	promptBuilder.WriteString("Previous tool calls requested by you returned the following results. They likely need formatting into a natural language response for the user:\n\n")

	for i, result := range results {
		if result.CallID != "" {
			promptBuilder.WriteString(fmt.Sprintf("Tool call %s result:\n", result.CallID))
		} else {
			promptBuilder.WriteString(fmt.Sprintf("Tool result %d:\n", i+1))
		}
		promptBuilder.WriteString(result.Content)
		promptBuilder.WriteString("\n\n")
	}

	return promptBuilder.String()
}

// emitMetric safely emits a metric event if a callback is configured.
// This method is called at key points during adapter operations to provide
// observability into performance and behavior.
//
// The method includes panic recovery to ensure that failures in user-provided
// metrics callbacks do not crash the adapter. Any panics are caught, logged,
// and the adapter continues normal operation. This is critical for production
// environments where metrics collection should never impact core functionality.
func (a *Adapter) emitMetric(data MetricEventData) {
	if a.metricsCallback == nil {
		return
	}

	// Protect against panics in user callbacks
	// Metrics are auxiliary functionality and should never crash the main operation
	defer func() {
		if r := recover(); r != nil {
			// Log the panic but don't propagate it
			// This ensures metrics collection failures don't impact core functionality
			a.logger.Error("Metrics callback panicked - metrics collection failed but operation continues",
				"panic", r,
				"event_type", data.EventType())
		}
	}()

	a.metricsCallback(data)
}

// GenerateToolCallID generates a unique ID for a tool call using UUIDv7.
// UUIDv7 provides the performance benefits of timestamp-based generation
// while maintaining full RFC 4122 compliance and battle-tested collision resistance.
//
// THREAD SAFETY: This method is safe for concurrent use by multiple goroutines.
// Each call generates a cryptographically unique ID with proper collision resistance.
//
// Benefits of UUIDv7 over UUIDv4:
// - Timestamp-based prefix enables natural sorting and better database performance
// - Better concurrent performance than gofrs/uuid implementation
// - Still cryptographically secure with proper collision resistance
// - Maintains standard UUID format that OpenAI expects
// - Includes timestamp extraction methods for debugging/analytics
//
// Performance: ~270ns/op vs ~210ns/op for UUIDv4, but provides ordering benefits.
func (a *Adapter) GenerateToolCallID() string {
	// UUIDv7 combines timestamp (48 bits) + version (4) + random (12) + variant (2) + random (62)
	// This gives us the speed benefits while maintaining RFC compliance
	id, err := uuid.NewV7()
	if err != nil {
		// This should never happen in normal operation - if it does, it indicates
		// a serious system issue (clock problems, entropy issues, etc.)
		a.logger.Error("UUIDv7 generation failed, falling back to UUIDv4",
			"error", err,
			"impact", "loss of timestamp-based ordering benefits")

		// Fallback to UUIDv4 to maintain functionality
		id = uuid.New()
	}
	return "call_" + id.String()
}
