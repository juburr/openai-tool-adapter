package tooladapter

import (
	"errors"
	"fmt"
	"io"
	"log/slog"
	"strings"
	"time"
)

// ToolPolicy defines how tool calls are handled during response processing.
// Different policies provide various trade-offs between latency, completeness, and behavior.
type ToolPolicy int

const (
	// ToolStopOnFirst stops processing on the first valid tool call (safest + lowest latency).
	// Emits only the first tool call detected and ignores subsequent content/calls.
	ToolStopOnFirst ToolPolicy = iota

	// ToolCollectThenStop halts content emission but collects tools until array closes,
	// timeout, or limits are reached (short batching window or structure-only).
	ToolCollectThenStop

	// ToolDrainAll reads to end of stream and collects all tool calls before emitting
	// (read to EOS, collect everything).
	ToolDrainAll

	// ToolAllowMixed streams both text content and tools together without suppression
	// (stream text and tools together).
	ToolAllowMixed
)

// String returns a human-readable string representation of the ToolPolicy.
func (tp ToolPolicy) String() string {
	switch tp {
	case ToolStopOnFirst:
		return "ToolStopOnFirst"
	case ToolCollectThenStop:
		return "ToolCollectThenStop"
	case ToolDrainAll:
		return "ToolDrainAll"
	case ToolAllowMixed:
		return "ToolAllowMixed"
	default:
		return fmt.Sprintf("ToolPolicy(%d)", int(tp))
	}
}

const (
	// DefaultPromptTemplate provides a robust, concise template that works across LLM families.
	// It emphasizes immediate, JSON-only tool calls when appropriate, and natural language otherwise.
	DefaultPromptTemplate = `System/tooling instructions:

You have access to the following functions. When a function call is needed, respond immediately (starting at the first token) with a single JSON array of tool calls, and include no natural-language text before or after the JSON.

Available functions:
%s

Formatting requirements:
- Output must be valid JSON only (no code fences).
- Structure: [{"name": "function_name", "parameters": {…}}] (use null if there are no parameters).
- If multiple calls are required, include them all in the single JSON array.

Decision policy:
- Use tools when they are required to answer correctly or efficiently; otherwise reply in natural language without calling any tools.`
)

// Option is a function that configures the Adapter.
// This functional options pattern provides several key benefits:
// 1. Backwards compatibility - new options don't break existing code
// 2. Optional parameters - users only specify what they want to change
// 3. Self-documenting - option names clearly indicate their purpose
// 4. Validation - each option can validate its input independently
type Option func(*Adapter)

// WithCustomPromptTemplate overrides the default system prompt template.
// This allows customization for specific LLM families or use cases.
//
// The template must contain exactly one %s placeholder where tool definitions
// will be inserted. The function validates this requirement.
func WithCustomPromptTemplate(template string) Option {
	return func(a *Adapter) {
		if template == "" {
			a.logger.Warn("Empty prompt template provided, using default")
			return
		}

		// Validate that the template has exactly one %s placeholder
		// This prevents runtime errors when formatting the prompt
		if err := validatePromptTemplate(template); err != nil {
			a.logger.Warn("Invalid prompt template, using default", "error", err)
			return
		}

		a.promptTemplate = template
		a.logger.Debug("Using custom prompt template", "template_length", len(template))
	}
}

// WithLogger sets a custom slog.Logger for the adapter.
// This enables structured logging for operational observability in production.
//
// Logging strategy:
// - INFO: Operational events (tool transformations, function call detection)
// - DEBUG: Detailed information (performance metrics)
// - WARN: Unexpected but recoverable situations
// - ERROR: Actual errors that affect functionality
//
// If no logger is provided, a no-op logger is used to avoid breaking existing code.
func WithLogger(logger *slog.Logger) Option {
	return func(a *Adapter) {
		if logger == nil {
			// Create a no-op logger when nil is provided
			a.logger = slog.New(slog.NewTextHandler(io.Discard, &slog.HandlerOptions{
				Level: slog.LevelError + 1, // Effectively disable all logging
			}))
			return
		}
		a.logger = logger
	}
}

// WithLogLevel sets the logging level for the default logger.
// This is a convenience option when you want to use slog.Default() but control the level.
// For production use, consider using WithLogger with a properly configured logger.
func WithLogLevel(level slog.Level) Option {
	return func(a *Adapter) {
		// Create a new logger with the specified level using discard for testing
		// In real usage, users should use WithLogger with a proper writer
		handler := slog.NewTextHandler(io.Discard, &slog.HandlerOptions{
			Level: level,
		})
		a.logger = slog.New(handler)
	}
}

// validatePromptTemplate ensures the template can be used safely with fmt.Sprintf.
// This prevents runtime panics from malformed templates.
func validatePromptTemplate(template string) error {
	// Count %s placeholders in the template
	// We need exactly one for the tool definitions
	placeholders := strings.Count(template, "%s")

	if placeholders == 0 {
		return errors.New("template validation failed: template must contain exactly one %%s placeholder for tool definitions")
	}
	if placeholders > 1 {
		return fmt.Errorf("template validation failed: template contains %d %%s placeholders but exactly one is required", placeholders)
	}

	// Test the template with a dummy string to catch other formatting issues
	testResult := fmt.Sprintf(template, "test")
	if testResult == template {
		return errors.New("template validation failed: %%s placeholder was not processed during formatting test")
	}

	return nil
}

// ApplyOptions applies a slice of options to an adapter, handling errors gracefully.
// This helper function makes it easier to apply multiple options while handling
// validation errors properly.
func ApplyOptions(adapter *Adapter, opts []Option) {
	for _, opt := range opts {
		opt(adapter)
	}
}

// DefaultOptions returns a set of sensible default options.
// This is useful for applications that want to start with good defaults
// and then customize specific aspects.
func DefaultOptions() []Option {
	return []Option{
		WithCustomPromptTemplate(DefaultPromptTemplate),
		WithLogger(slog.Default()), // Use the global default logger
	}
}

// NOTE: The preset helpers ProductionOptions, DevelopmentOptions, and QuietOptions
// have been removed. Prefer explicit configuration via WithLogger or WithLogLevel.

// WithMetricsCallback sets a callback function that receives metric events.
// This enables integration with monitoring systems like Prometheus, DataDog, or custom metrics collection.
//
// The callback receives typed event data that can be safely type-switched to access specific metrics.
// All event data includes performance metrics for operational monitoring.
//
// Example usage:
//
//	adapter := tooladapter.New(
//	    tooladapter.WithMetricsCallback(func(data tooladapter.MetricEventData) {
//	        switch eventData := data.(type) {
//	        case tooladapter.ToolTransformationData:
//	            // Handle tool transformation metrics
//	            myMetrics.ToolTransformations.Inc()
//	        case tooladapter.FunctionCallDetectionData:
//	            // Handle function call detection metrics
//	            myMetrics.FunctionCalls.Add(float64(eventData.FunctionCount))
//	            myMetrics.ProcessingDuration.Observe(float64(eventData.Performance.ProcessingDurationMs))
//	        }
//	    }),
//	)
//
// The callback is called synchronously during adapter operations, so it should be fast
// to avoid impacting request processing performance. For expensive operations like
// database writes, consider using a background goroutine or message queue.
//
// IMPORTANT: The adapter includes panic recovery for metrics callbacks. If your callback
// panics, the panic will be caught, logged, and the adapter will continue normal operation.
// This ensures that metrics collection failures never impact core functionality. However,
// you should still implement proper error handling in your callbacks for best practices.
func WithMetricsCallback(callback func(MetricEventData)) Option {
	return func(a *Adapter) {
		a.metricsCallback = callback
	}
}

// WithToolPolicy sets the tool processing policy for the adapter.
// This controls how tool calls are detected, collected, and emitted.
//
// Available policies:
//   - ToolStopOnFirst: Stop on first tool call (lowest latency, safest)
//   - ToolCollectThenStop: Collect tools until array closes or limits reached
//   - ToolDrainAll: Read entire response and collect all tools
//   - ToolAllowMixed: Allow both text content and tools to be emitted
//
// Default: ToolStopOnFirst
func WithToolPolicy(policy ToolPolicy) Option {
	return func(a *Adapter) {
		a.toolPolicy = policy
	}
}

// WithToolCollectWindow sets the maximum time to wait for additional tools
// when using ToolCollectThenStop policy in streaming mode.
//
// If set to 0, uses structure-only batching (no timer).
// This option is ignored for non-streaming mode and other policies.
// An upper bound is unnecessary, as overly high durations are equivalent
// in behavior to both non-streaming mode and the ToolDrainAll policy.
//
// Default: 200ms
func WithToolCollectWindow(duration time.Duration) Option {
	return func(a *Adapter) {
		if duration < 0 {
			a.logger.Warn("Negative duration not allowed for tool collection window",
				"supplied_duration", duration,
				"updated_duration", 0,
				"implication", "No time limit will be applied to the tool call collection window",
				"recommendation", "Supply a positive duration to WithToolCollectWindow()")
			duration = 0
		}

		a.toolCollectWindow = duration
	}
}

// WithToolMaxCalls sets the maximum number of tool calls to collect
// across both streaming and non-streaming modes.
//
// This provides a safety cap to prevent excessive tool call processing.
// Set to 0 for no limit (not recommended for production).
//
// Default: 8
func WithToolMaxCalls(maxCalls int) Option {
	return func(a *Adapter) {
		if maxCalls < 0 {
			a.logger.Warn("Negative tool count not allowed for ToolMaxCalls",
				"supplied_maxCalls", maxCalls,
				"updated_maxCalls", 0,
				"implication", "No limit will be applied to the number of tool calls",
				"recommendation", "Supply a positive number to WithToolMaxCalls()")
			maxCalls = 0
		}
		a.toolMaxCalls = maxCalls
	}
}

// WithToolCollectMaxBytes sets the maximum number of bytes to collect
// during JSON tool call processing as a safety cap.
//
// This prevents memory exhaustion from malformed or malicious responses.
// Set to 0 for no limit (not recommended for production).
//
// Default: 65536 (64KB) - provides DoS protection while allowing legitimate use cases
func WithToolCollectMaxBytes(maxBytes int) Option {
	return func(a *Adapter) {
		if maxBytes < 0 {
			a.logger.Warn("Negative byte count not allowed for ToolCollectMaxBytes",
				"supplied_maxBytes", maxBytes,
				"updated_maxBytes", 0,
				"implication", "No limit will be applied to the number of bytes collected",
				"recommendation", "Supply a positive number to WithToolCollectMaxBytes()")
			maxBytes = 0
		}
		a.toolCollectMaxBytes = maxBytes
	}
}

// WithCancelUpstreamOnStop controls whether the upstream stream is cancelled
// when stopping tool collection in streaming mode.
//
// This option only applies to streaming mode with ToolStopOnFirst or
// ToolCollectThenStop policies. When true, the adapter will cancel the
// upstream context to stop further content generation.
//
// Default: true
func WithCancelUpstreamOnStop(cancel bool) Option {
	return func(a *Adapter) {
		if !cancel {
			a.logger.Info("Upstream context cancellation has been disabled",
				"implication", "Upstream model will continue generating content after tool collection stops",
				"recommendation", "Enable context cancellation using WithCancelUpstreamOnStop() to reduce costs and unnecessary processing")
		}
		a.cancelUpstreamOnStop = cancel
	}
}

// WithStreamingToolBufferSize sets the maximum amount of content that can be buffered
// while parsing streaming responses for tool calls. This prevents memory exhaustion
// during streaming by limiting how much text is held in memory while searching
// for complete JSON tool call structures.
//
// When this limit is exceeded during streaming, the buffered content is processed
// as regular text rather than continuing to search for tool calls.
//
// Use cases:
//   - Increase for models that generate very large tool calls
//   - Decrease for memory-constrained environments
//   - Set very low for testing buffer overflow behavior
//
// Default: 10MB (10 * 1024 * 1024 bytes)
func WithStreamingToolBufferSize(limitBytes int) Option {
	return func(a *Adapter) {
		if limitBytes > 0 {
			a.streamBufferLimit = limitBytes
		}
	}
}

// WithStreamingEarlyDetection enables early tool call detection in streaming responses
// by looking ahead within the first N characters of content for tool call patterns.
// This improves buffering heuristics when models emit explanatory text before JSON.
//
// The adapter will search for tool call patterns like {"name": or [{"name": within
// the specified character limit. This helps catch tool calls that start after some
// preface text, improving content suppression for ToolStopOnFirst/ToolCollectThenStop.
//
// Recommended values:
//   - 80-100 characters: Good balance of recall vs false positives
//   - 120 characters: More generous, catches longer prefaces
//   - 0 (default): Disabled, uses only immediate JSON detection
//
// Use cases:
//   - Enable when models frequently add explanatory text before tool calls
//   - Keep disabled for maximum performance and minimal false positives
//   - Use with ToolStopOnFirst or ToolCollectThenStop for content suppression
//
// Note: ToolAllowMixed policy streams content regardless, so this mainly benefits
// policies that suppress content when tool calls are detected.
func WithStreamingEarlyDetection(lookAheadChars int) Option {
	return func(a *Adapter) {
		if lookAheadChars >= 0 {
			a.streamLookAheadLimit = lookAheadChars
		}
	}
}

// WithPromptBufferReuseLimit sets the maximum size of prompt generation buffers
// that will be returned to the buffer pool for reuse. Larger buffers are discarded
// to prevent the buffer pool from growing unbounded when processing very large
// tool definitions.
//
// This affects the internal buffer pool used for building tool prompts during
// request transformation. Buffers that exceed this threshold are garbage collected
// rather than being pooled for reuse.
//
// Use cases:
//   - Increase for applications with consistently large tool schemas
//   - Decrease for memory-sensitive environments
//   - Set very low for testing pool behavior
//
// Default: 64KB (64 * 1024 bytes)
func WithPromptBufferReuseLimit(thresholdBytes int) Option {
	return func(a *Adapter) {
		if thresholdBytes > 0 {
			a.bufferPoolThreshold = thresholdBytes
		}
	}
}

// WithNoSystemInstructionRole sets which role to use when no system message is present.
// Default is false to support models that ignore or lack a system role (e.g., Gemma 3),
// but you should set this to true if your model supports or requires a system message.
func WithSystemMessageSupport(supported bool) Option {
	return func(a *Adapter) {
		a.systemMessagesSupported = supported
	}
}
