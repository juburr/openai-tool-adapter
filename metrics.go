package tooladapter

import "time"

// MetricEvent represents the type of metric event being emitted.
// Each event corresponds to a significant operation within the tool adapter.
type MetricEvent string

const (
	// MetricEventToolTransformation fires when tools are injected into system prompt.
	// This event indicates that the adapter has successfully converted OpenAI-style
	// tool definitions into a prompt-based format for models without native tool support.
	MetricEventToolTransformation MetricEvent = "tool_transformation"

	// MetricEventFunctionCallDetection fires when function calls are parsed from responses.
	// This event indicates that the adapter has successfully extracted and converted
	// function calls from LLM response text back into OpenAI-compatible tool calls.
	MetricEventFunctionCallDetection MetricEvent = "function_call_detection"
)

// MetricEventData is implemented by all metric event data structures.
// This interface enables type-safe handling of different event types while
// maintaining a clean callback signature.
type MetricEventData interface {
	EventType() MetricEvent
}

// PerformanceMetrics contains timing and resource usage information.
// These metrics are included with most events to provide operational visibility
// into the adapter's performance characteristics.
//
// Thread Safety: PerformanceMetrics instances are immutable after creation.
// The SubOperations map is created fresh for each metric event and is never
// modified after the metric is emitted. This makes it safe for concurrent
// access by metric callbacks, even if they spawn goroutines.
type PerformanceMetrics struct {
	// ProcessingDuration is the total time spent processing the operation
	// Uses time.Duration for nanosecond precision - callers can convert as needed
	ProcessingDuration time.Duration `json:"processing_duration"`

	// MemoryAllocatedBytes tracks memory allocation during the operation (when available)
	MemoryAllocatedBytes int64 `json:"memory_allocated_bytes,omitempty"`

	// SubOperations provides timing breakdowns for complex operations
	// Keys might include: "prompt_generation", "json_parsing", "validation", etc.
	// Uses time.Duration for precise timing measurements
	// Note: This map is created fresh for each metric event and is never modified
	// after creation, making it safe for concurrent read access.
	SubOperations map[string]time.Duration `json:"sub_operations,omitempty"`
}

// ToolTransformationData contains metrics about tool-to-prompt transformations.
// This event is emitted when the adapter converts OpenAI tool definitions
// into system prompts for models without native tool support.
type ToolTransformationData struct {
	// ToolCount is the number of tools being transformed
	ToolCount int `json:"tool_count"`

	// ToolNames lists the names of all tools being transformed
	ToolNames []string `json:"tool_names"`

	// PromptLength is the length of the generated system prompt in characters
	PromptLength int `json:"prompt_length"`

	// Performance contains timing and resource metrics for this transformation
	Performance PerformanceMetrics `json:"performance"`
}

func (d ToolTransformationData) EventType() MetricEvent {
	return MetricEventToolTransformation
}

// FunctionCallDetectionData contains metrics about function call parsing.
// This event is emitted when the adapter extracts function calls from
// LLM responses and converts them back to OpenAI-compatible format.
type FunctionCallDetectionData struct {
	// FunctionCount is the number of function calls detected and parsed
	FunctionCount int `json:"function_count"`

	// FunctionNames lists the names of all functions called
	FunctionNames []string `json:"function_names"`

	// ContentLength is the length of the original response content in characters
	ContentLength int `json:"content_length"`

	// JSONCandidates is the number of potential JSON blocks found in the content
	JSONCandidates int `json:"json_candidates"`

	// Streaming indicates whether this detection occurred in streaming mode
	Streaming bool `json:"streaming"`

	// Performance contains timing and resource metrics for this detection
	Performance PerformanceMetrics `json:"performance"`
}

func (d FunctionCallDetectionData) EventType() MetricEvent {
	return MetricEventFunctionCallDetection
}
