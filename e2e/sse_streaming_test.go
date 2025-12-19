//go:build e2e

package e2e

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"testing"
	"time"

	tooladapter "github.com/juburr/openai-tool-adapter/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ============================================================================
// SSE Streaming E2E Tests
//
// These tests verify the raw SSE streaming functionality against a live
// vLLM instance running Gemma 3 4B. They test the full flow from HTTP
// request to SSE response with tool call detection.
// ============================================================================

func TestSSEStreaming_BasicPassthrough(t *testing.T) {
	config := LoadTestConfig()
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(config.Timeout)*time.Second)
	defer cancel()

	// Create a simple request without tools
	requestBody := map[string]interface{}{
		"model": config.Model,
		"messages": []map[string]interface{}{
			{"role": "user", "content": "Say hello in one word."},
		},
		"stream":     true,
		"max_tokens": 50,
	}

	resp, err := sendStreamingRequest(ctx, config, requestBody)
	require.NoError(t, err, "HTTP request should succeed")
	defer resp.Body.Close()

	require.Equal(t, http.StatusOK, resp.StatusCode, "Should get 200 OK")

	// Create SSE adapter for reading
	adapter := tooladapter.New(tooladapter.WithLogLevel(slog.LevelDebug))
	reader := tooladapter.NewHTTPSSEReader(resp)
	defer reader.Close()

	// Create a buffer writer to capture output
	var outputBuf bytes.Buffer
	writer := &testSSEWriter{Buffer: &outputBuf}

	sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)
	err = sseAdapter.Process(ctx)
	require.NoError(t, err, "SSE processing should succeed")

	// Verify we got output
	output := outputBuf.String()
	assert.NotEmpty(t, output, "Should have output")
	assert.Contains(t, output, "data:", "Should have SSE data lines")

	t.Logf("SSE passthrough output length: %d bytes", len(output))
}

func TestSSEStreaming_ToolCallDetection(t *testing.T) {
	config := LoadTestConfig()
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(config.Timeout)*time.Second)
	defer cancel()

	// Build the request with tools - use direct map construction for raw SSE testing
	request := map[string]interface{}{
		"model": config.Model,
		"messages": []map[string]interface{}{
			{"role": "user", "content": "What is the weather in Tokyo?"},
		},
		"tools": []map[string]interface{}{
			{
				"type": "function",
				"function": map[string]interface{}{
					"name":        "get_weather",
					"description": "Get current weather information for a specific location",
					"parameters": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"location": map[string]interface{}{
								"type":        "string",
								"description": "The location to get weather for",
							},
						},
						"required": []string{"location"},
					},
				},
			},
		},
		"stream": true,
	}

	resp, err := sendStreamingRequest(ctx, config, request)
	require.NoError(t, err, "HTTP request should succeed")
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		// Skip test if backend doesn't support native tool calling
		if strings.Contains(string(body), "tool choice requires") ||
			strings.Contains(string(body), "tool-call-parser") {
			t.Skip("Skipping: backend does not support native tool calling (requires --enable-auto-tool-choice)")
		}
		t.Fatalf("HTTP request failed: %d - %s", resp.StatusCode, string(body))
	}

	// Process with SSE adapter
	adapter := tooladapter.New(
		tooladapter.WithLogLevel(slog.LevelDebug),
		tooladapter.WithToolPolicy(tooladapter.ToolDrainAll),
	)

	reader := tooladapter.NewHTTPSSEReader(resp)
	defer reader.Close()

	// Use ProcessToResult to inspect before writing
	sseAdapter := adapter.NewSSEStreamAdapter(reader, nil)
	result, chunks, err := sseAdapter.ProcessToResult(ctx)
	require.NoError(t, err, "SSE processing should succeed")

	t.Logf("SSE Result - HasToolCalls: %v, Passthrough: %v, Content length: %d, Chunks: %d",
		result.HasToolCalls, result.Passthrough, len(result.Content), len(chunks))

	if result.HasToolCalls {
		t.Logf("Detected %d tool calls", len(result.ToolCalls))
		for i, call := range result.ToolCalls {
			t.Logf("  Tool %d: %s", i+1, call.Name)
			if len(call.Parameters) > 0 {
				t.Logf("    Parameters: %s", string(call.Parameters))
			}
		}

		// Verify the tool call is for weather
		assert.Equal(t, "get_weather", result.ToolCalls[0].Name, "Should call weather function")
	} else {
		t.Logf("Model did not output tool calls - Content: %s",
			result.Content[:min(200, len(result.Content))])
	}
}

func TestSSEStreaming_EarlyDetection(t *testing.T) {
	config := LoadTestConfig()
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(config.Timeout)*time.Second)
	defer cancel()

	// Request without tools - should use early detection passthrough
	requestBody := map[string]interface{}{
		"model": config.Model,
		"messages": []map[string]interface{}{
			{"role": "user", "content": "What is 2+2? Answer with just the number."},
		},
		"stream":     true,
		"max_tokens": 20,
	}

	resp, err := sendStreamingRequest(ctx, config, requestBody)
	require.NoError(t, err)
	defer resp.Body.Close()

	require.Equal(t, http.StatusOK, resp.StatusCode)

	adapter := tooladapter.New(tooladapter.WithLogLevel(slog.LevelDebug))
	reader := tooladapter.NewHTTPSSEReader(resp)
	defer reader.Close()

	var outputBuf bytes.Buffer
	writer := &testSSEWriter{Buffer: &outputBuf}

	sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)

	// Use early detection with 50-character lookahead
	err = sseAdapter.ProcessWithPassthrough(ctx, 50)
	require.NoError(t, err)

	output := outputBuf.String()
	assert.NotEmpty(t, output)
	assert.Contains(t, output, "data:")

	t.Logf("Early detection passthrough output: %d bytes", len(output))
}

func TestSSEStreaming_MultipleToolCalls(t *testing.T) {
	config := LoadTestConfig()
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(config.Timeout)*time.Second)
	defer cancel()

	// Request with multiple tools - use direct map construction for raw SSE testing
	request := map[string]interface{}{
		"model": config.Model,
		"messages": []map[string]interface{}{
			{"role": "user", "content": "What's the weather in Paris and calculate 15*7?"},
		},
		"tools": []map[string]interface{}{
			{
				"type": "function",
				"function": map[string]interface{}{
					"name":        "get_weather",
					"description": "Get current weather information for a specific location",
					"parameters": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"location": map[string]interface{}{
								"type":        "string",
								"description": "The location to get weather for",
							},
						},
						"required": []string{"location"},
					},
				},
			},
			{
				"type": "function",
				"function": map[string]interface{}{
					"name":        "calculate",
					"description": "Perform basic arithmetic calculations",
					"parameters": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"expression": map[string]interface{}{
								"type":        "string",
								"description": "Mathematical expression to evaluate",
							},
						},
						"required": []string{"expression"},
					},
				},
			},
		},
		"stream": true,
	}

	resp, err := sendStreamingRequest(ctx, config, request)
	require.NoError(t, err)
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		// Skip test if backend doesn't support native tool calling
		if strings.Contains(string(body), "tool choice requires") ||
			strings.Contains(string(body), "tool-call-parser") {
			t.Skip("Skipping: backend does not support native tool calling (requires --enable-auto-tool-choice)")
		}
		t.Fatalf("HTTP request failed: %d - %s", resp.StatusCode, string(body))
	}

	adapter := tooladapter.New(
		tooladapter.WithLogLevel(slog.LevelDebug),
		tooladapter.WithToolPolicy(tooladapter.ToolDrainAll),
		tooladapter.WithToolMaxCalls(10),
	)

	reader := tooladapter.NewHTTPSSEReader(resp)
	defer reader.Close()

	sseAdapter := adapter.NewSSEStreamAdapter(reader, nil)
	result, _, err := sseAdapter.ProcessToResult(ctx)
	require.NoError(t, err)

	t.Logf("Multiple tools result - HasToolCalls: %v, ToolCalls: %d",
		result.HasToolCalls, len(result.ToolCalls))

	if result.HasToolCalls {
		for i, call := range result.ToolCalls {
			t.Logf("  Tool %d: %s", i+1, call.Name)
		}
	} else {
		t.Logf("Model did not call multiple tools - Content preview: %s",
			result.Content[:min(100, len(result.Content))])
	}
}

func TestSSEStreaming_ToolPolicyStopOnFirst(t *testing.T) {
	config := LoadTestConfig()
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(config.Timeout)*time.Second)
	defer cancel()

	request := map[string]interface{}{
		"model": config.Model,
		"messages": []map[string]interface{}{
			{"role": "user", "content": "What time is it and what's the weather in London?"},
		},
		"tools": []map[string]interface{}{
			{
				"type": "function",
				"function": map[string]interface{}{
					"name":        "get_weather",
					"description": "Get current weather information for a specific location",
					"parameters": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"location": map[string]interface{}{
								"type":        "string",
								"description": "The location to get weather for",
							},
						},
						"required": []string{"location"},
					},
				},
			},
			{
				"type": "function",
				"function": map[string]interface{}{
					"name":        "get_time",
					"description": "Get current time and date information",
					"parameters": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"timezone": map[string]interface{}{
								"type":        "string",
								"description": "Timezone (e.g., 'UTC', 'EST', 'PST')",
							},
						},
						"required": []string{},
					},
				},
			},
		},
		"stream": true,
	}

	resp, err := sendStreamingRequest(ctx, config, request)
	require.NoError(t, err)
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		// Skip test if backend doesn't support native tool calling
		if strings.Contains(string(body), "tool choice requires") ||
			strings.Contains(string(body), "tool-call-parser") {
			t.Skip("Skipping: backend does not support native tool calling (requires --enable-auto-tool-choice)")
		}
		t.Fatalf("HTTP request failed: %d - %s", resp.StatusCode, string(body))
	}

	// Use ToolStopOnFirst policy
	adapter := tooladapter.New(
		tooladapter.WithLogLevel(slog.LevelDebug),
		tooladapter.WithToolPolicy(tooladapter.ToolStopOnFirst),
	)

	reader := tooladapter.NewHTTPSSEReader(resp)
	defer reader.Close()

	sseAdapter := adapter.NewSSEStreamAdapter(reader, nil)
	result, _, err := sseAdapter.ProcessToResult(ctx)
	require.NoError(t, err)

	t.Logf("ToolStopOnFirst result - HasToolCalls: %v", result.HasToolCalls)

	if result.HasToolCalls {
		// With ToolStopOnFirst, we should get exactly 1 tool call
		assert.Len(t, result.ToolCalls, 1, "ToolStopOnFirst should return exactly 1 tool")
		t.Logf("First tool: %s", result.ToolCalls[0].Name)
	}
}

func TestSSEStreaming_WriteOutput(t *testing.T) {
	config := LoadTestConfig()
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(config.Timeout)*time.Second)
	defer cancel()

	request := map[string]interface{}{
		"model": config.Model,
		"messages": []map[string]interface{}{
			{"role": "user", "content": "What is the weather in Seattle?"},
		},
		"tools": []map[string]interface{}{
			{
				"type": "function",
				"function": map[string]interface{}{
					"name":        "get_weather",
					"description": "Get current weather information for a specific location",
					"parameters": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"location": map[string]interface{}{
								"type":        "string",
								"description": "The location to get weather for",
							},
						},
						"required": []string{"location"},
					},
				},
			},
		},
		"stream": true,
	}

	resp, err := sendStreamingRequest(ctx, config, request)
	require.NoError(t, err)
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		// Skip test if backend doesn't support native tool calling
		if strings.Contains(string(body), "tool choice requires") ||
			strings.Contains(string(body), "tool-call-parser") {
			t.Skip("Skipping: backend does not support native tool calling (requires --enable-auto-tool-choice)")
		}
		t.Fatalf("HTTP request failed: %d - %s", resp.StatusCode, string(body))
	}

	adapter := tooladapter.New(
		tooladapter.WithLogLevel(slog.LevelDebug),
		tooladapter.WithToolPolicy(tooladapter.ToolDrainAll),
	)

	reader := tooladapter.NewHTTPSSEReader(resp)
	defer reader.Close()

	var outputBuf bytes.Buffer
	writer := &testSSEWriter{Buffer: &outputBuf}

	sseAdapter := adapter.NewSSEStreamAdapter(reader, writer)
	err = sseAdapter.Process(ctx)
	require.NoError(t, err)

	output := outputBuf.String()
	t.Logf("Written output length: %d bytes", len(output))

	// Verify SSE format
	assert.Contains(t, output, "data:", "Should have SSE data lines")

	// Check for [DONE] marker
	if strings.Contains(output, "[DONE]") {
		t.Log("Output contains [DONE] marker")
	}

	// Parse output to check for tool_calls
	if strings.Contains(output, "tool_calls") {
		t.Log("Output contains tool_calls")
		assert.Contains(t, output, "finish_reason")
	}
}

func TestSSEStreaming_LongConversation(t *testing.T) {
	config := LoadTestConfig()
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(config.Timeout)*time.Second)
	defer cancel()

	// Multi-turn conversation
	request := map[string]interface{}{
		"model": config.Model,
		"messages": []map[string]interface{}{
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": "Hi there!"},
			{"role": "assistant", "content": "Hello! How can I help you today?"},
			{"role": "user", "content": "Tell me a very short joke."},
		},
		"stream":     true,
		"max_tokens": 100,
	}

	resp, err := sendStreamingRequest(ctx, config, request)
	require.NoError(t, err)
	defer resp.Body.Close()

	require.Equal(t, http.StatusOK, resp.StatusCode)

	adapter := tooladapter.New(tooladapter.WithLogLevel(slog.LevelDebug))
	reader := tooladapter.NewHTTPSSEReader(resp)
	defer reader.Close()

	sseAdapter := adapter.NewSSEStreamAdapter(reader, nil)
	result, chunks, err := sseAdapter.ProcessToResult(ctx)
	require.NoError(t, err)

	t.Logf("Long conversation - Chunks: %d, Content length: %d",
		len(chunks), len(result.Content))
	t.Logf("Content: %s", result.Content[:min(200, len(result.Content))])
}

// ============================================================================
// Helper Functions
// ============================================================================

func sendStreamingRequest(ctx context.Context, config TestConfig, body interface{}) (*http.Response, error) {
	reqBody, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		config.BaseURL+"/chat/completions", bytes.NewReader(reqBody))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	if config.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+config.APIKey)
	}

	client := &http.Client{}
	return client.Do(req)
}

// testSSEWriter implements SSEStreamWriter for testing
type testSSEWriter struct {
	Buffer *bytes.Buffer
}

func (w *testSSEWriter) WriteChunk(chunk *tooladapter.SSEChunk) error {
	data, err := json.Marshal(chunk)
	if err != nil {
		return err
	}
	w.Buffer.WriteString("data: ")
	w.Buffer.Write(data)
	w.Buffer.WriteString("\n\n")
	return nil
}

func (w *testSSEWriter) WriteRaw(data []byte) error {
	_, err := w.Buffer.Write(data)
	return err
}

func (w *testSSEWriter) WriteDone() error {
	w.Buffer.WriteString("data: [DONE]\n\n")
	return nil
}

func (w *testSSEWriter) Flush() {}
