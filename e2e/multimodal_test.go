//go:build e2e

package e2e

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/openai/openai-go/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// loadNvidiaLogoBase64 reads the nvidia.jpg file and returns the base64 encoded string
func loadNvidiaLogoBase64(t *testing.T) string {
	imageData, err := os.ReadFile("nvidia.jpg")
	require.NoError(t, err, "Should be able to read nvidia.jpg file")

	base64String := base64.StdEncoding.EncodeToString(imageData)
	t.Logf("Successfully encoded nvidia.jpg: %d bytes -> %d base64 chars", len(imageData), len(base64String))

	return base64String
}

// Helper functions to create multimodal content parts
func createTextPart(text string) openai.ChatCompletionContentPartUnionParam {
	return openai.ChatCompletionContentPartUnionParam{
		OfText: &openai.ChatCompletionContentPartTextParam{
			Type: "text",
			Text: text,
		},
	}
}

func createImagePart(base64Data string) openai.ChatCompletionContentPartUnionParam {
	return openai.ChatCompletionContentPartUnionParam{
		OfImageURL: &openai.ChatCompletionContentPartImageParam{
			Type: "image_url",
			ImageURL: openai.ChatCompletionContentPartImageImageURLParam{
				URL: "data:image/jpeg;base64," + base64Data,
			},
		},
	}
}

func createImagePartFromFile(t *testing.T) openai.ChatCompletionContentPartUnionParam {
	return createImagePart(loadNvidiaLogoBase64(t))
}

// testMultimodalWithUnrelatedTool tests multimodal structure with unrelated tools
func testMultimodalWithUnrelatedTool(t *testing.T, client *TestClient, ctx context.Context) {
	// Test that adapter handles multimodal message structure correctly, even with a text-only model
	// Note: Gemma 3 4B is text-only, but we test that our adapter preserves the structure
	weatherTool := CreateWeatherTool()

	// Create a message with image structure asking about something unrelated to weather
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage([]openai.ChatCompletionContentPartUnionParam{
			createTextPart("I have an image attached, but please tell me about coding best practices instead."),
			createImagePartFromFile(t),
		}),
	}

	// Create request with weather tool
	req := openai.ChatCompletionNewParams{
		Model:    client.config.Model,
		Messages: messages,
		Tools:    []openai.ChatCompletionToolUnionParam{weatherTool},
	}

	// Transform the request - this should succeed without errors
	transformedReq, err := client.adapter.TransformCompletionsRequest(req)
	require.NoError(t, err, "Transform request should not fail")

	// Send the request
	response, err := client.client.Chat.Completions.New(ctx, transformedReq)
	require.NoError(t, err, "Request should not fail")

	// Transform the response
	finalResponse, err := client.adapter.TransformCompletionsResponse(*response)
	require.NoError(t, err, "Transform response should not fail")

	require.NotNil(t, finalResponse, "Response should not be nil")
	require.Len(t, finalResponse.Choices, 1, "Should have exactly one choice")

	choice := finalResponse.Choices[0]

	// Since we asked about coding practices (not weather), the model should respond with content
	// Note: Text-only model may still call tools if it finds them relevant
	if choice.FinishReason == "tool_calls" {
		t.Logf("Model chose to call tools: %v", choice.Message.ToolCalls)
	} else {
		assert.NotEmpty(t, choice.Message.Content, "Should have content response")
		content := strings.ToLower(choice.Message.Content)
		// Should respond about coding, not weather
		hasCoding := strings.Contains(content, "code") || strings.Contains(content, "programming") ||
			strings.Contains(content, "software") || strings.Contains(content, "development")
		t.Logf("Response content: %s", choice.Message.Content)
		if hasCoding {
			t.Logf("✅ Model correctly responded about coding practices")
		}
	}
}

// testToolCallWithImage tests that tool calling works with images present
func testToolCallWithImage(t *testing.T, client *TestClient, ctx context.Context) {
	// Create a weather tool that should be called
	weatherTool := CreateWeatherTool()

	// Create a message with an image but asking about weather (not the image)
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage([]openai.ChatCompletionContentPartUnionParam{
			createTextPart("What's the weather like in Seattle? Please use the weather tool to check."),
			createImagePartFromFile(t),
		}),
	}

	// Create request with tools
	req := openai.ChatCompletionNewParams{
		Model:    client.config.Model,
		Messages: messages,
		Tools:    []openai.ChatCompletionToolUnionParam{weatherTool},
	}

	// Transform the request
	transformedReq, err := client.adapter.TransformCompletionsRequest(req)
	require.NoError(t, err, "Transform request should not fail")

	// Send the request
	response, err := client.client.Chat.Completions.New(ctx, transformedReq)
	require.NoError(t, err, "Request should not fail")

	// Transform the response
	finalResponse, err := client.adapter.TransformCompletionsResponse(*response)
	require.NoError(t, err, "Transform response should not fail")

	require.NotNil(t, finalResponse, "Response should not be nil")
	require.Len(t, finalResponse.Choices, 1, "Should have exactly one choice")

	choice := finalResponse.Choices[0]

	// The model should call the weather tool, not analyze the image
	assert.Equal(t, "tool_calls", choice.FinishReason, "Should call the weather tool")
	assert.NotEmpty(t, choice.Message.ToolCalls, "Should have tool calls")

	// Verify it's calling the weather tool
	toolCall := choice.Message.ToolCalls[0]
	assert.Equal(t, "get_weather", toolCall.Function.Name, "Should call get_weather function")

	// Verify the location parameter
	var args map[string]interface{}
	err = json.Unmarshal([]byte(toolCall.Function.Arguments), &args)
	require.NoError(t, err, "Arguments should be valid JSON")

	location, exists := args["location"]
	require.True(t, exists, "Arguments should contain location")
	assert.Contains(t, strings.ToLower(location.(string)), "seattle",
		"Location should be Seattle")

	t.Logf("Tool call: %s with args: %s", toolCall.Function.Name, toolCall.Function.Arguments)
}

// testMultipleToolsWithMultimodal tests multiple tools with multimodal structure
func testMultipleToolsWithMultimodal(t *testing.T, client *TestClient, ctx context.Context) {
	// Test that adapter correctly handles multiple tools with multimodal message structure
	weatherTool := CreateWeatherTool()
	calcTool := CreateCalculatorTool()
	timeTool := CreateTimeTool()

	// Ask something that doesn't require any of the available tools
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage([]openai.ChatCompletionContentPartUnionParam{
			createTextPart("I have an image attached, but please tell me a fun fact about programming languages instead."),
			createImagePartFromFile(t),
		}),
	}

	// Create request with multiple unrelated tools
	req := openai.ChatCompletionNewParams{
		Model:    client.config.Model,
		Messages: messages,
		Tools: []openai.ChatCompletionToolUnionParam{
			weatherTool,
			calcTool,
			timeTool,
		},
	}

	// Transform the request
	transformedReq, err := client.adapter.TransformCompletionsRequest(req)
	require.NoError(t, err, "Transform request should not fail")

	// Send the request
	response, err := client.client.Chat.Completions.New(ctx, transformedReq)
	require.NoError(t, err, "Request should not fail")

	// Transform the response
	finalResponse, err := client.adapter.TransformCompletionsResponse(*response)
	require.NoError(t, err, "Transform response should not fail")

	require.NotNil(t, finalResponse, "Response should not be nil")
	require.Len(t, finalResponse.Choices, 1, "Should have exactly one choice")

	choice := finalResponse.Choices[0]

	// Test both possible behaviors - text response or tool call
	if choice.FinishReason == "tool_calls" {
		assert.NotEmpty(t, choice.Message.ToolCalls, "Should have tool calls if finish reason is tool_calls")
		t.Logf("Model chose to call tools instead: %v", choice.Message.ToolCalls)
	} else {
		assert.NotEmpty(t, choice.Message.Content, "Should have content response")
		content := strings.ToLower(choice.Message.Content)
		hasProgramming := strings.Contains(content, "programming") || strings.Contains(content, "language") ||
			strings.Contains(content, "code") || strings.Contains(content, "software")
		t.Logf("Response content: %s", choice.Message.Content)
		if hasProgramming {
			t.Logf("✅ Model correctly responded about programming")
		}
	}

	// Most importantly: the request didn't fail due to multimodal structure
	t.Logf("✅ Adapter successfully handled multimodal structure with multiple tools")
}

// testComplexMultimodalRequest tests complex multimodal requests with tool calls
func testComplexMultimodalRequest(t *testing.T, client *TestClient, ctx context.Context) {
	// Create calculator tool for the math question
	calcTool := CreateCalculatorTool()

	// Create a complex message with both an image and a tool-requiring task
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage([]openai.ChatCompletionContentPartUnionParam{
			createTextPart("I have two questions: 1) What is 42 multiplied by 17? Please use the calculator tool. 2) Also, can you see the logo in this image?"),
			createImagePartFromFile(t),
		}),
	}

	// Create request with calculator tool
	req := openai.ChatCompletionNewParams{
		Model:    client.config.Model,
		Messages: messages,
		Tools:    []openai.ChatCompletionToolUnionParam{calcTool},
	}

	// Transform the request
	transformedReq, err := client.adapter.TransformCompletionsRequest(req)
	require.NoError(t, err, "Transform request should not fail")

	// Send the request
	response, err := client.client.Chat.Completions.New(ctx, transformedReq)
	require.NoError(t, err, "Request should not fail")

	// Transform the response
	finalResponse, err := client.adapter.TransformCompletionsResponse(*response)
	require.NoError(t, err, "Transform response should not fail")

	require.NotNil(t, finalResponse, "Response should not be nil")
	require.Len(t, finalResponse.Choices, 1, "Should have exactly one choice")

	choice := finalResponse.Choices[0]

	// The model should prioritize the tool call for calculation
	// Note: Behavior may vary - model might call tool first or answer both
	if choice.FinishReason == "tool_calls" {
		assert.NotEmpty(t, choice.Message.ToolCalls, "Should have tool calls")

		// Verify it's calling the calculator
		toolCall := choice.Message.ToolCalls[0]
		assert.Equal(t, "calculate", toolCall.Function.Name, "Should call calculate function")

		// Check that it's calculating 42 * 17
		var args map[string]interface{}
		err = json.Unmarshal([]byte(toolCall.Function.Arguments), &args)
		require.NoError(t, err, "Arguments should be valid JSON")

		expression, exists := args["expression"]
		require.True(t, exists, "Arguments should contain expression")
		exprStr := strings.ToLower(expression.(string))
		assert.True(t,
			(strings.Contains(exprStr, "42") && strings.Contains(exprStr, "17")) ||
				strings.Contains(exprStr, "714"), // Model might compute it
			"Expression should involve 42 and 17. Got: %s", expression)

		t.Logf("Tool call for calculation: %s", toolCall.Function.Arguments)
	} else {
		// Model might answer both questions directly
		assert.NotEmpty(t, choice.Message.Content, "Should have content")
		content := strings.ToLower(choice.Message.Content)

		// Should mention either the calculation or NVIDIA or both
		hasCalculation := strings.Contains(content, "714") ||
			(strings.Contains(content, "42") && strings.Contains(content, "17"))
		hasNvidia := strings.Contains(content, "nvidia")

		assert.True(t, hasCalculation || hasNvidia,
			"Response should address at least one question. Got: %s", choice.Message.Content)

		t.Logf("Direct response to complex request: %s", choice.Message.Content)
	}
}

// TestMultimodalWithTools tests that the adapter correctly handles requests with both images and tools
func TestMultimodalWithTools(t *testing.T) {
	client := NewTestClient()
	ctx, cancel := client.CreateTimeoutContext()
	defer cancel()

	t.Run("MultimodalStructureWithUnrelatedTool", func(t *testing.T) {
		testMultimodalWithUnrelatedTool(t, client, ctx)
	})

	t.Run("ToolCallWithImage", func(t *testing.T) {
		testToolCallWithImage(t, client, ctx)
	})

	t.Run("MultipleToolsWithMultimodalStructure", func(t *testing.T) {
		testMultipleToolsWithMultimodal(t, client, ctx)
	})

	t.Run("ComplexMultimodalRequest", func(t *testing.T) {
		testComplexMultimodalRequest(t, client, ctx)
	})
}

// TestMultimodalEdgeCases tests edge cases with multimodal inputs
func TestMultimodalEdgeCases(t *testing.T) {
	client := NewTestClient()
	ctx, cancel := client.CreateTimeoutContext()
	defer cancel()

	t.Run("MultimodalStructureNoTools", func(t *testing.T) {
		// Test multimodal message structure without tools (should be processed by vLLM)
		// Note: Since Gemma 3 4B is text-only, we expect it to ignore image and respond to text
		messages := []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage([]openai.ChatCompletionContentPartUnionParam{
				createTextPart("I have an image attached, but please just say 'Hello, this is a test response.'"),
				createImagePartFromFile(t),
			}),
		}

		req := openai.ChatCompletionNewParams{
			Model:    client.config.Model,
			Messages: messages,
		}

		// This tests if vLLM can handle multimodal structure at all
		response, err := client.client.Chat.Completions.New(ctx, req)
		if err != nil {
			// If this fails, it means vLLM doesn't support multimodal structure
			t.Logf("❌ vLLM/Gemma 3 doesn't support multimodal message structure: %v", err)
			t.Skip("Skipping test - vLLM doesn't support multimodal input structure")
			return
		}

		require.NotNil(t, response, "Response should not be nil")
		require.Len(t, response.Choices, 1, "Should have exactly one choice")

		choice := response.Choices[0]
		assert.NotEmpty(t, choice.Message.Content, "Should have content response")

		content := strings.ToLower(choice.Message.Content)
		hasTest := strings.Contains(content, "test") || strings.Contains(content, "hello")

		t.Logf("Response to multimodal structure: %s", choice.Message.Content)
		if hasTest {
			t.Logf("✅ Model responded to text part of multimodal message")
		}
	})

	t.Run("TextOnlyWithTools", func(t *testing.T) {
		// Regular text request with tools (no image)
		weatherTool := CreateWeatherTool()

		req := client.CreateToolRequest("What's the weather in Tokyo?", []openai.ChatCompletionToolUnionParam{weatherTool})

		response, err := client.SendRequest(ctx, req)
		require.NoError(t, err, "Request should not fail")
		require.NotNil(t, response, "Response should not be nil")

		// Should call the weather tool
		choice := response.Choices[0]
		assert.Equal(t, "tool_calls", choice.FinishReason, "Should call weather tool")
		assert.NotEmpty(t, choice.Message.ToolCalls, "Should have tool calls")

		toolCall := choice.Message.ToolCalls[0]
		assert.Equal(t, "get_weather", toolCall.Function.Name, "Should call get_weather")

		t.Logf("Text-only tool call: %s", toolCall.Function.Arguments)
	})
}
