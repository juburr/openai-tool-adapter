//go:build e2e

package e2e

import (
	"encoding/base64"
	"encoding/json"
	"os"
	"testing"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/require"
)

func TestDebugMultimodalTransformation(t *testing.T) {
	client := NewTestClientWithVerboseLogging()

	// Read the actual nvidia.jpg file and encode it properly
	imageData, err := os.ReadFile("nvidia.jpg")
	require.NoError(t, err, "Should be able to read nvidia.jpg file")

	// Encode to base64 using Go's standard library
	testImageBase64 := base64.StdEncoding.EncodeToString(imageData)
	t.Logf("Successfully encoded nvidia.jpg: %d bytes -> %d base64 chars", len(imageData), len(testImageBase64))

	// Create multimodal message with tools
	weatherTool := CreateWeatherTool()

	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage([]openai.ChatCompletionContentPartUnionParam{
			createTextPart("Please tell me about weather patterns."),
			createImagePart(testImageBase64),
		}),
	}

	req := openai.ChatCompletionNewParams{
		Model:    client.config.Model,
		Messages: messages,
		Tools:    []openai.ChatCompletionToolParam{weatherTool},
	}

	// Transform the request
	transformedReq, err := client.adapter.TransformCompletionsRequest(req)
	require.NoError(t, err, "Transform request should not fail")

	// Print what we're sending to debug
	t.Logf("=== ORIGINAL REQUEST ===")
	originalJSON, _ := json.MarshalIndent(req, "", "  ")
	t.Logf("%s", originalJSON)

	t.Logf("=== TRANSFORMED REQUEST ===")
	transformedJSON, _ := json.MarshalIndent(transformedReq, "", "  ")
	t.Logf("%s", transformedJSON)

	// Check if the transformed request has preserved the image
	if len(transformedReq.Messages) > 0 {
		firstMsg := transformedReq.Messages[0]
		if firstMsg.OfUser != nil {
			content := firstMsg.OfUser.Content
			if parts := content.OfArrayOfContentParts; len(parts) > 0 {
				t.Logf("✅ Multimodal structure preserved - %d content parts", len(parts))
				for i, part := range parts {
					if part.OfText != nil {
						t.Logf("  Part %d: Text - %d chars", i, len(part.OfText.Text))
					} else if part.OfImageURL != nil {
						t.Logf("  Part %d: Image - URL length %d chars", i, len(part.OfImageURL.ImageURL.URL))
					}
				}
			} else if str := content.OfString.Or(""); str != "" {
				t.Logf("❌ Converted to simple text - %d chars", len(str))
			}
		}
	}

	// Try sending to vLLM
	ctx, cancel := client.CreateTimeoutContext()
	defer cancel()

	response, err := client.client.Chat.Completions.New(ctx, transformedReq)
	if err != nil {
		t.Logf("❌ vLLM Error: %v", err)

		// Try with a simple text-only version to see if that works
		t.Logf("=== TESTING SIMPLE TEXT VERSION ===")
		simpleReq := openai.ChatCompletionNewParams{
			Model: client.config.Model,
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Please tell me about weather patterns."),
			},
			Tools: []openai.ChatCompletionToolParam{weatherTool},
		}

		simpleTransformed, err := client.adapter.TransformCompletionsRequest(simpleReq)
		require.NoError(t, err)

		simpleResponse, err := client.client.Chat.Completions.New(ctx, simpleTransformed)
		if err != nil {
			t.Logf("❌ Even simple text fails: %v", err)
		} else {
			t.Logf("✅ Simple text works fine")
			t.Logf("Response: %s", simpleResponse.Choices[0].Message.Content)
		}

		return
	}

	t.Logf("✅ vLLM accepted multimodal request!")
	t.Logf("Response: %s", response.Choices[0].Message.Content)
}
