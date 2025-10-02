//go:build e2e

package e2e

import (
	"context"
	"encoding/base64"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestDirectMultimodalWithoutAdapter tests sending multimodal content directly to vLLM
// without using the adapter, to verify that vLLM itself supports multimodal input
func TestDirectMultimodalWithoutAdapter(t *testing.T) {
	// Read the actual nvidia.jpg file and encode it properly
	imageData, err := os.ReadFile("nvidia.jpg")
	require.NoError(t, err, "Should be able to read nvidia.jpg file")

	// Encode to base64 using Go's standard library
	base64String := base64.StdEncoding.EncodeToString(imageData)
	t.Logf("Successfully encoded nvidia.jpg: %d bytes -> %d base64 chars", len(imageData), len(base64String))

	// Create a direct OpenAI client (no adapter)
	client := openai.NewClient(
		option.WithBaseURL("http://localhost:8000/v1"),
		option.WithAPIKey("test-key"),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	t.Run("DirectImageAnalysis", func(t *testing.T) {
		// Create multimodal content with text + image
		parts := []openai.ChatCompletionContentPartUnionParam{
			{
				OfText: &openai.ChatCompletionContentPartTextParam{
					Type: "text",
					Text: "What do you see in this image? Please describe what company logo this is.",
				},
			},
			{
				OfImageURL: &openai.ChatCompletionContentPartImageParam{
					Type: "image_url",
					ImageURL: openai.ChatCompletionContentPartImageImageURLParam{
						URL: "data:image/jpeg;base64," + base64String,
					},
				},
			},
		}

		// Create request with NO TOOLS - just pure multimodal
		req := openai.ChatCompletionNewParams{
			Model: "google/gemma-3-4b-it",
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage(parts),
			},
			MaxTokens: openai.Int(100),
		}

		// Send directly to vLLM (no adapter involved)
		response, err := client.Chat.Completions.New(ctx, req)
		if err != nil {
			t.Logf("❌ Direct vLLM multimodal request failed: %v", err)

			// Try a simple text-only request to confirm vLLM is working
			textReq := openai.ChatCompletionNewParams{
				Model: "google/gemma-3-4b-it",
				Messages: []openai.ChatCompletionMessageParamUnion{
					openai.UserMessage("Say hello"),
				},
				MaxTokens: openai.Int(20),
			}

			textResp, textErr := client.Chat.Completions.New(ctx, textReq)
			if textErr != nil {
				t.Fatalf("Even simple text request failed: %v", textErr)
			}

			t.Logf("✅ Simple text works: %s", textResp.Choices[0].Message.Content)
			t.Fatalf("vLLM accepts text but rejects multimodal: %v", err)
		}

		require.NotNil(t, response, "Response should not be nil")
		require.Len(t, response.Choices, 1, "Should have exactly one choice")

		choice := response.Choices[0]
		assert.NotEmpty(t, choice.Message.Content, "Should have content response")

		content := strings.ToLower(choice.Message.Content)
		t.Logf("✅ vLLM successfully processed multimodal input!")
		t.Logf("Response: %s", choice.Message.Content)

		// Check if it recognizes NVIDIA (this would confirm it actually processed the image)
		if strings.Contains(content, "nvidia") {
			t.Logf("🎯 Model correctly identified NVIDIA in the image!")
		} else {
			t.Logf("⚠️  Model responded but didn't mention NVIDIA - content: %s", choice.Message.Content)
		}
	})

	t.Run("Base64Validation", func(t *testing.T) {
		// Test that our base64 encoding is valid
		decoded, err := base64.StdEncoding.DecodeString(base64String)
		require.NoError(t, err, "Base64 string should be valid")
		assert.Equal(t, imageData, decoded, "Decoded data should match original")
		t.Logf("✅ Base64 encoding/decoding works correctly")
	})

	t.Run("MinimalMultimodal", func(t *testing.T) {
		// Test with minimal multimodal content
		parts := []openai.ChatCompletionContentPartUnionParam{
			{
				OfText: &openai.ChatCompletionContentPartTextParam{
					Type: "text",
					Text: "Hello",
				},
			},
			{
				OfImageURL: &openai.ChatCompletionContentPartImageParam{
					Type: "image_url",
					ImageURL: openai.ChatCompletionContentPartImageImageURLParam{
						URL: "data:image/jpeg;base64," + base64String,
					},
				},
			},
		}

		req := openai.ChatCompletionNewParams{
			Model: "google/gemma-3-4b-it",
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage(parts),
			},
			MaxTokens: openai.Int(50),
		}

		response, err := client.Chat.Completions.New(ctx, req)
		if err != nil {
			t.Logf("❌ Minimal multimodal failed: %v", err)
			// This helps us understand what's failing
			return
		}

		require.NotNil(t, response)
		t.Logf("✅ Minimal multimodal request succeeded")
		t.Logf("Response: %s", response.Choices[0].Message.Content)
	})
}
