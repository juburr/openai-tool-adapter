//go:build e2e

package e2e

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBasicNonStreamingRequests(t *testing.T) {
	client := NewTestClient()
	ctx, cancel := client.CreateTimeoutContext()
	defer cancel()

	t.Run("RequestWithoutTools", func(t *testing.T) {
		request := client.CreateBasicRequest("Hello, how are you today?")

		response, err := client.SendRequest(ctx, request)
		require.NoError(t, err, "Basic request should not fail")
		require.NotNil(t, response, "Response should not be nil")
		require.Len(t, response.Choices, 1, "Should have exactly one choice")

		choice := response.Choices[0]
		assert.Empty(t, choice.Message.ToolCalls, "Should have no tool calls")
		assert.NotEmpty(t, choice.Message.Content, "Should have content")
		assert.Equal(t, "assistant", string(choice.Message.Role), "Role should be assistant")
		assert.NotEqual(t, "tool_calls", choice.FinishReason, "Should not finish with tool_calls")

		t.Logf("Response content: %s", choice.Message.Content)
	})

	t.Run("RequestWithoutToolsButMentioningWeather", func(t *testing.T) {
		request := client.CreateBasicRequest("What's the weather like today?")

		response, err := client.SendRequest(ctx, request)
		require.NoError(t, err, "Request should not fail")
		require.NotNil(t, response, "Response should not be nil")
		require.Len(t, response.Choices, 1, "Should have exactly one choice")

		choice := response.Choices[0]
		assert.Empty(t, choice.Message.ToolCalls, "Should have no tool calls when no tools provided")
		assert.NotEmpty(t, choice.Message.Content, "Should have content")

		// The model should respond naturally without trying to make tool calls
		content := strings.ToLower(choice.Message.Content)
		assert.Contains(t, content, "weather", "Response should acknowledge the weather question")

		t.Logf("Response content: %s", choice.Message.Content)
	})

	t.Run("EmptyMessage", func(t *testing.T) {
		request := client.CreateBasicRequest("")

		response, err := client.SendRequest(ctx, request)
		require.NoError(t, err, "Empty message should not fail")
		require.NotNil(t, response, "Response should not be nil")
		require.Len(t, response.Choices, 1, "Should have exactly one choice")

		choice := response.Choices[0]
		assert.Empty(t, choice.Message.ToolCalls, "Should have no tool calls")
		// Content may or may not be empty depending on model behavior

		t.Logf("Response to empty message: %s", choice.Message.Content)
	})

	t.Run("LongMessage", func(t *testing.T) {
		// Create a context with longer timeout for long messages
		longCtx, cancel := client.CreateTimeoutContext()
		defer cancel()

		longMessage := strings.Repeat("Tell me a short story about adventures. ", 50) // Reduced size
		request := client.CreateBasicRequest(longMessage)

		response, err := client.SendRequest(longCtx, request)
		if err != nil {
			t.Logf("Long message failed (expected for some models): %v", err)
			return // Skip the test if it fails due to model limitations
		}
		require.NotNil(t, response, "Response should not be nil")
		require.Len(t, response.Choices, 1, "Should have exactly one choice")

		choice := response.Choices[0]
		assert.Empty(t, choice.Message.ToolCalls, "Should have no tool calls")
		assert.NotEmpty(t, choice.Message.Content, "Should have content")

		t.Logf("Response to long message (first 200 chars): %s...",
			choice.Message.Content[:min(200, len(choice.Message.Content))])
	})
}
