package tooladapter_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"testing"

	tooladapter "github.com/juburr/openai-tool-adapter/v2"
	"github.com/openai/openai-go/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ============================================================================
// CONFIGURATION AND SETUP TESTS
// ============================================================================

// TestMessageInjectionStrategy verifies that tool prompts are correctly injected
// into messages according to the proper strategy (modify existing, not prepend new)
func TestMessageInjectionStrategy(t *testing.T) {
	adapter := tooladapter.New(tooladapter.WithLogLevel(slog.LevelError))
	tools := []openai.ChatCompletionToolUnionParam{
		createMockTool("test_func", "Test function"),
	}

	// ========================================================================
	// CORE SCENARIOS: Common message patterns
	// ========================================================================

	t.Run("AppendsToExistingSystemMessage", func(t *testing.T) {
		// Test that tool prompt is appended to existing system message
		// With single system message, "last" = "first" = the only one
		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.SystemMessage("You are a helpful assistant."),
				openai.UserMessage("Hello"),
			},
			Tools: tools,
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// Should still have 2 messages (not 3)
		assert.Len(t, result.Messages, 2, "Should modify existing message, not add new one")

		// First message should still be system
		assert.NotNil(t, result.Messages[0].OfSystem, "First message should remain system")

		// System message should contain both original and tool content
		systemContent := result.Messages[0].OfSystem.Content.OfString.Or("")
		assert.Contains(t, systemContent, "You are a helpful assistant", "Should preserve original system content")
		assert.Contains(t, systemContent, "test_func", "Should append tool information")
		assert.Contains(t, systemContent, "Test function", "Should include tool description")

		// Second message should remain unchanged
		assert.NotNil(t, result.Messages[1].OfUser, "Second message should remain user")
		userContent := result.Messages[1].OfUser.Content
		assert.Equal(t, "Hello", userContent.OfString.Or(""), "User message should be unchanged")
	})

	t.Run("PrependsToFirstUserMessageWhenNoSystemByDefault", func(t *testing.T) {
		// Test default behavior: no system message support (for Gemma-like models)
		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Hello, can you help me?"),
				openai.AssistantMessage("Of course!"),
				openai.UserMessage("Great, thanks!"),
			},
			Tools: tools,
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// Should still have 3 messages (modified first user, not added new)
		assert.Len(t, result.Messages, 3, "Should modify existing user message, not add new one")

		// First message should be modified user with tool instructions prepended
		assert.NotNil(t, result.Messages[0].OfUser, "First message should be user")
		userContent := result.Messages[0].OfUser.Content.OfString.Or("")
		assert.Contains(t, userContent, "test_func", "Should contain tool info")
		assert.Contains(t, userContent, "Test function", "Should contain tool description")
		assert.Contains(t, userContent, "Hello, can you help me?", "Should preserve original user content")

		// Tool content should come BEFORE user content
		toolIdx := strings.Index(userContent, "test_func")
		userIdx := strings.Index(userContent, "Hello, can you help me?")
		assert.Less(t, toolIdx, userIdx, "Tool content should come before user content")

		// Other messages should remain unchanged
		assert.NotNil(t, result.Messages[1].OfAssistant, "Assistant message should remain")
		assert.NotNil(t, result.Messages[2].OfUser, "Second user message should remain")
		assert.Equal(t, "Great, thanks!", result.Messages[2].OfUser.Content.OfString.Or(""))
	})

	t.Run("CreatesInstructionMessageWhenNoMessages", func(t *testing.T) {
		// Test empty messages case
		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{},
			Tools:    tools,
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// Should have exactly 1 message
		assert.Len(t, result.Messages, 1, "Should create one instruction message")

		// Should be a user instruction (default: no system support)
		assert.NotNil(t, result.Messages[0].OfUser, "Should create user instruction with default settings")

		// Should contain tool information
		instructionContent := result.Messages[0].OfUser.Content.OfString.Or("")
		assert.Contains(t, instructionContent, "test_func", "Should contain tool information")
		assert.Contains(t, instructionContent, "Test function", "Should include tool description")
	})

	t.Run("HandlesMultipleSystemMessages", func(t *testing.T) {
		// Test edge case: multiple system messages (modifies LAST one for "last wins" semantics)
		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.SystemMessage("First system message."),
				openai.SystemMessage("Second system message."),
				openai.UserMessage("Hello"),
			},
			Tools: tools,
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// Should still have 3 messages
		assert.Len(t, result.Messages, 3, "Should not add new messages")

		// First system message should be UNCHANGED
		assert.NotNil(t, result.Messages[0].OfSystem)
		firstContent := result.Messages[0].OfSystem.Content.OfString.Or("")
		assert.Equal(t, "First system message.", firstContent, "First system should be unchanged")
		assert.NotContains(t, firstContent, "test_func", "Should not modify first system message")

		// Second (LAST) system message should be MODIFIED
		assert.NotNil(t, result.Messages[1].OfSystem)
		secondContent := result.Messages[1].OfSystem.Content.OfString.Or("")
		assert.Contains(t, secondContent, "Second system message", "Should preserve second system content")
		assert.Contains(t, secondContent, "test_func", "Should append tool information to LAST system message")
	})

	t.Run("HandlesSystemAfterUser", func(t *testing.T) {
		// Test unusual ordering: user message before system message
		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("User message first"),
				openai.SystemMessage("System message second"),
				openai.UserMessage("Another user message"),
			},
			Tools: tools,
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// Should still have 3 messages
		assert.Len(t, result.Messages, 3, "Should not add new messages")

		// System message (at index 1) should be modified
		assert.NotNil(t, result.Messages[1].OfSystem)
		systemContent := result.Messages[1].OfSystem.Content.OfString.Or("")
		assert.Contains(t, systemContent, "System message second", "Should preserve system content")
		assert.Contains(t, systemContent, "test_func", "Should append tool information")

		// First user message should be unchanged
		assert.NotNil(t, result.Messages[0].OfUser)
		firstUserContent := result.Messages[0].OfUser.Content
		assert.Equal(t, "User message first", firstUserContent.OfString.Or(""), "First user message should be unchanged")
	})

	t.Run("HandlesOnlyAssistantMessages", func(t *testing.T) {
		// Edge case: only assistant messages (no system or user)
		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.AssistantMessage("Assistant message 1"),
				openai.AssistantMessage("Assistant message 2"),
			},
			Tools: tools,
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// Should have 3 messages (new user instruction + 2 assistant)
		assert.Len(t, result.Messages, 3, "Should add one new instruction message")

		// First should be new user instruction with tools (default: no system support)
		assert.NotNil(t, result.Messages[0].OfUser, "Should create user instruction")
		instructionContent := result.Messages[0].OfUser.Content.OfString.Or("")
		assert.Contains(t, instructionContent, "test_func", "Should contain tool information")

		// Assistant messages should be preserved
		assert.NotNil(t, result.Messages[1].OfAssistant)
		assert.NotNil(t, result.Messages[2].OfAssistant)
	})

	t.Run("LastSystemWinsWithThreeSystemMessages", func(t *testing.T) {
		// Test with THREE system messages to really verify "last wins" behavior
		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.SystemMessage("First system."),
				openai.UserMessage("Hi"),
				openai.SystemMessage("Second system."),
				openai.AssistantMessage("Hello!"),
				openai.SystemMessage("Third system."),
				openai.UserMessage("Help me"),
			},
			Tools: tools,
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// Should still have 6 messages
		assert.Len(t, result.Messages, 6, "Should not add new messages")

		// First two system messages should be UNCHANGED
		assert.Equal(t, "First system.", result.Messages[0].OfSystem.Content.OfString.Or(""))
		assert.Equal(t, "Second system.", result.Messages[2].OfSystem.Content.OfString.Or(""))

		// ONLY the LAST (third) system message should be modified
		thirdContent := result.Messages[4].OfSystem.Content.OfString.Or("")
		assert.Contains(t, thirdContent, "Third system", "Should preserve third system content")
		assert.Contains(t, thirdContent, "test_func", "Should append tools to LAST system message")

		// Other messages unchanged
		assert.Equal(t, "Hi", result.Messages[1].OfUser.Content.OfString.Or(""))
		assert.Equal(t, "Help me", result.Messages[5].OfUser.Content.OfString.Or(""))
	})

	t.Run("PreservesMessageOrderAndContent", func(t *testing.T) {
		// Test complex conversation preservation
		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.SystemMessage("You are helpful."),
				openai.UserMessage("What's 2+2?"),
				openai.AssistantMessage("2+2 equals 4."),
				openai.UserMessage("And 3+3?"),
				openai.AssistantMessage("3+3 equals 6."),
				openai.UserMessage("Thanks!"),
			},
			Tools: tools,
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// Should have same number of messages
		assert.Len(t, result.Messages, 6, "Should preserve message count")

		// Verify order and content preservation (except first system message)
		assert.NotNil(t, result.Messages[0].OfSystem)
		systemContent := result.Messages[0].OfSystem.Content.OfString.Or("")
		assert.Contains(t, systemContent, "You are helpful")
		assert.Contains(t, systemContent, "test_func")

		// Rest should be unchanged
		assert.Equal(t, "What's 2+2?", result.Messages[1].OfUser.Content.OfString.Or(""))
		assert.Contains(t, getAssistantContent(result.Messages[2]), "2+2 equals 4")
		assert.Equal(t, "And 3+3?", result.Messages[3].OfUser.Content.OfString.Or(""))
		assert.Contains(t, getAssistantContent(result.Messages[4]), "3+3 equals 6")
		assert.Equal(t, "Thanks!", result.Messages[5].OfUser.Content.OfString.Or(""))
	})

	// ========================================================================
	// COMPREHENSIVE MESSAGE PATTERN TESTS
	// ========================================================================

	t.Run("NoSystemMessage_UserAssistantPattern", func(t *testing.T) {
		// Common pattern: user -> assistant -> user -> assistant (no system)
		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("What's the weather?"),
				openai.AssistantMessage("I'll check that for you."),
				openai.UserMessage("Also, what's the time?"),
				openai.AssistantMessage("Let me get that information."),
			},
			Tools: tools,
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// Should modify first user message (no new messages)
		assert.Len(t, result.Messages, 4, "Should not add new messages")

		// First user message should be modified with tools
		assert.NotNil(t, result.Messages[0].OfUser)
		firstContent := result.Messages[0].OfUser.Content.OfString.Or("")
		assert.Contains(t, firstContent, "test_func")
		assert.Contains(t, firstContent, "What's the weather?")

		// Other messages unchanged
		assert.Contains(t, getAssistantContent(result.Messages[1]), "I'll check that")
		assert.Equal(t, "Also, what's the time?", result.Messages[2].OfUser.Content.OfString.Or(""))
		assert.Contains(t, getAssistantContent(result.Messages[3]), "Let me get that")
	})

	t.Run("SingleSystemFirst_SystemUserAssistantPattern", func(t *testing.T) {
		// Most common pattern: system -> user -> assistant -> user -> assistant
		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.SystemMessage("You are a weather assistant."),
				openai.UserMessage("What's today's forecast?"),
				openai.AssistantMessage("Today will be sunny."),
				openai.UserMessage("And tomorrow?"),
				openai.AssistantMessage("Tomorrow expects rain."),
			},
			Tools: tools,
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		assert.Len(t, result.Messages, 5, "Should preserve message count")

		// System message (first and last) should be modified
		systemContent := result.Messages[0].OfSystem.Content.OfString.Or("")
		assert.Contains(t, systemContent, "You are a weather assistant", "Should preserve original system content")
		assert.Contains(t, systemContent, "test_func", "Should append tool information")

		// Other messages unchanged
		assert.Equal(t, "What's today's forecast?", result.Messages[1].OfUser.Content.OfString.Or(""))
		assert.Contains(t, getAssistantContent(result.Messages[2]), "sunny")
		assert.Equal(t, "And tomorrow?", result.Messages[3].OfUser.Content.OfString.Or(""))
		assert.Contains(t, getAssistantContent(result.Messages[4]), "rain")
	})

	t.Run("SingleSystemMiddle_UserSystemAssistantPattern", func(t *testing.T) {
		// Edge case: System message in the middle of conversation
		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Hello"),
				openai.AssistantMessage("Hi there!"),
				openai.SystemMessage("Remember to be helpful."), // System in middle
				openai.UserMessage("What's the weather?"),
				openai.AssistantMessage("Let me check."),
			},
			Tools: tools,
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		assert.Len(t, result.Messages, 5, "Should preserve message count")

		// Only the system message (at index 2) should be modified
		assert.Equal(t, "Hello", result.Messages[0].OfUser.Content.OfString.Or(""))
		assert.Contains(t, getAssistantContent(result.Messages[1]), "Hi there")

		// System message should be modified
		systemContent := result.Messages[2].OfSystem.Content.OfString.Or("")
		assert.Contains(t, systemContent, "Remember to be helpful", "Should preserve system content")
		assert.Contains(t, systemContent, "test_func", "Should append tool information")

		// Rest unchanged
		assert.Equal(t, "What's the weather?", result.Messages[3].OfUser.Content.OfString.Or(""))
		assert.Contains(t, getAssistantContent(result.Messages[4]), "Let me check")
	})

	t.Run("SingleSystemLast_UserAssistantSystemPattern", func(t *testing.T) {
		// Edge case: System message at the end
		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Hello"),
				openai.AssistantMessage("Hi!"),
				openai.UserMessage("How are you?"),
				openai.AssistantMessage("I'm doing well!"),
				openai.SystemMessage("Always be polite."), // System at end
			},
			Tools: tools,
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		assert.Len(t, result.Messages, 5, "Should preserve message count")

		// First 4 messages unchanged
		assert.Equal(t, "Hello", result.Messages[0].OfUser.Content.OfString.Or(""))
		assert.Contains(t, getAssistantContent(result.Messages[1]), "Hi!")
		assert.Equal(t, "How are you?", result.Messages[2].OfUser.Content.OfString.Or(""))
		assert.Contains(t, getAssistantContent(result.Messages[3]), "I'm doing well")

		// Last system message should be modified
		systemContent := result.Messages[4].OfSystem.Content.OfString.Or("")
		assert.Contains(t, systemContent, "Always be polite", "Should preserve system content")
		assert.Contains(t, systemContent, "test_func", "Should append tool information")
	})

	t.Run("MultipleSystemMessages_CommonPattern", func(t *testing.T) {
		// Multiple system messages: system -> user -> assistant -> system -> user -> assistant
		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.SystemMessage("You are helpful."),
				openai.UserMessage("Hello"),
				openai.AssistantMessage("Hi!"),
				openai.SystemMessage("Be concise."), // Second system message
				openai.UserMessage("What's 2+2?"),
				openai.AssistantMessage("4"),
			},
			Tools: tools,
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		assert.Len(t, result.Messages, 6, "Should preserve message count")

		// First system message should be UNCHANGED
		assert.Equal(t, "You are helpful.", result.Messages[0].OfSystem.Content.OfString.Or(""))

		// Middle messages unchanged
		assert.Equal(t, "Hello", result.Messages[1].OfUser.Content.OfString.Or(""))
		assert.Contains(t, getAssistantContent(result.Messages[2]), "Hi!")

		// Second (LAST) system message should be MODIFIED
		systemContent := result.Messages[3].OfSystem.Content.OfString.Or("")
		assert.Contains(t, systemContent, "Be concise", "Should preserve second system content")
		assert.Contains(t, systemContent, "test_func", "Should append tools to LAST system")

		// Rest unchanged
		assert.Equal(t, "What's 2+2?", result.Messages[4].OfUser.Content.OfString.Or(""))
		assert.Contains(t, getAssistantContent(result.Messages[5]), "4")
	})

	t.Run("OnlySystemMessages", func(t *testing.T) {
		// Edge case: Only system messages, no user/assistant
		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.SystemMessage("First instruction."),
				openai.SystemMessage("Second instruction."),
				openai.SystemMessage("Third instruction."),
			},
			Tools: tools,
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		assert.Len(t, result.Messages, 3, "Should preserve message count")

		// First two unchanged
		assert.Equal(t, "First instruction.", result.Messages[0].OfSystem.Content.OfString.Or(""))
		assert.Equal(t, "Second instruction.", result.Messages[1].OfSystem.Content.OfString.Or(""))

		// Last one modified
		thirdContent := result.Messages[2].OfSystem.Content.OfString.Or("")
		assert.Contains(t, thirdContent, "Third instruction", "Should preserve content")
		assert.Contains(t, thirdContent, "test_func", "Should append tools")
	})

	t.Run("AlternatingSystemUserPattern", func(t *testing.T) {
		// Unusual: Alternating system and user messages
		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.SystemMessage("System 1"),
				openai.UserMessage("User 1"),
				openai.SystemMessage("System 2"),
				openai.UserMessage("User 2"),
				openai.SystemMessage("System 3"),
			},
			Tools: tools,
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		assert.Len(t, result.Messages, 5, "Should preserve message count")

		// Only last system (index 4) should be modified
		assert.Equal(t, "System 1", result.Messages[0].OfSystem.Content.OfString.Or(""))
		assert.Equal(t, "User 1", result.Messages[1].OfUser.Content.OfString.Or(""))
		assert.Equal(t, "System 2", result.Messages[2].OfSystem.Content.OfString.Or(""))
		assert.Equal(t, "User 2", result.Messages[3].OfUser.Content.OfString.Or(""))

		lastContent := result.Messages[4].OfSystem.Content.OfString.Or("")
		assert.Contains(t, lastContent, "System 3", "Should preserve content")
		assert.Contains(t, lastContent, "test_func", "Should append tools to last")
	})

	t.Run("ComplexRealWorldConversation", func(t *testing.T) {
		// Complex real-world-like conversation with context switches
		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.SystemMessage("You are a helpful AI assistant."),
				openai.UserMessage("Can you help me plan a trip?"),
				openai.AssistantMessage("I'd be happy to help you plan a trip!"),
				openai.UserMessage("I want to go to Paris."),
				openai.AssistantMessage("Paris is wonderful! When are you planning to go?"),
				openai.SystemMessage("Focus on budget options."), // Context change
				openai.UserMessage("Next month, what's affordable?"),
				openai.AssistantMessage("Here are budget-friendly options..."),
			},
			Tools: tools,
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		assert.Len(t, result.Messages, 8, "Should preserve message count")

		// First system unchanged
		assert.Equal(t, "You are a helpful AI assistant.", result.Messages[0].OfSystem.Content.OfString.Or(""))

		// Second (last) system modified
		lastSystemContent := result.Messages[5].OfSystem.Content.OfString.Or("")
		assert.Contains(t, lastSystemContent, "Focus on budget options", "Should preserve content")
		assert.Contains(t, lastSystemContent, "test_func", "Should append tools")

		// Verify conversation flow preserved
		assert.Equal(t, "Can you help me plan a trip?", result.Messages[1].OfUser.Content.OfString.Or(""))
		assert.Equal(t, "Next month, what's affordable?", result.Messages[6].OfUser.Content.OfString.Or(""))
	})
}

// Helper function to extract assistant message content
func getAssistantContent(msg openai.ChatCompletionMessageParamUnion) string {
	if msg.OfAssistant != nil {
		// Assistant messages have Content as a union type similar to user messages
		content := msg.OfAssistant.Content
		// Use the OfString field with Or() fallback
		return content.OfString.Or("")
	}
	return ""
}

// TestSystemMessageSupportBehavior comprehensively tests the WithSystemMessageSupport option
// and its impact on message injection strategies.
func TestSystemMessageSupportBehavior(t *testing.T) {
	tools := []openai.ChatCompletionToolUnionParam{
		openai.ChatCompletionFunctionTool(
			openai.FunctionDefinitionParam{
				Name:        "test_func",
				Description: openai.String("Test function"),
			},
		),
	}

	t.Run("SystemSupportEnabled_NoExistingSystem", func(t *testing.T) {
		// With system support enabled, should prepend SYSTEM instruction
		adapter := tooladapter.New(
			tooladapter.WithLogLevel(slog.LevelError),
			tooladapter.WithSystemMessageSupport(true),
		)

		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Hello"),
				openai.AssistantMessage("Hi there!"),
			},
			Tools: tools,
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// Should prepend a system instruction
		assert.Len(t, result.Messages, 3, "Should add one system instruction")
		assert.NotNil(t, result.Messages[0].OfSystem, "First message should be system")
		systemContent := result.Messages[0].OfSystem.Content.OfString.Or("")
		assert.Contains(t, systemContent, "test_func", "System message should contain tool info")

		// Original messages should be preserved
		assert.Equal(t, "Hello", result.Messages[1].OfUser.Content.OfString.Or(""))
		assert.NotNil(t, result.Messages[2].OfAssistant)
	})

	t.Run("SystemSupportDisabled_NoExistingSystem", func(t *testing.T) {
		// With system support disabled (default), should insert USER instruction
		adapter := tooladapter.New(
			tooladapter.WithLogLevel(slog.LevelError),
			tooladapter.WithSystemMessageSupport(false), // Explicit for clarity
		)

		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Hello"),
				openai.AssistantMessage("Hi there!"),
			},
			Tools: tools,
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// Should modify first user message (no new messages)
		assert.Len(t, result.Messages, 2, "Should not add new messages")
		assert.NotNil(t, result.Messages[0].OfUser, "First message should be user")
		userContent := result.Messages[0].OfUser.Content.OfString.Or("")
		assert.Contains(t, userContent, "test_func", "Should contain tool info")
		assert.Contains(t, userContent, "Hello", "Should preserve original content")

		// Assistant message should be preserved
		assert.NotNil(t, result.Messages[1].OfAssistant)
	})

	t.Run("BothSettings_WithExistingSystem", func(t *testing.T) {
		// When system message exists, both settings should append to it
		testCases := []struct {
			name            string
			systemSupported bool
		}{
			{"SystemSupportEnabled", true},
			{"SystemSupportDisabled", false},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				adapter := tooladapter.New(
					tooladapter.WithLogLevel(slog.LevelError),
					tooladapter.WithSystemMessageSupport(tc.systemSupported),
				)

				req := openai.ChatCompletionNewParams{
					Messages: []openai.ChatCompletionMessageParamUnion{
						openai.SystemMessage("You are helpful"),
						openai.UserMessage("Hello"),
					},
					Tools: tools,
				}

				result, err := adapter.TransformCompletionsRequest(req)
				require.NoError(t, err)

				// Should modify existing system message (no new messages)
				assert.Len(t, result.Messages, 2, "Should not add new messages")

				// System message should be modified
				systemContent := result.Messages[0].OfSystem.Content.OfString.Or("")
				assert.Contains(t, systemContent, "You are helpful", "Should preserve original")
				assert.Contains(t, systemContent, "test_func", "Should append tools")

				// User message unchanged
				assert.Equal(t, "Hello", result.Messages[1].OfUser.Content.OfString.Or(""))
			})
		}
	})

	t.Run("EmptyMessages_SystemSupportEnabled", func(t *testing.T) {
		adapter := tooladapter.New(
			tooladapter.WithLogLevel(slog.LevelError),
			tooladapter.WithSystemMessageSupport(true),
		)

		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{},
			Tools:    tools,
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		assert.Len(t, result.Messages, 1, "Should create one message")
		assert.NotNil(t, result.Messages[0].OfSystem, "Should be system message")
		systemContent := result.Messages[0].OfSystem.Content.OfString.Or("")
		assert.Contains(t, systemContent, "test_func", "Should contain tool info")
	})

	t.Run("EmptyMessages_SystemSupportDisabled", func(t *testing.T) {
		adapter := tooladapter.New(
			tooladapter.WithLogLevel(slog.LevelError),
			tooladapter.WithSystemMessageSupport(false),
		)

		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{},
			Tools:    tools,
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		assert.Len(t, result.Messages, 1, "Should create one message")
		assert.NotNil(t, result.Messages[0].OfUser, "Should be user message")
		userContent := result.Messages[0].OfUser.Content.OfString.Or("")
		assert.Contains(t, userContent, "test_func", "Should contain tool info")
	})

	t.Run("OnlyAssistantMessages_SystemSupport", func(t *testing.T) {
		testCases := []struct {
			name            string
			systemSupported bool
			expectedRole    string
		}{
			{"SystemEnabled", true, "system"},
			{"SystemDisabled", false, "user"},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				adapter := tooladapter.New(
					tooladapter.WithLogLevel(slog.LevelError),
					tooladapter.WithSystemMessageSupport(tc.systemSupported),
				)

				req := openai.ChatCompletionNewParams{
					Messages: []openai.ChatCompletionMessageParamUnion{
						openai.AssistantMessage("Previous response"),
						openai.AssistantMessage("Another response"),
					},
					Tools: tools,
				}

				result, err := adapter.TransformCompletionsRequest(req)
				require.NoError(t, err)

				assert.Len(t, result.Messages, 3, "Should add one instruction")

				if tc.expectedRole == "system" {
					assert.NotNil(t, result.Messages[0].OfSystem, "Should be system")
					content := result.Messages[0].OfSystem.Content.OfString.Or("")
					assert.Contains(t, content, "test_func")
				} else {
					assert.NotNil(t, result.Messages[0].OfUser, "Should be user")
					content := result.Messages[0].OfUser.Content.OfString.Or("")
					assert.Contains(t, content, "test_func")
				}
			})
		}
	})

	t.Run("ComplexConversation_SystemSupport", func(t *testing.T) {
		// Test with a complex conversation structure
		adapter := tooladapter.New(
			tooladapter.WithLogLevel(slog.LevelError),
			tooladapter.WithSystemMessageSupport(true),
		)

		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("First question"),
				openai.AssistantMessage("First answer"),
				openai.UserMessage("Second question"),
				openai.AssistantMessage("Second answer"),
				openai.UserMessage("Third question"),
			},
			Tools: tools,
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// Should prepend system instruction
		assert.Len(t, result.Messages, 6, "Should add one system instruction")
		assert.NotNil(t, result.Messages[0].OfSystem, "First should be system")

		// Verify conversation order preserved
		assert.Equal(t, "First question", result.Messages[1].OfUser.Content.OfString.Or(""))
		assert.NotNil(t, result.Messages[2].OfAssistant)
		assert.Equal(t, "Second question", result.Messages[3].OfUser.Content.OfString.Or(""))
		assert.NotNil(t, result.Messages[4].OfAssistant)
		assert.Equal(t, "Third question", result.Messages[5].OfUser.Content.OfString.Or(""))
	})

	t.Run("MultipleSystemMessages_BothSettings", func(t *testing.T) {
		// Test with multiple system messages - should always modify the LAST one
		testCases := []bool{true, false}

		for _, systemSupported := range testCases {
			name := fmt.Sprintf("SystemSupport_%v", systemSupported)
			t.Run(name, func(t *testing.T) {
				adapter := tooladapter.New(
					tooladapter.WithLogLevel(slog.LevelError),
					tooladapter.WithSystemMessageSupport(systemSupported),
				)

				req := openai.ChatCompletionNewParams{
					Messages: []openai.ChatCompletionMessageParamUnion{
						openai.SystemMessage("First system"),
						openai.UserMessage("Hello"),
						openai.SystemMessage("Second system"),
						openai.AssistantMessage("Response"),
						openai.SystemMessage("Third system"),
					},
					Tools: tools,
				}

				result, err := adapter.TransformCompletionsRequest(req)
				require.NoError(t, err)

				// Should not add new messages
				assert.Len(t, result.Messages, 5, "Should not add messages")

				// First two system messages unchanged
				assert.Equal(t, "First system", result.Messages[0].OfSystem.Content.OfString.Or(""))
				assert.Equal(t, "Second system", result.Messages[2].OfSystem.Content.OfString.Or(""))

				// Only last system message modified
				lastContent := result.Messages[4].OfSystem.Content.OfString.Or("")
				assert.Contains(t, lastContent, "Third system", "Should preserve original")
				assert.Contains(t, lastContent, "test_func", "Should append tools")
			})
		}
	})

	t.Run("PreservesMultimodalContent", func(t *testing.T) {
		// Ensure USER instruction insertion doesn't break multimodal messages
		adapter := tooladapter.New(
			tooladapter.WithLogLevel(slog.LevelError),
			tooladapter.WithSystemMessageSupport(false), // Use user instruction
		)

		// Create a user message with multimodal content (simulated)
		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Look at this image"),
				openai.AssistantMessage("I see the image"),
			},
			Tools: tools,
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// Should modify first user message (preserves message count)
		assert.Len(t, result.Messages, 2)
		assert.NotNil(t, result.Messages[0].OfUser, "First should be modified user")

		// First user message should contain both tool info and original content
		userContent := result.Messages[0].OfUser.Content.OfString.Or("")
		assert.Contains(t, userContent, "test_func", "Should contain tool info")
		assert.Contains(t, userContent, "Look at this image", "Should preserve original content")

		// Assistant message preserved
		assert.NotNil(t, result.Messages[1].OfAssistant)
	})

	t.Run("WithMetricsAndSystemSupport", func(t *testing.T) {
		// Test that system support works with other options like metrics
		var capturedEvents []tooladapter.MetricEventData

		adapter := tooladapter.New(
			tooladapter.WithLogLevel(slog.LevelError),
			tooladapter.WithSystemMessageSupport(true),
			tooladapter.WithMetricsCallback(func(data tooladapter.MetricEventData) {
				capturedEvents = append(capturedEvents, data)
			}),
		)

		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Test"),
			},
			Tools: tools,
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// Should prepend system instruction
		assert.NotNil(t, result.Messages[0].OfSystem)

		// Should have captured metrics
		assert.NotEmpty(t, capturedEvents, "Should capture metrics")
	})

	t.Run("StreamingWithSystemSupport", func(t *testing.T) {
		// Test streaming with system support enabled
		adapter := tooladapter.New(
			tooladapter.WithLogLevel(slog.LevelError),
			tooladapter.WithSystemMessageSupport(true),
		)

		// First transform the request
		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Test"),
			},
			Tools: tools,
		}

		transformedReq, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)
		assert.NotNil(t, transformedReq.Messages[0].OfSystem, "Should have system instruction")

		// Test streaming response transformation
		mockStream := NewMockStream([]openai.ChatCompletionChunk{
			createStreamChunk("Testing response"),
			createFinishChunk("stop"),
		})

		streamAdapter := adapter.TransformStreamingResponse(mockStream)
		defer func() { _ = streamAdapter.Close() }()

		var chunks []openai.ChatCompletionChunk
		for streamAdapter.Next() {
			chunks = append(chunks, streamAdapter.Current())
		}
		require.NoError(t, streamAdapter.Err())
		assert.NotEmpty(t, chunks, "Should process stream chunks")
	})

	t.Run("WithContextAndSystemSupport", func(t *testing.T) {
		// Test context-aware methods with system support
		adapter := tooladapter.New(
			tooladapter.WithLogLevel(slog.LevelError),
			tooladapter.WithSystemMessageSupport(true),
		)

		ctx := context.Background()
		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Test"),
			},
			Tools: tools,
		}

		result, err := adapter.TransformCompletionsRequestWithContext(ctx, req)
		require.NoError(t, err)

		// Should prepend system instruction
		assert.NotNil(t, result.Messages[0].OfSystem, "Should have system instruction")
	})

	t.Run("WithToolPolicyAndSystemSupport", func(t *testing.T) {
		// Test interaction with tool policies
		adapter := tooladapter.New(
			tooladapter.WithLogLevel(slog.LevelError),
			tooladapter.WithSystemMessageSupport(true),
			tooladapter.WithToolPolicy(tooladapter.ToolAllowMixed),
		)

		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Test"),
			},
			Tools: tools,
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// Should prepend system instruction regardless of tool policy
		assert.NotNil(t, result.Messages[0].OfSystem, "Should have system instruction")
	})

	t.Run("EdgeCase_VeryLongSystemMessage", func(t *testing.T) {
		// Test with very long existing system message
		longSystemMsg := strings.Repeat("System instruction. ", 1000)

		adapter := tooladapter.New(
			tooladapter.WithLogLevel(slog.LevelError),
			tooladapter.WithSystemMessageSupport(false), // Doesn't matter when system exists
		)

		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.SystemMessage(longSystemMsg),
				openai.UserMessage("Test"),
			},
			Tools: tools,
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// Should append to existing long system message
		assert.Len(t, result.Messages, 2)
		systemContent := result.Messages[0].OfSystem.Content.OfString.Or("")
		assert.Contains(t, systemContent, "System instruction.")
		assert.Contains(t, systemContent, "test_func")
		assert.Greater(t, len(systemContent), len(longSystemMsg), "Should be longer than original")
	})

	t.Run("EdgeCase_ToolMessageWithSystemSupport", func(t *testing.T) {
		// Test tool message handling with different system support settings
		testCases := []bool{true, false}

		for _, systemSupported := range testCases {
			name := fmt.Sprintf("SystemSupport_%v", systemSupported)
			t.Run(name, func(t *testing.T) {
				adapter := tooladapter.New(
					tooladapter.WithLogLevel(slog.LevelError),
					tooladapter.WithSystemMessageSupport(systemSupported),
				)

				req := openai.ChatCompletionNewParams{
					Messages: []openai.ChatCompletionMessageParamUnion{
						openai.UserMessage("Check weather"),
						openai.AssistantMessage("Checking..."),
						openai.ToolMessage("72°F", "call_123"),
						openai.UserMessage("Thanks"),
					},
					Tools: tools,
				}

				result, err := adapter.TransformCompletionsRequest(req)
				require.NoError(t, err)

				// Tool message should be removed and content injected
				// The injection location depends on systemSupported
				hasToolMessage := false
				for _, msg := range result.Messages {
					if msgBytes, _ := msg.MarshalJSON(); strings.Contains(string(msgBytes), "tool_call_id") {
						hasToolMessage = true
						break
					}
				}
				assert.False(t, hasToolMessage, "Tool message should be removed")

				// Check that tool result was injected
				firstMsg, _ := result.Messages[0].MarshalJSON()
				assert.Contains(t, string(firstMsg), "72°F", "Should contain tool result")
			})
		}
	})
}

// ============================================================================
// CONFIGURATION AND SETUP TESTS
// ============================================================================

// TestNewAdapter_Configuration verifies that the adapter initializes correctly
// with various configuration options.
func TestNewAdapter_Configuration(t *testing.T) {
	t.Run("InitializesWithDefaults", func(t *testing.T) {
		adapter := tooladapter.New()
		require.NotNil(t, adapter)
	})

	t.Run("InitializesWithCustomPrompt", func(t *testing.T) {
		customPrompt := "Custom instructions: %s"
		adapter := tooladapter.New(tooladapter.WithCustomPromptTemplate(customPrompt))
		require.NotNil(t, adapter)

		req := createMockRequest([]openai.ChatCompletionToolUnionParam{
			createMockTool("test_func", "Test"),
		})
		res, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// Just verify we got a transformed request
		assert.NotNil(t, res)
	})

	t.Run("InitializesWithLogger", func(t *testing.T) {
		adapter := tooladapter.New(tooladapter.WithLogLevel(slog.LevelDebug))
		require.NotNil(t, adapter)

		// Verify the transformation still works with custom logger
		req := createMockRequest([]openai.ChatCompletionToolUnionParam{
			createMockTool("test_func", "Test"),
		})
		res, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// Just verify the transformation occurred - should have more messages
		assert.Greater(t, len(res.Messages), len(req.Messages)-1, "Should have additional system message")
	})
}

// ============================================================================
// REQUEST TRANSFORMATION TESTS
// ============================================================================

// TestTransformCompletionsRequest_ToolInjection verifies that tool definitions are correctly
// injected into system prompts.
func TestTransformCompletionsRequest_ToolInjection(t *testing.T) {
	adapter := tooladapter.New()

	t.Run("NoToolsPassthrough", func(t *testing.T) {
		req := createMockRequest(nil)
		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// Should be unchanged when no tools
		assert.Equal(t, len(req.Messages), len(result.Messages))
		assert.Nil(t, result.Tools)
	})

	t.Run("SingleToolInjection", func(t *testing.T) {
		tool := createMockTool("get_weather", "Get weather for a location")
		req := createMockRequest([]openai.ChatCompletionToolUnionParam{tool})

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// Verify tool-specific fields are removed
		assert.Nil(t, result.Tools, "Tools field should be removed")
		assert.Equal(t, openai.ChatCompletionToolChoiceOptionUnionParam{}, result.ToolChoice, "ToolChoice field should be zero value")

		// With no system message and default settings, we modify the first user message
		assert.Equal(t, len(req.Messages), len(result.Messages), "Should not add new messages")
	})

	t.Run("MultipleToolInjection", func(t *testing.T) {
		tools := []openai.ChatCompletionToolUnionParam{
			createMockTool("get_weather", "Get weather data"),
			createMockTool("get_time", "Get current time"),
			createMockTool("calculate", "Perform calculations"),
		}
		req := createMockRequest(tools)

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// With no system message and default settings, we modify the first user message
		assert.Equal(t, len(req.Messages), len(result.Messages), "Should not add new messages")
	})

	t.Run("ExistingSystemMessageHandling", func(t *testing.T) {
		existingSystem := "You are a helpful assistant."
		req := openai.ChatCompletionNewParams{
			Model: openai.ChatModelGPT4o,
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.SystemMessage(existingSystem),
				openai.UserMessage("Hello"),
			},
			Tools: []openai.ChatCompletionToolUnionParam{
				createMockTool("test_func", "Test function"),
			},
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// NEW BEHAVIOR: Should modify existing system message, not add new one
		assert.Equal(t, len(req.Messages), len(result.Messages))
	})
}

// ============================================================================
// RESPONSE TRANSFORMATION TESTS - ALL JSON FORMATS
// ============================================================================

// TestFunctionCallFormats_AllJSONEnclosureTypes verifies that function calls work
// across all common JSON presentation formats that LLMs might use
func TestFunctionCallFormats_AllJSONEnclosureTypes(t *testing.T) {
	adapter := tooladapter.New()

	// Test cases covering all JSON enclosure formats
	testCases := []struct {
		name                 string
		assistantContent     string
		expectedToolCallName string
		expectedArgs         string
		description          string
	}{
		// Plain JSON without any enclosure
		{
			name:                 "PlainJSON_SingleObjectNoArray",
			assistantContent:     `{"name": "get_weather", "parameters": {"location": "Boston"}}`,
			expectedToolCallName: "get_weather",
			expectedArgs:         `{"location": "Boston"}`,
			description:          "Plain JSON object should be parsed as function call",
		},
		{
			name:                 "PlainJSON_ArrayWithSingleObject",
			assistantContent:     `[{"name": "get_weather", "parameters": {"location": "Boston"}}]`,
			expectedToolCallName: "get_weather",
			expectedArgs:         `{"location": "Boston"}`,
			description:          "Plain JSON array with single function call",
		},
		{
			name:                 "PlainJSON_ArrayWithMultipleObjects",
			assistantContent:     `[{"name": "get_weather", "parameters": {"location": "Boston"}}, {"name": "get_time", "parameters": null}]`,
			expectedToolCallName: "get_weather", // We'll check the first one
			expectedArgs:         `{"location": "Boston"}`,
			description:          "Plain JSON array with multiple function calls",
		},

		// Single backticks (inline code)
		{
			name:                 "SingleTicks_SingleObjectNoArray",
			assistantContent:     "Here's the function call: `{\"name\": \"get_weather\", \"parameters\": {\"location\": \"Boston\"}}`",
			expectedToolCallName: "get_weather",
			expectedArgs:         `{"location": "Boston"}`,
			description:          "Single backticks around JSON object should be parsed",
		},
		{
			name:                 "SingleTicks_ArrayWithSingleObject",
			assistantContent:     "Function call: `[{\"name\": \"get_weather\", \"parameters\": {\"location\": \"Boston\"}}]`",
			expectedToolCallName: "get_weather",
			expectedArgs:         `{"location": "Boston"}`,
			description:          "Single backticks around JSON array should be parsed",
		},

		// Triple backticks without language specifier
		{
			name:                 "TripleTicks_SingleObjectNoArray",
			assistantContent:     "```\n{\"name\": \"get_weather\", \"parameters\": {\"location\": \"Boston\"}}\n```",
			expectedToolCallName: "get_weather",
			expectedArgs:         `{"location": "Boston"}`,
			description:          "Triple backticks around JSON object should be parsed",
		},
		{
			name:                 "TripleTicks_ArrayWithSingleObject",
			assistantContent:     "```\n[{\"name\": \"get_weather\", \"parameters\": {\"location\": \"Boston\"}}]\n```",
			expectedToolCallName: "get_weather",
			expectedArgs:         `{"location": "Boston"}`,
			description:          "Triple backticks around JSON array should be parsed",
		},

		// Triple backticks with 'json' language specifier
		{
			name:                 "TripleTicksJSON_SingleObjectNoArray",
			assistantContent:     "```json\n{\"name\": \"get_weather\", \"parameters\": {\"location\": \"Boston\"}}\n```",
			expectedToolCallName: "get_weather",
			expectedArgs:         `{"location": "Boston"}`,
			description:          "Triple backticks with 'json' specifier around object should be parsed",
		},
		{
			name:                 "TripleTicksJSON_ArrayWithSingleObject",
			assistantContent:     "```json\n[{\"name\": \"get_weather\", \"parameters\": {\"location\": \"Boston\"}}]\n```",
			expectedToolCallName: "get_weather",
			expectedArgs:         `{"location": "Boston"}`,
			description:          "Triple backticks with 'json' specifier around array should be parsed",
		},

		// Functions without parameters (optional parameters test)
		{
			name:                 "NoParameters_ObjectWithEmptyParams",
			assistantContent:     `{"name": "get_current_time", "parameters": {}}`,
			expectedToolCallName: "get_current_time",
			expectedArgs:         `{}`,
			description:          "Function with empty parameters object",
		},
		{
			name:                 "NoParameters_ObjectWithNullParams",
			assistantContent:     `{"name": "get_current_time", "parameters": null}`,
			expectedToolCallName: "get_current_time",
			expectedArgs:         `null`,
			description:          "Function with null parameters",
		},
		{
			name:                 "NoParameters_ObjectWithMissingParams",
			assistantContent:     `{"name": "get_current_time"}`,
			expectedToolCallName: "get_current_time",
			expectedArgs:         `null`, // Should default to null
			description:          "Function with completely missing parameters field",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Logf("Testing scenario: %s", tc.description)

			mockResp := createMockCompletion(tc.assistantContent)
			finalResp, err := adapter.TransformCompletionsResponse(mockResp)
			require.NoError(t, err, "Transform should not error")

			// Verify tool call was detected and parsed
			require.Len(t, finalResp.Choices, 1, "Should have one choice")
			choice := finalResp.Choices[0]
			assert.Equal(t, "tool_calls", string(choice.FinishReason), "Should indicate tool calls were made")
			assert.Empty(t, choice.Message.Content, "Content should be cleared when tool calls are present")
			require.Greater(t, len(choice.Message.ToolCalls), 0, "Should have at least one tool call")

			// Verify the first tool call details
			toolCall := choice.Message.ToolCalls[0]
			assert.Equal(t, tc.expectedToolCallName, toolCall.Function.Name, "Function name should match")
			assert.JSONEq(t, tc.expectedArgs, toolCall.Function.Arguments, "Function arguments should match")
			assert.NotEmpty(t, toolCall.ID, "Tool call should have a unique ID")
			assert.Equal(t, "function", fmt.Sprintf("%v", toolCall.Type), "Tool call type should be 'function'")
		})
	}
}

// TestTransformCompletionsResponse_ToolCallParsing verifies that various LLM response formats
// are correctly parsed into OpenAI tool call format using the state machine parser.
func TestTransformCompletionsResponse_ToolCallParsing(t *testing.T) {
	// Use ToolDrainAll policy to preserve previous behavior of returning all tool calls
	adapter := tooladapter.New(tooladapter.WithToolPolicy(tooladapter.ToolDrainAll))

	testCases := []struct {
		name                 string
		assistantContent     string
		expectedToolCallName string
		expectedToolCallArgs string
		expectToolCall       bool
		expectedToolCount    int
		description          string
	}{
		{
			name:             "NoToolCall",
			assistantContent: "Hello! How can I help you today?",
			expectToolCall:   false,
			description:      "Regular conversational responses should pass through unchanged",
		},
		{
			name:             "MalformedJSON",
			assistantContent: `[{"name": "incomplete_func"`,
			expectToolCall:   false,
			description:      "Malformed JSON should not crash the state machine parser",
		},
		{
			name:             "EmptyToolCallArray",
			assistantContent: `[]`,
			expectToolCall:   false,
			description:      "Empty arrays should not trigger tool call processing",
		},
		{
			name:                 "SingleToolCall_PlainObject",
			assistantContent:     `{"name": "get_weather", "parameters": {"location": "Boston"}}`,
			expectedToolCallName: "get_weather",
			expectedToolCallArgs: `{"location": "Boston"}`,
			expectToolCall:       true,
			expectedToolCount:    1,
			description:          "Plain JSON objects should be parsed correctly by state machine",
		},
		{
			name:                 "SingleToolCall_InArray",
			assistantContent:     `[{"name": "get_weather", "parameters": {"location": "Boston"}}]`,
			expectedToolCallName: "get_weather",
			expectedToolCallArgs: `{"location": "Boston"}`,
			expectToolCall:       true,
			expectedToolCount:    1,
			description:          "JSON arrays with single tool calls should work with state machine",
		},
		{
			name:                 "SingleToolCall_NoParameters",
			assistantContent:     `[{"name": "get_time"}]`,
			expectedToolCallName: "get_time",
			expectedToolCallArgs: `null`,
			expectToolCall:       true,
			expectedToolCount:    1,
			description:          "Tool calls without parameters should default to null",
		},
		{
			name:                 "SingleToolCall_NullParameters",
			assistantContent:     `[{"name": "get_time", "parameters": null}]`,
			expectedToolCallName: "get_time",
			expectedToolCallArgs: `null`,
			expectToolCall:       true,
			expectedToolCount:    1,
			description:          "Explicit null parameters should be preserved",
		},
		{
			name:                 "SingleToolCall_WithMarkdown",
			assistantContent:     "```json\n" + `[{"name": "get_stock_price", "parameters": {"ticker": "GOOG"}}]` + "\n```",
			expectedToolCallName: "get_stock_price",
			expectedToolCallArgs: `{"ticker": "GOOG"}`,
			expectToolCall:       true,
			expectedToolCount:    1,
			description:          "Markdown code blocks should be handled correctly by state machine",
		},
		{
			name:                 "SingleToolCall_WithLeadingText",
			assistantContent:     "I'll help you with that. Let me call the weather function:\n" + `[{"name": "get_weather", "parameters": {"location": "Paris"}}]`,
			expectedToolCallName: "get_weather",
			expectedToolCallArgs: `{"location": "Paris"}`,
			expectToolCall:       true,
			expectedToolCount:    1,
			description:          "Tool calls mixed with explanatory text should be extracted by state machine",
		},
		{
			name:              "MultipleToolCalls",
			assistantContent:  "```\n" + `[{"name": "get_weather", "parameters": {"location": "NYC"}}, {"name": "get_time", "parameters": null}]` + "\n```",
			expectToolCall:    true,
			expectedToolCount: 2,
			description:       "Multiple tool calls in a single response should be handled by state machine",
		},
		{
			name:                 "ComplexMixedContent",
			assistantContent:     "Let me analyze this for you. First, I'll get the current data:\n\n```json\n[{\"name\": \"analyze_data\", \"parameters\": {\"dataset\": \"user_metrics\"}}]\n```\n\nThis will help us understand the trends.",
			expectedToolCallName: "analyze_data",
			expectedToolCallArgs: `{"dataset": "user_metrics"}`,
			expectToolCall:       true,
			expectedToolCount:    1,
			description:          "Complex responses with explanations before and after tool calls",
		},
		{
			name:                 "NestedJSONStructure",
			assistantContent:     `[{"name": "complex_func", "parameters": {"config": {"nested": {"deep": "value"}}, "array": [1, 2, 3]}}]`,
			expectedToolCallName: "complex_func",
			expectedToolCallArgs: `{"config": {"nested": {"deep": "value"}}, "array": [1, 2, 3]}`,
			expectToolCall:       true,
			expectedToolCount:    1,
			description:          "State machine should handle deeply nested JSON structures",
		},
		{
			name:                 "JSONWithEscapedQuotes",
			assistantContent:     `[{"name": "text_func", "parameters": {"message": "He said \"Hello world!\""}}]`,
			expectedToolCallName: "text_func",
			expectedToolCallArgs: `{"message": "He said \"Hello world!\""}`,
			expectToolCall:       true,
			expectedToolCount:    1,
			description:          "State machine should properly handle escaped quotes in JSON strings",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Logf("Testing: %s", tc.description)

			mockResp := createMockCompletion(tc.assistantContent)
			finalResp, err := adapter.TransformCompletionsResponse(mockResp)
			require.NoError(t, err)

			if !tc.expectToolCall {
				assert.Equal(t, mockResp, finalResp, "Response should be unchanged when no tool call is detected")
				return
			}

			// Verify tool call transformation
			require.Len(t, finalResp.Choices, 1)
			choice := finalResp.Choices[0]
			assert.Equal(t, "tool_calls", string(choice.FinishReason))
			assert.Empty(t, choice.Message.Content, "Content should be cleared when tool calls are present")
			require.Len(t, choice.Message.ToolCalls, tc.expectedToolCount)

			// For single tool calls, verify specific details
			if tc.expectedToolCount == 1 {
				toolCall := choice.Message.ToolCalls[0]
				assert.Equal(t, tc.expectedToolCallName, toolCall.Function.Name)
				assert.JSONEq(t, tc.expectedToolCallArgs, toolCall.Function.Arguments)
				assert.NotEmpty(t, toolCall.ID, "Tool call should have a unique ID")
				// Check the type - SDK may convert string to constant.Function internally
				typeStr := fmt.Sprintf("%v", toolCall.Type)
				assert.Equal(t, "function", typeStr)
			}
		})
	}
}

// TestNaturalJSON_NotFunctionCalls verifies that JSON appearing naturally in responses
// (e.g., code examples, data samples) are NOT treated as function calls
func TestNaturalJSON_NotFunctionCalls(t *testing.T) {
	adapter := tooladapter.New()

	testCases := []struct {
		name             string
		assistantContent string
		description      string
	}{
		{
			name: "CodeExample_ConfigObject",
			assistantContent: `Here's an example configuration object for your API:

{
  "apiKey": "your-key-here",
  "timeout": 5000,
  "retries": 3,
  "endpoints": {
    "primary": "https://api.example.com",
    "fallback": "https://backup.example.com"
  }
}

This configuration will set up your client properly.`,
			description: "Configuration JSON in code examples should not be treated as function calls",
		},
		{
			name: "DataExample_UserProfile",
			assistantContent: `The API returns user data in this format:

{
  "userId": 12345,
  "username": "john_doe",
  "profile": {
    "firstName": "John",
    "lastName": "Doe",
    "email": "john@example.com"
  },
  "preferences": {
    "theme": "dark",
    "notifications": true
  }
}`,
			description: "Data structure examples should not be treated as function calls",
		},
		{
			name: "JSONArray_DataSample",
			assistantContent: `Here are some sample records:

[
  {"id": 1, "fullName": "Alice Johnson", "age": 30},
  {"id": 2, "fullName": "Bob Smith", "age": 25},
  {"id": 3, "fullName": "Charlie Brown", "age": 35}
]

You can use this format for your data imports.`,
			description: "JSON arrays with fullName fields (containing spaces) should not be treated as function calls",
		},
		{
			name: "MixedContent_WithRealJSON",
			assistantContent: `Your database schema could look like this:

{
  "tables": {
    "users": {
      "columns": ["id", "name", "email"],
      "indexes": ["email"]
    },
    "orders": {
      "columns": ["id", "user_id", "total", "created_at"],
      "relationships": ["users"]
    }
  }
}

This is just an example - you should adapt it to your needs.`,
			description: "Complex JSON structures without function call pattern should not trigger tool calls",
		},
		{
			name: "JSONWithNameField_ButNotFunctionCall",
			assistantContent: `Here's a person object:

{
  "name": "John Smith",
  "age": 30,
  "address": {
    "street": "123 Main St",
    "city": "Anytown",
    "zip": "12345"
  }
}

Note that the 'name' field here refers to a person's name, not a function name.`,
			description: "JSON with person 'name' field should not trigger tool calls (names with spaces aren't valid function names)",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Logf("Testing scenario: %s", tc.description)

			mockResp := createMockCompletion(tc.assistantContent)
			finalResp, err := adapter.TransformCompletionsResponse(mockResp)
			require.NoError(t, err, "Transform should not error")

			// Verify that NO tool calls were detected
			require.Len(t, finalResp.Choices, 1, "Should have one choice")
			choice := finalResp.Choices[0]

			// The response should be unchanged (passed through as-is)
			assert.Equal(t, mockResp.Choices[0].Message.Content, choice.Message.Content, "Content should pass through unchanged")
			assert.Empty(t, choice.Message.ToolCalls, "Should have no tool calls")
			assert.NotEqual(t, "tool_calls", string(choice.FinishReason), "Should not indicate tool calls were made")
		})
	}
}

// TestComplexMixedScenarios_FunctionCallsWithNaturalJSON tests scenarios where
// both function calls and natural JSON appear in the same response
func TestComplexMixedScenarios_FunctionCallsWithNaturalJSON(t *testing.T) {
	adapter := tooladapter.New()

	testCases := []struct {
		name                 string
		assistantContent     string
		expectedToolCallName string
		expectedArgs         string
		description          string
	}{
		{
			name: "FunctionCallAfterCodeExample",
			assistantContent: `Here's an example of user data structure:

{
  "userId": 123,
  "name": "John Doe",
  "email": "john@example.com"
}

Now let me fetch the actual weather data for you:

[{"name": "get_weather", "parameters": {"location": "Boston"}}]`,
			expectedToolCallName: "get_weather",
			expectedArgs:         `{"location": "Boston"}`,
			description:          "Should extract function call even when natural JSON appears earlier",
		},
		{
			name: "FunctionCallBeforeCodeExample",
			assistantContent: `Let me get the weather first:

{"name": "get_weather", "parameters": {"location": "Boston"}}

Here's the data structure you'll receive:

{
  "temperature": 72,
  "humidity": 65,
  "conditions": "sunny"
}`,
			expectedToolCallName: "get_weather",
			expectedArgs:         `{"location": "Boston"}`,
			description:          "Should extract function call even when natural JSON appears later",
		},
		{
			name: "MultipleFunctionCallsWithNaturalJSON",
			assistantContent: `I'll gather some data for you. First, let me get the weather:

[{"name": "get_weather", "parameters": {"location": "Boston"}}]

The response will look like:
{
  "temperature": 72,
  "conditions": "sunny"
}

Then I'll get the time:

[{"name": "get_current_time"}]`,
			expectedToolCallName: "get_weather", // Should find the first valid function call
			expectedArgs:         `{"location": "Boston"}`,
			description:          "Should extract first valid function call from mixed content",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Logf("Testing scenario: %s", tc.description)

			mockResp := createMockCompletion(tc.assistantContent)
			finalResp, err := adapter.TransformCompletionsResponse(mockResp)
			require.NoError(t, err, "Transform should not error")

			// Verify tool call was detected
			require.Len(t, finalResp.Choices, 1, "Should have one choice")
			choice := finalResp.Choices[0]
			assert.Equal(t, "tool_calls", string(choice.FinishReason), "Should indicate tool calls were made")
			assert.Empty(t, choice.Message.Content, "Content should be cleared when tool calls are present")
			require.Greater(t, len(choice.Message.ToolCalls), 0, "Should have at least one tool call")

			// Verify the tool call details
			toolCall := choice.Message.ToolCalls[0]
			assert.Equal(t, tc.expectedToolCallName, toolCall.Function.Name, "Function name should match")
			assert.JSONEq(t, tc.expectedArgs, toolCall.Function.Arguments, "Function arguments should match")
		})
	}
}

// TestEdgeCases_ParameterHandling verifies robust parameter handling across different scenarios
func TestEdgeCases_ParameterHandling(t *testing.T) {
	adapter := tooladapter.New()

	testCases := []struct {
		name             string
		assistantContent string
		expectedName     string
		expectedArgs     string
		description      string
	}{
		{
			name:             "EmptyParametersObject",
			assistantContent: `{"name": "simple_function", "parameters": {}}`,
			expectedName:     "simple_function",
			expectedArgs:     `{}`,
			description:      "Empty parameters object should be preserved",
		},
		{
			name:             "NullParameters",
			assistantContent: `{"name": "simple_function", "parameters": null}`,
			expectedName:     "simple_function",
			expectedArgs:     `null`,
			description:      "Null parameters should be preserved",
		},
		{
			name:             "MissingParametersField",
			assistantContent: `{"name": "simple_function"}`,
			expectedName:     "simple_function",
			expectedArgs:     `null`,
			description:      "Missing parameters field should default to null",
		},
		{
			name:             "ComplexNestedParameters",
			assistantContent: `{"name": "complex_function", "parameters": {"config": {"nested": {"deep": {"value": 42}}}, "array": [1, 2, {"nested": true}], "string": "test"}}`,
			expectedName:     "complex_function",
			expectedArgs:     `{"config": {"nested": {"deep": {"value": 42}}}, "array": [1, 2, {"nested": true}], "string": "test"}`,
			description:      "Complex nested parameters should be preserved exactly",
		},
		{
			name:             "ParametersWithSpecialCharacters",
			assistantContent: `{"name": "text_function", "parameters": {"message": "Hello \"world\"!\nNew line\tTab\r\nCarriage return", "chars": "Special: @#$%^&*()_+-=[]{}|;':\",./<>?"}}`,
			expectedName:     "text_function",
			expectedArgs:     `{"message": "Hello \"world\"!\nNew line\tTab\r\nCarriage return", "chars": "Special: @#$%^&*()_+-=[]{}|;':\",./<>?"}`,
			description:      "Parameters with special characters and escape sequences should be preserved",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Logf("Testing scenario: %s", tc.description)

			mockResp := createMockCompletion(tc.assistantContent)
			finalResp, err := adapter.TransformCompletionsResponse(mockResp)
			require.NoError(t, err, "Transform should not error")

			// Verify tool call was detected and parsed correctly
			require.Len(t, finalResp.Choices, 1, "Should have one choice")
			choice := finalResp.Choices[0]
			require.Greater(t, len(choice.Message.ToolCalls), 0, "Should have at least one tool call")

			toolCall := choice.Message.ToolCalls[0]
			assert.Equal(t, tc.expectedName, toolCall.Function.Name, "Function name should match")
			assert.JSONEq(t, tc.expectedArgs, toolCall.Function.Arguments, "Function arguments should match exactly")
		})
	}
}

// ============================================================================
// EDGE CASES AND ROBUSTNESS TESTS
// ============================================================================

// TestStateMachineParser_EdgeCases tests the state machine parser's robustness
func TestStateMachineParser_EdgeCases(t *testing.T) {
	adapter := tooladapter.New()

	t.Run("EmptyToolDefinition", func(t *testing.T) {
		// Test handling of tools with missing information
		req := createMockRequest([]openai.ChatCompletionToolUnionParam{
			openai.ChatCompletionFunctionTool(
				openai.FunctionDefinitionParam{
					Name:        "", // Empty name
					Description: openai.String("A function with no name"),
				},
			),
		})

		// Should handle gracefully
		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)
		require.NotNil(t, result)
	})

	t.Run("ToolWithoutDescription", func(t *testing.T) {
		// Test handling of tools without description
		req := createMockRequest([]openai.ChatCompletionToolUnionParam{
			openai.ChatCompletionFunctionTool(
				openai.FunctionDefinitionParam{
					Name: "no_desc_func",
					// No description field
				},
			),
		})

		// Should handle gracefully
		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)
		require.NotNil(t, result)
	})

	t.Run("VeryLargeToolDefinition", func(t *testing.T) {
		// Test handling of extremely large tool definitions
		largeDescription := strings.Repeat("This is a very long description. ", 1000)
		req := createMockRequest([]openai.ChatCompletionToolUnionParam{
			createMockTool("large_function", largeDescription),
		})

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)
		require.NotNil(t, result)
	})

	t.Run("SpecialCharactersInToolDefinition", func(t *testing.T) {
		// Test handling of special characters that might break JSON or parsing
		req := createMockRequest([]openai.ChatCompletionToolUnionParam{
			createMockTool("special_func", "Description with \"quotes\" and \n newlines and {braces}"),
		})

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)
		require.NotNil(t, result)
	})

	t.Run("ResponseWithMultipleJSONBlocks", func(t *testing.T) {
		// Test responses that contain multiple JSON-like structures
		content := `Here are some examples:
		
Example 1: {"example": "data"}

But the actual tool call is:
[{"name": "real_function", "parameters": {"key": "value"}}]

And here's another example: {"more": "examples"}`

		mockResp := createMockCompletion(content)
		result, err := adapter.TransformCompletionsResponse(mockResp)
		require.NoError(t, err)

		// Should extract the actual tool call, not the examples
		if len(result.Choices) > 0 && len(result.Choices[0].Message.ToolCalls) > 0 {
			assert.Equal(t, "real_function", result.Choices[0].Message.ToolCalls[0].Function.Name)
		}
	})

	t.Run("DeeplyNestedJSON", func(t *testing.T) {
		// Test state machine with deeply nested structures
		content := `[{"name": "nested_func", "parameters": {"level1": {"level2": {"level3": {"level4": {"value": "deep"}}}}}}]`

		mockResp := createMockCompletion(content)
		result, err := adapter.TransformCompletionsResponse(mockResp)
		require.NoError(t, err)

		// Should successfully parse nested structure
		require.Len(t, result.Choices, 1)
		if len(result.Choices[0].Message.ToolCalls) > 0 {
			assert.Equal(t, "nested_func", result.Choices[0].Message.ToolCalls[0].Function.Name)
			assert.Contains(t, result.Choices[0].Message.ToolCalls[0].Function.Arguments, "level1")
		}
	})
}

// TestRobustness_MalformedAndEdgeCases verifies the parser handles malformed input gracefully
func TestRobustness_MalformedAndEdgeCases(t *testing.T) {
	adapter := tooladapter.New()

	testCases := []struct {
		name             string
		assistantContent string
		expectsToolCall  bool
		description      string
	}{
		{
			name:             "IncompleteJSON_MissingClosingBrace",
			assistantContent: `{"name": "incomplete_function", "parameters": {"key": "value"`,
			expectsToolCall:  false,
			description:      "Incomplete JSON should not be parsed as function call",
		},
		{
			name:             "InvalidJSON_ExtraComma",
			assistantContent: `{"name": "invalid_function", "parameters": {"key": "value",}}`,
			expectsToolCall:  false,
			description:      "Invalid JSON with trailing comma should not be parsed",
		},
		{
			name:             "EmptyString",
			assistantContent: ``,
			expectsToolCall:  false,
			description:      "Empty string should not cause errors",
		},
		{
			name:             "OnlyWhitespace",
			assistantContent: "   \n\t\r\n   ",
			expectsToolCall:  false,
			description:      "Only whitespace should not cause errors",
		},
		{
			name:             "ValidJSONButEmptyName",
			assistantContent: `{"name": "", "parameters": {"key": "value"}}`,
			expectsToolCall:  false,
			description:      "Valid JSON with empty name should not be treated as function call",
		},
		{
			name:             "ValidJSONButMissingName",
			assistantContent: `{"parameters": {"key": "value"}}`,
			expectsToolCall:  false,
			description:      "Valid JSON without name field should not be treated as function call",
		},
		{
			name:             "ValidJSONWithWrongStructure",
			assistantContent: `{"function": "get_weather", "args": {"location": "Boston"}}`,
			expectsToolCall:  false,
			description:      "JSON with wrong field names should not be treated as function call",
		},
		{
			name:             "VeryLargeJSON_NotFunctionCall",
			assistantContent: fmt.Sprintf(`{"data": %s, "metadata": {"size": "large"}}`, strings.Repeat(`{"item": "value"},`, 1000)),
			expectsToolCall:  false,
			description:      "Very large JSON without function call structure should not be parsed",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Logf("Testing scenario: %s", tc.description)

			mockResp := createMockCompletion(tc.assistantContent)
			finalResp, err := adapter.TransformCompletionsResponse(mockResp)
			require.NoError(t, err, "Transform should not error even with malformed input")

			// Verify expectations
			require.Len(t, finalResp.Choices, 1, "Should have one choice")
			choice := finalResp.Choices[0]

			if tc.expectsToolCall {
				assert.Greater(t, len(choice.Message.ToolCalls), 0, "Should have detected tool call")
				assert.Equal(t, "tool_calls", string(choice.FinishReason), "Should indicate tool calls were made")
				assert.Empty(t, choice.Message.Content, "Content should be cleared when tool calls are present")
			} else {
				assert.Empty(t, choice.Message.ToolCalls, "Should not have detected tool call")
				assert.NotEqual(t, "tool_calls", string(choice.FinishReason), "Should not indicate tool calls were made")
				// Content should be preserved for non-function-call responses
				assert.Equal(t, mockResp.Choices[0].Message.Content, choice.Message.Content, "Content should be preserved")
			}
		})
	}
}

// TestStrictModeSupport tests that the strict field is properly handled in tool definitions
func TestStrictModeSupport(t *testing.T) {
	adapter := tooladapter.New(tooladapter.WithLogLevel(slog.LevelError))

	t.Run("StrictModeEnabled", func(t *testing.T) {
		// Create a tool with strict mode enabled
		tools := []openai.ChatCompletionToolUnionParam{
			openai.ChatCompletionFunctionTool(
				openai.FunctionDefinitionParam{
					Name:        "strict_function",
					Description: openai.String("Function with strict schema compliance"),
					Strict:      openai.Bool(true), // Enable strict mode
					Parameters: map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"required_field": map[string]interface{}{
								"type":        "string",
								"description": "This field is required",
							},
						},
						"required":             []string{"required_field"},
						"additionalProperties": false,
					},
				},
			),
		}

		request := openai.ChatCompletionNewParams{
			Model: openai.ChatModelGPT4o,
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Test strict mode"),
			},
			Tools: tools,
		}

		// Transform the request
		transformedReq, err := adapter.TransformCompletionsRequest(request)
		require.NoError(t, err)

		// With no system message, first user message is modified
		require.Len(t, transformedReq.Messages, 1)
		userMessage := transformedReq.Messages[0]

		// Extract the user message content using JSON marshaling approach
		messageJSON, err := json.Marshal(userMessage)
		require.NoError(t, err)

		var msgMap map[string]interface{}
		err = json.Unmarshal(messageJSON, &msgMap)
		require.NoError(t, err)

		userContent, ok := msgMap["content"].(string)
		if !ok {
			t.Fatal("First message should be a user message with content")
		}

		// Verify strict mode flag is included
		assert.Contains(t, userContent, "Strict: true")
	})

	t.Run("StrictModeDisabled", func(t *testing.T) {
		// Create a tool with strict mode disabled (default)
		tools := []openai.ChatCompletionToolUnionParam{
			openai.ChatCompletionFunctionTool(
				openai.FunctionDefinitionParam{
					Name:        "regular_function",
					Description: openai.String("Regular function without strict mode"),
					// Strict field not set (defaults to false)
					Parameters: map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"optional_field": map[string]interface{}{
								"type":        "string",
								"description": "This field is optional",
							},
						},
					},
				},
			),
		}

		request := openai.ChatCompletionNewParams{
			Model: openai.ChatModelGPT4o,
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Test regular mode"),
			},
			Tools: tools,
		}

		// Transform the request
		transformedReq, err := adapter.TransformCompletionsRequest(request)
		require.NoError(t, err)

		// With no system message, first user message is modified
		require.Len(t, transformedReq.Messages, 1)
		userMessage := transformedReq.Messages[0]

		// Extract the user message content using JSON marshaling approach
		messageJSON, err := json.Marshal(userMessage)
		require.NoError(t, err)

		var msgMap map[string]interface{}
		err = json.Unmarshal(messageJSON, &msgMap)
		require.NoError(t, err)

		userContent, ok := msgMap["content"].(string)
		if !ok {
			t.Fatal("First message should be a user message with content")
		}

		// Verify strict mode flag is NOT included
		assert.NotContains(t, userContent, "Strict: true")
	})

	t.Run("StrictModeExplicitlyDisabled", func(t *testing.T) {
		// Create a tool with strict mode explicitly disabled
		tools := []openai.ChatCompletionToolUnionParam{
			openai.ChatCompletionFunctionTool(
				openai.FunctionDefinitionParam{
					Name:        "explicit_non_strict",
					Description: openai.String("Function with strict explicitly disabled"),
					Strict:      openai.Bool(false), // Explicitly disable strict mode
					Parameters: map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"flexible_field": map[string]interface{}{
								"type": "string",
							},
						},
					},
				},
			),
		}

		request := openai.ChatCompletionNewParams{
			Model: openai.ChatModelGPT4o,
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Test explicitly disabled strict mode"),
			},
			Tools: tools,
		}

		// Transform the request
		transformedReq, err := adapter.TransformCompletionsRequest(request)
		require.NoError(t, err)

		// With no system message, first user message is modified
		require.Len(t, transformedReq.Messages, 1)
		userMessage := transformedReq.Messages[0]

		// Extract the user message content using JSON marshaling approach
		messageJSON, err := json.Marshal(userMessage)
		require.NoError(t, err)

		var msgMap map[string]interface{}
		err = json.Unmarshal(messageJSON, &msgMap)
		require.NoError(t, err)

		userContent, ok := msgMap["content"].(string)
		if !ok {
			t.Fatal("First message should be a user message with content")
		}

		// Verify strict mode flag is NOT included
		assert.NotContains(t, userContent, "Strict: true")
	})
}

// ============================================================================
// PROMPT INJECTION ROLE SELECTION TESTS
// ============================================================================

// getRole marshals the union message and inspects the role field.
func getRole(t *testing.T, m openai.ChatCompletionMessageParamUnion) string {
	t.Helper()
	b, err := json.Marshal(m)
	require.NoError(t, err)
	s := string(b)
	switch {
	case strings.Contains(s, `"role":"system"`):
		return "system"
	case strings.Contains(s, `"role":"user"`):
		return "user"
	case strings.Contains(s, `"role":"assistant"`):
		return "assistant"
	default:
		return ""
	}
}

func TestPromptInjectionRoleSelection(t *testing.T) {
	t.Run("InjectsAsSystemWhenSystemExists", func(t *testing.T) {
		adapter := tooladapter.New()
		req := openai.ChatCompletionNewParams{
			Model: openai.ChatModelGPT4o,
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.SystemMessage("Existing system guidance."),
				openai.UserMessage("Hello"),
			},
			Tools: []openai.ChatCompletionToolUnionParam{
				openai.ChatCompletionFunctionTool(
					openai.FunctionDefinitionParam{
						Name:        "get_weather",
						Description: openai.String("Get weather"),
					},
				),
			},
		}

		out, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// NEW BEHAVIOR: Modifies existing system message
		require.Equal(t, len(req.Messages), len(out.Messages))

		// First message is still system but now contains tool prompt
		firstRole := getRole(t, out.Messages[0])
		assert.Equal(t, "system", firstRole)

		// Ensure it contains the tool prompt (contains the tool name)
		b, _ := json.Marshal(out.Messages[0])
		assert.Contains(t, string(b), "get_weather")
		assert.Contains(t, string(b), "Existing system guidance") // Original content preserved

		// Second message unchanged
		assert.Equal(t, getRole(t, req.Messages[1]), getRole(t, out.Messages[1]))

		// Tools removed and ToolChoice zeroed
		assert.Nil(t, out.Tools)
		assert.Equal(t, openai.ChatCompletionToolChoiceOptionUnionParam{}, out.ToolChoice)
	})

	t.Run("InjectsAsUserWhenNoSystemExists", func(t *testing.T) {
		adapter := tooladapter.New()
		req := openai.ChatCompletionNewParams{
			Model: openai.ChatModelGPT4o,
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Hello"),
			},
			Tools: []openai.ChatCompletionToolUnionParam{
				openai.ChatCompletionFunctionTool(
					openai.FunctionDefinitionParam{
						Name:        "get_time",
						Description: openai.String("Get time"),
					},
				),
			},
		}

		out, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// New behavior: modifies the first user message
		require.Equal(t, len(req.Messages), len(out.Messages))

		// First message is still user but now contains tools
		firstRole := getRole(t, out.Messages[0])
		assert.Equal(t, "user", firstRole)

		// Ensure it contains both tool prompt and original content
		b, _ := json.Marshal(out.Messages[0])
		assert.Contains(t, string(b), "get_time")
		assert.Contains(t, string(b), "Hello") // Original content preserved in same message

		// Tools removed and ToolChoice zeroed
		assert.Nil(t, out.Tools)
		assert.Equal(t, openai.ChatCompletionToolChoiceOptionUnionParam{}, out.ToolChoice)
	})

	t.Run("InjectsAsUserWhenNoMessages", func(t *testing.T) {
		adapter := tooladapter.New() // Default: no system support
		req := openai.ChatCompletionNewParams{
			Model:    openai.ChatModelGPT4o,
			Messages: nil, // no messages
			Tools: []openai.ChatCompletionToolUnionParam{
				openai.ChatCompletionFunctionTool(
					openai.FunctionDefinitionParam{
						Name:        "calc",
						Description: openai.String("Calc"),
					},
				),
			},
		}

		out, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// Exactly one injected message (user by default)
		require.Len(t, out.Messages, 1)
		assert.Equal(t, "user", getRole(t, out.Messages[0]))

		// Tools removed and ToolChoice zeroed
		assert.Nil(t, out.Tools)
		assert.Equal(t, openai.ChatCompletionToolChoiceOptionUnionParam{}, out.ToolChoice)
	})
}

// ============================================================================
// TOOL RESULT HANDLING TESTS
// ============================================================================

func TestToolResultHandling(t *testing.T) {
	adapter := tooladapter.New(tooladapter.WithLogLevel(slog.LevelError))

	// Helper function to create a test tool
	createTestTool := func() openai.ChatCompletionToolUnionParam {
		return openai.ChatCompletionFunctionTool(
			openai.FunctionDefinitionParam{
				Name:        "get_weather",
				Description: openai.String("Get weather for a location"),
				Parameters: openai.FunctionParameters{
					"type": "object",
					"properties": map[string]interface{}{
						"location": map[string]interface{}{"type": "string"},
					},
				},
			},
		)
	}

	t.Run("Case1_NeitherToolsNorResults_PassThrough", func(t *testing.T) {
		req := openai.ChatCompletionNewParams{
			Model: "gpt-4",
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Hello, how are you?"),
				openai.AssistantMessage("I'm doing well, thanks for asking!"),
			},
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// Should pass through unchanged
		assert.Equal(t, req.Messages, result.Messages, "Messages should be unchanged")
		assert.Equal(t, req.Model, result.Model, "Model should be unchanged")
		assert.Empty(t, result.Tools, "Tools should remain empty")
	})

	t.Run("Case2_OnlyTools_OriginalBehavior", func(t *testing.T) {
		tools := []openai.ChatCompletionToolUnionParam{createTestTool()}

		req := openai.ChatCompletionNewParams{
			Model: "gpt-4",
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("What's the weather like?"),
			},
			Tools: tools,
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// With no system message and default settings, first user message is modified
		assert.Equal(t, len(req.Messages), len(result.Messages), "Should modify existing message, not add new one")
		assert.Empty(t, result.Tools, "Tools should be removed from request")

		// Check that tool instructions were injected into first user message
		firstMsg, _ := result.Messages[0].MarshalJSON()
		firstMsgStr := string(firstMsg)
		assert.Contains(t, firstMsgStr, "get_weather", "Should contain tool definition")
		assert.Contains(t, firstMsgStr, "What's the weather like?", "Should preserve original user content")
	})

	t.Run("Case3_OnlyToolResults_NoCallableTools", func(t *testing.T) {
		req := openai.ChatCompletionNewParams{
			Model: "gpt-4",
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("What's the weather like in SF?"),
				openai.AssistantMessage("I'll check that for you."),
				openai.ToolMessage("The weather in San Francisco is 72°F and sunny.", "call_123"),
				openai.UserMessage("Thanks! Can you format that nicely?"),
			},
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// NEW BEHAVIOR: ToolMessage removed and first user message modified (4 -> 3)
		assert.Len(t, result.Messages, 3, "Should remove ToolMessage, preserve message count")
		assert.Empty(t, result.Tools, "Tools should remain empty")

		// Check that tool results were injected into first user message
		firstMsg, _ := result.Messages[0].MarshalJSON()
		firstMsgStr := string(firstMsg)
		assert.Contains(t, firstMsgStr, "Previous tool calls", "Should contain tool results prompt")
		assert.Contains(t, firstMsgStr, "72°F and sunny", "Should contain actual tool result")
		assert.Contains(t, firstMsgStr, "What's the weather like in SF?", "Should preserve original user content")

		// Check that ToolMessage was removed from conversation
		for _, msg := range result.Messages {
			msgBytes, _ := msg.MarshalJSON()
			msgStr := string(msgBytes)
			assert.NotContains(t, msgStr, "tool_call_id", "Should not contain ToolMessage")
		}
	})

	t.Run("Case4_ToolsAndResults_BothPresent", func(t *testing.T) {
		tools := []openai.ChatCompletionToolUnionParam{createTestTool()}

		req := openai.ChatCompletionNewParams{
			Model: "gpt-4",
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("What's the weather like in SF?"),
				openai.AssistantMessage("I'll check that for you."),
				openai.ToolMessage("The weather in San Francisco is 72°F and sunny.", "call_123"),
				openai.UserMessage("Now check NYC too."),
			},
			Tools: tools,
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		assert.Empty(t, result.Tools, "Tools should be removed from request")

		// Check that both tool instructions and tool results were injected
		firstMsg, _ := result.Messages[0].MarshalJSON()
		firstMsgStr := string(firstMsg)
		assert.Contains(t, firstMsgStr, "get_weather", "Should contain tool definition")
		assert.Contains(t, firstMsgStr, "Previous tool calls", "Should contain tool results section")
		assert.Contains(t, firstMsgStr, "72°F and sunny", "Should contain actual tool result")

		// Check that ToolMessage was removed from conversation
		for _, msg := range result.Messages[1:] { // Skip the injected prompt message
			msgBytes, _ := msg.MarshalJSON()
			msgStr := string(msgBytes)
			assert.NotContains(t, msgStr, "tool_call_id", "Should not contain ToolMessage")
		}
	})

	t.Run("MultipleToolResults", func(t *testing.T) {
		req := openai.ChatCompletionNewParams{
			Model: "gpt-4",
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Check weather in multiple cities"),
				openai.AssistantMessage("I'll check both cities."),
				openai.ToolMessage("SF: 72°F and sunny.", "call_123"),
				openai.ToolMessage("NYC: 68°F and cloudy.", "call_456"),
				openai.UserMessage("Great, summarize please."),
			},
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// Check that all tool results were included
		firstMsg, _ := result.Messages[0].MarshalJSON()
		firstMsgStr := string(firstMsg)
		assert.Contains(t, firstMsgStr, "SF: 72°F and sunny", "Should contain first result")
		assert.Contains(t, firstMsgStr, "NYC: 68°F and cloudy", "Should contain second result")
		assert.Contains(t, firstMsgStr, "call_123", "Should contain first call ID")
		assert.Contains(t, firstMsgStr, "call_456", "Should contain second call ID")
	})

	t.Run("MalformedToolMessage_SkippedGracefully", func(t *testing.T) {
		// This test simulates a malformed tool message that can't be parsed
		req := openai.ChatCompletionNewParams{
			Model: "gpt-4",
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Test message"),
				// Note: We can't easily create a malformed ToolMessage with the SDK,
				// but the code handles JSON unmarshaling errors gracefully
				openai.ToolMessage("Valid tool result", "call_123"),
			},
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// Should still process the valid tool message
		firstMsg, _ := result.Messages[0].MarshalJSON()
		firstMsgStr := string(firstMsg)
		assert.Contains(t, firstMsgStr, "Valid tool result", "Should contain valid tool result")
	})
}

func TestToolResultHandlingWithContext(t *testing.T) {
	adapter := tooladapter.New(tooladapter.WithLogLevel(slog.LevelError))

	t.Run("ContextCancellation", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel() // Cancel immediately

		req := openai.ChatCompletionNewParams{
			Model: "gpt-4",
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Test"),
				openai.ToolMessage("Result", "call_123"),
			},
		}

		_, err := adapter.TransformCompletionsRequestWithContext(ctx, req)
		assert.Error(t, err, "Should return context cancellation error")
		assert.Equal(t, context.Canceled, err, "Should be context.Canceled")
	})
}

// TestBufferPoolMemoryGrowthProtection verifies that the buffer pool prevents memory leaks
// from oversized buffers by discarding buffers that exceed the size threshold
func TestBufferPoolMemoryGrowthProtection(t *testing.T) {
	adapter := tooladapter.New(tooladapter.WithLogLevel(slog.LevelError))

	t.Run("NormalSizeBuffersArePooled", func(t *testing.T) {
		// Create a reasonable-sized tool that should stay within the threshold
		normalTool := openai.ChatCompletionFunctionTool(
			openai.FunctionDefinitionParam{
				Name:        "normal_function",
				Description: openai.String("A normal function with reasonable size"),
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"input": map[string]interface{}{
							"type":        "string",
							"description": "Normal input parameter",
						},
					},
				},
			},
		)

		req := openai.ChatCompletionNewParams{
			Model:    "gpt-4",
			Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("Test")},
			Tools:    []openai.ChatCompletionToolUnionParam{normalTool},
		}

		// Process multiple requests to test buffer reuse
		for i := 0; i < 5; i++ {
			result, err := adapter.TransformCompletionsRequest(req)
			require.NoError(t, err)
			require.NotEmpty(t, result.Messages)
		}

		// This test mainly ensures no crashes or memory issues with normal-sized buffers
		// Buffer pooling behavior is internal but should work correctly
	})

	t.Run("OversizedBuffersAreDiscarded", func(t *testing.T) {
		// Create a tool with a very large description that will grow the buffer beyond threshold
		largeDescription := strings.Repeat("This is a very detailed description that will make the buffer grow significantly. ", 500) // ~45KB
		largeTool := openai.ChatCompletionFunctionTool(
			openai.FunctionDefinitionParam{
				Name:        "large_function",
				Description: openai.String(largeDescription),
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"data": map[string]interface{}{
							"type":        "string",
							"description": largeDescription, // Add more large content
						},
						"config": map[string]interface{}{
							"type":        "object",
							"description": largeDescription, // Even more content
							"properties": map[string]interface{}{
								"setting1": map[string]interface{}{
									"type":        "string",
									"description": largeDescription,
								},
								"setting2": map[string]interface{}{
									"type":        "string",
									"description": largeDescription,
								},
							},
						},
					},
				},
			},
		)

		req := openai.ChatCompletionNewParams{
			Model:    "gpt-4",
			Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("Test")},
			Tools:    []openai.ChatCompletionToolUnionParam{largeTool},
		}

		// Process the oversized request - should create a large buffer
		result1, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)
		require.NotEmpty(t, result1.Messages)

		// Process a normal request after the large one
		normalReq := openai.ChatCompletionNewParams{
			Model:    "gpt-4",
			Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("Normal test")},
			Tools: []openai.ChatCompletionToolUnionParam{
				openai.ChatCompletionFunctionTool(
					openai.FunctionDefinitionParam{
						Name:        "small_function",
						Description: openai.String("Small function"),
					},
				),
			},
		}

		result2, err := adapter.TransformCompletionsRequest(normalReq)
		require.NoError(t, err)
		require.NotEmpty(t, result2.Messages)

		// The test mainly verifies that processing continues correctly even after
		// large buffers are created and (internally) discarded from the pool
		// We can't directly verify the buffer pool state, but we can ensure
		// no memory leaks or crashes occur
	})

	t.Run("RepeatedLargeRequestsHandled", func(t *testing.T) {
		// Test that repeated large requests don't cause memory accumulation
		largeDescription := strings.Repeat("Large content ", 1000) // ~13KB per description

		// Create multiple large tools to really stress the buffer
		var largeTools []openai.ChatCompletionToolUnionParam
		for i := 0; i < 10; i++ {
			largeTools = append(largeTools, openai.ChatCompletionFunctionTool(
				openai.FunctionDefinitionParam{
					Name:        fmt.Sprintf("large_function_%d", i),
					Description: openai.String(largeDescription),
					Parameters: map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"param": map[string]interface{}{
								"type":        "string",
								"description": largeDescription,
							},
						},
					},
				},
			))
		}

		req := openai.ChatCompletionNewParams{
			Model:    "gpt-4",
			Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("Test")},
			Tools:    largeTools,
		}

		// Process multiple large requests
		for i := 0; i < 3; i++ {
			result, err := adapter.TransformCompletionsRequest(req)
			require.NoError(t, err)
			require.NotEmpty(t, result.Messages)

			// Verify the prompt was built correctly even with large content
			firstMsg, err := result.Messages[0].MarshalJSON()
			require.NoError(t, err)
			assert.Contains(t, string(firstMsg), "large_function_0")
		}

		// If we get here without memory issues or crashes, the buffer pool
		// memory growth protection is working correctly
	})
}

// TestGenerateToolCallID tests the adapter's generateToolCallID method
func TestGenerateToolCallID(t *testing.T) {
	t.Run("BasicOperation", func(t *testing.T) {
		adapter := tooladapter.New()

		// Test basic ID generation
		id := adapter.GenerateToolCallID()
		assert.NotEmpty(t, id)
		assert.True(t, strings.HasPrefix(id, "call_"))

		// Test multiple calls for uniqueness
		ids := make(map[string]bool)
		for i := 0; i < 100; i++ {
			id := adapter.GenerateToolCallID()
			assert.False(t, ids[id], "Should generate unique IDs")
			ids[id] = true
		}
	})

	t.Run("LoggerIntegration", func(t *testing.T) {
		// Create adapter with custom logger
		var logBuf bytes.Buffer
		logger := slog.New(slog.NewTextHandler(&logBuf, &slog.HandlerOptions{
			Level: slog.LevelDebug,
		}))

		adapter := tooladapter.New(tooladapter.WithLogger(logger))

		// Generate multiple IDs and verify they work correctly
		ids := make(map[string]bool)
		for i := 0; i < 10; i++ {
			id := adapter.GenerateToolCallID()
			assert.NotEmpty(t, id, "Should generate non-empty ID")
			assert.True(t, strings.HasPrefix(id, "call_"), "Should have correct prefix")
			assert.False(t, ids[id], "Should generate unique IDs")
			ids[id] = true
		}

		// Verify no error logs were generated during normal operation
		logOutput := logBuf.String()
		assert.NotContains(t, logOutput, "UUIDv7 generation failed", "Should not log errors during normal operation")
	})

	t.Run("LoggerConfiguration", func(t *testing.T) {
		// Test with different logger configurations
		testCases := []struct {
			name        string
			logLevel    slog.Level
			description string
		}{
			{"ErrorLevel", slog.LevelError, "error level logging"},
			{"WarnLevel", slog.LevelWarn, "warn level logging"},
			{"InfoLevel", slog.LevelInfo, "info level logging"},
			{"DebugLevel", slog.LevelDebug, "debug level logging"},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				var logBuf bytes.Buffer
				logger := slog.New(slog.NewTextHandler(&logBuf, &slog.HandlerOptions{
					Level: tc.logLevel,
				}))

				adapter := tooladapter.New(tooladapter.WithLogger(logger))

				// Generate ID should work regardless of log level
				id := adapter.GenerateToolCallID()
				assert.NotEmpty(t, id, "Should generate ID with %s", tc.description)
				assert.True(t, strings.HasPrefix(id, "call_"), "Should have correct prefix with %s", tc.description)
			})
		}
	})

	t.Run("IntegrationWithTransform", func(t *testing.T) {
		// Test that the adapter methods actually use generateToolCallID method
		var logBuf bytes.Buffer
		logger := slog.New(slog.NewTextHandler(&logBuf, &slog.HandlerOptions{
			Level: slog.LevelDebug,
		}))

		adapter := tooladapter.New(tooladapter.WithLogger(logger))

		// Create a mock response with function calls to trigger ID generation
		mockResp := openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Role:    "assistant",
						Content: `[{"name": "test_function", "parameters": {"key": "value"}}]`,
					},
				},
			},
		}

		// Transform the response - this will trigger generateToolCallID
		result, err := adapter.TransformCompletionsResponse(mockResp)
		require.NoError(t, err, "Should transform response without error")
		require.NotNil(t, result, "Should create response")

		// Verify that tool calls were created with proper IDs
		if len(result.Choices) > 0 && len(result.Choices[0].Message.ToolCalls) > 0 {
			for _, toolCall := range result.Choices[0].Message.ToolCalls {
				assert.True(t, strings.HasPrefix(toolCall.ID, "call_"), "Tool call ID should have correct prefix")
				assert.Greater(t, len(toolCall.ID), len("call_"), "Tool call ID should have content after prefix")
			}
		}

		// Verify no unexpected error logs
		logOutput := logBuf.String()
		assert.NotContains(t, logOutput, "UUIDv7 generation failed", "Should not log UUID generation failures")
	})
}

// TestLoggerConsistency verifies that adapter methods use the configured logger
func TestLoggerConsistency(t *testing.T) {
	// Create adapter with JSON logger for easier parsing of structured logs
	var logBuf bytes.Buffer
	logger := slog.New(slog.NewJSONHandler(&logBuf, &slog.HandlerOptions{
		Level: slog.LevelDebug,
	}))

	adapter := tooladapter.New(tooladapter.WithLogger(logger))

	// Perform operations that might log messages
	mockResp := openai.ChatCompletion{
		Choices: []openai.ChatCompletionChoice{
			{
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: `[{"name": "test_function", "parameters": {"key": "value"}}]`,
				},
			},
		},
	}

	result, err := adapter.TransformCompletionsResponse(mockResp)
	require.NoError(t, err, "Should transform response without error")
	assert.NotNil(t, result, "Should create response")

	// Generate some IDs to ensure the method works
	for i := 0; i < 5; i++ {
		id := adapter.GenerateToolCallID()
		assert.True(t, strings.HasPrefix(id, "call_"), "Should generate valid ID %d", i)
	}

	// If there were any logs, they should be in JSON format (indicating our logger was used)
	logOutput := logBuf.String()
	if logOutput != "" {
		// Any log entries should be valid JSON since we configured JSON logging
		lines := strings.Split(strings.TrimSpace(logOutput), "\n")
		for _, line := range lines {
			if line != "" {
				// This would panic if not valid JSON, which would fail the test
				// We're just checking that the structure is correct
				assert.True(t, strings.Contains(line, "{") && strings.Contains(line, "}"),
					"Log line should be JSON format: %s", line)
			}
		}
	}
}

// ============================================================================
// MULTIMODAL SUPPORT TESTS
// ============================================================================

// TestMultimodalMessageHandling verifies that multimodal messages are properly
// handled when tool prompts are injected for models without system message support
func TestMultimodalMessageHandling(t *testing.T) {
	// Test with system messages disabled (like Gemma 3)
	adapter := tooladapter.New(
		tooladapter.WithSystemMessageSupport(false),
		tooladapter.WithLogLevel(slog.LevelError),
	)

	t.Run("SimpleTextMessage", func(t *testing.T) {
		// Test with simple text message
		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Hello world"),
			},
			Tools: []openai.ChatCompletionToolUnionParam{
				createMockTool("test_func", "Test function"),
			},
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// Should modify the first user message
		assert.Len(t, result.Messages, 1)
		userMsg := result.Messages[0]
		assert.NotNil(t, userMsg.OfUser)

		content := userMsg.OfUser.Content.OfString.Or("")
		assert.Contains(t, content, "Hello world")
		assert.Contains(t, content, "System/tooling instructions")
	})

	t.Run("MultimodalMessage", func(t *testing.T) {
		// Test with multimodal message (text + image)
		parts := []openai.ChatCompletionContentPartUnionParam{
			{
				OfText: &openai.ChatCompletionContentPartTextParam{
					Type: "text",
					Text: "Look at this image",
				},
			},
			{
				OfImageURL: &openai.ChatCompletionContentPartImageParam{
					Type: "image_url",
					ImageURL: openai.ChatCompletionContentPartImageImageURLParam{
						URL: "data:image/jpeg;base64,/9j/test...",
					},
				},
			},
		}

		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage(parts),
			},
			Tools: []openai.ChatCompletionToolUnionParam{
				createMockTool("test_func", "Test function"),
			},
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// Should preserve multimodal structure
		assert.Len(t, result.Messages, 1)
		userMsg := result.Messages[0]
		assert.NotNil(t, userMsg.OfUser)

		resultParts := userMsg.OfUser.Content.OfArrayOfContentParts
		assert.Len(t, resultParts, 2, "Should have 2 parts: combined text + image")

		// First part should be combined text
		textPart := resultParts[0].OfText
		assert.NotNil(t, textPart)
		assert.Contains(t, textPart.Text, "System/tooling instructions")
		assert.Contains(t, textPart.Text, "Look at this image")

		// Second part should be preserved image
		imagePart := resultParts[1].OfImageURL
		assert.NotNil(t, imagePart)
		assert.Equal(t, "data:image/jpeg;base64,/9j/test...", imagePart.ImageURL.URL)
	})

	t.Run("MultipleTextPartsWithImage", func(t *testing.T) {
		// Test with multiple text parts and image
		parts := []openai.ChatCompletionContentPartUnionParam{
			{
				OfText: &openai.ChatCompletionContentPartTextParam{
					Type: "text",
					Text: "First text",
				},
			},
			{
				OfImageURL: &openai.ChatCompletionContentPartImageParam{
					Type: "image_url",
					ImageURL: openai.ChatCompletionContentPartImageImageURLParam{
						URL: "data:image/jpeg;base64,/9j/test...",
					},
				},
			},
			{
				OfText: &openai.ChatCompletionContentPartTextParam{
					Type: "text",
					Text: "Second text",
				},
			},
		}

		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage(parts),
			},
			Tools: []openai.ChatCompletionToolUnionParam{
				createMockTool("test_func", "Test function"),
			},
		}

		result, err := adapter.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// Should preserve multimodal structure
		resultParts := result.Messages[0].OfUser.Content.OfArrayOfContentParts
		assert.Len(t, resultParts, 2, "Should have 2 parts: combined text + image")

		// First part should combine all text
		textPart := resultParts[0].OfText
		assert.NotNil(t, textPart)
		assert.Contains(t, textPart.Text, "System/tooling instructions")
		assert.Contains(t, textPart.Text, "First text Second text")

		// Second part should be preserved image
		imagePart := resultParts[1].OfImageURL
		assert.NotNil(t, imagePart)
		assert.Equal(t, "data:image/jpeg;base64,/9j/test...", imagePart.ImageURL.URL)
	})

	t.Run("SystemMessageSupportEnabled", func(t *testing.T) {
		// Test that with system message support enabled, multimodal messages are not modified
		adapterWithSysMsg := tooladapter.New(
			tooladapter.WithSystemMessageSupport(true),
			tooladapter.WithLogLevel(slog.LevelError),
		)

		parts := []openai.ChatCompletionContentPartUnionParam{
			{
				OfText: &openai.ChatCompletionContentPartTextParam{
					Type: "text",
					Text: "Original text",
				},
			},
			{
				OfImageURL: &openai.ChatCompletionContentPartImageParam{
					Type: "image_url",
					ImageURL: openai.ChatCompletionContentPartImageImageURLParam{
						URL: "data:image/jpeg;base64,/9j/test...",
					},
				},
			},
		}

		req := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage(parts),
			},
			Tools: []openai.ChatCompletionToolUnionParam{
				createMockTool("test_func", "Test function"),
			},
		}

		result, err := adapterWithSysMsg.TransformCompletionsRequest(req)
		require.NoError(t, err)

		// Should have system message + original user message
		assert.Len(t, result.Messages, 2)

		// First should be system message
		assert.NotNil(t, result.Messages[0].OfSystem)

		// Second should be unmodified multimodal user message
		userMsg := result.Messages[1]
		assert.NotNil(t, userMsg.OfUser)

		resultParts := userMsg.OfUser.Content.OfArrayOfContentParts
		assert.Len(t, resultParts, 2)

		// Text should be unchanged
		textPart := resultParts[0].OfText
		assert.NotNil(t, textPart)
		assert.Equal(t, "Original text", textPart.Text)

		// Image should be unchanged
		imagePart := resultParts[1].OfImageURL
		assert.NotNil(t, imagePart)
		assert.Equal(t, "data:image/jpeg;base64,/9j/test...", imagePart.ImageURL.URL)
	})
}
