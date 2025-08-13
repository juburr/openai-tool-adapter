package tooladapter

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestWithStreamingEarlyDetection tests the streaming early detection feature
// that improves buffering heuristics for catching tool calls that appear after
// explanatory text (addresses issue: "Streaming buffering heuristics can miss
// JSON that starts after some text").
func TestWithStreamingEarlyDetection(t *testing.T) {
	t.Run("DisabledByDefault", func(t *testing.T) {
		adapter := New() // Default behavior - early detection disabled
		stream := &StreamAdapter{adapter: adapter}

		// These should NOT be caught with default settings (early detection disabled)
		prefaceTexts := []string{
			`Let me call {"name": "get_weather"`,            // JSON at pos 12
			`Sure thing! I can help. [{"name": "calculate"`, // JSON at pos 28
			`I'll help you with that. {"name": "search"`,    // JSON at pos 26
		}

		for _, content := range prefaceTexts {
			result := stream.shouldStartBuffering(content)
			assert.False(t, result, "Should NOT catch with early detection disabled: %s", content)
		}
	})

	t.Run("EnabledWith80CharLimit", func(t *testing.T) {
		adapter := New(WithStreamingEarlyDetection(80))
		stream := &StreamAdapter{adapter: adapter}

		testCases := []struct {
			content     string
			expectCatch bool
			description string
		}{
			{
				content:     `Let me call {"name": "get_weather"`,
				expectCatch: true,
				description: "Short preface within 80 chars should be caught",
			},
			{
				content:     `Sure thing! I can help with that calculation. [{"name": "calculate"`,
				expectCatch: true,
				description: "Medium preface within 80 chars should be caught",
			},
			{
				content:     `I understand your request and I'll help you with a comprehensive analysis. {"name": "search"`,
				expectCatch: false,
				description: "Long preface exceeding 80 chars should NOT be caught",
			},
			{
				content:     `Here's information about John. His name is John and he works at {"company": "TechCorp"}`,
				expectCatch: false,
				description: "Non-tool JSON should NOT be caught (wrong structure)",
			},
			{
				content:     `Just some regular text without any tool calls in it.`,
				expectCatch: false,
				description: "Plain text should NOT be caught",
			},
		}

		for _, tc := range testCases {
			result := stream.shouldStartBuffering(tc.content)
			assert.Equal(t, tc.expectCatch, result, tc.description)

			jsonPos := findToolJSONStart(tc.content)
			t.Logf("Test: %s", tc.description)
			t.Logf("  JSON at pos: %d, Caught: %t, Expected: %t", jsonPos, result, tc.expectCatch)
		}
	})

	t.Run("EnabledWith120CharLimit", func(t *testing.T) {
		adapter := New(WithStreamingEarlyDetection(120))
		stream := &StreamAdapter{adapter: adapter}

		// This should be caught with the more generous 120-char limit
		longPrefaceContent := `I understand your request and I'll help you get that information. {"name": "get_weather"`
		result := stream.shouldStartBuffering(longPrefaceContent)
		assert.True(t, result, "Should catch longer preface with 120-char limit")

		jsonPos := findToolJSONStart(longPrefaceContent)
		t.Logf("JSON at position %d, caught with 120-char limit: %t", jsonPos, result)
	})

	t.Run("ImmediatePatternsStillWork", func(t *testing.T) {
		adapter := New(WithStreamingEarlyDetection(80))
		stream := &StreamAdapter{adapter: adapter}

		// These should still work regardless of early detection setting
		immediatePatterns := []string{
			`{"name": "get_weather"`,
			`[{"name": "calculate"`,
			"```json\n{\"name\": \"search\"",
			"```\n[{\"name\": \"translate\"",
			"Here's the call: `{\"name\": \"format\"`",
		}

		for _, content := range immediatePatterns {
			result := stream.shouldStartBuffering(content)
			assert.True(t, result, "Immediate pattern should always be caught: %s", content)
		}
	})

	t.Run("ConfigurableLimit", func(t *testing.T) {
		testContent := `I'll help you with that weather check. {"name": "get_weather"`
		jsonPos := findToolJSONStart(testContent) // Should be position 39
		t.Logf("Test content: %s", testContent)
		t.Logf("JSON position: %d, content length: %d", jsonPos, len(testContent))

		// We need the limit to be larger than jsonPos + pattern length to include the full pattern
		// Pattern {"name": is 8 characters, so position 39 needs limit > 46
		testLimits := []struct {
			limit       int
			expectCatch bool
		}{
			{30, false}, // Too small, won't reach JSON start (pos 39)
			{46, false}, // Reaches JSON start but not full pattern
			{47, true},  // Just enough to include full pattern {"name":
			{80, true},  // Well above needed
			{0, false},  // Disabled
		}

		for _, test := range testLimits {
			adapter := New(WithStreamingEarlyDetection(test.limit))
			stream := &StreamAdapter{adapter: adapter}

			result := stream.shouldStartBuffering(testContent)
			assert.Equal(t, test.expectCatch, result,
				"Limit %d should %s catch JSON at position %d",
				test.limit, map[bool]string{true: "", false: "NOT"}[test.expectCatch], jsonPos)
		}
	})
}

// findToolJSONStart finds the position where tool call JSON starts in content
func findToolJSONStart(content string) int {
	patterns := []string{
		`{"name":`,
		`{"name": `,
		`[{"name":`,
		`[{"name": `,
	}

	for _, pattern := range patterns {
		if pos := strings.Index(content, pattern); pos >= 0 {
			return pos
		}
	}
	return -1
}

// TestStreamingEarlyDetectionIntegration tests the feature with tool policies
func TestStreamingEarlyDetectionIntegration(t *testing.T) {
	t.Run("BenefitsStopOnFirstPolicy", func(t *testing.T) {
		// This feature is most beneficial for ToolStopOnFirst and ToolCollectThenStop
		// policies that suppress content when tool calls are detected
		adapter := New(
			WithStreamingEarlyDetection(100),
			WithToolPolicy(ToolStopOnFirst),
		)

		// Verify the configuration is applied
		assert.Equal(t, 100, adapter.streamLookAheadLimit)
		assert.Equal(t, ToolStopOnFirst, adapter.toolPolicy)

		stream := &StreamAdapter{adapter: adapter}

		// With early detection, this preface + tool call scenario should trigger buffering
		prefaceWithTool := `I'll check the weather for you. {"name": "get_weather"`
		result := stream.shouldStartBuffering(prefaceWithTool)
		assert.True(t, result, "Early detection should help ToolStopOnFirst catch preface scenarios")
	})

	t.Run("WorksWithAllPolicies", func(t *testing.T) {
		policies := []ToolPolicy{
			ToolStopOnFirst,
			ToolCollectThenStop,
			ToolDrainAll,
			ToolAllowMixed,
		}

		content := `Let me help you with that. {"name": "test_function"`

		for _, policy := range policies {
			adapter := New(
				WithStreamingEarlyDetection(80),
				WithToolPolicy(policy),
			)
			stream := &StreamAdapter{adapter: adapter}

			result := stream.shouldStartBuffering(content)
			assert.True(t, result, "Early detection should work with %s policy", policy.String())
		}
	})
}
