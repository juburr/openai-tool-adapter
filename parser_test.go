package tooladapter

import (
	"fmt"
	"strings"
	"testing"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestJSONExtractor_AllEnclosureFormats comprehensively tests all JSON presentation formats
func TestJSONExtractor_AllEnclosureFormats(t *testing.T) {
	testCases := []struct {
		name     string
		input    string
		expected []string
	}{
		// Plain JSON
		{
			name:     "PlainJSON_Object",
			input:    `{"name": "test_func", "parameters": {"key": "value"}}`,
			expected: []string{`{"name": "test_func", "parameters": {"key": "value"}}`},
		},
		{
			name:     "PlainJSON_Array",
			input:    `[{"name": "test_func", "parameters": {"key": "value"}}]`,
			expected: []string{`[{"name": "test_func", "parameters": {"key": "value"}}]`},
		},

		// Single backticks (inline code)
		{
			name:     "SingleTicks_Object",
			input:    "Function call: `{\"name\": \"test_func\", \"parameters\": {\"key\": \"value\"}}`",
			expected: []string{`{"name": "test_func", "parameters": {"key": "value"}}`},
		},
		{
			name:     "SingleTicks_Array",
			input:    "Function call: `[{\"name\": \"test_func\", \"parameters\": {\"key\": \"value\"}}]`",
			expected: []string{`[{"name": "test_func", "parameters": {"key": "value"}}]`},
		},
		{
			name:     "SingleTicks_Multiple",
			input:    "First: `{\"name\": \"func1\"}` and second: `{\"name\": \"func2\"}`",
			expected: []string{`{"name": "func1"}`, `{"name": "func2"}`},
		},

		// Triple backticks without language
		{
			name:     "TripleTicks_Object",
			input:    "```\n{\"name\": \"test_func\", \"parameters\": {\"key\": \"value\"}}\n```",
			expected: []string{`{"name": "test_func", "parameters": {"key": "value"}}`},
		},
		{
			name:     "TripleTicks_Array",
			input:    "```\n[{\"name\": \"test_func\", \"parameters\": {\"key\": \"value\"}}]\n```",
			expected: []string{`[{"name": "test_func", "parameters": {"key": "value"}}]`},
		},

		// Triple backticks with 'json' language specifier
		{
			name:     "TripleTicksJSON_Object",
			input:    "```json\n{\"name\": \"test_func\", \"parameters\": {\"key\": \"value\"}}\n```",
			expected: []string{`{"name": "test_func", "parameters": {"key": "value"}}`},
		},
		{
			name:     "TripleTicksJSON_Array",
			input:    "```json\n[{\"name\": \"test_func\", \"parameters\": {\"key\": \"value\"}}]\n```",
			expected: []string{`[{"name": "test_func", "parameters": {"key": "value"}}]`},
		},

		// Mixed formats in single input
		{
			name:     "MixedFormats_AllTypes",
			input:    "Plain: {\"name\": \"func1\"} and ticks: `{\"name\": \"func2\"}` and block:\n```json\n{\"name\": \"func3\"}\n```",
			expected: []string{`{"name": "func1"}`, `{"name": "func2"}`, `{"name": "func3"}`},
		},

		// Edge cases with whitespace and formatting
		{
			name:     "WhitespaceVariations_TripleTicks",
			input:    "```json\n  {\"name\": \"test_func\"}  \n```",
			expected: []string{`{"name": "test_func"}`},
		},
		{
			name:     "NoNewlines_TripleTicks",
			input:    "```{\"name\": \"test_func\"}```",
			expected: []string{`{"name": "test_func"}`},
		},

		// More edge cases
		{
			name:     "Deduplication",
			input:    "`{\"name\": \"dedupe\"}` some text {\"name\": \"dedupe\"}",
			expected: []string{`{"name": "dedupe"}`},
		},
		{
			name:     "EmptyInput",
			input:    "",
			expected: []string{},
		},
		{
			name:     "WhitespaceOnlyInput",
			input:    " \t\n\r ",
			expected: []string{},
		},
		{
			name:     "DanglingSingleBacktick",
			input:    "Some text `{\"name\": \"incomplete\"}",
			expected: []string{},
		},
		{
			name:     "DanglingTripleBacktick",
			input:    "Some text ```{\"name\": \"incomplete\"}",
			expected: []string{},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			extractor := NewJSONExtractor(tc.input)
			result := extractor.ExtractJSONBlocks()
			// Use ElementsMatch to assert that the results contain the expected elements,
			// regardless of the order in which the parser finds them.
			assert.ElementsMatch(t, tc.expected, result, "Extracted blocks should match expected")
		})
	}
}

// TestJSONExtractor_ParameterHandling tests various parameter scenarios
func TestJSONExtractor_ParameterHandling(t *testing.T) {
	testCases := []struct {
		name  string
		input string
	}{
		{name: "EmptyParameters", input: `{"name": "test_func", "parameters": {}}`},
		{name: "NullParameters", input: `{"name": "test_func", "parameters": null}`},
		{name: "MissingParameters", input: `{"name": "test_func"}`},
		{name: "ComplexNestedParameters", input: `{"name": "test_func", "parameters": {"config": {"nested": {"deep": {"value": 42}}}, "array": [1, 2, {"nested": true}]}}`},
		{name: "ParametersWithEscapedQuotes", input: `{"name": "test_func", "parameters": {"message": "He said \"Hello world!\""}}`},
		{name: "ParametersWithSpecialCharacters", input: `{"name": "test_func", "parameters": {"text": "Line1\nLine2\tTabbed\r\nCRLF", "symbols": "@#$%^&*()"}}`},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			extractor := NewJSONExtractor(tc.input)
			result := extractor.ExtractJSONBlocks()
			require.Len(t, result, 1, "Should extract exactly one JSON block")
			assert.Equal(t, tc.input, result[0], "Extracted JSON should match expected")
		})
	}
}

// TestExtractFunctionCalls_ComprehensiveValidation tests function call validation across scenarios
func TestExtractFunctionCalls_ComprehensiveValidation(t *testing.T) {
	testCases := []struct {
		name          string
		candidates    []string
		expectedCount int
		expectedNames []string
		description   string
	}{
		{
			name:          "ValidSingleCall_Object",
			candidates:    []string{`{"name": "get_weather", "parameters": {"location": "Boston"}}`},
			expectedCount: 1,
			expectedNames: []string{"get_weather"},
			description:   "Single valid function call as object",
		},
		{
			name:          "ValidSingleCall_Array",
			candidates:    []string{`[{"name": "get_weather", "parameters": {"location": "Boston"}}]`},
			expectedCount: 1,
			expectedNames: []string{"get_weather"},
			description:   "Single valid function call in array",
		},
		{
			name:          "ValidMultipleCalls_Array",
			candidates:    []string{`[{"name": "get_weather", "parameters": {"location": "Boston"}}, {"name": "get_time", "parameters": null}]`},
			expectedCount: 2,
			expectedNames: []string{"get_weather", "get_time"},
			description:   "Multiple valid function calls in array",
		},
		{
			name:          "ValidCall_NoParameters",
			candidates:    []string{`{"name": "get_current_time"}`},
			expectedCount: 1,
			expectedNames: []string{"get_current_time"},
			description:   "Valid function call without parameters field",
		},
		{
			name:          "ValidCall_EmptyParameters",
			candidates:    []string{`{"name": "get_current_time", "parameters": {}}`},
			expectedCount: 1,
			expectedNames: []string{"get_current_time"},
			description:   "Valid function call with empty parameters",
		},
		{
			name:          "InvalidJSON_MalformedSyntax",
			candidates:    []string{`{"name": "broken_func", "parameters"`},
			expectedCount: 0,
			description:   "Malformed JSON should not be parsed",
		},
		{
			name:          "EmptyArray",
			candidates:    []string{`[]`},
			expectedCount: 0,
			description:   "Empty array should not produce function calls",
		},
		{
			name:          "MissingName_Field",
			candidates:    []string{`{"parameters": {"key": "value"}}`},
			expectedCount: 0,
			description:   "JSON without name field should not be function call",
		},
		{
			name:          "EmptyName_Field",
			candidates:    []string{`{"name": "", "parameters": {"key": "value"}}`},
			expectedCount: 0,
			description:   "JSON with empty name should not be function call",
		},
		{
			name:          "WrongStructure_DifferentFields",
			candidates:    []string{`{"function": "get_weather", "args": {"location": "Boston"}}`},
			expectedCount: 0,
			description:   "JSON with wrong field names should not be function call",
		},
		{
			name:          "NaturalJSON_PersonData",
			candidates:    []string{`{"name": "John Smith", "age": 30, "city": "Boston"}`},
			expectedCount: 0,
			description:   "Person data with 'name' field should not be function call (name has space)",
		},
		{
			name:          "NaturalJSON_ParameterData",
			candidates:    []string{`{"parameters": ["age", "location"], "dataset": "numpy"}`},
			expectedCount: 0,
			description:   "Parameter data with 'parameters' field should not be function call (no name field)",
		},
		{
			name:          "NaturalJSON_CoincidentalLargeStructureMatch",
			candidates:    []string{`{"moreSugar": true, "moreEspresso": true, "name": "get_current_time", "parameters": {}, "location": "starbuckeroos"}`},
			expectedCount: 0,
			description:   "Simulates a large JSON structure that coincidentally contains 'name' and 'parameters' but is not a function call",
		},
		{
			name:          "NaturalJSON_ConfigData",
			candidates:    []string{`{"apiKey": "secret", "timeout": 5000, "retries": 3}`},
			expectedCount: 0,
			description:   "Configuration data should not be function call",
		},
		{
			name:          "NaturalJSON_ArrayOfObjects",
			candidates:    []string{`[{"id": 1, "fullName": "Alice Smith"}, {"id": 2, "fullName": "Bob Jones"}]`},
			expectedCount: 0,
			description:   "Array of data objects should not be function calls (fullName contains spaces, clearly not a function)",
		},
		{
			name:          "MultipleCandidates_FirstValid",
			candidates:    []string{`{"name": "get_weather", "parameters": {"location": "Boston"}}`, `{"invalid": "data"}`},
			expectedCount: 1,
			expectedNames: []string{"get_weather"},
			description:   "Should find valid function call among multiple candidates",
		},
		{
			name:          "MultipleCandidates_InvalidFirst",
			candidates:    []string{`{"invalid": "data"}`, `{"name": "get_weather", "parameters": {"location": "Boston"}}`},
			expectedCount: 1,
			expectedNames: []string{"get_weather"},
			description:   "Should find valid function call after an invalid candidate",
		},
		{
			name:          "MultipleCandidates_NoneValid",
			candidates:    []string{`{"invalid": "data1"}`, `{"invalid": "data2"}`},
			expectedCount: 0,
			description:   "Should return empty when no candidates are valid function calls",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Logf("Testing: %s", tc.description)
			result := ExtractFunctionCalls(tc.candidates)
			assert.Equal(t, tc.expectedCount, len(result), "Number of function calls should match")
			for i, expectedName := range tc.expectedNames {
				if i < len(result) {
					assert.Equal(t, expectedName, result[i].Name, "Function name %d should match", i)
				}
			}
		})
	}
}

// TestExtractFunctionCalls_ValidationEdgeCases tests comprehensive validation edge cases
// that ensure the parser correctly rejects malformed or invalid function call JSON.
// This test was added to prevent future regressions in validation logic.
func TestExtractFunctionCalls_ValidationEdgeCases(t *testing.T) {
	testCases := []struct {
		name        string
		candidates  []string
		description string
	}{
		{
			name:        "EmptyName_String",
			candidates:  []string{`{"name": "", "parameters": null}`},
			description: "Empty string name should be rejected",
		},
		{
			name:        "WhitespaceOnlyName_Space",
			candidates:  []string{`{"name": " ", "parameters": null}`},
			description: "Single space name should be rejected",
		},
		{
			name:        "WhitespaceOnlyName_MultipleSpaces",
			candidates:  []string{`{"name": "   ", "parameters": null}`},
			description: "Multiple spaces name should be rejected",
		},
		{
			name:        "WhitespaceOnlyName_Tab",
			candidates:  []string{`{"name": "\t", "parameters": null}`},
			description: "Tab character name should be rejected",
		},
		{
			name:        "WhitespaceOnlyName_Newline",
			candidates:  []string{`{"name": "\n", "parameters": null}`},
			description: "Newline character name should be rejected",
		},
		{
			name:        "WhitespaceOnlyName_Mixed",
			candidates:  []string{`{"name": " \t \n ", "parameters": null}`},
			description: "Mixed whitespace characters name should be rejected",
		},
		{
			name:        "MissingNameField",
			candidates:  []string{`{"parameters": null}`},
			description: "Missing name field should be rejected (unmarshals to empty string)",
		},
		{
			name:        "NullNameField",
			candidates:  []string{`{"name": null, "parameters": null}`},
			description: "Null name field should be rejected (unmarshals to empty string)",
		},
		{
			name:        "ExtraFields_WithValidName",
			candidates:  []string{`{"name": "valid_func", "parameters": null, "extra": "field"}`},
			description: "Valid name but extra fields should be rejected by DisallowUnknownFields",
		},
		{
			name:        "EmptyArray",
			candidates:  []string{`[]`},
			description: "Empty array should be rejected",
		},
		{
			name:        "ArrayWithEmptyName",
			candidates:  []string{`[{"name": "", "parameters": null}]`},
			description: "Array with empty name should be rejected",
		},
		{
			name:        "ArrayWithWhitespaceName",
			candidates:  []string{`[{"name": "   ", "parameters": null}]`},
			description: "Array with whitespace-only name should be rejected",
		},
		{
			name:        "InvalidStructure_WrongFields",
			candidates:  []string{`{"function": "test", "args": {}}`},
			description: "Wrong field names should be rejected",
		},
		{
			name:        "ValidStructure_ButInvalidName",
			candidates:  []string{`{"name": "invalid name with spaces", "parameters": null}`},
			description: "Correct structure but invalid name should be rejected",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Logf("Testing: %s", tc.description)
			result := ExtractFunctionCalls(tc.candidates)
			assert.Equal(t, 0, len(result), "Should reject invalid function call JSON")
		})
	}
}

// TestHasCompleteJSON_StreamingScenarios tests the streaming helper function
func TestHasCompleteJSON_StreamingScenarios(t *testing.T) {
	testCases := []struct {
		name     string
		content  string
		expected bool
	}{
		{
			name:     "CompleteToolCall_Array",
			content:  `[{"name": "test_func", "parameters": {"key": "value"}}]`,
			expected: true,
		},
		{
			name:     "CompleteToolCall_Object",
			content:  `{"name": "test_func", "parameters": {"key": "value"}}`,
			expected: true,
		},
		{
			name:     "CompleteToolCall_NoParameters",
			content:  `{"name": "test_func"}`,
			expected: true,
		},
		{
			name:     "CompleteToolCall_EscapedQuotes",
			content:  `[{"name": "text_func", "parameters": {"message": "He said \"Hello world!\"", "count": 5}}]`,
			expected: true,
		},
		{
			name:     "IncompleteJSON_MissingCloseBrace",
			content:  `[{"name": "test_func", "parameters": {`,
			expected: false,
		},
		{
			name:     "IncompleteJSON_MissingCloseBracket",
			content:  `[{"name": "test_func"}`,
			expected: false,
		},
		{
			name:     "IncompleteJSON_UnbalancedQuotes",
			content:  `{"name": "test_func", "parameters": {"message": "incomplete quote}`,
			expected: false,
		},
		{
			name:     "CompleteJSON_ButNotFunctionCall",
			content:  `{"config": "value", "timeout": 5000}`,
			expected: false,
		},
		{
			name:     "CompleteJSON_PersonData",
			content:  `{"name": "John Smith", "age": 30}`,
			expected: false,
		},
		{
			name:     "CompleteJSON_EmptyName",
			content:  `{"name": "", "parameters": {"key": "value"}}`,
			expected: false,
		},
		{
			name:     "EmptyContent",
			content:  "",
			expected: false,
		},
		{
			name:     "PlainText",
			content:  "This is just plain text",
			expected: false,
		},
		{
			name:     "WhitespaceOnly",
			content:  "   \n\t  ",
			expected: false,
		},
		{
			name:     "ComplexNested_Complete",
			content:  `[{"name": "complex_func", "parameters": {"config": {"nested": {"deep": {"value": "test"}}}, "array": [1, 2, 3]}}]`,
			expected: true,
		},
		{
			name:     "ComplexNested_Incomplete",
			content:  `[{"name": "complex_func", "parameters": {"config": {"nested": {"deep": {"value": "test"}`,
			expected: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := HasCompleteJSON(tc.content)
			assert.Equal(t, tc.expected, result, "HasCompleteJSON result should match expected")
		})
	}
}

// TestJSONExtractor_EdgeCases tests edge cases and error conditions
func TestJSONExtractor_EdgeCases(t *testing.T) {
	t.Run("VeryNestedStructure", func(t *testing.T) {
		input := `{"level1": {"level2": {"level3": {"level4": {"level5": "deep"}}}}}`
		extractor := NewJSONExtractor(input)
		result := extractor.ExtractJSONBlocks()

		require.Len(t, result, 1)
		assert.Contains(t, result[0], "level5")
	})

	t.Run("MultipleJSONBlocks", func(t *testing.T) {
		input := `{"first": "block"} some text {"second": "block"} more text [{"third": "block"}]`
		extractor := NewJSONExtractor(input)
		result := extractor.ExtractJSONBlocks()

		assert.GreaterOrEqual(t, len(result), 2, "Should find multiple JSON blocks")
	})

	t.Run("UnbalancedBraces", func(t *testing.T) {
		input := `{"unbalanced": "json"`
		extractor := NewJSONExtractor(input)
		result := extractor.ExtractJSONBlocks()

		// Should not crash and should not return incomplete JSON
		assert.Equal(t, 0, len(result), "Should not return incomplete JSON")
	})

	t.Run("ComplexEscaping", func(t *testing.T) {
		input := `{"message": "Text with \"quotes\" and \\backslashes\\ and \nnewlines"}`
		extractor := NewJSONExtractor(input)
		result := extractor.ExtractJSONBlocks()

		require.Len(t, result, 1)
		assert.Contains(t, result[0], "quotes")
		assert.Contains(t, result[0], "backslashes")
	})

	t.Run("SingleTicksExtraction", func(t *testing.T) {
		input := "Here's the call: `{\"name\": \"test_func\", \"parameters\": {\"key\": \"value\"}}`"
		extractor := NewJSONExtractor(input)
		result := extractor.ExtractJSONBlocks()

		require.Len(t, result, 1)
		assert.Equal(t, `{"name": "test_func", "parameters": {"key": "value"}}`, result[0])
	})

	t.Run("VeryLargeJSON", func(t *testing.T) {
		// Create a large JSON structure
		largeArray := strings.Repeat(`{"id": 123, "data": "value"}, `, 1000)
		input := fmt.Sprintf(`{"items": [%s], "count": 1000}`, strings.TrimSuffix(largeArray, ", "))

		extractor := NewJSONExtractor(input)
		result := extractor.ExtractJSONBlocks()

		require.Len(t, result, 1, "Should handle large JSON without issues")
		assert.Contains(t, result[0], "count")
		assert.Greater(t, len(result[0]), 10000, "Should preserve large JSON structure")
	})

	t.Run("MixedMarkdownAndPlainJSON", func(t *testing.T) {
		input := `Plain JSON: {"name": "func1"} and markdown:
		
` + "```json\n{\"name\": \"func2\", \"parameters\": {\"key\": \"value\"}}\n```" + `

And more plain: {"name": "func3"}`

		extractor := NewJSONExtractor(input)
		result := extractor.ExtractJSONBlocks()

		assert.GreaterOrEqual(t, len(result), 3, "Should find all JSON blocks")

		// Check that we found the expected function names
		foundFunctions := make(map[string]bool)
		for _, block := range result {
			if strings.Contains(block, "func1") {
				foundFunctions["func1"] = true
			}
			if strings.Contains(block, "func2") {
				foundFunctions["func2"] = true
			}
			if strings.Contains(block, "func3") {
				foundFunctions["func3"] = true
			}
		}

		assert.True(t, foundFunctions["func1"], "Should find func1")
		assert.True(t, foundFunctions["func2"], "Should find func2")
		assert.True(t, foundFunctions["func3"], "Should find func3")
	})
}

// TestStreamingEscapeQuotePattern tests the specific pattern that fails in streaming
func TestStreamingEscapeQuotePattern(t *testing.T) {
	// Test the exact pattern that's failing in streaming
	content := `[{"name": "text_func", "parameters": {"message": "He said \"Hello world!\"", "count": 5}}]`
	result := HasCompleteJSON(content)
	assert.True(t, result, "Should recognize streaming escape quote pattern as complete JSON")

	// Also test extraction
	extractor := NewJSONExtractor(content)
	candidates := extractor.ExtractJSONBlocks()
	require.Len(t, candidates, 1, "Should extract one JSON block")

	calls := ExtractFunctionCalls(candidates)
	require.Len(t, calls, 1, "Should extract one function call")
	assert.Equal(t, "text_func", calls[0].Name)
}

// TestValidateFunctionName_OpenAISpec tests function name validation according to OpenAI specification
func TestValidateFunctionName_OpenAISpec(t *testing.T) {
	testCases := []struct {
		name         string
		functionName string
		expectValid  bool
		description  string
	}{
		// Valid standard function names
		{
			name:         "ValidName_Letters",
			functionName: "get_weather",
			expectValid:  true,
			description:  "Function name with letters and underscore should be valid",
		},
		{
			name:         "ValidName_Numbers",
			functionName: "function123",
			expectValid:  true,
			description:  "Function name with letters and numbers should be valid",
		},
		{
			name:         "ValidName_Hyphens",
			functionName: "get-current-time",
			expectValid:  true,
			description:  "Function name with hyphens should be valid",
		},
		{
			name:         "ValidName_Underscores",
			functionName: "calculate_tax_rate",
			expectValid:  true,
			description:  "Function name with underscores should be valid",
		},
		{
			name:         "ValidName_Mixed",
			functionName: "func_123-test",
			expectValid:  true,
			description:  "Function name with mixed valid characters should be valid",
		},
		{
			name:         "ValidName_SingleChar",
			functionName: "a",
			expectValid:  true,
			description:  "Single character function name should be valid",
		},
		{
			name:         "ValidName_MaxLength",
			functionName: strings.Repeat("a", 64),
			expectValid:  true,
			description:  "Function name at max length (64 chars) should be valid",
		},
		{
			name:         "ValidName_CaseSensitive",
			functionName: "Get_Weather",
			expectValid:  true,
			description:  "Mixed case function names should be valid (case sensitive)",
		},

		// Invalid standard function names
		{
			name:         "InvalidName_Empty",
			functionName: "",
			expectValid:  false,
			description:  "Empty function name should be invalid",
		},
		{
			name:         "InvalidName_TooLong",
			functionName: strings.Repeat("a", 65),
			expectValid:  false,
			description:  "Function name longer than 64 chars should be invalid",
		},
		{
			name:         "InvalidName_Spaces",
			functionName: "get weather",
			expectValid:  false,
			description:  "Function name with spaces should be invalid",
		},
		{
			name:         "InvalidName_SpecialChars",
			functionName: "get_weather!",
			expectValid:  false,
			description:  "Function name with special characters should be invalid",
		},
		{
			name:         "ValidName_MCPFormat",
			functionName: "get.weather",
			expectValid:  true,
			description:  "MCP prefixed function name with period should be valid",
		},
		{
			name:         "InvalidName_AtSymbol",
			functionName: "get@weather",
			expectValid:  false,
			description:  "Function name with @ symbol should be invalid",
		},
		{
			name:         "InvalidName_MultiplePeriods",
			functionName: "server.sub.function",
			expectValid:  false,
			description:  "Function name with multiple periods should be invalid",
		},
		{
			name:         "InvalidName_Hash",
			functionName: "get#weather",
			expectValid:  false,
			description:  "Function name with # symbol should be invalid",
		},
		{
			name:         "InvalidName_Parentheses",
			functionName: "get_weather()",
			expectValid:  false,
			description:  "Function name with parentheses should be invalid",
		},

		// Valid MCP prefixed function names
		{
			name:         "ValidMCP_Basic",
			functionName: "weather.get_current",
			expectValid:  true,
			description:  "Basic MCP prefixed function should be valid",
		},
		{
			name:         "ValidMCP_Numbers",
			functionName: "server123.function456",
			expectValid:  true,
			description:  "MCP prefix with numbers should be valid",
		},
		{
			name:         "ValidMCP_LongPrefix",
			functionName: strings.Repeat("a", 32) + ".get_weather",
			expectValid:  true,
			description:  "MCP prefix should be valid within total limit",
		},
		{
			name:         "ValidMCP_LongFunction",
			functionName: "server." + strings.Repeat("a", 50),
			expectValid:  true,
			description:  "MCP function name should be valid within total limit",
		},
		{
			name:         "ValidMCP_MixedCase",
			functionName: "WeatherServer.Get_Current_Conditions",
			expectValid:  true,
			description:  "Mixed case MCP names should be valid",
		},

		// Invalid MCP prefixed function names
		{
			name:         "InvalidMCP_EmptyPrefix",
			functionName: ".get_weather",
			expectValid:  false,
			description:  "Empty MCP prefix should be invalid",
		},
		{
			name:         "InvalidMCP_EmptyFunction",
			functionName: "server.",
			expectValid:  false,
			description:  "Empty function name in MCP format should be invalid",
		},
		{
			name:         "InvalidMCP_PrefixTooLong",
			functionName: strings.Repeat("a", 65) + ".get_weather",
			expectValid:  false,
			description:  "MCP prefix longer than 64 chars should be invalid",
		},
		{
			name:         "InvalidMCP_FunctionTooLong",
			functionName: "server." + strings.Repeat("a", 65),
			expectValid:  false,
			description:  "MCP function name longer than 64 chars should be invalid",
		},
		{
			name:         "InvalidMCP_MultiplePeriods",
			functionName: "server.sub.get_weather",
			expectValid:  false,
			description:  "Multiple periods in MCP name should be invalid",
		},
		{
			name:         "InvalidMCP_PrefixWithUnderscore",
			functionName: "weather_server.get_current",
			expectValid:  false,
			description:  "MCP prefix with underscore should be invalid",
		},
		{
			name:         "InvalidMCP_PrefixWithHyphen",
			functionName: "weather-server.get_current",
			expectValid:  false,
			description:  "MCP prefix with hyphen should be invalid",
		},
		{
			name:         "InvalidMCP_PrefixWithSpecialChar",
			functionName: "weather@server.get_current",
			expectValid:  false,
			description:  "MCP prefix with special characters should be invalid",
		},
		{
			name:         "InvalidMCP_FunctionWithInvalidChar",
			functionName: "server.get_weather!",
			expectValid:  false,
			description:  "MCP function name with invalid characters should be invalid",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateFunctionName(tc.functionName)
			if tc.expectValid {
				assert.NoError(t, err, "Function name %q should be valid", tc.functionName)
			} else {
				assert.Error(t, err, "Function name %q should be invalid", tc.functionName)
			}
		})
	}
}

// TestValidateFunctionName_ErrorMessages tests that validation errors provide helpful messages
func TestValidateFunctionName_ErrorMessages(t *testing.T) {
	testCases := []struct {
		name          string
		functionName  string
		expectedError string
	}{
		{
			name:          "EmptyName",
			functionName:  "",
			expectedError: "function name validation failed: name cannot be empty",
		},
		{
			name:          "TooLong",
			functionName:  strings.Repeat("a", 65),
			expectedError: "name \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\" is 65 characters long but maximum allowed is 64",
		},
		{
			name:          "InvalidChars",
			functionName:  "get@weather",
			expectedError: "name \"get@weather\" contains invalid characters, must match pattern ^[a-zA-Z0-9_-]{1,64}$",
		},
		{
			name:          "MultiplePeriods",
			functionName:  "server.sub.function",
			expectedError: "name \"server.sub.function\" contains 2 periods but only one is allowed for MCP server prefixes",
		},
		{
			name:          "EmptyMCPPrefix",
			functionName:  ".function",
			expectedError: "MCP server prefix cannot be empty in \".function\"",
		},
		{
			name:          "EmptyMCPFunction",
			functionName:  "server.",
			expectedError: "function name part cannot be empty in \"server.\"",
		},
		{
			name:          "MCPTotalTooLong",
			functionName:  strings.Repeat("a", 32) + "." + strings.Repeat("b", 33),
			expectedError: "MCP format name \"" + strings.Repeat("a", 32) + "." + strings.Repeat("b", 33) + "\" is 66 characters long but maximum allowed is 64",
		},
		{
			name:          "MCPPrefixTooLong",
			functionName:  strings.Repeat("a", 65) + ".f",
			expectedError: "MCP format name \"" + strings.Repeat("a", 65) + ".f\" is 67 characters long but maximum allowed is 64",
		},
		{
			name:          "MCPPrefixInvalidChars",
			functionName:  "server_name.function",
			expectedError: "MCP server prefix \"server_name\" contains invalid characters, must only contain letters and numbers (a-zA-Z0-9)",
		},
		{
			name:          "MCPFunctionInvalidChars",
			functionName:  "server.function@name",
			expectedError: "function name part \"function@name\" contains invalid characters, must match pattern ^[a-zA-Z0-9_-]{1,64}$",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateFunctionName(tc.functionName)
			assert.Error(t, err, "Function name should be invalid")
			assert.Contains(t, err.Error(), tc.expectedError, "Error message should contain expected text")
		})
	}
}

// TestValidateFunctionCall_Integration tests that ValidateFunctionCall uses the new validation
func TestValidateFunctionCall_Integration(t *testing.T) {
	testCases := []struct {
		name         string
		functionCall functionCall
		expectValid  bool
		description  string
	}{
		{
			name: "ValidStandardCall",
			functionCall: functionCall{
				Name:       "get_weather",
				Parameters: []byte(`{"location": "Boston"}`),
			},
			expectValid: true,
			description: "Standard function call should be valid",
		},
		{
			name: "ValidMCPCall",
			functionCall: functionCall{
				Name:       "weather.get_current",
				Parameters: []byte(`{"location": "NYC"}`),
			},
			expectValid: true,
			description: "MCP prefixed function call should be valid",
		},
		{
			name: "InvalidNameWithSpaces",
			functionCall: functionCall{
				Name:       "get weather",
				Parameters: []byte(`{"location": "Boston"}`),
			},
			expectValid: false,
			description: "Function call with spaces in name should be invalid",
		},
		{
			name: "InvalidMCPMultiplePeriods",
			functionCall: functionCall{
				Name:       "server.sub.function",
				Parameters: []byte(`{}`),
			},
			expectValid: false,
			description: "Function call with multiple periods should be invalid",
		},
		{
			name: "ValidCaseSensitive",
			functionCall: functionCall{
				Name:       "Get_Weather",
				Parameters: []byte(`{"location": "LA"}`),
			},
			expectValid: true,
			description: "Case sensitive function names should be valid",
		},
		{
			name: "EmptyName",
			functionCall: functionCall{
				Name:       "",
				Parameters: []byte(`{}`),
			},
			expectValid: false,
			description: "Empty function name should be invalid",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Logf("Testing: %s", tc.description)

			result := ValidateFunctionCall(tc.functionCall)

			if tc.expectValid {
				assert.True(t, result, "Function call should be valid")
			} else {
				assert.False(t, result, "Function call should be invalid")
			}
		})
	}
}

// TestValidationRegexPatterns tests the regex patterns directly
func TestValidationRegexPatterns(t *testing.T) {
	t.Run("FunctionNamePattern", func(t *testing.T) {
		validNames := []string{
			"a", "get_weather", "func123", "test-name", "A", "Z", "0", "9", "_", "-",
			"get_weather_data", "calculate-tax-rate", "func_123_test",
			strings.Repeat("a", 64), // Max length
		}

		invalidNames := []string{
			"",                      // Empty
			"get weather",           // Space
			"get@weather",           // Special char
			"get.weather",           // Period
			"get_weather!",          // Exclamation
			"get#weather",           // Hash
			strings.Repeat("a", 65), // Too long
		}

		for _, name := range validNames {
			assert.True(t, functionNamePattern.MatchString(name), "Should match valid name: %q", name)
		}

		for _, name := range invalidNames {
			assert.False(t, functionNamePattern.MatchString(name), "Should not match invalid name: %q", name)
		}
	})

	t.Run("PrefixPattern", func(t *testing.T) {
		validPrefixes := []string{
			"a", "server", "Server123", "A", "Z", "0", "9",
			"weatherServer", "server123", "ABC123",
			strings.Repeat("a", 64), // Max length
		}

		invalidPrefixes := []string{
			"",                      // Empty
			"server_name",           // Underscore
			"server-name",           // Hyphen
			"server name",           // Space
			"server@name",           // Special char
			"server.name",           // Period
			strings.Repeat("a", 65), // Too long
		}

		for _, prefix := range validPrefixes {
			assert.True(t, prefixPattern.MatchString(prefix), "Should match valid prefix: %q", prefix)
		}

		for _, prefix := range invalidPrefixes {
			assert.False(t, prefixPattern.MatchString(prefix), "Should not match invalid prefix: %q", prefix)
		}
	})
}

// TestCaseSensitivity ensures that function names are properly case sensitive
func TestCaseSensitivity(t *testing.T) {
	testCases := []struct {
		name1 string
		name2 string
	}{
		{"get_weather", "Get_weather"},
		{"get_weather", "GET_WEATHER"},
		{"calculateTax", "calculatetax"},
		{"server.function", "Server.function"},
		{"server.function", "server.Function"},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%s_vs_%s", tc.name1, tc.name2), func(t *testing.T) {
			// Both should be valid
			err1 := ValidateFunctionName(tc.name1)
			err2 := ValidateFunctionName(tc.name2)

			assert.NoError(t, err1, "First name should be valid")
			assert.NoError(t, err2, "Second name should be valid")

			// But they should be treated as different (case sensitive)
			assert.NotEqual(t, tc.name1, tc.name2, "Names should be different (case sensitive)")
		})
	}
}

// === INTEGRATION TESTS ===
// These tests verify that the function name validation integrates properly with the adapter

// TestValidationIntegration_OpenAISpecCompliance verifies that the adapter correctly validates
// function names according to OpenAI specification with MCP prefix support
func TestValidationIntegration_OpenAISpecCompliance(t *testing.T) {
	adapter := New()

	testCases := []struct {
		name                 string
		assistantContent     string
		shouldDetectToolCall bool
		expectedFunctionName string
		description          string
	}{
		// Valid function names that should be detected
		{
			name:                 "ValidStandardName_Underscore",
			assistantContent:     `[{"name": "get_weather", "parameters": {"location": "Boston"}}]`,
			shouldDetectToolCall: true,
			expectedFunctionName: "get_weather",
			description:          "Standard function name with underscore should be valid",
		},
		{
			name:                 "ValidStandardName_Hyphen",
			assistantContent:     `[{"name": "get-current-time", "parameters": null}]`,
			shouldDetectToolCall: true,
			expectedFunctionName: "get-current-time",
			description:          "Standard function name with hyphens should be valid",
		},
		{
			name:                 "ValidStandardName_Numbers",
			assistantContent:     `[{"name": "function123", "parameters": {"key": "value"}}]`,
			shouldDetectToolCall: true,
			expectedFunctionName: "function123",
			description:          "Standard function name with numbers should be valid",
		},
		{
			name:                 "ValidStandardName_Mixed",
			assistantContent:     `[{"name": "calc_tax_2024", "parameters": {"income": 50000}}]`,
			shouldDetectToolCall: true,
			expectedFunctionName: "calc_tax_2024",
			description:          "Standard function name with mixed valid characters should be valid",
		},
		{
			name:                 "ValidMCPName_Basic",
			assistantContent:     `[{"name": "weather.get_current", "parameters": {"location": "NYC"}}]`,
			shouldDetectToolCall: true,
			expectedFunctionName: "weather.get_current",
			description:          "Basic MCP prefixed function name should be valid",
		},
		{
			name:                 "ValidMCPName_WithNumbers",
			assistantContent:     `[{"name": "server123.function456", "parameters": {"data": "test"}}]`,
			shouldDetectToolCall: true,
			expectedFunctionName: "server123.function456",
			description:          "MCP prefixed function name with numbers should be valid",
		},
		{
			name:                 "ValidMCPName_MixedCase",
			assistantContent:     `[{"name": "WeatherAPI.Get_Forecast", "parameters": {"city": "LA"}}]`,
			shouldDetectToolCall: true,
			expectedFunctionName: "WeatherAPI.Get_Forecast",
			description:          "Mixed case MCP function name should be valid",
		},
		{
			name:                 "ValidName_MaxLength",
			assistantContent:     `[{"name": "` + strings.Repeat("a", 64) + `", "parameters": {}}]`,
			shouldDetectToolCall: true,
			expectedFunctionName: strings.Repeat("a", 64),
			description:          "Function name at maximum length should be valid",
		},

		// Invalid function names that should NOT be detected as tool calls
		{
			name:                 "InvalidName_WithSpaces",
			assistantContent:     `[{"name": "get weather", "parameters": {"location": "Boston"}}]`,
			shouldDetectToolCall: false,
			description:          "Function name with spaces should be invalid per OpenAI spec",
		},
		{
			name:                 "InvalidName_WithSpecialChars",
			assistantContent:     `[{"name": "get@weather", "parameters": {"location": "Boston"}}]`,
			shouldDetectToolCall: false,
			description:          "Function name with special characters should be invalid",
		},
		{
			name:                 "InvalidName_TooLong",
			assistantContent:     `[{"name": "` + strings.Repeat("a", 65) + `", "parameters": {}}]`,
			shouldDetectToolCall: false,
			description:          "Function name longer than 64 characters should be invalid",
		},
		{
			name:                 "InvalidMCP_MultiplePeriods",
			assistantContent:     `[{"name": "server.sub.function", "parameters": {}}]`,
			shouldDetectToolCall: false,
			description:          "MCP function name with multiple periods should be invalid",
		},
		{
			name:                 "InvalidMCP_PrefixWithUnderscore",
			assistantContent:     `[{"name": "weather_server.get_current", "parameters": {}}]`,
			shouldDetectToolCall: false,
			description:          "MCP prefix with underscore should be invalid",
		},
		{
			name:                 "InvalidMCP_PrefixWithHyphen",
			assistantContent:     `[{"name": "weather-server.get_current", "parameters": {}}]`,
			shouldDetectToolCall: false,
			description:          "MCP prefix with hyphen should be invalid",
		},
		{
			name:                 "InvalidMCP_EmptyPrefix",
			assistantContent:     `[{"name": ".get_weather", "parameters": {}}]`,
			shouldDetectToolCall: false,
			description:          "Empty MCP prefix should be invalid",
		},
		{
			name:                 "InvalidMCP_EmptyFunction",
			assistantContent:     `[{"name": "server.", "parameters": {}}]`,
			shouldDetectToolCall: false,
			description:          "Empty function name in MCP format should be invalid",
		},

		// Natural JSON that should not be detected as function calls
		{
			name:                 "NaturalJSON_PersonName",
			assistantContent:     `[{"name": "John Smith", "age": 30, "city": "Boston"}]`,
			shouldDetectToolCall: false,
			description:          "Person data with name containing spaces should not be function call",
		},
		{
			name:                 "NaturalJSON_ProductName",
			assistantContent:     `[{"name": "iPhone 15 Pro", "price": 999, "color": "black"}]`,
			shouldDetectToolCall: false,
			description:          "Product data with name containing spaces should not be function call",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Logf("Testing: %s", tc.description)

			mockResp := createMockCompletionForValidation(tc.assistantContent)
			finalResp, err := adapter.TransformCompletionsResponse(mockResp)
			require.NoError(t, err, "Transform should not error")

			require.Len(t, finalResp.Choices, 1, "Should have one choice")
			choice := finalResp.Choices[0]

			if tc.shouldDetectToolCall {
				// Should detect tool call
				assert.Equal(t, "tool_calls", string(choice.FinishReason), "Should indicate tool calls were made")
				assert.Empty(t, choice.Message.Content, "Content should be cleared when tool calls are present")
				require.Greater(t, len(choice.Message.ToolCalls), 0, "Should have at least one tool call")

				// Verify the function name
				toolCall := choice.Message.ToolCalls[0]
				assert.Equal(t, tc.expectedFunctionName, toolCall.Function.Name, "Function name should match expected")
				assert.NotEmpty(t, toolCall.ID, "Tool call should have a unique ID")
				assert.Equal(t, "function", string(toolCall.Type), "Tool call type should be 'function'")
			} else {
				// Should NOT detect tool call
				assert.Empty(t, choice.Message.ToolCalls, "Should not have detected tool call")
				assert.NotEqual(t, "tool_calls", string(choice.FinishReason), "Should not indicate tool calls were made")
				// Content should be preserved for non-function-call responses
				assert.Equal(t, mockResp.Choices[0].Message.Content, choice.Message.Content, "Content should be preserved")
			}
		})
	}
}

// TestValidationIntegration_CaseSensitivity verifies that function names are case sensitive
func TestValidationIntegration_CaseSensitivity(t *testing.T) {
	adapter := New()

	testCases := []struct {
		functionName1 string
		functionName2 string
		description   string
	}{
		{
			functionName1: "get_weather",
			functionName2: "Get_weather",
			description:   "Standard function names should be case sensitive",
		},
		{
			functionName1: "weather.get_current",
			functionName2: "Weather.get_current",
			description:   "MCP prefix should be case sensitive",
		},
		{
			functionName1: "server.get_data",
			functionName2: "server.Get_data",
			description:   "MCP function name should be case sensitive",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			// Test first function name
			content1 := `[{"name": "` + tc.functionName1 + `", "parameters": {"test": "value"}}]`
			mockResp1 := createMockCompletionForValidation(content1)
			finalResp1, err := adapter.TransformCompletionsResponse(mockResp1)
			require.NoError(t, err)

			// Test second function name
			content2 := `[{"name": "` + tc.functionName2 + `", "parameters": {"test": "value"}}]`
			mockResp2 := createMockCompletionForValidation(content2)
			finalResp2, err := adapter.TransformCompletionsResponse(mockResp2)
			require.NoError(t, err)

			// Both should be valid
			require.Len(t, finalResp1.Choices, 1)
			require.Len(t, finalResp2.Choices, 1)
			require.Greater(t, len(finalResp1.Choices[0].Message.ToolCalls), 0)
			require.Greater(t, len(finalResp2.Choices[0].Message.ToolCalls), 0)

			// Function names should be preserved exactly (case sensitive)
			name1 := finalResp1.Choices[0].Message.ToolCalls[0].Function.Name
			name2 := finalResp2.Choices[0].Message.ToolCalls[0].Function.Name

			assert.Equal(t, tc.functionName1, name1, "First function name should be preserved exactly")
			assert.Equal(t, tc.functionName2, name2, "Second function name should be preserved exactly")
			assert.NotEqual(t, name1, name2, "Function names should be different (case sensitive)")
		})
	}
}

// TestValidationIntegration_EdgeCases tests edge cases of the validation
func TestValidationIntegration_EdgeCases(t *testing.T) {
	adapter := New()

	testCases := []struct {
		name             string
		assistantContent string
		shouldDetect     bool
		description      string
	}{
		{
			name:             "SingleCharacterName",
			assistantContent: `[{"name": "a", "parameters": {}}]`,
			shouldDetect:     true,
			description:      "Single character function name should be valid",
		},
		{
			name:             "NumberOnlyName",
			assistantContent: `[{"name": "123", "parameters": {}}]`,
			shouldDetect:     true,
			description:      "Number-only function name should be valid",
		},
		{
			name:             "EmptyParametersObject",
			assistantContent: `[{"name": "test_func", "parameters": {}}]`,
			shouldDetect:     true,
			description:      "Function with empty parameters object should be valid",
		},
		{
			name:             "NullParameters",
			assistantContent: `[{"name": "test_func", "parameters": null}]`,
			shouldDetect:     true,
			description:      "Function with null parameters should be valid",
		},
		{
			name:             "MissingParameters",
			assistantContent: `[{"name": "test_func"}]`,
			shouldDetect:     true,
			description:      "Function without parameters field should be valid",
		},
		{
			name:             "MCPWithinTotalLimit",
			assistantContent: `[{"name": "` + strings.Repeat("a", 30) + `.test_func", "parameters": {}}]`,
			shouldDetect:     true,
			description:      "MCP name within total 64-char limit should be valid",
		},
		{
			name:             "MCPAtTotalLimit",
			assistantContent: `[{"name": "` + strings.Repeat("a", 32) + `.` + strings.Repeat("b", 31) + `", "parameters": {}}]`,
			shouldDetect:     true,
			description:      "MCP name at exactly 64 chars should be valid",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Logf("Testing: %s", tc.description)

			mockResp := createMockCompletionForValidation(tc.assistantContent)
			finalResp, err := adapter.TransformCompletionsResponse(mockResp)
			require.NoError(t, err, "Transform should not error")

			require.Len(t, finalResp.Choices, 1)
			choice := finalResp.Choices[0]

			if tc.shouldDetect {
				assert.Greater(t, len(choice.Message.ToolCalls), 0, "Should detect tool call")
				assert.Equal(t, "tool_calls", string(choice.FinishReason), "Should indicate tool calls")
			} else {
				assert.Empty(t, choice.Message.ToolCalls, "Should not detect tool call")
				assert.NotEqual(t, "tool_calls", string(choice.FinishReason), "Should not indicate tool calls")
			}
		})
	}
}

// Helper functions for integration tests
func createMockCompletionForValidation(content string) openai.ChatCompletion {
	return openai.ChatCompletion{
		Choices: []openai.ChatCompletionChoice{
			{
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: content,
				},
			},
		},
	}
}

// TestHasCompleteJSON_Coverage tests uncovered paths in hasCompleteJSON
func TestHasCompleteJSON_Coverage(t *testing.T) {
	t.Run("InvalidJSONStructure", func(t *testing.T) {
		// Test HasCompleteJSON directly with invalid JSON
		result := HasCompleteJSON("{invalid json")
		assert.False(t, result, "Invalid JSON should return false")
	})

	t.Run("ValidJSONButNotFunctionCall", func(t *testing.T) {
		// Test with valid JSON that's not a function call
		result := HasCompleteJSON(`{"data": "value", "other": 123}`)
		assert.False(t, result, "Valid JSON that's not a function call should return false")
	})

	t.Run("EmptyContent", func(t *testing.T) {
		// Test HasCompleteJSON with empty content
		result := HasCompleteJSON("")
		assert.False(t, result, "Empty content should return false")
	})

	t.Run("WhitespaceOnlyContent", func(t *testing.T) {
		// Test HasCompleteJSON with whitespace-only content
		result := HasCompleteJSON("   \t\n   ")
		assert.False(t, result, "Whitespace-only content should return false")
	})
}

// TestFunctionNameValidation_Coverage tests uncovered paths in function name validation
func TestFunctionNameValidation_Coverage(t *testing.T) {
	t.Run("MCPFormatTotalLengthExceeded", func(t *testing.T) {
		// Create a name that exceeds MaxFunctionNameLength (64 chars)
		longName := strings.Repeat("a", 32) + "." + strings.Repeat("b", 32) // 65 chars total
		err := ValidateFunctionName(longName)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "maximum allowed is 64")
	})

	t.Run("MCPFormatPrefixTooLong", func(t *testing.T) {
		// Create a prefix that exceeds MaxPrefixLength (64 chars)
		longPrefix := strings.Repeat("a", 65) + ".function"
		err := ValidateFunctionName(longPrefix)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "maximum allowed is 64")
	})

	t.Run("MCPFormatFunctionPartTooLong", func(t *testing.T) {
		// Create a function part that exceeds MaxFunctionNameLength (64 chars)
		longFunction := "prefix." + strings.Repeat("a", 65)
		err := ValidateFunctionName(longFunction)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "maximum allowed is 64")
	})

	t.Run("MCPFormatPrefixWithInvalidChar", func(t *testing.T) {
		// MCP prefix should only contain alphanumeric characters
		invalidPrefix := "pre-fix.function"
		err := ValidateFunctionName(invalidPrefix)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "prefix")
		assert.Contains(t, err.Error(), "invalid characters")
	})
}

// TestCandidatePoolManagement tests that the candidate pool properly resets all fields
func TestCandidatePoolManagement(t *testing.T) {
	t.Run("PoolFieldReset", func(t *testing.T) {
		// Create an extractor with JSON content that will create candidates
		input := `Here is some JSON: {"name": "test_func", "parameters": {"key": "value"}}`
		extractor := NewJSONExtractor(input)

		// Extract JSON blocks - this will create candidates and return them to the pool
		results := extractor.ExtractJSONBlocks()
		require.NotEmpty(t, results, "Should extract at least one JSON block")
		assert.Contains(t, results[0], "test_func", "Should extract the test function")

		// Now create another extractor and extract from different content
		// This should reuse pooled candidates, and we want to verify they're properly reset
		input2 := `Different content with JSON: {"name": "other_func", "parameters": null}`
		extractor2 := NewJSONExtractor(input2)
		results2 := extractor2.ExtractJSONBlocks()

		require.NotEmpty(t, results2, "Should extract JSON from second input")
		assert.Contains(t, results2[0], "other_func", "Should extract the other function")

		// Verify the results are completely independent (no stale data from pool reuse)
		assert.NotEqual(t, results[0], results2[0], "Results should be different")
		assert.NotContains(t, results2[0], "test_func", "Second result should not contain first function name")
	})

	t.Run("PoolReuseWithMultipleCandidates", func(t *testing.T) {
		// Test with multiple JSON candidates to stress the pool management
		input := `First: {"name": "func1"} and second: {"name": "func2"} and third: {"name": "func3"}`
		extractor := NewJSONExtractor(input)

		results := extractor.ExtractJSONBlocks()
		require.Len(t, results, 3, "Should extract exactly 3 JSON blocks")

		// Verify all results are unique and correct
		expectedFuncs := []string{"func1", "func2", "func3"}
		for i, result := range results {
			assert.Contains(t, result, expectedFuncs[i], "Result %d should contain expected function name", i)
		}

		// Extract from different content to test pool reuse
		input2 := `Only one here: {"name": "single_func", "parameters": {"test": true}}`
		extractor2 := NewJSONExtractor(input2)
		results2 := extractor2.ExtractJSONBlocks()

		require.Len(t, results2, 1, "Should extract exactly 1 JSON block from second input")
		assert.Contains(t, results2[0], "single_func", "Should extract the single function")

		// Verify no contamination between extractions
		for _, result := range results2 {
			for _, expectedFunc := range expectedFuncs {
				assert.NotContains(t, result, expectedFunc, "Second extraction should not contain functions from first")
			}
		}
	})

	t.Run("PoolManagementWithComplexStructures", func(t *testing.T) {
		// Test with complex nested JSON structures
		complexJSON := `{
			"name": "complex_func",
			"parameters": {
				"nested": {
					"array": [1, 2, 3],
					"object": {"key": "value"}
				},
				"string": "test with spaces and special chars !@#$%"
			}
		}`

		extractor := NewJSONExtractor(complexJSON)
		results := extractor.ExtractJSONBlocks()

		require.Len(t, results, 1, "Should extract one complex JSON block")

		// Verify the JSON contains expected structure
		assert.Contains(t, results[0], "complex_func", "Should contain function name")
		assert.Contains(t, results[0], "nested", "Should contain nested structure")
		assert.Contains(t, results[0], "special chars", "Should preserve complex string content")

		// Test pool reuse with simple structure
		simpleJSON := `{"name": "simple"}`
		extractor2 := NewJSONExtractor(simpleJSON)
		results2 := extractor2.ExtractJSONBlocks()

		require.Len(t, results2, 1, "Should extract simple JSON")
		assert.NotContains(t, results2[0], "complex_func", "Simple result should not contain complex function")
		assert.NotContains(t, results2[0], "nested", "Simple result should not contain nested structure")
	})

	t.Run("PoolResetAfterFailedParsing", func(t *testing.T) {
		// Test that pool candidates are properly reset even when JSON parsing fails
		invalidJSON := `This is not JSON: {"incomplete": "object" and some trailing text`
		extractor := NewJSONExtractor(invalidJSON)
		results := extractor.ExtractJSONBlocks()

		// Should find no valid JSON
		assert.Empty(t, results, "Should not extract invalid JSON")

		// Now extract valid JSON to ensure pool candidates weren't contaminated
		validJSON := `{"name": "valid_func", "parameters": null}`
		extractor2 := NewJSONExtractor(validJSON)
		results2 := extractor2.ExtractJSONBlocks()

		require.Len(t, results2, 1, "Should extract valid JSON after failed extraction")
		assert.Contains(t, results2[0], "valid_func", "Should extract correct function name")
		assert.NotContains(t, results2[0], "incomplete", "Should not contain failed parsing artifacts")
	})
}
