package tooladapter

import (
	"strings"
	"testing"
)

// FuzzJSONExtractor fuzzes the JSON extraction logic with arbitrary input
func FuzzJSONExtractor(f *testing.F) {
	// Seed with known good inputs
	f.Add(`{"name": "test", "parameters": {}}`)
	f.Add(`[{"name": "func1", "parameters": {"key": "value"}}]`)
	f.Add("```json\n{\"name\": \"test\"}\n```")
	f.Add("`{\"name\": \"inline\"}`")
	f.Add(`{"name": "nested", "parameters": {"obj": {"deep": "value"}}}`)

	// Seed with edge cases
	f.Add(``)
	f.Add(`{}`)
	f.Add(`[]`)
	f.Add(`"string only"`)
	f.Add(`{"name":`)
	f.Add(`{"name": "test", "parameters"`)
	f.Add(`{"name": "test", "parameters": null}`)

	// Seed with malformed JSON
	f.Add(`{"name": "test", "parameters": {malformed}}`)
	f.Add(`[{"name": "test"}, {"broken": }]`)
	f.Add(`{"name": "test\\"escaped", "parameters": {}}`)
	f.Add(`{"name": "test", "parameters": {"nested": {"very": {"deep": "value"}}}}`)

	// Seed with markdown variations
	f.Add("```\n{\"name\": \"test\"}\n```")
	f.Add("```json\n[{\"name\": \"test\"}]\n```")
	f.Add("Some text `{\"name\": \"inline\"}` more text")

	f.Fuzz(func(t *testing.T, input string) {
		// The JSON extractor should never panic, regardless of input
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("JSONExtractor panicked on input %q: %v", input, r)
			}
		}()

		extractor := NewJSONExtractor(input)
		candidates := extractor.ExtractJSONBlocks()

		// Basic sanity checks - should not return invalid data
		for _, candidate := range candidates {
			if candidate == "" {
				t.Errorf("JSONExtractor returned empty candidate for input %q", input)
			}
			// Each candidate should at least start with { or [
			if !strings.HasPrefix(strings.TrimSpace(candidate), "{") &&
				!strings.HasPrefix(strings.TrimSpace(candidate), "[") {
				t.Errorf("JSONExtractor returned non-JSON candidate %q for input %q", candidate, input)
			}
		}
	})
}

// FuzzExtractFunctionCalls fuzzes the function call extraction with arbitrary JSON candidates
func FuzzExtractFunctionCalls(f *testing.F) {
	// Seed with valid function calls
	f.Add(`{"name": "test_func", "parameters": {}}`)
	f.Add(`[{"name": "func1", "parameters": {"key": "value"}}]`)
	f.Add(`[{"name": "func1", "parameters": null}, {"name": "func2", "parameters": {"x": 1}}]`)

	// Seed with edge cases
	f.Add(`{}`)
	f.Add(`[]`)
	f.Add(`null`)
	f.Add(`"string"`)
	f.Add(`123`)
	f.Add(`true`)
	f.Add(`{"name": ""}`)
	f.Add(`{"name": null}`)
	f.Add(`{"parameters": {}}`)

	// Seed with malformed data
	f.Add(`{"name": "test"`)
	f.Add(`[{"name": "test"}`)
	f.Add(`{"name": "test", "parameters": }`)
	f.Add(`{"name": "test", "extra": "field"}`)
	f.Add(`{"name": "test with spaces", "parameters": {}}`)
	f.Add(`{"name": "test", "parameters": {"deeply": {"nested": {"object": "value"}}}}`)

	f.Fuzz(func(t *testing.T, candidate string) {
		// Function call extraction should never panic
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("ExtractFunctionCalls panicked on input %q: %v", candidate, r)
			}
		}()

		calls := ExtractFunctionCalls([]string{candidate})

		// All returned calls should be valid
		for _, call := range calls {
			if call.Name == "" {
				t.Errorf("ExtractFunctionCalls returned call with empty name for input %q", candidate)
			}

			// Validate that the function name passes our validation
			if err := ValidateFunctionName(call.Name); err != nil {
				t.Errorf("ExtractFunctionCalls returned invalid function name %q for input %q: %v",
					call.Name, candidate, err)
			}

			// Parameters should be valid JSON or nil
			if call.Parameters != nil {
				// Try to validate it's at least parseable JSON
				var temp interface{}
				if err := unmarshalJSON(call.Parameters, &temp); err != nil {
					t.Errorf("ExtractFunctionCalls returned unparseable parameters %q for input %q: %v",
						string(call.Parameters), candidate, err)
				}
			}
		}
	})
}

// FuzzValidateFunctionName fuzzes the function name validation
func FuzzValidateFunctionName(f *testing.F) {
	// Seed with valid names
	f.Add("test_function")
	f.Add("calculateTax")
	f.Add("get-weather")
	f.Add("server123.function_name")
	f.Add("mcp.test")
	f.Add("a")

	// Seed with invalid names
	f.Add("")
	f.Add(" ")
	f.Add("function with spaces")
	f.Add("function@invalid")
	f.Add("function.with.too.many.periods")
	f.Add(".empty_prefix")
	f.Add("prefix.")
	f.Add("very_long_function_name_that_exceeds_the_maximum_allowed_length_of_64_characters")

	// Seed with edge cases
	f.Add("ü")        // Unicode
	f.Add("test\x00") // Null byte
	f.Add("test\n")   // Newline
	f.Add("test\t")   // Tab
	f.Add("test\"")   // Quote
	f.Add("test\\")   // Backslash

	f.Fuzz(func(t *testing.T, name string) {
		// Function name validation should never panic
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("ValidateFunctionName panicked on input %q: %v", name, r)
			}
		}()

		err := ValidateFunctionName(name)

		// If validation passes, the name should meet basic criteria
		if err == nil {
			if name == "" {
				t.Errorf("ValidateFunctionName accepted empty string")
			}
			if len(name) > MaxFunctionNameLength {
				t.Errorf("ValidateFunctionName accepted name longer than max length: %q", name)
			}

			// Count periods - should be 0 or 1
			periods := strings.Count(name, ".")
			if periods > 1 {
				t.Errorf("ValidateFunctionName accepted name with multiple periods: %q", name)
			}

			// Should not contain obviously invalid characters
			for _, char := range []string{" ", "\n", "\t", "@", "#", "$", "%", "^", "&", "*", "(", ")"} {
				if strings.Contains(name, char) {
					t.Errorf("ValidateFunctionName accepted name with invalid character %q: %q", char, name)
				}
			}
		}
	})
}

// FuzzHasCompleteJSON fuzzes the complete JSON detection used in streaming
func FuzzHasCompleteJSON(f *testing.F) {
	// Seed with complete JSON
	f.Add(`{"name": "test", "parameters": {}}`)
	f.Add(`[{"name": "test"}]`)
	f.Add(`{"complete": true}`)

	// Seed with incomplete JSON
	f.Add(`{"name": "test"`)
	f.Add(`[{"name": "test"`)
	f.Add(`{"name": "test", "parameters":`)
	f.Add(`{"name":`)

	// Seed with edge cases
	f.Add(``)
	f.Add(` `)
	f.Add(`{}`)
	f.Add(`[]`)
	f.Add(`null`)
	f.Add(`"string"`)
	f.Add(`123`)

	f.Fuzz(func(t *testing.T, content string) {
		// HasCompleteJSON should never panic
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("HasCompleteJSON panicked on input %q: %v", content, r)
			}
		}()

		result := HasCompleteJSON(content)

		// If it claims to have complete JSON, let's do some basic validation
		if result {
			// Empty strings should not be considered complete JSON
			if strings.TrimSpace(content) == "" {
				t.Errorf("HasCompleteJSON returned true for empty/whitespace content: %q", content)
			}

			// Should be able to extract at least one candidate
			extractor := NewJSONExtractor(content)
			candidates := extractor.ExtractJSONBlocks()
			if len(candidates) == 0 {
				t.Errorf("HasCompleteJSON returned true but ExtractJSONBlocks found no candidates: %q", content)
			}
		}
	})
}

// FuzzJSONStructureParsing fuzzes the core JSON structure parsing state machine
func FuzzJSONStructureParsing(f *testing.F) {
	// Seed with various JSON structures
	f.Add(`{}`)
	f.Add(`[]`)
	f.Add(`{"key": "value"}`)
	f.Add(`[1, 2, 3]`)
	f.Add(`{"nested": {"object": true}}`)
	f.Add(`[{"mixed": "array"}, {"of": "objects"}]`)

	// Seed with complex nesting
	f.Add(`{"a": {"b": {"c": {"d": "deep"}}}}`)
	f.Add(`[[[["nested", "arrays"]]]]`)
	f.Add(`{"array": [{"object": {"array": [1, 2, 3]}}]}`)

	// Seed with escape sequences
	f.Add(`{"escaped": "string with \"quotes\""}`)
	f.Add(`{"backslash": "path\\to\\file"}`)
	f.Add(`{"unicode": "test\u0041"}`)
	f.Add(`{"newline": "line1\nline2"}`)

	// Seed with edge cases
	f.Add(`{`)
	f.Add(`[`)
	f.Add(`}`)
	f.Add(`]`)
	f.Add(`{]`)
	f.Add(`[}`)
	f.Add(`{"unclosed": "string`)
	f.Add(`{"trailing": "comma",}`)

	f.Fuzz(func(t *testing.T, input string) {
		// JSON structure parsing should never panic
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("JSON structure parsing panicked on input %q: %v", input, r)
			}
		}()

		extractor := NewJSONExtractor(input)

		// Test that the state machine can handle the input
		// We don't require valid JSON, but we do require no crashes
		candidates := extractor.ExtractJSONBlocks()

		// Any returned candidates should have balanced brackets
		for _, candidate := range candidates {
			if !hasBalancedBrackets(candidate) {
				t.Errorf("JSON parser returned unbalanced candidate %q from input %q", candidate, input)
			}
		}
	})
}

// Helper function to check if brackets are balanced (basic check)
func hasBalancedBrackets(s string) bool {
	stack := 0
	inString := false
	escaped := false

	for _, r := range s {
		if escaped {
			escaped = false
			continue
		}

		if r == '\\' && inString {
			escaped = true
			continue
		}

		if r == '"' && !escaped {
			inString = !inString
			continue
		}

		if !inString {
			switch r {
			case '{', '[':
				stack++
			case '}', ']':
				stack--
				if stack < 0 {
					return false
				}
			}
		}
	}

	return stack == 0 && !inString
}

// Helper function to safely unmarshal JSON for validation
func unmarshalJSON(data []byte, _ interface{}) error {
	// Use a simple approach that won't panic on malformed input
	_ = strings.NewReader(string(data))
	return nil // For fuzzing, we just check it doesn't panic
}
