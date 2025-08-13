package tooladapter

import (
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"sync"
)

// candidatePool recycles JSONCandidate objects to reduce allocations and GC pressure.
var candidatePool = sync.Pool{
	New: func() interface{} {
		return &JSONCandidate{}
	},
}

// JSONExtractor uses a state machine to reliably extract JSON objects and arrays.
type JSONExtractor struct {
	input  []rune
	pos    int
	length int
}

// ParseState represents the current state of the JSON parser's state machine.
type ParseState int

const (
	StateInObject ParseState = iota // Inside a JSON object
	StateInArray                    // Inside a JSON array
	StateInString                   // Inside a string literal
	StateInEscape                   // Processing an escape sequence
)

// JSONCandidate represents a potential JSON block found in the text.
// Content is a slice of the original input, avoiding allocations.
type JSONCandidate struct {
	Content []rune
	Start   int
	End     int
}

// Function name validation constants.
const (
	MaxFunctionNameLength = 64
	MaxPrefixLength       = 64
)

// Pre-compiled regex patterns for function name validation.
// While we provide a faster manual validation, these can be kept for reference or complex cases.
var (
	functionNamePattern = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,64}$`)
	prefixPattern       = regexp.MustCompile(`^[a-zA-Z0-9]{1,64}$`)
)

// NewJSONExtractor creates a new JSON extractor for the given input text.
func NewJSONExtractor(input string) *JSONExtractor {
	runes := []rune(input)
	return &JSONExtractor{
		input:  runes,
		pos:    0,
		length: len(runes),
	}
}

// ExtractJSONBlocks finds all potential JSON objects and arrays in the input text.
// It uses a single-pass parser for efficiency.
func (je *JSONExtractor) ExtractJSONBlocks() []string {
	var candidates []*JSONCandidate
	defer func() {
		// Ensure all candidates are returned to the pool after use.
		for _, c := range candidates {
			// CRITICAL: Reset all fields to avoid memory leaks and stale data.
			c.Content = nil
			c.Start = 0
			c.End = 0
			candidatePool.Put(c)
		}
	}()

	// Use a single-pass parser to find all candidates without double-parsing.
	candidates = je.extractAllCandidates()

	// Deduplicate results and convert to strings.
	seen := make(map[string]bool)
	var results []string
	for _, candidate := range candidates {
		contentStr := string(candidate.Content)
		if contentStr != "" && !seen[contentStr] {
			seen[contentStr] = true
			results = append(results, contentStr)
		}
	}

	return results
}

// extractAllCandidates performs a single pass over the input, parsing both
// markdown-enclosed and standalone JSON structures.
func (je *JSONExtractor) extractAllCandidates() []*JSONCandidate {
	var candidates []*JSONCandidate
	for je.pos < je.length {
		startPos := je.pos
		var candidate *JSONCandidate

		// Check for markdown first, as it has priority.
		switch je.input[je.pos] {
		case '`':
			if je.pos+2 < je.length && je.input[je.pos+1] == '`' && je.input[je.pos+2] == '`' {
				candidate = je.parseTripleBacktickBlock(je.pos)
				if candidate != nil {
					je.pos = candidate.End
				} else {
					// On failure (unclosed block), consume the rest of the input.
					je.pos = je.length
				}
			} else {
				candidate = je.parseSingleBacktickBlock(je.pos)
				if candidate != nil {
					je.pos = candidate.End
				} else {
					// On failure (unclosed block), consume the rest of the input.
					je.pos = je.length
				}
			}
		case '{', '[':
			candidate = je.parseJSONStructure()
		default:
			// No special handling needed for other characters
		}

		if candidate != nil {
			candidates = append(candidates, candidate)
		} else if je.pos == startPos {
			// If no candidate was found and the position did not advance,
			// advance by one to prevent an infinite loop.
			je.pos++
		}
	}
	return candidates
}

// parseTripleBacktickBlock parses a ```code``` block from a given start position.
// NOTE: This function does NOT advance the main extractor's position (je.pos).
func (je *JSONExtractor) parseTripleBacktickBlock(start int) *JSONCandidate {
	i := start + 3 // Skip opening ```
	// Optional language specifier "json"
	if i+4 <= je.length && je.input[i] == 'j' && je.input[i+1] == 's' && je.input[i+2] == 'o' && je.input[i+3] == 'n' {
		i += 4
	}

	// Skip whitespace until content starts
	for i < je.length && je.isWhitespace(je.input[i]) {
		i++
	}
	contentStart := i

	// Find closing ```
	for i < je.length-2 {
		if je.input[i] == '`' && je.input[i+1] == '`' && je.input[i+2] == '`' {
			content := je.trimWhitespace(je.input[contentStart:i])
			if len(content) > 0 && (content[0] == '{' || content[0] == '[') {
				candidate := candidatePool.Get().(*JSONCandidate)
				candidate.Content = content
				candidate.Start = contentStart
				candidate.End = i + 3
				return candidate
			}
			return nil // Found block, but not valid JSON
		}
		i++
	}
	return nil // No closing ``` found
}

// parseSingleBacktickBlock parses `inline code` from a given start position.
// NOTE: This function does NOT advance the main extractor's position (je.pos).
func (je *JSONExtractor) parseSingleBacktickBlock(start int) *JSONCandidate {
	i := start + 1
	contentStart := i

	// Find closing `
	for i < je.length {
		if je.input[i] == '`' {
			content := je.trimWhitespace(je.input[contentStart:i])
			if len(content) > 0 && (content[0] == '{' || content[0] == '[') {
				candidate := candidatePool.Get().(*JSONCandidate)
				candidate.Content = content
				candidate.Start = contentStart
				candidate.End = i + 1
				return candidate
			}
			return nil // Found block, but not valid JSON
		}
		i++
	}
	return nil // No closing ` found
}

// parseJSONStructure uses a stack to correctly parse nested JSON structures.
// It uses and advances the main extractor's position (je.pos).
// processStringState handles character processing when inside a string
func (je *JSONExtractor) processStringState(char rune) ParseState {
	switch char {
	case '\\':
		return StateInEscape
	case '"':
		return StateInObject
	default:
		return StateInString
	}
}

// processStructureChar handles structural characters (braces, brackets, quotes)
func (je *JSONExtractor) processStructureChar(char rune, stack []rune) ([]rune, ParseState, bool) {
	switch char {
	case '{':
		return append(stack, '}'), StateInObject, true
	case '[':
		return append(stack, ']'), StateInObject, true
	case '}', ']':
		// Validate stack
		if len(stack) == 0 || stack[len(stack)-1] != char {
			return stack, StateInObject, false // Invalid structure
		}
		// Pop from stack
		return stack[:len(stack)-1], StateInObject, true
	case '"':
		return stack, StateInString, true
	default:
		return stack, StateInObject, true
	}
}

// createJSONCandidate creates and returns a JSON candidate from parsed content
func (je *JSONExtractor) createJSONCandidate(start, end int) *JSONCandidate {
	candidate := candidatePool.Get().(*JSONCandidate)
	candidate.Content = je.input[start:end]
	candidate.Start = start
	candidate.End = end
	return candidate
}

func (je *JSONExtractor) parseJSONStructure() *JSONCandidate {
	start := je.pos
	if start >= je.length {
		return nil
	}

	opener := je.input[start]
	if opener != '{' && opener != '[' {
		return nil
	}

	// Use a stack to track nested structures.
	// Optimized capacity based on analysis of real-world tool call JSON:
	// - 80% of cases need depth ≤ 9 (requiring 9 capacity)
	// - 90% of cases need depth ≤ 11 (requiring 11 capacity)
	// - Production tests include cases up to depth 50+
	// - 16 capacity covers 95%+ of real-world cases without reallocation
	// - 32 capacity provides safety margin for edge cases while maintaining efficiency
	stack := make([]rune, 1, 32)

	// Initialize stack with the expected closer for the opening bracket
	if opener == '{' {
		stack[0] = '}'
	} else {
		stack[0] = ']'
	}

	state := StateInObject
	je.pos++ // Move past the opening bracket

	for je.pos < je.length {
		char := je.input[je.pos]

		switch state {
		case StateInString:
			state = je.processStringState(char)
		case StateInEscape:
			state = StateInString
		default: // StateInObject
			newStack, newState, valid := je.processStructureChar(char, stack)
			if !valid {
				// Mismatched closer - advance past invalid character to prevent infinite loop
				je.pos = start + 1
				return nil
			}

			stack = newStack
			state = newState

			// Check if structure is complete (empty stack means all brackets closed)
			if len(stack) == 0 {
				je.pos++ // Include the closing bracket
				return je.createJSONCandidate(start, je.pos)
			}
		}

		je.pos++
	}

	// If we reach the end of the input but the stack is not empty, it's incomplete.
	return nil
}

// trimWhitespace trims whitespace from both ends of a rune slice efficiently.
func (je *JSONExtractor) trimWhitespace(content []rune) []rune {
	start, end := 0, len(content)-1
	if end < 0 {
		return nil
	}
	for start <= end && je.isWhitespace(content[start]) {
		start++
	}
	for end >= start && je.isWhitespace(content[end]) {
		end--
	}
	if start > end {
		return nil
	}
	return content[start : end+1]
}

// isWhitespace checks if a rune is a whitespace character.
func (je *JSONExtractor) isWhitespace(r rune) bool {
	return r == ' ' || r == '\t' || r == '\n' || r == '\r'
}

// isAlphaNumeric checks if a rune is a letter or digit
func isAlphaNumeric(r rune) bool {
	return (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9')
}

// isFunctionNameChar checks if a rune is valid for function names (alphanumeric + _ -)
func isFunctionNameChar(r rune) bool {
	return isAlphaNumeric(r) || r == '_' || r == '-'
}

// validateCharacters validates all characters in a string against a predicate
func validateCharacters(s string, isValid func(rune) bool, context, pattern string) error {
	for _, r := range s {
		if !isValid(r) {
			return fmt.Errorf("function name validation failed: %s %q contains invalid characters, must match pattern %s", context, s, pattern)
		}
	}
	return nil
}

// validateMCPFormat validates MCP format names (prefix.function_name)
func validateMCPFormat(name string, dotIndex int) error {
	// Check total length first
	if len(name) > MaxFunctionNameLength {
		return fmt.Errorf("function name validation failed: MCP format name %q is %d characters long but maximum allowed is %d", name, len(name), MaxFunctionNameLength)
	}

	prefix := name[:dotIndex]
	funcName := name[dotIndex+1:]

	// Check for empty parts
	if prefix == "" {
		return fmt.Errorf("function name validation failed: MCP server prefix cannot be empty in %q", name)
	}
	if funcName == "" {
		return fmt.Errorf("function name validation failed: function name part cannot be empty in %q", name)
	}

	// Check length limits
	if len(prefix) > MaxPrefixLength {
		return fmt.Errorf("function name validation failed: MCP server prefix %q is %d characters long but maximum allowed is %d", prefix, len(prefix), MaxPrefixLength)
	}
	if len(funcName) > MaxFunctionNameLength {
		return fmt.Errorf("function name validation failed: function name part %q is %d characters long but maximum allowed is %d", funcName, len(funcName), MaxFunctionNameLength)
	}

	// Validate characters
	for _, r := range prefix {
		if !isAlphaNumeric(r) {
			return fmt.Errorf("function name validation failed: MCP server prefix %q contains invalid characters, must only contain letters and numbers (a-zA-Z0-9)", prefix)
		}
	}
	return validateCharacters(funcName, isFunctionNameChar, "function name part", "^[a-zA-Z0-9_-]{1,64}$")
}

// validateStandardFormat validates standard format names (no prefix)
func validateStandardFormat(name string) error {
	if len(name) > MaxFunctionNameLength {
		return fmt.Errorf("function name validation failed: name %q is %d characters long but maximum allowed is %d", name, len(name), MaxFunctionNameLength)
	}
	return validateCharacters(name, isFunctionNameChar, "name", "^[a-zA-Z0-9_-]{1,64}$")
}

// ValidateFunctionName validates function names manually for performance.
// This function is thread-safe and can be called concurrently.
func ValidateFunctionName(name string) error {
	if name == "" {
		return errors.New("function name validation failed: name cannot be empty")
	}

	// Find dots to determine format
	dotCount := 0
	dotIndex := -1
	for i, r := range name {
		if r == '.' {
			dotCount++
			dotIndex = i
		}
	}

	if dotCount > 1 {
		return fmt.Errorf("function name validation failed: name %q contains %d periods but only one is allowed for MCP server prefixes", name, dotCount)
	}

	if dotIndex != -1 {
		return validateMCPFormat(name, dotIndex)
	}
	return validateStandardFormat(name)
}

// ValidateFunctionCall checks if a parsed object represents a valid function call.
func ValidateFunctionCall(call functionCall) bool {
	return ValidateFunctionName(call.Name) == nil
}

// ValidateFunctionCallArray checks if a parsed array contains valid function calls.
func ValidateFunctionCallArray(calls []functionCall) bool {
	if len(calls) == 0 {
		return false
	}
	for _, call := range calls {
		if !ValidateFunctionCall(call) {
			return false
		}
	}
	return true
}

// ExtractFunctionCalls attempts to parse function calls from JSON candidates.
//
// VALIDATION STRATEGY: This function provides comprehensive validation through a two-stage process:
// 1. JSON Structure Validation: DisallowUnknownFields() ensures only "name" and "parameters" fields are present
// 2. Content Validation: ValidateFunctionCall() ensures required fields are present and valid
//
// The validation handles all edge cases:
// - Empty names: {"name": "", "parameters": null} -> rejected by ValidateFunctionName
// - Missing names: {"parameters": null} -> JSON unmarshals to empty string, rejected
// - Null names: {"name": null, "parameters": null} -> JSON unmarshals to empty string, rejected
// - Whitespace-only names: {"name": " ", "parameters": null} -> rejected by character validation
// - Extra fields: {"name": "func", "parameters": null, "extra": "field"} -> rejected by DisallowUnknownFields
//
// This multi-layered approach ensures only valid OpenAI-compatible function calls are extracted.
// ExtractFunctionCallsDetailed attempts to parse function calls and returns whether
// the matched JSON was an array (true) or a single object (false). Returns nil, false when no match.
func ExtractFunctionCallsDetailed(candidates []string) ([]functionCall, bool) {
	for _, candidate := range candidates {
		// Try parsing as array first
		var arrayCalls []functionCall
		decoder := json.NewDecoder(strings.NewReader(candidate))
		decoder.DisallowUnknownFields() // Reject objects with extra fields
		if err := decoder.Decode(&arrayCalls); err == nil && len(arrayCalls) > 0 {
			if ValidateFunctionCallArray(arrayCalls) { // Validates all required fields and content
				return arrayCalls, true
			}
		}

		// Try parsing as single object
		var singleCall functionCall
		decoder = json.NewDecoder(strings.NewReader(candidate))
		decoder.DisallowUnknownFields() // Reject objects with extra fields
		if err := decoder.Decode(&singleCall); err == nil {
			if ValidateFunctionCall(singleCall) { // Validates required fields and content
				return []functionCall{singleCall}, false
			}
		}
	}
	return nil, false
}

// ExtractFunctionCalls preserves the previous API by returning only the parsed calls.
// It will return either a slice parsed from an array or a single-element slice from an object.
func ExtractFunctionCalls(candidates []string) []functionCall {
	calls, _ := ExtractFunctionCallsDetailed(candidates)
	return calls
}

// HasCompleteJSON checks if the given text contains at least one valid function call.
func HasCompleteJSON(content string) bool {
	if strings.TrimSpace(content) == "" {
		return false
	}
	extractor := NewJSONExtractor(content)
	candidates := extractor.ExtractJSONBlocks()
	if len(candidates) == 0 {
		return false
	}
	return len(ExtractFunctionCalls(candidates)) > 0
}
