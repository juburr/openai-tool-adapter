package tooladapter

import (
	"fmt"
	"strings"
	"testing"
)

// Declare package-level variables to store the results of the benchmarked functions.
// This prevents the compiler from optimizing away the function calls.
var (
	benchmarkResult []string
	benchmarkErr    error
)

// BenchmarkJSONExtractor_FullExtraction benchmarks the entire end-to-end extraction process.
func BenchmarkJSONExtractor_FullExtraction(b *testing.B) {
	smallContent := `
Here's some text with embedded JSON:

` + "```json" + `
{"name": "get_weather", "parameters": {"location": "NYC"}}
` + "```" + `

And some inline code: ` + "`{\"name\": \"get_time\", \"parameters\": null}`" + `

Another code block:
` + "```" + `
[{"name": "search", "parameters": {"query": "test"}}, {"name": "format", "parameters": {"style": "json"}}]
` + "```" + `

More text and another inline: ` + "`[{\"name\": \"calculate\"}]`" + `
`
	largeContent := strings.Repeat(smallContent, 100)
	complexJSON := `{"name": "complex_function", "parameters": {"config": {"nested": {"deep": {"value": "test", "options": ["a", "b", "c"], "metadata": {"created": "2023-01-01", "tags": ["tag1", "tag2", "tag3"]}}}}}}`
	veryLargeContent := strings.Repeat("Text before\n```json\n"+complexJSON+"\n```\nText after.\n", 500)

	b.Run("SmallContent", func(b *testing.B) {
		extractor := NewJSONExtractor(smallContent)
		b.ReportAllocs()
		b.ResetTimer()
		var r []string // Local variable to avoid race conditions if run in parallel
		for i := 0; i < b.N; i++ {
			// Reset the extractor's position for each run in the loop
			extractor.pos = 0
			r = extractor.ExtractJSONBlocks()
		}
		benchmarkResult = r // Assign the result of the last operation to the package-level variable
	})

	b.Run("LargeContent", func(b *testing.B) {
		extractor := NewJSONExtractor(largeContent)
		b.ReportAllocs()
		b.ResetTimer()
		var r []string
		for i := 0; i < b.N; i++ {
			extractor.pos = 0
			r = extractor.ExtractJSONBlocks()
		}
		benchmarkResult = r
	})

	b.Run("VeryLargeContent", func(b *testing.B) {
		extractor := NewJSONExtractor(veryLargeContent)
		b.ReportAllocs()
		b.ResetTimer()
		var r []string
		for i := 0; i < b.N; i++ {
			extractor.pos = 0
			r = extractor.ExtractJSONBlocks()
		}
		benchmarkResult = r
	})
}

// validateFunctionNameRegex is a regex-based function validator for comparison.
func validateFunctionNameRegex(name string) error {
	if name == "" {
		return fmt.Errorf("function name cannot be empty")
	}

	periodCount := strings.Count(name, ".")
	if periodCount > 1 {
		return fmt.Errorf("function name cannot contain more than one period, got %d periods in %q", periodCount, name)
	}

	if periodCount == 1 {
		parts := strings.SplitN(name, ".", 2)
		if len(parts) != 2 {
			return fmt.Errorf("invalid MCP prefixed function name format: %q", name)
		}

		prefix := parts[0]
		functionName := parts[1]

		if prefix == "" {
			return fmt.Errorf("MCP server prefix cannot be empty in %q", name)
		}
		if len(prefix) > MaxPrefixLength {
			return fmt.Errorf("MCP server prefix too long: %d characters (max %d) in %q", len(prefix), MaxPrefixLength, name)
		}
		if !prefixPattern.MatchString(prefix) {
			return fmt.Errorf("MCP server prefix %q must contain only letters and numbers (a-zA-Z0-9)", prefix)
		}

		if functionName == "" {
			return fmt.Errorf("function name part cannot be empty in %q", name)
		}
		if len(functionName) > MaxFunctionNameLength {
			return fmt.Errorf("function name part too long: %d characters (max %d) in %q", len(functionName), MaxFunctionNameLength, name)
		}
		if !functionNamePattern.MatchString(functionName) {
			return fmt.Errorf("function name part %q must match pattern ^[a-zA-Z0-9_-]{1,64}$", functionName)
		}

		return nil
	}

	if len(name) > MaxFunctionNameLength {
		return fmt.Errorf("function name too long: %d characters (max %d) in %q", len(name), MaxFunctionNameLength, name)
	}
	if !functionNamePattern.MatchString(name) {
		return fmt.Errorf("function name %q must match pattern ^[a-zA-Z0-9_-]{1,64}$", name)
	}

	return nil
}

// BenchmarkValidateFunctionName compares the performance of the manual vs. regex-based validation.
func BenchmarkValidateFunctionName(b *testing.B) {
	// Mixed valid and invalid names to test branching logic
	mixedTestCases := []string{
		"get_weather",
		"function123",
		"my-func",
		"a",
		strings.Repeat("a", 64),
		"server.get_weather",
		"server1.func2",
		strings.Repeat("s", 64) + "." + strings.Repeat("f", 64),
		"",                      // invalid
		strings.Repeat("a", 65), // invalid
		"get weather",           // invalid
		"server.sub.func",       // invalid
	}

	// Production-like scenario with only valid names, but no MCP prefixes in use.
	validNoMCPCases := []string{
		"get_current_weather",
		"getUserProfile",
		"update-database-record",
		"process_image_data_v2",
		"calculate-complex-tax-rate-2024",
		strings.Repeat("a", 64),
		"short",
		"another_long_and_descriptive_function_name",
	}

	// Production-like scenario with only valid names and everything MCP-prefixed.
	validOnlyMCPCases := []string{
		"weather.getCurrent",
		"database.updateRecord",
		"userAPI.fetchProfile",
		"billing.processPayment-v3",
		"longservernamepart12345.longfunctionnamepart67890",
		strings.Repeat("s", 64) + ".func",
		"api." + strings.Repeat("f", 64),
		"auth.validate_user_token",
	}

	// Original benchmark with mixed cases
	b.Run("MixedAll", func(b *testing.B) {
		b.Run("Manual", func(b *testing.B) {
			b.ReportAllocs()
			var err error
			for i := 0; i < b.N; i++ {
				for _, name := range mixedTestCases {
					err = ValidateFunctionName(name)
				}
			}
			benchmarkErr = err
		})

		b.Run("Regex", func(b *testing.B) {
			b.ReportAllocs()
			var err error
			for i := 0; i < b.N; i++ {
				for _, name := range mixedTestCases {
					err = validateFunctionNameRegex(name)
				}
			}
			benchmarkErr = err
		})
	})

	// --- NEW: Benchmark for valid, non-MCP names ---
	b.Run("ValidNoMCP", func(b *testing.B) {
		b.Run("Manual", func(b *testing.B) {
			b.ReportAllocs()
			var err error
			for i := 0; i < b.N; i++ {
				for _, name := range validNoMCPCases {
					err = ValidateFunctionName(name)
				}
			}
			benchmarkErr = err
		})

		b.Run("Regex", func(b *testing.B) {
			b.ReportAllocs()
			var err error
			for i := 0; i < b.N; i++ {
				for _, name := range validNoMCPCases {
					err = validateFunctionNameRegex(name)
				}
			}
			benchmarkErr = err
		})
	})

	// --- NEW: Benchmark for valid, MCP-only names ---
	b.Run("ValidOnlyMCP", func(b *testing.B) {
		b.Run("Manual", func(b *testing.B) {
			b.ReportAllocs()
			var err error
			for i := 0; i < b.N; i++ {
				for _, name := range validOnlyMCPCases {
					err = ValidateFunctionName(name)
				}
			}
			benchmarkErr = err
		})

		b.Run("Regex", func(b *testing.B) {
			b.ReportAllocs()
			var err error
			for i := 0; i < b.N; i++ {
				for _, name := range validOnlyMCPCases {
					err = validateFunctionNameRegex(name)
				}
			}
			benchmarkErr = err
		})
	})
}
