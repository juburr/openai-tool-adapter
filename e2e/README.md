# E2E Test Suite

This directory contains end-to-end tests for the OpenAI Tool Adapter package. These tests require a running LLM instance and are designed to validate the adapter's functionality against real model responses.

## Test Structure

### Test Files

- **`config_test.go`** - Configuration management and environment variable handling
- **`testutils_test.go`** - Common test utilities and helper functions
- **`basic_test.go`** - Basic non-streaming requests without tools
- **`tool_calling_test.go`** - Non-streaming tool calling scenarios
- **`streaming_test.go`** - Streaming requests and tool calling
- **`edge_cases_test.go`** - Edge cases, error handling, and resource limits
- **`main.go`** - Interactive testing tool (can be run independently)

### Test Categories

1. **Basic Functionality Tests**
   - Requests without tools
   - Empty messages
   - Long messages
   - Ambiguous requests

2. **Tool Calling Tests**
   - Single tool scenarios (weather, calculator)
   - Multiple available tools
   - Tool selection logic
   - Argument parsing and validation

3. **Streaming Tests**
   - Basic streaming without tools
   - Streaming with tool calls
   - Different tool policies (StopOnFirst, AllowMixed)
   - Real-time tool detection

4. **Edge Cases and Error Handling**
   - Timeout scenarios
   - Invalid tool definitions
   - Very long arguments
   - Unicode and special characters
   - Resource limits
   - Malformed responses

## Configuration

The tests use environment variables for configuration:

| Variable | Description | Default |
|----------|-------------|---------|
| `E2E_BASE_URL` | LLM service base URL | `http://localhost:8000/v1` |
| `E2E_API_KEY` | API key (if required) | `dummy-key` |
| `E2E_MODEL` | Model name | `google/gemma-3-4b-it` |
| `E2E_TIMEOUT_SECONDS` | Request timeout | `60` (increased for slow models) |

## Running Tests

### Prerequisites

1. **Running LLM Instance**: You need a compatible LLM service running (e.g., vLLM, Ollama, or any OpenAI-compatible API)
2. **Go Dependencies**: Ensure all test dependencies are installed

### Basic Usage

```bash
# Run all e2e tests (requires build tag)
go test -tags e2e ./e2e

# Run specific test
go test -tags e2e ./e2e -run TestBasicNonStreamingRequests

# Run with verbose output
go test -tags e2e ./e2e -v

# Run streaming tests only
go test -tags e2e ./e2e -run TestStreaming
```

### With Custom Configuration

```bash
# Test against different endpoint
E2E_BASE_URL="http://remote-server:8080/v1" go test -tags e2e ./e2e

# Test with different model
E2E_MODEL="mistral-7b-instruct" go test -tags e2e ./e2e

# Test with authentication
E2E_API_KEY="your-actual-key" go test -tags e2e ./e2e
```

### Interactive Testing

```bash
# Run the interactive test tool
go run -tags e2e cmd/main.go

# With verbose output
go run -tags e2e cmd/main.go -verbose

# With custom configuration
E2E_MODEL="custom-model" go run -tags e2e cmd/main.go -verbose
```

## Expected Model Behavior

### Tool Calling Format

The tests expect models to respond with JSON arrays in this format:

```json
[{"name": "function_name", "parameters": {"arg": "value"}}]
```

### Supported Tools

The test suite includes these predefined tools:

1. **Weather Tool** (`get_weather`)
   - Parameters: `location` (required), `unit` (optional)
   - Used for testing location-based queries

2. **Calculator Tool** (`calculate`)
   - Parameters: `expression` (required)
   - Used for testing mathematical queries

### Model Requirements

- Must support instruction-following
- Should be able to produce structured JSON output
- Needs to understand tool usage context from prompts

### Expected Behavior Patterns

**Non-streaming vs Streaming:**
- Non-streaming tool calling tends to be more reliable
- Streaming tool calling may be less consistent depending on the model
- Some models may not call tools as reliably in streaming mode

**Timeout Considerations:**
- Local models (especially on CPU) can be slow
- Default timeout increased to 60 seconds for slow models
- Some tests gracefully handle timeouts for very slow responses

**Tool Calling Reliability:**
- Not all models will call tools for every request
- Tests handle cases where models choose not to use tools
- Streaming tests are more tolerant of inconsistent tool calling behavior

## Test Validation

### What's Tested

- **Request Transformation**: Tools properly injected into prompts
- **Response Parsing**: JSON tool calls correctly extracted
- **Tool Selection**: Appropriate tool chosen for the query
- **Argument Extraction**: Correct parameters passed to tools
- **Streaming Behavior**: Real-time tool detection works
- **Error Handling**: Graceful handling of edge cases

### Common Assertions

- Response structure validation
- Tool call ID generation
- JSON argument validity
- Content vs tool call separation
- Streaming chunk consistency

## Troubleshooting

### Tests Failing

1. **Connection Issues**
   ```bash
   # Verify your LLM service is running
   curl -X POST http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"your-model","messages":[{"role":"user","content":"test"}]}'
   ```

2. **Model Not Calling Tools**
   - Check if model supports function calling via prompts
   - Verify prompt template is appropriate for your model
   - Try with verbose logging: `go run -tags e2e main.go -verbose`

3. **Timeouts**
   - Increase timeout: `E2E_TIMEOUT_SECONDS=60`
   - Check model response speed
   - Verify network connectivity

4. **Parsing Errors**
   - Enable debug logging in testutils
   - Check model's JSON output format
   - Verify model follows expected tool calling format

### Debugging

```bash
# Run with verbose test output
go test -tags e2e ./e2e -v -run TestToolCallingNonStreaming

# Use the interactive tool for debugging
go run -tags e2e main.go -verbose

# Check specific edge cases
go test -tags e2e ./e2e -run TestEdgeCases -v
```

## Adding New Tests

### Test Naming Convention

- Non-streaming: `Test*NonStreaming`
- Streaming: `TestStreaming*`
- Edge cases: `TestEdgeCases*` or `Test*EdgeCases`

### Common Patterns

```go
func TestNewScenario(t *testing.T) {
    client := NewTestClient()
    ctx, cancel := client.CreateTimeoutContext()
    defer cancel()
    
    // Create request
    request := client.CreateToolRequest("query", []openai.ChatCompletionToolParam{tool})
    
    // Send and validate
    response, err := client.SendRequest(ctx, request)
    require.NoError(t, err)
    
    // Assertions
    assert.NotEmpty(t, response.Choices[0].Message.ToolCalls)
}
```

### Best Practices

1. **Use Test Utilities**: Leverage `NewTestClient()` and helper functions
2. **Handle Timeouts**: Always use context with appropriate timeouts
3. **Validate Structure**: Check both success and failure scenarios
4. **Log Information**: Use `t.Logf()` for debugging information
5. **Test Edge Cases**: Include error conditions and boundary cases

## Contributing

When adding new tests:

1. Follow existing naming conventions
2. Add appropriate assertions
3. Include error handling
4. Update this README if adding new test categories
5. Test against multiple models when possible

The e2e test suite is designed to be comprehensive yet flexible, supporting various LLM backends while maintaining consistent validation of the adapter's core functionality.