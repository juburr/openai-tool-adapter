//go:build e2e

package e2e

import (
	"context"
	"log/slog"
	"time"

	tooladapter "github.com/juburr/openai-tool-adapter/v2"
	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
)

// TestClient wraps OpenAI client and adapter for testing
type TestClient struct {
	client  openai.Client
	adapter *tooladapter.Adapter
	config  TestConfig
}

// NewTestClient creates a new test client with configuration
func NewTestClient() *TestClient {
	config := LoadTestConfig()

	client := openai.NewClient(
		option.WithBaseURL(config.BaseURL),
		option.WithAPIKey(config.APIKey),
	)

	adapter := tooladapter.New(tooladapter.WithLogLevel(slog.LevelError))

	return &TestClient{
		client:  client,
		adapter: adapter,
		config:  config,
	}
}

// NewTestClientWithVerboseLogging creates a test client with debug logging
func NewTestClientWithVerboseLogging() *TestClient {
	config := LoadTestConfig()

	client := openai.NewClient(
		option.WithBaseURL(config.BaseURL),
		option.WithAPIKey(config.APIKey),
	)

	adapter := tooladapter.New(tooladapter.WithLogLevel(slog.LevelDebug))

	return &TestClient{
		client:  client,
		adapter: adapter,
		config:  config,
	}
}

// CreateWeatherTool creates a standard weather tool for testing
func CreateWeatherTool() openai.ChatCompletionToolParam {
	return openai.ChatCompletionToolParam{
		Type: "function",
		openai.FunctionDefinitionParam{
			Name:        "get_weather",
			Description: openai.String("Get current weather information for a specific location"),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]interface{}{
					"location": map[string]interface{}{
						"type":        "string",
						"description": "The location to get weather for",
					},
					"unit": map[string]interface{}{
						"type":        "string",
						"enum":        []string{"celsius", "fahrenheit"},
						"description": "Temperature unit: celsius or fahrenheit",
					},
				},
				"required": []string{"location"},
			},
		},
	}
}

// CreateCalculatorTool creates a calculator tool for testing
func CreateCalculatorTool() openai.ChatCompletionToolParam {
	return openai.ChatCompletionToolParam{
		Type: "function",
		openai.FunctionDefinitionParam{
			Name:        "calculate",
			Description: openai.String("Perform basic arithmetic calculations"),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]interface{}{
					"expression": map[string]interface{}{
						"type":        "string",
						"description": "Mathematical expression to evaluate (e.g., '2+2', '10*5')",
					},
				},
				"required": []string{"expression"},
			},
		},
	}
}

// CreateTimeTool creates a time/date tool for testing
func CreateTimeTool() openai.ChatCompletionToolParam {
	return openai.ChatCompletionToolParam{
		Type: "function",
		openai.FunctionDefinitionParam{
			Name:        "get_time",
			Description: openai.String("Get current time and date information"),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]interface{}{
					"timezone": map[string]interface{}{
						"type":        "string",
						"description": "Timezone (e.g., 'UTC', 'EST', 'PST')",
						"default":     "UTC",
					},
					"format": map[string]interface{}{
						"type":        "string",
						"description": "Time format ('12h' or '24h')",
						"enum":        []string{"12h", "24h"},
						"default":     "24h",
					},
				},
				"required": []string{},
			},
		},
	}
}

// CreateTranslationTool creates a translation tool for testing
func CreateTranslationTool() openai.ChatCompletionToolParam {
	return openai.ChatCompletionToolParam{
		Type: "function",
		openai.FunctionDefinitionParam{
			Name:        "translate_text",
			Description: openai.String("Translate text between different languages"),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]interface{}{
					"text": map[string]interface{}{
						"type":        "string",
						"description": "Text to translate",
					},
					"from_lang": map[string]interface{}{
						"type":        "string",
						"description": "Source language code (e.g., 'en', 'es', 'fr')",
					},
					"to_lang": map[string]interface{}{
						"type":        "string",
						"description": "Target language code (e.g., 'en', 'es', 'fr')",
					},
				},
				"required": []string{"text", "from_lang", "to_lang"},
			},
		},
	}
}

// CreateSearchTool creates a search tool for testing
func CreateSearchTool() openai.ChatCompletionToolParam {
	return openai.ChatCompletionToolParam{
		Type: "function",
		openai.FunctionDefinitionParam{
			Name:        "web_search",
			Description: openai.String("Search the web for information"),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "Search query string",
					},
					"limit": map[string]interface{}{
						"type":        "integer",
						"description": "Maximum number of results (1-10)",
						"minimum":     1,
						"maximum":     10,
						"default":     5,
					},
				},
				"required": []string{"query"},
			},
		},
	}
}

// CreateBasicRequest creates a basic chat request without tools
func (tc *TestClient) CreateBasicRequest(message string) openai.ChatCompletionNewParams {
	return openai.ChatCompletionNewParams{
		Model: tc.config.Model,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage(message),
		},
	}
}

// CreateToolRequest creates a chat request with tools
func (tc *TestClient) CreateToolRequest(message string, tools []openai.ChatCompletionToolParam) openai.ChatCompletionNewParams {
	return openai.ChatCompletionNewParams{
		Model: tc.config.Model,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage(message),
		},
		Tools: tools,
	}
}

// SendRequest sends a request and returns the response
func (tc *TestClient) SendRequest(ctx context.Context, request openai.ChatCompletionNewParams) (*openai.ChatCompletion, error) {
	transformedRequest, err := tc.adapter.TransformCompletionsRequest(request)
	if err != nil {
		return nil, err
	}

	response, err := tc.client.Chat.Completions.New(ctx, transformedRequest)
	if err != nil {
		return nil, err
	}

	transformedResponse, err := tc.adapter.TransformCompletionsResponse(*response)
	if err != nil {
		return nil, err
	}

	return &transformedResponse, nil
}

// SendStreamingRequest sends a streaming request and returns the adapter
func (tc *TestClient) SendStreamingRequest(ctx context.Context, request openai.ChatCompletionNewParams) (*tooladapter.StreamAdapter, error) {
	transformedRequest, err := tc.adapter.TransformCompletionsRequest(request)
	if err != nil {
		return nil, err
	}

	stream := tc.client.Chat.Completions.NewStreaming(ctx, transformedRequest)
	return tc.adapter.TransformStreamingResponseWithContext(ctx, stream), nil
}

// CreateTimeoutContext creates a context with the configured timeout
func (tc *TestClient) CreateTimeoutContext() (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), time.Duration(tc.config.Timeout)*time.Second)
}
