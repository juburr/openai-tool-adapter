package tooladapter_test

import (
	"github.com/openai/openai-go/v3"
)

// MockChatCompletionStream provides a simple mock for testing streaming functionality
type MockChatCompletionStream struct {
	chunks  []openai.ChatCompletionChunk
	current int
	err     error
}

func NewMockStream(chunks []openai.ChatCompletionChunk) *MockChatCompletionStream {
	return &MockChatCompletionStream{
		chunks:  chunks,
		current: -1,
	}
}

func (m *MockChatCompletionStream) Next() bool {
	m.current++
	return m.current < len(m.chunks)
}

func (m *MockChatCompletionStream) Current() openai.ChatCompletionChunk {
	if m.current >= 0 && m.current < len(m.chunks) {
		return m.chunks[m.current]
	}
	return openai.ChatCompletionChunk{}
}

func (m *MockChatCompletionStream) Err() error {
	return m.err
}

func (m *MockChatCompletionStream) Close() error {
	return nil
}

func (m *MockChatCompletionStream) SetError(err error) {
	m.err = err
}

// Common test helper functions
func createMockTool(name, description string) openai.ChatCompletionToolUnionParam {
	functionDef := openai.FunctionDefinitionParam{
		Name: name,
		Parameters: openai.FunctionParameters{
			"type": "object",
			"properties": map[string]interface{}{
				"param1": map[string]interface{}{
					"type":        "string",
					"description": "A parameter",
				},
			},
		},
	}

	// Set description if it's not empty
	if description != "" {
		functionDef.Description = openai.String(description)
	}

	return openai.ChatCompletionFunctionTool(functionDef)
}

func createMockRequest(tools []openai.ChatCompletionToolUnionParam) openai.ChatCompletionNewParams {
	return openai.ChatCompletionNewParams{
		Model: openai.ChatModelGPT4o,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("Hello, please help me."),
		},
		Tools: tools,
	}
}

func createMockCompletion(content string) openai.ChatCompletion {
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
