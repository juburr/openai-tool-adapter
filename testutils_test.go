package tooladapter_test

import (
	"github.com/openai/openai-go"
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
func createMockTool(name, description string) openai.ChatCompletionToolParam {
	tool := openai.ChatCompletionToolParam{
		Type: "function",
		Function: openai.FunctionDefinitionParam{
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
		},
	}

	// Only set description if it's not empty
	if description != "" {
		tool.Function.Description = openai.String(description)
	}

	return tool
}

func createMockRequest(tools []openai.ChatCompletionToolParam) openai.ChatCompletionNewParams {
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
