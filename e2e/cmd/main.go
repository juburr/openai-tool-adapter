//go:build e2e

package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"log/slog"

	tooladapter "github.com/juburr/openai-tool-adapter/v3"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

type WeatherRequest struct {
	Location string `json:"location" description:"The location to get weather for"`
	Unit     string `json:"unit,omitempty" description:"Temperature unit: celsius or fahrenheit"`
}

type WeatherResponse struct {
	Location    string  `json:"location"`
	Temperature float64 `json:"temperature"`
	Unit        string  `json:"unit"`
	Description string  `json:"description"`
	Humidity    int     `json:"humidity"`
}

func getWeather(location, unit string) WeatherResponse {
	tempCelsius := 22.0
	if unit == "fahrenheit" {
		tempCelsius = tempCelsius*9/5 + 32
	}

	if unit == "" {
		unit = "celsius"
	}

	return WeatherResponse{
		Location:    location,
		Temperature: tempCelsius,
		Unit:        unit,
		Description: "Partly cloudy with light winds",
		Humidity:    65,
	}
}

func createWeatherTool() openai.ChatCompletionToolUnionParam {
	return openai.ChatCompletionFunctionTool(
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

func printOriginalRequest(params openai.ChatCompletionNewParams, verbose bool) {
	if verbose {
		fmt.Println("=== Original Request ===")
		fmt.Printf("Model: %s\n", params.Model)
		fmt.Printf("Messages: %+v\n", params.Messages)
		fmt.Printf("Tools: %+v\n", params.Tools)
	}
}

func printTransformedRequest(params openai.ChatCompletionNewParams, verbose bool) {
	if verbose {
		fmt.Println("\n=== Transformed Request ===")
		fmt.Printf("Model: %s\n", params.Model)
		fmt.Printf("Tools: %+v\n", params.Tools)
		for i, msg := range params.Messages {
			fmt.Printf("Message %d: %+v\n", i, msg)
		}
	}
}

func printRawResponse(completion *openai.ChatCompletion, verbose bool) {
	if verbose {
		fmt.Println("\n=== Raw LLM Response ===")
		fmt.Printf("Response: %+v\n", completion)
		if len(completion.Choices) > 0 {
			content := completion.Choices[0].Message.Content
			fmt.Printf("Content: %s\n", content)
		}
	}
}

func printTransformedResponse(completion openai.ChatCompletion, verbose bool) {
	if verbose {
		fmt.Println("\n=== Transformed Response ===")
		fmt.Printf("Response: %+v\n", completion)
	}
}

func processToolCalls(completion openai.ChatCompletion, verbose bool) {
	if len(completion.Choices) == 0 || len(completion.Choices[0].Message.ToolCalls) == 0 {
		if len(completion.Choices) > 0 {
			content := completion.Choices[0].Message.Content
			fmt.Printf("Assistant: %s\n", content)
		}
		return
	}

	if verbose {
		fmt.Println("\n=== Tool Calls Detected ===")
		for i, toolCall := range completion.Choices[0].Message.ToolCalls {
			fmt.Printf("Tool Call %d:\n", i)
			fmt.Printf("  ID: %s\n", toolCall.ID)
			fmt.Printf("  Function: %s\n", toolCall.Function.Name)
			fmt.Printf("  Arguments: %s\n", toolCall.Function.Arguments)
		}
	}

	for _, toolCall := range completion.Choices[0].Message.ToolCalls {
		processWeatherToolCall(toolCall, verbose)
	}
}

func processWeatherToolCall(toolCall openai.ChatCompletionMessageToolCall, verbose bool) {
	if toolCall.Function.Name != "get_weather" {
		return
	}

	var req WeatherRequest
	if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &req); err != nil {
		log.Printf("Failed to parse weather request: %v", err)
		return
	}

	weather := getWeather(req.Location, req.Unit)
	if verbose {
		fmt.Printf("  Result: %+v\n", weather)
	} else {
		fmt.Printf("Weather in %s: %.1f°%s, %s (Humidity: %d%%)\n",
			weather.Location, weather.Temperature, weather.Unit, weather.Description, weather.Humidity)
	}
}

func main() {
	verbose := flag.Bool("verbose", false, "Enable verbose output")
	flag.Parse()

	client := openai.NewClient(
		option.WithBaseURL("http://localhost:8000/v1"),
		option.WithAPIKey("dummy-key"),
	)

	logLevel := slog.LevelError
	if *verbose {
		logLevel = slog.LevelDebug
	}
	adapter := tooladapter.New(tooladapter.WithLogLevel(logLevel))

	weatherTool := createWeatherTool()

	originalParams := openai.ChatCompletionNewParams{
		Model: "google/gemma-3-4b-it",
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("What's the weather like in San Francisco?"),
		},
		Tools: []openai.ChatCompletionToolUnionParam{weatherTool},
	}

	printOriginalRequest(originalParams, *verbose)

	transformedParams, err := adapter.TransformCompletionsRequest(originalParams)
	if err != nil {
		log.Fatalf("Failed to transform request: %v", err)
	}

	printTransformedRequest(transformedParams, *verbose)

	completion, err := client.Chat.Completions.New(context.Background(), transformedParams)
	if err != nil {
		log.Fatalf("Failed to create completion: %v", err)
	}

	printRawResponse(completion, *verbose)

	transformedCompletion, err := adapter.TransformCompletionsResponse(*completion)
	if err != nil {
		log.Fatalf("Failed to transform response: %v", err)
	}

	printTransformedResponse(transformedCompletion, *verbose)

	processToolCalls(transformedCompletion, *verbose)
}
