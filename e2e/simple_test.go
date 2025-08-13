//go:build e2e

package e2e

import (
	"testing"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/stretchr/testify/assert"
)

func TestSimpleCompilation(t *testing.T) {
	config := LoadTestConfig()

	client := openai.NewClient(
		option.WithBaseURL(config.BaseURL),
		option.WithAPIKey(config.APIKey),
	)

	assert.NotNil(t, client)
}
