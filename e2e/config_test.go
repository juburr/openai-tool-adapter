//go:build e2e

package e2e

import (
	"os"
	"strconv"
)

// TestConfig holds configuration for e2e tests
type TestConfig struct {
	BaseURL string
	APIKey  string
	Model   string
	Timeout int // seconds
}

// LoadTestConfig loads configuration from environment variables
func LoadTestConfig() TestConfig {
	config := TestConfig{
		BaseURL: getEnvOrDefault("E2E_BASE_URL", "http://localhost:8000/v1"),
		APIKey:  getEnvOrDefault("E2E_API_KEY", "dummy-key"),
		Model:   getEnvOrDefault("E2E_MODEL", "google/gemma-3-4b-it"),
		Timeout: getEnvIntOrDefault("E2E_TIMEOUT_SECONDS", 60), // Increased for slow models
	}
	return config
}

func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvIntOrDefault(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}
