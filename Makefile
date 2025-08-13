.PHONY: bench check e2e e2e-basic e2e-interactive fuzz fuzz-quick help lint reportcard test test-fast vulncheck 

# Default target
help:
	@echo "Available targets:"
	@echo "  check        - Run Go code quality checks (fmt, vet)"
	@echo "  lint         - Run Go linter (requires golangci-lint)"
	@echo "  test         - Run Go unit tests"
	@echo "  test-fast    - Run Go unit tests (fast version)"
	@echo "  e2e          - Run Go end-to-end tests"
	@echo "  e2e-basic    - Run basic e2e tests only"
	@echo "  e2e-interactive - Run interactive e2e test tool"
	@echo "  bench        - Run Go benchmarks"
	@echo "  fuzz         - Run fuzzing tests"
	@echo "  fuzz-quick   - Run quick fuzzing tests (for CI)"
	@echo "  vulncheck    - Run Go vulnerability checks (requires govulncheck)"
	@echo "  reportcard   - Run Go report card (requires goreportcard-cli)"

# Run Go code quality checks
check:
	@echo "Running code quality checks..."
	go fmt ./...
	go vet ./...

# Run Go linter
# Requires golangci-lint to be installed: https://golangci-lint.run/welcome/install/
lint:
	@echo "Running linter..."
	golangci-lint run --timeout 5m

# Run Go unit tests
test:
	@echo "Running unit tests..."
	go test -v -race ./... -coverprofile=coverage.out
	go tool cover -html=coverage.out -o coverage.html
	@echo "Test coverage report generated: coverage.html"
	@echo "Updating coverage badge in README..."
	@COVERAGE=$$(go tool cover -func=coverage.out | tail -1 | awk '{print $$3}' | sed 's/%//'); \
	sed -i "s/coverage-[0-9.]*%25/coverage-$$COVERAGE%25/g" README.md; \
	echo " - Coverage badge updated to $$COVERAGE%"

# Run Go unit tests (fast version)
test-fast:
	@echo "Running fast unit tests..."
	go test -v -race -short -skip 'TestBufferLimitExceeded*|Fuzz*' ./...

# Run Go vulnerability checks
# Requires govulncheck to be installed: https://go.dev/security/govulncheck
vulncheck:
	@echo "Running Go vulnerability checks..."
	govulncheck ./...

# Run Go report card
reportcard:
	@echo "Running Go report card..."
	goreportcard-cli -v

# Run benchmarks
bench:
	@echo "Running benchmarks..."
	go test -bench=. -benchmem ./...

# Run fuzzing tests
fuzz:
	@echo "Running fuzzing tests..."
	@echo "Testing JSON parser fuzzing (30s)..."
	go test -fuzz=FuzzJSONExtractor -fuzztime=30s
	@echo "Testing function name validation fuzzing (30s)..."
	go test -fuzz=FuzzValidateFunctionName -fuzztime=30s
	@echo "Testing response transformation fuzzing (30s)..."  
	go test -fuzz=FuzzTransformCompletionsResponse -fuzztime=30s
	@echo "All fuzzing tests completed!"

# Run quick fuzzing tests (for CI)
fuzz-quick:
	@echo "Running quick fuzzing tests..."
	go test -fuzz=FuzzJSONExtractor -fuzztime=5s
	go test -fuzz=FuzzValidateFunctionName -fuzztime=5s
	go test -fuzz=FuzzTransformCompletionsResponse -fuzztime=5s

# Run end-to-end tests against an actual vLLM instance
e2e:
	@echo "Running end-to-end tests..."
	@echo "Note: Requires a running LLM service at E2E_BASE_URL (default: http://localhost:8000/v1)"
	cd e2e && go test -tags e2e -v .

# Run basic end-to-end tests only (faster)
e2e-basic:
	@echo "Running basic e2e tests only..."
	@echo "Note: Requires a running LLM service at E2E_BASE_URL (default: http://localhost:8000/v1)"
	cd e2e && go test -tags e2e -v . -run "TestBasic|TestToolCallingNonStreaming"

# Run interactive e2e test tool
e2e-interactive:
	@echo "Running interactive e2e test tool..."
	@echo "Use -verbose flag for detailed output"
	@echo "Configure via environment variables: E2E_BASE_URL, E2E_MODEL, E2E_API_KEY"
	cd e2e && go run -tags e2e cmd/main.go