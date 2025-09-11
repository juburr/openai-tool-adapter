//go:build e2e

module e2e

go 1.24.5

require (
	github.com/juburr/openai-tool-adapter v0.0.0
	github.com/openai/openai-go/v2 v2.4.1
	github.com/stretchr/testify v1.10.0
)

require (
	github.com/davecgh/go-spew v1.1.1 // indirect
	github.com/google/uuid v1.6.0 // indirect
	github.com/kr/text v0.2.0 // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	github.com/tidwall/gjson v1.14.4 // indirect
	github.com/tidwall/match v1.1.1 // indirect
	github.com/tidwall/pretty v1.2.1 // indirect
	github.com/tidwall/sjson v1.2.5 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)

replace github.com/juburr/openai-tool-adapter => ../
