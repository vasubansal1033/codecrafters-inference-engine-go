// Package cli implements the inference engine command line.
package cli

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"

	"github.com/codecrafters-io/inference-engine-starter-go/internal/generate"
	"github.com/codecrafters-io/inference-engine-starter-go/internal/models"
	"github.com/codecrafters-io/inference-engine-starter-go/internal/models/gpt2"
	"github.com/codecrafters-io/inference-engine-starter-go/internal/safetensors"
	"github.com/codecrafters-io/inference-engine-starter-go/internal/tensor"
)

type command struct {
	args  int
	usage string
	run   func([]string, io.Writer) error
}

var commands = map[string]command{
	"inspect":  {1, "inspect model.safetensors", runInspect},
	"embed":    {2, "embed model.safetensors ids.json", runEmbed},
	"attn":     {4, "attn model.safetensors config.json <layer_num> input.json", runAttention},
	"block":    {4, "block model.safetensors config.json <layer_num> input.json", runBlock},
	"logits":   {3, "logits model.safetensors config.json ids.json", runLogits},
	"generate": {4, "generate model.safetensors config.json ids.json tokens.json", runGenerate},
}

func Run(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		fmt.Fprintln(stderr, "usage: your_program.sh <command> [<args>...]")
		return 1
	}
	selected, ok := commands[args[0]]
	if !ok {
		fmt.Fprintf(stderr, "Unknown command %s\n", args[0])
		return 1
	}
	if len(args)-1 < selected.args {
		fmt.Fprintf(stderr, "usage: your_program.sh %s\n", selected.usage)
		return 1
	}
	if err := selected.run(args[1:], stdout); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	return 0
}

func runInspect(args []string, output io.Writer) error {
	weights, err := safetensors.Load(args[0])
	if err != nil {
		return fmt.Errorf("Error reading tensors: %w", err)
	}
	names := make([]string, 0, len(weights))
	for name := range weights {
		names = append(names, name)
	}
	sort.Strings(names)
	type summary struct {
		Name  string    `json:"name"`
		DType string    `json:"dtype"`
		Shape []int     `json:"shape"`
		First []float32 `json:"first"`
	}
	summaries := make([]summary, 0, len(names))
	for _, name := range names {
		value := weights[name]
		summaries = append(summaries, summary{name, value.DType, value.Shape, value.Values[:min(4, len(value.Values))]})
	}
	return writeJSON(output, summaries)
}

func runEmbed(args []string, output io.Writer) error {
	weights, err := safetensors.Load(args[0])
	if err != nil {
		return fmt.Errorf("Error reading tensors: %w", err)
	}
	ids, err := readJSON[[]int](args[1])
	if err != nil {
		return fmt.Errorf("Error reading token ids: %w", err)
	}
	embedded, err := gpt2.EmbedTokenIDs(weights, ids)
	if err != nil {
		return fmt.Errorf("Error embedding tokens: %w", err)
	}
	return writeJSON(output, embedded.Rows())
}

func loadModel(modelPath, configPath string) (models.Model, error) {
	weights, err := safetensors.Load(modelPath)
	if err != nil {
		return nil, fmt.Errorf("Error reading tensors: %w", err)
	}
	model, err := models.Load(configPath, weights)
	if err != nil {
		return nil, fmt.Errorf("Error while loading configs: %w", err)
	}
	return model, nil
}

func runAttention(args []string, output io.Writer) error {
	model, err := loadModel(args[0], args[1])
	if err != nil {
		return err
	}
	layer, err := strconv.Atoi(args[2])
	if err != nil {
		return fmt.Errorf("Error while parsing layer number: %w", err)
	}
	input, err := loadInputRows(args[3])
	if err != nil {
		return fmt.Errorf("Error while loading inputs: %w", err)
	}
	result, err := model.Attention(layer, input)
	if err != nil {
		return fmt.Errorf("Error while calculating attention: %w", err)
	}
	return writeJSON(output, result.Rows())
}

func runBlock(args []string, output io.Writer) error {
	model, err := loadModel(args[0], args[1])
	if err != nil {
		return err
	}
	layer, err := strconv.Atoi(args[2])
	if err != nil {
		return fmt.Errorf("Error while parsing layer number: %w", err)
	}
	input, err := loadInputRows(args[3])
	if err != nil {
		return fmt.Errorf("Error while loading inputs: %w", err)
	}
	result, err := model.Block(layer, input)
	if err != nil {
		return fmt.Errorf("Error while calculating block: %w", err)
	}
	return writeJSON(output, result.Rows())
}

func runLogits(args []string, output io.Writer) error {
	model, err := loadModel(args[0], args[1])
	if err != nil {
		return err
	}
	ids, err := readJSON[[]int](args[2])
	if err != nil {
		return fmt.Errorf("Error while loading ids: %w", err)
	}
	input, err := model.Embed(ids)
	if err != nil {
		return fmt.Errorf("Error embedding tokens: %w", err)
	}
	result, err := models.Logits(model, input)
	if err != nil {
		return fmt.Errorf("Error while calculating logits: %w", err)
	}
	return writeJSON(output, result.Rows()[0])
}

func runGenerate(args []string, output io.Writer) error {
	model, err := loadModel(args[0], args[1])
	if err != nil {
		return err
	}
	ids, err := readJSON[[]int](args[2])
	if err != nil {
		return fmt.Errorf("Error while loading ids: %w", err)
	}
	config, err := readJSON[tokenConfig](args[3])
	if err != nil {
		return fmt.Errorf("Error while loading token configs: %w", err)
	}
	result, err := generate.Greedy(model, ids, config.Count)
	if err != nil {
		return fmt.Errorf("Error while performing generation: %w", err)
	}
	return writeJSON(output, result)
}

type tokenConfig struct {
	Count       int     `json:"count"`
	Temperature float32 `json:"temperature"`
	TopP        float32 `json:"top_p"`
}

func loadInputRows(path string) (tensor.Tensor, error) {
	rows, err := readJSON[[][]float32](path)
	if err != nil {
		return tensor.Tensor{}, err
	}
	input := tensor.Tensor{DType: "F32", Shape: []int{len(rows), len(rows[0])}}
	for _, row := range rows {
		input.Values = append(input.Values, row...)
	}
	return input, nil
}

func readJSON[T any](path string) (T, error) {
	var value T
	data, err := os.ReadFile(path)
	if err != nil {
		return value, err
	}
	err = json.Unmarshal(data, &value)
	return value, err
}

func writeJSON(output io.Writer, value any) error {
	encoded, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("Error encoding output: %w", err)
	}
	fmt.Fprintln(output, string(encoded))
	return nil
}
