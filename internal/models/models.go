// Package models defines the model-independent inference interface and registry.
package models

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/codecrafters-io/inference-engine-starter-go/internal/models/gpt2"
	"github.com/codecrafters-io/inference-engine-starter-go/internal/tensor"
)

type Model interface {
	NumLayers() int
	Embed(ids []int) (tensor.Tensor, error)
	Attention(layer int, input tensor.Tensor) (tensor.Tensor, error)
	Block(layer int, input tensor.Tensor) (tensor.Tensor, error)
	Head(last tensor.Tensor) (tensor.Tensor, error)
}

type Builder func(config []byte, weights map[string]tensor.Tensor) (Model, error)

var builders = map[string]Builder{
	"gpt2": func(config []byte, weights map[string]tensor.Tensor) (Model, error) {
		return gpt2.New(config, weights)
	},
}

func Load(configPath string, weights map[string]tensor.Tensor) (Model, error) {
	config, err := os.ReadFile(configPath)
	if err != nil {
		return nil, err
	}
	var metadata struct {
		ModelType string `json:"model_type"`
	}
	if err := json.Unmarshal(config, &metadata); err != nil {
		return nil, err
	}
	builder, ok := builders[metadata.ModelType]
	if !ok {
		return nil, fmt.Errorf("unsupported model_type %q", metadata.ModelType)
	}
	return builder(config, weights)
}

func Logits(model Model, input tensor.Tensor) (tensor.Tensor, error) {
	current := input
	for layer := range model.NumLayers() {
		next, err := model.Block(layer, current)
		if err != nil {
			return tensor.Tensor{}, err
		}
		current = next
	}
	if len(current.Shape) != 2 || current.Shape[0] == 0 {
		return tensor.Tensor{}, fmt.Errorf("invalid final hidden state shape %v", current.Shape)
	}
	width := current.Shape[1]
	start := (current.Shape[0] - 1) * width
	last := tensor.Tensor{
		DType:  current.DType,
		Shape:  []int{1, width},
		Values: current.Values[start : start+width],
	}
	return model.Head(last)
}
