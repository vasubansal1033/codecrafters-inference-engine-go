// Package models defines the model-independent inference interface and registry.
package models

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/codecrafters-io/inference-engine-starter-go/internal/models/gpt2"
	kvcache "github.com/codecrafters-io/inference-engine-starter-go/internal/models/kv_cache"
	"github.com/codecrafters-io/inference-engine-starter-go/internal/tensor"
)

type Model interface {
	NumLayers() int
	Embed(ids []int) (tensor.Tensor, error)
	EmbedAt(ids []int, position int) (tensor.Tensor, error)
	Attention(layer int, input tensor.Tensor) (tensor.Tensor, error)
	Block(layer int, input tensor.Tensor) (tensor.Tensor, error)
	Head(last tensor.Tensor) (tensor.Tensor, error)
	Forward(input tensor.Tensor, cache *kvcache.KVCache) (tensor.Tensor, error)
	GetKVCache() (*kvcache.KVCache, error)
}

type ModelBuilder func(config []byte, weights map[string]tensor.Tensor) (Model, error)

var modelBuilder = map[string]ModelBuilder{
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
	if metadata.ModelType == "" {
		metadata.ModelType = "gpt2"
	}
	model, ok := modelBuilder[metadata.ModelType]
	if !ok {
		return nil, fmt.Errorf("unsupported model_type %q", metadata.ModelType)
	}
	return model(config, weights)
}

func Logits(model Model, input tensor.Tensor) (tensor.Tensor, error) {
	return model.Forward(input, nil)
}
