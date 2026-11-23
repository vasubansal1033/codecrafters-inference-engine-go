package gpt2

import (
	"encoding/json"

	"github.com/codecrafters-io/inference-engine-starter-go/internal/nn"
	"github.com/codecrafters-io/inference-engine-starter-go/internal/tensor"
)

type Model struct {
	config  GPT2Config
	weights map[string]tensor.Tensor
}

func New(configData []byte, weights map[string]tensor.Tensor) (*Model, error) {
	var config GPT2Config
	if err := json.Unmarshal(configData, &config); err != nil {
		return nil, err
	}
	// Avoiding a constant row divide by zero in test stages.
	if config.LayerNormEpsilon == 0 {
		config.LayerNormEpsilon = 1e-5
	}
	return &Model{config: config, weights: weights}, nil
}

func (m *Model) NumLayers() int {
	return m.config.NumberOfLayers
}

func (m *Model) Embed(ids []int) (tensor.Tensor, error) {
	return EmbedTokenIDs(m.weights, ids)
}

func (m *Model) Attention(layer int, input tensor.Tensor) (tensor.Tensor, error) {
	return CalculateAttention(m.weights, m.config, layer, input)
}

func (m *Model) Block(layer int, input tensor.Tensor) (tensor.Tensor, error) {
	return CalculateBlock(m.weights, m.config, layer, input)
}

func (m *Model) Head(last tensor.Tensor) (tensor.Tensor, error) {
	normalized, err := (nn.LayerNorm{
		W:   m.weights["ln_f.weight"],
		B:   m.weights["ln_f.bias"],
		Eps: m.config.LayerNormEpsilon,
	}).Forward(last)
	if err != nil {
		return tensor.Tensor{}, err
	}
	embeddingWeight, err := m.weights["wte.weight"].Transpose()
	if err != nil {
		return tensor.Tensor{}, err
	}
	return (nn.Linear{W: embeddingWeight}).Forward(normalized)
}
