package gpt2

import (
	"encoding/json"
	"fmt"
	"os"

	kvcache "github.com/codecrafters-io/inference-engine-starter-go/internal/models/kv_cache"
	"github.com/codecrafters-io/inference-engine-starter-go/internal/nn"
	"github.com/codecrafters-io/inference-engine-starter-go/internal/tensor"
)

// ref: https://huggingface.co/openai-community/gpt2/blob/main/config.json
type GPT2Config struct {
	ActivationFunction     string                 `json:"activation_function"`
	Architectures          []string               `json:"architectures"`
	AttentionPDrop         float32                `json:"attn_pdrop"`
	BoSTokenId             int                    `json:"bos_token_id"`
	EmbeddingPDrop         float32                `json:"embd_pdrop"`
	EoSTokenId             int                    `json:"eos_token_id"`
	InitializerRange       float32                `json:"initializer_range"`
	LayerNormEpsilon       float32                `json:"layer_norm_epsilon"`
	ModelType              string                 `json:"model_type"`
	ContextWindowSize      int                    `json:"n_ctx"`
	NumberOfEmbeddings     int                    `json:"n_embd"`
	NumberOfHeads          int                    `json:"n_head"`
	NumberOfLayers         int                    `json:"n_layer"`
	NumberOfPositions      int                    `json:"n_positions"`
	ResidualPDrop          float32                `json:"resid_pdrop"`
	SummaryActivation      interface{}            `json:"summary_activation"`
	SummaryFirstDropout    float32                `json:"summary_first_dropout"`
	SummaryProjectToLabels bool                   `json:"summary_proj_to_labels"`
	SummaryType            string                 `json:"summary_type"`
	SummaryUseProj         bool                   `json:"summary_use_proj"`
	TaskSpecificParams     map[string]interface{} `json:"task_specific_params"`
	VocabularySize         int                    `json:"vocab_size"`
}

type Model struct {
	config  GPT2Config
	weights map[string]tensor.Tensor
	kvCache kvcache.KVCache
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
	return &Model{
		config:  config,
		weights: weights,
		kvCache: kvcache.NewKVCache(),
	}, nil
}

func LoadConfig(path string) (GPT2Config, error) {
	return readJSON[GPT2Config](path)
}

func LoadInputRows(path string) (tensor.Tensor, error) {
	inputRows, err := readJSON[[][]float32](path)
	if err != nil {
		return tensor.Tensor{}, err
	}
	inputTensor := tensor.Tensor{
		DType: "F32",
		Shape: []int{len(inputRows), len(inputRows[0])},
	}
	for i := range inputRows {
		inputTensor.Values = append(inputTensor.Values, inputRows[i]...)
	}
	return inputTensor, nil
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

func (m *Model) NumLayers() int {
	return m.config.NumberOfLayers
}

func (m *Model) Embed(ids []int) (tensor.Tensor, error) {
	return embedTokenIDs(m.weights, ids, 0)
}

func (m *Model) EmbedAt(ids []int, position int) (tensor.Tensor, error) {
	return embedTokenIDs(m.weights, ids, position)
}

// Attention is the attn stage. It recomputes the whole input and does not use the cache.
func (m *Model) Attention(layer int, input tensor.Tensor) (tensor.Tensor, error) {
	return m.calculateAttention(layer, input, nil)
}

// Block is the block stage. It recomputes the whole input and does not use the cache.
func (m *Model) Block(layer int, input tensor.Tensor) (tensor.Tensor, error) {
	return m.calculateBlock(layer, input, nil)
}

// Forward runs every block. Generation passes its cache. Logits passes nil.
func (m *Model) Forward(input tensor.Tensor, cache *kvcache.KVCache) (tensor.Tensor, error) {
	current := input
	for layer := range m.NumLayers() {
		next, err := m.calculateBlock(layer, current, cache)
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
	return m.Head(last)
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

func (m *Model) GetKVCache() (*kvcache.KVCache, error) {
	return &m.kvCache, nil
}
