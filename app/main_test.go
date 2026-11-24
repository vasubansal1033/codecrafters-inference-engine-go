package main

import (
	"fmt"
	"github.com/codecrafters-io/inference-engine-starter-go/internal/generate"
	"github.com/codecrafters-io/inference-engine-starter-go/internal/models"
	"github.com/codecrafters-io/inference-engine-starter-go/internal/models/gpt2"
	"github.com/codecrafters-io/inference-engine-starter-go/internal/safetensors"
	tensorpkg "github.com/codecrafters-io/inference-engine-starter-go/internal/tensor"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func closeValues(t *testing.T, got, want []float32, tolerance float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if diff := float32(math.Abs(float64(got[i] - want[i]))); diff > tolerance {
			t.Fatalf("value[%d] = %v, want %v (tolerance %v)", i, got[i], want[i], tolerance)
		}
	}
}

func TestTensorOperations(t *testing.T) {
	matrix := tensorpkg.Tensor{DType: "F32", Shape: []int{2, 3}, Values: []float32{1, 2, 3, 4, 5, 6}}
	vector := tensorpkg.Tensor{DType: "F32", Shape: []int{3}, Values: []float32{10, 20, 30}}

	added, err := matrix.Add(vector)
	if err != nil {
		t.Fatal(err)
	}
	closeValues(t, added.Values, []float32{11, 22, 33, 14, 25, 36}, 0)

	multiplied, err := matrix.BroadcastMultiply(vector)
	if err != nil {
		t.Fatal(err)
	}
	closeValues(t, multiplied.Values, []float32{10, 40, 90, 40, 100, 180}, 0)

	column := tensorpkg.Tensor{DType: "F32", Shape: []int{3, 1}, Values: []float32{10, 20, 30}}
	product, err := matrix.MatMul(column)
	if err != nil {
		t.Fatal(err)
	}
	closeValues(t, product.Values, []float32{140, 320}, 0)

	transposed, err := matrix.Transpose()
	if err != nil {
		t.Fatal(err)
	}
	closeValues(t, transposed.Values, []float32{1, 4, 2, 5, 3, 6}, 0)

	probabilities, err := (tensorpkg.Tensor{DType: "F32", Shape: []int{1, 2}, Values: []float32{0, 1}}).Softmax(1, 1)
	if err != nil {
		t.Fatal(err)
	}
	closeValues(t, probabilities.Values, []float32{0.26894143, 0.7310586}, 1e-6)

	normalized, err := (tensorpkg.Tensor{DType: "F32", Shape: []int{1, 2}, Values: []float32{1, 2}}).LayerNorm(1, 1e-5)
	if err != nil {
		t.Fatal(err)
	}
	closeValues(t, normalized.Values, []float32{-0.99998, 0.99998}, 1e-5)
}

func attentionWeights(nEmbeddings int) map[string]tensorpkg.Tensor {
	qkv := make([]float32, nEmbeddings*3*nEmbeddings)
	for i := 0; i < nEmbeddings; i++ {
		qkv[i*3*nEmbeddings+i] = 1
		qkv[i*3*nEmbeddings+nEmbeddings+i] = 1
		qkv[i*3*nEmbeddings+2*nEmbeddings+i] = 1
	}
	proj := make([]float32, nEmbeddings*nEmbeddings)
	for i := 0; i < nEmbeddings; i++ {
		proj[i*nEmbeddings+i] = 1
	}
	return map[string]tensorpkg.Tensor{
		"h.0.attn.c_attn.weight": {DType: "F32", Shape: []int{nEmbeddings, 3 * nEmbeddings}, Values: qkv},
		"h.0.attn.c_attn.bias":   {DType: "F32", Shape: []int{3 * nEmbeddings}, Values: make([]float32, 3*nEmbeddings)},
		"h.0.attn.c_proj.weight": {DType: "F32", Shape: []int{nEmbeddings, nEmbeddings}, Values: proj},
		"h.0.attn.c_proj.bias":   {DType: "F32", Shape: []int{nEmbeddings}, Values: make([]float32, nEmbeddings)},
	}
}

func newTestModel(t *testing.T, nEmbeddings, nHeads int, weights map[string]tensorpkg.Tensor) *gpt2.Model {
	t.Helper()
	config := fmt.Sprintf(`{"n_embd":%d,"n_head":%d,"n_layer":1,"layer_norm_epsilon":1e-5}`, nEmbeddings, nHeads)
	model, err := gpt2.New([]byte(config), weights)
	if err != nil {
		t.Fatal(err)
	}
	return model
}

func TestAttentionGolden(t *testing.T) {
	model := newTestModel(t, 1, 1, attentionWeights(1))
	got, err := model.Attention(0, tensorpkg.Tensor{DType: "F32", Shape: []int{2, 1}, Values: []float32{1, 2}})

	if err != nil {
		t.Fatal(err)
	}
	closeValues(t, got.Values, []float32{1, 1.880797}, 1e-6)
}

func TestTwoHeadAttentionGolden(t *testing.T) {
	model := newTestModel(t, 2, 2, attentionWeights(2))
	got, err := model.Attention(0, tensorpkg.Tensor{DType: "F32", Shape: []int{2, 2}, Values: []float32{1, 0, 0, 1}})

	if err != nil {
		t.Fatal(err)
	}
	closeValues(t, got.Values, []float32{1, 0, 0.5, 0.7310586}, 1e-6)
}

func TestBlockGolden(t *testing.T) {
	weights := attentionWeights(2)
	weights["h.0.ln_1.weight"] = tensorpkg.Tensor{DType: "F32", Shape: []int{2}, Values: []float32{1, 1}}
	weights["h.0.ln_1.bias"] = tensorpkg.Tensor{DType: "F32", Shape: []int{2}, Values: []float32{0, 0}}
	weights["h.0.ln_2.weight"] = tensorpkg.Tensor{DType: "F32", Shape: []int{2}, Values: []float32{1, 1}}
	weights["h.0.ln_2.bias"] = tensorpkg.Tensor{DType: "F32", Shape: []int{2}, Values: []float32{0, 0}}
	weights["h.0.mlp.c_fc.weight"] = tensorpkg.Tensor{
		DType: "F32", Shape: []int{2, 8},
		Values: []float32{1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
	}
	weights["h.0.mlp.c_fc.bias"] = tensorpkg.Tensor{DType: "F32", Shape: []int{8}, Values: make([]float32, 8)}
	weights["h.0.mlp.c_proj.weight"] = tensorpkg.Tensor{
		DType: "F32", Shape: []int{8, 2},
		Values: []float32{1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	}
	weights["h.0.mlp.c_proj.bias"] = tensorpkg.Tensor{DType: "F32", Shape: []int{2}, Values: []float32{0, 0}}

	model := newTestModel(t, 2, 1, weights)
	got, err := model.Block(0, tensorpkg.Tensor{DType: "F32", Shape: []int{1, 2}, Values: []float32{1, 2}})

	if err != nil {
		t.Fatal(err)
	}
	closeValues(t, got.Values, []float32{-0.15878819, 3.8411696}, 1e-4)
}

func TestRealModelLogitsAndGenerate(t *testing.T) {
	modelPath := os.Getenv("GPT2_MODEL")
	if modelPath == "" {
		t.Skip("set GPT2_MODEL to run the real-model regression")
	}
	weights, err := safetensors.Load(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	config, err := gpt2.LoadConfig(filepath.Join(filepath.Dir(modelPath), "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	input, err := gpt2.EmbedTokenIDs(weights, []int{464, 3139, 286, 4881, 318})
	if err != nil {
		t.Fatal(err)
	}
	model, err := models.Load(filepath.Join(filepath.Dir(modelPath), "config.json"), weights)
	if err != nil {
		t.Fatal(err)
	}
	logits, err := models.Logits(model, input)
	if err != nil {
		t.Fatal(err)
	}
	if len(logits.Values) != config.VocabularySize {
		t.Fatalf("logits length = %d, want %d", len(logits.Values), config.VocabularySize)
	}
	generated, err := generate.Greedy(model, []int{464, 3139, 286, 4881, 318}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(generated) != 1 {
		t.Fatalf("generated %d tokens, want 1", len(generated))
	}
}
