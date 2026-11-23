package models_test

import (
	"math"
	"testing"

	"github.com/codecrafters-io/inference-engine-starter-go/internal/models"
	"github.com/codecrafters-io/inference-engine-starter-go/internal/models/gpt2"
	"github.com/codecrafters-io/inference-engine-starter-go/internal/tensor"
)

func TestGPT2ImplementsModel(t *testing.T) {
	var _ models.Model = (*gpt2.Model)(nil)
}

func TestTiedLogitsStayFiniteWithoutEpsilon(t *testing.T) {
	weights := map[string]tensor.Tensor{
		"wte.weight":             {DType: "F32", Shape: []int{2, 2}, Values: []float32{1, 1, 1, -1}},
		"wpe.weight":             {DType: "F32", Shape: []int{4, 2}, Values: []float32{0, 0, 1, 0, 0, 0, 0, 0}},
		"ln_f.weight":            {DType: "F32", Shape: []int{2}, Values: []float32{0, 0}},
		"ln_f.bias":              {DType: "F32", Shape: []int{2}, Values: []float32{1, -1}},
		"h.0.ln_1.weight":        {DType: "F32", Shape: []int{2}, Values: []float32{1, 1}},
		"h.0.ln_1.bias":          {DType: "F32", Shape: []int{2}, Values: []float32{0, 0}},
		"h.0.attn.c_attn.weight": {DType: "F32", Shape: []int{2, 6}, Values: make([]float32, 12)},
		"h.0.attn.c_attn.bias":   {DType: "F32", Shape: []int{6}, Values: make([]float32, 6)},
		"h.0.attn.c_proj.weight": {DType: "F32", Shape: []int{2, 2}, Values: make([]float32, 4)},
		"h.0.attn.c_proj.bias":   {DType: "F32", Shape: []int{2}, Values: make([]float32, 2)},
		"h.0.ln_2.weight":        {DType: "F32", Shape: []int{2}, Values: []float32{1, 1}},
		"h.0.ln_2.bias":          {DType: "F32", Shape: []int{2}, Values: []float32{0, 0}},
		"h.0.mlp.c_fc.weight":    {DType: "F32", Shape: []int{2, 8}, Values: make([]float32, 16)},
		"h.0.mlp.c_fc.bias":      {DType: "F32", Shape: []int{8}, Values: make([]float32, 8)},
		"h.0.mlp.c_proj.weight":  {DType: "F32", Shape: []int{8, 2}, Values: make([]float32, 16)},
		"h.0.mlp.c_proj.bias":    {DType: "F32", Shape: []int{2}, Values: make([]float32, 2)},
	}
	model, err := gpt2.New([]byte(`{"n_embd":2,"n_head":1,"n_layer":1,"n_positions":4,"vocab_size":2}`), weights)
	if err != nil {
		t.Fatal(err)
	}
	embedded, err := model.Embed([]int{0})
	if err != nil {
		t.Fatal(err)
	}
	logits, err := models.Logits(model, embedded)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{0, 2}
	if len(logits.Values) != len(want) {
		t.Fatalf("logits = %v", logits.Values)
	}
	for i, value := range logits.Values {
		if math.IsNaN(float64(value)) || math.Abs(float64(value-want[i])) > 1e-5 {
			t.Fatalf("logits[%d] = %v, want %v", i, value, want[i])
		}
	}
}
