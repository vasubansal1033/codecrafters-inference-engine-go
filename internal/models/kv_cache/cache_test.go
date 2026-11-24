package kvcache

import (
	"testing"

	"github.com/codecrafters-io/inference-engine-starter-go/internal/tensor"
)

func TestGetMissingSlotIsEmpty(t *testing.T) {
	cache := NewKVCache()
	k, v, err := cache.Get(0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if k.Shape[0] != 0 || k.Shape[1] != 0 || v.Shape[0] != 0 || v.Shape[1] != 0 {
		t.Fatalf("empty shapes = %v %v", k.Shape, v.Shape)
	}
}

func TestPutKeepsHeadsAndRowShape(t *testing.T) {
	cache := NewKVCache()
	promptK := tensorOf([]int{5, 4}, 1)
	promptV := tensorOf([]int{5, 4}, 2)
	if err := cache.Put(0, 0, promptK, promptV); err != nil {
		t.Fatal(err)
	}
	if err := cache.Put(0, 1, tensorOf([]int{5, 4}, 3), tensorOf([]int{5, 4}, 4)); err != nil {
		t.Fatal(err)
	}
	if err := cache.Put(0, 0, tensorOf([]int{1, 4}, 5), tensorOf([]int{1, 4}, 6)); err != nil {
		t.Fatal(err)
	}

	k, v, err := cache.Get(0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if k.Shape[0] != 6 || k.Shape[1] != 4 || v.Shape[0] != 6 || v.Shape[1] != 4 {
		t.Fatalf("shapes = %v %v", k.Shape, v.Shape)
	}
	if k.Values[0] != 1 || k.Values[len(k.Values)-1] != 5 || v.Values[0] != 2 || v.Values[len(v.Values)-1] != 6 {
		t.Fatalf("k = %v, v = %v", k.Values, v.Values)
	}

	other, _, err := cache.Get(0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if other.Shape[0] != 5 || other.Values[0] != 3 {
		t.Fatalf("head 1 was wiped: %v %v", other.Shape, other.Values[:1])
	}
}

func tensorOf(shape []int, fill float32) tensor.Tensor {
	count := 1
	for _, dim := range shape {
		count *= dim
	}
	values := make([]float32, count)
	for i := range values {
		values[i] = fill
	}
	return tensor.Tensor{DType: "F32", Shape: shape, Values: values}
}
