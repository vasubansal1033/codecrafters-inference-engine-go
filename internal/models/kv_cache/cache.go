package kvcache

import (
	"fmt"

	"github.com/codecrafters-io/inference-engine-starter-go/internal/tensor"
)

type KVCache struct {
	// map of layer_num to a list of tensors
	Cache map[int][]tensor.Tensor
}

func (kv *KVCache) Get(layer_num int) (tensor.Tensor, error) {
	return tensor.Tensor{}, fmt.Errorf("KVCache.Get not implemented yet")
}

func (kv *KVCache) Put(layer_num int) (tensor.Tensor, error) {
	return tensor.Tensor{}, fmt.Errorf("KVCache.Put not implemented yet")
}
