package kvcache

import (
	"fmt"

	"github.com/codecrafters-io/inference-engine-starter-go/internal/tensor"
)

type KVCache struct {
	// list of tensors indexed by layer_num, head
	Cache map[int]map[int][]tensor.Tensor
}

func (kv *KVCache) Get(layer_num int, head int) (tensor.Tensor, error) {
	return tensor.Tensor{}, fmt.Errorf("KVCache.Get not implemented yet")
}

func (kv *KVCache) Put(layer_num int, head int, t tensor.Tensor) (tensor.Tensor, error) {
	return tensor.Tensor{}, fmt.Errorf("KVCache.Put not implemented yet")
}
