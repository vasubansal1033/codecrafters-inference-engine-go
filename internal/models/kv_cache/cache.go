package kvcache

import (
	"fmt"

	"github.com/codecrafters-io/inference-engine-starter-go/internal/tensor"
)

type KVCache struct {
	// list of tensors indexed by layer_num, head
	KCache map[int]map[int][]tensor.Tensor
	VCache map[int]map[int][]tensor.Tensor
}

func NewKVCache() KVCache {
	return KVCache{
		KCache: make(map[int]map[int][]tensor.Tensor),
		VCache: make(map[int]map[int][]tensor.Tensor),
	}
}

func (kv *KVCache) Get(layer_num int, head int) (tensor.Tensor, tensor.Tensor, error) {
	// A missing slot means Prefill stage.
	// First touch of a layer and a missing slot both land here.
	kLayerTensorsList, _ := kv.getTensorList(layer_num, head, "K", false)
	vLayerTensorsList, _ := kv.getTensorList(layer_num, head, "V", false)

	return stackRows(kLayerTensorsList),
		stackRows(vLayerTensorsList),
		nil
}

func stackRows(list []tensor.Tensor) tensor.Tensor {
	// handling stacking of 1D and 2D tensors
	rows := 0
	width := 0
	values := []float32{}
	for _, t := range list {
		switch len(t.Shape) {
		case 1:
			rows++
			width = t.Shape[0]
		case 2:
			rows += t.Shape[0]
			width = t.Shape[1]
		}
		values = append(values, t.Values...)
	}

	return tensor.Tensor{
		DType:  "F32",
		Shape:  []int{rows, width},
		Values: values,
	}
}

func (kv *KVCache) Put(layer_num int, head int, k tensor.Tensor, v tensor.Tensor) error {
	kLayerTensorsList, _ := kv.getTensorList(layer_num, head, "K", true)
	vLayerTensorsList, _ := kv.getTensorList(layer_num, head, "V", true)

	kv.KCache[layer_num][head] = append(kLayerTensorsList, k)
	kv.VCache[layer_num][head] = append(vLayerTensorsList, v)

	return nil
}

func (kv *KVCache) getTensorList(layer_num int, head int, queryType string, skip_error bool) ([]tensor.Tensor, error) {
	cache := kv.KCache
	if queryType == "V" {
		cache = kv.VCache
	}

	if innerMap, ok := cache[layer_num]; ok {
		if t, ok := innerMap[head]; ok {
			return t, nil
		}

		if skip_error {
			innerMap[head] = make([]tensor.Tensor, 0)
			return []tensor.Tensor{}, nil
		}
	}

	if skip_error {
		cache[layer_num] = make(map[int][]tensor.Tensor)
		cache[layer_num][head] = make([]tensor.Tensor, 0)

		return cache[layer_num][head], nil
	}

	return nil, fmt.Errorf("KVCache.Get error: No entry found for layer_num %d, head: %d", layer_num, head)
}
