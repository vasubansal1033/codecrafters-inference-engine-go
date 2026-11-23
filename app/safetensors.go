package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
)

type dtypeDecoder struct {
	width  int
	decode func([]byte) float32
}

var dtypeDecoders = map[string]dtypeDecoder{
	"F32": {width: 4, decode: func(raw []byte) float32 {
		return math.Float32frombits(binary.LittleEndian.Uint32(raw))
	}},
	"I32": {width: 4, decode: func(raw []byte) float32 {
		return float32(int32(binary.LittleEndian.Uint32(raw)))
	}},
}

// loadTensors reads every tensor in a safetensors file, with all of its numbers.
func loadTensors(path string) (map[string]tensor, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var headerLen uint64
	if err := binary.Read(f, binary.LittleEndian, &headerLen); err != nil {
		return nil, err
	}
	headerBytes := make([]byte, headerLen)
	if _, err := io.ReadFull(f, headerBytes); err != nil {
		return nil, err
	}
	var header map[string]struct {
		DType       string `json:"dtype"`
		Shape       []int  `json:"shape"`
		DataOffsets [2]int `json:"data_offsets"`
	}
	if err := json.Unmarshal(headerBytes, &header); err != nil {
		return nil, err
	}
	data, err := io.ReadAll(f)
	if err != nil {
		return nil, err
	}

	tensors := make(map[string]tensor)
	for name, info := range header {
		if name == "__metadata__" {
			continue
		}
		decoder, ok := dtypeDecoders[info.DType]
		if !ok {
			return nil, fmt.Errorf("unsupported dtype %s", info.DType)
		}
		raw := data[info.DataOffsets[0]:info.DataOffsets[1]]
		values := make([]float32, len(raw)/decoder.width)
		for i := range values {
			values[i] = decoder.decode(raw[i*decoder.width:])
		}
		tensors[name] = tensor{DType: info.DType, Shape: info.Shape, Values: values}
	}
	return tensors, nil
}
