package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
)

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
		raw := data[info.DataOffsets[0]:info.DataOffsets[1]]
		values := make([]float32, len(raw)/4)
		for i := range values {
			bits := binary.LittleEndian.Uint32(raw[i*4:])
			switch info.DType {
			case "F32":
				values[i] = math.Float32frombits(bits)
			case "I32":
				values[i] = float32(int32(bits))
			default:
				return nil, fmt.Errorf("unsupported dtype %s", info.DType)
			}
		}
		tensors[name] = tensor{DType: info.DType, Shape: info.Shape, Values: values}
	}
	return tensors, nil
}
