package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	tensorpkg "github.com/codecrafters-io/inference-engine-starter-go/internal/tensor"
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

// Load reads every tensor in a safetensors file, with all of its numbers.
func Load(path string) (map[string]tensorpkg.Tensor, error) {
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
	dataStart, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, err
	}

	tensors := make(map[string]tensorpkg.Tensor)
	var raw []byte
	for name, info := range header {
		if name == "__metadata__" {
			continue
		}
		decoder, ok := dtypeDecoders[info.DType]
		if !ok {
			return nil, fmt.Errorf("unsupported dtype %s", info.DType)
		}
		nbytes := info.DataOffsets[1] - info.DataOffsets[0]
		if nbytes < 0 || nbytes%decoder.width != 0 {
			return nil, fmt.Errorf("invalid data range for %s", name)
		}
		if cap(raw) < nbytes {
			raw = make([]byte, nbytes)
		}
		raw = raw[:nbytes]
		if _, err := f.ReadAt(raw, dataStart+int64(info.DataOffsets[0])); err != nil {
			return nil, err
		}
		values := make([]float32, nbytes/decoder.width)
		for i := range values {
			values[i] = decoder.decode(raw[i*decoder.width : (i+1)*decoder.width])
		}
		tensors[name] = tensorpkg.Tensor{DType: info.DType, Shape: info.Shape, Values: values}
	}
	return tensors, nil
}
