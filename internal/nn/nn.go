// Package nn provides reusable neural-network layers.
package nn

import (
	"fmt"
	"math"
	"slices"

	"github.com/codecrafters-io/inference-engine-starter-go/internal/tensor"
)

type Linear struct {
	W tensor.Tensor
	B *tensor.Tensor
}

func (l Linear) Forward(input tensor.Tensor) (tensor.Tensor, error) {
	output, err := input.MatMul(l.W)
	if err != nil {
		return tensor.Tensor{}, err
	}
	if l.B == nil {
		return output, nil
	}
	return output.Add(*l.B)
}

type LayerNorm struct {
	W   tensor.Tensor
	B   tensor.Tensor
	Eps float32
}

func (l LayerNorm) Forward(input tensor.Tensor) (tensor.Tensor, error) {
	output, err := input.LayerNorm(1, l.Eps)
	if err != nil {
		return tensor.Tensor{}, err
	}
	output, err = output.BroadcastMultiply(l.W)
	if err != nil {
		return tensor.Tensor{}, err
	}
	return output.Add(l.B)
}

func GELU(input tensor.Tensor) tensor.Tensor {
	output := input
	output.Values = slices.Clone(input.Values)
	const sqrtTwoOverPi = 0.7978845608028654
	for i, x := range output.Values {
		value := float64(x)
		output.Values[i] = 0.5 * x * (1 + float32(math.Tanh(sqrtTwoOverPi*(value+0.044715*math.Pow(value, 3)))))
	}
	return output
}

func CausalMask(input tensor.Tensor, pastContextLen int) (tensor.Tensor, error) {
	output := input
	output.Values = slices.Clone(input.Values)
	for i := 0; i < output.Shape[0]; i++ {
		for j := pastContextLen + i + 1; j < output.Shape[1]; j++ {
			index, err := output.FlatIndex(i, j)
			if err != nil {
				return tensor.Tensor{}, err
			}
			output.Values[index] = float32(math.Inf(-1))
		}
	}
	return output, nil
}

func SplitHeads(input tensor.Tensor, numHeads int) ([]tensor.Tensor, error) {
	if len(input.Shape) != 2 || numHeads <= 0 || input.Shape[1]%numHeads != 0 {
		return nil, fmt.Errorf("cannot split shape %v into %d heads", input.Shape, numHeads)
	}
	rows, width := input.Shape[0], input.Shape[1]
	headWidth := width / numHeads
	heads := make([]tensor.Tensor, numHeads)
	for head := range numHeads {
		heads[head] = tensor.Tensor{
			DType:  input.DType,
			Shape:  []int{rows, headWidth},
			Values: make([]float32, rows*headWidth),
		}
		for row := range rows {
			start := row*width + head*headWidth
			copy(heads[head].Values[row*headWidth:(row+1)*headWidth], input.Values[start:start+headWidth])
		}
	}
	return heads, nil
}

func MergeHeads(heads []tensor.Tensor) (tensor.Tensor, error) {
	if len(heads) == 0 || len(heads[0].Shape) != 2 {
		return tensor.Tensor{}, fmt.Errorf("cannot merge empty or invalid heads")
	}
	rows, headWidth := heads[0].Shape[0], heads[0].Shape[1]
	output := tensor.Tensor{
		DType:  heads[0].DType,
		Shape:  []int{rows, headWidth * len(heads)},
		Values: make([]float32, rows*headWidth*len(heads)),
	}
	for head, value := range heads {
		if len(value.Shape) != 2 || value.Shape[0] != rows || value.Shape[1] != headWidth {
			return tensor.Tensor{}, fmt.Errorf("head %d has incompatible shape %v", head, value.Shape)
		}
		for row := range rows {
			dst := row*output.Shape[1] + head*headWidth
			copy(output.Values[dst:dst+headWidth], value.Values[row*headWidth:(row+1)*headWidth])
		}
	}
	return output, nil
}
