package main

import (
	"fmt"
	"math"
	"slices"
)

// tensor is one table of numbers from a safetensors file.
type tensor struct {
	DType  string
	Shape  []int
	Values []float32
}

// TODO[improvement]: current implementations work only for 2D tensors
func (t1 tensor) add(t2 tensor) (tensor, error) {
	// Shape [n] is one row. [m,n] + [n] and [m,n] + [1,n] broadcast on that row.
	bothRank1 := len(t1.Shape) == 1 && len(t2.Shape) == 1
	dtype := t1.DType
	if len(t1.Shape) == 1 {
		t1.Shape = []int{1, t1.Shape[0]}
	}
	if len(t2.Shape) == 1 {
		t2.Shape = []int{1, t2.Shape[0]}
	}

	t1Shape, t2Shape := t1.Shape, t2.Shape
	if len(t1Shape) != 2 || len(t2Shape) != 2 || t1Shape[1] != t2Shape[1] || (t1Shape[0] != t2Shape[0] && t1Shape[0] != 1 && t2Shape[0] != 1) {
		return tensor{}, fmt.Errorf("Addition error: invalid shapes %v + %v", t1Shape, t2Shape)
	}
	if t1Shape[0] == 1 && t2Shape[0] != 1 {
		t1, t2 = t2, t1
		t1Shape, t2Shape = t1.Shape, t2.Shape
	}
	if t2Shape[0] == 1 && t1Shape[0] != 1 {
		repeated := make([]float32, t1Shape[0]*t1Shape[1])
		for i := 0; i < t1Shape[0]; i++ {
			copy(repeated[i*t1Shape[1]:(i+1)*t1Shape[1]], t2.Values)
		}
		t2 = tensor{DType: t2.DType, Shape: []int{t1Shape[0], t1Shape[1]}, Values: repeated}
		t2Shape = t2.Shape
	}

	out := t1
	out.DType = dtype
	out.Shape = []int{t1Shape[0], t1Shape[1]}
	out.Values = slices.Clone(t1.Values)
	for i := 0; i < t1Shape[0]; i++ {
		for j := 0; j < t2Shape[1]; j++ {
			t1FlatIdx, err := t1.getFlatArrayIdx(i, j)
			if err != nil {
				return tensor{}, fmt.Errorf("Addition error[1]: invalid shapes %v + %v", t1Shape, t2Shape)
			}
			t2FlatIdx, err := t2.getFlatArrayIdx(i, j)
			if err != nil {
				return tensor{}, fmt.Errorf("Addition error[2]: invalid shapes %v + %v", t1Shape, t2Shape)
			}

			out.Values[t1FlatIdx] += t2.Values[t2FlatIdx]
		}
	}
	if bothRank1 {
		out.Shape = []int{t1Shape[1]}
	}

	return out, nil
}

func (t1 tensor) scalerMultiply(n float32) (tensor, error) {
	rank1 := len(t1.Shape) == 1
	if rank1 {
		t1.Shape = []int{1, t1.Shape[0]}
	}
	t1Shape := t1.Shape
	out := t1
	out.Values = slices.Clone(t1.Values)
	for i := 0; i < t1Shape[0]; i++ {
		for j := 0; j < t1Shape[1]; j++ {
			t1FlatIdx, err := t1.getFlatArrayIdx(i, j)
			if err != nil {
				return tensor{}, fmt.Errorf("Scale error[1]: invalid shapes %v", t1Shape)
			}

			out.Values[t1FlatIdx] *= n
		}
	}
	if rank1 {
		out.Shape = []int{t1Shape[1]}
	}

	return out, nil
}

func (t tensor) transpose() (tensor, error) {
	rank1 := len(t.Shape) == 1
	if rank1 {
		t.Shape = []int{1, t.Shape[0]}
	}
	if len(t.Shape) != 2 {
		return tensor{}, fmt.Errorf("Transpose error: invalid shape %v", t.Shape)
	}

	rows, cols := t.Shape[0], t.Shape[1]
	out := tensor{
		DType:  t.DType,
		Shape:  []int{cols, rows},
		Values: make([]float32, rows*cols),
	}
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			src, err := t.getFlatArrayIdx(i, j)
			if err != nil {
				return tensor{}, fmt.Errorf("Transpose error[1]: invalid shapes %v", t.Shape)
			}
			// out shape is [cols, rows], so element (j, i) is the transposed spot.
			dst, err := out.getFlatArrayIdx(j, i)
			if err != nil {
				return tensor{}, fmt.Errorf("Transpose error[2]: invalid shapes %v", out.Shape)
			}
			out.Values[dst] = t.Values[src]
		}
	}

	return out, nil
}

func (t1 tensor) exp() (tensor, error) {
	rank1 := len(t1.Shape) == 1
	if rank1 {
		t1.Shape = []int{1, t1.Shape[0]}
	}
	t1Shape := t1.Shape
	out := t1
	out.Values = slices.Clone(t1.Values)
	for i := 0; i < t1Shape[0]; i++ {
		for j := 0; j < t1Shape[1]; j++ {
			t1FlatIdx, err := t1.getFlatArrayIdx(i, j)
			if err != nil {
				return tensor{}, fmt.Errorf("Scale error[1]: invalid shapes %v", t1Shape)
			}

			out.Values[t1FlatIdx] = float32(math.Exp(float64(t1.Values[t1FlatIdx])))
		}
	}
	if rank1 {
		out.Shape = []int{t1Shape[1]}
	}

	return out, nil
}

func (t1 *tensor) multiply(t2 tensor) (tensor, error) {
	t1Shape, t2Shape := t1.Shape, t2.Shape
	// mxn times nxp -> mxp
	if t1Shape[1] != t2Shape[0] {
		return tensor{}, fmt.Errorf("Multiplication error: invalid shapes %v x %v", t1Shape, t2Shape)
	}

	resRows, resCols := t1Shape[0], t2Shape[1]
	res := tensor{
		DType:  t1.DType, // assuming both have same DType,
		Shape:  []int{resRows, resCols},
		Values: make([]float32, resRows*resCols),
	}

	/*
		res[i][j] = sum over k t1[i][k].t2[k][j]
	*/
	commonDim := t1Shape[1]
	for i := 0; i < resRows; i++ {
		for j := 0; j < resCols; j++ {
			flatIdx, err := res.getFlatArrayIdx(i, j)
			if err != nil {
				return tensor{}, fmt.Errorf("Multiplication error[1]: invalid shapes %v x %v", t1Shape, t2Shape)
			}

			for k := 0; k < commonDim; k++ {
				t1FlatIdx, err := t1.getFlatArrayIdx(i, k)
				if err != nil {
					return tensor{}, fmt.Errorf("Multiplication error[2]: invalid shapes %v x %v", t1Shape, t2Shape)
				}
				t2FlatIdx, err := t2.getFlatArrayIdx(k, j)
				if err != nil {
					return tensor{}, fmt.Errorf("Multiplication error[3]: invalid shapes %v x %v", t1Shape, t2Shape)
				}

				res.Values[flatIdx] += t1.Values[t1FlatIdx] * t2.Values[t2FlatIdx]
			}
		}
	}

	return res, nil
}

func (t tensor) softmax(dim int, temp float32) (tensor, error) {
	if dim != 1 {
		return tensor{}, fmt.Errorf("Softmax error: Invalid dimension %d passed for shape %v", dim, t.Shape)
	}
	if temp <= 0 {
		return tensor{}, fmt.Errorf("Softmax error: Invalid temperature value %v passed", temp)
	}

	// current challenge requires normalization along rows (dim=1).
	// TODO[improvement]: handle generic case

	rows, cols := t.Shape[0], t.Shape[1]
	out := t
	out.Values = slices.Clone(t.Values)
	if dim == 1 {
		for i := 0; i < rows; i++ {
			rowStartFlatIdx, err := t.getFlatArrayIdx(i, 0)
			if err != nil {
				return tensor{}, fmt.Errorf("Softmax error[1]: Invalid dimension %d passed for shape %v", dim, t.Shape)
			}
			rowEndFlatIdx, err := t.getFlatArrayIdx(i, cols-1)
			if err != nil {
				return tensor{}, fmt.Errorf("Softmax error[1]: Invalid dimension %d passed for shape %v", dim, t.Shape)
			}

			/*
				For numerical stability do exp(z-zmax)/(sum(exp(z-zmax)))
			*/
			rowMax := slices.Max(out.Values[rowStartFlatIdx : rowEndFlatIdx+1])
			rowSum := float32(0)
			for j := rowStartFlatIdx; j < rowEndFlatIdx+1; j++ {
				exponent := float64((out.Values[j] - rowMax) / temp)
				out.Values[j] = float32(math.Exp(exponent))
				rowSum += out.Values[j]
			}

			// normalize
			for j := rowStartFlatIdx; j < rowEndFlatIdx+1; j++ {
				out.Values[j] /= rowSum
			}
		}
	}

	return out, nil
}

func (t *tensor) getFlatArrayIdx(i int, j int) (int, error) {
	/*
		2, 3
		0(0, 0) 1(0, 1) 2(0, 2)
		3(1, 0) 4(1, 1) 5(1, 2)
		6(2, 0) 7(2, 1) 8(2, 2)

	*/
	var rows, cols int
	if len(t.Shape) == 1 {
		rows, cols = 1, t.Shape[0]
	} else if len(t.Shape) == 2 {
		rows, cols = t.Shape[0], t.Shape[1]
	} else {
		return -1, fmt.Errorf("Invalid indices: %d %d for shape %v", i, j, t.Shape)
	}
	if i < 0 || j < 0 || i >= rows || j >= cols {
		return -1, fmt.Errorf("Invalid indices: %d %d for shape %v", i, j, t.Shape)
	}
	idx := i*cols + j
	if idx >= len(t.Values) {
		return -1, fmt.Errorf("Invalid indices: %d %d for shape %v", i, j, t.Shape)
	}

	return idx, nil
}
