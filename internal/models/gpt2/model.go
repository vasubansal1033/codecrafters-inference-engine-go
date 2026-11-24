package gpt2

import (
	"fmt"
	"math"

	kvcache "github.com/codecrafters-io/inference-engine-starter-go/internal/models/kv_cache"
	"github.com/codecrafters-io/inference-engine-starter-go/internal/nn"
	tensorpkg "github.com/codecrafters-io/inference-engine-starter-go/internal/tensor"
)

// One transformer layer. gpt2.go decides whether a call receives the KV cache.

type layerWeights struct {
	attnWeight, attnBias tensorpkg.Tensor
	attnProjWeight       tensorpkg.Tensor
	attnProjBias         tensorpkg.Tensor
	ln1Weight, ln1Bias   tensorpkg.Tensor
	ln2Weight, ln2Bias   tensorpkg.Tensor
	mlpWeight, mlpBias   tensorpkg.Tensor
	mlpProjWeight        tensorpkg.Tensor
	mlpProjBias          tensorpkg.Tensor
}

func weightsForLayer(tmap map[string]tensorpkg.Tensor, layer int) layerWeights {
	prefix := fmt.Sprintf("h.%d.", layer)
	return layerWeights{
		attnWeight:     tmap[prefix+"attn.c_attn.weight"],
		attnBias:       tmap[prefix+"attn.c_attn.bias"],
		attnProjWeight: tmap[prefix+"attn.c_proj.weight"],
		attnProjBias:   tmap[prefix+"attn.c_proj.bias"],
		ln1Weight:      tmap[prefix+"ln_1.weight"],
		ln1Bias:        tmap[prefix+"ln_1.bias"],
		ln2Weight:      tmap[prefix+"ln_2.weight"],
		ln2Bias:        tmap[prefix+"ln_2.bias"],
		mlpWeight:      tmap[prefix+"mlp.c_fc.weight"],
		mlpBias:        tmap[prefix+"mlp.c_fc.bias"],
		mlpProjWeight:  tmap[prefix+"mlp.c_proj.weight"],
		mlpProjBias:    tmap[prefix+"mlp.c_proj.bias"],
	}
}

func (m *Model) calculateAttention(layerNum int, inputTensor tensorpkg.Tensor, cache *kvcache.KVCache) (tensorpkg.Tensor, error) {
	weights := weightsForLayer(m.weights, layerNum)

	/*
		TENSOR CALCULATION:
		input_rows * c_attn.weight + c_attn.bias
			(num_pos, n_embd)*(n_embd, 3 * n_embd) + [3 * n_embd]
			bias will be broadcasted horizontally
		output -> (num_pos, 3 * n_embd) -> (num_pos, [n_embd | n_embd | n_embd])
														Q        K         V
		Each row corresponding to a position is divided into [Q, K, V]
		Q, K, V -> each has (n_pos, n_embd)
		attn score: QK' / sqrt(d_k) -> (n_pos, n_pos)

		mask(score) -> upper triangular replaced with -inf
		softmax(mask(score)) * V
		(num_pos, num_pos) * (num_pos, n_embd) -> (num_pos, n_embd)

		result = causally attended embeddings (dim is same as input embeddings)
			   = (num_pod, n_embd)

		result = result * c_proj.weight + c_proj.bias
			(num_pos, n_embd) * (n_embd, n_embd) + [n_embd]
			= (num_pos, n_embd)
	*/

	layerProjection, err := (nn.Linear{
		W: weights.attnWeight,
		B: &weights.attnBias,
	}).Forward(inputTensor)

	if err != nil {
		return tensorpkg.Tensor{}, err
	}

	numPositions := inputTensor.Shape[0]
	numEmbeddings := m.config.NumberOfEmbeddings
	numHeads := m.config.NumberOfHeads
	if numHeads <= 0 || numEmbeddings%numHeads != 0 {
		return tensorpkg.Tensor{}, fmt.Errorf("Error while calculating score: invalid dims n_embd: %d and n_heads: %d", numEmbeddings, numHeads)
	}

	// MHA calculation
	/*
		Multi Head Attention
		Q, K, V -> (n_pos, n_embd)
		Each divided into n_heads
		For e.g., Q ->
			Qi -> (n_pos, head_dim) where head_dim = n_embd/n_heads
			Q = [Q1 | Q2 .... | Qn_heads]
	*/
	headOutputs := make([]tensorpkg.Tensor, 0, numHeads)

	// attention for hth head
	for h := 0; h < numHeads; h++ {
		score, V, err := calculateScore(layerProjection, cache, layerNum, numPositions, numEmbeddings, numHeads, h)
		if err != nil {
			return tensorpkg.Tensor{}, err
		}
		score, err = nn.CausalMask(score)
		if err != nil {
			return tensorpkg.Tensor{}, err
		}

		softmaxScore, err := score.Softmax(1, 1)
		if err != nil {
			return tensorpkg.Tensor{}, err
		}

		// causally attended embeddings
		headOut, err := softmaxScore.MatMul(V)
		if err != nil {
			return tensorpkg.Tensor{}, err
		}

		headOutputs = append(headOutputs, headOut)
	}

	attended, err := nn.MergeHeads(headOutputs)
	if err != nil {
		return tensorpkg.Tensor{}, err
	}
	res, err := (nn.Linear{W: weights.attnProjWeight, B: &weights.attnProjBias}).Forward(attended)
	if err != nil {
		return tensorpkg.Tensor{}, err
	}

	return res, nil
}

func (m *Model) calculateBlock(layerNum int, inputTensor tensorpkg.Tensor, cache *kvcache.KVCache) (tensorpkg.Tensor, error) {
	weights := weightsForLayer(m.weights, layerNum)
	layerNorm, err := (nn.LayerNorm{W: weights.ln1Weight, B: weights.ln1Bias, Eps: m.config.LayerNormEpsilon}).Forward(inputTensor)
	if err != nil {
		return tensorpkg.Tensor{}, err
	}

	// calculate attn on layernormed input
	attn, err := m.calculateAttention(layerNum, layerNorm, cache)

	if err != nil {
		return tensorpkg.Tensor{}, err
	}

	resOutput, err := inputTensor.Add(attn)
	if err != nil {
		return tensorpkg.Tensor{}, err
	}

	resOutput, err = m.performMLP(layerNum, resOutput)
	if err != nil {
		return tensorpkg.Tensor{}, err
	}

	return resOutput, nil
}

// split [Q | K | V] and returns Q K^T / sqrt(d_k) and V
// In MHA performs calculation for Qh, Kh, Vh
func calculateScore(layerProjection tensorpkg.Tensor, cache *kvcache.KVCache, layerNum int, numPositions int, numEmbeddings int, numHeads int, head int) (tensorpkg.Tensor, tensorpkg.Tensor, error) {
	tshape := layerProjection.Shape
	if len(tshape) != 2 || numPositions < 0 || numEmbeddings <= 0 || numHeads <= 0 || head < 0 || head >= numHeads || numEmbeddings%numHeads != 0 || tshape[0] != numPositions || tshape[1] != 3*numEmbeddings || len(layerProjection.Values) < numPositions*3*numEmbeddings {
		return tensorpkg.Tensor{}, tensorpkg.Tensor{}, fmt.Errorf("Error while calculating score: invalid dims provided: %v, whereas n_pos: %d, n_embd: %d and n_heads: %d", tshape, numPositions, numEmbeddings, numHeads)
	}

	// Fused row is [Q | K | V], and each block is [head 0 | head 1 | ...].
	// Copy only this head's columns. Shape is [n_pos, head_dim].
	/*
		KV Cache implementation

		Prefill phase:
			While generating the first token, the token cache will be empty. So we need to "prefill" it with context
			K and V.
		Decode phase:
			Using KVCache and layer_num as key, retrieve K and V of all previous tokens inform of tensor
			K, V -> (num_positions-1, num_embeddings)

		And from current layer's projection tensor, we can retrieve K and V for current token.
		We can reconstruct full K tensor -> (num_positions, num_embeddings)
		And using current token's q (1, num_embeddings) get  -> (1, num_embeddings) the attention
		for current token.
		Otherwise, in basic implementation we are anyways throwing away the attention for the previous
		i-1 tokens.

	*/

	headDim := numEmbeddings / numHeads
	var kcached, vcached tensorpkg.Tensor

	var err error
	if cache == nil {
		// treating no cache case as prefill phase
		kcached = tensorpkg.Tensor{
			DType:  "F32",
			Shape:  []int{0, 0},
			Values: []float32{},
		}
		vcached = tensorpkg.Tensor{
			DType:  "F32",
			Shape:  []int{0, 0},
			Values: []float32{},
		}
	} else {
		kcached, vcached, err = cache.Get(layerNum, head)
		if err != nil {
			return tensorpkg.Tensor{}, tensorpkg.Tensor{}, err
		}
	}

	var q, k, v tensorpkg.Tensor
	if kcached.IsEmpty() && vcached.IsEmpty() {
		// prefill phase
		q = tensorpkg.Tensor{
			DType:  layerProjection.DType,
			Shape:  []int{numPositions, headDim},
			Values: make([]float32, numPositions*headDim),
		}
		k = tensorpkg.Tensor{
			DType:  layerProjection.DType,
			Shape:  []int{numPositions, headDim},
			Values: make([]float32, numPositions*headDim),
		}
		v = tensorpkg.Tensor{
			DType:  layerProjection.DType,
			Shape:  []int{numPositions, headDim},
			Values: make([]float32, numPositions*headDim),
		}

		stride := 3 * numEmbeddings
		qBase := head * headDim
		kBase := numEmbeddings + qBase
		vBase := 2*numEmbeddings + qBase
		for i := 0; i < numPositions; i++ {
			row := layerProjection.Values[i*stride : (i+1)*stride]
			copy(q.Values[i*headDim:(i+1)*headDim], row[qBase:qBase+headDim])
			copy(k.Values[i*headDim:(i+1)*headDim], row[kBase:kBase+headDim])
			copy(v.Values[i*headDim:(i+1)*headDim], row[vBase:vBase+headDim])
		}

		// prefill cache store
		if cache != nil {
			err := cache.Put(layerNum, head, k, v)
			if err != nil {
				return tensorpkg.Tensor{}, tensorpkg.Tensor{}, err
			}
		}
	} else {
		// decode phase
	}

	kT, err := k.Transpose()
	if err != nil {
		return tensorpkg.Tensor{}, tensorpkg.Tensor{}, err
	}
	qk, err := q.MatMul(kT)
	if err != nil {
		return tensorpkg.Tensor{}, tensorpkg.Tensor{}, err
	}
	// Dot is over key_dim = n_embd/n_heads.
	dK := float32(numEmbeddings / numHeads)
	res, err := qk.Scale(1 / float32(math.Sqrt(float64(dK))))
	return res, v, err
}

func (m *Model) performMLP(layerNum int, inputTensor tensorpkg.Tensor) (tensorpkg.Tensor, error) {
	weights := weightsForLayer(m.weights, layerNum)
	layerNormOutput, err := (nn.LayerNorm{W: weights.ln2Weight, B: weights.ln2Bias, Eps: m.config.LayerNormEpsilon}).Forward(inputTensor)
	if err != nil {
		return tensorpkg.Tensor{}, err
	}

	// apply MLP weights
	mlpOutput, err := (nn.Linear{W: weights.mlpWeight, B: &weights.mlpBias}).Forward(layerNormOutput)
	if err != nil {
		return tensorpkg.Tensor{}, err
	}

	// apply gelu approx. which is element wise operation
	mlpOutput = nn.GELU(mlpOutput)

	// apply MLP projection
	mlpOutput, err = (nn.Linear{W: weights.mlpProjWeight, B: &weights.mlpProjBias}).Forward(mlpOutput)
	if err != nil {
		return tensorpkg.Tensor{}, err
	}

	mlpOutput, err = inputTensor.Add(mlpOutput)
	if err != nil {
		return tensorpkg.Tensor{}, err
	}

	return mlpOutput, nil
}
