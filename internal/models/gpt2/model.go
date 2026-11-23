package gpt2

import (
	"encoding/json"
	"fmt"
	"github.com/codecrafters-io/inference-engine-starter-go/internal/nn"
	tensorpkg "github.com/codecrafters-io/inference-engine-starter-go/internal/tensor"
	"math"
	"os"
	"slices"
)

/*
attention
*/

// ref: https://huggingface.co/openai-community/gpt2/blob/main/config.json
type GPT2Config struct {
	ActivationFunction     string                 `json:"activation_function"`
	Architectures          []string               `json:"architectures"`
	AttentionPDrop         float32                `json:"attn_pdrop"`
	BoSTokenId             int                    `json:"bos_token_id"`
	EmbeddingPDrop         float32                `json:"embd_pdrop"`
	EoSTokenId             int                    `json:"eos_token_id"`
	InitializerRange       float32                `json:"initializer_range"`
	LayerNormEpsilon       float32                `json:"layer_norm_epsilon"`
	ModelType              string                 `json:"model_type"`
	ContextWindowSize      int                    `json:"n_ctx"`
	NumberOfEmbeddings     int                    `json:"n_embd"`
	NumberOfHeads          int                    `json:"n_head"`
	NumberOfLayers         int                    `json:"n_layer"`
	NumberOfPositions      int                    `json:"n_positions"`
	ResidualPDrop          float32                `json:"resid_pdrop"`
	SummaryActivation      interface{}            `json:"summary_activation"`
	SummaryFirstDropout    float32                `json:"summary_first_dropout"`
	SummaryProjectToLabels bool                   `json:"summary_proj_to_labels"`
	SummaryType            string                 `json:"summary_type"`
	SummaryUseProj         bool                   `json:"summary_use_proj"`
	TaskSpecificParams     map[string]interface{} `json:"task_specific_params"`
	VocabularySize         int                    `json:"vocab_size"`
}

type TokenConfig struct {
	Count int `json:"count"`
}

func LoadConfig(path string) (GPT2Config, error) {
	return readJSON[GPT2Config](path)
}

func LoadInputRows(path string) (tensorpkg.Tensor, error) {
	inputRows, err := readJSON[[][]float32](path)
	if err != nil {
		return tensorpkg.Tensor{}, err
	}

	// convert to tensor output
	// [num_pos, n_embd]
	tensorShape := []int{len(inputRows), len(inputRows[0])}
	inputTensor := tensorpkg.Tensor{
		DType:  "F32",
		Shape:  tensorShape,
		Values: []float32{},
	}

	for i := range len(inputRows) {
		inputTensor.Values = append(inputTensor.Values, inputRows[i][:]...)
	}

	return inputTensor, nil
}

func LoadTokenConfig(path string) (TokenConfig, error) {
	return readJSON[TokenConfig](path)
}

func readJSON[T any](path string) (T, error) {
	var value T
	data, err := os.ReadFile(path)
	if err != nil {
		return value, err
	}
	err = json.Unmarshal(data, &value)
	return value, err
}

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

func CalculateAttention(tmap map[string]tensorpkg.Tensor, config GPT2Config, layerNum int, inputTensor tensorpkg.Tensor) (tensorpkg.Tensor, error) {
	weights := weightsForLayer(tmap, layerNum)

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

	add, err := (nn.Linear{W: weights.attnWeight, B: &weights.attnBias}).Forward(inputTensor)
	if err != nil {
		return tensorpkg.Tensor{}, err
	}

	numPositions := inputTensor.Shape[0]
	numEmbeddings := config.NumberOfEmbeddings
	numHeads := config.NumberOfHeads
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
		score, V, err := calculateScore(add, numPositions, numEmbeddings, numHeads, h)
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

func CalculateBlock(tmap map[string]tensorpkg.Tensor, config GPT2Config, layerNum int, inputTensor tensorpkg.Tensor) (tensorpkg.Tensor, error) {
	weights := weightsForLayer(tmap, layerNum)
	layerNorm, err := (nn.LayerNorm{W: weights.ln1Weight, B: weights.ln1Bias, Eps: config.LayerNormEpsilon}).Forward(inputTensor)
	if err != nil {
		return tensorpkg.Tensor{}, err
	}

	// calculate attn on layernormed input
	attn, err := CalculateAttention(
		tmap,
		config,
		layerNum,
		layerNorm)

	if err != nil {
		return tensorpkg.Tensor{}, err
	}

	resOutput, err := inputTensor.Add(attn)
	if err != nil {
		return tensorpkg.Tensor{}, err
	}

	resOutput, err = performMLP(tmap, config, layerNum, resOutput)
	if err != nil {
		return tensorpkg.Tensor{}, err
	}

	return resOutput, nil
}

func CalculateLogits(tmap map[string]tensorpkg.Tensor, config GPT2Config, inputTensor tensorpkg.Tensor) (tensorpkg.Tensor, error) {
	attendedEmbeddings := tensorpkg.Tensor{
		DType:  inputTensor.DType,
		Shape:  inputTensor.Shape,
		Values: slices.Clone(inputTensor.Values),
	}

	for i := range config.NumberOfLayers {
		a, err := CalculateBlock(tmap, config, i, attendedEmbeddings)
		if err != nil {
			return tensorpkg.Tensor{}, err
		}

		attendedEmbeddings = a
	}

	embdLength := inputTensor.Shape[1]
	lastAttendedEmbeddingStart := len(inputTensor.Values) - embdLength
	lastAttendedEmbedding := tensorpkg.Tensor{
		DType:  attendedEmbeddings.DType,
		Shape:  []int{1, embdLength},
		Values: attendedEmbeddings.Values[lastAttendedEmbeddingStart:],
	}

	// apply last layer's layer norm
	layerNormWeights := tmap["ln_f.weight"]
	layerNormBias := tmap["ln_f.bias"]

	lastAttendedEmbedding, err := (nn.LayerNorm{W: layerNormWeights, B: layerNormBias, Eps: config.LayerNormEpsilon}).Forward(lastAttendedEmbedding)
	if err != nil {
		return tensorpkg.Tensor{}, err
	}

	/*
		for each token t:
		    logit[t] = dot(last, row t of wte.weight)

		last -> (1, n_embd)
		wte.weight -> (num_pos, n_embd)

		dot prod. logits = last x (wte.weight).Tranpose
		(1, num_pos)
	*/

	wteWeight := tmap["wte.weight"]
	wteWeight, err = wteWeight.Transpose()
	if err != nil {
		return tensorpkg.Tensor{}, err
	}

	logits, err := lastAttendedEmbedding.MatMul(wteWeight)
	if err != nil {
		return tensorpkg.Tensor{}, err
	}

	return logits, nil
}

func PerformGeneration(tmap map[string]tensorpkg.Tensor, config GPT2Config, tokenConfig TokenConfig, inputTensor tensorpkg.Tensor) ([]int, error) {

	generatedIds := []int{}
	for range tokenConfig.Count {
		logits, err := CalculateLogits(tmap, config, inputTensor)
		if err != nil {
			return []int{}, err
		}

		maxLogitIdx := slices.Index(logits.Values, slices.Max(logits.Values))
		generatedIds = append(generatedIds, maxLogitIdx)
		inputTensor, err = addTokenIdEmbedding(tmap, inputTensor, maxLogitIdx)
		if err != nil {
			return nil, err
		}
	}

	return generatedIds, nil

}

// split [Q | K | V] and returns Q K^T / sqrt(d_k) and V
// In MHA performs calculation for Qh, Kh, Vh
func calculateScore(t tensorpkg.Tensor, numPositions int, numEmbeddings int, numHeads int, head int) (tensorpkg.Tensor, tensorpkg.Tensor, error) {
	tshape := t.Shape
	if len(tshape) != 2 || numPositions < 0 || numEmbeddings <= 0 || numHeads <= 0 || head < 0 || head >= numHeads || numEmbeddings%numHeads != 0 || tshape[0] != numPositions || tshape[1] != 3*numEmbeddings || len(t.Values) < numPositions*3*numEmbeddings {
		return tensorpkg.Tensor{}, tensorpkg.Tensor{}, fmt.Errorf("Error while calculating score: invalid dims provided: %v, whereas n_pos: %d, n_embd: %d and n_heads: %d", tshape, numPositions, numEmbeddings, numHeads)
	}

	q := tensorpkg.Tensor{
		DType:  t.DType,
		Shape:  []int{numPositions, numEmbeddings},
		Values: make([]float32, numPositions*numEmbeddings),
	}
	k := tensorpkg.Tensor{
		DType:  t.DType,
		Shape:  []int{numPositions, numEmbeddings},
		Values: make([]float32, numPositions*numEmbeddings),
	}
	v := tensorpkg.Tensor{
		DType:  t.DType,
		Shape:  []int{numPositions, numEmbeddings},
		Values: make([]float32, numPositions*numEmbeddings),
	}

	stride := 3 * numEmbeddings
	for i := 0; i < numPositions; i++ {
		row := t.Values[i*stride : (i+1)*stride]
		copy(q.Values[i*numEmbeddings:(i+1)*numEmbeddings], row[:numEmbeddings])
		copy(k.Values[i*numEmbeddings:(i+1)*numEmbeddings], row[numEmbeddings:2*numEmbeddings])
		copy(v.Values[i*numEmbeddings:(i+1)*numEmbeddings], row[2*numEmbeddings:3*numEmbeddings])
	}

	// q, k, v are sliced heads representing the head indexed "head" argument
	// dimension is (n_pos, head_dim)
	/*
		attention input: [Q|K\V] -> (n_pos, 3*n_embd)
		Q, K, V -> (n_pos, n_embd)
		Now these are again sliced
		Q -> [q0|q1|...|qm] where m = head_dim
		K -> [k0|k1|...|km]
		V -> [v0|v1|...|vm]
	*/
	qHeads, err := nn.SplitHeads(q, numHeads)
	if err != nil {
		return tensorpkg.Tensor{}, tensorpkg.Tensor{}, err
	}
	kHeads, err := nn.SplitHeads(k, numHeads)
	if err != nil {
		return tensorpkg.Tensor{}, tensorpkg.Tensor{}, err
	}
	vHeads, err := nn.SplitHeads(v, numHeads)
	if err != nil {
		return tensorpkg.Tensor{}, tensorpkg.Tensor{}, err
	}
	q, k, v = qHeads[head], kHeads[head], vHeads[head]

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

func performMLP(tmap map[string]tensorpkg.Tensor, config GPT2Config, layerNum int, inputTensor tensorpkg.Tensor) (tensorpkg.Tensor, error) {
	weights := weightsForLayer(tmap, layerNum)
	layerNormOutput, err := (nn.LayerNorm{W: weights.ln2Weight, B: weights.ln2Bias, Eps: config.LayerNormEpsilon}).Forward(inputTensor)
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
