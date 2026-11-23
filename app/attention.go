package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"slices"
)

const (
	sqrtTwoOverPi float32 = 0.7978845608028654 // sqrt(2 / pi)
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

func loadConfig(path string) (GPT2Config, error) {
	f, err := os.Open(path)
	if err != nil {
		return GPT2Config{}, err
	}
	defer f.Close()

	reader := bufio.NewReader(f)
	decoder := json.NewDecoder(reader)

	config := GPT2Config{}
	if err := decoder.Decode(&config); err != nil {
		return GPT2Config{}, err
	}

	return config, nil
}

func loadInputRows(path string) (tensor, error) {
	f, err := os.Open(path)
	if err != nil {
		return tensor{}, err
	}
	defer f.Close()

	reader := bufio.NewReader(f)
	decoder := json.NewDecoder(reader)

	var inputRows [][]float32
	if err := decoder.Decode(&inputRows); err != nil {
		return tensor{}, err
	}

	// convert to tensor output
	// [num_pos, n_embd]
	tensorShape := []int{len(inputRows), len(inputRows[0])}
	inputTensor := tensor{
		DType:  "F32",
		Shape:  tensorShape,
		Values: []float32{},
	}

	for i := range len(inputRows) {
		inputTensor.Values = append(inputTensor.Values, inputRows[i][:]...)
	}

	return inputTensor, nil
}

func loadTokenConfig(path string) (TokenConfig, error) {
	f, err := os.Open(path)
	if err != nil {
		return TokenConfig{}, err
	}
	defer f.Close()

	reader := bufio.NewReader(f)
	decoder := json.NewDecoder(reader)

	var tokenConfig TokenConfig
	if err := decoder.Decode(&tokenConfig); err != nil {
		return TokenConfig{}, err
	}

	return tokenConfig, nil
}

func calculateAttention(tmap map[string]tensor, config GPT2Config, layerNum int, inputTensor tensor) (tensor, error) {
	// fmt.Printf("config: %v\nlayerNum: %d\ninputs: %v", config, layerNum, inputs)

	layerAttnWeights := tmap[fmt.Sprintf("h.%d.attn.c_attn.weight", layerNum)]
	layerAttnBias := tmap[fmt.Sprintf("h.%d.attn.c_attn.bias", layerNum)]
	layerProjWeights := tmap[fmt.Sprintf("h.%d.attn.c_proj.weight", layerNum)]
	layerProjBias := tmap[fmt.Sprintf("h.%d.attn.c_proj.bias", layerNum)]

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

	mul, err := inputTensor.matmul(layerAttnWeights)
	if err != nil {
		return tensor{}, err
	}

	add, err := mul.add(layerAttnBias)
	if err != nil {
		return tensor{}, err
	}

	numPositions := inputTensor.Shape[0]
	numEmbeddings := config.NumberOfEmbeddings
	numHeads := config.NumberOfHeads
	if numHeads <= 0 || numEmbeddings%numHeads != 0 {
		return tensor{}, fmt.Errorf("Error while calculating score: invalid dims n_embd: %d and n_heads: %d", numEmbeddings, numHeads)
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
	headDim := numEmbeddings / numHeads
	attended := tensor{
		DType:  add.DType,
		Shape:  []int{numPositions, numEmbeddings},
		Values: make([]float32, numPositions*numEmbeddings),
	}

	// attention for hth head
	for h := 0; h < numHeads; h++ {
		score, V, err := calculateScore(add, numPositions, numEmbeddings, numHeads, h)
		if err != nil {
			return tensor{}, err
		}
		err = applyCausalMask(score)
		if err != nil {
			return tensor{}, err
		}

		softmaxScore, err := score.softmax(1, 1)
		if err != nil {
			return tensor{}, err
		}

		// causally attended embeddings
		headOut, err := softmaxScore.matmul(V)
		if err != nil {
			return tensor{}, err
		}

		// placing this sub-head's attention (n_pos, n_embd/n_head)
		// in over head's attention (n_pos, n_embd)
		for i := 0; i < numPositions; i++ {
			dst := i*numEmbeddings + h*headDim
			copy(attended.Values[dst:dst+headDim], headOut.Values[i*headDim:(i+1)*headDim])
		}
	}

	mul, err = attended.matmul(layerProjWeights)
	if err != nil {
		return tensor{}, err
	}

	res, err := mul.add(layerProjBias)
	if err != nil {
		return tensor{}, err
	}

	return res, nil
}

func calculateBlock(tmap map[string]tensor, config GPT2Config, layerNum int, inputTensor tensor) (tensor, error) {
	layerNorm, err := inputTensor.layerNorm(1)
	if err != nil {
		return tensor{}, err
	}

	// apply layer norm-1 wandb
	layerNormWeights := tmap[fmt.Sprintf("h.%d.ln_1.weight", layerNum)]
	layerNormBias := tmap[fmt.Sprintf("h.%d.ln_1.bias", layerNum)]

	layerNorm, err = layerNorm.broadcastMultiply(layerNormWeights)
	if err != nil {
		return tensor{}, err
	}
	layerNorm, err = layerNorm.add(layerNormBias)
	if err != nil {
		return tensor{}, err
	}

	// calculate attn on layernormed input
	attn, err := calculateAttention(
		tmap,
		config,
		layerNum,
		layerNorm,
	)
	if err != nil {
		return tensor{}, err
	}

	resOutput, err := inputTensor.add(attn)
	if err != nil {
		return tensor{}, err
	}

	resOutput, err = performMLP(tmap, config, layerNum, resOutput)
	if err != nil {
		return tensor{}, err
	}

	return resOutput, nil
}

func calculateLogits(tmap map[string]tensor, config GPT2Config, inputTensor tensor) (tensor, error) {
	attendedEmbeddings := tensor{
		DType:  inputTensor.DType,
		Shape:  inputTensor.Shape,
		Values: slices.Clone(inputTensor.Values),
	}

	for i := range config.NumberOfLayers {
		a, err := calculateBlock(tmap, config, i, attendedEmbeddings)
		if err != nil {
			return tensor{}, err
		}

		attendedEmbeddings = a
	}

	embdLength := inputTensor.Shape[1]
	lastAttendedEmbeddingStart := len(inputTensor.Values) - embdLength
	lastAttendedEmbedding := tensor{
		DType:  attendedEmbeddings.DType,
		Shape:  []int{1, embdLength},
		Values: attendedEmbeddings.Values[lastAttendedEmbeddingStart:],
	}

	// apply last layer's layer norm
	lastAttendedEmbedding, err := lastAttendedEmbedding.layerNorm(1)
	if err != nil {
		return tensor{}, err
	}

	layerNormWeights := tmap["ln_f.weight"]
	layerNormBias := tmap["ln_f.bias"]

	lastAttendedEmbedding, err = lastAttendedEmbedding.broadcastMultiply(layerNormWeights)
	if err != nil {
		return tensor{}, err
	}

	lastAttendedEmbedding, err = lastAttendedEmbedding.add(layerNormBias)
	if err != nil {
		return tensor{}, err
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
	wteWeight, err = wteWeight.transpose()
	if err != nil {
		return tensor{}, err
	}

	logits, err := lastAttendedEmbedding.matmul(wteWeight)
	if err != nil {
		return tensor{}, err
	}

	return logits, nil
}

func performGeneration(tmap map[string]tensor, config GPT2Config, tokenConfig TokenConfig, inputTensor tensor) ([]int, error) {

	generatedIds := []int{}
	for range tokenConfig.Count {
		logits, err := calculateLogits(tmap, config, inputTensor)
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
func calculateScore(t tensor, numPositions int, numEmbeddings int, numHeads int, head int) (tensor, tensor, error) {
	tshape := t.Shape
	if len(tshape) != 2 || numPositions < 0 || numEmbeddings <= 0 || numHeads <= 0 || head < 0 || head >= numHeads || numEmbeddings%numHeads != 0 || tshape[0] != numPositions || tshape[1] != 3*numEmbeddings || len(t.Values) < numPositions*3*numEmbeddings {
		return tensor{}, tensor{}, fmt.Errorf("Error while calculating score: invalid dims provided: %v, whereas n_pos: %d, n_embd: %d and n_heads: %d", tshape, numPositions, numEmbeddings, numHeads)
	}

	q := tensor{
		DType:  t.DType,
		Shape:  []int{numPositions, numEmbeddings},
		Values: make([]float32, numPositions*numEmbeddings),
	}
	k := tensor{
		DType:  t.DType,
		Shape:  []int{numPositions, numEmbeddings},
		Values: make([]float32, numPositions*numEmbeddings),
	}
	v := tensor{
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
	headDim := numEmbeddings / numHeads
	// q, k, v actually are qi's above
	q, k, v = sliceHead(q, head, headDim), sliceHead(k, head, headDim), sliceHead(v, head, headDim)

	kT, err := k.transpose()
	if err != nil {
		return tensor{}, tensor{}, err
	}
	qk, err := q.matmul(kT)
	if err != nil {
		return tensor{}, tensor{}, err
	}
	// Dot is over key_dim = n_embd/n_heads.
	dK := float32(headDim)
	res, err := qk.scale(1 / float32(math.Sqrt(float64(dK))))
	return res, v, err
}

func performMLP(tmap map[string]tensor, config GPT2Config, layerNum int, inputTensor tensor) (tensor, error) {
	// apply layer norm-2 wandb
	layerNormWeights := tmap[fmt.Sprintf("h.%d.ln_2.weight", layerNum)]
	layerNormBias := tmap[fmt.Sprintf("h.%d.ln_2.bias", layerNum)]

	inputTensorLayerNormed, err := inputTensor.layerNorm(1)
	if err != nil {
		return tensor{}, err
	}

	layerNormOutput, err := inputTensorLayerNormed.broadcastMultiply(layerNormWeights)
	if err != nil {
		return tensor{}, err
	}
	layerNormOutput, err = layerNormOutput.add(layerNormBias)
	if err != nil {
		return tensor{}, err
	}

	// apply MLP weights
	mlpWeights := tmap[fmt.Sprintf("h.%d.mlp.c_fc.weight", layerNum)]
	mlpBias := tmap[fmt.Sprintf("h.%d.mlp.c_fc.bias", layerNum)]

	mlpOutput, err := layerNormOutput.matmul(mlpWeights)
	if err != nil {
		return tensor{}, err
	}
	mlpOutput, err = mlpOutput.add(mlpBias)
	if err != nil {
		return tensor{}, err
	}

	// apply gelu approx. which is element wise operation
	for i := 0; i < len(mlpOutput.Values); i++ {
		mlpOutput.Values[i] = geluApprox(mlpOutput.Values[i])
	}

	// apply MLP projection
	mlpProjWeights := tmap[fmt.Sprintf("h.%d.mlp.c_proj.weight", layerNum)]
	mlpProjBias := tmap[fmt.Sprintf("h.%d.mlp.c_proj.bias", layerNum)]

	mlpOutput, err = mlpOutput.matmul(mlpProjWeights)
	if err != nil {
		return tensor{}, err
	}
	mlpOutput, err = mlpOutput.add(mlpProjBias)
	if err != nil {
		return tensor{}, err
	}

	mlpOutput, err = inputTensor.add(mlpOutput)
	if err != nil {
		return tensor{}, err
	}

	return mlpOutput, nil
}

func sliceHead(t tensor, head int, headDim int) tensor {
	numPositions, numEmbeddings := t.Shape[0], t.Shape[1]
	start := head * headDim
	out := tensor{
		DType:  t.DType,
		Shape:  []int{numPositions, headDim},
		Values: make([]float32, numPositions*headDim),
	}
	for i := 0; i < numPositions; i++ {
		copy(out.Values[i*headDim:(i+1)*headDim], t.Values[i*numEmbeddings+start:i*numEmbeddings+start+headDim])
	}
	return out
}

func applyCausalMask(t tensor) error {
	minf := math.Inf(-1)
	tshape := t.Shape
	rows, cols := tshape[0], tshape[1]

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if j > i {
				flatIdx, err := t.getFlatArrayIdx(i, j)
				if err != nil {
					return err
				}
				t.Values[flatIdx] = float32(minf)
			}
		}
	}

	return nil
}

// geluApprox(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
func geluApprox(x float32) float32 {
	X := float64(x)
	return 0.5 * x * (1 + float32(math.Tanh(float64(sqrtTwoOverPi)*(X+(float64(0.044715)*math.Pow(X, float64(3)))))))
}
