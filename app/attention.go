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
	SQRT_TWO_OVER_PI float32 = 0.7978845608028654 // sqrt(2 / pi)
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

func calculateAttention(tmap map[string]tensor, config GPT2Config, layer_num int, inputTensor tensor) (tensor, error) {
	// fmt.Printf("config: %v\nlayer_num: %d\ninputs: %v", config, layer_num, inputs)

	layerAttnWeights := tmap[fmt.Sprintf("h.%d.attn.c_attn.weight", layer_num)]
	layerAttnBias := tmap[fmt.Sprintf("h.%d.attn.c_attn.bias", layer_num)]
	layerProjWeights := tmap[fmt.Sprintf("h.%d.attn.c_proj.weight", layer_num)]
	layerProjBias := tmap[fmt.Sprintf("h.%d.attn.c_proj.bias", layer_num)]

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

	// n_pos := config.NumberOfPositions
	n_pos := inputTensor.Shape[0]
	n_embd := config.NumberOfEmbeddings
	n_heads := config.NumberOfHeads
	if n_heads <= 0 || n_embd%n_heads != 0 {
		return tensor{}, fmt.Errorf("Error while calculating score: invalid dims n_embd: %d and n_heads: %d", n_embd, n_heads)
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
	head_dim := n_embd / n_heads
	attended := tensor{
		DType:  add.DType,
		Shape:  []int{n_pos, n_embd},
		Values: make([]float32, n_pos*n_embd),
	}

	// attention for hth head
	for h := 0; h < n_heads; h++ {
		score, V, err := calculateScore(add, n_pos, n_embd, n_heads, h)
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
		for i := 0; i < n_pos; i++ {
			dst := i*n_embd + h*head_dim
			copy(attended.Values[dst:dst+head_dim], headOut.Values[i*head_dim:(i+1)*head_dim])
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

func calculateBlock(tmap map[string]tensor, config GPT2Config, layer_num int, inputTensor tensor) (tensor, error) {
	layerNorm, err := inputTensor.layerNorm(1)
	if err != nil {
		return tensor{}, err
	}

	// apply layer norm-1 wandb
	layerNormWeights := tmap[fmt.Sprintf("h.%d.ln_1.weight", layer_num)]
	layerNormBias := tmap[fmt.Sprintf("h.%d.ln_1.bias", layer_num)]

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
		layer_num,
		layerNorm,
	)
	if err != nil {
		return tensor{}, err
	}

	resOutput, err := inputTensor.add(attn)
	if err != nil {
		return tensor{}, err
	}

	resOutput, err = performMLP(tmap, config, layer_num, resOutput)
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
func calculateScore(t tensor, n_pos int, n_embd int, n_heads int, head int) (tensor, tensor, error) {
	tshape := t.Shape
	if len(tshape) != 2 || n_pos < 0 || n_embd <= 0 || n_heads <= 0 || head < 0 || head >= n_heads || n_embd%n_heads != 0 || tshape[0] != n_pos || tshape[1] != 3*n_embd || len(t.Values) < n_pos*3*n_embd {
		return tensor{}, tensor{}, fmt.Errorf("Error while calculating score: invalid dims provided: %v, whereas n_pos: %d, n_embd: %d and n_heads: %d", tshape, n_pos, n_embd, n_heads)
	}

	q := tensor{
		DType:  t.DType,
		Shape:  []int{n_pos, n_embd},
		Values: make([]float32, n_pos*n_embd),
	}
	k := tensor{
		DType:  t.DType,
		Shape:  []int{n_pos, n_embd},
		Values: make([]float32, n_pos*n_embd),
	}
	v := tensor{
		DType:  t.DType,
		Shape:  []int{n_pos, n_embd},
		Values: make([]float32, n_pos*n_embd),
	}

	stride := 3 * n_embd
	for i := 0; i < n_pos; i++ {
		row := t.Values[i*stride : (i+1)*stride]
		copy(q.Values[i*n_embd:(i+1)*n_embd], row[:n_embd])
		copy(k.Values[i*n_embd:(i+1)*n_embd], row[n_embd:2*n_embd])
		copy(v.Values[i*n_embd:(i+1)*n_embd], row[2*n_embd:3*n_embd])
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
	head_dim := n_embd / n_heads
	// q, k, v actually are qi's above
	q, k, v = sliceHead(q, head, head_dim), sliceHead(k, head, head_dim), sliceHead(v, head, head_dim)

	kT, err := k.transpose()
	if err != nil {
		return tensor{}, tensor{}, err
	}
	qk, err := q.matmul(kT)
	if err != nil {
		return tensor{}, tensor{}, err
	}
	// Dot is over key_dim = n_embd/n_heads.
	dK := float32(head_dim)
	res, err := qk.scale(1 / float32(math.Sqrt(float64(dK))))
	return res, v, err
}

func performMLP(tmap map[string]tensor, config GPT2Config, layer_num int, inputTensor tensor) (tensor, error) {
	// apply layer norm-2 wandb
	layerNormWeights := tmap[fmt.Sprintf("h.%d.ln_2.weight", layer_num)]
	layerNormBias := tmap[fmt.Sprintf("h.%d.ln_2.bias", layer_num)]

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
	mlpWeights := tmap[fmt.Sprintf("h.%d.mlp.c_fc.weight", layer_num)]
	mlpBias := tmap[fmt.Sprintf("h.%d.mlp.c_fc.bias", layer_num)]

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
	mlpProjWeights := tmap[fmt.Sprintf("h.%d.mlp.c_proj.weight", layer_num)]
	mlpProjBias := tmap[fmt.Sprintf("h.%d.mlp.c_proj.bias", layer_num)]

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

func sliceHead(t tensor, head int, head_dim int) tensor {
	n_pos, n_embd := t.Shape[0], t.Shape[1]
	start := head * head_dim
	out := tensor{
		DType:  t.DType,
		Shape:  []int{n_pos, head_dim},
		Values: make([]float32, n_pos*head_dim),
	}
	for i := 0; i < n_pos; i++ {
		copy(out.Values[i*head_dim:(i+1)*head_dim], t.Values[i*n_embd+start:i*n_embd+start+head_dim])
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
	return 0.5 * x * (1 + float32(math.Tanh(float64(SQRT_TWO_OVER_PI)*(X+(float64(0.044715)*math.Pow(X, float64(3)))))))
}
