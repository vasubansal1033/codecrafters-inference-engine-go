package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
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

func loadInputRows(path string) ([][]int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	reader := bufio.NewReader(f)
	decoder := json.NewDecoder(reader)

	var inputRows [][]int
	if err := decoder.Decode(&inputRows); err != nil {
		return nil, err
	}

	return inputRows, nil
}

func calculateAttenion(tmap map[string]tensor, config GPT2Config, layer_num int, inputs [][]int) ([][]float32, error) {
	// fmt.Printf("config: %v\nlayer_num: %d\ninputs: %v", config, layer_num, inputs)

	layerAttnWeights := tmap["h.0.attn.c_attn.weight"]
	layerAttnBias := tmap["h.0.attn.c_attn.bias"]
	layerProjWeights := tmap["h.0.attn.c_proj.weight"]
	layerProjBias := tmap["h.0.attn.c_proj.bias"]

	fmt.Println("Shapes:")
	fmt.Printf("n_embd: %d\n", config.NumberOfEmbeddings)          // 768
	fmt.Printf("attention weights: %v\n", layerAttnWeights.Shape)  // [768, 2304]
	fmt.Printf("attention bias: %v\n", layerAttnBias.Shape)        // [2304]
	fmt.Printf("projection weights: %v\n", layerProjWeights.Shape) // [768, 768]
	fmt.Printf("projection bias: %v\n", layerProjBias.Shape)       // [768]

	t1 := tensor{
		DType:  "float32",
		Shape:  []int{1, 2},
		Values: []float32{1.0, 2.0},
	}
	t2 := tensor{
		DType:  "float32",
		Shape:  []int{2, 2},
		Values: []float32{3.0, 4.0, 5.0, 6.0},
	}

	res, err := t1.multiply(t2)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Multiplication error: %s\n", err)
	}
	fmt.Printf("Mult result: %+v\n", res)

	return nil, nil
}
