package main

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
)

// tensor is one table of numbers from a safetensors file.
type tensor struct {
	DType  string
	Shape  []int
	Values []float32
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

/*
  embeddings
*/

// embedTokens calculates the embedding and returns result as 2D array
// (token embedding + position embedding of token in prompt)
// tokenId[i] -> i position, tokenId[i] -> token id
func embedTokenIds(tmap map[string]tensor, tokenIds []int) ([][]float32, error) {
	numTokenIds := len(tokenIds)
	embedding := make([][]float32, numTokenIds)
	for pos := range numTokenIds {
		tokenId := tokenIds[pos]

		tokenEmbedding, err := getEmbeddingFromTensorValues(tmap["wte.weight"], tokenId)
		if err != nil {
			return nil, err
		}
		posEmbedding, err := getEmbeddingFromTensorValues(tmap["wpe.weight"], pos)
		if err != nil {
			return nil, err
		}

		// merge the two embeddings
		mergedEmbedding, err := mergeTokenAndPosEmbeddings(tokenEmbedding, posEmbedding)
		if err != nil {
			return nil, err
		}

		embedding[pos] = mergedEmbedding
	}

	return embedding, nil
}

func getEmbeddingFromTensorValues(t tensor, id int) ([]float32, error) {
	tensorShape := t.Shape
	tensorValues := t.Values

	// basic sanity check
	if tensorShape[0]*tensorShape[1] != len(tensorValues) {
		return nil, fmt.Errorf("Invalid tensor with shape %d x %d and number of values %d", tensorShape[0], tensorShape[1], len(tensorValues))
	}

	/*
		we actually want tensorValues[tokenId]
		e.g.
		[1, 2, 3, 4, 5, 6] and shape [2, 3]
		-> [[1,  2,  3],[4,  5,  6]]
		   0,0 0,1 0,2 1,0 1,1 1,2
		      token1      token2
		each row represents token embedding

		For tokenId 1 ->
			skip shape[1]*tokenId elements and take shape[1] elements
	*/
	// shape is [rows, dim]. One row is dim floats, not rows floats.
	embedding := make([]float32, tensorShape[1])
	for i := 0; i < tensorShape[1]; i++ {
		tensorValuesOffset := tensorShape[1] * id
		embedding[i] = tensorValues[tensorValuesOffset+i]
	}

	return embedding, nil
}

func mergeTokenAndPosEmbeddings(tokenEmbedding []float32, posEmbedding []float32) ([]float32, error) {
	if len(tokenEmbedding) != len(posEmbedding) {
		return nil, fmt.Errorf("Position mismatch: len(tokenEmbedding) %d, len(posEmbedding) %d", len(tokenEmbedding), len(posEmbedding))
	}

	l := len(tokenEmbedding)
	mergedEmbedding := make([]float32, l)
	for i := 0; i < l; i++ {
		mergedEmbedding[i] = tokenEmbedding[i] + posEmbedding[i]
	}

	return mergedEmbedding, nil
}

func loadTokenIds(path string) ([]int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	reader := bufio.NewReader(f)
	decoder := json.NewDecoder(reader)

	var ids []int
	if err = decoder.Decode(&ids); err != nil {
		return nil, err
	}

	return ids, nil
}

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

// Usage: your_program.sh <command> <arg1> <arg2> ...
func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "usage: your_program.sh <command> [<args>...]\n")
		os.Exit(1)
	}

	switch command := os.Args[1]; command {
	case "inspect":
		tensors, err := loadTensors(os.Args[2])
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading tensors: %s\n", err)
			os.Exit(1)
		}

		names := make([]string, 0, len(tensors))
		for name := range tensors {
			names = append(names, name)
		}
		sort.Strings(names)

		type summary struct {
			Name  string    `json:"name"`
			DType string    `json:"dtype"`
			Shape []int     `json:"shape"`
			First []float32 `json:"first"`
		}
		summaries := make([]summary, 0, len(names))
		for _, name := range names {
			t := tensors[name]
			first := t.Values[:min(4, len(t.Values))]
			summaries = append(summaries, summary{name, t.DType, t.Shape, first})
		}
		encoded, err := json.Marshal(summaries)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error encoding output: %s\n", err)
			os.Exit(1)
		}
		fmt.Println(string(encoded))
	case "embed":
		if (len(os.Args)) < 4 {
			fmt.Fprintf(os.Stderr, "usage: your_program.sh embed model.safetensors ids.json\n")
			os.Exit(1)
		}

		tensors, err := loadTensors(os.Args[2])
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading tensors: %s\n", err)
			os.Exit(1)
		}
		tokenIds, err := loadTokenIds(os.Args[3])
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading token ids: %s\n", err)
			os.Exit(1)
		}

		mergedEmbeddings, err := embedTokenIds(tensors, tokenIds)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error embedding tokens: %s\n", err)
			os.Exit(1)
		}
		encoded, err := json.Marshal(mergedEmbeddings)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error encoding output: %s\n", err)
			os.Exit(1)
		}
		fmt.Println(string(encoded))
	case "attn":
		if len(os.Args) < 5 {
			fmt.Fprintf(os.Stderr, "usage: your_program.sh attn model.safetensors config.json <layer_num> input.json\n")
			os.Exit(1)
		}

		configPath := os.Args[3]
		config, err := loadConfig(configPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error while loading configs")
			os.Exit(1)
		}

		configMarshal, err := json.Marshal(config)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error encoding config: %s\n", err)
			os.Exit(1)
		}
		fmt.Println(string(configMarshal))
	default:
		fmt.Fprintf(os.Stderr, "Unknown command %s\n", command)
		os.Exit(1)
	}
}
