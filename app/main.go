package main

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strconv"
)

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

		modelPath := os.Args[2]
		configPath := os.Args[3]
		layerNumInput := os.Args[4]
		inputsPath := os.Args[5]

		tensors, err := loadTensors(modelPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading tensors: %s\n", err)
			os.Exit(1)
		}

		config, err := loadConfig(configPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error while loading configs: %s\n", err)
			os.Exit(1)
		}

		layerNum, err := strconv.Atoi(layerNumInput)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error while parisng layer number: %s\n", err)
			os.Exit(1)
		}

		inputs, err := loadInputRows(inputsPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error while loading inputs: %s\n", err)
			os.Exit(1)
		}

		attn, err := calculateAttenion(
			tensors,
			config,
			layerNum,
			inputs,
		)

		fmt.Printf("%v", attn)

	default:
		fmt.Fprintf(os.Stderr, "Unknown command %s\n", command)
		os.Exit(1)
	}
}
