package main

import (
	"encoding/json"
	"os"
)

type tokenConfig struct {
	Count int `json:"count"`
}

func loadTokenConfig(path string) (tokenConfig, error) {
	var config tokenConfig
	data, err := os.ReadFile(path)
	if err != nil {
		return config, err
	}
	err = json.Unmarshal(data, &config)
	return config, err
}
