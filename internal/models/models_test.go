package models_test

import (
	"testing"

	"github.com/codecrafters-io/inference-engine-starter-go/internal/models"
	"github.com/codecrafters-io/inference-engine-starter-go/internal/models/gpt2"
)

func TestGPT2ImplementsModel(t *testing.T) {
	var _ models.Model = (*gpt2.Model)(nil)
}
