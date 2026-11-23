// Package generate provides model-independent text generation.
package generate

import (
	"slices"

	"github.com/codecrafters-io/inference-engine-starter-go/internal/models"
)

func Greedy(model models.Model, ids []int, count int) ([]int, error) {
	context := slices.Clone(ids)
	generated := make([]int, 0, count)
	for range count {
		input, err := model.Embed(context)
		if err != nil {
			return nil, err
		}
		logits, err := models.Logits(model, input)
		if err != nil {
			return nil, err
		}
		next := slices.Index(logits.Values, slices.Max(logits.Values))
		generated = append(generated, next)
		context = append(context, next)
	}
	return generated, nil
}
