// Package generate provides model-independent text generation.
package generate

import (
	"cmp"
	"fmt"
	"slices"

	"github.com/codecrafters-io/inference-engine-starter-go/internal/models"
	"github.com/codecrafters-io/inference-engine-starter-go/internal/tensor"
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

func TopP(model models.Model, ids []int, count int, temperature float32, topP float32, random []float64) ([]int, error) {
	if len(random) != count {
		return nil, fmt.Errorf("top-p sampling: expected %d random numbers, got %d", count, len(random))
	}
	context := slices.Clone(ids)
	generated := make([]int, 0, count)
	for step := range count {
		input, err := model.Embed(context)
		if err != nil {
			return nil, err
		}
		logits, err := models.Logits(model, input)
		if err != nil {
			return nil, err
		}

		probs, err := logits.Softmax(1, temperature)
		if err != nil {
			return nil, err
		}

		topPSampledIdx, err := topPSample(probs, topP, random[step])
		if err != nil {
			return nil, err
		}

		generated = append(generated, topPSampledIdx)
		context = append(context, topPSampledIdx)
	}
	return generated, nil
}

type tokenProb struct {
	index int
	prob  float32
}

func topPSample(probs tensor.Tensor, topP float32, random float64) (int, error) {
	if topP <= 0 || topP > 1 {
		return -1, fmt.Errorf("top-p sampling: topP must be in (0, 1], got %v", topP)
	}

	numSamples := probs.Shape[1]
	sampleArray := make([]tokenProb, numSamples)
	for i := 0; i < numSamples; i++ {
		sampleArray[i] = tokenProb{
			index: i,
			prob:  probs.Values[i],
		}
	}

	// Higher probability first. Equal probabilities stay in id order.
	slices.SortFunc(sampleArray, func(x, y tokenProb) int {
		if x.prob > y.prob {
			return -1
		}
		if x.prob < y.prob {
			return 1
		}
		return cmp.Compare(x.index, y.index)
	})

	cumProb := float64(0)
	thresholdIdx := numSamples
	for i, sample := range sampleArray {
		cumProb += float64(sample.prob)
		if cumProb >= float64(topP) {
			thresholdIdx = i + 1
			break
		}
	}

	nucleus := sampleArray[:thresholdIdx]
	cum := 0.0
	for _, sample := range nucleus {
		cum += float64(sample.prob) / cumProb
		if cum > random {
			return sample.index, nil
		}
	}
	return nucleus[len(nucleus)-1].index, nil
}
