// Package generate provides model-independent text generation.
package generate

import (
	"fmt"
	"math/rand"
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

func TopP(model models.Model, ids []int, count int, temperature float32, topP float32) ([]int, error) {
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

		probs, err := logits.Softmax(1, temperature)
		if err != nil {
			return nil, err
		}

		topPSampledIdx, err := topPSample(probs, topP)
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

func topPSample(probs tensor.Tensor, topP float32) (int, error) {
	if topP > 1 {
		return -1, fmt.Errorf("Top P sampling error: Invalid topP %d provided\n", topP)
	}

	numSamples := probs.Shape[1]
	sampleArray := make([]tokenProb, numSamples)
	for i := 0; i < numSamples; i++ {
		sampleArray[i] = tokenProb{
			index: i,
			prob:  probs.Values[i],
		}
	}

	// sort in decreasing order
	slices.SortFunc(sampleArray, func(x, y tokenProb) int {
		if x.prob == y.prob {
			return 0
		}
		if x.prob > y.prob {
			return 1
		}
		return -1
	})

	cumProb := float32(0)
	thresholdIdx := numSamples
	for i := 0; i < numSamples; i++ {
		cumProb += sampleArray[i].prob
		if cumProb > topP {
			thresholdIdx = numSamples
			break
		}
	}

	nucleus := sampleArray[:thresholdIdx+1]
	r := rand.Float32() * cumProb // insted of normalizing nucleus
	runProb := float32(0)
	for i := 0; i < len(nucleus); i++ {
		runProb += nucleus[i].prob
		if runProb <= r {
			return nucleus[i].index, nil
		}
	}

	return nucleus[thresholdIdx-1].index, nil
}
