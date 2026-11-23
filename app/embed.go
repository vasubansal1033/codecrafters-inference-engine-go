package main

import (
	"fmt"
	tensorpkg "github.com/codecrafters-io/inference-engine-starter-go/internal/tensor"
	"slices"
)

/*
  embeddings
*/

// embedTokens calculates the embedding and returns result as 2D array
// (token embedding + position embedding of token in prompt)
// tokenId[i] -> i position, tokenId[i] -> token id
func embedTokenIds(tmap map[string]tensorpkg.Tensor, tokenIds []int) (tensorpkg.Tensor, error) {
	numTokenIds := len(tokenIds)
	var embedding []float32
	for pos := range numTokenIds {
		tokenId := tokenIds[pos]

		tokenEmbedding, err := getEmbeddingFromTensorValues(tmap["wte.weight"], tokenId)
		if err != nil {
			return tensorpkg.Tensor{}, err
		}
		posEmbedding, err := getEmbeddingFromTensorValues(tmap["wpe.weight"], pos)
		if err != nil {
			return tensorpkg.Tensor{}, err
		}

		// merge the two embeddings
		mergedEmbedding, err := mergeTokenAndPosEmbeddings(tokenEmbedding, posEmbedding)
		if err != nil {
			return tensorpkg.Tensor{}, err
		}

		embedding = append(embedding, mergedEmbedding...)
	}

	numEmbeddings := len(embedding) / numTokenIds

	return tensorpkg.Tensor{
		DType:  "F32",
		Shape:  []int{numTokenIds, numEmbeddings},
		Values: embedding,
	}, nil
}

func addTokenIdEmbedding(tmap map[string]tensorpkg.Tensor, inputTensor tensorpkg.Tensor, tokenId int) (tensorpkg.Tensor, error) {
	numTokenIds := inputTensor.Shape[0]
	numEmbeddings := inputTensor.Shape[1]

	tokenEmbedding, err := getEmbeddingFromTensorValues(tmap["wte.weight"], tokenId)
	if err != nil {
		return tensorpkg.Tensor{}, err
	}
	posEmbedding, err := getEmbeddingFromTensorValues(tmap["wpe.weight"], numTokenIds)
	if err != nil {
		return tensorpkg.Tensor{}, err
	}

	// merge the two embeddings
	mergedEmbedding, err := mergeTokenAndPosEmbeddings(tokenEmbedding, posEmbedding)
	if err != nil {
		return tensorpkg.Tensor{}, err
	}

	values := append(slices.Clone(inputTensor.Values), mergedEmbedding...)

	return tensorpkg.Tensor{
		DType:  "F32",
		Shape:  []int{numTokenIds + 1, numEmbeddings},
		Values: values,
	}, nil
}

func getEmbeddingFromTensorValues(t tensorpkg.Tensor, id int) ([]float32, error) {
	tensorShape := t.Shape
	tensorValues := t.Values

	// basic sanity check
	if tensorShape[0]*tensorShape[1] != len(tensorValues) {
		return nil, fmt.Errorf("Invalid tensor with shape %d x %d and number of values %d", tensorShape[0], tensorShape[1], len(tensorValues))
	}
	if id < 0 || id >= tensorShape[0] {
		return nil, fmt.Errorf("id %d out of range for shape %v", id, tensorShape)
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
	return readJSON[[]int](path)
}
