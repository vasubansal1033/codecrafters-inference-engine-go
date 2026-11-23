package generate

import (
	"testing"

	"github.com/codecrafters-io/inference-engine-starter-go/internal/tensor"
)

func TestTopPSample(t *testing.T) {
	tests := []struct {
		name   string
		values []float32
		topP   float32
		random float64
		want   int
	}{
		{name: "tie above midpoint", values: []float32{0.5, 0.5}, topP: 1, random: 0.6, want: 1},
		{name: "tie below midpoint", values: []float32{0.5, 0.5}, topP: 1, random: 0.4, want: 0},
		{name: "equal probabilities stay in id order", values: []float32{0.1, 0.4, 0.4, 0.1}, topP: 1, random: 0.2, want: 1},
		{name: "second tied token keeps its id", values: []float32{0.1, 0.4, 0.4, 0.1}, topP: 1, random: 0.5, want: 2},
		{name: "renormalized nucleus", values: []float32{0.2, 0.5, 0.3}, topP: 0.8, random: 0.55, want: 1},
		{name: "single token nucleus", values: []float32{0.2, 0.5, 0.3}, topP: 0.5, random: 0.99, want: 1},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := topPSample(tensor.Tensor{
				Shape:  []int{1, len(test.values)},
				Values: test.values,
			}, test.topP, test.random)
			if err != nil {
				t.Fatal(err)
			}
			if got != test.want {
				t.Fatalf("token = %d, want %d", got, test.want)
			}
		})
	}
}

func TestTopPSampleRejectsInvalidTopP(t *testing.T) {
	_, err := topPSample(tensor.Tensor{Shape: []int{1, 1}, Values: []float32{1}}, 0, 0.1)
	if err == nil {
		t.Fatal("expected an error")
	}
}
