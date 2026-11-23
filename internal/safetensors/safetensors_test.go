package safetensors

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestLoadReadsTensorsWithoutWholeFileCopy(t *testing.T) {
	header := []byte(`{"weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"ids":{"dtype":"I32","shape":[2],"data_offsets":[16,24]}}`)
	payload := make([]byte, 24)
	binary.LittleEndian.PutUint32(payload[0:], math.Float32bits(1.5))
	binary.LittleEndian.PutUint32(payload[4:], math.Float32bits(-2))
	binary.LittleEndian.PutUint32(payload[8:], math.Float32bits(3))
	binary.LittleEndian.PutUint32(payload[12:], math.Float32bits(4))
	binary.LittleEndian.PutUint32(payload[16:], 7)
	neg := int32(-9)
	binary.LittleEndian.PutUint32(payload[20:], uint32(neg))

	dir := t.TempDir()
	path := filepath.Join(dir, "model.safetensors")
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := binary.Write(file, binary.LittleEndian, uint64(len(header))); err != nil {
		t.Fatal(err)
	}
	if _, err := file.Write(header); err != nil {
		t.Fatal(err)
	}
	if _, err := file.Write(payload); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}

	tensors, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	weight := tensors["weight"]
	if len(weight.Shape) != 2 || weight.Shape[0] != 2 || weight.Shape[1] != 2 {
		t.Fatalf("weight shape = %v", weight.Shape)
	}
	want := []float32{1.5, -2, 3, 4}
	for i, value := range want {
		if weight.Values[i] != value {
			t.Fatalf("weight[%d] = %v, want %v", i, weight.Values[i], value)
		}
	}
	if got := tensors["ids"].Values; len(got) != 2 || got[0] != 7 || got[1] != -9 {
		t.Fatalf("ids = %v", got)
	}
}
