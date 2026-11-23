## Inference engine

Go implementation of the CodeCrafters inference engine. GPT-2 is the first model. Tensor math, safetensors loading, layers, generation, and the command line stay model-independent.

Attention stays visible in order: score, causal mask, softmax, value product, head merge, output projection. Shared packages supply those steps.

## Package layout

```text
app/main.go                    thin process entry point
internal/cli/                  command table, input files, JSON output
internal/generate/             model-independent greedy generation
internal/models/               Model interface, registry, shared logits loop
internal/models/gpt2/          GPT-2 configuration, weights, and forward pass
internal/nn/                   reusable layers and attention helpers
internal/safetensors/          safetensors file loader
internal/tensor/               tensor type and arithmetic
```

`internal/models.Model` is the boundary between shared orchestration and model-specific math:

```go
type Model interface {
	NumLayers() int
	Embed(ids []int) (tensor.Tensor, error)
	Attention(layer int, input tensor.Tensor) (tensor.Tensor, error)
	Block(layer int, input tensor.Tensor) (tensor.Tensor, error)
	Head(last tensor.Tensor) (tensor.Tensor, error)
}
```

`models.Logits` runs every block, takes the last position, and calls `Head`. `generate.Greedy` re-embeds the full id list each step. From `[10, 20]`, it scores `[10, 20]`, appends the winner, then scores `[10, 20, winner]`.

## Running commands

```sh
./your_program.sh inspect model.safetensors
./your_program.sh embed model.safetensors ids.json
./your_program.sh attn model.safetensors config.json 0 input.json
./your_program.sh block model.safetensors config.json 0 input.json
./your_program.sh logits model.safetensors config.json ids.json
./your_program.sh generate model.safetensors config.json ids.json tokens.json
```

The CodeCrafters build script compiles `app/*.go`.

## Verification

```sh
test -z "$(gofmt -l .)"
go vet ./...
go build app/*.go
go test ./...
```

Set `GPT2_MODEL` to a real `model.safetensors` to run the optional logits and generation check. Put the matching `config.json` in the same directory.

## Adding a model

Add a package such as `internal/models/llama`, implement `models.Model`, and register its constructor in `internal/models/models.go` under that config's `model_type`. The package owns its config, tensor names, embedding, attention, blocks, and output head.

Shared packages only gain operations another model can reuse. Llama would add `RMSNorm`, `SiLU`, and rotary embeddings to `internal/nn`, and transpose Hugging Face `[out, in]` weights once in its constructor so `nn.Linear` keeps one layout. New dtypes go in the `internal/safetensors` decoder table. New samplers go next to `generate.Greedy`.

## Why loading GPT-2 fits in 2 GB

The Run GPT-2 stage kills a process past 2 GB. The weights file is 548 MB. The first `Load` kept that data region alive with `io.ReadAll` and also copied every tensor into `[]float32`. Both stayed reachable until `Load` returned.

`/usr/bin/time -l` on this Mac reported 1,461,141,504 bytes (1.36 GiB) for the stage prompt. That is the process high-water mark. The tester also counts the kernel file pages, another 548 MB, so the same run sits near 1.9 GB and the overlapped copies cross 2 GB. The message is `process exceeded memory limit (2 GB)`.

`Load` now reads the 8-byte length and the JSON header, then reads one tensor at a time with `ReadAt` into one reusable buffer. `wte.weight` is the big tensor: `50257 * 768 * 4` = 154,389,504 bytes. Smaller tensors reuse that buffer. After `Load` returns, only the floats remain.

A later Mac run stayed near 1.36 GiB. The peak had moved to generation, because each token allocates another transposed `wte` of the same size. Go's `GOGC` default is 100, so collection waits until the heap grows by about the live size. Half a gigabyte of weights plus one transpose lands near 1.4 GB. The tester passes because the raw byte slice no longer sits on the kernel file pages.

## When something fails

| What you see | What is causing it |
| --- | --- |
| `unsupported model_type` | The configuration names a model with no builder in the registry. |
| A matrix multiplication shape error | The weights use a different `[in, out]` layout from `nn.Linear`. |
| A missing command argument prints usage | The command table checks the argument list before running. |
| The optional real-model test is skipped | `GPT2_MODEL` is not set. |
| `process exceeded memory limit (2 GB)` on Run GPT-2 | Load held the raw file bytes and the float copy together, and the tester also counts the kernel file pages. |
| Mac `maximum resident set size` stays near 1.36 GiB after the loader change | The high point is generation: the float weights, the `wte` transpose (154,389,504 bytes), and `GOGC=100` spare heap. |
