## Inference engine

This repository implements the CodeCrafters inference engine challenge in Go. GPT-2 is the first registered model, while tensor math, safetensors loading, neural-network layers, generation, and the command line remain model-independent.

The implementation keeps the arithmetic explicit. Attention still shows the score, causal mask, softmax, value product, head merge, and output projection in order. The reusable packages provide each operation without hiding that flow.

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

`models.Logits` applies every block, selects the final position, and calls the model's output head. `generate.Greedy` re-embeds the complete token sequence on each step, calls that shared logits loop, and appends the highest-scoring token. For example, generating two tokens from `[10, 20]` evaluates `[10, 20]`, appends the first result, then evaluates `[10, 20, result]`.

## Running commands

```sh
./your_program.sh inspect model.safetensors
./your_program.sh embed model.safetensors ids.json
./your_program.sh attn model.safetensors config.json 0 input.json
./your_program.sh block model.safetensors config.json 0 input.json
./your_program.sh logits model.safetensors config.json ids.json
./your_program.sh generate model.safetensors config.json ids.json tokens.json
```

The entry point remains compatible with the CodeCrafters build script, which builds `app/*.go`.

## Verification

Run the same checks used after each refactor commit:

```sh
test -z "$(gofmt -l .)"
go vet ./...
go build app/*.go
go test ./...
```

Set `GPT2_MODEL` to a real `model.safetensors` file to enable the optional logits and generation regression. Put the matching `config.json` in the same directory.

## Adding a model

Create a package such as `internal/models/llama` and implement `models.Model`. The package owns its configuration, tensor names, embedding rules, attention, blocks, and output head. Register its constructor in `internal/models/models.go` under the `model_type` stored in its configuration.

Shared packages should only gain operations that another model can reuse. A Llama implementation would add `RMSNorm`, `SiLU`, and rotary position embeddings to `internal/nn`. Its constructor can transpose Hugging Face weights once if their layout differs from the layout accepted by `nn.Linear`. GPT-2 code does not need to change.

Safetensors support for a new numeric format belongs in the decoder table in `internal/safetensors`. Sampling strategies belong beside `generate.Greedy`, rather than inside a model package.

## When something fails

| What you see | What is causing it |
| --- | --- |
| `unsupported model_type` | The configuration names a model that has no builder in the registry. |
| A matrix multiplication shape error | A model supplied weights in a different `[in, out]` layout from `nn.Linear`. |
| A missing command argument prints usage | The command table validates each command's argument list before running it. |
| The optional real-model test is skipped | `GPT2_MODEL` is not set. |
