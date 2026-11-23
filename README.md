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

## Why loading GPT-2 fits in 2 GB

The Run GPT-2 stage stops a process that goes past 2 GB. On 23 Nov 2026 the stage log described GPT-2's weights file as 548 MB. The first `Load` pulled that whole data region into one byte slice with `io.ReadAll`, then copied every tensor into a `[]float32`. The byte slice stayed reachable until `Load` returned, because each copy read from it. The heap therefore held the file bytes and the floats at the same time: about 548 MB plus another 548 MB, before the collector's spare room.

On this Mac, `/usr/bin/time -l` reported a maximum resident set size of 1,461,141,504 bytes (1.36 GiB) for `generate` on the stage prompt. That counter is the process's own high-water mark. The tester also charges the file pages the kernel keeps after the read. Those pages are the same 548 MB file, so the tester's total sits near 1.36 GiB plus 548 MB, around 1.9 GB. The overlapped load allocations push the rest of the way past 2 GB, and the stage prints `process exceeded memory limit (2 GB)`.

`Load` now reads the 8-byte header length and the JSON header, then records the file offset where the data region begins. Each tensor is read from that region with `ReadAt` into one reusable buffer.

`wte.weight` is the large example. The stage config has `vocab_size` 50257 and `n_embd` 768, so the tensor is `50257 * 768 * 4` bytes, which is 154,389,504 bytes. The loop grows `raw` to that length, reads those bytes at `dataStart + data_offsets[0]`, and turns each 4 bytes into one float. A smaller tensor reuses the same buffer. A larger one replaces it. After `Load` returns, nothing references `raw`. The floats for each tensor are what the model keeps.

A later Mac run stayed near 1.36 GiB and rose a little. The high-water mark had moved from load to generation. Generation keeps the floats, and each new token allocates a transposed `wte` for the output projection, another 154,389,504 bytes. Go's `GOGC` default is 100, including the Go 1.26 toolchain in `go.mod`: the collector waits until the heap has grown by about the size of the live heap. Live weights of roughly half a gigabyte, plus one transpose, put that threshold near 1.4 GB. The Mac counter records this later peak, so deleting the load-time byte slice does not lower it. The tester accepts the run because the load no longer stacks the raw byte slice on top of the kernel's file pages.

## When something fails

| What you see | What is causing it |
| --- | --- |
| `unsupported model_type` | The configuration names a model that has no builder in the registry. |
| A matrix multiplication shape error | A model supplied weights in a different `[in, out]` layout from `nn.Linear`. |
| A missing command argument prints usage | The command table validates each command's argument list before running it. |
| The optional real-model test is skipped | `GPT2_MODEL` is not set. |
| `process exceeded memory limit (2 GB)` on Run GPT-2 | Load held the raw file bytes and the float copy together, and the tester also counts the kernel's file pages. |
| Mac `maximum resident set size` stays near 1.36 GiB after the loader change | The counter's high point is generation: the float weights, the `wte` transpose (154,389,504 bytes), and `GOGC=100` spare heap. |
