## Inference engine

This repository is a Go implementation of the CodeCrafters inference engine. GPT-2 is the model it runs today. Tensor math, the safetensors loader, the shared layers, generation, and the command line do not depend on GPT-2.

A GPT-2 attention step stays in this order: score, causal mask, softmax, product with the values, merge of the heads, then the output projection. The shared packages provide those operations. The GPT-2 package chooses the weights for each step.

## Package layout

```text
app/main.go                         process entry point
internal/cli/                       command table, input files, JSON output
internal/generate/                  greedy and top-p generation
internal/models/                    Model interface, registry, logits helper
internal/models/gpt2/gpt2.go        GPT-2 model, config, and the steps callers use
internal/models/gpt2/model.go       one transformer layer: attention, block, MLP
internal/models/gpt2/embed.go       token and position rows
internal/models/kv_cache/           keys and values stored per layer and head
internal/nn/                        reusable layers and attention helpers
internal/safetensors/               safetensors file loader
internal/tensor/                    tensor type and arithmetic
```

`app/main.go` calls `cli.Run`. `internal/models/gpt2/gpt2.go` holds the loaded model: its config, its weights, and its key-value cache. The methods on that model are the steps the rest of the engine calls. `internal/models/gpt2/model.go` is one transformer layer. It reads the model's weights and config, and it receives a cache pointer when the caller has one.

`internal/models.Model` is the boundary between shared orchestration and a model's own math:

```go
type Model interface {
	NumLayers() int
	Embed(ids []int) (tensor.Tensor, error)
	Attention(layer int, input tensor.Tensor) (tensor.Tensor, error)
	Block(layer int, input tensor.Tensor) (tensor.Tensor, error)
	Head(last tensor.Tensor) (tensor.Tensor, error)
	Forward(input tensor.Tensor, cache *kvcache.KVCache) (tensor.Tensor, error)
	GetKVCache() (*kvcache.KVCache, error)
}
```

`Attention` and `Block` serve the `attn` and `block` commands. Each one recomputes every row of the input it is given, and it passes a nil cache into the layer. `models.Logits` calls `Forward` with a nil cache, so the `logits` command does the same full recompute, then `Head` on the last row. `Head` applies the final layer norm and multiplies by the transposed token-embedding matrix.

`Greedy` and `TopP` are the generation path. Each one calls `GetKVCache` once, at the start of that generation, and passes the returned pointer into `Forward` for every new token. `Forward` walks the layers and gives that same pointer to each block. The layer's attention is the step that reads stored keys and values and appends the rows from the current call. The cache keeps K and V in separate maps, indexed by layer and then by head. `Get` stacks the stored rows for one slot. An empty slot has shape `[0, 0]`. `Put` appends the tensor from the current call onto that slot.

#### One greedy generation

Take the prompt ids `[10, 20]` and a request for two tokens. The first step embeds `[10, 20]`, runs `Forward` with the model's cache, and appends the vocabulary id with the largest logit. If that id is `7`, the second step embeds `[10, 20, 7]`, runs `Forward` with the same cache, and appends the next id. The command prints those two new ids, in order. When two logits tie, the smaller id wins, because the search returns the first maximum.

#### Top-p sampling

When `tokens.json` sets `temperature` to a value other than zero, generation uses top-p. The same file carries `top_p`. Standard input supplies one float64 for each token to generate, one number per line, and the count of numbers must equal `count`. Each logit is divided by the temperature and turned into a probability. The probabilities are sorted from high to low, and equal probabilities stay in id order. The shortest prefix whose probabilities add up to at least `top_p` is kept. Those kept probabilities are divided by their sum. Walking that prefix, the first id whose running total is greater than the stdin number is the token. A temperature of `0`, or a file that leaves temperature out, stays on the greedy path. A greedy file is `{"count": n}`.

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

Add a package such as `internal/models/llama`, implement `models.Model`, and register its constructor in `internal/models/models.go` under that config's `model_type`. The package owns its config, tensor names, embedding, attention, blocks, output head, and the cache its `Forward` method uses.

Shared packages gain an operation when another model can reuse it. Llama would add `RMSNorm`, `SiLU`, and rotary embeddings to `internal/nn`, and would transpose Hugging Face `[out, in]` weights once in its constructor so `nn.Linear` keeps one layout. New dtypes go in the `internal/safetensors` decoder table. New samplers go in `internal/generate`, next to `Greedy` and `TopP`.

## Memory while running GPT-2

The Run GPT-2 stage stops a process that passes 2 GB. The weights file is 548 MB. `Load` reads the 8-byte header length, reads the JSON header, and records the file position where the numbers begin. It then reads one tensor at a time with `ReadAt` into a single buffer. That buffer grows to the largest tensor and is reused for the smaller ones. `wte.weight` is the largest: `50257 * 768 * 4` = 154,389,504 bytes. Each tensor is decoded into its own `[]float32`. When `Load` returns, the process holds those floats.

`/usr/bin/time -l` on this Mac reported a maximum resident set size of 1,461,141,504 bytes (1.36 GiB) for the stage prompt. That figure is the process high-water mark. The tester's 2 GB limit also counts the kernel pages that cache the open file, about another 548 MB, so the limit sees the process and those file pages together.

During generation, each token builds a transpose of `wte.weight`, another 154,389,504 bytes. Go's default `GOGC` is 100, so collection waits until the heap has grown by about as much as the live data. The float weights plus that transpose sit near 1.4 GB of process memory.

## When something fails

| What you see | What is causing it |
| --- | --- |
| `unsupported model_type` | The configuration names a model with no builder in the registry. |
| A matrix multiplication shape error | The weights use a different `[in, out]` layout from `nn.Linear`. |
| A missing command argument prints usage | The command table checks the argument list before running. |
| The optional real-model test is skipped | `GPT2_MODEL` is not set. |
| `process exceeded memory limit (2 GB)` on Run GPT-2 | The tester counts process memory and the kernel's cached copy of the 548 MB file. A second full copy of the raw file, kept alongside the float weights, crosses the limit. |
| Mac `maximum resident set size` stays near 1.36 GiB | The reported figure is the process high-water mark: the float weights, the `wte` transpose (154,389,504 bytes), and the spare heap from `GOGC=100`. The tester adds the kernel file cache on top of that. |
| `top-p sampling: expected N random numbers, got M` | Standard input did not contain one float64 per generated token. |
| Top-p picks a different id from the tester | The kept probabilities were not divided by their sum, the stdin number was compared with `<=` instead of `>`, or equal probabilities were not left in id order. |
