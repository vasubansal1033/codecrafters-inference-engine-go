## Inference engine

Go implementation of the CodeCrafters inference engine. The model it runs is GPT-2.

Running example: "The capital of France is".

| Position | Id | Token |
| --- | --- | --- |
| 0 | 464 | The |
| 1 | 3139 | ` capital` |
| 2 | 286 | ` of` |
| 3 | 4881 | ` France` |
| 4 | 318 | ` is` |

| Topic | Where |
| --- | --- |
| Stage arithmetic | `concepts.md` |
| Prefill, decode, causal mask | `kv-cache.md` |
| Commands, layout, memory | this file |

Attention order: score, causal mask, softmax, product with the values, merge the heads, output projection. `internal/nn` provides the operations. `internal/models/gpt2` chooses the weights.

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

`app/main.go` calls `cli.Run`. `gpt2.go` holds the config, the weights, and the key-value cache. `model.go` is one transformer layer. It reads those weights and receives a cache pointer when the caller has one.

#### Model interface

Generation, logits, and the command line call this.

```go
type Model interface {
	NumLayers() int
	Embed(ids []int) (tensor.Tensor, error)
	EmbedAt(ids []int, position int) (tensor.Tensor, error)
	Attention(layer int, input tensor.Tensor) (tensor.Tensor, error)
	Block(layer int, input tensor.Tensor) (tensor.Tensor, error)
	Head(last tensor.Tensor) (tensor.Tensor, error)
	Forward(input tensor.Tensor, cache *kvcache.KVCache) (tensor.Tensor, error)
	GetKVCache() (*kvcache.KVCache, error)
}
```

| Call | What it does |
| --- | --- |
| `Attention`, `Block` | `attn` and `block`. Recompute every input row. Cache is nil. |
| `models.Logits` | `Forward` with a nil cache, then `Head` on the last row. |
| `Head` | Final layer norm, then multiply by the transposed token-embedding matrix. |
| `Greedy`, `TopP` | `GetKVCache` once. Pass that pointer into every `Forward`. |

`Embed` places ids at positions `0` onward. `EmbedAt` places ids at a chosen position. Shapes are in `kv-cache.md`.

#### Greedy

Request: two tokens from the prompt above. A tie goes to the smaller id, because the search returns the first maximum.

| Step | Embed | Then |
| --- | --- | --- |
| 1 | ids at positions `0` through `4` | `Forward` with the model's cache. Append the id with the largest logit. |
| 2, if that id is `262` (" the") | `EmbedAt(262, 5)` | `Forward` with the same cache. |

The command prints the two new ids, in order. Context after step 1: `[464, 3139, 286, 4881, 318, 262]`.

#### Top-p

Used when `tokens.json` sets `temperature` above zero. The file also carries `top_p`. Temperature `0`, or a missing temperature, stays on greedy. A greedy file is `{"count": n}`.

1. Standard input supplies one float64 per generated token, one number per line. The count must equal `count`.
2. Divide each logit by the temperature. Turn the result into probabilities.
3. Sort by descending probability. Equal probabilities stay in id order.
4. Keep the shortest prefix whose probabilities add up to at least `top_p`.
5. Divide those kept probabilities by their sum.
6. Walk the prefix. The token is the first id whose running total is greater than the stdin number.

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

1. Add a package such as `internal/models/llama`.
2. Implement `models.Model`. The package owns its config, tensor names, embedding, attention, blocks, output head, and the cache its `Forward` uses.
3. Register the constructor in `internal/models/models.go` under that config's `model_type`.

Shared code changes only when another model can reuse the operation.

| New piece | Where it goes |
| --- | --- |
| `RMSNorm`, `SiLU`, rotary embeddings | `internal/nn` |
| Hugging Face `[out, in]` transpose, once in the constructor | the model package, so `nn.Linear` keeps one layout |
| A new dtype | `internal/safetensors` decoder table |
| A new sampler | `internal/generate`, next to `Greedy` and `TopP` |

## Memory while running GPT-2

The Run GPT-2 stage stops a process that passes 2 GB. Two allocations decide that. Shapes for the second are in `kv-cache.md`.

#### Loading the weights

The weights file is 548 MB.

| | Heap during `Load` | After `Load` returns |
| --- | --- | --- |
| `io.ReadAll`, until 23 Nov 2026 | The 548 MB data region and the decoded `[]float32` weights, both live. | Floats remain. The byte slice can be collected only after the copy finishes. |
| `ReadAt`, current `Load` | One reusable byte buffer. It grows to the largest tensor and is reused for the smaller ones. | The buffer is unreferenced. The model keeps the floats. |

Current `Load`:

1. Read the 8-byte header length.
2. Read the JSON header.
3. Record the file position where the numbers begin.
4. `ReadAt` one tensor into the shared buffer.
5. Decode those bytes into that tensor's `[]float32`.

Largest tensor: `wte.weight`, `50257 * 768 * 4` = 154,389,504 bytes.

| Measurement | Value |
| --- | --- |
| Mac `/usr/bin/time -l` maximum resident set size, `generate` on the stage prompt, 23 Nov 2026 | 1,461,141,504 bytes (1.36 GiB). After the loader change, before the key-value cache. Process high-water mark only. |
| Also counted by the tester | Kernel pages for the open 548 MB file. |
| 1.36 GiB plus that file | About 1.9 GB. |
| Same total with the `ReadAll` byte slice still live | Crosses 2 GB. The stage prints `process exceeded memory limit (2 GB)`. |

The Mac figure stayed near 1.36 GiB after the byte slice was removed. The high point is generation, not loading. Each token's `Head` (`internal/models/gpt2/gpt2.go`) transposes `wte.weight`, another 154,389,504 bytes. Go 1.26 (`go.mod`) uses `GOGC` 100, so collection waits until the heap has grown by about as much as the live data. Float weights plus that transpose sit near 1.4 GB of process memory.

#### Each token after the prompt

GPT-2 cache size: 12 layers, 12 heads, 64 floats per row, keys and values. One token is `2 * 12 * 12 * 64 * 4` = 73,728 bytes. The five prompt tokens are `5 * 73728` = 368,640 bytes.

| | Second token, prompt plus `262` | What is allocated |
| --- | --- | --- |
| Until 25 Nov 2026 | `Embed` of all six ids, positions `0` through `5` | Layer norm, `c_attn`, scores, and the MLP for six rows in every layer. Stored keys and values do not shrink this while the block input is still the full list. |
| Current `embedStep` | `EmbedAt(262, 5)` | Block input `[1, 768]`. Per head, new Q, K, V are `[1, 64]`. Score is `[1, past+1]`. |
| Every token, both versions | `Head` | `wte` transpose, 154,389,504 bytes. The cache is unused here. |

## When something fails

| What you see | What is causing it |
| --- | --- |
| `unsupported model_type` | The configuration names a model with no builder in the registry. |
| A matrix multiplication shape error | The weights use a different `[in, out]` layout from `nn.Linear`. |
| A missing command argument prints usage | The command table checks the argument list before running. |
| The optional real-model test is skipped | `GPT2_MODEL` is not set. |
| `process exceeded memory limit (2 GB)` on Run GPT-2 | `Load` held the raw file bytes from `io.ReadAll` and the float weights at the same time. The tester also counts the kernel's cached copy of the 548 MB file. |
| Mac `maximum resident set size` stays near 1.36 GiB | Measured on 23 Nov 2026 after the loader change and before the key-value cache. The high-water mark is the float weights, the `wte` transpose (154,389,504 bytes), and the spare heap from `GOGC=100`. The tester adds the kernel file cache on top. |
| A later token still runs the whole context | That step embedded every id. After the first token, the loop embeds only the new id, at its index in the full list. |
| `top-p sampling: expected N random numbers, got M` | Standard input did not contain one float64 per generated token. |
| Top-p picks a different id from the tester | The kept probabilities were not divided by their sum, the stdin number was compared with `<=` instead of `>`, or equal probabilities were not left in id order. |
