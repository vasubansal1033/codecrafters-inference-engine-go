## Extensions

GPT-2 is multi-head attention with a learned position table (`wpe`). The models below are small enough to keep next to that 548 MB file and run on a laptop. Each one is here because it adds one idea the current engine does not have. Falcon-7B and DeepSeek-V2-Lite are the well-known versions of multi-query attention and latent attention, and both are too large for this list.

A new architecture is a new package under `internal/models`, registered in `internal/models/models.go` under the config's `model_type`. How that registration works is in Adding another model in `README.md`.

The loader decodes `F32` and `I32` only. Several files below are stored as float16 or bfloat16. Widening those dtypes to `float32` in `internal/safetensors` comes before any of those models can load. A 16-bit file then occupies about twice its download size as floats.

## What to add, in order

| Step | Checkpoint | Download | New idea | Why this step |
| --- | --- | --- | --- | --- |
| 1 | `bigcode/tiny_starcoder_py` | 657 MB, float32 safetensors | Multi-query attention | The file loads with today's decoder. Positions stay a learned table, so the only new piece is one key/value head shared by every query head. |
| 2 | `EleutherAI/pythia-70m` | 166 MB, float16 safetensors | Rotary positions | Smallest RoPE file here. Attention is still one key/value head per query head. |
| 3 | `HuggingFaceTB/SmolLM2-135M` | 269 MB, bfloat16 safetensors | Grouped-query attention | Three key/value heads and nine query heads. Reuses RoPE. Also brings RMSNorm and SwiGLU. |
| 4 | `facebook/xglm-564M` | 1.13 GB, float16 pickle | Sinusoidal positions | The position rows are computed, not stored. The file has to be converted to safetensors before `Load` can read it. |
| 5 | `bigscience/bloom-560m` | about 1.1 GB, float16 | ALiBi | Scores get a per-head distance penalty. Attention stays multi-head. There is no position table. |
| 6 | `google/gemma-3-270m` | 536 MB, bfloat16 safetensors | Sliding-window attention | Most layers only see the last 512 positions. The repo is gated. This model is also multi-query, so it comes after step 1. |
| 7 | the GPT-2 weights already in use | 548 MB, float32 | Flash attention | Same math as the current attention. The check is that the logits still match. |
| 8 | none | | Multi-head latent attention | No published checkpoint is near this size. DeepSeek-V2-Lite is 15.7B parameters. |

## Multi-query attention

`bigcode/tiny_starcoder_py` is a 164M model, `model_type` `gpt_bigcode`. It has 20 layers, width 768, and 12 query heads. The key and the value are a single head of width 64. The fused attention projection is `768 + 2 * 64` = 896 columns, not GPT-2's `3 * 768`.

The position table is learned, with 8,192 rows, so `Embed` and `EmbedAt` stay the same shape of operation. The cache stores one key row and one value row per layer per token, and every query head reads that pair. Fill-in-the-middle tokens are in the vocabulary. A normal left-to-right prompt does not have to use them.

`inspect` can open this file today. `generate` cannot, until a constructor is registered for `gpt_bigcode`.

## Rotary positions

`EleutherAI/pythia-70m` is `model_type` `gpt_neox`: 6 layers, width 512, 8 heads. `rotary_pct` is 0.25 and `rotary_emb_base` is 10,000, so RoPE rotates the first quarter of each head and leaves the rest alone. There is no `wpe` table. The position index already passed to `EmbedAt` is the index RoPE needs.

The block is also different. Pythia uses a parallel residual: attention and the MLP both read the block input, and both results are added back to it. GPT-2 runs the MLP on the output of attention. A Pythia package has to do both, or the logits will not match a reference.

## Grouped-query attention

`HuggingFaceTB/SmolLM2-135M` is `model_type` `llama`: 30 layers, width 576, 9 query heads, 3 key/value heads, head width 64. Each key/value head is shared by three query heads. Multi-query attention is the same layout with `num_key_value_heads` set to 1, so the cache from step 1 should already accept a head count that is smaller than the query head count.

Positions are full-head RoPE with `rope_theta` 100,000. That is a different setting from Pythia's partial RoPE. The block uses RMSNorm and a SwiGLU MLP (`hidden_act` `silu`, intermediate width 1,536). The token table is tied to the output head. The published `model.safetensors` is bfloat16.

## Sinusoidal positions

`facebook/xglm-564M` is `model_type` `xglm`: 24 layers, width 1,024, 16 heads, vocabulary 256,008. The published weights are a float16 `pytorch_model.bin` of 1.13 GB. There is no safetensors file in that repo, and this loader does not read pickles.

The position rows are the usual sine and cosine table. XGLM builds them with an offset of 2, and `scale_embedding` multiplies the token row by the square root of the width before the add. Nothing named like `wpe` is in the checkpoint. Attention itself is multi-head, with layer norm and GELU, so this step is the position table and the embedding scale.

## ALiBi

`bigscience/bloom-560m` is `model_type` `bloom`: 24 layers, width 1,024, 16 heads, vocabulary 250,680, about 559M parameters stored in float16 (about 1.1 GB). If the repo only has a pickle, it needs the same safetensors conversion as XGLM.

ALiBi does not add a position vector to the token row. For head `i` of `n` heads, the score between a query at position `q` and a key at position `k` is reduced by `slope_i * (q - k)`, with `slope_i = 2^(-8 * i / n)`. The slopes are constants, not weights. The causal mask still hides future keys. BLOOM also applies a layer norm to the token embedding before the first block.

## Sliding-window attention

`google/gemma-3-270m` is a gated repo. The weights file is 536 MB of bfloat16. The text model has 18 layers, width 640, 4 query heads, and 1 key/value head, so it needs the multi-query cache from step 1. Head width is 256.

Fifteen layers use a window of 512: a query may read keys in `(position - 511)` through its own position, and the causal mask still applies. Every sixth layer is global and may read the whole past. Positions are RoPE, with one base for the local layers and another for the global layers. The MLP is gated. There is also a query/key norm. Those extras are why this file is last among the real checkpoints: the new idea to check is the window, after multi-query attention and RoPE already exist.

## Flash attention

Flash attention is not a different model. It computes the same scores, mask, softmax, and product with the values, but it tiles the work so the full score matrix does not have to sit in memory. The GPT-2 weights are the test. A flash path that disagrees with the current attention by more than the tester's `0.0001` is wrong, even if it is faster. The naive path in `internal/models/gpt2/model.go` stays, so the comparison has something to run against.

## Multi-head latent attention

There is no checkpoint on this list. DeepSeek-V2-Lite uses latent attention, and it is a 15.7B mixture-of-experts model (about 2.4B parameters active per token). That is a different project from the files above.

Latent attention stores one compressed vector per token per layer, plus a small extra key that carries RoPE, and expands that vector into per-head keys and values with learned matrices. The current cache stores a full key and a full value for every head. A first implementation can use a tiny config with random weights, or weights copied out of a reference forward pass, and check one layer's output before any attempt to load a DeepSeek file.

## When a new model is wrong

| What you see | What is causing it |
| --- | --- |
| `unsupported dtype` on Pythia, SmolLM2, BLOOM, or Gemma | The file is float16 or bfloat16. The decoder table still has only `F32` and `I32`. |
| `unsupported model_type` | The constructor for that `model_type` is not in the registry. A missing `model_type` is treated as `gpt2`. |
| `tiny_starcoder_py` loads and the attention projection is 896 wide | That width is the multi-query fused projection, `n_embd + 2 * head_dim`. GPT-2's `3 * n_embd` split will read past the row. |
| Pythia logits are wrong after RoPE looks right | The block still runs the MLP on the attention output. Pythia adds attention and the MLP onto the same input. |
| A Gemma layer attends to the whole past | That layer is one of the global layers, or the window mask was not combined with the causal mask. |
| A flash-attention path is faster and the logits moved | The tiled softmax is not the same reduction as the current one. Compare against the GPT-2 path. |
