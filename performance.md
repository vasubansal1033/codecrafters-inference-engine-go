## Performance

Memory numbers for the Run GPT-2 stage are below. CPU time has not been measured yet. The prompt used for the memory run is "The capital of France is".

## Memory

The Run GPT-2 stage stops any process that uses more than 2 GB. Two things decide whether the engine stays under that limit: how the weights are loaded, and how much work each generated token does.

#### Loading the weights

The weights file is 548 MB. The original loader read the whole file into memory and then decoded it:

| Loader | Memory while `Load` runs | Memory after `Load` returns |
| --- | --- | --- |
| `io.ReadAll` (until 23 Nov 2026) | The 548 MB byte slice and the decoded `[]float32` weights are both alive. | The floats remain. The byte slice can only be garbage collected after decoding finishes. |
| `ReadAt` (current) | One byte buffer, reused for every tensor. It grows to the size of the largest tensor. | The buffer is no longer referenced. Only the floats remain. |

The current `Load` works like this:

1. Read the 8-byte header length.
2. Read the JSON header.
3. Record the file offset where the tensor data begins.
4. For each tensor, use `ReadAt` to read its bytes into the shared buffer.
5. Decode those bytes into the tensor's own `[]float32`.

The largest tensor is `wte.weight`, at `50257 * 768 * 4` = 154,389,504 bytes, so that is the largest the buffer ever grows.

I measured the process on a Mac with `/usr/bin/time -l` on 23 Nov 2026, running `generate` on the stage prompt after the loader change and before the key-value cache existed:

| Measurement | Value |
| --- | --- |
| Maximum resident set size | 1,461,141,504 bytes (1.36 GiB) |
| What the tester also counts | The kernel's cached pages for the open 548 MB file |
| Measured size plus the file cache | About 1.9 GB, under the limit |
| The same total with the `io.ReadAll` byte slice still alive | Over 2 GB, so the stage prints `process exceeded memory limit (2 GB)` |

The peak stayed near 1.36 GiB even after the byte slice was removed, which told me the peak happens during generation rather than loading. Every call to `Head` (in `internal/models/gpt2/gpt2.go`) transposes `wte.weight`, allocating another 154,389,504 bytes per token. Go 1.26 (from `go.mod`) runs with `GOGC=100` by default, which lets the heap grow to roughly twice the live data before collecting. The float weights plus that transposed copy and the garbage collector's headroom add up to about 1.4 GB.

#### Work done for each token after the prompt

The GPT-2 cache stores keys and values for 12 layers and 12 heads, 64 floats per row. One token therefore adds `2 * 12 * 12 * 64 * 4` = 73,728 bytes to the cache, and the five prompt tokens take `5 * 73728` = 368,640 bytes. The cache itself is small. What it saves is the work of running every earlier row through the block again. Prefill and decode are in `kv-cache.md`.

| Second generated token | What gets embedded | What runs |
| --- | --- | --- |
| Before the cache (until 25 Nov 2026) | All six ids, at positions 0 through 5 | Layer norm, `c_attn`, the scores, and the MLP all run on six rows in every layer, even though only the last row's output is used. |
| With the cache (current) | Only id `262`, at position 5 | Each block processes a `[1, 768]` input. Each head computes a new `[1, 64]` query, key, and value, and the score is `[1, past + 1]`. |
| Both versions | Not applicable | `Head` still transposes `wte`, 154,389,504 bytes per token. The cache does not help here. |

## CPU

No CPU profile has been recorded. `/usr/bin/time -l`, the command used for the memory table, also prints user time, system time, and involuntary context switches. When you profile, keep the prompt and the requested token count next to those numbers, and split the run into three parts: loading the file, the first generated token, and each later token. The first token is the prefill. Later tokens are the decode path.

The places that will show up in a CPU profile of the current engine are the same places that dominate memory during generation: the per-layer `c_attn` and MLP products, the attention scores, and the `wte` transpose inside `Head`.

| Run | User time | System time | What was timed |
| --- | --- | --- | --- |
| Load | | | `Load` returning the float weights |
| First token | | | Prefill of the five prompt ids |
| Later token | | | One decode step, id embedded at its real position |
