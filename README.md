This repository is a Go program for the CodeCrafters "Build Your Own Inference Engine" challenge. It loads GPT-2 weights from a safetensors file, embeds token ids, runs multi-head attention for one layer, and runs the attention half of one transformer block. The longest section below is how `add` and `multiply` line up a matrix with a vector. The shapes and formulas were checked against `app/tensor.go`, `app/attention.go`, `app/main.go`, and `go.mod` (`go 1.26.0`) on 22 November 2026.

The walkthrough with the small tensors, from the weight file through the block, is in `concepts.md`.

## What is implemented

`inspect` lists every tensor name with its dtype, shape, and first few values. `embed` looks up one row of `wte.weight` per token id and one row of `wpe.weight` per position, then adds those two rows element by element. That addition lives in `mergeTokenAndPosEmbeddings` and only accepts two slices of equal length. `attn` runs one attention layer on a JSON matrix of rows. `block` layer-normalizes those rows, applies `ln_1`, runs attention, and adds the attention output back onto the original rows.

A tensor is a flat `[]float32` plus a shape, stored row by row. Shape `[n]` is a single list of `n` numbers. Shape `[m, n]` is `m` rows of `n` numbers. `getFlatArrayIdx` reads a shape `[n]` as one row of `n` columns, and it rejects a negative index or an index past the row or column count.

## Broadcasting

> Note. In GPT-2 the internal weights (`c_attn`, `c_proj`, and the MLP matrices) are matrix products. Layer-norm scales (`ln_1.weight`, `ln_2.weight`, `ln_f.weight`) are vectors, one scale per column, multiplied onto every token. Both paths live in `multiply`. The pictures are in `concepts.md`, under What multiply does.

GPT-2 stores several parameters as one list whose length is the embedding size: `h.{layer}.attn.c_attn.bias`, `h.{layer}.attn.c_proj.bias`, `h.{layer}.ln_1.weight`, and `h.{layer}.ln_1.bias`. A forward pass has one row per token, so the activation is `[n_pos, n_embd]`, or `[n_pos, 3*n_embd]` after the fused query, key, and value projection. The vector has to meet every row. That stretch is the broadcast.

The same `multiply` function also has to compute full matrix products. `c_attn.weight` is `[n_embd, 3*n_embd]`, and the forward step is the activation times that matrix. A one-token prompt is `[1, n_embd]`. That single row is a real product with the weight, and it is also shaped like a row that broadcasting would copy. `multiply` decides which of those it is from the shapes as they were stored, before any vector is viewed as a row.

#### The shared row-repeat steps

`add` and the second half of `multiply` follow the same steps. `add` uses `+=`. The broadcast half of `multiply` uses `*=`. Both methods take the receiver by value, so the rewrites below happen on a copy.

1. Record whether both arguments started with a one-dimensional shape.
2. Save the left tensor's dtype. A later swap must leave the result reporting that dtype.
3. Read a shape `[n]` as one row, `[1, n]`.
4. Both sides must now have two dimensions, both rows must have the same width, and the row counts must match or one side must have exactly one row. Anything else returns `Addition error` or `Multiplication error` with the two shapes.
5. If the left side is the single row and the right side has more rows, swap the arguments. After this step the values that get copied are always on the right.
6. If the right side has one row and the left side has several, allocate a new buffer and copy that row once for each left-hand row.
7. Clone the left-hand values into the result, then add or multiply the right-hand value at the same position.
8. When both arguments started as shape `[n]`, set the result shape back to `[n]`. The extra dimension existed so the row logic could treat every input the same way.

The clone in step 7 is required because a value receiver copies the struct, including the slice header, and that header still points at the caller's array. Writing into `Values` without cloning would change the tensor the caller passed in. Replacing `Shape` with a newly allocated slice stays on the copy, which is why step 3 leaves the caller unchanged.

`applyCausalMask` is the method that writes through that shared array on purpose. It sets `score[i][j]` to negative infinity when `j > i`, and the diagonal stays. The caller's score tensor sees those writes.

#### A worked multiply of a vector

Left operand, shape `[2, 3]`:

```text
1 2 3
4 5 6
```

Right operand, shape `[3]`: `10, 20, 30`.

The right side has one dimension, so the matrix-product check does not apply. That check requires two dimensions on both sides. The vector is viewed as shape `[1, 3]`. The widths are both 3, and the right side has one row, so that row is copied twice:

```text
10 20 30
10 20 30
```

Position-wise multiplication produces:

```text
10  40  90
40 100 180
```

The result shape is `[2, 3]`. The caller's tensor is still `[2, 3]` holding `1, 2, 3, 4, 5, 6`.

The same alignment with addition produces `11, 22, 33` on the first row and `14, 25, 36` on the second.

Two length-3 vectors, `1, 2, 3` and `4, 5, 6`, take this path as well. The arithmetic gives `4, 10, 18`, and step 8 folds the shape back to `[3]`.

#### When multiply is a matrix product

The product path runs only when both arguments already have two dimensions and the left column count equals the right row count. The value at result row `i`, column `j` is the sum of `left[i][k] * right[k][j]` over that shared dimension. The result shape is `[left rows, right columns]`.

A one-token case:

```text
left  shape [1, 2]  values 1, 2
right shape [2, 2]  values 3, 4, 5, 6
```

The right-hand matrix is:

```text
3 4
5 6
```

The inner sizes match, so this is a product even though the left side has a single row. The result is `1*3 + 2*5 = 13` and `1*4 + 2*6 = 16`, with shape `[1, 2]`.

This pair also satisfies the row-repeat rule: the rows have the same width, and one side has a single row. The product check runs first, while both tensors are still two-dimensional, so a one-token prompt times `c_proj.weight` (shape `[n_embd, n_embd]`) stays a matrix product.

#### Where each call lands in the block

`calculateBlock` normalizes each input row, multiplies by `ln_1.weight`, adds `ln_1.bias`, runs attention on those rows, and adds the attention output onto the row it started from.

- `ln_1.weight` has shape `[n_embd]`. `multiply` broadcasts it across the normalized rows.
- `ln_1.bias`, `c_attn.bias`, and `c_proj.bias` are vectors. `add` broadcasts each one across rows.
- `input * c_attn.weight`, `attended * c_proj.weight`, `Q * K^T`, and the softmax scores times `V` are two-dimensional with matching inner sizes. `multiply` computes a matrix product for each of them.

`layerNorm` works per row. It subtracts the row mean and divides by the square root of (the mean of squared deviations plus `1e-5`). The mean of those squared deviations divides by the row length. The comment on that line calls the variance unbiased; the divisor in the code is the row length. Gamma and beta are applied afterwards, by the broadcast multiply and the broadcast add, so they sit outside `layerNorm`.

Attention takes the position count from `inputTensor.Shape[0]`. `n_positions` in the config is the context length the model was built with, and a short prompt has fewer rows than that.

The feed-forward tensors for a block (`ln_2`, `mlp.c_fc`, `mlp.c_proj`) are in the weight file. `calculateBlock` does not read them.

Multi-head attention splits the fused `[n_pos, 3*n_embd]` row into Q, K, and V, then takes a contiguous column block of width `n_embd/n_heads` for each head. The score is `Q K^T / sqrt(head_dim)`. The mask, the row softmax, the multiply by V, and the write into the concatenated result stay in `calculateAttention`, in that order. `sliceHead` copies one head's columns into a new `[n_pos, head_dim]` tensor.

## Layout

- `app/main.go` switches on `inspect`, `embed`, `attn`, and `block`.
- `app/safetensors.go` reads the weight file.
- `app/embed.go` looks up token and position rows.
- `app/tensor.go` holds the tensor operations, including both broadcast paths.
- `app/attention.go` holds the config loader, multi-head attention, the causal mask, and the block.

`your_program.sh` builds with `go build` on `app/*.go`. Each file is `package main`.

## Running

```sh
./your_program.sh inspect model.safetensors
./your_program.sh embed model.safetensors ids.json
./your_program.sh attn model.safetensors config.json 0 input.json
./your_program.sh block model.safetensors config.json 0 input.json
codecrafters submit
```

The full GPT-2 weight file is about 548 MB. Download it outside this repository so it does not end up in a commit:

```sh
mkdir -p /tmp/testing && cd /tmp/testing
curl -L -o model.safetensors https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors
```

## When a shape goes wrong

| What you see | What is causing it |
| --- | --- |
| `Multiplication error: invalid shapes [2 3] x [2 2]` | The inner sizes are 3 and 2, so there is no matrix product. The broadcast path then requires equal row widths, and 3 does not equal 2. |
| `Addition error: invalid shapes [2 3] + [4]` | After `[4]` is viewed as `[1, 4]`, the row widths are 3 and 4. `add` has no second interpretation of the shapes. |
| A one-token projection comes out shaped like the weight matrix, with each column scaled | `[1, n_embd]` times `[n_embd, n_embd]` took the row-repeat path. The product check has to see both shapes while they are still two-dimensional. |
| The tensor passed into `add` or into a broadcast `multiply` has different values afterwards | The result wrote into `Values` without cloning. The copied slice header still points at the caller's array. |
| Layer-norm of `[[1, 2]]` is about `0.00199598` in the first column | The denominator was `sqrt(variance) + eps`. `layerNorm` now divides by `sqrt(variance + 1e-5)`. |
| `invalid dims provided: [2 6], whereas n_pos: 4` | The position count was `n_positions` from the config. A two-row prompt has to use `inputTensor.Shape[0]`, which `calculateAttention` does. |
