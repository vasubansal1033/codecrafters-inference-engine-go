This is a starting point for Go solutions to the ["Build Your Own Inference Engine" Challenge](https://codecrafters.io/challenges/inference-engine).

The program loads GPT-2's weights and writes the next token IDs. Along the way it covers attention and the transformer.

**Note:** On GitHub, go to [codecrafters.io](https://codecrafters.io) to try the challenge.

## Passing the first stage

The entry point is `app/main.go`. It calls `cli.Run`, and the commands are the ones listed under Running commands in the repository README. Submit with:

```sh
codecrafters submit
```

## Stage 2 and beyond

1. Install `go (1.26)` locally.
2. Run `./your_program.sh` with one of the commands under Running commands in the repository README.
3. Run `codecrafters submit`. Test output streams to the terminal.

## Testing locally

`your_program.sh` reads the files you pass on the command line:

```sh
mkdir -p /tmp/testing && cd /tmp/testing
/path/to/your/repo/your_program.sh inspect model.safetensors
```

A [shell alias](https://shapeshed.com/unix-alias/) shortens that:

```sh
alias myengine=/path/to/your/repo/your_program.sh

mkdir -p /tmp/testing && cd /tmp/testing
myengine inspect model.safetensors
```

## Running GPT-2

The last stage runs the engine on GPT-2's real weights. Download them into a testing folder outside this repository, so the 548 MB file stays out of git:

```sh
mkdir -p /tmp/testing && cd /tmp/testing
curl -L -o model.safetensors https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors
```

That stage's instructions list the `config.json`, `ids.json`, and `tokens.json` to run with.
