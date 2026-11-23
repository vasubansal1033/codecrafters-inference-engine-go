package main

import (
	"os"

	"github.com/codecrafters-io/inference-engine-starter-go/internal/cli"
)

func main() {
	os.Exit(cli.Run(os.Args[1:], os.Stdout, os.Stderr))
}
