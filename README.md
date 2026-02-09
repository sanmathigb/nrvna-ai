# nrvna-ai

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Build agents, pipelines, and batch workflows with local LLMs. Three commands. Filesystem is the queue. Shell is the orchestrator.

```
          wrk                    nrvnad                    flw
           │                        │                        │
   "prompt" ──▶ input/ready/ ──▶ processing/ ──▶ output/ ──▶ result
           │                        │                        │
       (submit)              (workers churn)            (collect)
```

No frameworks. No Python dependencies. Just three binaries that compose like Unix pipes — small enough to understand in minutes, powerful enough to build agent systems with shell scripts.

## Install

```bash
git clone --recursive https://github.com/sanmathigb/nrvna-ai.git
cd nrvna-ai
cmake -S . -B build && cmake --build build -j4
sudo cmake --install build
```

All dependencies are vendored. llama.cpp is built automatically — no separate install, no conflicts.

## Quick Start

```bash
# Interactive — pick a model, assign a workspace, start
nrvnad

# Or start directly
nrvnad model.gguf workspace

# Submit work (returns immediately)
JOB=$(wrk workspace "What is 2+2?")

# Collect result
flw workspace $JOB
```

Place GGUF models in `./models/` or set `NRVNA_MODELS_DIR`.

## Three Primitives

| Tool | What it does |
|------|-------------|
| `nrvnad` | Load a model, watch a workspace, process jobs |
| `wrk` | Submit a prompt, get back a job ID |
| `flw` | Retrieve a result by job ID |

That's the entire API. Everything else is composition.

## What Emerges

The primitives are small. What you build with them isn't.

**Agent loop** — feed results back as prompts:
```bash
for i in {1..5}; do
  result=$(wrk workspace "Continue: $memory" | xargs flw workspace -w)
  memory="$memory\n$result"
done
```

**Fan-out / fan-in** — parallelize, then synthesize:
```bash
a=$(wrk workspace "Research: databases")
b=$(wrk workspace "Research: caching")
c=$(wrk workspace "Research: queuing")
wrk workspace "Synthesize: $(flw workspace $a) $(flw workspace $b) $(flw workspace $c)"
```

**Multi-model routing** — different models for different tasks:
```bash
nrvnad qwen-vl.gguf   ws-vision    # mmproj auto-detected
nrvnad codellama.gguf  ws-code
nrvnad phi-3.gguf      ws-fast

wrk ws-vision "Describe this" --image photo.jpg
wrk ws-code   "Refactor: $(cat main.py)"
wrk ws-fast   "Classify: bug or feature?"
```

**Batch processing** — queue hundreds, workers churn through them:
```bash
for img in photos/*.jpg; do
  wrk workspace "Caption this" --image "$img"
done
```

**Memory** — job history is context history:
```bash
flw workspace $job1 >> memory.txt
flw workspace $job2 >> memory.txt
wrk workspace "Given this context: $(cat memory.txt) — what next?"
```

## How It Works

Jobs are directories. State is location. Transitions are atomic renames.

```
workspace/
├── input/ready/    ← queued jobs
├── processing/     ← jobs being worked
├── output/         ← completed results
└── failed/         ← errors
```

No database. No message broker. No runtime dependencies. The filesystem is the coordination layer — you can inspect it with `ls`, debug it with `cat`, monitor it with `watch`.

## Platform Support

| Platform | Backend |
|----------|---------|
| macOS (Apple Silicon) | Metal GPU acceleration |
| macOS (Intel) | CPU |
| Linux | CPU, CUDA if available |

## Requirements

- macOS or Linux
- CMake 3.16+ and a C++17 compiler
- A GGUF model ([HuggingFace](https://huggingface.co/models?search=gguf))

## Documentation

| Doc | Description |
|-----|-------------|
| [QUICKSTART.md](QUICKSTART.md) | Get running in 5 minutes |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Internals: components, threading, state machine |
| [ADVANCED.md](ADVANCED.md) | Patterns: batch, fan-out, loops, memory, routing |

## License

MIT
