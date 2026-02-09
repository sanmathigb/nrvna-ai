# nrvna-ai

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Async inference primitives for local LLMs. Submit jobs, collect results, build pipelines.

```
          wrk                    nrvnad                    flw
           │                        │                        │
   "prompt" ──▶ input/ready/ ──▶ processing/ ──▶ output/ ──▶ result
           │                        │                        │
       (submit)              (workers churn)            (collect)
```

## Install

```bash
git clone --recursive https://github.com/sanmathigb/nrvna-ai.git
cd nrvna-ai
cmake -S . -B build && cmake --build build -j4
sudo cmake --install build
```

All dependencies are vendored — llama.cpp is built automatically as part of the build. No separate installation needed, and it won't conflict with any existing llama.cpp on your system.

## Quick Start

```bash
# Interactive mode — pick a model, assign a workspace, start
nrvnad

# Or start directly
nrvnad model.gguf workspace

# Submit job (returns immediately)
JOB=$(wrk workspace "What is 2+2?")

# Collect result
flw workspace $JOB
```

Place GGUF models in `./models/` or set `NRVNA_MODELS_DIR`.

## Why

Every LLM API is synchronous: call, wait, return. nrvna-ai provides true async:

- **Fire and forget** — submit jobs, come back later
- **Batch processing** — queue hundreds of jobs, workers process in parallel
- **Multi-model** — different models for different workspaces
- **Vision support** — mmproj auto-detected, images via `--image`
- **Composable** — build agents and pipelines with shell scripts
- **No infrastructure** — filesystem is the queue (no Redis, no Kafka)

## The Primitives

| Tool | Purpose | Example |
|------|---------|---------|
| `nrvnad` | Start daemon | `nrvnad model.gguf workspace` |
| `wrk` | Submit prompt | `wrk workspace "prompt"` |
| `flw` | Collect result | `flw workspace job-id` |

Jobs are directories. State is location. Transitions are atomic renames.

```
workspace/
├── input/ready/    ← queued jobs
├── processing/     ← jobs being worked
├── output/         ← completed jobs (result.txt)
└── failed/         ← failed jobs (error.txt)
```

## Platform Support

| Platform | Backend | Status |
|----------|---------|--------|
| macOS (Apple Silicon) | Metal | Full GPU acceleration |
| macOS (Intel) | CPU | CPU only |
| Linux | CPU / CUDA | CPU, CUDA if available |

## Multi-Model / Multi-Workspace

```bash
# Different models for different tasks
nrvnad qwen-vl.gguf   ws-vision    # mmproj auto-detected
nrvnad codellama.gguf  ws-code
nrvnad phi-3.gguf      ws-fast

# Submit to the right workspace
wrk ws-vision "Describe this" --image photo.jpg
wrk ws-code   "Refactor: $(cat main.py)"
wrk ws-fast   "Classify: bug or feature?"
```

## Batch Processing

```bash
# Submit 100 images for captioning
for img in photos/*.jpg; do
  wrk workspace "Caption this image" --image "$img"
done

# Results accumulate in output/
ls workspace/output/
```

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
