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

## Quick Start

```bash
# Build
git clone --recursive https://github.com/sanmathigb/nrvna-ai.git
cd nrvna-ai && cmake -S . -B build && cmake --build build -j4

# Start daemon (2 workers)
./build/nrvnad ./models/your-model.gguf ./workspace 2

# Submit job (returns immediately)
JOB=$(./build/wrk ./workspace "What is 2+2?")

# Collect result
./build/flw ./workspace $JOB
```

## Why

Every LLM API is synchronous: call, wait, return. nrvna-ai provides true async:

- **Fire and forget** — submit jobs, come back later
- **Batch processing** — queue hundreds of jobs, workers process in parallel
- **Multi-model** — different models for different workspaces
- **Vision support** — images via `--mmproj` and `--image`
- **Composable** — build agents and pipelines with shell scripts
- **No infrastructure** — filesystem is the queue (no Redis, no Kafka)

## The Primitives

| Tool | Purpose | Example |
|------|---------|---------|
| `nrvnad` | Daemon: model + workspace + workers | `nrvnad model.gguf ./ws 4` |
| `wrk` | Submit: prompt → job ID | `wrk ./ws "prompt"` |
| `flw` | Collect: job ID → result | `flw ./ws <job-id>` |

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
| macOS (Apple Silicon) | Metal | ✅ Full GPU acceleration |
| macOS (Intel) | CPU | ✅ CPU only |
| Linux | CPU / CUDA | ✅ CPU, CUDA if available |

## Multi-Model / Multi-Workspace

```bash
# Different models for different tasks
nrvnad qwen-vl.gguf    ./ws-vision 2 --mmproj qwen-vl-mmproj.gguf
nrvnad codellama.gguf  ./ws-code   4
nrvnad phi-3-mini.gguf ./ws-fast   2

# Submit to the right workspace
wrk ./ws-vision "Describe this" --image photo.jpg
wrk ./ws-code   "Refactor: $(cat main.py)"
wrk ./ws-fast   "Classify: bug or feature?"
```

## Batch Processing

```bash
# Submit 100 images for captioning
for img in photos/*.jpg; do
  wrk ./workspace "Caption this image" --image "$img"
done

# Results accumulate in output/
ls ./workspace/output/
```

## Requirements

- C++17 compiler
- CMake 3.16+
- macOS or Linux
- GGUF model ([HuggingFace](https://huggingface.co/models?search=gguf))

## Documentation

| Doc | Description |
|-----|-------------|
| [QUICKSTART.md](QUICKSTART.md) | Get running in 5 minutes |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Internals: components, threading, state machine |
| [ADVANCED.md](ADVANCED.md) | Patterns: batch, fan-out, loops, memory, routing |

## License

MIT
