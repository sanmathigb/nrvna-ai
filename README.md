# nrvna-ai

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Local inference as durable jobs. Three binaries. Filesystem is the queue.

## Quick Start

One prompt, one result — proof of life in 60 seconds:

```bash
# Build
git clone --recursive https://github.com/sanmathigb/nrvna-ai.git
cd nrvna-ai && cmake -S . -B build && cmake --build build -j4

# Start a daemon with any GGUF model
./build/nrvnad models/your-model.gguf /tmp/ws -w 1 &
while [ ! -f /tmp/ws/.nrvnad.pid ]; do sleep 1; done
JOB=$(./build/wrk /tmp/ws "Explain the CAP theorem in two sentences")
./build/flw /tmp/ws -w $JOB
```

## See It Work

Submit multiple jobs, inspect workspace progress, and collect results:

```bash
# Queue a few jobs
JOB1=$(./build/wrk /tmp/ws "Explain Raft in two sentences")
JOB2=$(./build/wrk /tmp/ws "Summarize the CAP theorem in one sentence")

# Inspect workspace status
./build/flw /tmp/ws
./build/flw /tmp/ws --json

# Retrieve results
./build/flw /tmp/ws -w $JOB1
./build/flw /tmp/ws -w $JOB2
```

The substrate is the point: durable jobs, inspectable state, and predictable retrieval through the same three binaries.

## Three Primitives

| Tool | What it does |
|------|-------------|
| `nrvnad` | Load a model, watch a workspace, process jobs |
| `wrk` | Submit a prompt, get back a job ID |
| `flw` | Retrieve a result by job ID |

That's the entire API. Everything else is composition.

```bash
# Start a daemon
./build/nrvnad models/Qwen2.5-7B-Instruct-Q4_K_M.gguf ./ws -w 1 &
while [ ! -f ./ws/.nrvnad.pid ]; do sleep 1; done

# Submit work
JOB=$(./build/wrk ./ws "Explain the CAP theorem in two sentences")

# Collect result
./build/flw ./ws -w $JOB
```

## Job Types

```bash
# Text (default)
wrk ./ws "Summarize this document"

# Vision — caption, describe, OCR (mmproj auto-detected)
wrk ./ws "What's in this image?" --image photo.jpg

# Embeddings — vectors for search/similarity
wrk ./ws "sentence to embed" --embed

# Text-to-speech — audio output (vocoder auto-detected)
wrk ./ws "Hello, world" --tts
```

## How It Works

Jobs are directories. State is location. Transitions are atomic renames.

```
workspace/
├── input/ready/    ← queued jobs
├── processing/     ← jobs being worked
├── output/         ← results (result.txt, embedding.json, or audio.wav)
└── failed/         ← errors (error.txt)
```

No database. No message broker. No runtime dependencies. Every job is fresh — bounded context, no session drift, predictable output.

## Workflows

Workflows sit above the core release surface. Common patterns are:

- multi-job text processing
- multimodal ingestion
- chunked TTS or transcription
- map-reduce over large documents

## Why

nrvna is compelling when the job is bigger than one prompt and smaller than a whole framework.

- **Not a chat app** — async jobs, not conversations
- **Not an agent framework** — primitives you build on
- **Not a model runtime** — that's llama.cpp underneath. nrvna adds jobs, workspaces, and composition.

## Platform

| Platform | Backend |
|----------|---------|
| macOS (Apple Silicon) | Metal GPU acceleration |
| macOS (Intel + discrete GPU) | Metal with local patch |
| Linux | CPU, CUDA if available |

## Requirements

- macOS or Linux
- CMake 3.16+ and a C++17 compiler
- GGUF models you provide yourself ([HuggingFace](https://huggingface.co/models?search=gguf))

## License

MIT
