# nrvna-ai Architecture

## Overview

nrvna-ai is an **asynchronous inference primitive** - a directory-based job queue for LLM inference using llama.cpp. Jobs are represented as filesystem directories that move through states via atomic renames.

## Directory Structure

```
WORKSPACE/
├── input/
│   ├── writing/      <- Jobs being created (staging area)
│   └── ready/        <- Jobs waiting to be processed
├── processing/       <- Jobs currently running inference
├── output/           <- Completed jobs with results
└── failed/           <- Failed jobs with error messages
```

## Components

| Component | File | Role |
|-----------|------|------|
| **Work** | `work.hpp/cpp` | Client API - submits jobs |
| **Flow** | `flow.hpp/cpp` | Client API - queries job status/results |
| **Server** | `server.hpp/cpp` | Orchestrates Scanner + Pool + Processor |
| **Scanner** | `scanner.hpp/cpp` | Finds jobs in `input/ready/` |
| **Pool** | `pool.hpp/cpp` | Thread pool for workers |
| **Processor** | `processor.hpp/cpp` | Routes jobs by type, manages Runners, moves jobs through states |
| **Runner** | `runner.hpp/cpp` | Wraps llama.cpp for text, vision, and embedding inference |
| **TtsRunner** | `runner_tts.hpp/cpp` | Text-to-speech inference with OuteTTS + vocoder |
| **Logger** | `logger.hpp/cpp` | Thread-safe logging to stderr |

## Workflow: Job Submission (Client Side)

```
1. Client calls Work::submit(prompt)
         |
         v
2. Create directory: input/writing/<job_id>/
         |
         v
3. Write prompt to: input/writing/<job_id>/prompt.txt
   (+ type.txt for embed/tts, images/ for vision)
         |
         v
4. ATOMIC RENAME: input/writing/<job_id> -> input/ready/<job_id>
         |
         v
5. Return job_id to client
```

## Workflow: Job Processing (Server Side)

```
SERVER (main thread)
  |
  +-- Scanner Thread (scanLoop)
  |     +-- Every 5s: scan input/ready/ for new jobs
  |     +-- Submit found job IDs to Pool
  |
  +-- Worker Threads (Pool)
        +-- Worker-0, Worker-1, ... Worker-N
        +-- Each pulls jobs from queue
        +-- Each has its own Runner + TtsRunner instance
```

### Per-Job Processing

```
1. Scanner finds job in input/ready/<job_id>
         |
         v
2. Pool assigns to worker thread
         |
         v
3. Processor::process() called:
   a. ATOMIC RENAME: input/ready/<job_id> -> processing/<job_id>
   b. Read prompt from processing/<job_id>/prompt.txt
   c. Read type from processing/<job_id>/type.txt (default: text)
   d. Route by type:
      - text/vision → Runner::run()    → result.txt
      - embed       → Runner::embed()  → embedding.json
      - tts         → TtsRunner::run() → audio.wav
   e. On success: write output file, RENAME -> output/<job_id>
   f. On failure: write error.txt, RENAME -> failed/<job_id>
```

## Workflow: Result Retrieval (Client Side)

```
Client calls Flow::status(job_id)
         |
         v
Check directories in order:
  - output/<job_id>      -> Status::Done
  - failed/<job_id>      -> Status::Failed
  - processing/<job_id>  -> Status::Running
  - input/ready/<job_id> -> Status::Queued
  - none found           -> Status::Missing

Client calls Flow::get(job_id)
         |
         v
Read output/<job_id>/result.txt (or error.txt if failed)
```

## Job States

| State | Directory | Description |
|-------|-----------|-------------|
| STAGING | `input/writing/<id>` | Being created, not yet visible |
| QUEUED | `input/ready/<id>` | Waiting for worker |
| RUNNING | `processing/<id>` | Inference in progress |
| DONE | `output/<id>` | Completed successfully |
| FAILED | `failed/<id>` | Error occurred |

## Inference Pipeline

### Text/Vision (Runner)

Based on llama.cpp `examples/simple/simple.cpp` and `tools/mtmd/mtmd-cli.cpp`.

- Shared `llama_model` across all workers (thread-safe)
- Per-worker `llama_context` created fresh for each job, freed after
- Per-worker `mtmd_context` for vision (NOT thread-safe)
- Vision encoding serialized via mutex (GGML shared compute graph state)
- Chat template applied via `llama_chat_apply_template` (falls back to raw prompt for base models)
- Sampler chain: penalties → top_k → top_p → min_p → temp → dist
- `stripThinkBlocks()` removes `<think>...</think>` from reasoning models

### TTS (TtsRunner)

Based on llama.cpp `tools/tts/tts.cpp`.

- Shared TTS model + vocoder model across workers
- OuteTTS v0.2/v0.3 auto-detected by vocabulary probing
- Audio code generation with top_k=4 sampler
- Code extraction via `<|N|>` token text parsing
- Vocoder encodes codes → embeddings → ISTFT spectral conversion → 24kHz PCM

### Embeddings (Runner::embed)

- Creates context with `embeddings=true`, mean pooling
- Returns float vector (dimension depends on model)

## Logging

All log output goes to **stderr**. Stdout is reserved for job status lines.

### Log Levels

| Level | Value | Description |
|-------|-------|-------------|
| ERROR | 0 | Errors only |
| WARN | 1 | Warnings and above |
| INFO | 2 | General info (default) |
| DEBUG | 3 | Detailed debugging |
| TRACE | 4 | Very verbose tracing |

### Configuration

```bash
export NRVNA_LOG_LEVEL=debug    # Options: error, warn, info, debug, trace
export LLAMA_LOG_LEVEL=error    # Controls llama.cpp verbosity (default: error)
```

### Log Format

```
[YYYY-MM-DD HH:MM:SS.mmm] [LEVEL] [ThreadName] Message
```

## CLI Tools

| Tool | Purpose | Example |
|------|---------|---------|
| `nrvnad` | Start daemon | `nrvnad model.gguf workspace` |
| `wrk` | Submit jobs | `wrk workspace "prompt"` |
| `flw` | Collect results | `flw workspace job-id` |

## Key Design Decisions

1. **Atomic renames** - Directory moves are atomic on POSIX filesystems, ensuring thread-safe state transitions without locks
2. **Directory = State** - Job's location IS its state (no database needed)
3. **Shared model, per-thread context** - llama.cpp model loaded once, each worker gets own inference context
4. **Filesystem-based** - Survives process crashes, easy to inspect/debug
5. **Stuck job recovery** - If `finalizeSuccess` fails, attempts `finalizeFailure` to prevent jobs stuck in `processing/`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NRVNA_WORKERS` | 4 | Worker threads |
| `NRVNA_LOG_LEVEL` | info | Log verbosity |
| `NRVNA_GPU_LAYERS` | 99 (Mac) / 0 (other) | GPU layers for model |
| `NRVNA_PREDICT` | 2048 | Max tokens to generate |
| `NRVNA_MAX_CTX` | 8192 | Context window size |
| `NRVNA_BATCH` | 2048 | Batch size |
| `NRVNA_TEMP` | 0.8 | Sampling temperature |
| `NRVNA_VISION_TEMP` | 0.3 | Vision sampling temperature |
| `NRVNA_TOP_K` | 40 | Top-K sampling |
| `NRVNA_TOP_P` | 0.9 | Top-P sampling |
| `NRVNA_MIN_P` | 0.05 | Min-P sampling |
| `NRVNA_REPEAT_PENALTY` | 1.1 | Repetition penalty |
| `NRVNA_REPEAT_LAST_N` | 64 | Repeat penalty window |
| `NRVNA_SEED` | 0 | Random seed |
| `NRVNA_MODELS_DIR` | ./models/ | Model search path |
| `NRVNA_MAX_IMAGE_SIZE` | 50MB | Max image file size |
| `NRVNA_QUIET` | (unset) | Suppress mtmd timing logs |
| `LLAMA_LOG_LEVEL` | error | llama.cpp log verbosity |

## Thread Model

```
Main Thread
    |
    +-- creates Server
    |       |
    |       +-- creates Scanner (1 thread)
    |       +-- creates Pool (N worker threads)
    |       +-- creates Processor (shared, thread-safe)
    |       |       +-- pre-initializes N Runners
    |       |       +-- pre-initializes N TtsRunners (if vocoder present)
    |       |
    |       +-- recoverOrphanedJobs (processing/ -> ready/ or failed/)
    |
    +-- waits for shutdown signal (SIGINT/SIGTERM)

Scanner Thread
    +-- loops every 5 seconds
    +-- scans input/ready/
    +-- submits jobs to Pool queue

Worker Threads (N)
    +-- wait on condition variable
    +-- pop job from queue
    +-- call Processor::process(job_id, worker_id)
    +-- each has dedicated Runner + TtsRunner instance
```
