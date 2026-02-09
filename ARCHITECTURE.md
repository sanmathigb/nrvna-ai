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
| **Server** | `server.hpp/cpp` | Orchestrates everything |
| **Scanner** | `scanner.hpp/cpp` | Finds jobs in `input/ready/` |
| **Pool** | `pool.hpp/cpp` | Thread pool for workers |
| **Processor** | `processor.hpp/cpp` | Moves jobs through states, calls Runner |
| **Runner** | `runner.hpp/cpp` | Wraps llama.cpp for inference |
| **Logger** | `logger.hpp/cpp` | Thread-safe logging with levels |

## Workflow: Job Submission (Client Side)

```
1. Client calls Work::submit(prompt)
         |
         v
2. Create directory: input/writing/<job_id>/
         |
         v
3. Write prompt to: input/writing/<job_id>/prompt.txt
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
  |     +-- Every 1s: scan input/ready/ for new jobs
  |     +-- Submit found job IDs to Pool
  |
  +-- Worker Threads (Pool)
        +-- Worker-0, Worker-1, ... Worker-N
        +-- Each pulls jobs from queue
        +-- Each has its own Runner instance
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
   c. Runner::run(prompt) - llama.cpp inference
   d. On success: write result.txt, RENAME -> output/<job_id>
   e. On failure: write error.txt, RENAME -> failed/<job_id>
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

## Logging System

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
# Set via environment variable
export NRVNA_LOG_LEVEL=debug    # Options: error, warn, info, debug, trace

# llama.cpp has separate log control
export LLAMA_LOG_LEVEL=error    # Options: error, warn, info, debug
```

### Log Format

```
[YYYY-MM-DD HH:MM:SS.mmm] [LEVEL] [ThreadName] Message
```

Example:
```
[2026-01-12 10:30:45.123] [INFO ] [Scanner] Found 3 ready jobs
[2026-01-12 10:30:45.125] [INFO ] [Worker-0] Processing job: 1736700000_12345_0
[2026-01-12 10:30:46.789] [INFO ] [Worker-0] Job completed: 1736700000_12345_0
```

### Thread Names

- `Main` - Server main thread
- `Scanner` - Directory scanning thread
- `Worker-N` - Worker threads (N = 0, 1, 2, ...)

### Usage in Code

```cpp
#include "nrvna/logger.hpp"

LOG_ERROR("Something went wrong: " + error);
LOG_WARN("Warning message");
LOG_INFO("Informational message");
LOG_DEBUG("Debug details");
LOG_TRACE("Very detailed tracing");
```

### Thread Safety

- All logging is mutex-protected
- Safe to call from any thread
- Errors go to stderr, everything else to stdout

## CLI Tools

| Tool | Purpose | Example |
|------|---------|---------|
| `nrvnad` | Start daemon | `nrvnad model.gguf workspace` |
| `wrk` | Submit jobs | `wrk workspace "What is AI?"` |
| `flw` | Collect results | `flw workspace job-id` |

## Key Design Decisions

1. **Atomic renames** - Directory moves are atomic on POSIX filesystems, ensuring thread-safe state transitions without locks
2. **Directory = State** - Job's location IS its state (no database needed)
3. **Shared model, per-thread context** - llama.cpp model loaded once, each worker gets own inference context
4. **Filesystem-based** - Survives process crashes, easy to inspect/debug

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NRVNA_WORKERS` | 4 | Worker threads |
| `NRVNA_LOG_LEVEL` | info | Log verbosity |
| `NRVNA_GPU_LAYERS` | 99 (Mac) / 0 (other) | GPU layers for model |
| `NRVNA_PREDICT` | 2048 | Max tokens to generate |
| `NRVNA_MAX_CTX` | 8192 | Context window size |
| `NRVNA_TEMP` | 0.8 | Sampling temperature |
| `NRVNA_TOP_K` | 40 | Top-K sampling |
| `NRVNA_TOP_P` | 0.9 | Top-P sampling |
| `NRVNA_MIN_P` | 0.05 | Min-P sampling |
| `NRVNA_REPEAT_PENALTY` | 1.1 | Repetition penalty |
| `NRVNA_SEED` | 0 | Random seed |
| `NRVNA_MODELS_DIR` | ./models/ | Model search path |

## Thread Model

```
Main Thread
    |
    +-- creates Server
    |       |
    |       +-- creates Scanner (1 thread)
    |       +-- creates Pool (N worker threads)
    |       +-- creates Processor (shared, thread-safe)
    |
    +-- waits for shutdown signal

Scanner Thread
    +-- loops every 1 second
    +-- scans input/ready/
    +-- submits jobs to Pool queue

Worker Threads (N)
    +-- wait on condition variable
    +-- pop job from queue
    +-- call Processor::process(job_id, worker_id)
    +-- each has dedicated Runner instance
```
