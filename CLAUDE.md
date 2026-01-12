# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nrvna-ai is an asynchronous inference primitive for LLM processing, implementing a directory-based job system with llama.cpp as the inference backend. The core architecture uses filesystem directories to represent job states with atomic transitions for thread-safe operation.

## Build System

This is a C++17 CMake project. Use these commands:

```bash
# Configure and build
mkdir -p build && cd build
cmake ..
make

# Run header compilation check
./nrvna_ai_header_check
```

## Architecture

### Core Components

**Work** (`include/nrvna/work.hpp`): Job submission API that creates and atomically publishes job directories. Handles prompt validation, staging in `input/writing/`, and atomic publication to `input/ready/`. Not thread-safe.

**Flow** (`include/nrvna/flow.hpp`, `src/flow.cpp`): Read-only facade over the workspace directory structure. Provides job status queries and result retrieval. Thread-safe for concurrent reads.

**Server** (`include/nrvna/server.hpp`): High-level orchestrator that manages the entire system lifecycle, owns Monitor instances, and handles workspace initialization.

**Monitor** (`include/nrvna/monitor.hpp`): Directory-based job processor with scanner and worker threads. Discovers jobs in `input/ready/`, moves them through processing pipeline to `output/` or `failed/`.

**Runner** (`include/nrvna/runner.hpp`): Synchronous inference wrapper around a shared llama.cpp model. One instance per worker thread, not thread-safe. Handles tokenization, prompt formatting, and inference execution.

### Directory-Based Job States

Jobs flow through directories representing lifecycle states:
- `input/writing/<id>` → STAGING (Work creates and writes `prompt.txt`)
- `input/ready/<id>` → QUEUED (atomic rename from writing/, contains `prompt.txt`)
- `processing/<id>` → RUNNING (Monitor moves from ready/, job being processed)
- `output/<id>` → DONE (contains `result.txt`, may retain `prompt.txt`)
- `failed/<id>` → FAILED (contains `error.txt`, may retain `prompt.txt`)

State transitions use atomic directory renames for thread safety. The critical publication step is the rename from `writing/` to `ready/`.

### Key Types

- `JobId`: String-based job identifier (in `nrvna/types.hpp`)
- `Status`: Enum for job lifecycle states (Queued, Running, Done, Failed, Missing)
- `Job`: In-memory representation with id, status, content, and timestamp
- `SubmitResult`: Result type for job submission with success/error status and messages
- `RunResult`: Result type for inference execution with output or error details

## Development Notes

- All public headers are in `include/nrvna/`
- Single implementation file: `src/flow.cpp`
- Header-only design for most components
- `compile_check.cpp` verifies all headers compile successfully
- Error handling: noexcept methods return sentinel values, others may throw
- Thread safety: Flow is read-only thread-safe, Monitor coordinates workers internally