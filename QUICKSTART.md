# Quick Start

nrvna-ai is an async inference primitive. You submit jobs, they process in the background, you collect results.

## 1. Build

```bash
git clone --recursive https://github.com/sanmathigb/nrvna-ai.git
cd nrvna-ai
cmake -S . -B build && cmake --build build -j4
```

## 2. Get a Model

Download any GGUF model from [HuggingFace](https://huggingface.co/models?search=gguf) and place it in `./models/`:

```bash
mkdir -p models
# Example: download a small model for testing
# https://huggingface.co/TheBloke/phi-2-GGUF
```

Or set `NRVNA_MODELS_DIR` to point to your existing models directory.

## 3. Start the Daemon

Interactive mode (recommended for first run):

```bash
./build/nrvnad
```

This shows a dashboard with your models and workspaces. Pick a number to get started.

Or start directly:

```bash
./build/nrvnad ./models/your-model.gguf workspace
```

Leave this running.

## 4. Submit a Job

```bash
./build/wrk workspace "What is the capital of France?"
```

Output:
```
abc123
```

The job ID is printed. The job is now processing in the background.

## 5. Collect the Result

```bash
./build/flw workspace abc123
```

Output:
```
The capital of France is Paris.
```

Use `-w` to wait for a job that's still processing:

```bash
./build/flw workspace -w abc123
```

## Piping

Submit and wait in one line:

```bash
./build/wrk workspace "Hello" | xargs ./build/flw workspace -w
```

## Batch Processing

Submit many jobs, collect results later:

```bash
# Submit 3 jobs in parallel
./build/wrk workspace "Explain quantum computing"
./build/wrk workspace "Explain machine learning"
./build/wrk workspace "Explain neural networks"

# All processing simultaneously. Collect when ready:
./build/flw workspace    # gets latest result
```

## Next Steps

- [ADVANCED.md](ADVANCED.md) — batch, fan-out, loops, multi-model routing
- [ARCHITECTURE.md](ARCHITECTURE.md) — internals, threading, state machine
- `./build/wrk --help` — all wrk options
- `./build/flw --help` — all flw options
- `./build/nrvnad --help` — server configuration
