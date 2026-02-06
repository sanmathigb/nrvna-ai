# Quick Start: Async AI in 5 Minutes

nrvna-ai is an async inference primitive. You submit jobs, they process in the background, you collect results. This lets you batch prompts, fan-out to parallel workers, or chain jobs in pipelines.

## 1. Build

```bash
git clone --recursive https://github.com/nrvna-ai/nrvna-ai.git
cd nrvna-ai
mkdir build && cd build && cmake .. && make -j4
cd ..
```

## 2. Get a Model

```bash
./bin/models list                    # See available models
./bin/models pull qwen2.5-coder-3b   # Download (~2.1GB)
```

Or download any GGUF model to `./models/`.

## 3. Start the Server

```bash
./build/nrvnad qwen ./workspace
```

The server auto-detects models in `./models/`. Leave this running.

## 4. Submit a Job (Returns Immediately!)

```bash
./build/wrk ./workspace "What is the capital of France?"
```

Output:
```
Job submitted: abc123
Run: flw ./workspace -w abc123
abc123
```

The job is now processing in the background.

## 5. Collect the Result

```bash
./build/flw ./workspace -w abc123
```

Output:
```
The capital of France is Paris.
```

## Why Two Commands?

Because you can submit many jobs and collect them later:

```bash
# Submit 3 jobs in parallel
./build/wrk ./workspace "Explain quantum computing"
./build/wrk ./workspace "Explain machine learning"
./build/wrk ./workspace "Explain neural networks"

# All processing simultaneously! Collect when ready.
./build/flw ./workspace   # Get latest result
```

## Convenience Wrapper

For quick tests, use the `ask` script (submits + waits):

```bash
./bin/ask "Hello, world!"
```

But remember: `wrk` + `flw` are the real primitives. They unlock batch processing, job chaining, and parallel inference.

## Next Steps

- `./build/wrk --help` - See all wrk options
- `./build/flw --help` - See all flw options
- `./build/nrvnad --help` - Server configuration
- `./build/swarm --help` - Map-reduce parallel agent
