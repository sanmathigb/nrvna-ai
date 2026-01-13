# nrvna-ai

models get wrk. you get flw.

Async filesystem-based job queue for LLM inference. Small building blocks, not a framework.

---

## Why

Async primitives for model inference don't exist. We have entire ecosystems built on synchronous APIs. nrvna-ai is an attempt to build the async equivalents — small, simple, Unix tool-like. Ready to use, or build on top of.

---

## Usage

### Single Model

```bash
./nrvnad model.gguf ./workspace 4
./wrk ./workspace "What is 2+2?"
./flw ./workspace <job-id>
```

### Multiple Models, Multiple Intents

```bash
# Terminal 1: Writing assistant
./nrvnad mistral.gguf ./writing_workspace 4

# Terminal 2: Learning companion
./nrvnad phi3.gguf ./learning_workspace 2

# Submit to whichever intent you need
./wrk ./writing_workspace "Write a blog post"
./wrk ./learning_workspace "Explain this code"
```

---

## Requirements

- C++17, CMake 3.14+
- macOS (Metal) or Linux
- GGUF model file

---

## Build

```bash
git clone --recursive https://github.com/sanmathigb/nrvna-ai.git
cd nrvna-ai && mkdir build && cd build
cmake .. && make -j4
```

---

## License

MIT
