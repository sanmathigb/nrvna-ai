# nrvna-ai

Asynchronous inference primitive.

> while in nrvna-ai, models get wrk. you get flw.

---

## Philosophy

Inference shouldn't block your workflow. Submit prompts, do other work, collect results when ready. Directories are state machines. Atomic renames are transactions. No database, no complexity.

## This is for

- Batch processing prompts while you work
- Local LLM workflows that need to scale
- Developers who value simplicity over features

## This is not for

- Interactive chat (use llama.cpp directly)
- Cloud inference (use an API)
- Production at scale (this is an MVP)

---

## Quickstart

```bash
# Build
git clone --recursive https://github.com/sanmathigb/nrvna-ai.git
cd nrvna-ai && mkdir build && cd build && cmake .. && make -j4

# Run
./nrvnad model.gguf ./workspace 4    # start server
./wrk ./workspace "your prompt"       # submit work
./flw ./workspace <job_id>            # get result
```

## How it works

```
input/ready/ → processing/ → output/
```

Jobs are directories. State is location. No polling, no callbacks.

## Requirements

- C++17, CMake 3.14+
- macOS (Metal) or Linux
- GGUF model

## License

MIT
