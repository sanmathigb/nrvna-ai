# nrvna-ai

asynchronous inference primitive. while in nrvna-ai, models get wrk. you get flw.

## Build

```bash
git clone --recursive https://github.com/sanmathigb/nrvna-ai.git
cd nrvna-ai && mkdir build && cd build
cmake .. && make -j4
```

## Use

```bash
# 1. Start server
./nrvnad model.gguf ./workspace 4

# 2. Submit work
./wrk ./workspace "What is 2+2?"
# → 1736700000_12345_0

# 3. Get result
./flw ./workspace 1736700000_12345_0
# → The answer is 4.
```

## How it works

Jobs flow through directories:
```
input/ready/ → processing/ → output/
```

State = location. Atomic renames = thread safety. No database.

## Requirements

- C++17, CMake 3.14+
- macOS (Metal) or Linux
- GGUF model file

## License

MIT
