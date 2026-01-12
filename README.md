# nrvna-ai

**Async batch inference for local LLMs. You queue work, nrvna processes it, you check results later.**

> **"nrvna gets wrk. you get flw."**

---

## What is nrvna?

nrvna is a **robust, Unix-style runtime** for local LLM workflows. It is an exploration of asynchronous, file-based AI.

It's designed for users who've already found their perfect settings with tools like llama.cpp and want to scale up with async batch processing.

**Not a replacement for chat.** If you're still testing prompts and tweaking temperatures, use llama.cpp directly. Once you know what works, nrvna scales it.

### The Model

```bash
# You submit work
wrk ~/project "Write User class with validation"
wrk ~/project "Add comprehensive tests"  
wrk ~/project "Review for SQL injection"

# Go do other work (coding, meetings, coffee...)

# Check results later
flw ~/project
# → All 3 jobs done ✅
```

**Key insight:** The daemon processes work in the background. You maintain flow state. No waiting for responses.

---

## Quick Start

### 1. Build

```bash
cd ~/ws/nrvna-ai
mkdir -p build && cd build
cmake ..
make -j8
```

**Requirements:**
- C++17 compiler (clang/gcc)
- CMake 3.20+
- Metal GPU (macOS) or CUDA (Linux) for performance
- 4GB+ VRAM recommended

### 2. Get a Model

**Download from Hugging Face:**

```bash
# Recommended: Qwen 2.5 Coder (fast + good quality)
huggingface-cli download Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF \
  qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
  --local-dir ~/models

# Or: TinyLlama (very fast, basic quality)
huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --local-dir ~/models

# Or: Phi-3 Mini (analytical, good for code review)
huggingface-cli download microsoft/Phi-3-mini-4k-instruct-gguf \
  Phi-3-mini-4k-instruct-q4.gguf \
  --local-dir ~/models
```

**What models work?**
- ✅ **ANY GGUF model** with chat templates
- ✅ Qwen, Llama, Mistral, Phi, Gemma, DeepSeek, etc.
- ✅ Instruct/Chat models preferred (better at following instructions)
- ⚠️ Base models work but need careful prompting

**How to know if a model will work:**

1. **Check for GGUF format** - Must be `.gguf` file
2. **Check for "Instruct" or "Chat"** in name - Better instruction following
3. **Check size vs your VRAM:**
   - 1-2GB models: 4GB VRAM minimum
   - 3-4GB models: 6GB VRAM recommended  
   - 7B+ models: 8GB+ VRAM needed

```bash
# Verify model has chat template
strings ~/models/your-model.gguf | grep "tokenizer.chat_template"
# → Should show Jinja2 template (means it will work!)
```

### 3. Start the Daemon

```bash
# Basic (uses good defaults)
./nrvnad ~/models/qwen-coder-1.5b.gguf ~/workspace &

# With custom settings (optional)
export NRVNA_TEMP=0.7        # Creativity (0.5-0.9)
export NRVNA_TOP_K=40        # Token pool size
export NRVNA_TOP_P=0.9       # Nucleus sampling  
export NRVNA_PREDICT=512     # Max output tokens
export NRVNA_CTX=2048        # Context window (0=auto)
export NRVNA_GPU_LAYERS=99   # GPU offload (99=all)

./nrvnad ~/models/qwen-coder-1.5b.gguf ~/workspace &
```

**Verify it's running:**
```bash
ps aux | grep nrvnad
# Should show process
```

### 4. Submit Work

```bash
# Submit jobs
./wrk ~/workspace "Write a binary search function in C++ with comments"
./wrk ~/workspace "Explain how the function handles edge cases"
./wrk ~/workspace "Add unit tests for the implementation"

# Returns immediately with job IDs
# Jobs process in background
```

### 5. Check Results

```bash
# See completed jobs
./flw ~/workspace

# Results are in:
# ~/workspace/input/   - Original prompts
# ~/workspace/output/  - Generated results
# ~/workspace/failed/  - Any failures
```

---

## Configuration Guide

### Environment Variables

**Quality Control (defaults match llama.cpp):**
```bash
export NRVNA_TEMP=0.8             # Temperature (default: 0.8, same as llama.cpp)
export NRVNA_TOP_K=40             # Top-K sampling (default: 40)
export NRVNA_TOP_P=0.95           # Top-P nucleus (default: 0.95, same as llama.cpp)
export NRVNA_MIN_P=0.05           # Min-P threshold (default: 0.05, same as llama.cpp)
export NRVNA_REPEAT_PENALTY=1.0   # Repetition penalty (default: 1.0=OFF, same as llama.cpp)
export NRVNA_PREDICT=512          # Max tokens to generate
```

**Performance:**
```bash
export NRVNA_CTX=2048      # Context window (0=auto, 2048=safe)
export NRVNA_GPU_LAYERS=99 # GPU offload (0=CPU only, 99=all layers)
export NRVNA_THREADS=8     # CPU threads for inference
```

**Debugging:**
```bash
export LLAMA_LOG_LEVEL=info  # error|warn|info|debug
```

### Recommended Settings by Task

**Code Generation (use defaults, they work!):**
```bash
# Defaults are optimized for code (tested with llama.cpp)
# Just use: ./nrvnad model.gguf workspace
# 
# Or customize if needed:
NRVNA_TEMP=0.8              # Default (llama.cpp uses 0.8 for code)
NRVNA_TOP_K=40              # Default
NRVNA_TOP_P=0.95            # Default
NRVNA_MIN_P=0.05            # Default
NRVNA_PREDICT=512
```

**Documentation Writing:**
```bash
NRVNA_TEMP=0.8      # More creative
NRVNA_TOP_K=50
NRVNA_TOP_P=0.95
NRVNA_PREDICT=768   # Longer outputs
```

**Code Review (strict/analytical):**
```bash
NRVNA_TEMP=0.6      # More focused
NRVNA_TOP_K=30
NRVNA_TOP_P=0.85
NRVNA_PREDICT=512
```

**Deterministic (same input → same output):**
```bash
NRVNA_TEMP=0.0      # Greedy sampling
NRVNA_SEED=42       # Fixed seed
```

---

## Model Selection Guide

### Choosing the Right Model

**By Hardware (VRAM constraints):**

| Your VRAM | Recommended Model | Size | Speed | Quality |
|-----------|-------------------|------|-------|---------|
| 4GB | TinyLlama 1.1B | 640MB | ⚡⚡⚡ Fast | ⭐⭐ Basic |
| 4GB | Qwen 1.5B | 1GB | ⚡⚡ Fast | ⭐⭐⭐ Good |
| 6GB+ | Qwen 3B | 2GB | ⚡ Medium | ⭐⭐⭐⭐ Great |
| 8GB+ | Phi-3 Mini | 2.2GB | ⚡ Medium | ⭐⭐⭐⭐ Analytical |
| 8GB+ | Mistral 7B | 4GB | 🐢 Slow | ⭐⭐⭐⭐⭐ Excellent |

**By Task:**

| Task | Best Models | Why |
|------|-------------|-----|
| **Code generation** | **Qwen 2.5 Coder, CodeLlama** | **Specialized for code** |
| Documentation | Mistral, Llama 3 | Good prose |
| Code review | Phi-3, Qwen | Analytical |
| Quick tasks | Qwen 1.5B (not TinyLlama) | Fast + acceptable quality |
| Production | Qwen 1.5B/3B Coder | Balance of speed and quality |

### What Makes a Model Compatible?

**✅ Will work:**
- GGUF format (`.gguf` file extension)
- Has embedded chat template (check with `strings model.gguf | grep chat_template`)
- Instruct or Chat variant
- Q4_K_M or Q5_K_M quantization (good quality/size balance)

**⚠️ Might work (requires tuning):**
- Base models (not instruct-tuned) - need careful prompting
- Very old models - may lack proper templates
- Experimental quantizations (Q2, Q8) - quality/size tradeoffs

**❌ Won't work:**
- Non-GGUF formats (PyTorch `.bin`, SafeTensors `.safetensors`)
- Models without tokenizer metadata
- Models larger than your VRAM capacity

### Finding Models on Hugging Face

**Search tips:**
1. Go to huggingface.co/models
2. Filter by "GGUF" in search
3. Look for "Instruct", "Chat", or "Coder" in name
4. Check model card for:
   - Context length (2048+ recommended)
   - Recommended quantization (Q4_K_M is standard)
   - Chat template format (should be documented)

**Popular model families (all GGUF-compatible):**
- **Qwen/Qwen2.5-Coder** - Best for code (1.5B, 3B, 7B)
- **microsoft/Phi-3** - Analytical, good reasoning (mini 4K)
- **mistralai/Mistral** - Strong general model (7B, 8x7B)
- **meta-llama/Llama-3** - High quality, good prose (8B, 70B)
- **TinyLlama** - Fast baseline (1.1B)
- **deepseek-ai/DeepSeek-Coder** - Code specialist (1.3B, 6.7B, 33B)

**Download command pattern:**
```bash
huggingface-cli download <org>/<repo> <filename> --local-dir ~/models
```

---

## How It Works

### Architecture

```
┌─────────────┐
│   nrvnad    │  ← Daemon (loads model, watches workspace)
│  (daemon)   │
└──────┬──────┘
       │
       ├─ Monitors: ~/workspace/input/
       ├─ Processes: jobs with LLM
       └─ Outputs:  ~/workspace/output/
       
┌──────────────┐
│  wrk (CLI)   │  ← Submit work (creates job in input/)
└──────────────┘

┌──────────────┐
│  flw (CLI)   │  ← Check results (lists output/)
└──────────────┘
```

**Job lifecycle:**
1. `wrk` creates directory in `workspace/input/`
2. Writes `prompt.txt` with your request
3. `nrvnad` picks up job
4. Runs inference with loaded model
5. Moves result to `workspace/output/`
6. `flw` reads from `output/` and displays

### Why Async?

**Traditional (blocking):**
```bash
$ llama-cli -m model.gguf -p "Write quicksort"
# You wait 45 seconds staring at terminal... 😴
```

**nrvna (async):**
```bash
$ wrk . "Write quicksort"
Job submitted: job_abc123
$ wrk . "Write binary search"  
Job submitted: job_def456
$ wrk . "Write merge sort"
Job submitted: job_ghi789

# Go write code, attend meetings, live life...

$ flw .
✅ job_abc123 - quicksort implementation (done)
✅ job_def456 - binary search (done)  
✅ job_ghi789 - merge sort (done)
```

**You save 135 seconds of waiting. More importantly: you never lose flow state.**

---

## Technical Details

### What's New (v0.2.0)

**🎯 Model-Agnostic Chat Templates**
- Automatically reads template from GGUF model metadata
- Works with ANY model (Qwen, Llama, Mistral, Phi, etc.)
- Based on llama.cpp's `llama_chat_apply_template()` API

**Before (hardcoded):**
```cpp
return "<|im_start|>user\n" + prompt + "<|im_end|>";  // Only Qwen!
```

**After (universal):**
```cpp
// Reads template from model, works with everything
const char* tmpl = llama_model_chat_template(model);
llama_chat_apply_template(tmpl, &msg, 1, ...);
```

**🎨 Quality Sampling (no more repetition)**
- Temperature 0.7 (creative but focused)
- Top-K 40 (filter noise)
- Top-P 0.9 (nucleus sampling)
- Deterministic seed (reproducible)

**Before (greedy):**
```
Write User class
Output: class User { int getUserint getUserint getUser... ❌
```

**After (temp=0.7):**
```cpp
class User {
private:
    std::string name;
    std::string email;
    int id;
    
public:
    User(const std::string& name, const std::string& email)
        : name(name), email(email), id(generateId()) {}
    
    // Validation
    bool isValidEmail() const {
        return email.find('@') != std::string::npos;
    }
    
    // Getters
    std::string getName() const { return name; }
    std::string getEmail() const { return email; }
    int getId() const { return id; }
};
```
✅ **Professional, complete, working code**

**📦 Better Token Allocation**
- Increased from 256 → 512 tokens default
- Complete functions with comments and examples
- Configurable via `NRVNA_PREDICT`

### Under the Hood

**Model Loading:**
```cpp
// llama.cpp integration
llama_model* model = llama_load_model_from_file(path);
llama_context* ctx = llama_new_context_with_model(model, params);
```

**Sampling Chain:**
```cpp
llama_sampler_chain_add(chain, llama_sampler_init_top_k(40));
llama_sampler_chain_add(chain, llama_sampler_init_top_p(0.9));
llama_sampler_chain_add(chain, llama_sampler_init_temp(0.7));
llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));
```

**Chat Template Application:**
```cpp
// Model-agnostic formatting
llama_chat_message msg = {"user", prompt};
llama_chat_apply_template(template, &msg, 1, add_assistant, buf, size);
```

**Based on:** `llama.cpp` examples/simple-chat/simple-chat.cpp

---

## Troubleshooting

### Model Won't Load

**Error:** `Failed to load model`

**Check:**
```bash
# 1. File exists and is GGUF
ls -lh ~/models/your-model.gguf

# 2. Not corrupted
strings ~/models/your-model.gguf | head -5
# Should show: GGUF, version, metadata

# 3. Has chat template
strings ~/models/your-model.gguf | grep "tokenizer.chat_template"
# Should show template (if missing, model may still work with fallback)
```

**Fix:**
- Re-download model if corrupted
- Check VRAM capacity: `nvidia-smi` (CUDA) or Activity Monitor (macOS Metal)
- Try smaller quantization: Q4_K_M instead of Q6_K

### Low Quality Output

**Symptoms:** Repetitive, boring, or cut-off responses

**Check configuration:**
```bash
echo $NRVNA_TEMP      # Should be 0.6-0.8 (not 0.0!)
echo $NRVNA_PREDICT   # Should be 512+ for code
echo $NRVNA_TOP_K     # Should be 30-50
```

**Common fixes:**
```bash
# If repetitive: increase temperature
export NRVNA_TEMP=0.7

# If cut off: increase tokens
export NRVNA_PREDICT=768

# If too random: decrease temperature
export NRVNA_TEMP=0.6
```

### Slow Inference

**Check GPU offload:**
```bash
echo $NRVNA_GPU_LAYERS  # Should be 99 for full GPU
```

**Metal (macOS):**
```bash
# Verify Metal is working
log show --predicate 'subsystem == "com.apple.Metal"' --last 1m | grep llama
```

**Performance expectations:**
- **CPU-only:** 1-3 tokens/sec (very slow)
- **Partial GPU:** 5-8 tokens/sec (okay)
- **Full GPU:** 10-15 tokens/sec (good)

**Fixes:**
```bash
# Enable full GPU offload
export NRVNA_GPU_LAYERS=99

# Reduce context if tight on VRAM
export NRVNA_CTX=2048

# Use smaller model
# Qwen 1.5B instead of 3B
```

### Jobs Stuck in `processing/`

**Cause:** Daemon crashed or killed

**Fix:**
```bash
# Kill old daemon
pkill nrvnad

# Manually reset stuck jobs
mv ~/workspace/processing/* ~/workspace/input/

# Restart daemon
./nrvnad ~/models/qwen-coder.gguf ~/workspace &
```

### Out of Memory (OOM)

**Error:** `Metal out of memory` or crash

**Check model size vs VRAM:**
```bash
ls -lh ~/models/your-model.gguf
# 1-2GB = needs 4GB VRAM
# 3-4GB = needs 6-8GB VRAM
# 7GB+ = needs 10GB+ VRAM
```

**Fixes:**
1. Use smaller model (Qwen 1.5B instead of 3B)
2. Reduce context: `export NRVNA_CTX=2048`
3. Use lower quantization: Q4_K_M instead of Q6_K
4. Reduce GPU layers: `export NRVNA_GPU_LAYERS=20` (hybrid CPU/GPU)

---

## Advanced Usage

### Multiple Workspaces

```bash
# Code generation workspace
./nrvnad ~/models/qwen-coder.gguf ~/workspaces/code &

# Documentation workspace (different model)
./nrvnad ~/models/mistral-7b.gguf ~/workspaces/docs &

# Submit to different workspaces
./wrk ~/workspaces/code "Write API handler"
./wrk ~/workspaces/docs "Explain architecture"
```

### Custom Prompts

**Workspace-specific context:**
```bash
# Create .nrvna/context.txt in workspace
echo "This is a C++ project using modern standards" > ~/workspace/.nrvna/context.txt

# Jobs will include this context (future feature)
```

### Batch Processing

```bash
# Submit many jobs at once
for func in binary_search quicksort merge_sort heap_sort; do
    ./wrk ~/workspace "Write ${func} in C++ with tests"
done

# Check all results
./flw ~/workspace
```

### Integration with Scripts

```bash
#!/bin/bash
# auto-review.sh - Submit code reviews for all .cpp files

for file in src/*.cpp; do
    content=$(cat "$file")
    ./wrk ~/reviews "Review this code for bugs:\n\n${content}"
done

echo "Reviews submitted. Check results with: flw ~/reviews"
```

---

## Comparison: nrvna vs llama.cpp

| Feature | llama.cpp | nrvna |
|---------|-----------|-------|
| **Use Case** | Interactive experimentation | Production batch processing |
| **Workflow** | Synchronous (wait for output) | Async (submit and continue) |
| **Learning Curve** | Learn all flags/options | Set env vars once, forget |
| **Best For** | Testing prompts, finding settings | Scaling proven workflows |
| **Model Support** | All GGUF | All GGUF (via templates) |
| **Output** | Terminal (must capture) | Organized filesystem |
| **Multi-task** | One at a time | Queue unlimited jobs |

**When to use llama.cpp:**
- Testing new models
- Experimenting with prompts
- Interactive chat sessions
- Learning what works

**When to use nrvna:**
- You found your perfect settings
- You have recurring tasks
- You want to batch process
- You value flow state over immediacy

**The journey:** Experiment with llama.cpp → Graduate to nrvna

---

## Development

### Building from Source

```bash
git clone https://github.com/yourusername/nrvna-ai.git
cd nrvna-ai
git submodule update --init --recursive  # Get llama.cpp

mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8

# Binaries in: build/nrvnad, build/wrk, build/flw
```

### Testing

```bash
cd build
ctest --output-on-failure

# Run specific test
./unit_tests
./integration_tests
```

### Project Structure

```
nrvna-ai/
├── cli/          - Command-line tools (wrk, flw, nrvnad)
├── src/          - Core implementation
│   ├── runner.cpp    - LLM inference wrapper
│   ├── work.cpp      - Job management
│   ├── monitor.cpp   - Filesystem watcher
│   └── server.cpp    - Daemon (future: HTTP API)
├── include/      - Public headers
├── third_party/  - llama.cpp submodule
└── build/        - Build artifacts
```

---

## FAQ

**Q: Why not just use llama.cpp directly?**

A: You should! nrvna is for when you've found what works and want to scale. If you're still experimenting, llama.cpp is perfect.

**Q: Does this upload my code to a server?**

A: No. Everything runs locally. Your code never leaves your machine.

**Q: What about OpenAI/Claude API?**

A: nrvna is for local models only. If you have API credits, use those! If you want privacy, cost control, or offline capability, use nrvna.

**Q: Can I use this for chat/conversation?**

A: Technically yes, but not recommended. nrvna is designed for single-shot jobs. For chat, use llama.cpp's server mode or a proper chat client.

**Q: How do I know my model will work before downloading?**

A: Check model card on Hugging Face:
1. Format: Must be GGUF
2. Type: "Instruct" or "Chat" preferred
3. Size: Must fit in your VRAM (see guide above)
4. Template: Check if model card mentions chat template support

**Q: Why "nrvna"?**

A: Nirvana = state of perfect peace. You achieve flow state by delegating work to the daemon. Also sounds like "nerve-ana" (neural analysis).

**Q: Can I contribute?**

A: Absolutely! See CONTRIBUTING.md (future). Key areas:
- Testing with different models
- Documentation improvements
- Platform support (Windows, Linux)
- Performance optimizations

---

## Changelog

### v0.2.1 (Nov 2025) - Match llama.cpp Defaults
- ✅ **Defaults now match llama.cpp exactly** (temp=0.8, top_p=0.95, min_p=0.05)
- ✅ Min-P sampling support (configurable)
- ✅ Repetition penalty support (configurable)
- ✅ Improved job completion logging (✅ JOB COMPLETED visible in logs)
- ✅ Focus on code-specialized models (Qwen Coder, CodeLlama recommended)
- 📝 Sampling comparison docs (no more guessing!)

### v0.2.0 (Nov 2025) - Model Agnostic
- ✅ Universal chat template support (works with ANY GGUF)
- ✅ Quality sampling (temp=0.7, top-k=40, top-p=0.9)
- ✅ Increased token allocation (256→512 default)
- ✅ Environment variable configuration
- 📝 Comprehensive documentation (LEARNINGS.md)

### v0.1.0 (Oct 2025) - Initial MVP
- ✅ Basic async job processing
- ✅ Filesystem-based queue
- ✅ llama.cpp integration
- ⚠️ Hardcoded ChatML (Qwen-only)
- ⚠️ Greedy sampling (repetition issues)

---

## License

MIT License - see LICENSE file

---

## Resources

- **llama.cpp:** https://github.com/ggerganov/llama.cpp
- **GGUF Models:** https://huggingface.co/models?library=gguf
- **Documentation:** See LEARNINGS.md for deep technical dive
- **Issues:** https://github.com/yourusername/nrvna-ai/issues

---

**Ready to scale your LLM workflow? Start with one model, find your settings, then nrvna takes it from there.**

**"nrvna gets wrk. you get flw."**
