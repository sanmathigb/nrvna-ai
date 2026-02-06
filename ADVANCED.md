# Advanced Patterns

The primitives (`nrvnad`, `wrk`, `flw`) are building blocks. Here's what you can build with them.

---

## Batch Processing

Submit many jobs, collect results later:

```bash
# Submit 100 jobs in seconds
for img in photos/*.jpg; do
  wrk ./workspace "Caption this image" --image "$img" >> jobs.txt
done

# Check progress
ls ./workspace/output/ | wc -l

# Collect all results
for job in $(cat jobs.txt); do
  echo "=== $job ==="
  flw ./workspace $job
done
```

---

## Fan-Out / Fan-In (Map-Reduce)

Break a task into parallel subtasks, synthesize:

```bash
# Fan-out: submit subtasks
job1=$(wrk ./workspace "Research: PostgreSQL scalability")
job2=$(wrk ./workspace "Research: PostgreSQL ecosystem")
job3=$(wrk ./workspace "Research: PostgreSQL vs MongoDB")

# Wait for all
result1=$(flw ./workspace $job1)
result2=$(flw ./workspace $job2)
result3=$(flw ./workspace $job3)

# Fan-in: synthesize
wrk ./workspace "Synthesize these findings into a recommendation:
$result1
$result2
$result3"
```

---

## Self-Refinement Loop

Generate, critique, improve:

```bash
GOAL="Write a cover letter for a senior engineer position"

# First draft
draft=$(wrk ./workspace "$GOAL" | xargs flw ./workspace)

# Critique
critique=$(wrk ./workspace "Critique this draft. What's weak? $draft" | xargs flw ./workspace)

# Improve
final=$(wrk ./workspace "Improve this draft based on feedback:
Draft: $draft
Feedback: $critique" | xargs flw ./workspace)

echo "$final"
```

---

## Agent Loop

Iterate until done:

```bash
GOAL="Write a Python tutorial covering variables, loops, and functions"
memory=""

for i in {1..5}; do
  result=$(wrk ./workspace "Goal: $GOAL
Previous work: $memory
Continue. Write the next section. Say DONE if complete." | xargs flw ./workspace)

  echo "=== Iteration $i ==="
  echo "$result"

  if echo "$result" | grep -q "DONE"; then
    break
  fi

  memory="$memory\n---\n$result"
done
```

---

## Event-Driven (Watch for Results)

Monitor completions with `fswatch`:

```bash
# Terminal 1: Watch for results
fswatch -0 ./workspace/output | while read -d '' path; do
  [[ "$path" == */result.txt ]] && cat "$path"
done

# Terminal 2: Submit jobs
wrk ./workspace "Question 1"
wrk ./workspace "Question 2"
```

---

## Memory / Context Accumulation

Build up knowledge across jobs:

```bash
# Save results to memory
flw ./workspace $job1 >> memory.txt
flw ./workspace $job2 >> memory.txt
flw ./workspace $job3 >> memory.txt

# Use memory as context
wrk ./workspace "Given this context:
$(cat memory.txt)

Answer: What's the best approach for a startup?"
```

---

## Multi-Model Routing

Different models for different tasks:

```bash
# Start specialized daemons
nrvnad qwen-vl.gguf    ./ws-vision 2 --mmproj qwen-vl-mmproj.gguf &
nrvnad codellama.gguf  ./ws-code   4 &
nrvnad phi-3-mini.gguf ./ws-fast   2 &

# Route by task type
classify() {
  case "$1" in
    *.jpg|*.png) echo "./ws-vision" ;;
    *.py|*.js)   echo "./ws-code" ;;
    *)           echo "./ws-fast" ;;
  esac
}

# Submit to appropriate workspace
ws=$(classify "$input")
wrk "$ws" "Process this: $input"
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NRVNA_PREDICT` | `768` | Max tokens to generate |
| `NRVNA_TEMP` | `0.8` | Sampling temperature |
| `NRVNA_CTX` | `8192` | Context window size |
| `NRVNA_BATCH` | `2048` | Batch size |
| `NRVNA_GPU_LAYERS` | `99` (Mac) | Layers offloaded to GPU |

---

## Tips

1. **More workers = more parallelism** — but diminishing returns past CPU cores
2. **Vision is serialized** — mutex prevents parallel vision corruption
3. **Jobs are directories** — inspect with `ls`, `cat`, `tree`
4. **Atomic state** — job location *is* job state
5. **Compose with shell** — the primitives are designed for piping
