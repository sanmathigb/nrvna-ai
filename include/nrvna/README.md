# nrvna Headers - C++ Design Concepts & Architecture

## Directory Overview
This directory contains all public interfaces for the nrvna-ai asynchronous inference system.

## C++ Design Patterns Applied

### [[nodiscard]] Usage Pattern
```cpp
[[nodiscard]] SubmitResult submit(const std::string& prompt);
[[nodiscard]] bool start();
[[nodiscard]] std::optional<Job> get(const JobId& id);
```
**When to use**: Functions returning values that must be checked (errors, resources, important data)
**Why**: Prevents accidentally ignoring critical return values that indicate success/failure

### Class vs Struct Decision Framework
```cpp
struct SubmitResult {     // Pure data holder
    bool ok;              // All members public
    JobId id;            // No invariants to maintain
    std::string message; // Simple composition
};

class Work {             // Behavior + encapsulation  
private:                 // Hidden implementation
    std::filesystem::path workspace_;
public:                  // Controlled interface
    [[nodiscard]] SubmitResult submit(const std::string& prompt);
};
```
**Rule**: `struct` = pure data containers, `class` = behavior with encapsulation

### noexcept Specifications Strategy
```cpp
void setMaxSize(std::size_t maxBytes) noexcept;     // Cannot fail
[[nodiscard]] std::size_t queueSize() const noexcept;  // Read-only, safe
[[nodiscard]] bool isRunning() const noexcept;     // Atomic read, guaranteed safe

// NOT noexcept - filesystem operations can fail
[[nodiscard]] SubmitResult submit(const std::string& prompt);
```
**Guidelines**:
- Use `noexcept` for operations that cannot fail (setters, atomic reads, destructors)
- Avoid `noexcept` for filesystem, memory allocation, or complex operations

### Static Method Design
```cpp
static JobId generateId() noexcept;              // Pure function, no instance state
static bool isValidPrompt(const std::string&) noexcept;  // Utility function
```
**When static**: Function belongs to class conceptually but needs no instance data

### Move Semantics Pattern
```cpp
Work(Work&&) noexcept = default;                // Enable efficient moves
Work& operator=(Work&&) noexcept = default;

// But disable copying for resource-owning classes
Work(const Work&) = delete;                     // No accidental copying
Work& operator=(const Work&) = delete;
```
**Pattern**: Enable moves, disable copies for resource-owning classes

## Architectural Principles

### Single Responsibility Principle (SRP) 
- **Scanner**: Only discovers jobs in directories
- **Pool**: Only manages worker threads  
- **Processor**: Only executes individual jobs
- **Work**: Only job submission and validation
- **Flow**: Only job result retrieval

### Error Handling Philosophy
**No Exceptions Approach**:
```cpp
// Return structured results instead of throwing
struct SubmitResult { bool ok; JobId id; SubmissionError error; };

// Use std::optional for nullable returns  
[[nodiscard]] std::optional<Job> get(const JobId& id) const noexcept;

// Use enum classes for typed errors
enum class SubmissionError : uint8_t { None, IoError, InvalidSize };
```

### Dependency Injection Points
```cpp
// Constructor injection for dependencies
explicit Processor(const std::filesystem::path& workspace, std::shared_ptr<Runner> runner);

// Function injection for behavior
using JobProcessor = std::function<void(const JobId&)>;
[[nodiscard]] bool Pool::start(JobProcessor processor);
```

### RAII Resource Management
```cpp
class Pool {
    ~Pool();  // Destructor automatically stops threads and cleans up
private:
    std::vector<std::thread> workerThreads_;  // Automatic cleanup
};
```

## Component Interaction Design

### Directory-Based State Machine
```
input/writing/<id>/    →  input/ready/<id>/    →  processing/<id>/  →  output/<id>/
     (Work)                  (Scanner)              (Processor)          (Flow)
```

### Atomic Operations Guarantee
- **Work**: Atomic directory rename from `writing/` to `ready/`
- **Processor**: Atomic directory rename from `ready/` to `processing/`
- **Final**: Atomic directory rename to `output/` or `failed/`

## Memory Management Patterns

### Smart Pointer Usage
```cpp
std::unique_ptr<Scanner> scanner_;    // Exclusive ownership
std::shared_ptr<Runner> runner_;      // Shared across components  
```

### Thread Safety Approach  
- **Immutable by design**: Most operations are read-only
- **Atomic primitives**: For simple state (`std::atomic<bool>`)
- **Mutex protection**: Only where necessary (`std::mutex queueMutex_`)

## Testing Interfaces
All components designed for easy testing:
- Constructor dependency injection
- Pure functions where possible
- Clear separation between I/O and logic
- Mockable interfaces (virtual destructors, function injection)

This design enables reliable, testable, maintainable C++ code following industry best practices.