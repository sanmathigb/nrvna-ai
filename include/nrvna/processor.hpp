/*
 * nrvna ai - Asynchronous Inference Primitive
 * Copyright (c) 2025 Sanmathi Bharamgouda
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include <filesystem>
#include <memory>

#include "nrvna/types.hpp"
#include <unordered_map>
#include <mutex>

namespace nrvnaai {

class Runner;

enum class ProcessResult : uint8_t {
    Success,
    Failed,
    NotFound,
    SystemError
};

class Processor {
public:
    explicit Processor(const std::filesystem::path& workspace, const std::string& modelPath);
    
    Processor(const Processor&) = delete;
    Processor& operator=(const Processor&) = delete;
    Processor(Processor&&) = delete;
    Processor& operator=(Processor&&) = delete;

    // Pre-initialize runners for all worker threads (MUST be called before threads start)
    bool initializeRunners(int numWorkers);

    [[nodiscard]] ProcessResult process(const JobId& jobId, int workerId) noexcept;

private:
    std::filesystem::path workspace_;
    std::string modelPath_;
    
    // Per-thread Runner instances for Metal compatibility
    std::unordered_map<int, std::unique_ptr<Runner>> runners_;
    std::mutex runnersMutex_;
    
    [[nodiscard]] bool moveReadyToProcessing(const JobId& jobId) noexcept;
    [[nodiscard]] bool finalizeSuccess(const JobId& jobId, const std::string& result) noexcept;
    [[nodiscard]] bool finalizeFailure(const JobId& jobId, const std::string& error) noexcept;
    
    [[nodiscard]] std::string readPrompt(const JobId& jobId) const noexcept;
    [[nodiscard]] std::filesystem::path getJobPath(const char* phase, const JobId& jobId) const noexcept;
    
    // Metal-compatible per-thread Runner management
    std::unique_ptr<Runner>& getRunnerForWorker(int workerId);
};

}