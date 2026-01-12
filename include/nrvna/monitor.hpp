/*
 * nrvna ai - Asynchronous Inference Primitive
 * Copyright (c) 2025 Sanmathi Bharamgouda
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "nrvna/types.hpp"

namespace nrvnaai {

class Runner;

class Monitor final {
public:
    Monitor(const std::filesystem::path& workspace, const std::string& modelPath, int workers);
    ~Monitor();

    Monitor(const Monitor&) = delete;
    Monitor& operator=(const Monitor&) = delete;
    Monitor(Monitor&&) = delete;
    Monitor& operator=(Monitor&&) = delete;

    bool start();
    bool stop();
    [[nodiscard]] bool isRunning() const noexcept { return running_.load(); }
    [[nodiscard]] std::size_t queueSize() const noexcept;
    [[nodiscard]] const std::filesystem::path& workspace() const noexcept { return workspace_; }

private:
    bool createDirectories();
    void scanLoop();
    void workerLoop(int workerIndex);
    bool enqueueNewJobs();
    bool processJob(const JobId& id, Runner& runner);

    bool moveReadyToProcessing(const JobId& jobId);
    bool finalizeSuccess(const JobId& jobId);
    bool finalizeFailure(const JobId& jobId, const std::string& errorMessage);

    mutable std::mutex queueMutex_;
    std::condition_variable queueCondition_;
    std::queue<JobId> jobQueue_;

    std::filesystem::path workspace_;
    std::string modelPath_;
    int workers_;

    std::atomic<bool> running_{false};
    std::thread scannerThread_;
    std::vector<std::thread> workerThreads_;
    std::vector<std::unique_ptr<Runner>> runners_;
};

}