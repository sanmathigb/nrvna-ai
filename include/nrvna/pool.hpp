/*
 * nrvna ai - Asynchronous Inference Primitive
 * Copyright (c) 2025 Sanmathi Bharamgouda
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "nrvna/types.hpp"

namespace nrvnaai {

using JobProcessor = std::function<void(const JobId&, int workerId)>;

class Pool {
public:
    explicit Pool(int workers) noexcept;
    ~Pool();

    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;
    Pool(Pool&&) = delete;
    Pool& operator=(Pool&&) = delete;

    [[nodiscard]] bool start(JobProcessor processor);
    void stop() noexcept;
    void submit(const JobId& jobId) noexcept;
    
    [[nodiscard]] bool isRunning() const noexcept { return running_.load(); }
    [[nodiscard]] std::size_t queueSize() const noexcept;
    [[nodiscard]] int workerCount() const noexcept { return workers_; }

private:
    void workerLoop(int workerId);
    
    int workers_;
    JobProcessor processor_;
    
    std::atomic<bool> running_{false};
    std::atomic<bool> shutdown_{false};
    
    mutable std::mutex queueMutex_;
    std::condition_variable jobAvailable_;
    std::queue<JobId> jobQueue_;
    
    std::vector<std::thread> workerThreads_;
};

}