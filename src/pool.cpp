/*
 * nrvna ai - Asynchronous Inference Primitive
 * Copyright (c) 2025 Sanmathi Bharamgouda
 * SPDX-License-Identifier: MIT
 */

#include "nrvna/pool.hpp"
#include "nrvna/logger.hpp"
#include <chrono>

namespace nrvnaai {

Pool::Pool(int workers) noexcept : workers_(workers) {
    LOG_DEBUG("Pool created with " + std::to_string(workers) + " workers");
}

Pool::~Pool() {
    stop();
}

bool Pool::start(JobProcessor processor) {
    if (running_.load()) {
        LOG_WARN("Pool already running");
        return false;
    }

    if (!processor) {
        LOG_ERROR("Invalid job processor provided");
        return false;
    }

    processor_ = processor;
    running_.store(true);
    shutdown_.store(false);

    try {
        workerThreads_.reserve(workers_);
        for (int i = 0; i < workers_; ++i) {
            workerThreads_.emplace_back(&Pool::workerLoop, this, i);
            // LOG_DEBUG("Started worker thread: Worker-" + std::to_string(i));
        }
        
        LOG_INFO("Pool started successfully with " + std::to_string(workers_) + " worker threads");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to start pool: " + std::string(e.what()));
        running_.store(false);
        return false;
    }
}

void Pool::stop() noexcept {
    if (!running_.load()) {
        return;
    }

    LOG_DEBUG("Stopping pool...");
    
    // Signal shutdown
    shutdown_.store(true);
    running_.store(false);
    
    // Wake up all waiting threads
    jobAvailable_.notify_all();
    
    // Wait for all threads to finish
    for (auto& thread : workerThreads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    workerThreads_.clear();
    
    // Clear remaining jobs
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        while (!jobQueue_.empty()) {
            jobQueue_.pop();
        }
    }
    
    LOG_INFO("Pool stopped");
}

void Pool::submit(const JobId& jobId) noexcept {
    if (!running_.load() || shutdown_.load()) {
        LOG_DEBUG("Cannot submit job to stopped pool: " + jobId);
        return;
    }

    try {
        {
            std::lock_guard<std::mutex> lock(queueMutex_);
            jobQueue_.push(jobId);
        }
        
        jobAvailable_.notify_one();
        LOG_DEBUG("Job queued: " + jobId);
    } catch (...) {
        LOG_ERROR("Failed to queue job: " + jobId);
    }
}

std::size_t Pool::queueSize() const noexcept {
    try {
        std::lock_guard<std::mutex> lock(queueMutex_);
        return jobQueue_.size();
    } catch (...) {
        return 0;
    }
}

void Pool::workerLoop(int workerId) {
    // Name this thread for logging
    setThreadName("Worker-" + std::to_string(workerId));
    LOG_DEBUG("Worker-" + std::to_string(workerId) + " thread started");
    
    try {
        while (!shutdown_.load()) {
            JobId jobId;
            
            // Get next job
            {
                std::unique_lock<std::mutex> lock(queueMutex_);
                
                // Wait for job or shutdown signal
                jobAvailable_.wait(lock, [this] { 
                    return !jobQueue_.empty() || shutdown_.load(); 
                });
                
                if (shutdown_.load()) {
                    break;
                }
                
                if (jobQueue_.empty()) {
                    continue;
                }
                
                jobId = jobQueue_.front();
                jobQueue_.pop();
            }
            
            // Process job outside of lock
            if (!jobId.empty()) {
                LOG_INFO("Worker-" + std::to_string(workerId) + " claimed job: " + jobId);
                
                try {
                    processor_(jobId, workerId);
                } catch (const std::exception& e) {
                    LOG_ERROR("Worker " + std::to_string(workerId) + " job processing error: " + 
                             std::string(e.what()) + " (job: " + jobId + ")");
                } catch (...) {
                    LOG_ERROR("Worker " + std::to_string(workerId) + " unknown job processing error (job: " + jobId + ")");
                }
            }
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Worker " + std::to_string(workerId) + " fatal error: " + std::string(e.what()));
    } catch (...) {
        LOG_ERROR("Worker " + std::to_string(workerId) + " unknown fatal error");
    }
    
    LOG_DEBUG("Worker " + std::to_string(workerId) + " stopped");
}

}