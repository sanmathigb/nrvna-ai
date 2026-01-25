/*
 * nrvna ai - Asynchronous Inference Primitive
 * Copyright (c) 2025 Sanmathi Bharamgouda
 * SPDX-License-Identifier: MIT
 */

#include "nrvna/processor.hpp"
#include "nrvna/runner.hpp"
#include "nrvna/logger.hpp"
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>

namespace {
std::mutex g_output_mutex;

std::string timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    char buf[16];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", std::localtime(&time));
    return buf;
}

}

namespace nrvnaai {

Processor::Processor(const std::filesystem::path& workspace, const std::string& modelPath)
    : workspace_(workspace), modelPath_(modelPath) {
    LOG_DEBUG("Processor created for workspace: " + workspace_.string() + " with model: " + modelPath_);
}

ProcessResult Processor::process(const JobId& jobId, int workerId) noexcept {
    // CRITICAL: Get or create Runner instance for this worker thread
    // This implements the original nrvna pattern for Metal compatibility
    std::unique_ptr<Runner>& runner = getRunnerForWorker(workerId);
    LOG_DEBUG("Processing job: " + jobId);
    
    try {
        // Step 1: Move from ready to processing (atomic)
        if (!moveReadyToProcessing(jobId)) {
            LOG_DEBUG("Job not found or already claimed by another worker: " + jobId);
            return ProcessResult::NotFound;
        }

        {
            std::lock_guard<std::mutex> lock(g_output_mutex);
            std::cout << "    \033[90m" << timestamp() << "\033[0m  " << jobId << "  \033[33mrunning\033[0m\n" << std::flush;
        }
        auto startTime = std::chrono::steady_clock::now();

        // Step 2: Read prompt
        std::string prompt = readPrompt(jobId);
        if (prompt.empty()) {
            {
                std::lock_guard<std::mutex> lock(g_output_mutex);
                std::cout << "    \033[90m" << timestamp() << "\033[0m  " << jobId << "  \033[31mfailed\033[0m  empty prompt\n" << std::flush;
            }
            (void)finalizeFailure(jobId, "Failed to read prompt file");
            return ProcessResult::Failed;
        }

        // Step 3: Run inference
        if (!runner) {
            {
                std::lock_guard<std::mutex> lock(g_output_mutex);
                std::cout << "    \033[90m" << timestamp() << "\033[0m  " << jobId << "  \033[31mfailed\033[0m  no runner\n" << std::flush;
            }
            (void)finalizeFailure(jobId, "No runner available");
            return ProcessResult::SystemError;
        }

        RunResult result = runner->run(prompt);
        
        // Step 4: Finalize based on result
        if (result.ok) {
            if (finalizeSuccess(jobId, result.output)) {
                auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();
                {
                    std::lock_guard<std::mutex> lock(g_output_mutex);
                    std::cout << "    \033[90m" << timestamp() << "\033[0m  " << jobId << "  \033[32mdone\033[0m  " << std::fixed << std::setprecision(1) << elapsed << "s\n" << std::flush;
                }
                LOG_INFO("JOB COMPLETED: " + jobId + " -> " + std::to_string(result.output.size()) + " chars");
                return ProcessResult::Success;
            } else {
                LOG_ERROR("Failed to finalize successful job: " + jobId);
                return ProcessResult::SystemError;
            }
        } else {
            auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();
            {
                std::lock_guard<std::mutex> lock(g_output_mutex);
                std::cout << "    \033[90m" << timestamp() << "\033[0m  " << jobId << "  \033[31mfailed\033[0m  " << std::fixed << std::setprecision(1) << elapsed << "s\n" << std::flush;
            }
            (void)finalizeFailure(jobId, result.error);
            LOG_WARN("Job failed during inference: " + jobId + " - " + result.error);
            return ProcessResult::Failed;
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR("Exception processing job " + jobId + ": " + std::string(e.what()));
        (void)finalizeFailure(jobId, "Internal processing error: " + std::string(e.what()));
        return ProcessResult::SystemError;
    } catch (...) {
        LOG_ERROR("Unknown exception processing job: " + jobId);
        (void)finalizeFailure(jobId, "Unknown internal processing error");
        return ProcessResult::SystemError;
    }
}

bool Processor::moveReadyToProcessing(const JobId& jobId) noexcept {
    try {
        auto readyPath = getJobPath("input/ready", jobId);
        auto processingPath = getJobPath("processing", jobId);
        
        // Use std::error_code to handle race condition gracefully
        std::error_code ec;
        std::filesystem::rename(readyPath, processingPath, ec);
        
        if (ec) {
            // Another worker already claimed it, or job disappeared
            LOG_DEBUG("Job already claimed or missing: " + jobId);
            return false;
        }
        
        LOG_DEBUG("Job moved to processing: " + jobId);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to move job to processing: " + std::string(e.what()));
        return false;
    } catch (...) {
        LOG_ERROR("Unknown error moving job to processing");
        return false;
    }
}

bool Processor::finalizeSuccess(const JobId& jobId, const std::string& result) noexcept {
    try {
        auto processingPath = getJobPath("processing", jobId);
        auto outputPath = getJobPath("output", jobId);
        
        // Write result to temporary file first
        auto tempResultPath = processingPath / "result.txt.tmp";
        {
            std::ofstream file(tempResultPath, std::ios::binary);
            if (!file) return false;
            file << result;
            file.flush();
            if (!file.good()) return false;
        }
        
        // Rename temp file to final name
        auto finalResultPath = processingPath / "result.txt";
        std::filesystem::rename(tempResultPath, finalResultPath);
        
        // Atomic move entire job to output
        std::filesystem::rename(processingPath, outputPath);
        
        LOG_DEBUG("Job finalized successfully: " + jobId);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to finalize success for job " + jobId + ": " + std::string(e.what()));
        return false;
    } catch (...) {
        LOG_ERROR("Unknown error finalizing success for job: " + jobId);
        return false;
    }
}

bool Processor::finalizeFailure(const JobId& jobId, const std::string& error) noexcept {
    try {
        auto processingPath = getJobPath("processing", jobId);
        auto failedPath = getJobPath("failed", jobId);
        
        // Write error to file
        auto errorPath = processingPath / "error.txt";
        {
            std::ofstream file(errorPath, std::ios::binary);
            if (file) {
                file << error;
                file.flush();
            }
            // Continue even if error file write fails
        }
        
        // Atomic move to failed directory
        std::filesystem::rename(processingPath, failedPath);
        
        LOG_DEBUG("Job moved to failed: " + jobId);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to finalize failure for job " + jobId + ": " + std::string(e.what()));
        return false;
    } catch (...) {
        LOG_ERROR("Unknown error finalizing failure for job: " + jobId);
        return false;
    }
}

std::string Processor::readPrompt(const JobId& jobId) const noexcept {
    try {
        auto promptPath = getJobPath("processing", jobId) / "prompt.txt";
        
        if (!std::filesystem::exists(promptPath)) {
            LOG_ERROR("Prompt file not found: " + promptPath.string());
            return "";
        }
        
        std::ifstream file(promptPath, std::ios::binary);
        if (!file) {
            LOG_ERROR("Failed to open prompt file: " + promptPath.string());
            return "";
        }
        
        std::string content((std::istreambuf_iterator<char>(file)), 
                           std::istreambuf_iterator<char>());
        
        if (content.empty()) {
            LOG_WARN("Empty prompt file: " + promptPath.string());
        }
        
        return content;
    } catch (const std::exception& e) {
        LOG_ERROR("Exception reading prompt for job " + jobId + ": " + std::string(e.what()));
        return "";
    } catch (...) {
        LOG_ERROR("Unknown error reading prompt for job: " + jobId);
        return "";
    }
}

std::filesystem::path Processor::getJobPath(const char* phase, const JobId& jobId) const noexcept {
    try {
        return workspace_ / phase / jobId;
    } catch (...) {
        return {};
    }
}

// Pre-initialize all Runner instances before worker threads start
// This ensures ggml_backend_load_all() is called sequentially from main thread
bool Processor::initializeRunners(int numWorkers) {
    std::lock_guard<std::mutex> lock(runnersMutex_);
    
    try {
        for (int i = 0; i < numWorkers; ++i) {
            LOG_DEBUG("Pre-creating Runner instance for worker " + std::to_string(i));
            runners_[i] = std::make_unique<Runner>(modelPath_);
        }
        LOG_DEBUG("All " + std::to_string(numWorkers) + " Runner instances initialized");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to initialize runners: " + std::string(e.what()));
        return false;
    }
}

// CRITICAL: Metal-compatible per-thread Runner management
std::unique_ptr<Runner>& Processor::getRunnerForWorker(int workerId) {
    std::lock_guard<std::mutex> lock(runnersMutex_);
    
    auto it = runners_.find(workerId);
    if (it == runners_.end()) {
        // This should never happen if initializeRunners() was called properly
        LOG_ERROR("Runner not found for worker " + std::to_string(workerId) + " - was initializeRunners() called?");
        throw std::runtime_error("Runner not initialized for worker " + std::to_string(workerId));
    }
    
    return it->second;
}

}