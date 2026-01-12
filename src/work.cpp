/*
 * nrvna ai - Asynchronous Inference Primitive
 * Copyright (c) 2025 Sanmathi Bharamgouda
 * SPDX-License-Identifier: MIT
 */

#include "nrvna/work.hpp"
#include "nrvna/logger.hpp"
#include <filesystem>
#include <fstream>
#include <chrono>
#include <sstream>
#include <atomic>
#include <unistd.h>

namespace nrvnaai {

Work::Work(const std::filesystem::path& workspace, bool createIfMissing) 
    : workspace_(workspace) {
    if (!createWorkspace(createIfMissing)) {
        LOG_ERROR("Failed to initialize workspace: " + workspace_.string());
    }
}

SubmitResult Work::submit(const std::string& prompt) {
    if (!isValidPrompt(prompt)) {
        LOG_DEBUG("Invalid prompt: empty or too large");
        return {false, "", SubmissionError::InvalidContent, "Prompt is empty or exceeds size limit"};
    }

    if (prompt.size() > maxBytes_) {
        LOG_DEBUG("Prompt exceeds size limit: " + std::to_string(prompt.size()) + " > " + std::to_string(maxBytes_));
        return {false, "", SubmissionError::InvalidSize, "Prompt exceeds maximum size limit"};
    }

    JobId jobId = generateId();
    LOG_DEBUG("Generated job ID: " + jobId);

    // Step 1: Create job directory in writing area
    if (!createJobDirectory(jobId)) {
        LOG_ERROR("Failed to create job directory for: " + jobId);
        return {false, "", SubmissionError::IoError, "Failed to create job directory"};
    }

    // Step 2: Write prompt file
    if (!writePromptFile(jobId, prompt)) {
        LOG_ERROR("Failed to write prompt file for: " + jobId);
        // Cleanup failed job directory
        std::filesystem::remove_all(workspace_ / "input" / "writing" / jobId);
        return {false, "", SubmissionError::IoError, "Failed to write prompt file"};
    }

    // Step 3: Atomic publish to ready directory
    if (!atomicPublish(jobId)) {
        LOG_ERROR("Failed to publish job: " + jobId);
        // Cleanup failed job directory
        std::filesystem::remove_all(workspace_ / "input" / "writing" / jobId);
        return {false, "", SubmissionError::IoError, "Failed to publish job"};
    }

    LOG_INFO("Job submitted successfully: " + jobId);
    return {true, jobId, SubmissionError::None, ""};
}

bool Work::createWorkspace(bool createIfMissing) noexcept {
    try {
        if (!std::filesystem::exists(workspace_)) {
            if (!createIfMissing) {
                return false;
            }
            std::filesystem::create_directories(workspace_);
        }

        // Work only needs input directories for job submission
        // Server creates processing/output/failed directories
        std::filesystem::create_directories(workspace_ / "input" / "writing");
        std::filesystem::create_directories(workspace_ / "input" / "ready");

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to create workspace: " + std::string(e.what()));
        return false;
    } catch (...) {
        LOG_ERROR("Unknown error creating workspace");
        return false;
    }
}

JobId Work::generateId() noexcept {
    static std::atomic<uint64_t> counter{0};
    
    try {
        auto now = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        uint64_t unique_counter = counter.fetch_add(1);
        
        std::stringstream ss;
        ss << std::to_string(now) << "_" << getpid() << "_" << unique_counter;
        return ss.str();
    } catch (...) {
        // Fallback to simpler ID generation if anything fails
        uint64_t unique_counter = counter.fetch_add(1);
        return "fallback_" + std::to_string(getpid()) + "_" + std::to_string(unique_counter);
    }
}

bool Work::isValidPrompt(const std::string& prompt) noexcept {
    return !prompt.empty() && prompt.size() <= 1'000'000; // 1MB max for safety
}

bool Work::createJobDirectory(const JobId& jobId) const noexcept {
    try {
        auto jobPath = workspace_ / "input" / "writing" / jobId;
        return std::filesystem::create_directories(jobPath);
    } catch (...) {
        return false;
    }
}

bool Work::writePromptFile(const JobId& jobId, const std::string& prompt) const noexcept {
    try {
        auto promptPath = workspace_ / "input" / "writing" / jobId / "prompt.txt";
        std::ofstream file(promptPath, std::ios::binary);
        if (!file) return false;
        
        file << prompt;
        file.flush();
        file.close();
        
        return file.good();
    } catch (...) {
        return false;
    }
}

bool Work::atomicPublish(const JobId& jobId) const noexcept {
    try {
        auto writingPath = workspace_ / "input" / "writing" / jobId;
        auto readyPath = workspace_ / "input" / "ready" / jobId;
        
        // Atomic directory rename - the key operation
        std::filesystem::rename(writingPath, readyPath);
        return true;
    } catch (...) {
        return false;
    }
}

}