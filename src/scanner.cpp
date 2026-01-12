/*
 * nrvna ai - Asynchronous Inference Primitive
 * Copyright (c) 2025 Sanmathi Bharamgouda
 * SPDX-License-Identifier: MIT
 */

#include "nrvna/scanner.hpp"
#include "nrvna/logger.hpp"
#include <algorithm>

namespace nrvnaai {

Scanner::Scanner(const std::filesystem::path& workspace) noexcept 
    : workspace_(workspace), readyPath_(workspace / "input" / "ready") {
}

std::vector<JobId> Scanner::scan() const noexcept {
    std::vector<JobId> jobs;
    
    try {
        if (!std::filesystem::exists(readyPath_)) {
            LOG_DEBUG("Ready directory does not exist: " + readyPath_.string());
            return jobs;
        }

        for (const auto& entry : std::filesystem::directory_iterator(readyPath_)) {
            if (entry.is_directory()) {
                JobId jobId = extractJobId(entry.path());
                if (!jobId.empty() && isValidJobDirectory(entry.path())) {
                    jobs.push_back(jobId);
                    LOG_TRACE("Found job: " + jobId);
                }
            }
        }

        // Sort by job ID for consistent ordering (timestamp is in the ID)
        std::sort(jobs.begin(), jobs.end());
        
        if (!jobs.empty()) {
            LOG_DEBUG("Scanner found " + std::to_string(jobs.size()) + " ready jobs");
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR("Scanner error: " + std::string(e.what()));
    } catch (...) {
        LOG_ERROR("Unknown scanner error");
    }
    
    return jobs;
}

bool Scanner::hasNewJobs() const noexcept {
    try {
        if (!std::filesystem::exists(readyPath_)) {
            return false;
        }

        for (const auto& entry : std::filesystem::directory_iterator(readyPath_)) {
            if (entry.is_directory() && isValidJobDirectory(entry.path())) {
                return true;
            }
        }
    } catch (...) {
        // Swallow errors in this quick check
    }
    
    return false;
}

std::size_t Scanner::readyJobCount() const noexcept {
    std::size_t count = 0;
    
    try {
        if (!std::filesystem::exists(readyPath_)) {
            return 0;
        }

        for (const auto& entry : std::filesystem::directory_iterator(readyPath_)) {
            if (entry.is_directory() && isValidJobDirectory(entry.path())) {
                ++count;
            }
        }
    } catch (...) {
        // Return what we counted so far
    }
    
    return count;
}

bool Scanner::isValidJobDirectory(const std::filesystem::path& dir) const noexcept {
    try {
        // Must be a directory
        if (!std::filesystem::is_directory(dir)) {
            return false;
        }

        // Must contain prompt.txt file
        auto promptFile = dir / "prompt.txt";
        if (!std::filesystem::exists(promptFile) || !std::filesystem::is_regular_file(promptFile)) {
            LOG_DEBUG("Invalid job directory (missing prompt.txt): " + dir.string());
            return false;
        }

        // Prompt file must not be empty
        if (std::filesystem::file_size(promptFile) == 0) {
            LOG_DEBUG("Invalid job directory (empty prompt.txt): " + dir.string());
            return false;
        }

        return true;
    } catch (...) {
        return false;
    }
}

JobId Scanner::extractJobId(const std::filesystem::path& dir) const noexcept {
    try {
        return dir.filename().string();
    } catch (...) {
        return "";
    }
}

}