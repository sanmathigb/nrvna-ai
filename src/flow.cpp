/*
 * nrvna ai - Asynchronous Inference Primitive
 * Copyright (c) 2025 Sanmathi Bharamgouda
 * SPDX-License-Identifier: MIT
 */

#include "nrvna/flow.hpp"
#include "nrvna/logger.hpp"
#include <fstream>
#include <algorithm>

namespace nrvnaai {

Flow::Flow(const std::filesystem::path& workspace) noexcept 
    : workspace_(workspace) {
    LOG_DEBUG("Flow created for workspace: " + workspace_.string());
}

std::optional<Job> Flow::latest() const noexcept {
    try {
        std::vector<Job> jobs = list(1);
        if (jobs.empty()) {
            return std::nullopt;
        }
        return jobs[0];
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<Job> Flow::get(const JobId& id) const noexcept {
    try {
        Status jobStatus = status(id);
        
        if (jobStatus == Status::Done) {
            auto outputDir = workspace_ / "output" / id;
            auto resultFile = outputDir / "result.txt";
            
            if (!std::filesystem::exists(resultFile)) {
                LOG_DEBUG("Result file not found for job: " + id);
                return std::nullopt;
            }
            
            std::string content = readResultContent(id);
            auto timestamp = std::filesystem::last_write_time(outputDir);
            auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                timestamp - std::filesystem::file_time_type::clock::now() + std::chrono::system_clock::now());
            
            return Job{id, Status::Done, content, sctp};
            
        } else if (jobStatus == Status::Failed) {
            auto failedDir = workspace_ / "failed" / id;
            auto errorFile = failedDir / "error.txt";
            
            std::string errorContent = "";
            if (std::filesystem::exists(errorFile)) {
                std::ifstream file(errorFile);
                std::string line;
                while (std::getline(file, line)) {
                    errorContent += line + "\n";
                }
            }
            
            auto timestamp = std::filesystem::last_write_time(failedDir);
            auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                timestamp - std::filesystem::file_time_type::clock::now() + std::chrono::system_clock::now());
            return Job{id, Status::Failed, errorContent, sctp};
            
        } else {
            auto sctp = std::chrono::system_clock::now();
            return Job{id, jobStatus, "", sctp};
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR("Error retrieving job " + id + ": " + e.what());
        return std::nullopt;
    }
}

std::vector<Job> Flow::list(std::size_t max) const noexcept {
    std::vector<Job> jobs;
    try {
        // Check output directory for completed jobs
        auto outputDir = workspace_ / "output";
        if (std::filesystem::exists(outputDir)) {
            for (const auto& entry : std::filesystem::directory_iterator(outputDir)) {
                if (entry.is_directory()) {
                    auto timestamp = std::filesystem::last_write_time(entry);
                    auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                        timestamp - std::filesystem::file_time_type::clock::now() + std::chrono::system_clock::now());
                    jobs.push_back({entry.path().filename().string(), Status::Done, "", sctp});
                }
            }
        }
        
        // Check failed directory for failed jobs  
        auto failedDir = workspace_ / "failed";
        if (std::filesystem::exists(failedDir)) {
            for (const auto& entry : std::filesystem::directory_iterator(failedDir)) {
                if (entry.is_directory()) {
                    auto timestamp = std::filesystem::last_write_time(entry);
                    auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                        timestamp - std::filesystem::file_time_type::clock::now() + std::chrono::system_clock::now());
                    jobs.push_back({entry.path().filename().string(), Status::Failed, "", sctp});
                }
            }
        }
        
        // Sort by timestamp (newest first)
        std::sort(jobs.begin(), jobs.end(), [](const Job& a, const Job& b) {
            return a.timestamp > b.timestamp;
        });
        
        // Limit to max
        if (jobs.size() > max) {
            jobs.resize(max);
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR("Error listing jobs: " + std::string(e.what()));
    }
    
    return jobs;
}

Status Flow::status(const JobId& id) const noexcept {
    try {
        // Check in order of completion states
        if (std::filesystem::exists(workspace_ / "output" / id)) {
            return Status::Done;
        }
        if (std::filesystem::exists(workspace_ / "failed" / id)) {
            return Status::Failed;
        }
        if (std::filesystem::exists(workspace_ / "processing" / id)) {
            return Status::Running;
        }
        if (std::filesystem::exists(workspace_ / "pending" / id)) {
            return Status::Queued;
        }
        
        return Status::Missing;
        
    } catch (...) {
        return Status::Missing;
    }
}

bool Flow::exists(const JobId& id) const noexcept {
    return status(id) != Status::Missing;
}

std::optional<std::string> Flow::error(const JobId& id) const {
    try {
        auto errorFile = workspace_ / "failed" / id / "error.txt";
        if (!std::filesystem::exists(errorFile)) {
            return std::nullopt;
        }
        
        std::ifstream file(errorFile);
        std::string content, line;
        while (std::getline(file, line)) {
            content += line + "\n";
        }
        return content;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Error reading error file for job " + id + ": " + e.what());
        return std::nullopt;
    }
}

std::optional<std::string> Flow::prompt(const JobId& id) const {
    try {
        // Check all possible directories for the prompt file
        std::vector<std::filesystem::path> searchDirs = {
            workspace_ / "output" / id,
            workspace_ / "failed" / id,
            workspace_ / "processing" / id,
            workspace_ / "pending" / id
        };
        
        for (const auto& dir : searchDirs) {
            auto promptFile = dir / "prompt.txt";
            if (std::filesystem::exists(promptFile)) {
                std::ifstream file(promptFile);
                std::string content, line;
                while (std::getline(file, line)) {
                    content += line + "\n";
                }
                return content;
            }
        }
        
        return std::nullopt;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Error reading prompt file for job " + id + ": " + e.what());
        return std::nullopt;
    }
}

std::string Flow::readResultContent(const JobId& id) const noexcept {
    try {
        auto resultFile = workspace_ / "output" / id / "result.txt";
        std::ifstream file(resultFile);
        std::string content, line;
        while (std::getline(file, line)) {
            content += line + "\n";
        }
        return content;
    } catch (...) {
        return "";
    }
}

}