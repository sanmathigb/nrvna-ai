/*
 * nrvna ai - Asynchronous Inference Primitive
 * Copyright (c) 2025 Sanmathi Bharamgouda
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include "nrvna/work.hpp"
#include "nrvna/types.hpp"
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace nrvnaai {

struct JobMeta {
    // Submission phase (written by Work)
    std::string submitted_at;
    std::string mode;           // "text", "embed", "vision", "tts"
    JobId parent;               // empty if none
    std::vector<std::string> tags;

    // Completion phase (written by Processor)
    std::string completed_at;
    double duration_s = -1.0;   // negative = not yet completed
    std::vector<std::string> artifacts;
    std::string status;         // "done" or "failed"
};

bool writeMetaJson(const std::filesystem::path& dir, const JobMeta& meta);
std::optional<JobMeta> readMetaJson(const std::filesystem::path& dir);

std::string formatTimestamp();
std::string jobTypeToString(JobType type);

} // namespace nrvnaai
