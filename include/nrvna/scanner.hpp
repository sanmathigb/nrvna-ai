/*
 * nrvna ai - Asynchronous Inference Primitive
 * Copyright (c) 2025 Sanmathi Bharamgouda
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include <filesystem>
#include <vector>

#include "nrvna/types.hpp"

namespace nrvnaai {

class Scanner {
public:
    explicit Scanner(const std::filesystem::path& workspace) noexcept;
    
    Scanner(const Scanner&) = delete;
    Scanner& operator=(const Scanner&) = delete;
    Scanner(Scanner&&) noexcept = default;
    Scanner& operator=(Scanner&&) noexcept = default;

    [[nodiscard]] std::vector<JobId> scan() const noexcept;
    [[nodiscard]] bool hasNewJobs() const noexcept;
    [[nodiscard]] std::size_t readyJobCount() const noexcept;

private:
    std::filesystem::path workspace_;
    std::filesystem::path readyPath_;
    
    [[nodiscard]] bool isValidJobDirectory(const std::filesystem::path& dir) const noexcept;
    [[nodiscard]] JobId extractJobId(const std::filesystem::path& dir) const noexcept;
};

}