/*
 * nrvna ai - Asynchronous Inference Primitive
 * Copyright (c) 2025 Sanmathi Bharamgouda
 * SPDX-License-Identifier: MIT
 */

#pragma once
#include <filesystem>
#include <optional>
#include <string>
#include <vector>
#include <chrono>

#include "nrvna/types.hpp"

namespace nrvnaai {

struct Job {
    JobId id;
    Status status;
    std::string content;
    std::chrono::system_clock::time_point timestamp;
};

class Flow {
public:
    explicit Flow(const std::filesystem::path& workspace) noexcept;

    Flow(const Flow&) = delete;
    Flow& operator=(const Flow&) = delete;
    Flow(Flow&&) noexcept = default;
    Flow& operator=(Flow&&) noexcept = default;

    [[nodiscard]] std::optional<Job> latest() const noexcept;
    [[nodiscard]] std::optional<Job> get(const JobId& id) const noexcept;
    [[nodiscard]] std::vector<Job> list(std::size_t max = 10) const noexcept;
    [[nodiscard]] Status status(const JobId& id) const noexcept;

    [[nodiscard]] bool exists(const JobId& id) const noexcept;
    [[nodiscard]] std::optional<std::string> error(const JobId& id) const;
    [[nodiscard]] std::optional<std::string> prompt(const JobId& id) const;

private:
    std::filesystem::path workspace_;
    
    [[nodiscard]] std::string readResultContent(const JobId& id) const;
};

}






