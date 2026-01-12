/*
 * nrvna ai - Asynchronous Inference Primitive
 * Copyright (c) 2025 Sanmathi Bharamgouda
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include <filesystem>
#include <string>

#include "nrvna/types.hpp"

namespace nrvnaai {

enum class SubmissionError : uint8_t {
    None = 0,
    IoError,
    InvalidSize,
    InvalidContent,
    WorkspaceError
};

struct SubmitResult {
    bool ok = false;
    JobId id;
    SubmissionError error = SubmissionError::None;
    std::string message;
    explicit operator bool() const noexcept { return ok; }
};

class Work final {
public:
    explicit Work(const std::filesystem::path& workspace, bool createIfMissing = true);

    Work(const Work&) = delete;
    Work& operator=(const Work&) = delete;
    Work(Work&&) noexcept = default;
    Work& operator=(Work&&) noexcept = default;

    [[nodiscard]] SubmitResult submit(const std::string& prompt);

    void setMaxSize(std::size_t maxBytes) noexcept { maxBytes_ = maxBytes; }
    [[nodiscard]] std::size_t maxSize() const noexcept { return maxBytes_; }

private:
    std::filesystem::path workspace_;
    std::size_t maxBytes_ = 10'000'000; // 10MB

    [[nodiscard]] bool createWorkspace(bool createIfMissing) noexcept;
    [[nodiscard]] static JobId generateId() noexcept;
    [[nodiscard]] static bool isValidPrompt(const std::string& prompt) noexcept;
    
    [[nodiscard]] bool createJobDirectory(const JobId& jobId) const noexcept;
    [[nodiscard]] bool writePromptFile(const JobId& jobId, const std::string& prompt) const noexcept;
    [[nodiscard]] bool atomicPublish(const JobId& jobId) const noexcept;
};

}
