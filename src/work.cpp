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
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <sys/stat.h>
#include <unistd.h>

namespace nrvnaai {

namespace {
std::size_t env_size(const char* name, std::size_t defv) {
    const char* val = std::getenv(name);
    if (!val || !*val) {
        return defv;
    }
    try {
        std::size_t parsed = static_cast<std::size_t>(std::stoull(val));
        return parsed == 0 ? defv : parsed;
    } catch (...) {
        return defv;
    }
}

std::string toLowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool sameFilesystem(const std::filesystem::path& src, const std::filesystem::path& destDir) {
    struct stat src_stat;
    struct stat dest_stat;
    if (::stat(src.c_str(), &src_stat) != 0) {
        return false;
    }
    if (::stat(destDir.c_str(), &dest_stat) != 0) {
        return false;
    }
    return src_stat.st_dev == dest_stat.st_dev;
}

bool validateImagePath(const std::filesystem::path& path, SubmissionError& code, std::string& error) {
    if (!std::filesystem::exists(path)) {
        error = "Image file not found: " + path.string();
        code = SubmissionError::InvalidContent;
        return false;
    }
    if (!std::filesystem::is_regular_file(path)) {
        error = "Image path is not a file: " + path.string();
        code = SubmissionError::InvalidContent;
        return false;
    }
    std::string ext = toLowerCopy(path.extension().string());
    if (ext.empty()) {
        error = "Image file has no extension: " + path.string();
        code = SubmissionError::InvalidContent;
        return false;
    }
    static const std::vector<std::string> valid_ext = {".jpg", ".jpeg", ".png", ".gif", ".webp"};
    if (std::find(valid_ext.begin(), valid_ext.end(), ext) == valid_ext.end()) {
        error = "Unsupported image extension: " + path.string();
        code = SubmissionError::InvalidContent;
        return false;
    }
    std::size_t max_bytes = env_size("NRVNA_MAX_IMAGE_SIZE", 50ULL * 1024 * 1024);
    std::error_code ec;
    auto size = std::filesystem::file_size(path, ec);
    if (ec) {
        error = "Failed to read image size: " + path.string();
        code = SubmissionError::IoError;
        return false;
    }
    if (size > max_bytes) {
        error = "Image exceeds size limit (" + std::to_string(max_bytes) + " bytes): " + path.string();
        code = SubmissionError::InvalidSize;
        return false;
    }
    return true;
}
}

Work::Work(const std::filesystem::path& workspace, bool createIfMissing)
    : workspace_(workspace) {
    if (!createWorkspace(createIfMissing)) {
        LOG_ERROR("Failed to initialize workspace: " + workspace_.string());
    }
}

SubmitResult Work::submit(const std::string& prompt, JobType type) {
    if (!isValidPrompt(prompt)) {
        if (prompt.empty()) {
            LOG_DEBUG("Invalid prompt: empty");
            return {false, "", SubmissionError::InvalidContent, "Prompt is empty"};
        } else {
            LOG_DEBUG("Prompt exceeds size limit: " + std::to_string(prompt.size()) + " > " + std::to_string(maxBytes_));
            return {false, "", SubmissionError::InvalidSize, "Prompt exceeds maximum size limit (" + std::to_string(maxBytes_) + " bytes)"};
        }
    }

    JobId jobId = generateId();
    LOG_DEBUG("Generated job ID: " + jobId);

    if (!createJobDirectory(jobId)) {
        LOG_ERROR("Failed to create job directory for: " + jobId);
        return {false, "", SubmissionError::IoError, "Failed to create job directory"};
    }

    if (!writePromptFile(jobId, prompt)) {
        LOG_ERROR("Failed to write prompt file for: " + jobId);
        cleanupFailedJob(jobId);
        return {false, "", SubmissionError::IoError, "Failed to write prompt file"};
    }

    if (type != JobType::Text && !writeTypeFile(jobId, type)) {
        LOG_ERROR("Failed to write type file for: " + jobId);
        cleanupFailedJob(jobId);
        return {false, "", SubmissionError::IoError, "Failed to write type file"};
    }

    if (!atomicPublish(jobId)) {
        LOG_ERROR("Failed to publish job: " + jobId);
        cleanupFailedJob(jobId);
        return {false, "", SubmissionError::IoError, "Failed to publish job"};
    }

    LOG_INFO("Job submitted successfully: " + jobId);
    return {true, jobId, SubmissionError::None, ""};
}

SubmitResult Work::submit(const std::string& prompt, const std::vector<std::filesystem::path>& imagePaths) {
    if (!isValidPrompt(prompt)) {
        if (prompt.empty()) {
            LOG_DEBUG("Invalid prompt: empty");
            return {false, "", SubmissionError::InvalidContent, "Prompt is empty"};
        } else {
            LOG_DEBUG("Prompt exceeds size limit: " + std::to_string(prompt.size()) + " > " + std::to_string(maxBytes_));
            return {false, "", SubmissionError::InvalidSize, "Prompt exceeds maximum size limit (" + std::to_string(maxBytes_) + " bytes)"};
        }
    }

    if (!imagePaths.empty()) {
        for (const auto& path : imagePaths) {
            std::string error;
            SubmissionError code = SubmissionError::None;
            if (!validateImagePath(path, code, error)) {
                LOG_ERROR(error);
                return {false, "", code, error};
            }
        }
    }

    JobId jobId = generateId();
    LOG_DEBUG("Generated job ID: " + jobId);

    if (!createJobDirectory(jobId)) {
        LOG_ERROR("Failed to create job directory for: " + jobId);
        return {false, "", SubmissionError::IoError, "Failed to create job directory"};
    }

    if (!writePromptFile(jobId, prompt)) {
        LOG_ERROR("Failed to write prompt file for: " + jobId);
        cleanupFailedJob(jobId);
        return {false, "", SubmissionError::IoError, "Failed to write prompt file"};
    }

    if (!imagePaths.empty()) {
        if (!writeImageFiles(jobId, imagePaths)) {
            LOG_ERROR("Failed to write image files for: " + jobId);
            cleanupFailedJob(jobId);
            return {false, "", SubmissionError::IoError, "Failed to write image files"};
        }
        // Write vision type for jobs with images
        if (!writeTypeFile(jobId, JobType::Vision)) {
            LOG_ERROR("Failed to write type file for: " + jobId);
            cleanupFailedJob(jobId);
            return {false, "", SubmissionError::IoError, "Failed to write type file"};
        }
    }

    if (!atomicPublish(jobId)) {
        LOG_ERROR("Failed to publish job: " + jobId);
        cleanupFailedJob(jobId);
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

        std::filesystem::create_directories(workspace_ / "input" / "writing");
        std::filesystem::create_directories(workspace_ / "input" / "ready");
        std::filesystem::create_directories(workspace_ / "processing");
        std::filesystem::create_directories(workspace_ / "output");
        std::filesystem::create_directories(workspace_ / "failed");

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to create workspace: " + std::string(e.what()));
        return false;
    } catch (...) {
        LOG_ERROR("Unknown error creating workspace");
        return false;
    }
}

JobId Work::generateId() {
    static std::atomic<uint64_t> counter{0};

    auto now = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    uint64_t unique_counter = counter.fetch_add(1);

    std::stringstream ss;
    ss << std::to_string(now) << "_" << getpid() << "_" << unique_counter;
    return ss.str();
}

bool Work::isValidPrompt(const std::string& prompt) const noexcept {
    return !prompt.empty() && prompt.size() <= maxBytes_;
}

bool Work::createJobDirectory(const JobId& jobId) const noexcept {
    try {
        auto jobPath = workspace_ / "input" / "writing" / jobId;
        std::filesystem::create_directories(jobPath);
        return std::filesystem::exists(jobPath) && std::filesystem::is_directory(jobPath);
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

bool Work::writeImageFiles(const JobId& jobId, const std::vector<std::filesystem::path>& imagePaths) const noexcept {
    try {
        auto jobPath = workspace_ / "input" / "writing" / jobId;
        auto imagesDir = jobPath / "images";
        std::filesystem::create_directories(imagesDir);

        int idx = 0;
        for (const auto& srcPath : imagePaths) {
            if (!std::filesystem::exists(srcPath)) {
                LOG_ERROR("Image file not found: " + srcPath.string());
                return false;
            }
            std::string ext = srcPath.extension().string();
            std::string destFilename = "image_" + std::to_string(idx) + ext;
            auto destPath = imagesDir / destFilename;
            std::error_code ec;
            if (sameFilesystem(srcPath, imagesDir)) {
                // Use absolute path so symlink survives job directory moves
                std::filesystem::create_symlink(std::filesystem::absolute(srcPath), destPath, ec);
                if (ec) {
                    ec.clear();
                    std::filesystem::copy_file(srcPath, destPath, std::filesystem::copy_options::overwrite_existing, ec);
                }
            } else {
                std::filesystem::copy_file(srcPath, destPath, std::filesystem::copy_options::overwrite_existing, ec);
            }
            if (ec) {
                LOG_ERROR("Failed to write image file: " + srcPath.string());
                return false;
            }
            ++idx;
        }

        return true;
    } catch (...) {
        return false;
    }
}

bool Work::writeTypeFile(const JobId& jobId, JobType type) const noexcept {
    try {
        auto typePath = workspace_ / "input" / "writing" / jobId / "type.txt";
        std::ofstream file(typePath, std::ios::binary);
        if (!file) return false;

        switch (type) {
            case JobType::Embed:
                file << "embed";
                break;
            case JobType::Vision:
                file << "vision";
                break;
            default:
                file << "text";
                break;
        }
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

        std::filesystem::rename(writingPath, readyPath);
        return true;
    } catch (...) {
        return false;
    }
}

void Work::cleanupFailedJob(const JobId& jobId) const noexcept {
    std::error_code ec;
    std::filesystem::remove_all(workspace_ / "input" / "writing" / jobId, ec);
}

}
