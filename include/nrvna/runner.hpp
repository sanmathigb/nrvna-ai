/*
 * nrvna ai - Asynchronous Inference Primitive
 * Copyright (c) 2025 Sanmathi Bharamgouda
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct llama_model;
struct llama_context;
struct llama_context_params;
struct llama_sampler;
struct mtmd_context;
struct mtmd_bitmap;

namespace nrvnaai {

struct RunResult {
    bool ok = false;
    std::string output;
    std::string error;
};

struct EmbedResult {
    bool ok = false;
    std::vector<float> embedding;
    std::string error;
};

class Runner final {
public:
    explicit Runner(const std::string& modelPath);
    explicit Runner(const std::string& modelPath, const std::string& mmprojPath, int numWorkers = 1);
    ~Runner();

    Runner(const Runner&) = delete;
    Runner& operator=(const Runner&) = delete;
    Runner(Runner&&) = delete;
    Runner& operator=(Runner&&) = delete;

    [[nodiscard]] RunResult run(const std::string& prompt);
    [[nodiscard]] RunResult run(const std::string& prompt, const std::vector<std::filesystem::path>& imagePaths);
    [[nodiscard]] EmbedResult embed(const std::string& text);
    [[nodiscard]] bool isMultimodal() const noexcept { return mtmd_ctx_ != nullptr; }

private:
    struct SamplingConfig {
        int n_predict = 0;
        int max_ctx = 0;
        float temp = 0.8f;
        int top_k = 40;
        float top_p = 0.9f;
        float min_p = 0.05f;
        float repeat_penalty = 1.1f;
        int repeat_last_n = 64;
        uint32_t seed = 0;
    };

    // Shared model (thread-safe), per-worker mtmd context (not thread-safe)
    static std::shared_ptr<llama_model> shared_model_;
    static std::string current_model_path_;
    static std::mutex model_mutex_;

    // Per-instance mtmd context for thread-safe vision processing
    std::shared_ptr<mtmd_context> mtmd_owned_;
    std::string mmproj_path_;

    [[nodiscard]] bool initializeModel(const std::string& modelPath) noexcept;
    void cleanup() noexcept;
    std::string formatPrompt(const std::string& content);
    std::string formatMultimodalPrompt(const std::string& prompt, size_t imageCount, const char* marker);
    SamplingConfig buildSamplingConfig() const;
    void buildContextParams(int n_prompt, const SamplingConfig& config, llama_context_params& params) const;
    llama_sampler* buildSampler(const SamplingConfig& config) const;
    RunResult runText(const std::string& prompt);
    RunResult runVision(const std::string& prompt, const std::vector<std::filesystem::path>& imagePaths);
    std::vector<mtmd_bitmap*> loadImages(const std::vector<std::filesystem::path>& imagePaths) const;
    void freeBitmaps(std::vector<mtmd_bitmap*>& bitmaps) const noexcept;

    llama_context* context_ = nullptr;
    mtmd_context* mtmd_ctx_ = nullptr;
};

}
