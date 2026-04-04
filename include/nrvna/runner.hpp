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
struct common_chat_templates;

namespace nrvnaai {

struct ModelInfo {
    bool        valid = false;
    std::string desc;                 // llama_model_desc() — display only, not for policy
    std::string arch;                 // general.architecture — "llama", "qwen2", "bert", etc.
    int         n_ctx_train = 0;      // training context length
    uint64_t    model_size_bytes = 0; // total parameter bytes
    bool        has_chat_template = false;
    bool        has_encoder = false;
    bool        has_decoder = true;
    int         n_embd_out = 0;       // embedding output dimension
};

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
    [[nodiscard]] EmbedResult embedVision(const std::string& prompt, const std::vector<std::filesystem::path>& imagePaths);
    [[nodiscard]] bool isMultimodal() const noexcept { return mtmd_ctx_ != nullptr; }

    // Probe GGUF metadata without starting a server — loads model briefly, returns info
    [[nodiscard]] static ModelInfo probeModelInfo(const std::string& modelPath);

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

    // GGUF sampling defaults — resolved once at model load, used as fallbacks in env_*() calls.
    // If GGUF has no value, these hold the hardcoded defaults.
    static float gguf_temp_;
    static int   gguf_top_k_;
    static float gguf_top_p_;
    static float gguf_min_p_;
    static float gguf_repeat_penalty_;
    static int   gguf_repeat_last_n_;

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

    // Per-instance chat templates (initialized at model load, destroyed with model)
    common_chat_templates* chat_templates_ = nullptr;
};

}
