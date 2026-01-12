/*
 * nrvna ai - Asynchronous Inference Primitive
 * Copyright (c) 2025 Sanmathi Bharamgouda
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>

struct llama_model;
struct llama_context;

namespace nrvnaai {

struct RunResult {
    bool ok = false;
    std::string output;
    std::string error;
};

struct SamplingParams {
    float temperature = 0.8f;
    int top_k = 40;
    float top_p = 0.95f;
    float typical_p = 1.0f;
    float min_p = 0.05f;
    float repeat_penalty = 1.06f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    int repeat_last_n = 112;
};

enum class ModelType { 
    CODE,      // CodeLlama - needs low temp
    CHAT,      // Chat models - standard params  
    GENERAL,   // General models - balanced params
    EMBEDDING, // Embedding models - generate vectors not text
    UNKNOWN    // Fallback to safe defaults
};

class Runner final {
public:
    explicit Runner(const std::string& modelPath);
    explicit Runner(const std::string& modelPath, const SamplingParams& params);
    ~Runner();

    Runner(const Runner&) = delete;
    Runner& operator=(const Runner&) = delete;
    Runner(Runner&&) = delete;
    Runner& operator=(Runner&&) = delete;

    [[nodiscard]] RunResult run(const std::string& prompt);

private:
    // Shared model pattern for stability
    static std::shared_ptr<llama_model> shared_model_;
    static std::string current_model_path_;
    static std::mutex model_mutex_;
    static ModelType detected_model_type_;
    static std::atomic<bool> sampler_logged_;
    
    // Cached model metadata
    static std::string cached_model_name_;
    static std::string cached_architecture_;
    static bool cached_has_template_;
    static uint32_t cached_context_length_;

    [[nodiscard]] bool initializeModel(const std::string& modelPath) noexcept;
    void cleanup() noexcept;
    
    // Intelligence functions from stable implementation
    ModelType detectModelType(const std::string& modelPath);
    SamplingParams getCodeParams();
    SamplingParams getChatParams();
    SamplingParams getGeneralParams();
    int detectOptimalThreads();
    int detectGpuLayers(const std::string& modelPath);
    
    // Structured logging functions
    void logModelLoadSummary(int gpu_layers);
    void logSamplerConfigurationOnce();
    std::string modelTypeToString(ModelType type);
    std::string fmtFloat(float v);
    std::string formatPrompt(const std::string& content);

    llama_context* context_ = nullptr;
    SamplingParams sampling_params_;
};

}