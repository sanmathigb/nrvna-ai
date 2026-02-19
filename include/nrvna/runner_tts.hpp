/*
 * nrvna ai - Asynchronous Inference Primitive
 * Copyright (c) 2025 Sanmathi Bharamgouda
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct llama_model;

namespace nrvnaai {

enum class TtsVersion { V0_2, V0_3 };

struct TtsResult {
    bool ok = false;
    std::vector<float> audio;
    int sample_rate = 24000;
    std::string error;
};

class TtsRunner final {
public:
    explicit TtsRunner(const std::string& modelPath, const std::string& vocoderPath);
    ~TtsRunner();

    TtsRunner(const TtsRunner&) = delete;
    TtsRunner& operator=(const TtsRunner&) = delete;
    TtsRunner(TtsRunner&&) = delete;
    TtsRunner& operator=(TtsRunner&&) = delete;

    [[nodiscard]] TtsResult run(const std::string& text);

private:
    static std::shared_ptr<llama_model> shared_tts_model_;
    static std::shared_ptr<llama_model> shared_vocoder_;
    static std::string current_tts_model_path_;
    static std::string current_vocoder_path_;
    static std::mutex tts_model_mutex_;

    static TtsVersion detected_version_;
    static std::string v3_audio_text_;
    static std::string v3_audio_data_;
};

} // namespace nrvnaai
