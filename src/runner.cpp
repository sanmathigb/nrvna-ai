/*
 * nrvna ai - Asynchronous Inference Primitive
 * Copyright (c) 2025 Sanmathi Bharamgouda
 * SPDX-License-Identifier: MIT
 */

#include "nrvna/runner.hpp"
#include "nrvna/logger.hpp"
#include "llama.h"
#include <cstdlib>
#include <thread>
#include <algorithm>

namespace nrvnaai {

// Static member definitions
std::shared_ptr<llama_model> Runner::shared_model_ = nullptr;
std::string Runner::current_model_path_ = "";
std::mutex Runner::model_mutex_;

// Helper: Get integer from env with default
static int env_int(const char* name, int defv) {
    if (const char* v = std::getenv(name)) return std::atoi(v);
    return defv;
}

// Helper: Get float from env with default
static float env_float(const char* name, float defv) {
    if (const char* v = std::getenv(name)) return std::atof(v);
    return defv;
}

// Configurable llama.cpp log filtering - keep UI clean
static void filtered_llama_log(enum ggml_log_level level, const char* text, void* /*user_data*/) {
    // Skip progress dots and other noise for clean UI
    if (!text || text[0] == '.' || text[0] == '\n' || text[0] == '\0') {
        return;
    }

    static int filter_level = -1;

    if (filter_level == -1) {
        const char* env = std::getenv("LLAMA_LOG_LEVEL");
        filter_level = env ?
            (std::string(env) == "info" ? GGML_LOG_LEVEL_INFO :
             std::string(env) == "warn" ? GGML_LOG_LEVEL_WARN :
             std::string(env) == "debug" ? GGML_LOG_LEVEL_DEBUG :
             GGML_LOG_LEVEL_ERROR) : GGML_LOG_LEVEL_ERROR;
    }

    if (level >= filter_level) {
        fprintf(stderr, "%s", text);
    }
}

Runner::Runner(const std::string& modelPath) {
    llama_log_set(filtered_llama_log, nullptr);
    ggml_backend_load_all();

    std::lock_guard<std::mutex> lock(model_mutex_);

    // Load model only if different path or not loaded
    if (!shared_model_ || current_model_path_ != modelPath) {
        LOG_INFO("Loading model: " + modelPath);
        
        llama_model_params model_params = llama_model_default_params();
        
        // GPU layers from env, default to auto-detect Metal
        #if defined(__APPLE__)
            model_params.n_gpu_layers = env_int("NRVNA_GPU_LAYERS", 99);
        #else
            model_params.n_gpu_layers = env_int("NRVNA_GPU_LAYERS", 0);
        #endif
        
        llama_model* model = llama_model_load_from_file(modelPath.c_str(), model_params);
        if (!model) {
            LOG_ERROR("Failed to load model: " + modelPath);
            throw std::runtime_error("Failed to load model: " + modelPath);
        }

        shared_model_ = std::shared_ptr<llama_model>(model, llama_model_free);
        current_model_path_ = modelPath;
        
        LOG_INFO("Model loaded successfully");
    }
}

Runner::~Runner() {
    if (context_) {
        llama_free(context_);
        context_ = nullptr;
    }
}

RunResult Runner::run(const std::string& prompt) {
    if (!shared_model_) {
        return {false, "", "Model not loaded"};
    }
    
    try {
        // Get config from environment variables
        const int n_predict = env_int("NRVNA_PREDICT", 1024);
        const int n_ctx = env_int("NRVNA_CTX", 0); // 0 = auto-size
        const float temp = env_float("NRVNA_TEMP", 0.8f);        // llama.cpp default
        const int top_k = env_int("NRVNA_TOP_K", 40);            // llama.cpp default
        const float top_p = env_float("NRVNA_TOP_P", 0.95f);     // llama.cpp default
        const float min_p = env_float("NRVNA_MIN_P", 0.05f);     // llama.cpp default
        const float repeat_penalty = env_float("NRVNA_REPEAT_PENALTY", 1.0f);  // llama.cpp default (OFF)
        const int repeat_last_n = env_int("NRVNA_REPEAT_LAST_N", 64);
        const uint32_t seed = env_int("NRVNA_SEED", 0);
        
        // Log generation settings
        LOG_INFO("Generation settings: n_predict=" + std::to_string(n_predict) + 
                 " n_ctx=" + std::to_string(n_ctx) + 
                 " temp=" + std::to_string(temp) + 
                 " top_p=" + std::to_string(top_p) + 
                 " min_p=" + std::to_string(min_p));
        
        // Format prompt
        std::string formatted_prompt = formatPrompt(prompt);
        
        // Tokenize
        const llama_vocab* vocab = llama_model_get_vocab(shared_model_.get());
        const int n_prompt = -llama_tokenize(vocab, formatted_prompt.c_str(), formatted_prompt.size(), NULL, 0, true, true);
        if (n_prompt <= 0) {
            return {false, "", "Failed to tokenize input"};
        }
        
        std::vector<llama_token> prompt_tokens(n_prompt);
        if (llama_tokenize(vocab, formatted_prompt.c_str(), formatted_prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
            return {false, "", "Failed to tokenize the prompt"};
        }

        // Create context
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = n_ctx > 0 ? n_ctx : (n_prompt + n_predict + 1);
        ctx_params.n_batch = n_prompt;
        ctx_params.no_perf = false;

        context_ = llama_init_from_model(shared_model_.get(), ctx_params);
        if (!context_) {
            return {false, "", "Failed to create context"};
        }
        
        LOG_DEBUG("Context: " + std::to_string(ctx_params.n_ctx) + " tokens");

        // Build sampler chain (order matters!)
        auto sparams = llama_sampler_chain_default_params();
        sparams.no_perf = false;
        llama_sampler* smpl = llama_sampler_chain_init(sparams);
        
        // Add repetition penalty first (affects logits)
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
            repeat_last_n,      // last n tokens to penalize
            repeat_penalty,     // repeat penalty
            0.0f,              // frequency penalty
            0.0f               // presence penalty
        ));
        
        // Then sampling filters
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_min_p(min_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));

        // Prepare batch
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
        
        // Handle encoder models (if needed)
        llama_token decoder_start_token_id = 0;
        if (llama_model_has_encoder(shared_model_.get())) {
            if (llama_encode(context_, batch)) {
                LOG_ERROR("Failed to encode");
                llama_sampler_free(smpl);
                llama_free(context_);
                context_ = nullptr;
                return {false, "", "Failed to encode"};
            }
            
            decoder_start_token_id = llama_model_decoder_start_token(shared_model_.get());
            if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
                decoder_start_token_id = llama_vocab_bos(vocab);
            }
            
            batch = llama_batch_get_one(&decoder_start_token_id, 1);
        }
        
        std::string output;
        llama_token new_token_id;
        int n_pos = 0;

        // Main generation loop
        for (; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
            if (llama_decode(context_, batch)) {
                LOG_ERROR("Failed to decode");
                break;
            }
            
            n_pos += batch.n_tokens;
            
            // Sample next token
            new_token_id = llama_sampler_sample(smpl, context_, -1);
            llama_sampler_accept(smpl, new_token_id);
            
            // Check for end of generation
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }
            
            // Convert token to text
            char buf[128];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                LOG_ERROR("Failed to convert token to piece");
                break;
            }
            
            output.append(buf, n);
            
            // Prepare next batch
            batch = llama_batch_get_one(&new_token_id, 1);
        }

        // Cleanup
        llama_sampler_free(smpl);
        llama_free(context_);
        context_ = nullptr;
        
        LOG_INFO("Generated " + std::to_string(output.size()) + " bytes");
        return {true, output, ""};
        
    } catch (const std::exception& e) {
        if (context_) {
            llama_free(context_);
            context_ = nullptr;
        }
        LOG_ERROR("Inference error: " + std::string(e.what()));
        return {false, "", "Inference error: " + std::string(e.what())};
    }
}

std::string Runner::formatPrompt(const std::string& content) {
    // Single-shot prompt formatting using model's template
    const char* tmpl = llama_model_chat_template(shared_model_.get(), nullptr);
    if (!tmpl) return content;
    
    llama_chat_message msg = {"user", content.c_str()};
    
    // First pass: get required length
    int len = llama_chat_apply_template(tmpl, &msg, 1, true, nullptr, 0);
    if (len < 0) return content; // Error or no template support
    
    // Second pass: format into buffer
    std::vector<char> buf(len + 1);
    int res = llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), buf.size());
    
    return (res > 0) ? std::string(buf.data(), res) : content;
}

}
