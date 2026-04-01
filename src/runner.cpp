/*
 * nrvna ai - Asynchronous Inference Primitive
 * Copyright (c) 2025 Sanmathi Bharamgouda
 * SPDX-License-Identifier: MIT
 */

#include "nrvna/runner.hpp"
#include "nrvna/logger.hpp"
#include "llama_util.hpp"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include <chrono>
#include <thread>
#include <algorithm>
#include <regex>

namespace nrvnaai {

// Static member definitions (model is shared, mtmd context is per-worker)
std::shared_ptr<llama_model> Runner::shared_model_ = nullptr;
std::string Runner::current_model_path_ = "";
std::mutex Runner::model_mutex_;

// GGUF sampling defaults — hardcoded fallbacks until model is loaded
float Runner::gguf_temp_           = 0.8f;
int   Runner::gguf_top_k_          = 40;
float Runner::gguf_top_p_          = 0.9f;
float Runner::gguf_min_p_          = 0.05f;
float Runner::gguf_repeat_penalty_ = 1.1f;
int   Runner::gguf_repeat_last_n_  = 64;

// Vision encoding mutex - serializes mtmd_helper_eval_chunks across all workers
// because the underlying GGML compute graph has shared state that corrupts
// when multiple vision encodings run simultaneously
static std::mutex vision_encoding_mutex_;

// Strip <think>...</think> blocks from reasoning models (DeepSeek-R1, QwQ, Qwen3, etc.)
// Handles both closed (<think>...</think>) and unclosed (<think>... to end) blocks —
// unclosed blocks occur when the model exhausts n_predict tokens while still reasoning.
static std::string stripThinkBlocks(const std::string& text) {
    std::string result = text;
    size_t pos = 0;
    while ((pos = result.find("<think>", pos)) != std::string::npos) {
        size_t end = result.find("</think>", pos);
        if (end != std::string::npos) {
            // Closed block: remove <think>...</think> and trailing whitespace
            end += 8; // length of "</think>"
            while (end < result.size() && (result[end] == ' ' || result[end] == '\t' || result[end] == '\n' || result[end] == '\r'))
                ++end;
            result.erase(pos, end - pos);
        } else {
            // Unclosed block: model ran out of tokens mid-think, strip to end
            result.erase(pos);
        }
    }
    // Trim leading whitespace left behind
    size_t start = result.find_first_not_of(" \t\n\r");
    return (start == std::string::npos) ? "" : result.substr(start);
}

// Read GGUF metadata helpers
static std::string readModelStrMeta(const llama_model* model, const char* key) {
    char buf[256] = {};
    int32_t n = llama_model_meta_val_str(model, key, buf, sizeof(buf));
    return (n > 0) ? std::string(buf, static_cast<size_t>(n)) : std::string();
}

static float readModelFloatMeta(const llama_model* model, const char* key, float fallback) {
    char buf[64] = {};
    int32_t n = llama_model_meta_val_str(model, key, buf, sizeof(buf));
    if (n > 0) {
        try { return std::stof(std::string(buf, static_cast<size_t>(n))); }
        catch (...) {}
    }
    return fallback;
}

static int readModelIntMeta(const llama_model* model, const char* key, int fallback) {
    char buf[64] = {};
    int32_t n = llama_model_meta_val_str(model, key, buf, sizeof(buf));
    if (n > 0) {
        try { return std::stoi(std::string(buf, static_cast<size_t>(n))); }
        catch (...) {}
    }
    return fallback;
}

static void restrictModelToCpu(llama_model_params& params) {
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    static ggml_backend_dev_t cpu_only_devices[2] = { nullptr, nullptr };
    cpu_only_devices[0] = cpu_dev;
    cpu_only_devices[1] = nullptr;
    if (cpu_dev) {
        params.devices = cpu_only_devices;
    }
}

// Populate ModelInfo from a loaded model pointer
static ModelInfo buildModelInfoFromModel(const llama_model* model) {
    ModelInfo info;
    info.valid = true;

    char desc_buf[256] = {};
    llama_model_desc(model, desc_buf, sizeof(desc_buf));
    info.desc = desc_buf;

    info.arch = readModelStrMeta(model, "general.architecture");
    info.n_ctx_train = llama_model_n_ctx_train(model);
    info.model_size_bytes = llama_model_size(model);
    info.has_chat_template = (llama_model_chat_template(model, nullptr) != nullptr);
    info.has_encoder = llama_model_has_encoder(model);
    info.has_decoder = llama_model_has_decoder(model);
    info.n_embd_out = llama_model_n_embd_out(model);

    return info;
}

ModelInfo Runner::probeModelInfo(const std::string& modelPath) {
    llama_log_set(filtered_llama_log, nullptr);
    ggml_backend_load_all();

    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 0;  // CPU only for probing — fast, no GPU contention
    restrictModelToCpu(params);

    llama_model* model = llama_model_load_from_file(modelPath.c_str(), params);
    if (!model) {
        LOG_ERROR("Failed to probe model: " + modelPath);
        return ModelInfo{};
    }

    ModelInfo info = buildModelInfoFromModel(model);
    llama_model_free(model);
    return info;
}

Runner::Runner(const std::string& modelPath) : Runner(modelPath, "", 1) {
}

Runner::Runner(const std::string& modelPath, const std::string& mmprojPath, int numWorkers)
    : mmproj_path_(mmprojPath) {
    llama_log_set(filtered_llama_log, nullptr);
    ggml_backend_load_all();

    // Model loading is shared across workers (thread-safe)
    {
        std::lock_guard<std::mutex> lock(model_mutex_);
        if (!shared_model_ || current_model_path_ != modelPath) {
            LOG_INFO("Loading model: " + modelPath);

            llama_model_params model_params = llama_model_default_params();

            #if defined(__APPLE__)
                model_params.n_gpu_layers = env_int("NRVNA_GPU_LAYERS", 99);
            #else
                model_params.n_gpu_layers = env_int("NRVNA_GPU_LAYERS", 0);
            #endif
            if (model_params.n_gpu_layers <= 0) {
                restrictModelToCpu(model_params);
            }

            llama_model* model = llama_model_load_from_file(modelPath.c_str(), model_params);
            if (!model) {
                LOG_ERROR("Failed to load model: " + modelPath);
                throw std::runtime_error("Failed to load model: " + modelPath);
            }

            shared_model_ = std::shared_ptr<llama_model>(model, llama_model_free);
            current_model_path_ = modelPath;

            // Resolve GGUF sampling defaults once — log only when model provides a value
            auto resolveGgufFloat = [&](const char* key, float hardcoded, float& out) {
                float v = readModelFloatMeta(model, key, -1.0f);
                if (v >= 0.0f) {
                    LOG_INFO(std::string("Model sampling hint: ") + key + "=" + std::to_string(v));
                    out = v;
                } else {
                    out = hardcoded;
                }
            };
            auto resolveGgufInt = [&](const char* key, int hardcoded, int& out) {
                int v = readModelIntMeta(model, key, -1);
                if (v >= 0) {
                    LOG_INFO(std::string("Model sampling hint: ") + key + "=" + std::to_string(v));
                    out = v;
                } else {
                    out = hardcoded;
                }
            };
            resolveGgufFloat("general.sampling.temp",           0.8f,  gguf_temp_);
            resolveGgufInt  ("general.sampling.top_k",          40,    gguf_top_k_);
            resolveGgufFloat("general.sampling.top_p",          0.9f,  gguf_top_p_);
            resolveGgufFloat("general.sampling.min_p",          0.05f, gguf_min_p_);
            resolveGgufFloat("general.sampling.penalty_repeat", 1.1f,  gguf_repeat_penalty_);
            resolveGgufInt  ("general.sampling.penalty_last_n", 64,    gguf_repeat_last_n_);

            LOG_INFO("Model loaded successfully");
        }
    }

    // Each worker gets its own mtmd context (NOT thread-safe, so per-instance)
    // CRITICAL: Divide CPU threads among workers to prevent contention
    if (!mmprojPath.empty()) {
        LOG_INFO("Loading mmproj: " + mmprojPath);
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu = env_int("NRVNA_GPU_LAYERS", 0) > 0;

        // Divide threads among workers to prevent parallel vision corruption
        int total_threads = std::thread::hardware_concurrency();
        mparams.n_threads = std::max(1, total_threads / std::max(1, numWorkers));
        LOG_INFO("Vision threads per worker: " + std::to_string(mparams.n_threads) +
                 " (total: " + std::to_string(total_threads) + ", workers: " + std::to_string(numWorkers) + ")");
        mparams.print_timings = false;

        mtmd_context* ctx = mtmd_init_from_file(mmprojPath.c_str(), shared_model_.get(), mparams);
        if (!ctx) {
            LOG_WARN("Failed to load mmproj: " + mmprojPath + " - running in text-only mode");
            mtmd_ctx_ = nullptr;
        } else {
            mtmd_owned_ = std::shared_ptr<mtmd_context>(ctx, mtmd_free);
            mtmd_ctx_ = ctx;
            LOG_INFO("Multimodal support enabled");
        }
    } else {
        mtmd_ctx_ = nullptr;
    }
}

Runner::~Runner() {
    if (context_) {
        llama_free(context_);
        context_ = nullptr;
    }
}

Runner::SamplingConfig Runner::buildSamplingConfig() const {
    SamplingConfig config;
    const llama_model* model = shared_model_.get();
    const int n_ctx_train = llama_model_n_ctx_train(model);

    // Precedence: env var > GGUF metadata (cached at model load) > hardcoded default
    config.temp           = env_float("NRVNA_TEMP",           gguf_temp_);
    config.top_k          = env_int  ("NRVNA_TOP_K",          gguf_top_k_);
    config.top_p          = env_float("NRVNA_TOP_P",          gguf_top_p_);
    config.min_p          = env_float("NRVNA_MIN_P",          gguf_min_p_);
    config.repeat_penalty = env_float("NRVNA_REPEAT_PENALTY", gguf_repeat_penalty_);
    config.repeat_last_n  = env_int  ("NRVNA_REPEAT_LAST_N",  gguf_repeat_last_n_);
    config.seed = static_cast<uint32_t>(env_int("NRVNA_SEED", 0));

    config.max_ctx = std::min(n_ctx_train, env_int("NRVNA_MAX_CTX", 8192));
    config.n_predict = env_int("NRVNA_PREDICT", 2048);

    LOG_INFO("Model context: " + std::to_string(n_ctx_train) +
             ", using max_ctx=" + std::to_string(config.max_ctx) +
             ", n_predict=" + std::to_string(config.n_predict));

    return config;
}

void Runner::buildContextParams(int n_prompt, const SamplingConfig& config, llama_context_params& params) const {
    params = llama_context_default_params();
    params.n_ctx = std::min(n_prompt + config.n_predict + 64, config.max_ctx);
    params.n_batch = env_int("NRVNA_BATCH", 2048);  // Match reference CLI default
    params.no_perf = false;

    if (env_int("NRVNA_GPU_LAYERS", 0) <= 0) {
        params.offload_kqv = false;
        params.op_offload = false;
    }
}

llama_sampler* Runner::buildSampler(const SamplingConfig& config) const {
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler* smpl = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
        config.repeat_last_n,
        config.repeat_penalty,
        0.0f,
        0.0f
    ));

    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(config.top_k));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(config.top_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(config.min_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(config.temp));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(config.seed));

    return smpl;
}

RunResult Runner::run(const std::string& prompt) {
    return runText(prompt);
}

EmbedResult Runner::embed(const std::string& text) {
    if (!shared_model_) {
        return {false, {}, "Model not loaded"};
    }

    try {
        const llama_vocab* vocab = llama_model_get_vocab(shared_model_.get());

        // Tokenize input
        const int n_tokens = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, true, true);
        if (n_tokens <= 0) {
            return {false, {}, "Failed to tokenize input"};
        }

        std::vector<llama_token> tokens(n_tokens);
        if (llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), true, true) < 0) {
            return {false, {}, "Failed to tokenize input"};
        }

        // Create context with embedding mode enabled
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = tokens.size() + 1;
        ctx_params.n_batch = tokens.size();
        ctx_params.embeddings = true;
        ctx_params.pooling_type = LLAMA_POOLING_TYPE_MEAN;  // Mean pooling for sentence embeddings
        if (env_int("NRVNA_GPU_LAYERS", 0) <= 0) {
            ctx_params.offload_kqv = false;
            ctx_params.op_offload = false;
        }

        llama_context* ctx = llama_init_from_model(shared_model_.get(), ctx_params);
        if (!ctx) {
            return {false, {}, "Failed to create embedding context"};
        }

        // Create batch and decode
        llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
        if (llama_decode(ctx, batch) != 0) {
            llama_free(ctx);
            return {false, {}, "Failed to decode for embeddings"};
        }

        // Get embeddings
        float* emb = llama_get_embeddings_seq(ctx, 0);
        if (!emb) {
            // Fall back to getting embeddings from last token
            emb = llama_get_embeddings_ith(ctx, -1);
        }

        if (!emb) {
            llama_free(ctx);
            return {false, {}, "Failed to get embeddings"};
        }

        int n_embd = llama_model_n_embd(shared_model_.get());
        std::vector<float> embedding(emb, emb + n_embd);

        llama_free(ctx);

        LOG_INFO("Generated embedding with " + std::to_string(n_embd) + " dimensions");
        return {true, std::move(embedding), ""};

    } catch (const std::exception& e) {
        LOG_ERROR("Embedding error: " + std::string(e.what()));
        return {false, {}, "Embedding error: " + std::string(e.what())};
    }
}

EmbedResult Runner::embedVision(const std::string& prompt, const std::vector<std::filesystem::path>& imagePaths) {
    if (!shared_model_) {
        return {false, {}, "Model not loaded"};
    }

    if (!mtmd_ctx_) {
        return {false, {}, "Vision embedding requires --mmproj flag"};
    }

    if (imagePaths.empty()) {
        return {false, {}, "No images provided for vision embedding"};
    }

    mtmd_input_chunks* chunks = nullptr;
    std::vector<mtmd_bitmap*> bitmaps;
    llama_context* ctx = nullptr;
    try {
        const char* marker = mtmd_default_marker();
        std::string formatted_prompt = formatMultimodalPrompt(prompt, imagePaths.size(), marker);

        bitmaps = loadImages(imagePaths);
        if (bitmaps.empty()) {
            return {false, {}, "Failed to load image(s)"};
        }

        mtmd_input_text text;
        text.text = formatted_prompt.c_str();
        text.add_special = true;
        text.parse_special = true;

        chunks = mtmd_input_chunks_init();
        if (!chunks) {
            freeBitmaps(bitmaps);
            return {false, {}, "Failed to init image chunks"};
        }

        std::vector<const mtmd_bitmap*> bitmap_ptrs;
        bitmap_ptrs.reserve(bitmaps.size());
        for (auto* bmp : bitmaps) {
            bitmap_ptrs.push_back(bmp);
        }

        int32_t res = mtmd_tokenize(mtmd_ctx_, chunks, &text, bitmap_ptrs.data(), bitmap_ptrs.size());
        if (res != 0) {
            mtmd_input_chunks_free(chunks);
            chunks = nullptr;
            freeBitmaps(bitmaps);
            return {false, {}, "Failed to tokenize multimodal prompt"};
        }

        const int n_prompt = static_cast<int>(mtmd_helper_get_n_tokens(chunks));
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = std::max(n_prompt + 8, 128);
        ctx_params.n_batch = std::max(1, std::min(n_prompt, env_int("NRVNA_BATCH", 2048)));
        ctx_params.embeddings = true;
        ctx_params.pooling_type = LLAMA_POOLING_TYPE_MEAN;
        ctx_params.no_perf = false;
        if (env_int("NRVNA_GPU_LAYERS", 0) <= 0) {
            ctx_params.offload_kqv = false;
            ctx_params.op_offload = false;
        }

        ctx = llama_init_from_model(shared_model_.get(), ctx_params);
        if (!ctx) {
            mtmd_input_chunks_free(chunks);
            chunks = nullptr;
            freeBitmaps(bitmaps);
            return {false, {}, "Failed to create embedding context"};
        }

        llama_pos n_past = 0;
        {
            std::lock_guard<std::mutex> vision_lock(vision_encoding_mutex_);
            if (mtmd_helper_eval_chunks(mtmd_ctx_, ctx, chunks, 0, 0, ctx_params.n_batch, true, &n_past) != 0) {
                llama_free(ctx);
                mtmd_input_chunks_free(chunks);
                chunks = nullptr;
                freeBitmaps(bitmaps);
                return {false, {}, "Failed to eval multimodal prompt"};
            }
        }

        mtmd_input_chunks_free(chunks);
        chunks = nullptr;
        freeBitmaps(bitmaps);

        float* emb = llama_get_embeddings_seq(ctx, 0);
        if (!emb) {
            emb = llama_get_embeddings_ith(ctx, -1);
        }
        if (!emb) {
            llama_free(ctx);
            return {false, {}, "Failed to get multimodal embeddings"};
        }

        int n_embd = llama_model_n_embd_out(shared_model_.get());
        if (n_embd <= 0) {
            n_embd = llama_model_n_embd(shared_model_.get());
        }
        if (n_embd <= 0) {
            llama_free(ctx);
            return {false, {}, "Invalid embedding dimension"};
        }

        std::vector<float> embedding(emb, emb + n_embd);
        llama_free(ctx);

        LOG_INFO("Generated multimodal embedding with " + std::to_string(n_embd) +
                 " dimensions from " + std::to_string(imagePaths.size()) + " image(s)");
        return {true, std::move(embedding), ""};

    } catch (const std::exception& e) {
        if (ctx) {
            llama_free(ctx);
        }
        if (chunks) {
            mtmd_input_chunks_free(chunks);
        }
        freeBitmaps(bitmaps);
        LOG_ERROR("Vision embedding error: " + std::string(e.what()));
        return {false, {}, "Vision embedding error: " + std::string(e.what())};
    }
}

RunResult Runner::run(const std::string& prompt, const std::vector<std::filesystem::path>& imagePaths) {
    if (imagePaths.empty()) {
        return runText(prompt);
    }
    return runVision(prompt, imagePaths);
}

RunResult Runner::runText(const std::string& prompt) {
    if (!shared_model_) {
        return {false, "", "Model not loaded"};
    }
    
    llama_sampler* smpl = nullptr;
    try {
        SamplingConfig config = buildSamplingConfig();
        std::string formatted_prompt = formatPrompt(prompt);
        const llama_vocab* vocab = llama_model_get_vocab(shared_model_.get());
        const int n_prompt = -llama_tokenize(vocab, formatted_prompt.c_str(), formatted_prompt.size(), NULL, 0, true, true);
        if (n_prompt <= 0) {
            return {false, "", "Failed to tokenize input"};
        }

        int max_predict = config.max_ctx - n_prompt - 64;
        if (max_predict < 0) max_predict = 0;
        if (config.n_predict > max_predict) {
            config.n_predict = max_predict;
        }

        std::vector<llama_token> prompt_tokens(n_prompt);
        if (llama_tokenize(vocab, formatted_prompt.c_str(), formatted_prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
            return {false, "", "Failed to tokenize the prompt"};
        }

        llama_context_params ctx_params;
        buildContextParams(n_prompt, config, ctx_params);
        context_ = llama_init_from_model(shared_model_.get(), ctx_params);
        if (!context_) {
            return {false, "", "Failed to create context"};
        }

        LOG_DEBUG("Context: " + std::to_string(ctx_params.n_ctx) + " tokens");

        smpl = buildSampler(config);
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
        
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

        for (; n_pos + batch.n_tokens < n_prompt + config.n_predict; ) {
            if (llama_decode(context_, batch)) {
                LOG_ERROR("Failed to decode");
                break;
            }
            
            n_pos += batch.n_tokens;
            
            new_token_id = llama_sampler_sample(smpl, context_, -1);
            llama_sampler_accept(smpl, new_token_id);
            
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }
            
            char buf[128];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                LOG_ERROR("Failed to convert token to piece");
                break;
            }
            
            output.append(buf, n);
            
            batch = llama_batch_get_one(&new_token_id, 1);
        }

        llama_sampler_free(smpl);
        llama_free(context_);
        context_ = nullptr;
        
        LOG_INFO("Generated " + std::to_string(output.size()) + " bytes");
        output = stripThinkBlocks(output);
        return {true, output, ""};
        
    } catch (const std::exception& e) {
        if (smpl) {
            llama_sampler_free(smpl);
            smpl = nullptr;
        }
        if (context_) {
            llama_free(context_);
            context_ = nullptr;
        }
        LOG_ERROR("Inference error: " + std::string(e.what()));
        return {false, "", "Inference error: " + std::string(e.what())};
    }
}

RunResult Runner::runVision(const std::string& prompt, const std::vector<std::filesystem::path>& imagePaths) {
    if (!shared_model_) {
        return {false, "", "Model not loaded"};
    }

    if (!mtmd_ctx_) {
        return {false, "", "Vision job requires --mmproj flag"};
    }

    try {
        SamplingConfig config = buildSamplingConfig();

        // Lower temperature for vision tasks (more accurate OCR/descriptions)
        config.temp = env_float("NRVNA_VISION_TEMP", 0.3f);

        LOG_INFO("Vision job: " + std::to_string(imagePaths.size()) + " image(s), temp=" + std::to_string(config.temp));

        const char* marker = mtmd_default_marker();
        std::string formatted_prompt = formatMultimodalPrompt(prompt, imagePaths.size(), marker);

        auto loadStart = std::chrono::steady_clock::now();
        std::vector<mtmd_bitmap*> bitmaps = loadImages(imagePaths);
        if (bitmaps.empty()) {
            return {false, "", "Failed to load image(s)"};
        }
        auto loadTime = std::chrono::duration<double>(std::chrono::steady_clock::now() - loadStart).count();
        LOG_DEBUG("Image load time: " + std::to_string(loadTime) + "s");

        mtmd_input_text text;
        text.text = formatted_prompt.c_str();
        text.add_special = true;  // Add BOS token
        text.parse_special = true;

        mtmd_input_chunks* chunks = mtmd_input_chunks_init();
        if (!chunks) {
            freeBitmaps(bitmaps);
            return {false, "", "Failed to init image chunks"};
        }

        std::vector<const mtmd_bitmap*> bitmap_ptrs;
        bitmap_ptrs.reserve(bitmaps.size());
        for (auto* bmp : bitmaps) {
            bitmap_ptrs.push_back(bmp);
        }

        int32_t res = mtmd_tokenize(mtmd_ctx_, chunks, &text, bitmap_ptrs.data(), bitmap_ptrs.size());
        if (res != 0) {
            mtmd_input_chunks_free(chunks);
            freeBitmaps(bitmaps);
            return {false, "", "Failed to tokenize multimodal prompt"};
        }

        size_t n_prompt = mtmd_helper_get_n_tokens(chunks);
        int max_predict = config.max_ctx - static_cast<int>(n_prompt) - 64;
        if (max_predict < 0) max_predict = 0;
        if (config.n_predict > max_predict) {
            config.n_predict = max_predict;
        }
        llama_context_params ctx_params;
        buildContextParams(static_cast<int>(n_prompt), config, ctx_params);
        context_ = llama_init_from_model(shared_model_.get(), ctx_params);
        if (!context_) {
            mtmd_input_chunks_free(chunks);
            freeBitmaps(bitmaps);
            return {false, "", "Failed to create context"};
        }

        llama_sampler* smpl = buildSampler(config);
        llama_pos n_past = 0;

        // CRITICAL: Serialize vision encoding across all workers
        // The GGML compute graph has shared state that corrupts when multiple
        // vision encodings run simultaneously, even with separate mtmd contexts
        auto encodeStart = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> vision_lock(vision_encoding_mutex_);
            if (mtmd_helper_eval_chunks(mtmd_ctx_, context_, chunks, 0, 0, ctx_params.n_batch, true, &n_past) != 0) {
                llama_sampler_free(smpl);
                llama_free(context_);
                context_ = nullptr;
                mtmd_input_chunks_free(chunks);
                freeBitmaps(bitmaps);
                return {false, "", "Failed to eval multimodal prompt"};
            }
        }

        mtmd_input_chunks_free(chunks);
        freeBitmaps(bitmaps);

        auto encodeTime = std::chrono::duration<double>(std::chrono::steady_clock::now() - encodeStart).count();
        LOG_INFO("Vision encoding: " + std::to_string(encodeTime) + "s for " + std::to_string(n_past) + " tokens");

        // Token generation loop with explicit position tracking (matches reference)
        const llama_vocab* vocab = llama_model_get_vocab(shared_model_.get());
        std::string output;
        llama_token new_token_id;
        llama_batch batch = llama_batch_init(1, 0, 1);

        for (int i = 0; i < config.n_predict; ++i) {
            new_token_id = llama_sampler_sample(smpl, context_, -1);
            llama_sampler_accept(smpl, new_token_id);

            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            char buf[128];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                break;
            }
            output.append(buf, n);

            // Decode next token with explicit position (like reference)
            batch.n_tokens = 1;
            batch.token[0] = new_token_id;
            batch.pos[0] = n_past++;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = true;

            if (llama_decode(context_, batch)) {
                break;
            }
        }

        llama_batch_free(batch);

        llama_sampler_free(smpl);
        llama_free(context_);
        context_ = nullptr;

        LOG_INFO("Generated " + std::to_string(output.size()) + " bytes before strip");
        output = stripThinkBlocks(output);
        return {true, output, ""};

    } catch (const std::exception& e) {
        if (context_) {
            llama_free(context_);
            context_ = nullptr;
        }
        return {false, "", "Multimodal inference error: " + std::string(e.what())};
    }
}

std::string Runner::formatPrompt(const std::string& content) {
    // Apply chat template if model has one (instruct models)
    // Pass through raw if no template (base models)
    const char* tmpl = llama_model_chat_template(shared_model_.get(), nullptr);
    if (!tmpl) {
        return content;
    }

    llama_chat_message msg = {"user", content.c_str()};
    int len = llama_chat_apply_template(tmpl, &msg, 1, true, nullptr, 0);
    if (len < 0) {
        return content;
    }

    std::vector<char> buf(len + 1);
    int res = llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), buf.size());
    return (res > 0) ? std::string(buf.data(), res) : content;
}

std::string Runner::formatMultimodalPrompt(const std::string& prompt, size_t imageCount, const char* marker) {
    // Insert media markers if not present
    std::string content = prompt;
    if (prompt.find(marker) == std::string::npos) {
        // Prepend markers (image before text, like reference)
        std::string prefix;
        for (size_t i = 0; i < imageCount; ++i) {
            prefix += marker;
        }
        content = prefix + prompt;
    }

    // Apply chat template if model has one (instruct models)
    // Pass through raw if no template (base models)
    const char* tmpl = llama_model_chat_template(shared_model_.get(), nullptr);
    if (!tmpl) {
        return content;
    }

    llama_chat_message msg = {"user", content.c_str()};
    int len = llama_chat_apply_template(tmpl, &msg, 1, true, nullptr, 0);
    if (len < 0) {
        return content;
    }

    std::vector<char> buf(len + 1);
    int res = llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), buf.size());
    return (res > 0) ? std::string(buf.data(), res) : content;
}

std::vector<mtmd_bitmap*> Runner::loadImages(const std::vector<std::filesystem::path>& imagePaths) const {
    std::vector<mtmd_bitmap*> bitmaps;
    bitmaps.reserve(imagePaths.size());
    for (const auto& path : imagePaths) {
        mtmd_bitmap* bmp = mtmd_helper_bitmap_init_from_file(mtmd_ctx_, path.c_str());
        if (!bmp) {
            freeBitmaps(bitmaps);
            return {};
        }
        bitmaps.push_back(bmp);
    }
    return bitmaps;
}

void Runner::freeBitmaps(std::vector<mtmd_bitmap*>& bitmaps) const noexcept {
    for (auto* bmp : bitmaps) {
        mtmd_bitmap_free(bmp);
    }
    bitmaps.clear();
}

}
