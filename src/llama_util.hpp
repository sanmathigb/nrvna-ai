/*
 * nrvna ai - Shared llama.cpp utilities (internal)
 * Copyright (c) 2025 Sanmathi Bharamgouda
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <string>

namespace nrvnaai {

// Get integer from env with default
inline int env_int(const char* name, int defv) {
    if (const char* v = std::getenv(name)) return std::atoi(v);
    return defv;
}

// Get float from env with default
inline float env_float(const char* name, float defv) {
    if (const char* v = std::getenv(name)) return std::atof(v);
    return defv;
}

// Configurable llama.cpp log filtering — keep UI clean
inline void filtered_llama_log(enum ggml_log_level level, const char* text, void* /*user_data*/) {
    if (!text || text[0] == '.' || text[0] == '\n' || text[0] == '\0') return;

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

} // namespace nrvnaai
