/*
 * nrvna ai - Asynchronous Inference Primitive
 * Copyright (c) 2025 Sanmathi Bharamgouda
 * SPDX-License-Identifier: MIT
 */

#include "nrvna/logger.hpp"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace nrvnaai {

static LogLevel g_level = LogLevel::INFO;
static std::mutex g_log_mutex;
static bool g_level_initialized = false;
static std::unordered_map<std::thread::id, std::string> g_thread_names;

void Logger::setLevel(LogLevel level) noexcept {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_level = level;
    g_level_initialized = true;
}

void Logger::initFromEnv() noexcept {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_level = parseEnvLevel();
    g_level_initialized = true;
}

LogLevel Logger::level() noexcept {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    if (!g_level_initialized) {
        g_level = parseEnvLevel();
        g_level_initialized = true;
    }
    return g_level;
}

void Logger::log(LogLevel level, const std::string& message) noexcept {
    try {
        if (static_cast<uint8_t>(level) > static_cast<uint8_t>(Logger::level())) {
            return; // Skip if below threshold
        }

        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        // Get thread name/ID
        std::string thread_info;
        {
            std::lock_guard<std::mutex> lock(g_log_mutex);
            auto tid = std::this_thread::get_id();
            auto it = g_thread_names.find(tid);
            if (it != g_thread_names.end()) {
                thread_info = it->second;
            } else {
                std::ostringstream oss;
                oss << "T" << tid;
                thread_info = oss.str();
            }
        }

        std::stringstream ss;
        ss << "[" << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        ss << "." << std::setfill('0') << std::setw(3) << ms.count() << "]";
        ss << " [" << levelToString(level) << "]";
        ss << " [" << thread_info << "]";
        ss << " " << message;

        {
            // All logs go to stderr - keep stdout pure for UI
            std::lock_guard<std::mutex> lock(g_log_mutex);
            std::cerr << ss.str() << std::endl;
        }
    } catch (...) {
        // Never throw from logging - would cause infinite loops
    }
}

LogLevel Logger::parseEnvLevel() noexcept {
    const char* env_val = std::getenv("NRVNA_LOG_LEVEL");
    if (!env_val) return LogLevel::INFO;

    // Case-insensitive comparison
    std::string level_str(env_val);
    for (char& c : level_str) {
        c = std::tolower(c);
    }

    if (level_str == "error") return LogLevel::ERROR;
    if (level_str == "warn" || level_str == "warning") return LogLevel::WARN;
    if (level_str == "info") return LogLevel::INFO;
    if (level_str == "debug") return LogLevel::DEBUG;
    if (level_str == "trace") return LogLevel::TRACE;
    
    return LogLevel::INFO; // Default fallback
}

const char* Logger::levelToString(LogLevel level) noexcept {
    switch (level) {
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::WARN:  return "WARN ";
        case LogLevel::INFO:  return "INFO ";
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::TRACE: return "TRACE";
        default: return "UNKN ";
    }
}

// Helper function to name threads for better logging
void setThreadName(const std::string& name) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_thread_names[std::this_thread::get_id()] = name;
}

std::string getThreadName(int worker_id) {
    return "Worker-" + std::to_string(worker_id);
}

}