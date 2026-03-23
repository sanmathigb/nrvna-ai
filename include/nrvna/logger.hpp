/*
 * nrvna ai - Asynchronous Inference Primitive
 * Copyright (c) 2025 Sanmathi Bharamgouda
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include <string>

namespace nrvnaai {

enum class LogLevel : uint8_t { 
    ERROR = 0, 
    WARN = 1, 
    INFO = 2, 
    DEBUG = 3, 
    TRACE = 4 
};

class Logger {
public:
    static void setLevel(LogLevel level) noexcept;
    static void initFromEnv() noexcept;
    [[nodiscard]] static LogLevel level() noexcept;
    
    static void log(LogLevel level, const std::string& message) noexcept;
    
    static void error(const std::string& msg) noexcept { log(LogLevel::ERROR, msg); }
    static void warn(const std::string& msg) noexcept { log(LogLevel::WARN, msg); }
    static void info(const std::string& msg) noexcept { log(LogLevel::INFO, msg); }
    static void debug(const std::string& msg) noexcept { log(LogLevel::DEBUG, msg); }
    static void trace(const std::string& msg) noexcept { log(LogLevel::TRACE, msg); }

private:
    static LogLevel parseEnvLevel() noexcept;
    static const char* levelToString(LogLevel level) noexcept;
};

// Thread naming for better logging context
void setThreadName(const std::string& name);
std::string getThreadName(int worker_id);

}

// Convenience macros for common usage
#define LOG_ERROR(msg) do { ::nrvnaai::Logger::error(msg); } while(0)
#define LOG_WARN(msg)  do { ::nrvnaai::Logger::warn(msg); } while(0)
#define LOG_INFO(msg)  do { ::nrvnaai::Logger::info(msg); } while(0)
#define LOG_DEBUG(msg) do { ::nrvnaai::Logger::debug(msg); } while(0)
#define LOG_TRACE(msg) do { ::nrvnaai::Logger::trace(msg); } while(0)
