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
#define LOG_ERROR(msg) ::nrvnaai::Logger::error(msg)
#define LOG_WARN(msg)  ::nrvnaai::Logger::warn(msg)  
#define LOG_INFO(msg)  ::nrvnaai::Logger::info(msg)
#define LOG_DEBUG(msg) ::nrvnaai::Logger::debug(msg)
#define LOG_TRACE(msg) ::nrvnaai::Logger::trace(msg)