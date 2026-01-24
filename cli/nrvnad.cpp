/*
 * nrvna ai - Server daemon (nrvnad)
 * Copyright (c) 2025 Sanmathi Bharamgouda
 * SPDX-License-Identifier: MIT
 */

#include "nrvna/server.hpp"
#include "nrvna/logger.hpp"
#include <iostream>
#include <chrono>
#include <thread>
#include <csignal>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace nrvnaai;

constexpr const char* VERSION = "0.1.0";

// Async-signal-safe: only set flag, no complex operations
static volatile sig_atomic_t g_shutdown_requested = 0;

void signalHandler(int signal) {
    (void)signal;
    g_shutdown_requested = 1;
}

void printBanner() {
    std::cout << "\n\n\n";
    std::cout << "                    \033[1m███╗   ██╗██████╗ ██╗   ██╗███╗   ██╗ █████╗ \033[0m\n";
    std::cout << "                    \033[1m████╗  ██║██╔══██╗██║   ██║████╗  ██║██╔══██╗\033[0m\n";
    std::cout << "                    \033[1m██╔██╗ ██║██████╔╝██║   ██║██╔██╗ ██║███████║\033[0m\n";
    std::cout << "                    \033[1m██║╚██╗██║██╔══██╗╚██╗ ██╔╝██║╚██╗██║██╔══██║\033[0m\n";
    std::cout << "                    \033[1m██║ ╚████║██║  ██║ ╚████╔╝ ██║ ╚████║██║  ██║\033[0m\n";
    std::cout << "                    \033[1m╚═╝  ╚═══╝╚═╝  ╚═╝  ╚═══╝  ╚═╝  ╚═══╝╚═╝  ╚═╝\033[0m\n";
    std::cout << "\n";
    std::cout << "                         \033[90masync  ·  inference  ·  primitive\033[0m\n";
    std::cout << "\n\n";
}

void printUsage(const char* progName) {
    (void)progName;
    printBanner();

    std::cout << "  \033[1mUsage:\033[0m nrvnad <model> <workspace> [workers]\n";
    std::cout << "         nrvnad --help | --version\n\n";
    std::cout << "  \033[1mArguments:\033[0m\n";
    std::cout << "    model       Path to .gguf or model name (e.g., mistral, qwen)\n";
    std::cout << "    workspace   Directory for job storage\n";
    std::cout << "    workers     Number of worker threads (default: 4, max: 64)\n\n";
    std::cout << "  \033[1mOptions:\033[0m\n";
    std::cout << "    -h, --help     Show this help message\n";
    std::cout << "    -v, --version  Show version\n\n";
    std::cout << "  \033[1mWorkflow:\033[0m\n";
    std::cout << "    1. Start daemon:  nrvnad mistral ./workspace\n";
    std::cout << "    2. Submit prompt: wrk ./workspace \"Hello\"\n";
    std::cout << "    3. Get result:    flw ./workspace -w <job_id>\n\n";
}

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool containsToken(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

std::optional<std::filesystem::path> resolveModelPath(const std::string& modelArg) {
    std::filesystem::path candidate(modelArg);
    if (std::filesystem::exists(candidate)) {
        return candidate;
    }

    std::filesystem::path modelsDir = std::filesystem::current_path() / "models";
    if (!std::filesystem::exists(modelsDir)) {
        return std::nullopt;
    }

    std::string needle = toLower(modelArg);
    std::vector<std::filesystem::path> matches;

    for (const auto& entry : std::filesystem::directory_iterator(modelsDir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto& path = entry.path();
        if (path.extension() != ".gguf") {
            continue;
        }
        std::string filename = toLower(path.filename().string());
        if (containsToken(filename, "mmproj")) {
            continue;
        }
        if (containsToken(filename, needle)) {
            matches.push_back(path);
        }
    }

    if (matches.empty()) {
        return std::nullopt;
    }

    std::sort(matches.begin(), matches.end());
    return matches.front();
}

std::optional<std::filesystem::path> resolveMmprojPath(const std::filesystem::path& modelPath) {
    std::filesystem::path dir = modelPath.parent_path();
    if (dir.empty() || !std::filesystem::exists(dir)) {
        return std::nullopt;
    }

    std::string stem = toLower(modelPath.stem().string());
    std::vector<std::filesystem::path> matches;

    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto& path = entry.path();
        if (path.extension() != ".gguf") {
            continue;
        }
        std::string filename = toLower(path.filename().string());
        if (containsToken(filename, "mmproj") && (stem.empty() || containsToken(filename, stem))) {
            matches.push_back(path);
        }
    }

    if (matches.empty()) {
        return std::nullopt;
    }

    std::sort(matches.begin(), matches.end());
    return matches.front();
}

void applyDefaultEnv(const char* key,
                     const std::string& value,
                     const std::unordered_set<std::string>& lockedKeys,
                     std::unordered_map<std::string, std::string>& applied) {
    if (lockedKeys.count(key) > 0) {
        return;
    }
    setenv(key, value.c_str(), 1);
    applied[key] = value;
}

void applyModelDefaults(const std::filesystem::path& modelPath) {
    std::string filename = toLower(modelPath.filename().string());
    std::unordered_set<std::string> lockedKeys;
    const std::vector<const char*> keys = {
        "NRVNA_TEMP",
        "NRVNA_TOP_P",
        "NRVNA_TOP_K",
        "NRVNA_PREDICT",
        "NRVNA_REPEAT_PENALTY",
        "NRVNA_REPEAT_LAST_N"
    };

    for (const auto* key : keys) {
        if (std::getenv(key)) {
            lockedKeys.insert(key);
        }
    }

    std::unordered_map<std::string, std::string> applied;

    applyDefaultEnv("NRVNA_TEMP", "0.7", lockedKeys, applied);
    applyDefaultEnv("NRVNA_TOP_P", "0.9", lockedKeys, applied);
    applyDefaultEnv("NRVNA_TOP_K", "40", lockedKeys, applied);
    applyDefaultEnv("NRVNA_PREDICT", "512", lockedKeys, applied);
    applyDefaultEnv("NRVNA_REPEAT_PENALTY", "1.1", lockedKeys, applied);
    applyDefaultEnv("NRVNA_REPEAT_LAST_N", "64", lockedKeys, applied);

    if (containsToken(filename, "coder") || containsToken(filename, "code")) {
        applyDefaultEnv("NRVNA_TEMP", "0.3", lockedKeys, applied);
        applyDefaultEnv("NRVNA_PREDICT", "256", lockedKeys, applied);
    }

    if (containsToken(filename, "deepseek") || containsToken(filename, "r1")) {
        applyDefaultEnv("NRVNA_TEMP", "0.4", lockedKeys, applied);
    }

    if (containsToken(filename, "3b") || containsToken(filename, "500m") || containsToken(filename, "mini")) {
        applyDefaultEnv("NRVNA_PREDICT", "256", lockedKeys, applied);
    }

    if (!applied.empty()) {
        LOG_INFO("Applied default params: " + std::to_string(applied.size()));
        for (const auto& entry : applied) {
            LOG_DEBUG("  " + entry.first + "=" + entry.second);
        }
    }
}

int main(int argc, char* argv[]) {
    // Suppress noisy logs for clean UI
    Logger::setLevel(LogLevel::ERROR);

    // Handle --help and --version before anything else
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help" || arg == "--design") {
            std::cout << "\033[2J\033[1;1H";
            printUsage(argv[0]);
            return 0;
        }
        if (arg == "-v" || arg == "--version") {
            std::cout << VERSION << "\n";
            return 0;
        }
    }

    if (argc < 3 || argc > 4) {
        printUsage(argv[0]);
        return 1;
    }

    std::string modelPath = argv[1];
    std::string workspace = argv[2];
    int workers = 4;

    if (argc == 4) {
        try {
            workers = std::stoi(argv[3]);
            if (workers < 1 || workers > 64) {
                std::cerr << "Error: Workers must be between 1 and 64\n";
                return 1;
            }
        } catch (...) {
            std::cerr << "Error: Invalid worker count: " << argv[3] << "\n";
            return 1;
        }
    }

    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    if (auto resolved = resolveModelPath(modelPath)) {
        modelPath = resolved->string();
    }

    if (!std::filesystem::exists(std::filesystem::path(modelPath))) {
        std::cerr << "Error: Model not found: " << modelPath << "\n";
        return 1;
    }

    applyModelDefaults(std::filesystem::path(modelPath));

    std::cout << "\033[2J\033[1;1H";
    printBanner();

    std::string modelName = std::filesystem::path(modelPath).filename().string();
    std::cout << "    Loading " << modelName << "\n" << std::flush;

    try {
        auto server = std::make_unique<Server>(modelPath, workspace, workers);

        if (!server->start()) {
            std::cout << " \033[31mfailed\033[0m\n";
            return 1;
        }

        std::cout << "\n";

        std::cout << "    \033[1mModel\033[0m      " << modelName << "\n";
        std::cout << "    \033[1mWorkers\033[0m    " << workers << " ready\n";
        std::cout << "    \033[1mWorkspace\033[0m  " << workspace << "\n";
        std::cout << "\n";
        std::cout << "    \033[90mSubmit\033[0m  wrk " << workspace << " \"prompt\"\n";
        std::cout << "    \033[90mResult\033[0m  flw " << workspace << " -w <job-id>\n";
        std::cout << "\n";
        std::cout << "    \033[90m────────────────────────────────────────────────────────────\033[0m\n";
        std::cout << "\n";

        while (!g_shutdown_requested && server->isRunning()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        if (g_shutdown_requested) {
            std::cout << "\nShutdown requested, stopping server..." << std::endl;
        }
        LOG_DEBUG("Shutdown requested, stopping server...");
        server->shutdown();

    } catch (const std::exception& e) {
        LOG_ERROR("Server error: " + std::string(e.what()));
        return 1;
    } catch (...) {
        LOG_ERROR("Unknown server error");
        return 1;
    }

    LOG_DEBUG("nrvna-ai daemon stopped");
    return 0;
}
