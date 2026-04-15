/*
 * nrvna ai - Server daemon (nrvnad)
 * Copyright (c) 2025 Sanmathi Bharamgouda
 * SPDX-License-Identifier: MIT
 */

#include "nrvna/logger.hpp"
#include "nrvna/runner.hpp"
#include "nrvna/server.hpp"
#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <unistd.h>
#include <optional>

using namespace nrvnaai;

constexpr const char * VERSION = "0.1.0";

static volatile sig_atomic_t g_shutdown_requested = 0;
static std::filesystem::path g_models_dir;

void signalHandler(int signal) {
    (void) signal;
    g_shutdown_requested = 1;
}

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool containsToken(const std::string & haystack, const std::string & needle) {
    return haystack.find(needle) != std::string::npos;
}

std::filesystem::path resolveModelsDir(const char * argv0) {
    if (const char * env = std::getenv("NRVNA_MODELS_DIR")) {
        return std::filesystem::path(env);
    }

    std::filesystem::path exePath(argv0 ? argv0 : "");
    if (!exePath.empty()) {
        std::error_code ec;
        exePath = std::filesystem::absolute(exePath, ec).lexically_normal();
        if (!ec && std::filesystem::exists(exePath)) {
            auto base = exePath.parent_path();
            if (std::filesystem::exists(base / "models")) {
                return base / "models";
            }
            if (std::filesystem::exists(base.parent_path() / "models")) {
                return base.parent_path() / "models";
            }
        }
    }

    return std::filesystem::current_path() / "models";
}

std::optional<std::filesystem::path> resolveModelPath(const std::string & modelArg) {
    std::filesystem::path candidate(modelArg);
    if (std::filesystem::exists(candidate)) {
        return candidate;
    }

    if (!std::filesystem::exists(g_models_dir)) {
        return std::nullopt;
    }

    std::string needle = toLower(modelArg);
    std::vector<std::filesystem::path> matches;

    for (const auto & entry : std::filesystem::directory_iterator(g_models_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto & path = entry.path();
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

std::optional<std::filesystem::path> resolveMmprojPath(const std::filesystem::path & modelPath) {
    std::filesystem::path dir = modelPath.parent_path();
    if (dir.empty() || !std::filesystem::exists(dir)) {
        return std::nullopt;
    }

    std::string stem = toLower(modelPath.stem().string());
    std::vector<std::filesystem::path> matches;

    for (const auto & entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto & path = entry.path();
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

std::optional<std::filesystem::path> resolveVocoderPath(const std::filesystem::path & modelPath) {
    std::filesystem::path dir = modelPath.parent_path();
    if (dir.empty() || !std::filesystem::exists(dir)) {
        return std::nullopt;
    }

    std::string modelName = toLower(modelPath.stem().string());
    if (!containsToken(modelName, "outetts") && !containsToken(modelName, "oute")) {
        return std::nullopt;
    }

    std::vector<std::filesystem::path> matches;
    for (const auto & entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".gguf") continue;
        std::string filename = toLower(entry.path().filename().string());
        if (containsToken(filename, "vocoder") || containsToken(filename, "wavtokenizer")) {
            matches.push_back(entry.path());
        }
    }

    if (matches.empty()) return std::nullopt;
    std::sort(matches.begin(), matches.end());
    return matches.front();
}

void applyDefaultEnv(const char * key,
                     const std::string & value,
                     const std::unordered_set<std::string> & lockedKeys,
                     std::unordered_map<std::string, std::string> & applied) {
    if (lockedKeys.count(key) > 0) {
        return;
    }
    setenv(key, value.c_str(), 1);
    applied[key] = value;
}

void applyModelDefaults(const std::filesystem::path & modelPath, const ModelInfo & info) {
    std::string filename = toLower(modelPath.filename().string());
    std::unordered_set<std::string> lockedKeys;

    if (std::getenv("NRVNA_TEMP")) lockedKeys.insert("NRVNA_TEMP");

    std::unordered_map<std::string, std::string> applied;
    std::string archLower = toLower(info.arch);
    std::string descLower = toLower(info.desc);
    if (containsToken(archLower, "code") || containsToken(descLower, "coder") ||
        containsToken(filename, "coder") || containsToken(filename, "code")) {
        applyDefaultEnv("NRVNA_TEMP", "0.3", lockedKeys, applied);
    } else if (archLower == "deepseek" || containsToken(descLower, "r1") ||
               containsToken(filename, "deepseek") || containsToken(filename, "r1")) {
        applyDefaultEnv("NRVNA_TEMP", "0.6", lockedKeys, applied);
    }

    constexpr uint64_t VRAM_4GB = 4ULL * 1024 * 1024 * 1024;
    if (info.model_size_bytes > VRAM_4GB) {
        int gpuLayers = 99;
#if !defined(__APPLE__)
        gpuLayers = 0;
#endif
        if (std::getenv("NRVNA_GPU_LAYERS")) {
            gpuLayers = std::atoi(std::getenv("NRVNA_GPU_LAYERS"));
        }
        if (gpuLayers > 0) {
            double gb = static_cast<double>(info.model_size_bytes) / (1024.0 * 1024.0 * 1024.0);
            LOG_WARN("Model size " + std::to_string(gb).substr(0, 4) + " GB may exceed efficient GPU fit (4 GB VRAM). "
                     "Set NRVNA_GPU_LAYERS=0 to force CPU if inference produces garbage.");
        }
    }

    if (info.n_embd_out > 0 && info.has_encoder && !info.has_decoder) {
        LOG_INFO("Embedding model detected: output dim=" + std::to_string(info.n_embd_out));
    }

    if (!applied.empty()) {
        std::string summary = "Applied defaults:";
        for (const auto & entry : applied) {
            summary += " " + entry.first + "=" + entry.second;
        }
        LOG_INFO(summary);
    }
}

void printHelp() {
    std::cout << "nrvna " << VERSION << "                        async · inference · primitive\n\n";
    std::cout << "USAGE\n\n";
    std::cout << "  nrvnad <model.gguf> <workspace> [options]    start daemon\n";
    std::cout << "  wrk <workspace> \"prompt\"                     submit work\n";
    std::cout << "  flw <workspace> [job-id]                     collect results\n\n";
    std::cout << "OPTIONS\n\n";
    std::cout << "  --mmproj <path>     Vision projection model\n";
    std::cout << "  --vocoder <path>    TTS vocoder model\n";
    std::cout << "  -w, --workers <n>   Worker threads (default: 4)\n";
    std::cout << "  -v, --version       Show version\n";
    std::cout << "  -h, --help          Show this help\n\n";
    std::cout << "NOTES\n\n";
    std::cout << "  Models are .gguf files. Set NRVNA_MODELS_DIR or pass a full path.\n";
    std::cout << "  MMProj and vocoder are auto-detected from the model directory.\n";
}

int main(int argc, char * argv[]) {
    g_models_dir = resolveModelsDir(argv[0]);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printHelp();
            return 0;
        }
        if (arg == "-v" || arg == "--version") {
            std::cout << VERSION << "\n";
            return 0;
        }
    }

    std::string modelPath;
    std::string workspace;
    std::string mmprojPath;
    std::string vocoderPath;
    int workers = 4;

    std::vector<std::string> positionalArgs;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-w" || arg == "--workers") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --workers requires a value\n";
                return 1;
            }
            try {
                workers = std::stoi(argv[++i]);
            } catch (...) {
                std::cerr << "Error: Invalid worker count\n";
                return 1;
            }
            if (workers < 1 || workers > 64) {
                std::cerr << "Error: worker count must be between 1 and 64\n";
                return 1;
            }
        } else if (arg == "--mmproj") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --mmproj requires a path\n";
                return 1;
            }
            mmprojPath = argv[++i];
        } else if (arg == "--vocoder") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --vocoder requires a path\n";
                return 1;
            }
            vocoderPath = argv[++i];
        } else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "Error: unknown option: " << arg << "\n";
            return 1;
        } else if (!arg.empty() && arg[0] != '-') {
            positionalArgs.push_back(arg);
        }
    }

    if (!positionalArgs.empty()) {
        modelPath = positionalArgs[0];
    }
    if (positionalArgs.size() > 1) {
        workspace = positionalArgs[1];
    }
    if (positionalArgs.size() > 2) {
        std::cerr << "Error: unexpected extra positional argument: " << positionalArgs[2] << "\n";
        return 1;
    }

    if (modelPath.empty()) {
        printHelp();
        return 0;
    }

    if (workspace.empty()) {
        std::cerr << "Error: workspace required\n";
        std::cerr << "Usage: nrvnad <model> <workspace> [--mmproj <path>] [--vocoder <path>] [-w <n>]\n";
        return 1;
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

    if (mmprojPath.empty()) {
        if (auto resolved = resolveMmprojPath(std::filesystem::path(modelPath))) {
            mmprojPath = resolved->string();
        }
    }

    if (vocoderPath.empty()) {
        if (auto resolved = resolveVocoderPath(std::filesystem::path(modelPath))) {
            vocoderPath = resolved->string();
        }
    }

    if (!vocoderPath.empty() && !std::filesystem::exists(std::filesystem::path(vocoderPath))) {
        std::cerr << "Error: Vocoder not found: " << vocoderPath << "\n";
        return 1;
    }

    ModelInfo probeInfo = Runner::probeModelInfo(modelPath);
    if (!probeInfo.valid) {
        std::cerr << "Error: Failed to probe model metadata: " << modelPath << "\n";
        return 1;
    }

    applyModelDefaults(std::filesystem::path(modelPath), probeInfo);

    std::cout << "\n";
    std::cout << "  \033[1mnrvna\033[0m " << VERSION << "                        \033[90masync · inference · primitive\033[0m\n";
    std::cout << "  \033[90m─────────────────────────────────────────────────────────────────\033[0m\n\n";

    {
        double gb = static_cast<double>(probeInfo.model_size_bytes) / (1024.0 * 1024.0 * 1024.0);
        std::string sizeStr = std::to_string(gb).substr(0, 3) + " GB";
        std::cout << "  \033[90mModel: " << probeInfo.desc
                  << ", ctx=" << probeInfo.n_ctx_train
                  << ", " << sizeStr
                  << ", template=" << (probeInfo.has_chat_template ? "yes" : "no")
                  << ", encoder=" << (probeInfo.has_encoder ? "yes" : "no")
                  << "\033[0m\n";
    }

    std::string modelName = std::filesystem::path(modelPath).filename().string();
    std::cout << "  Loading " << modelName << "\n" << std::flush;

    try {
        auto server = std::make_unique<Server>(modelPath, workspace, workers, mmprojPath, vocoderPath);

        if (!server->start()) {
            std::cout << "  \033[31mFailed to start\033[0m\n";
            return 1;
        }

        std::filesystem::path pidPath = std::filesystem::path(workspace) / ".nrvnad.pid";
        {
            std::ofstream pf(pidPath);
            if (pf) {
                pf << getpid();
            }
        }

        std::cout << "\n";
        std::cout << "  \033[1mRUNNING\033[0m\n\n";
        std::cout << "    Model      " << modelName << "\n";
        std::cout << "    Workers    " << workers << "\n";
        std::cout << "    Workspace  " << workspace << "\n";
        if (!mmprojPath.empty()) {
            std::cout << "    MMProj     " << mmprojPath << "\n";
        }
        if (!vocoderPath.empty()) {
            std::cout << "    Vocoder    " << vocoderPath << "\n";
        }
        std::cout << "\n";
        std::cout << "  \033[90m─────────────────────────────────────────────────────────────────\033[0m\n\n";
        std::cout << "  Submit:  ./wrk " << workspace << " \"prompt\"\n";
        std::cout << "  Results: ./flw " << workspace << " <job-id>\n\n";
        std::cout << "  \033[90m─────────────────────────────────────────────────────────────────\033[0m\n\n";

        while (!g_shutdown_requested && server->isRunning()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        if (g_shutdown_requested) {
            std::cout << "\nShutdown requested, stopping server..." << std::endl;
        }
        LOG_DEBUG("Shutdown requested, stopping server...");
        {
            std::error_code ec;
            std::filesystem::remove(pidPath, ec);
        }
        server->shutdown();

    } catch (const std::exception & e) {
        LOG_ERROR("Server error: " + std::string(e.what()));
        return 1;
    } catch (...) {
        LOG_ERROR("Unknown server error");
        return 1;
    }

    LOG_DEBUG("nrvna-ai daemon stopped");
    return 0;
}
