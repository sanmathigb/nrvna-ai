/*
 * nrvna ai - Server daemon (nrvnad)
 * Copyright (c) 2025 Sanmathi Bharamgouda
 * SPDX-License-Identifier: MIT
 */

#include "nrvna/server.hpp"
#include "nrvna/logger.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
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

struct WorkspaceInfo {
    std::string path;
    std::string model;
    size_t queued = 0;
    size_t processing = 0;
    size_t done = 0;
    size_t failed = 0;
};

struct ModelInfo {
    std::string filename;
    std::string shortName;
    uintmax_t size;
};

size_t countDirEntries(const std::filesystem::path& dir) {
    size_t count = 0;
    if (std::filesystem::exists(dir)) {
        for (const auto& entry : std::filesystem::directory_iterator(dir)) {
            if (entry.is_directory()) ++count;
        }
    }
    return count;
}

bool isWorkspace(const std::filesystem::path& dir) {
    return std::filesystem::exists(dir / "input" / "ready") &&
           std::filesystem::exists(dir / "output");
}

std::vector<WorkspaceInfo> scanWorkspaces() {
    std::vector<WorkspaceInfo> workspaces;
    auto cwd = std::filesystem::current_path();

    for (const auto& entry : std::filesystem::directory_iterator(cwd)) {
        if (!entry.is_directory()) continue;
        if (entry.path().filename().string()[0] == '.') continue;
        if (!isWorkspace(entry.path())) continue;

        WorkspaceInfo ws;
        ws.path = "./" + entry.path().filename().string();
        ws.queued = countDirEntries(entry.path() / "input" / "ready");
        ws.processing = countDirEntries(entry.path() / "processing");
        ws.done = countDirEntries(entry.path() / "output");
        ws.failed = countDirEntries(entry.path() / "failed");

        // Read model if available
        std::ifstream mf(entry.path() / ".model");
        if (mf) std::getline(mf, ws.model);

        workspaces.push_back(ws);
    }
    std::sort(workspaces.begin(), workspaces.end(),
              [](const auto& a, const auto& b) { return a.path < b.path; });
    return workspaces;
}

std::string extractShortName(const std::string& filename) {
    std::string lower = filename;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower.find("llama") != std::string::npos) return "llama";
    if (lower.find("mistral") != std::string::npos) return "mistral";
    if (lower.find("qwen") != std::string::npos && lower.find("coder") != std::string::npos) return "qwen-coder";
    if (lower.find("qwen") != std::string::npos) return "qwen";
    if (lower.find("deepseek") != std::string::npos) return "deepseek";
    if (lower.find("phi") != std::string::npos) return "phi";
    if (lower.find("gemma") != std::string::npos) return "gemma";
    if (lower.find("smol") != std::string::npos) return "smol";

    // Fallback: first part before dash or dot
    size_t pos = filename.find_first_of("-_.");
    if (pos != std::string::npos && pos > 0) {
        return filename.substr(0, pos);
    }
    return filename.substr(0, 8);
}

std::vector<ModelInfo> scanModels() {
    std::vector<ModelInfo> models;
    std::filesystem::path modelsDir = std::filesystem::current_path() / "models";

    if (!std::filesystem::exists(modelsDir)) return models;

    for (const auto& entry : std::filesystem::directory_iterator(modelsDir)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".gguf") continue;
        std::string filename = entry.path().filename().string();
        if (filename.find("mmproj") != std::string::npos) continue;

        ModelInfo m;
        m.filename = filename;
        m.shortName = extractShortName(filename);
        m.size = entry.file_size();
        models.push_back(m);
    }
    std::sort(models.begin(), models.end(),
              [](const auto& a, const auto& b) { return a.shortName < b.shortName; });
    return models;
}

struct DashboardResult {
    std::vector<WorkspaceInfo> selectable;  // All non-running workspaces (for interactive start)
    std::vector<ModelInfo> models;
};

DashboardResult printDashboard() {
    auto models = scanModels();
    auto workspaces = scanWorkspaces();

    // Split workspaces into running and selectable (pending + completed)
    std::vector<WorkspaceInfo> running, selectable;
    for (const auto& ws : workspaces) {
        if (ws.model.empty()) continue;
        if (ws.processing > 0) {
            running.push_back(ws);
        } else {
            selectable.push_back(ws);
        }
    }
    // Sort selectable: pending first (have queued jobs), then completed
    std::sort(selectable.begin(), selectable.end(), [](const auto& a, const auto& b) {
        if ((a.queued > 0) != (b.queued > 0)) return a.queued > 0;
        return a.path < b.path;
    });

    // Header
    std::cout << "\n";
    std::cout << "  \033[1mnrvna\033[0m " << VERSION << "                        \033[90masync · inference · primitive\033[0m\n";
    std::cout << "  \033[90m─────────────────────────────────────────────────────────────────\033[0m\n";
    std::cout << "\n";

    // Running workspaces (daemon active)
    if (!running.empty()) {
        std::cout << "  \033[1mRUNNING\033[0m  \033[90mdaemon processing\033[0m\n\n";
        for (const auto& ws : running) {
            std::string displayPath = ws.path;
            if (displayPath.size() > 16) displayPath = displayPath.substr(0, 13) + "...";
            std::cout << "    \033[36m" << std::left << std::setw(20) << displayPath << "\033[0m  ";
            std::cout << "\033[90m" << std::left << std::setw(10) << ws.model << "\033[0m  ";
            std::cout << "\033[32m" << ws.processing << " processing\033[0m";
            if (ws.queued > 0) std::cout << "  \033[33m" << ws.queued << " queued\033[0m";
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    // Selectable workspaces (can start daemon)
    if (!selectable.empty()) {
        std::cout << "  \033[1mWORKSPACES\033[0m  \033[90mselect to start daemon\033[0m\n\n";
        constexpr size_t maxDisplay = 6;
        int idx = 1;
        for (const auto& ws : selectable) {
            if (static_cast<size_t>(idx) > maxDisplay) {
                std::cout << "    \033[90m+" << (selectable.size() - maxDisplay) << " more\033[0m\n";
                break;
            }
            std::string displayPath = ws.path;
            if (displayPath.size() > 16) displayPath = displayPath.substr(0, 13) + "...";
            std::cout << "    \033[33m[" << idx << "]\033[0m  ";
            std::cout << "\033[36m" << std::left << std::setw(16) << displayPath << "\033[0m  ";
            std::cout << "\033[90m" << std::left << std::setw(10) << ws.model << "\033[0m  ";
            if (ws.queued > 0) {
                std::cout << "\033[33m" << ws.queued << " pending\033[0m  ";
            }
            if (ws.done > 0) {
                std::cout << "\033[32m" << ws.done << " done\033[0m  ";
            }
            if (ws.failed > 0) {
                std::cout << "\033[31m" << ws.failed << " failed\033[0m";
            }
            std::cout << "\n";
            ++idx;
        }
        std::cout << "\n";
    }

    // Models
    if (!models.empty()) {
        std::cout << "  \033[1mMODELS\033[0m  \033[90m./models/\033[0m\n\n";
        constexpr size_t maxDisplay = 4;
        size_t displayed = 0;
        for (const auto& m : models) {
            if (displayed >= maxDisplay) {
                std::cout << "    \033[90m+" << (models.size() - maxDisplay) << " more\033[0m\n";
                break;
            }
            double gb = static_cast<double>(m.size) / (1024.0 * 1024.0 * 1024.0);
            std::cout << "    \033[36m" << std::left << std::setw(12) << m.shortName << "\033[0m"
                      << std::setw(40) << m.filename
                      << "\033[90m" << std::fixed << std::setprecision(1) << gb << " GB\033[0m\n";
            ++displayed;
        }
        std::cout << "\n";
    } else {
        std::cout << "  \033[1mMODELS\033[0m  \033[90m./models/\033[0m\n\n";
        std::cout << "    \033[33mNo models found\033[0m\n";
        std::cout << "    \033[90mDownload: ./scripts/models pull llama\033[0m\n\n";
    }

    // Commands
    std::cout << "  \033[90m─────────────────────────────────────────────────────────────────\033[0m\n";
    std::cout << "\n";
    std::cout << "  \033[1mSTART\033[0m       nrvnad <model> <workspace>\n";
    std::cout << "  \033[1mSUBMIT\033[0m      wrk <workspace> \"prompt\"\n";
    std::cout << "  \033[1mRETRIEVE\033[0m    flw <workspace> [job-id]\n";
    std::cout << "\n";

    return {selectable, models};
}

int promptAndStartDaemon(const std::vector<WorkspaceInfo>& pending, int& workers) {
    if (pending.empty()) return -1;

    std::cout << "  \033[90mStart daemon? [1";
    if (pending.size() > 1) std::cout << "-" << pending.size();
    std::cout << "] or Enter to quit:\033[0m ";
    std::cout.flush();

    std::string input;
    if (!std::getline(std::cin, input) || input.empty() || input == "n" || input == "N") {
        return -1;
    }

    int choice = -1;
    try {
        choice = std::stoi(input);
        if (choice < 1 || choice > static_cast<int>(pending.size())) {
            return -1;
        }
    } catch (...) {
        return -1;
    }

    const auto& ws = pending[choice - 1];
    std::cout << "\n  Starting \033[36m" << ws.path << "\033[0m with \033[36m" << ws.model << "\033[0m\n";
    std::cout << "  \033[90mWorkers [4]:\033[0m ";
    std::cout.flush();

    std::string workerInput;
    if (std::getline(std::cin, workerInput) && !workerInput.empty()) {
        try {
            int w = std::stoi(workerInput);
            if (w >= 1 && w <= 64) workers = w;
        } catch (...) {}
    }

    return choice - 1;
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
        if (arg == "-h" || arg == "--help") {
            std::cout << "\033[2J\033[1;1H";
            printDashboard();
            return 0;
        }
        if (arg == "-v" || arg == "--version") {
            std::cout << VERSION << "\n";
            return 0;
        }
    }

    std::string modelPath;
    std::string workspace;
    int workers = 4;

    if (argc < 3) {
        // Interactive mode - show dashboard and prompt
        std::cout << "\033[2J\033[1;1H";
        auto result = printDashboard();

        if (!result.selectable.empty()) {
            int choice = promptAndStartDaemon(result.selectable, workers);
            if (choice >= 0) {
                const auto& ws = result.selectable[choice];
                modelPath = ws.model;
                workspace = ws.path;
            } else {
                return 0;
            }
        } else {
            return 0;
        }
    } else if (argc > 4) {
        std::cout << "\033[2J\033[1;1H";
        printDashboard();
        return 1;
    } else {
        modelPath = argv[1];
        workspace = argv[2];
    }

    if (argc >= 4) {
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
    std::cout << "\n";
    std::cout << "  \033[1mnrvna\033[0m " << VERSION << "                        \033[90masync · inference · primitive\033[0m\n";
    std::cout << "  \033[90m─────────────────────────────────────────────────────────────────\033[0m\n";
    std::cout << "\n";

    std::string modelName = std::filesystem::path(modelPath).filename().string();
    std::cout << "  Loading " << modelName << "\n" << std::flush;

    try {
        auto server = std::make_unique<Server>(modelPath, workspace, workers);

        if (!server->start()) {
            std::cout << "  \033[31mFailed to start\033[0m\n";
            return 1;
        }

        // Write model info to workspace for dashboard
        {
            std::ofstream mf(workspace + "/.model");
            if (mf) mf << extractShortName(modelName);
        }

        std::cout << "\n";
        std::cout << "  \033[1mRUNNING\033[0m\n\n";
        std::cout << "    Model      " << modelName << "\n";
        std::cout << "    Workers    " << workers << "\n";
        std::cout << "    Workspace  " << workspace << "\n";
        std::cout << "\n";
        std::cout << "  \033[90m─────────────────────────────────────────────────────────────────\033[0m\n";
        std::cout << "\n";
        std::cout << "  \033[90mSubmit:\033[0m  wrk " << workspace << " \"prompt\"\n";
        std::cout << "  \033[90mResults:\033[0m flw " << workspace << "\n";
        std::cout << "\n";
        std::cout << "  \033[90m─────────────────────────────────────────────────────────────────\033[0m\n";
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
