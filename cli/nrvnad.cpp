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
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <unistd.h>

using namespace nrvnaai;

constexpr const char* VERSION = "0.1.0";

// Async-signal-safe: only set flag, no complex operations
static volatile sig_atomic_t g_shutdown_requested = 0;
static std::filesystem::path g_models_dir;

void signalHandler(int signal) {
    (void)signal;
    g_shutdown_requested = 1;
}

struct WorkspaceInfo {
    std::string path;
    std::string model;
    std::string mmproj;
    size_t queued = 0;
    size_t processing = 0;
    size_t done = 0;
    size_t failed = 0;
    bool daemonRunning = false;   // daemon currently active
    bool daemonStopped = false;   // daemon was running but stopped (stale pid)
};

struct ModelInfo {
    std::string filename;
    std::string shortName;
    uintmax_t size;
};

std::string trim(std::string value);

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
           std::filesystem::exists(dir / "input" / "writing");
}

std::optional<pid_t> readPidFile(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) {
        return std::nullopt;
    }
    long pid = 0;
    file >> pid;
    if (pid <= 0) {
        return std::nullopt;
    }
    return static_cast<pid_t>(pid);
}

std::filesystem::path resolveModelsDir(const char* argv0) {
    if (const char* env = std::getenv("NRVNA_MODELS_DIR")) {
        return std::filesystem::path(env);
    }

    std::filesystem::path exePath(argv0 ? argv0 : "");
    if (!exePath.empty()) {
        std::error_code ec;
        exePath = std::filesystem::absolute(exePath, ec);
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

std::string displayPath(const std::filesystem::path& path) {
    std::error_code ec;
    auto abs = std::filesystem::absolute(path, ec);
    auto cwd = std::filesystem::current_path();
    if (!ec) {
        auto rel = abs.lexically_relative(cwd);
        auto relStr = rel.string();
        if (!rel.empty() && relStr.rfind("..", 0) != 0) {
            if (relStr == ".") {
                return "./";
            }
            if (relStr.rfind("./", 0) == 0) {
                return relStr;
            }
            return std::string("./") + relStr;
        }
    }
    return path.string();
}

bool isProcessAlive(pid_t pid) {
    if (pid <= 0) {
        return false;
    }
    if (::kill(pid, 0) == 0) {
        return true;
    }
    return errno == EPERM;
}

std::filesystem::path workspaceHistoryFile() {
    return std::filesystem::current_path() / ".nrvna-workspaces";
}

std::filesystem::path normalizePath(const std::filesystem::path& path) {
    return std::filesystem::absolute(path).lexically_normal();
}

void recordWorkspacePath(const std::filesystem::path& workspace) {
    auto cwd = std::filesystem::current_path();
    auto normalized = normalizePath(workspace);

    if (normalized.parent_path() == cwd) {
        return;
    }

    std::unordered_set<std::string> seen;
    auto historyPath = workspaceHistoryFile();
    {
        std::ifstream file(historyPath);
        std::string line;
        while (std::getline(file, line)) {
            line = trim(line);
            if (!line.empty()) {
                seen.insert(line);
            }
        }
    }

    auto normalizedStr = normalized.string();
    if (seen.count(normalizedStr) > 0) {
        return;
    }

    std::ofstream file(historyPath, std::ios::app);
    if (file) {
        file << normalizedStr << "\n";
    }
}

WorkspaceInfo readWorkspaceInfo(const std::filesystem::path& path, const std::string& displayPath) {
    WorkspaceInfo ws;
    ws.path = displayPath;
    ws.queued = countDirEntries(path / "input" / "ready");
    ws.processing = countDirEntries(path / "processing");
    ws.done = countDirEntries(path / "output");
    ws.failed = countDirEntries(path / "failed");

    if (auto pid = readPidFile(path / ".nrvnad.pid")) {
        ws.daemonRunning = isProcessAlive(*pid);
        ws.daemonStopped = !ws.daemonRunning;
    }

    std::ifstream mf(path / ".model");
    if (mf) std::getline(mf, ws.model);
    std::ifstream mp(path / ".mmproj");
    if (mp) std::getline(mp, ws.mmproj);

    return ws;
}

std::vector<WorkspaceInfo> scanWorkspaces() {
    std::vector<WorkspaceInfo> workspaces;
    auto cwd = std::filesystem::current_path();
    std::unordered_set<std::string> seen;

    for (const auto& entry : std::filesystem::directory_iterator(cwd)) {
        if (!entry.is_directory()) continue;
        if (entry.path().filename().string()[0] == '.') continue;
        if (!isWorkspace(entry.path())) continue;

        workspaces.push_back(readWorkspaceInfo(entry.path(), "./" + entry.path().filename().string()));
        seen.insert(normalizePath(entry.path()).string());
    }

    std::ifstream history(workspaceHistoryFile());
    if (history) {
        std::string line;
        while (std::getline(history, line)) {
            line = trim(line);
            if (line.empty()) continue;
            auto path = normalizePath(std::filesystem::path(line));
            if (seen.count(path.string()) > 0) continue;
            if (!std::filesystem::exists(path) || !isWorkspace(path)) continue;

            workspaces.push_back(readWorkspaceInfo(path, path.string()));
            seen.insert(path.string());
        }
    }
    std::sort(workspaces.begin(), workspaces.end(),
              [](const auto& a, const auto& b) { return a.path < b.path; });
    return workspaces;
}

std::string extractShortName(const std::string& filename) {
    size_t pos = filename.find_first_of("-_.");
    std::string name = (pos != std::string::npos && pos > 0)
        ? filename.substr(0, pos)
        : filename.substr(0, 8);
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);
    return name;
}

std::vector<ModelInfo> scanModels() {
    std::vector<ModelInfo> models;
    std::filesystem::path modelsDir = g_models_dir;

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

std::optional<std::string> latestJobId(const std::filesystem::path& workspace) {
    std::optional<std::string> latest;
    std::optional<std::filesystem::file_time_type> latestTime;

    auto scanDir = [&](const std::filesystem::path& dir) {
        if (!std::filesystem::exists(dir)) {
            return;
        }
        for (const auto& entry : std::filesystem::directory_iterator(dir)) {
            if (!entry.is_directory()) {
                continue;
            }
            auto ts = std::filesystem::last_write_time(entry);
            if (!latestTime || ts > *latestTime) {
                latestTime = ts;
                latest = entry.path().filename().string();
            }
        }
    };

    scanDir(workspace / "output");
    scanDir(workspace / "failed");
    return latest;
}

struct DashboardResult {
    std::vector<WorkspaceInfo> workspaces;  // All non-running workspaces (for interactive start)
    std::vector<ModelInfo> models;
};

struct DaemonSelection {
    std::string modelPath;
    std::string workspace;
    std::string mmprojPath;
};

DashboardResult printDashboard() {
    auto models = scanModels();
    auto allWorkspaces = scanWorkspaces();

    // Only show workspaces where daemon is not currently running
    std::vector<WorkspaceInfo> selectable;
    for (const auto& ws : allWorkspaces) {
        if (!ws.daemonRunning) selectable.push_back(ws);
    }
    // Sort: queued jobs first, then stopped daemons, then idle
    std::sort(selectable.begin(), selectable.end(), [](const auto& a, const auto& b) {
        auto priority = [](const WorkspaceInfo& ws) {
            if (ws.queued > 0) return 0;
            if (ws.daemonStopped) return 1;
            return 2;
        };
        int pa = priority(a), pb = priority(b);
        if (pa != pb) return pa < pb;
        return a.path < b.path;
    });

    // Header
    std::cout << "\n";
    std::cout << "  \033[1mnrvna\033[0m " << VERSION << "                        \033[90masync · inference · primitive\033[0m\n";
    std::cout << "  \033[90m─────────────────────────────────────────────────────────────────\033[0m\n";
    std::cout << "\n";

    // Selectable workspaces with inline status tags
    if (!selectable.empty()) {
        std::cout << "  \033[1mWORKSPACES\033[0m\n\n";
        constexpr size_t maxDisplay = 8;
        int idx = 1;
        size_t displayed = 0;
        for (const auto& ws : selectable) {
            if (displayed >= maxDisplay) {
                std::cout << "    \033[90m+" << (selectable.size() - maxDisplay) << " more\033[0m\n";
                break;
            }
            std::string dp = ws.path;
            if (dp.size() > 16) dp = dp.substr(0, 13) + "...";
            std::string modelDisplay = ws.model.empty() ? "(no model)" : extractShortName(ws.model);
            std::cout << "    \033[33m[" << idx << "]\033[0m  ";
            std::cout << "\033[36m" << std::left << std::setw(16) << dp << "\033[0m  ";
            std::cout << "\033[90m" << std::left << std::setw(10) << modelDisplay << "\033[0m  ";
            if (ws.queued > 0) std::cout << "\033[33;1m" << ws.queued << " queued\033[0m  ";
            if (ws.done > 0) std::cout << "\033[32m" << ws.done << " done\033[0m  ";
            if (ws.failed > 0) std::cout << "\033[31m" << ws.failed << " failed\033[0m";
            std::cout << "\n";
            ++idx;
            ++displayed;
        }
        std::cout << "\n";
    }

    // Models - numbers continue from workspace count
    bool isDefaultModelsDir = (std::getenv("NRVNA_MODELS_DIR") == nullptr);
    if (!models.empty()) {
        std::cout << "  \033[1mMODELS\033[0m  \033[90m" << displayPath(g_models_dir) << "/";
        if (models.size() > 6) {
            std::cout << "  (" << models.size() << " available)";
        }
        std::cout << "\033[0m\n\n";
        constexpr size_t maxDisplay = 6;
        size_t modelOffset = selectable.size();
        size_t displayed = 0;
        for (size_t i = 0; i < models.size(); ++i) {
            if (displayed >= maxDisplay) {
                size_t remaining = models.size() - maxDisplay;
                if (remaining <= 3) {
                    const auto& m = models[i];
                    double gb = static_cast<double>(m.size) / (1024.0 * 1024.0 * 1024.0);
                    std::cout << "    \033[33m[" << (modelOffset + i + 1) << "]\033[0m  ";
                    std::cout << "\033[36m" << std::left << std::setw(12) << m.shortName << "\033[0m"
                              << std::setw(40) << m.filename
                              << "\033[90m" << std::fixed << std::setprecision(1) << gb << " GB\033[0m\n";
                    ++displayed;
                    continue;
                }
                std::cout << "    \033[90m+" << remaining << " more (type name to search)\033[0m\n";
                break;
            }
            const auto& m = models[i];
            double gb = static_cast<double>(m.size) / (1024.0 * 1024.0 * 1024.0);
            std::cout << "    \033[33m[" << (modelOffset + i + 1) << "]\033[0m  ";
            std::cout << "\033[36m" << std::left << std::setw(12) << m.shortName << "\033[0m"
                      << std::setw(40) << m.filename
                      << "\033[90m" << std::fixed << std::setprecision(1) << gb << " GB\033[0m\n";
            ++displayed;
        }
        std::cout << "\n";
    } else {
        std::cout << "  \033[1mMODELS\033[0m  \033[90m" << displayPath(g_models_dir) << "/\033[0m\n\n";
        std::cout << "    \033[33mNo .gguf models found\033[0m\n\n";
        std::cout << "    \033[90mDownload GGUF models from huggingface.co\033[0m\n";
        if (isDefaultModelsDir) {
            std::cout << "    \033[90mPlace in ./models/ or set NRVNA_MODELS_DIR\033[0m\n";
        }
        std::cout << "\n";
    }

    // Footer legend
    std::cout << "  \033[90m─────────────────────────────────────────────────────────────────\033[0m\n";
    std::cout << "  \033[90m";
    if (!selectable.empty()) {
        std::cout << "[1";
        if (selectable.size() > 1) std::cout << "-" << selectable.size();
        std::cout << "] workspace";
        if (selectable.size() > 1) std::cout << "s";
    }
    if (!models.empty()) {
        if (!selectable.empty()) std::cout << "    ";
        std::cout << "[" << (selectable.size() + 1);
        if (models.size() > 1) std::cout << "-" << (selectable.size() + models.size());
        std::cout << "] model";
        if (models.size() > 1) std::cout << "s";
    }
    if (!selectable.empty() || !models.empty()) std::cout << "    ";
    std::cout << "m = all models    q = quit\033[0m\n";

    return {selectable, models};
}

std::string toLower(std::string value);
bool isNumber(const std::string& value);
std::optional<std::filesystem::path> resolveModelPath(const std::string& modelArg);

int promptWorkers(int defaultVal = 4) {
    std::cout << "  \033[90mWorkers [" << defaultVal << "]:\033[0m ";
    std::cout.flush();
    std::string input;
    if (std::getline(std::cin, input) && !input.empty()) {
        try {
            int w = std::stoi(input);
            if (w >= 1 && w <= 64) return w;
        } catch (...) {}
    }
    return defaultVal;
}
std::optional<std::string> promptWorkspacePath();

std::optional<std::filesystem::path> resolveMmprojPath(const std::filesystem::path& modelPath);

// Helper: select a model, ask workspace + workers, auto-detect mmproj
std::optional<DaemonSelection> selectModel(const ModelInfo& model, int& workers) {
    std::cout << "\n  Selected \033[36m" << model.filename << "\033[0m\n";

    auto workspace = promptWorkspacePath();
    if (!workspace) return std::nullopt;

    workers = promptWorkers();
    auto modelPath = (g_models_dir / model.filename).string();

    // Auto-detect mmproj
    std::string mmprojPath;
    if (auto resolved = resolveMmprojPath(std::filesystem::path(modelPath))) {
        mmprojPath = resolved->string();
        std::cout << "  \033[90mMMProj: " << resolved->filename().string() << "\033[0m\n";
    }

    return DaemonSelection{modelPath, *workspace, mmprojPath};
}

std::optional<DaemonSelection> promptUnifiedSelection(
    const std::vector<WorkspaceInfo>& workspaces,
    const std::vector<ModelInfo>& models,
    int& workers
) {
    size_t wsCount = workspaces.size();
    size_t modelCount = models.size();

    while (true) {
        std::cout << "\n  \033[90m>\033[0m ";
        std::cout.flush();

        std::string input;
        if (!std::getline(std::cin, input)) {
            return std::nullopt;
        }
        input = trim(input);
        if (input.empty()) continue;

        if (input == "q" || input == "Q" || input == "quit") {
            return std::nullopt;
        }

        if (input == "m" || input == "M" || input == "more") {
            std::cout << "\n  \033[1mALL MODELS\033[0m\n\n";
            for (size_t i = 0; i < models.size(); ++i) {
                const auto& m = models[i];
                double gb = static_cast<double>(m.size) / (1024.0 * 1024.0 * 1024.0);
                std::cout << "    \033[33m[" << (wsCount + i + 1) << "]\033[0m  ";
                std::cout << "\033[36m" << std::left << std::setw(12) << m.shortName << "\033[0m"
                          << std::setw(40) << m.filename
                          << "\033[90m" << std::fixed << std::setprecision(1) << gb << " GB\033[0m\n";
            }
            std::cout << "\n";
            continue;
        }

        // Try as number
        if (isNumber(input)) {
            int choice = std::stoi(input);

            // Workspace selection
            if (choice >= 1 && choice <= static_cast<int>(wsCount)) {
                const auto& ws = workspaces[choice - 1];
                if (ws.model.empty()) {
                    std::cout << "  \033[33mNo model set. Use: nrvnad <model> " << ws.path << "\033[0m\n";
                    continue;
                }
                std::string modelDisplay = extractShortName(ws.model);
                std::cout << "\n  Starting \033[36m" << ws.path << "\033[0m with \033[36m" << modelDisplay << "\033[0m\n";
                workers = promptWorkers();

                // Auto-detect mmproj if stored one is empty or stale
                std::string mmprojPath = ws.mmproj;
                if (mmprojPath.empty() || !std::filesystem::exists(mmprojPath)) {
                    if (auto resolved = resolveModelPath(ws.model)) {
                        if (auto mmResolved = resolveMmprojPath(*resolved)) {
                            mmprojPath = mmResolved->string();
                            std::cout << "  \033[90mMMProj: " << mmResolved->filename().string() << "\033[0m\n";
                        }
                    }
                }

                return DaemonSelection{ws.model, ws.path, mmprojPath};
            }

            // Model selection
            int modelIdx = choice - static_cast<int>(wsCount) - 1;
            if (modelIdx >= 0 && modelIdx < static_cast<int>(modelCount)) {
                auto result = selectModel(models[modelIdx], workers);
                if (result) return result;
                continue;
            }

            std::cout << "  \033[31mInvalid number\033[0m\n";
            continue;
        }

        // Try as model name search
        std::string needle = toLower(input);
        std::vector<size_t> matches;
        for (size_t i = 0; i < models.size(); ++i) {
            std::string fileLower = toLower(models[i].filename);
            if (fileLower.find(needle) != std::string::npos) {
                matches.push_back(i);
            }
        }

        if (matches.empty()) {
            std::cout << "  \033[31mNo matching model found\033[0m\n";
            continue;
        }

        if (matches.size() > 1) {
            std::cout << "\n  \033[33mMultiple matches:\033[0m\n";
            for (size_t idx : matches) {
                const auto& m = models[idx];
                std::cout << "    \033[33m[" << (wsCount + idx + 1) << "]\033[0m  "
                          << "\033[36m" << m.shortName << "\033[0m  " << m.filename << "\n";
            }
            std::cout << "  \033[90mPick a number to select\033[0m\n";
            continue;
        }

        // Single match
        auto result = selectModel(models[matches[0]], workers);
        if (result) return result;
    }
}

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string trim(std::string value) {
    auto notSpace = [](unsigned char c) { return !std::isspace(c); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), notSpace));
    value.erase(std::find_if(value.rbegin(), value.rend(), notSpace).base(), value.end());
    return value;
}

std::optional<std::string> promptWorkspacePath() {
    std::cout << "  Workspace path [workspace]: ";
    std::cout.flush();
    std::string input;
    if (!std::getline(std::cin, input)) return std::nullopt;
    input = trim(input);
    return input.empty() ? "workspace" : input;
}

bool isNumber(const std::string& value) {
    return !value.empty() && std::all_of(value.begin(), value.end(), [](unsigned char c) { return std::isdigit(c); });
}

bool containsToken(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

std::optional<std::filesystem::path> resolveModelPath(const std::string& modelArg) {
    std::filesystem::path candidate(modelArg);
    if (std::filesystem::exists(candidate)) {
        return candidate;
    }

    std::filesystem::path modelsDir = g_models_dir;
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

    // Check which keys user has already set
    if (std::getenv("NRVNA_TEMP")) lockedKeys.insert("NRVNA_TEMP");

    std::unordered_map<std::string, std::string> applied;

    // Sampling defaults - let runner.cpp use model-aware defaults
    // Only override temperature for specific model types
    if (containsToken(filename, "coder") || containsToken(filename, "code")) {
        applyDefaultEnv("NRVNA_TEMP", "0.3", lockedKeys, applied);  // more deterministic for code
    } else if (containsToken(filename, "deepseek") || containsToken(filename, "r1")) {
        applyDefaultEnv("NRVNA_TEMP", "0.6", lockedKeys, applied);  // reasoning models
    }
    // No artificial n_predict limits - runner.cpp uses model's context size

    if (!applied.empty()) {
        LOG_INFO("Applied default params: " + std::to_string(applied.size()));
        for (const auto& entry : applied) {
            LOG_DEBUG("  " + entry.first + "=" + entry.second);
        }
    }
}

int main(int argc, char* argv[]) {
    g_models_dir = resolveModelsDir(argv[0]);

    // Handle --help and --version before anything else
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            std::cout << "\033[2J\033[1;1H";
            printDashboard();
            std::cout << "\n";
            std::cout << "  \033[1mUSAGE\033[0m\n\n";
            std::cout << "    nrvnad <model.gguf> <workspace>  select model · assign workspace · start\n";
            std::cout << "    wrk <workspace> \"prompt\"         submit work\n";
            std::cout << "    flw <workspace> [job-id]         collect results\n";
            std::cout << "\n";
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
    int workers = 4;

    // Parse all flags first, collect positional args
    std::vector<std::string> positionalArgs;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-w" || arg == "--workers") && i + 1 < argc) {
            try {
                workers = std::stoi(argv[++i]);
            } catch (...) {
                std::cerr << "Error: Invalid worker count\n";
                return 1;
            }
        } else if (arg == "--mmproj" && i + 1 < argc) {
            mmprojPath = argv[++i];
        } else if (arg == "--workspace" && i + 1 < argc) {
            workspace = argv[++i];
        } else if (arg[0] != '-') {
            positionalArgs.push_back(arg);
        }
    }

    // Extract model and workspace from positional args if not set via flags
    if (!positionalArgs.empty() && modelPath.empty()) {
        modelPath = positionalArgs[0];
    }
    if (positionalArgs.size() > 1 && workspace.empty()) {
        workspace = positionalArgs[1];
    }
    if (positionalArgs.size() > 2) {
        try {
            int w = std::stoi(positionalArgs[2]);
            if (w >= 1 && w <= 64) workers = w;
        } catch (...) {}
    }

    bool cliMode = !modelPath.empty();
    Logger::setLevel(cliMode ? LogLevel::INFO : LogLevel::ERROR);
    if (!cliMode) {
        setenv("NRVNA_QUIET", "1", 1);
    }

    if (!cliMode) {
        // Interactive mode - show dashboard then go straight to selection
        std::cout << "\033[2J\033[1;1H";
        auto result = printDashboard();

        auto selection = promptUnifiedSelection(result.workspaces, result.models, workers);
        if (!selection) {
            return 0;
        }
        modelPath = selection->modelPath;
        workspace = selection->workspace;
        mmprojPath = selection->mmprojPath;
    } else if (workspace.empty()) {
        std::cerr << "Error: workspace required\n";
        std::cerr << "Usage: nrvnad <model> <workspace> [--mmproj <path>] [-w <n>]\n";
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

    applyModelDefaults(std::filesystem::path(modelPath));
    recordWorkspacePath(std::filesystem::path(workspace));

    std::cout << "\033[2J\033[1;1H";
    std::cout << "\n";
    std::cout << "  \033[1mnrvna\033[0m " << VERSION << "                        \033[90masync · inference · primitive\033[0m\n";
    std::cout << "  \033[90m─────────────────────────────────────────────────────────────────\033[0m\n";
    std::cout << "\n";

    std::string modelName = std::filesystem::path(modelPath).filename().string();
    std::cout << "  Loading " << modelName << "\n" << std::flush;

    try {
        std::unique_ptr<Server> server;
        if (!mmprojPath.empty()) {
            server = std::make_unique<Server>(modelPath, mmprojPath, workspace, workers);
        } else {
            server = std::make_unique<Server>(modelPath, workspace, workers);
        }

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

        // Write model info to workspace for dashboard
        {
            std::ofstream mf(workspace + "/.model");
            if (mf) mf << modelName;
        }
        if (!mmprojPath.empty()) {
            std::ofstream mp(workspace + "/.mmproj");
            if (mp) mp << mmprojPath;
        }

        std::cout << "\n";
        std::cout << "  \033[1mRUNNING\033[0m\n\n";
        std::cout << "    Model      " << modelName << "\n";
        std::cout << "    Workers    " << workers << "\n";
        std::cout << "    Workspace  " << workspace << "\n";
        if (!mmprojPath.empty()) {
            std::cout << "    MMProj     " << mmprojPath << "\n";
        }
        std::cout << "\n";
        std::cout << "  \033[90m─────────────────────────────────────────────────────────────────\033[0m\n";
        std::cout << "\n";
        auto latest = latestJobId(std::filesystem::path(workspace));
        std::cout << "  Submit:  ./wrk " << workspace << " \"prompt\"\n";
        std::cout << "  Results: ./flw " << workspace;
        if (latest) {
            std::cout << " " << *latest;
        } else {
            std::cout << " <job-id>";
        }
        std::cout << "\n";
        if (latest) {
            std::cout << "  \033[90mLatest job:\033[0m " << *latest << "\n";
        }
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
        {
            std::error_code ec;
            std::filesystem::remove(pidPath, ec);
        }
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
