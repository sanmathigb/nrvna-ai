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
    std::vector<WorkspaceInfo> selectable;  // All non-running workspaces (for interactive start)
    std::vector<ModelInfo> models;
};

struct DaemonSelection {
    std::string modelPath;
    std::string workspace;
    std::string mmprojPath;
};

DashboardResult printDashboard() {
    auto models = scanModels();
    auto workspaces = scanWorkspaces();

    // Split workspaces into running and selectable (pending + completed)
    std::vector<WorkspaceInfo> running, selectable;
    for (const auto& ws : workspaces) {
        if (ws.model.empty()) continue;
        auto pidPath = std::filesystem::path(ws.path) / ".nrvnad.pid";
        if (auto pid = readPidFile(pidPath); pid && isProcessAlive(*pid)) {
            running.push_back(ws);
        } else {
            selectable.push_back(ws);
        }
    }
    // Sort selectable: pending first, then stopped, then idle (by path within each)
    std::sort(selectable.begin(), selectable.end(), [](const auto& a, const auto& b) {
        // Priority: pending (queued > 0) > stopped (daemonStopped) > idle
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

    // Running workspaces (daemon active)
    if (!running.empty()) {
        std::cout << "  \033[1mRUNNING\033[0m  \033[90mdaemon processing\033[0m\n\n";
        constexpr size_t maxDisplay = 6;
        size_t displayed = 0;
        for (const auto& ws : running) {
            if (displayed >= maxDisplay) {
                std::cout << "    \033[90m+" << (running.size() - maxDisplay) << " more\033[0m\n";
                break;
            }
            std::string displayPath = ws.path;
            if (displayPath.size() > 16) displayPath = displayPath.substr(0, 13) + "...";
            std::cout << "    \033[36m" << std::left << std::setw(20) << displayPath << "\033[0m  ";
            std::cout << "\033[90m" << std::left << std::setw(10) << ws.model << "\033[0m  ";
            std::cout << "\033[32m" << ws.processing << " processing\033[0m";
            if (ws.queued > 0) std::cout << "  \033[33m" << ws.queued << " queued\033[0m";
            std::cout << "\n";
            ++displayed;
        }
        std::cout << "\n";
    }

    // Selectable workspaces (can start daemon)
    // Split into: pending (queued jobs) vs stopped (daemon killed) vs idle
    std::vector<WorkspaceInfo> pending, stopped, idle;
    for (const auto& ws : selectable) {
        if (ws.queued > 0) {
            pending.push_back(ws);
        } else if (ws.daemonStopped) {
            stopped.push_back(ws);
        } else {
            idle.push_back(ws);
        }
    }

    // PENDING - urgent, jobs waiting
    if (!pending.empty()) {
        std::cout << "  \033[1mPENDING\033[0m  \033[33mjobs queued - needs restart\033[0m\n\n";
        int idx = 1;
        for (const auto& ws : pending) {
            std::string displayPath = ws.path;
            if (displayPath.size() > 16) displayPath = displayPath.substr(0, 13) + "...";
            std::cout << "    \033[33m[" << idx << "]\033[0m  ";
            std::cout << "\033[36m" << std::left << std::setw(16) << displayPath << "\033[0m  ";
            std::cout << "\033[90m" << std::left << std::setw(10) << ws.model << "\033[0m  ";
            std::cout << "\033[33;1m" << ws.queued << " queued\033[0m";
            if (ws.done > 0) std::cout << "  \033[32m" << ws.done << " done\033[0m";
            if (ws.failed > 0) std::cout << "  \033[31m" << ws.failed << " failed\033[0m";
            std::cout << "\n";
            ++idx;
        }
        std::cout << "\n";
    }

    // STOPPED - daemon was killed, can resume
    if (!stopped.empty()) {
        std::cout << "  \033[1mSTOPPED\033[0m  \033[90mdaemon exited - can resume\033[0m\n\n";
        int idx = static_cast<int>(pending.size()) + 1;
        for (const auto& ws : stopped) {
            std::string displayPath = ws.path;
            if (displayPath.size() > 16) displayPath = displayPath.substr(0, 13) + "...";
            std::cout << "    \033[33m[" << idx << "]\033[0m  ";
            std::cout << "\033[36m" << std::left << std::setw(16) << displayPath << "\033[0m  ";
            std::cout << "\033[90m" << std::left << std::setw(10) << ws.model << "\033[0m  ";
            if (ws.done > 0) std::cout << "\033[32m" << ws.done << " done\033[0m  ";
            if (ws.failed > 0) std::cout << "\033[31m" << ws.failed << " failed\033[0m";
            std::cout << "\n";
            ++idx;
        }
        std::cout << "\n";
    }

    // IDLE - old workspaces
    if (!idle.empty()) {
        std::cout << "  \033[1mWORKSPACES\033[0m  \033[90midle\033[0m\n\n";
        constexpr size_t maxDisplay = 4;
        int idx = static_cast<int>(pending.size() + stopped.size()) + 1;
        size_t displayed = 0;
        for (const auto& ws : idle) {
            if (displayed >= maxDisplay) {
                std::cout << "    \033[90m+" << (idle.size() - maxDisplay) << " more\033[0m\n";
                break;
            }
            std::string displayPath = ws.path;
            if (displayPath.size() > 16) displayPath = displayPath.substr(0, 13) + "...";
            std::cout << "    \033[33m[" << idx << "]\033[0m  ";
            std::cout << "\033[36m" << std::left << std::setw(16) << displayPath << "\033[0m  ";
            std::cout << "\033[90m" << std::left << std::setw(10) << ws.model << "\033[0m  ";
            if (ws.done > 0) std::cout << "\033[32m" << ws.done << " done\033[0m  ";
            if (ws.failed > 0) std::cout << "\033[31m" << ws.failed << " failed\033[0m";
            std::cout << "\n";
            ++idx;
            ++displayed;
        }
        std::cout << "\n";
    }

    // Models - show with numbers for direct selection (offset by workspace count)
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
                    // Just show them all if only a few more
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

    return {selectable, models};
}

void printHelpCommands() {
    std::cout << "\n";
    std::cout << "  \033[90m─────────────────────────────────────────────────────────────────\033[0m\n";
    std::cout << "\n";
    std::cout << "  ./nrvnad <model> <workspace>      start daemon\n";
    std::cout << "  ./wrk <workspace> \"prompt\"        submit job\n";
    std::cout << "  ./flw <workspace> [job-id]        retrieve result\n";
    std::cout << "\n";
}

std::string toLower(std::string value);
bool isNumber(const std::string& value);
bool confirmWorkspaceReuse(const std::string& workspace);

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

// Prompt for mmproj - Enter to skip, or provide path
std::string promptMmproj() {
    std::cout << "  \033[90mMMProj [Enter=skip, or path]:\033[0m ";
    std::cout.flush();

    std::string input;
    if (!std::getline(std::cin, input)) {
        return "";
    }
    input = trim(input);

    if (input.empty()) {
        return "";
    }

    // Check if path exists
    if (std::filesystem::exists(std::filesystem::path(input))) {
        return input;
    }

    // Try in models directory
    auto inModels = g_models_dir / input;
    if (std::filesystem::exists(inModels)) {
        return inModels.string();
    }

    std::cout << "  \033[31mMMProj not found, skipping\033[0m\n";
    return "";
}

std::optional<DaemonSelection> promptUnifiedSelection(
    const std::vector<WorkspaceInfo>& workspaces,
    const std::vector<ModelInfo>& models,
    int& workers
);

std::optional<DaemonSelection> promptUnifiedSelection(
    const std::vector<WorkspaceInfo>& workspaces,
    const std::vector<ModelInfo>& models,
    int& workers
) {
    size_t wsCount = workspaces.size();
    size_t modelCount = models.size();

    // Count workspaces by category (matches dashboard order: pending, stopped, idle)
    size_t pendingCount = 0, stoppedCount = 0;
    for (const auto& ws : workspaces) {
        if (ws.queued > 0) ++pendingCount;
        else if (ws.daemonStopped) ++stoppedCount;
    }
    size_t idleCount = wsCount - pendingCount - stoppedCount;

    while (true) {
        // Show hint about what numbers mean
        std::cout << "\n  \033[1mSTART DAEMON\033[0m\n\n";
        size_t nextIdx = 1;
        if (pendingCount > 0) {
            std::cout << "    \033[33m[" << nextIdx;
            if (pendingCount > 1) std::cout << "-" << (nextIdx + pendingCount - 1);
            std::cout << "]\033[0m  Resume pending (jobs waiting)\n";
            nextIdx += pendingCount;
        }
        if (stoppedCount > 0) {
            std::cout << "    \033[90m[" << nextIdx;
            if (stoppedCount > 1) std::cout << "-" << (nextIdx + stoppedCount - 1);
            std::cout << "]\033[0m  Resume stopped workspace\n";
            nextIdx += stoppedCount;
        }
        if (idleCount > 0) {
            std::cout << "    \033[90m[" << nextIdx;
            if (idleCount > 1) std::cout << "-" << (nextIdx + idleCount - 1);
            std::cout << "]\033[0m  Resume idle workspace\n";
            nextIdx += idleCount;
        }
        if (modelCount > 0) {
            std::cout << "    \033[90m[" << nextIdx;
            if (modelCount > 1) std::cout << "-" << (nextIdx + modelCount - 1);
            std::cout << "]\033[0m  New workspace with model\n";
            std::cout << "    \033[90mor type model name (e.g., mistral, qwen)\033[0m\n";
        }
        std::cout << "\n";
        std::cout << "    \033[90mm = list all models    h = help    Enter = back\033[0m\n";
        std::cout << "\n  \033[90mChoice:\033[0m ";
        std::cout.flush();

        std::string input;
        if (!std::getline(std::cin, input)) {
            return std::nullopt;
        }
        input = trim(input);
        if (input.empty()) {
            return std::nullopt;
        }

        if (input == "h" || input == "H" || input == "help") {
            printHelpCommands();
            continue;
        }

        if (input == "m" || input == "M" || input == "more") {
            // Show all models with numbers
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

        // Try as number first
        if (isNumber(input)) {
            int choice = std::stoi(input);

            // Is it a workspace selection?
            if (choice >= 1 && choice <= static_cast<int>(wsCount)) {
                const auto& ws = workspaces[choice - 1];
                std::cout << "\n  Starting \033[36m" << ws.path << "\033[0m with \033[36m" << ws.model << "\033[0m\n";
                workers = promptWorkers();
                return DaemonSelection{ws.model, ws.path, ws.mmproj};
            }

            // Is it a model selection by number?
            int modelIdx = choice - static_cast<int>(wsCount) - 1;
            if (modelIdx >= 0 && modelIdx < static_cast<int>(modelCount)) {
                const auto& model = models[modelIdx];
                std::cout << "\n  Selected \033[36m" << model.filename << "\033[0m\n";

                auto workspace = promptWorkspacePath();
                if (!workspace) {
                    continue;
                }

                workers = promptWorkers();
                auto modelPath = (g_models_dir / model.filename).string();
                std::string mmprojPath = promptMmproj();

                return DaemonSelection{modelPath, *workspace, mmprojPath};
            }

            std::cout << "  \033[31mInvalid number\033[0m\n";
            continue;
        }

        // Try as model name search
        std::string needle = toLower(input);
        std::vector<size_t> matches;
        for (size_t i = 0; i < models.size(); ++i) {
            std::string shortLower = toLower(models[i].shortName);
            std::string fileLower = toLower(models[i].filename);
            if (shortLower.find(needle) != std::string::npos ||
                fileLower.find(needle) != std::string::npos) {
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

        // Single match - use it
        {
            const auto& model = models[matches[0]];
            std::cout << "\n  Selected \033[36m" << model.filename << "\033[0m\n";

            auto workspace = promptWorkspacePath();
            if (!workspace) {
                continue;
            }

            workers = promptWorkers();
            auto modelPath = (g_models_dir / model.filename).string();
            std::string mmprojPath = promptMmproj();

            return DaemonSelection{modelPath, *workspace, mmprojPath};
        }
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

bool confirmWorkspaceReuse(const std::string& workspace) {
    if (!std::filesystem::exists(std::filesystem::path(workspace))) {
        return true;
    }

    std::cout << "  \033[33mWorkspace exists. Reuse? [y/N]:\033[0m ";
    std::cout.flush();
    std::string input;
    if (!std::getline(std::cin, input)) {
        return false;
    }
    input = toLower(trim(input));
    return input == "y" || input == "yes";
}

std::optional<std::string> promptWorkspacePath() {
    while (true) {
        std::cout << "\n  \033[90mWorkspace name \033[0m\033[90m(created if new)\033[0m\n";
        std::cout << "  \033[90mExamples: ws_coding, ./experiments/run1\033[0m\n";
        std::cout << "  \033[90m[workspace]:\033[0m ";
        std::cout.flush();

        std::string workspace;
        if (!std::getline(std::cin, workspace)) {
            return std::nullopt;
        }
        workspace = trim(workspace);
        if (workspace.empty()) {
            workspace = "workspace";
        }
        if (!confirmWorkspaceReuse(workspace)) {
            continue;
        }
        return workspace;
    }
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

    // Set log level based on mode:
    // - CLI mode (with args): show INFO logs for visibility
    // - Interactive mode: suppress logs for clean dashboard UI
    bool cliMode = (argc >= 3);
    Logger::setLevel(cliMode ? LogLevel::INFO : LogLevel::ERROR);
    if (!cliMode) {
        setenv("NRVNA_QUIET", "1", 1);  // Suppress mtmd timing logs in interactive mode
    }

    if (argc < 3) {
        // Interactive mode - show status dashboard first
        std::cout << "\033[2J\033[1;1H";
        auto result = printDashboard();

        // Quick tips
        std::cout << "  \033[90m───────────────────────────────────────────────────────────────\033[0m\n";
        std::cout << "\n";
        std::cout << "  ./nrvnad <model> <workspace>      start daemon\n";
        std::cout << "  ./wrk <workspace> \"prompt\"        submit job\n";
        std::cout << "  ./flw <workspace> [job-id]        retrieve result\n";
        std::cout << "\n";
        std::cout << "  \033[90mEnter = start interactive    q = quit\033[0m\n";
        std::cout << "  ";
        std::cout.flush();

        std::string gate;
        if (!std::getline(std::cin, gate)) {
            return 0;
        }
        gate = trim(gate);
        if (gate == "q" || gate == "Q" || gate == "quit") {
            return 0;
        }

        // Action tray - select what to do
        auto selection = promptUnifiedSelection(result.selectable, result.models, workers);
        if (!selection) {
            return 0;
        }
        modelPath = selection->modelPath;
        workspace = selection->workspace;
        mmprojPath = selection->mmprojPath;
    } else {
        // CLI mode: nrvnad <model> <workspace> [--mmproj <path>] [-w <n>]
        modelPath = argv[1];
        workspace = argv[2];
    }

    for (int i = 3; i < argc; ++i) {
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
        } else if (isNumber(arg) && workers == 4) {
            try {
                workers = std::stoi(arg);
            } catch (...) {
                std::cerr << "Error: Invalid worker count\n";
                return 1;
            }
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
            if (mf) mf << extractShortName(modelName);
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
