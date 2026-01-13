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

using namespace nrvnaai;

// Async-signal-safe: only set flag, no complex operations
static volatile sig_atomic_t g_shutdown_requested = 0;

void signalHandler(int signal) {
    (void)signal;
    g_shutdown_requested = 1;
}

void printBanner() {
    std::cout << "\n";
    std::cout << "   \033[1m_ __      _ __      __   __     _ __       __ _ \033[0m\n";
    std::cout << "  \033[1m| '_ \\    | '__|     \\ \\ / /    | '_ \\     / _` |\033[0m\n";
    std::cout << "  \033[1m| | | |   | |         \\ V /     | | | |   | (_| |\033[0m\n";
    std::cout << "  \033[1m|_| |_|   |_|          \\_/      |_| |_|    \\__,_|\033[0m\n\n";

    std::cout << "             \033[90masync   ·   inference primitive\033[0m\n\n";
}

void printUsage(const char* progName) {
    (void)progName;
    printBanner();

    std::cout << "  \033[1mwrkflw\033[0m\n";
    std::cout << "    1. start daemon       $ ./nrvnad model.gguf workspace\n";
    std::cout << "    2. submit prompt      $ ./wrk workspace \"prompt\"\n";
    std::cout << "    3. retrieve inference $ ./flw workspace <job_id>\n\n";
}

int main(int argc, char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "--design") {
        std::cout << "\033[2J\033[1;1H";
        printUsage(argv[0]);
        return 0;
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

    std::cout << "\033[2J\033[1;1H";
    printBanner();

    LOG_DEBUG("nrvna-ai daemon starting...");

    try {
        Server server(modelPath, workspace, workers);

        if (!server.start()) {
            LOG_ERROR("Failed to start server");
            return 1;
        }

        std::cout << "\n";
        std::cout << "  \033[32m●\033[0m listening on " << workspace << "\n";
        std::cout << "  \033[90m" << modelPath << "\033[0m\n";
        std::cout << "\n";

        while (!g_shutdown_requested && server.isRunning()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        if (g_shutdown_requested) {
            std::cout << "\nShutdown requested, stopping server..." << std::endl;
        }
        LOG_DEBUG("Shutdown requested, stopping server...");
        server.shutdown();

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
