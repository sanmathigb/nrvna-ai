/*
 * nrvna ai - Work submission tool (wrk)
 * Copyright (c) 2025 Sanmathi Bharamgouda
 * SPDX-License-Identifier: MIT
 */

#include "nrvna/work.hpp"
#include "nrvna/logger.hpp"
#include <filesystem>
#include <iostream>
#include <sstream>
#include <iterator>
#include <unistd.h>

using namespace nrvnaai;

constexpr const char* VERSION = "0.1.0";

void printUsage(const char* progName) {
    std::cout << "nrvna-ai Work Submission Tool v" << VERSION << "\n\n";
    std::cout << "Usage: " << progName << " <workspace> <prompt...> [--image <path> ...]\n";
    std::cout << "       " << progName << " <workspace> <text> --embed\n";
    std::cout << "       " << progName << " <workspace> -     (read prompt from stdin)\n";
    std::cout << "       " << progName << " --help | --version\n\n";
    std::cout << "Arguments:\n";
    std::cout << "  workspace     Directory for job storage\n";
    std::cout << "  prompt        Text prompt for inference (can be multiple words)\n";
    std::cout << "  -             Read prompt from stdin\n\n";
    std::cout << "Options:\n";
    std::cout << "  --image <path>  Attach image (repeatable)\n";
    std::cout << "  --embed         Submit as embedding job (returns vector)\n";
    std::cout << "  -h, --help      Show this help message\n";
    std::cout << "  -v, --version   Show version\n\n";
    std::cout << "Environment Variables:\n";
    std::cout << "  NRVNA_LOG_LEVEL    Log level (ERROR, WARN, INFO, DEBUG, TRACE)\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << progName << " ./workspace \"What is the capital of France?\"\n";
    std::cout << "  " << progName << " ./workspace Write a hello world program\n";
    std::cout << "  " << progName << " ./workspace \"Machine learning is...\" --embed\n";
    std::cout << "  echo \"Hello\" | " << progName << " ./workspace -\n";
}

int main(int argc, char* argv[]) {
    // Default to WARN for clean piping; NRVNA_LOG_LEVEL overrides
    if (!std::getenv("NRVNA_LOG_LEVEL"))
        Logger::setLevel(LogLevel::WARN);

    // Handle --help and --version before anything else
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
        if (arg == "-v" || arg == "--version") {
            std::cout << VERSION << "\n";
            return 0;
        }
    }

    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string workspace = argv[1];
    std::string prompt;
    std::vector<std::filesystem::path> imagePaths;
    bool useEmbed = false;

    // Check for stdin input
    bool readStdin = false;
    if (argc == 2 && !isatty(fileno(stdin))) {
        readStdin = true;
    } else if (argc >= 3 && std::string(argv[2]) == "-") {
        readStdin = true;
    }

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--image" || arg == "-i") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --image requires a path\n";
                return 1;
            }
            imagePaths.emplace_back(argv[++i]);
        } else if (arg == "--embed") {
            useEmbed = true;
        }
    }

    if (readStdin) {
        prompt.assign((std::istreambuf_iterator<char>(std::cin)),
                       std::istreambuf_iterator<char>());
        // Remove strictly trailing newline if prompt is just a one-liner
        if (!prompt.empty() && prompt.back() == '\n') {
            prompt.pop_back();
        }
    } else {
        if (argc < 3) {
            printUsage(argv[0]);
            return 1;
        }

        std::ostringstream promptStream;
        bool first = true;
        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--image" || arg == "-i") {
                ++i;
                continue;
            }
            if (arg == "--embed") continue;  // Skip --embed from prompt
            if (!first) promptStream << " ";
            promptStream << argv[i];
            first = false;
        }
        prompt = promptStream.str();
    }

    if (prompt.empty()) {
        std::cerr << "Error: Empty prompt provided\n";
        return 1;
    }

    try {
        Work work(workspace, true); // Create workspace if missing

        SubmitResult result;
        if (useEmbed) {
            result = work.submit(prompt, JobType::Embed);
        } else if (!imagePaths.empty()) {
            result = work.submit(prompt, imagePaths);
        } else {
            result = work.submit(prompt);
        }

        if (result.ok) {
            // Just the job ID - clean for piping, no noise
            std::cout << result.id << std::endl;
            return 0;
        } else {
            std::cerr << "Error: " << result.message << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Error: Unknown error submitting job\n";
        return 1;
    }
}
