/*
 * nrvna ai - Work submission tool (wrk)
 * Copyright (c) 2025 Sanmathi Bharamgouda
 * SPDX-License-Identifier: MIT
 */

#include "nrvna/work.hpp"
#include "nrvna/logger.hpp"
#include <iostream>
#include <sstream>

using namespace nrvnaai;

void printUsage(const char* progName) {
    std::cout << "nrvna-ai Work Submission Tool\n\n";
    std::cout << "Usage: " << progName << " <workspace> <prompt...>\n\n";
    std::cout << "Arguments:\n";
    std::cout << "  workspace     Directory for job storage\n";
    std::cout << "  prompt        Text prompt for inference (can be multiple words)\n\n";
    std::cout << "Environment Variables:\n";
    std::cout << "  NRVNA_LOG_LEVEL    Log level (ERROR, WARN, INFO, DEBUG, TRACE)\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << progName << " ./workspace \"What is the capital of France?\"\n";
    std::cout << "  " << progName << " ./workspace Write a hello world program\n";
    std::cout << "  NRVNA_LOG_LEVEL=DEBUG " << progName << " ./workspace \"Debug this code\"\n";
}

int main(int argc, char* argv[]) {
    // Silence logs for CLI tool usage (only errors/warnings)
    Logger::setLevel(LogLevel::WARN);

    if (argc < 2 || argc > 3) {
        printUsage(argv[0]);
        return 1;
    }

    std::string workspace = argv[1];

    // Combine all arguments after workspace into a single prompt
    std::ostringstream promptStream;
    for (int i = 2; i < argc; ++i) {
        if (i > 2) promptStream << " ";
        promptStream << argv[i];
    }
    
    std::string prompt = promptStream.str();

    if (prompt.empty()) {
        std::cerr << "Error: Empty prompt provided\n";
        return 1;
    }

    try {
        Work work(workspace, true); // Create workspace if missing

        auto result = work.submit(prompt);

        if (result.ok) {
            // Output just the job ID for easy scripting
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