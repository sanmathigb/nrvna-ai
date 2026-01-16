/*
 * nrvna ai - Flow retrieval tool (flw)
 * Copyright (c) 2025 Sanmathi Bharamgouda
 * SPDX-License-Identifier: MIT
 */

#include "nrvna/flow.hpp"
#include "nrvna/logger.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <unistd.h>

using namespace nrvnaai;

void printUsage(const char* progName) {
    std::cout << "nrvna-ai Flow Retrieval Tool\n\n";
    std::cout << "Usage: " << progName << " <workspace> [job_id]\n\n";
    std::cout << "Arguments:\n";
    std::cout << "  workspace     Directory for job storage\n";
    std::cout << "  job_id        Specific job ID to retrieve (optional)\n\n";
    std::cout << "Behavior:\n";
    std::cout << "  - If job_id provided: retrieve specific job\n";
    std::cout << "  - If no job_id: retrieve latest completed job\n\n";
    std::cout << "Environment Variables:\n";
    std::cout << "  NRVNA_LOG_LEVEL    Log level (ERROR, WARN, INFO, DEBUG, TRACE)\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << progName << " ./workspace\n";
    std::cout << "  " << progName << " ./workspace 1731808123456_12345_0\n";
    std::cout << "  NRVNA_LOG_LEVEL=DEBUG " << progName << " ./workspace\n";
}

const char* statusToString(Status status) {
    switch (status) {
        case Status::Queued: return "QUEUED";
        case Status::Running: return "RUNNING";
        case Status::Done: return "DONE";
        case Status::Failed: return "FAILED";
        case Status::Missing: return "MISSING";
        default: return "UNKNOWN";
    }
}

int main(int argc, char* argv[]) {
    // Silence logs for CLI tool usage
    Logger::setLevel(LogLevel::WARN);

    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string workspace = argv[1];
    std::string jobId = "";
    bool wait = false;
    
    // Parse args
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-w" || arg == "--wait") {
            wait = true;
        } else {
            jobId = arg;
        }
    }

    // Check piped input for JobID if not provided
    if (jobId.empty() && !isatty(fileno(stdin))) {
        std::cin >> jobId;
    }

    try {
        Flow flow(workspace);

        // Resolve ID (Specific or Latest)
        if (jobId.empty()) {
             auto latest = flow.latest();
             if (latest) jobId = latest->id;
             else {
                 std::cerr << "No jobs found" << std::endl;
                 return 1;
             }
        }

        // Wait loop
        if (wait) {
            while (true) {
                Status s = flow.status(jobId);
                if (s == Status::Done || s == Status::Failed) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }

        if (!jobId.empty()) {
            // Retrieve specific job
            auto job = flow.get(jobId);
            
            if (!job.has_value()) {
                std::cerr << "Job not found: " << jobId << std::endl;
                return 1;
            }

            if (job->status == Status::Done) {
                std::cout << job->content << std::endl;
                return 0;
            } else if (job->status == Status::Failed) {
                std::cerr << "Job failed: " << jobId << std::endl;
                if (!job->content.empty()) {
                    std::cerr << "Error: " << job->content << std::endl;
                }
                return 1;
            } else {
                std::cerr << "Job not ready: " << jobId << " (status: " << statusToString(job->status) << ")" << std::endl;
                return 2; // Different exit code for "not ready"
            }

        } else {
            // Should be unreachable now given logic above
            return 1; 
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Error: Unknown error retrieving job\n";
        return 1;
    }
}