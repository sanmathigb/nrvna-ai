/*
 * nrvna ai - Flow retrieval tool (flw)
 * Copyright (c) 2025 Sanmathi Bharamgouda
 * SPDX-License-Identifier: MIT
 */

#include "nrvna/flow.hpp"
#include "nrvna/logger.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <chrono>
#include <unistd.h>

using namespace nrvnaai;

constexpr const char* VERSION = "0.1.0";

void printUsage(const char* progName) {
    std::cout << "nrvna-ai Flow Retrieval Tool v" << VERSION << "\n\n";
    std::cout << "Usage: " << progName << " <workspace> [options] [job_id]\n";
    std::cout << "       " << progName << " --help | --version\n\n";
    std::cout << "Arguments:\n";
    std::cout << "  workspace     Directory for job storage\n";
    std::cout << "  job_id        Specific job ID to retrieve (optional)\n\n";
    std::cout << "Options:\n";
    std::cout << "  -w, --wait    Wait for job to complete before returning\n";
    std::cout << "  --json        Output structured JSON\n";
    std::cout << "  -h, --help    Show this help message\n";
    std::cout << "  -v, --version Show version\n\n";
    std::cout << "Behavior:\n";
    std::cout << "  - No job_id: show workspace status (counts + recent jobs)\n";
    std::cout << "  - With job_id: retrieve that job's result\n";
    std::cout << "  - With -w and job_id: wait for job to complete, then print result\n";
    std::cout << "  - Piped input: reads job_id from stdin (wrk ... | flw <ws> -w)\n\n";
    std::cout << "Environment Variables:\n";
    std::cout << "  NRVNA_LOG_LEVEL    Log level (ERROR, WARN, INFO, DEBUG, TRACE)\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << progName << " ./ws                      Show workspace status\n";
    std::cout << "  " << progName << " ./ws --json               Status as JSON\n";
    std::cout << "  " << progName << " ./ws -w <job_id>          Wait and print result\n";
    std::cout << "  wrk ./ws \"Hello\" | " << progName << " ./ws -w   Submit and collect\n";
}

std::string escapeJson(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c; break;
        }
    }
    return out;
}

std::string readFileRaw(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
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

const char* statusToJsonString(Status status) {
    switch (status) {
        case Status::Queued: return "queued";
        case Status::Running: return "running";
        case Status::Done: return "done";
        case Status::Failed: return "failed";
        case Status::Missing: return "missing";
        default: return "unknown";
    }
}

int main(int argc, char* argv[]) {
    // Default to WARN for clean output; NRVNA_LOG_LEVEL overrides
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
    std::string jobId = "";
    bool wait = false;
    bool json = false;
    
    // Parse args
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-w" || arg == "--wait") {
            wait = true;
        } else if (arg == "--json") {
            json = true;
        } else {
            jobId = arg;
        }
    }

    // Check piped input for JobID if not provided
    if (jobId.empty() && !isatty(fileno(stdin))) {
        std::cin >> jobId;
    }

    // Validate job ID format
    if (!jobId.empty() && !Flow::isValidJobId(jobId)) {
        std::cerr << "Invalid job ID: " << jobId << std::endl;
        return 1;
    }

    try {
        Flow flow(workspace);

        // No job ID and no pipe: show workspace status
        if (jobId.empty() && !wait) {
            auto c = flow.counts();
            if (json) {
                std::cout << "{\"queued\":" << c.queued
                          << ",\"running\":" << c.running
                          << ",\"done\":" << c.done
                          << ",\"failed\":" << c.failed << "}\n";
                return 0;
            }

            std::cout << "queued:     " << c.queued << "\n"
                      << "running:    " << c.running << "\n"
                      << "done:       " << c.done << "\n"
                      << "failed:     " << c.failed << "\n";

            // Show recent jobs with duration from meta.json
            auto recentJobs = flow.list(5);
            if (!recentJobs.empty()) {
                std::cout << "\nrecent:\n";
                for (const auto& job : recentJobs) {
                    const char* tag = job.status == Status::Failed ? "failed" : "done";
                    auto m = flow.meta(job.id);
                    if (m && m->duration_s >= 0.0) {
                        char dur[16];
                        std::snprintf(dur, sizeof(dur), "%5.1fs", m->duration_s);
                        std::cout << "  [" << tag << "] " << dur << "  " << job.id << "\n";
                    } else {
                        std::cout << "  [" << tag << "]        " << job.id << "\n";
                    }
                }
            }
            return 0;
        }

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
                if (s == Status::Done || s == Status::Failed || s == Status::Missing) break;
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

            if (json) {
                auto meta = flow.meta(jobId);
                auto outputDir = std::filesystem::path(workspace) / "output" / jobId;
                std::ostringstream out;
                out << "{";
                out << "\"id\":\"" << escapeJson(jobId) << "\"";
                out << ",\"status\":\"" << escapeJson(std::string(statusToJsonString(job->status))) << "\"";
                if (meta) {
                    if (!meta->mode.empty()) out << ",\"mode\":\"" << escapeJson(meta->mode) << "\"";
                    if (!meta->submitted_at.empty()) out << ",\"submitted_at\":\"" << escapeJson(meta->submitted_at) << "\"";
                    if (!meta->completed_at.empty()) out << ",\"completed_at\":\"" << escapeJson(meta->completed_at) << "\"";
                    if (meta->duration_s >= 0.0) out << ",\"duration_s\":" << meta->duration_s;
                    if (!meta->parent.empty()) out << ",\"parent\":\"" << escapeJson(meta->parent) << "\"";
                    if (!meta->tags.empty()) {
                        out << ",\"tags\":[";
                        for (size_t i = 0; i < meta->tags.size(); ++i) {
                            if (i > 0) out << ",";
                            out << "\"" << escapeJson(meta->tags[i]) << "\"";
                        }
                        out << "]";
                    }
                    if (!meta->artifacts.empty()) {
                        out << ",\"artifacts\":[";
                        for (size_t i = 0; i < meta->artifacts.size(); ++i) {
                            if (i > 0) out << ",";
                            out << "\"" << escapeJson(meta->artifacts[i]) << "\"";
                        }
                        out << "]";
                    }
                }

                if (job->status == Status::Done) {
                    auto resultPath = outputDir / "result.txt";
                    auto audioPath = outputDir / "audio.wav";
                    auto embeddingPath = outputDir / "embedding.json";
                    if (std::filesystem::exists(resultPath)) {
                        out << ",\"result\":\"" << escapeJson(job->content) << "\"";
                    } else if (std::filesystem::exists(audioPath)) {
                        out << ",\"audio_path\":\"" << escapeJson(std::filesystem::absolute(audioPath).string()) << "\"";
                    } else if (std::filesystem::exists(embeddingPath)) {
                        out << ",\"embedding\":" << readFileRaw(embeddingPath);
                    }
                } else if (job->status == Status::Failed) {
                    out << ",\"error\":\"" << escapeJson(job->content) << "\"";
                }
                out << "}\n";
                std::cout << out.str();
                return job->status == Status::Failed ? 1 : 0;
            }

            if (job->status == Status::Done) {
                // Check for audio output — print path instead of binary content
                auto audioPath = std::filesystem::path(workspace) / "output" / jobId / "audio.wav";
                if (std::filesystem::exists(audioPath)) {
                    std::cout << std::filesystem::absolute(audioPath).string() << std::endl;
                    return 0;
                }
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

        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Error: Unknown error retrieving job\n";
        return 1;
    }
}
