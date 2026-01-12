/*
 * nrvna ai - Asynchronous Inference Primitive
 * Copyright (c) 2025 Sanmathi Bharamgouda
 * SPDX-License-Identifier: MIT
 */

#include "nrvna/server.hpp"
#include "nrvna/scanner.hpp"
#include "nrvna/pool.hpp"
#include "nrvna/processor.hpp"
#include "nrvna/runner.hpp"
#include "nrvna/logger.hpp"
#include <csignal>
#include <chrono>
#include <thread>

namespace nrvnaai {

static std::atomic<bool> g_shutdown_requested{false};

Server::Server(const std::string& modelPath, const std::filesystem::path& workspace, int workers)
    : modelPath_(modelPath), workspace_(workspace), workers_(workers) {
    LOG_DEBUG("Server created - model: " + modelPath + ", workspace: " + workspace_.string() + 
              ", workers: " + std::to_string(workers));
}

Server::~Server() {
    shutdown();
}

bool Server::start() {
    if (running_.load()) {
        LOG_WARN("Server already running");
        return false;
    }

    LOG_INFO("Starting nrvna-ai server...");

    // Setup signal handlers
    std::signal(SIGINT, handleSignal);
    std::signal(SIGTERM, handleSignal);

    // Create workspace
    if (!createWorkspace()) {
        LOG_ERROR("Failed to create workspace");
        return false;
    }

    // Name the main thread
    setThreadName("Main");

    // Log startup banner
    LOG_DEBUG("========================================");
    LOG_DEBUG("nrvna-ai Server Starting");
    LOG_DEBUG("========================================");
    LOG_DEBUG("Model: " + modelPath_);
    LOG_DEBUG("Workspace: " + workspace_.string());
    LOG_DEBUG("Workers: " + std::to_string(workers_));
    LOG_DEBUG("nrvna Log Level: " + std::string(getenv("NRVNA_LOG_LEVEL") ? getenv("NRVNA_LOG_LEVEL") : "INFO"));
    LOG_DEBUG("llama.cpp Log Level: " + std::string(getenv("LLAMA_LOG_LEVEL") ? getenv("LLAMA_LOG_LEVEL") : "ERROR"));
    LOG_DEBUG("========================================");

    // Create components
    try {
        scanner_ = std::make_unique<Scanner>(workspace_);
        pool_ = std::make_unique<Pool>(workers_);
        processor_ = std::make_unique<Processor>(workspace_, modelPath_);

        // CRITICAL: Pre-initialize all Runners BEFORE starting worker threads
        // This ensures ggml_backend_load_all() is called sequentially from main thread
        // Fixes Metal backend threading crashes on AMD GPU
        LOG_DEBUG("Pre-initializing " + std::to_string(workers_) + " Runner instances...");
        if (!processor_->initializeRunners(workers_)) {
            LOG_ERROR("Failed to initialize runners");
            return false;
        }
        LOG_DEBUG("All " + std::to_string(workers_) + " Runner instances initialized successfully");

        // Start pool with processor function
        LOG_DEBUG("Starting worker pool with " + std::to_string(workers_) + " threads...");
        if (!pool_->start([this](const JobId& jobId, int workerId) {
            (void)processor_->process(jobId, workerId);
        })) {
            LOG_ERROR("Failed to start worker pool");
            return false;
        }

        running_.store(true);

        // Start scanner loop in background
        scannerThread_ = std::thread(&Server::scanLoop, this);

        LOG_DEBUG("Server started successfully");
        return true;

    } catch (const std::exception& e) {
        LOG_ERROR("Failed to start server: " + std::string(e.what()));
        return false;
    }
}

void Server::shutdown() noexcept {
    if (!running_.load()) {
        return;
    }

    LOG_INFO("Shutting down server...");
    
    shutdown_.store(true);
    running_.store(false);

    // Stop scanner thread
    if (scannerThread_.joinable()) {
        scannerThread_.join();
    }

    // Stop pool
    if (pool_) {
        pool_->stop();
    }

    // Clean up components
    processor_.reset();
    pool_.reset();
    scanner_.reset();

    LOG_INFO("Server shutdown complete");
}

bool Server::createWorkspace() noexcept {
    try {
        std::filesystem::create_directories(workspace_ / "input" / "writing");
        std::filesystem::create_directories(workspace_ / "input" / "ready");
        std::filesystem::create_directories(workspace_ / "processing");
        std::filesystem::create_directories(workspace_ / "output");
        std::filesystem::create_directories(workspace_ / "failed");
        
        LOG_DEBUG("Workspace created: " + workspace_.string());
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to create workspace: " + std::string(e.what()));
        return false;
    }
}

void Server::scanLoop() {
    LOG_DEBUG("Scanner loop started");

    const auto scanInterval = std::chrono::milliseconds(1000); // 1 second scan interval

    while (!shutdown_.load()) {
        try {
            // Scan for new jobs
            auto jobs = scanner_->scan();
            
            // Submit found jobs to pool
            for (const auto& jobId : jobs) {
                if (shutdown_.load()) break;
                pool_->submit(jobId);
            }

            if (!jobs.empty()) {
                LOG_DEBUG("Submitted " + std::to_string(jobs.size()) + " jobs to pool");
            }

            // Sleep with interruption check
            auto sleepEnd = std::chrono::steady_clock::now() + scanInterval;
            while (std::chrono::steady_clock::now() < sleepEnd && !shutdown_.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

        } catch (const std::exception& e) {
            LOG_ERROR("Scanner loop error: " + std::string(e.what()));
            std::this_thread::sleep_for(scanInterval);
        } catch (...) {
            LOG_ERROR("Unknown scanner loop error");
            std::this_thread::sleep_for(scanInterval);
        }
    }

    LOG_DEBUG("Scanner loop stopped");
}

void Server::handleSignal(int signal) {
    LOG_INFO("Received signal " + std::to_string(signal) + ", requesting shutdown...");
    g_shutdown_requested.store(true);
}

}