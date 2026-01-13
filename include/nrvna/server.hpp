/*
 * nrvna ai - Asynchronous Inference Primitive
 * Copyright (c) 2025 Sanmathi Bharamgouda
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include <atomic>
#include <filesystem>
#include <memory>
#include <thread>

namespace nrvnaai {

class Scanner;
class Pool;
class Processor;

class Server final {
public:
    Server(const std::string& modelPath, const std::filesystem::path& workspace, int workers = 4);
    ~Server();

    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;
    Server(Server&&) = delete;
    Server& operator=(Server&&) = delete;

    [[nodiscard]] bool start();
    void shutdown() noexcept;
    [[nodiscard]] const std::filesystem::path& workspace() const noexcept { return workspace_; }
    [[nodiscard]] bool isRunning() const noexcept { return running_.load(); }

private:
    [[nodiscard]] bool createWorkspace() noexcept;
    [[nodiscard]] bool recoverOrphanedJobs() noexcept;
    void scanLoop();

    std::string modelPath_;
    std::filesystem::path workspace_;
    int workers_;
    
    std::atomic<bool> running_{false};
    std::atomic<bool> shutdown_{false};
    
    std::unique_ptr<Scanner> scanner_;
    std::unique_ptr<Pool> pool_;
    std::unique_ptr<Processor> processor_;
    
    std::thread scannerThread_;
};

}