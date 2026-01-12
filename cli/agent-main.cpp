// agent-main.cpp — Minimal autonomous agent with dynamic n_predict via env
// Safest, simplest, no regressions.

#include <iostream>
#include <fstream>
#include <filesystem>
#include <thread>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include "nrvna/work.hpp"

namespace fs = std::filesystem;

std::string readfile(const fs::path& p) {
    std::ifstream f(p);
    if (!f) return "";
    return std::string((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
}

// ------------------ Minimal memory strategy ------------------
std::string load_memory(const fs::path& ws, size_t max_chars = 2000) {
    std::vector<fs::path> dirs;
    for (auto& e : fs::directory_iterator(ws / "output"))
        if (e.is_directory()) dirs.push_back(e);

    if (dirs.empty()) return "";

    std::sort(dirs.begin(), dirs.end(),
        [](auto& a, auto& b) {
            return fs::last_write_time(a) < fs::last_write_time(b);
        });

    std::string memory;

    // Always include first (plan)
    {
        std::string first = readfile(dirs.front() / "result.txt");
        if (!first.empty()) {
            memory += "[PLAN]\n";
            memory += first.substr(0, 500);
            memory += "\n\n";
        }
    }

    // Then add recent outputs until full
    for (auto it = dirs.rbegin(); it != dirs.rend(); ++it) {
        std::string out = readfile(*it / "result.txt");
        if (out.empty()) continue;

        if (memory.size() + out.size() < max_chars) {
            memory += out + "\n---\n";
        } else {
            size_t remaining = max_chars - memory.size();
            if (remaining > 50)
                memory += out.substr(0, remaining);
            break;
        }
    }

    return memory;
}

// ---------------------------- Wait ---------------------------
void wait_for(const fs::path& ws, const std::string& job_id) {
    fs::path out = ws / "output";

    for (;;) {
        for (auto& e : fs::directory_iterator(out)) {
            if (!e.is_directory()) continue;

            if (e.path().filename().string().find(job_id) != std::string::npos) {
                fs::path r = e.path() / "result.txt";
                if (fs::exists(r) && fs::file_size(r) > 0)
                    return;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

// -------------------- Token schedule -------------------------
int token_budget_for_step(int step, int total) {
    if (step == 1) return 256;        // outline
    if (step < total) return 768;    // content writing
    return 1500;                     // final merge
}

// ----------------------------- Main --------------------------
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: ./agent <workspace> \"goal\" [iterations]\n";
        return 1;
    }

    fs::path ws = argv[1];
    std::string goal = argv[2];
    int iters = (argc > 3) ? std::stoi(argv[3]) : 4;

    fs::create_directories(ws / "input/ready");
    fs::create_directories(ws / "output");

    nrvnaai::Work work(ws);

    for (int i = 1; i <= iters; i++) {
        std::cout << "\n\033[1;34m=== AGENT LOOP: ITERATION " << i << " ===\033[0m\n";

        // Note: Dynamic token budget via setenv("NRVNA_PREDICT") only works if 
        // the agent IS the runner. Since we are using a remote nrvnad server,
        // we rely on the server's global NRVNA_PREDICT setting (set to 2048 in script).

        std::cout << "[AGENT] 🧠 Reading workspace memory (context)..." << std::endl;
        std::string memory = load_memory(ws);

        std::string prompt =
            "You are an autonomous agent.\n"
            "Goal: " + goal + "\n\n"
            "Memory:\n" + memory + "\n\n"
            "Continue the task.\n"
            "DO NOT describe steps.\n"
            "Write the actual content for the next step.\n"
            "If the ENTIRE Goal is met, end with EXACTLY: DONE";

        std::cout << "[AGENT] ⚡ Using 'Work' primitive to submit job..." << std::endl;
        auto job = work.submit(prompt);
        std::cout << "[AGENT] 🆔 Job ID: " << job.id << " (Async processing started)" << std::endl;
        
        std::cout << "[AGENT] ⏳ Waiting for async inference..." << std::endl;
        wait_for(ws, job.id);

        // show snippet
        std::vector<fs::path> outs;
        for (auto& e : fs::directory_iterator(ws / "output"))
            if (e.is_directory()) outs.push_back(e);

        std::sort(outs.begin(), outs.end(),
            [](auto& a, auto& b) {
                return fs::last_write_time(a) < fs::last_write_time(b);
            });

        std::string result = readfile(outs.back() / "result.txt");
        std::cout << "[AGENT] 📥 Retrieved result (" << result.size() << " bytes)\n";
        std::cout << "\033[1;32m[OUTPUT]\033[0m " << result.substr(0, 200) << "...\n";

        if (result.find("DONE") != std::string::npos) {
            std::cout << "\033[1;32m[AGENT] ✅ Goal Achieved (DONE signal received).\033[0m\n";
            break;
        }
    }

    std::cout << "\nFinal outputs in: " << (ws / "output") << "\n";
    return 0;
}
