// agent-tools.cpp - Agent with bash tool calling
// Usage: ./agent-tools <workspace> <goal> [iterations]

#include <iostream>
#include <fstream>
#include <filesystem>
#include <thread>
#include <vector>
#include <algorithm>
#include <array>
#include "nrvna/work.hpp"

namespace fs = std::filesystem;

std::string read(const fs::path& p) {
    std::ifstream f(p);
    return f ? std::string((std::istreambuf_iterator<char>(f)), {}) : "";
}

std::string latest(const fs::path& ws) {
    std::vector<fs::path> dirs;
    for (auto& e : fs::directory_iterator(ws / "output"))
        if (e.is_directory()) dirs.push_back(e);
    if (dirs.empty()) return "";
    std::sort(dirs.begin(), dirs.end(), [](auto& a, auto& b) {
        return fs::last_write_time(a) > fs::last_write_time(b);
    });
    return read(dirs[0] / "result.txt");
}

void wait_for(const std::string& job, const fs::path& ws) {
    for (int i = 0; i < 300; i++) {
        for (auto& e : fs::directory_iterator(ws / "output")) {
            if (e.path().filename().string().find(job) != std::string::npos) {
                auto r = e.path() / "result.txt";
                if (fs::exists(r) && fs::file_size(r) > 0) return;
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

std::string exec(const std::string& cmd) {
    std::array<char, 128> buf;
    std::string out;
    FILE* pipe = popen((cmd + " 2>&1").c_str(), "r");
    if (!pipe) return "ERROR";
    while (fgets(buf.data(), buf.size(), pipe)) out += buf.data();
    pclose(pipe);
    return out.substr(0, 2000);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <workspace> <goal> [iters]\n";
        return 1;
    }
    
    fs::path ws = argv[1];
    std::string goal = argv[2];
    int iters = (argc > 3) ? std::stoi(argv[3]) : 5;
    
    fs::create_directories(ws / "input/ready");
    fs::create_directories(ws / "output");
    nrvnaai::Work work(ws);
    
    for (int i = 1; i <= iters; i++) {
        std::cout << "\n=== ITERATION " << i << " ===\n";
        
        std::string mem = latest(ws);
        std::string prompt = "Goal: " + goal + 
            "\nPrevious: " + mem.substr(0, 500) +
            "\n\nNext step? Reply with bash command OR explanation.";
        
        auto r = work.submit(prompt);
        wait_for(r.id, ws);
        
        std::string action = latest(ws);
        std::cout << "Action: " << action.substr(0, 100) << "...\n";
        
        // Try executing as bash (simple heuristic: has $, |, or ends with common commands)
        if (action.find("$") != std::string::npos || 
            action.find("|") != std::string::npos ||
            action.find("ls") != std::string::npos ||
            action.find("curl") != std::string::npos) {
            
            std::cout << "[EXEC] " << action.substr(0, 80) << "\n";
            std::string out = exec(action);
            std::cout << "Output: " << out.substr(0, 150) << "...\n";
            
            // Feed back
            r = work.submit("Command output:\n" + out + "\n\nWhat did you learn?");
            wait_for(r.id, ws);
        }
    }
    
    std::cout << "\nDone: " << ws / "output" << "\n";
    return 0;
}
