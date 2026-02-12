/*
 * nrvna ai - Web dashboard (nrvna-web)
 * Copyright (c) 2025 Sanmathi Bharamgouda
 * SPDX-License-Identifier: MIT
 */

#include "nrvna/work.hpp"
#include "nrvna/logger.hpp"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <csignal>
#include <cstdlib>
#include <cctype>
#include <unistd.h>

// Vendored in third_party/llama.cpp
#include "httplib.h"
#include "json.hpp"

using json = nlohmann::json;
using namespace nrvnaai;

constexpr const char* VERSION = "0.1.0";

static volatile sig_atomic_t g_shutdown_requested = 0;
static std::string g_api_token;

void signalHandler(int) {
    g_shutdown_requested = 1;
}

static bool authRequired() {
    return !g_api_token.empty();
}

static bool isAuthorized(const httplib::Request& req) {
    if (!authRequired()) {
        return true;
    }

    const std::string xToken = req.get_header_value("X-NRVNA-TOKEN");
    if (xToken == g_api_token) {
        return true;
    }

    const std::string auth = req.get_header_value("Authorization");
    const std::string prefix = "Bearer ";
    if (auth.rfind(prefix, 0) == 0 && auth.substr(prefix.size()) == g_api_token) {
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Workspace scanning — mirrors nrvnad.cpp logic
// ---------------------------------------------------------------------------

struct WorkspaceInfo {
    std::string path;
    std::string model;
    size_t queued = 0;
    size_t running = 0;
    size_t done = 0;
    size_t failed = 0;
    bool daemonRunning = false;
    bool daemonStopped = false;  // had PID file but process is dead
};

static std::string trim(std::string s) {
    auto notSpace = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), notSpace));
    s.erase(std::find_if(s.rbegin(), s.rend(), notSpace).base(), s.end());
    return s;
}

static size_t countDirs(const std::filesystem::path& dir) {
    size_t n = 0;
    if (std::filesystem::exists(dir)) {
        for (const auto& e : std::filesystem::directory_iterator(dir))
            if (e.is_directory()) ++n;
    }
    return n;
}

static bool isWorkspace(const std::filesystem::path& dir) {
    return std::filesystem::exists(dir / "input" / "ready") &&
           std::filesystem::exists(dir / "input" / "writing");
}

static bool isProcessAlive(pid_t pid) {
    if (pid <= 0) return false;
    if (::kill(pid, 0) == 0) return true;
    return errno == EPERM;
}

static std::filesystem::path normalizePath(const std::filesystem::path& p) {
    return std::filesystem::absolute(p).lexically_normal();
}

static std::string displayPath(const std::filesystem::path& path) {
    std::error_code ec;
    auto abs = std::filesystem::absolute(path, ec);
    auto cwd = std::filesystem::current_path();
    if (!ec) {
        auto rel = abs.lexically_relative(cwd);
        auto s = rel.string();
        if (!rel.empty() && s.rfind("..", 0) != 0) {
            if (s == ".") return "./";
            if (s.rfind("./", 0) == 0) return s;
            return "./" + s;
        }
    }
    return path.string();
}

static WorkspaceInfo readInfo(const std::filesystem::path& p, const std::string& dp) {
    WorkspaceInfo ws;
    ws.path = dp;
    ws.queued   = countDirs(p / "input" / "ready");
    ws.running  = countDirs(p / "processing");
    ws.done     = countDirs(p / "output");
    ws.failed   = countDirs(p / "failed");

    std::ifstream pidFile(p / ".nrvnad.pid");
    if (pidFile) {
        long pid = 0;
        pidFile >> pid;
        if (pid > 0) {
            ws.daemonRunning = isProcessAlive(static_cast<pid_t>(pid));
            ws.daemonStopped = !ws.daemonRunning;
        }
    }

    std::ifstream mf(p / ".model");
    if (mf) std::getline(mf, ws.model);

    return ws;
}

static std::vector<WorkspaceInfo> scanWorkspaces() {
    std::vector<WorkspaceInfo> out;
    auto cwd = std::filesystem::current_path();
    std::unordered_set<std::string> seen;

    // Local directories
    for (const auto& e : std::filesystem::directory_iterator(cwd)) {
        if (!e.is_directory()) continue;
        if (e.path().filename().string()[0] == '.') continue;
        if (!isWorkspace(e.path())) continue;
        out.push_back(readInfo(e.path(), "./" + e.path().filename().string()));
        seen.insert(normalizePath(e.path()).string());
    }

    // Remote paths from .nrvna-workspaces
    std::ifstream hist(cwd / ".nrvna-workspaces");
    if (hist) {
        std::string line;
        while (std::getline(hist, line)) {
            line = trim(line);
            if (line.empty()) continue;
            auto p = normalizePath(std::filesystem::path(line));
            if (seen.count(p.string())) continue;
            if (!std::filesystem::exists(p) || !isWorkspace(p)) continue;
            out.push_back(readInfo(p, p.string()));
            seen.insert(p.string());
        }
    }

    std::sort(out.begin(), out.end(),
              [](const auto& a, const auto& b) { return a.path < b.path; });
    return out;
}

// ---------------------------------------------------------------------------
// Per-workspace Work + mutex
// ---------------------------------------------------------------------------

struct WorkEntry {
    std::mutex mu;
    Work work;
    explicit WorkEntry(const std::filesystem::path& ws) : work(ws, false) {}
};

static std::mutex g_map_mutex;
static std::unordered_map<std::string, std::unique_ptr<WorkEntry>> g_work_map;

static WorkEntry& getWork(const std::string& workspace) {
    std::lock_guard<std::mutex> lk(g_map_mutex);
    auto it = g_work_map.find(workspace);
    if (it != g_work_map.end()) return *it->second;
    auto [inserted, ok] = g_work_map.emplace(
        workspace, std::make_unique<WorkEntry>(workspace));
    return *inserted->second;
}

// ---------------------------------------------------------------------------
// Embedded HTML dashboard
// ---------------------------------------------------------------------------

static const char* DASHBOARD_HTML = R"HTML(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>nrvna</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#111;color:#e0e0e0;padding:16px;max-width:640px;margin:0 auto}
h1{font-size:1.4rem;margin-bottom:4px}
.sub{color:#888;font-size:.85rem;margin-bottom:20px}
.card{background:#1a1a1a;border:1px solid #333;border-radius:8px;padding:14px;margin-bottom:12px}
.card h2{font-size:1rem;color:#4fc3f7;margin-bottom:6px;display:flex;align-items:center;gap:8px}
.model{color:#888;font-size:.85rem}
.counts{display:flex;gap:10px;margin-top:8px;flex-wrap:wrap}
.badge{font-size:.8rem;padding:2px 8px;border-radius:4px;font-weight:600}
.q{background:#33302a;color:#ffa726}.r{background:#1a2a33;color:#4fc3f7}
.d{background:#1a331a;color:#66bb6a}.f{background:#331a1a;color:#ef5350}
.status{font-size:.75rem;padding:2px 8px;border-radius:4px;font-weight:600}
.status.on{background:#1a331a;color:#66bb6a}.status.stopped{background:#33302a;color:#ffa726}.status.off{background:#222;color:#666}
hr{border:none;border-top:1px solid #333;margin:20px 0}
label{display:block;font-size:.9rem;margin-bottom:6px;color:#aaa}
select,textarea{width:100%;background:#1a1a1a;color:#e0e0e0;border:1px solid #333;border-radius:6px;padding:10px;font-size:1rem;font-family:inherit}
textarea{min-height:100px;resize:vertical}
button{background:#4fc3f7;color:#111;border:none;border-radius:6px;padding:12px 24px;font-size:1rem;font-weight:600;cursor:pointer;margin-top:10px;min-height:44px;min-width:44px}
button:active{opacity:.8}
button:disabled{opacity:.5;cursor:not-allowed}
.msg{margin-top:10px;padding:10px;border-radius:6px;font-size:.9rem}
.msg.ok{background:#1a331a;color:#66bb6a}.msg.warn{background:#33302a;color:#ffa726}.msg.err{background:#331a1a;color:#ef5350}
.empty{color:#666;text-align:center;padding:40px 0}
</style>
</head>
<body>
<h1>nrvna</h1>
<p class="sub">async &middot; inference &middot; primitive</p>

<div id="workspaces"><p class="empty">loading&hellip;</p></div>

<hr>

<label for="ws">Workspace</label>
<select id="ws"></select>

<label for="prompt" style="margin-top:12px">Prompt</label>
<textarea id="prompt" placeholder="Type your prompt here..."></textarea>

<button id="submit" onclick="submitJob()">Submit</button>
<div id="msg"></div>

<script>
function render(data){
  const c=document.getElementById('workspaces');
  const s=document.getElementById('ws');
  if(!data.length){c.innerHTML='<p class="empty">No workspaces found</p>';s.innerHTML='';return}
  c.innerHTML=data.map(w=>`<div class="card">
    <h2>${w.path}<span class="status ${w.daemonRunning?'on':w.daemonStopped?'stopped':'off'}">${w.daemonRunning?'running':w.daemonStopped?'stopped':'idle'}</span></h2>
    <div class="model">${w.model||'(no model)'}</div>
    <div class="counts">
      ${w.queued?`<span class="badge q">${w.queued} queued</span>`:''}
      ${w.running?`<span class="badge r">${w.running} running</span>`:''}
      ${w.done?`<span class="badge d">${w.done} done</span>`:''}
      ${w.failed?`<span class="badge f">${w.failed} failed</span>`:''}
    </div></div>`).join('');
  const prev=s.value;
  s.innerHTML=data.map(w=>`<option value="${w.path}">${w.path}</option>`).join('');
  if(prev&&[...s.options].some(o=>o.value===prev))s.value=prev;
}
var _status=[];
function refresh(){fetch('/api/status').then(r=>r.json()).then(d=>{_status=d;render(d)}).catch(()=>{})}
refresh();setInterval(refresh,5000);

function submitJob(){
  const ws=document.getElementById('ws').value;
  const p=document.getElementById('prompt').value.trim();
  const m=document.getElementById('msg');
  const btn=document.getElementById('submit');
  if(!ws){m.className='msg err';m.textContent='Select a workspace';return}
  if(!p){m.className='msg err';m.textContent='Enter a prompt';return}
  btn.disabled=true;
  fetch('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({workspace:ws,prompt:p})})
  .then(r=>r.json()).then(d=>{
    if(d.ok){
      var info=_status.find(w=>w.path===ws);
      if(info&&!info.daemonRunning){m.className='msg warn';m.textContent='Queued (daemon not running) \u2014 start nrvnad to process'}
      else{m.className='msg ok';m.textContent='Submitted: '+d.id}
      document.getElementById('prompt').value='';refresh();
    } else{m.className='msg err';m.textContent=d.error||'Submit failed'}
  }).catch(e=>{m.className='msg err';m.textContent='Network error'})
  .finally(()=>{btn.disabled=false});
}
document.getElementById('prompt').addEventListener('keydown',e=>{
  if((e.metaKey||e.ctrlKey)&&e.key==='Enter')submitJob();
});
</script>
</body>
</html>
)HTML";

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    int port = 8080;
    std::string bind = "127.0.0.1";
    if (const char* token = std::getenv("NRVNA_WEB_TOKEN")) {
        g_api_token = token;
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-p" || arg == "--port") && i + 1 < argc) {
            try { port = std::stoi(argv[++i]); }
            catch (...) { std::cerr << "Error: invalid port\n"; return 1; }
        } else if ((arg == "-b" || arg == "--bind") && i + 1 < argc) {
            bind = argv[++i];
        } else if ((arg == "-t" || arg == "--token") && i + 1 < argc) {
            g_api_token = argv[++i];
        } else if (arg == "-v" || arg == "--version") {
            std::cout << VERSION << "\n";
            return 0;
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "nrvna-web — HTTP dashboard for nrvna workspaces\n\n"
                      << "Usage: nrvna-web [options]\n\n"
                      << "  -p, --port <port>   Port (default: 8080)\n"
                      << "  -b, --bind <addr>   Bind address (default: 127.0.0.1)\n"
                      << "  -t, --token <tok>   API token for /api/jobs (or NRVNA_WEB_TOKEN)\n"
                      << "  -v, --version\n"
                      << "  -h, --help\n";
            return 0;
        }
    }

    // Suppress nrvna logs for clean output (web binary doesn't do inference)
    Logger::setLevel(LogLevel::WARN);

    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    // Show discovered workspaces at startup
    auto workspaces = scanWorkspaces();
    std::cerr << "nrvna-web " << VERSION << "\n";
    std::cerr << "Discovered " << workspaces.size() << " workspace"
              << (workspaces.size() != 1 ? "s" : "") << "\n";
    for (const auto& ws : workspaces) {
        std::cerr << "  " << ws.path;
        if (ws.daemonRunning) std::cerr << "  [running]";
        else if (ws.daemonStopped) std::cerr << "  [stopped]";
        std::cerr << "\n";
    }
    std::cerr << "Listening on " << bind << ":" << port << "\n";
    if (authRequired()) {
        std::cerr << "API token auth enabled for POST /api/jobs\n";
    }

    httplib::Server svr;

    // GET / — embedded dashboard
    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(DASHBOARD_HTML, "text/html");
    });

    // GET /api/status — all workspaces with counts
    svr.Get("/api/status", [](const httplib::Request&, httplib::Response& res) {
        auto workspaces = scanWorkspaces();
        json arr = json::array();
        for (const auto& ws : workspaces) {
            arr.push_back({
                {"path",    ws.path},
                {"model",   ws.model},
                {"daemonRunning",  ws.daemonRunning},
                {"daemonStopped",  ws.daemonStopped},
                {"queued",  ws.queued},
                {"running", ws.running},
                {"done",    ws.done},
                {"failed",  ws.failed}
            });
        }
        res.set_content(arr.dump(), "application/json");
    });

    // POST /api/jobs — submit text prompt
    svr.Post("/api/jobs", [](const httplib::Request& req, httplib::Response& res) {
        if (!isAuthorized(req)) {
            res.status = 401;
            res.set_content(json({{"ok", false}, {"error", "Unauthorized"}}).dump(),
                            "application/json");
            return;
        }

        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(json({{"ok", false}, {"error", "Invalid JSON"}}).dump(),
                            "application/json");
            return;
        }

        if (!body.contains("workspace") || !body.contains("prompt") ||
            !body["workspace"].is_string() || !body["prompt"].is_string()) {
            res.status = 400;
            res.set_content(json({{"ok", false}, {"error", "Missing workspace or prompt"}}).dump(),
                            "application/json");
            return;
        }

        std::string workspace = body["workspace"].get<std::string>();
        std::string prompt = body["prompt"].get<std::string>();

        if (prompt.empty()) {
            res.status = 400;
            res.set_content(json({{"ok", false}, {"error", "Empty prompt"}}).dump(),
                            "application/json");
            return;
        }

        // Validate workspace exists
        std::filesystem::path wsPath(workspace);
        if (!isWorkspace(wsPath)) {
            res.status = 400;
            res.set_content(json({{"ok", false}, {"error", "Invalid workspace"}}).dump(),
                            "application/json");
            return;
        }

        auto& entry = getWork(workspace);
        std::lock_guard<std::mutex> lk(entry.mu);
        auto result = entry.work.submit(prompt);

        if (result.ok) {
            res.set_content(json({{"ok", true}, {"id", result.id}}).dump(),
                            "application/json");
        } else {
            res.status = 500;
            res.set_content(json({{"ok", false}, {"error", result.message}}).dump(),
                            "application/json");
        }
    });

    // Shutdown hook — poll flag so we can break out of listen()
    svr.set_keep_alive_timeout(1);

    // Run server in a thread so we can check shutdown flag
    std::thread serverThread([&]() {
        svr.listen(bind, port);
    });

    while (!g_shutdown_requested) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    std::cerr << "\nShutting down...\n";
    svr.stop();
    serverThread.join();

    return 0;
}
