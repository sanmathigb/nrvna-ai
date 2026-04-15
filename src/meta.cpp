/*
 * nrvna ai - Asynchronous Inference Primitive
 * Copyright (c) 2025 Sanmathi Bharamgouda
 * SPDX-License-Identifier: MIT
 */

#include "nrvna/meta.hpp"
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace nrvnaai {

std::string escapeJson(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:   out += c; break;
        }
    }
    return out;
}

namespace {

std::string unescapeJson(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            ++i;
            switch (s[i]) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                default: out += s[i]; break;
            }
        } else {
            out += s[i];
        }
    }
    return out;
}

std::string extractString(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\": \"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return "";
    pos += needle.size();

    std::string result;
    bool escape = false;
    for (size_t i = pos; i < json.size(); ++i) {
        char c = json[i];
        if (escape) {
            result += '\\';
            result += c;
            escape = false;
            continue;
        }
        if (c == '\\') {
            escape = true;
            continue;
        }
        if (c == '"') {
            return unescapeJson(result);
        }
        result += c;
    }
    return "";
}

double extractDouble(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\": ";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return -1.0;
    pos += needle.size();
    auto end = json.find_first_of(",\n}", pos);
    try {
        return std::stod(json.substr(pos, end - pos));
    } catch (...) {
        return -1.0;
    }
}

std::vector<std::string> extractStringArray(const std::string& json, const std::string& key) {
    std::vector<std::string> result;
    std::string needle = "\"" + key + "\": [";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return result;
    pos += needle.size();
    auto end = json.find(']', pos);
    if (end == std::string::npos) return result;

    bool in_string = false;
    bool escape = false;
    std::string current;
    for (size_t i = pos; i < end; ++i) {
        char c = json[i];
        if (!in_string) {
            if (c == '"') {
                in_string = true;
                current.clear();
            }
            continue;
        }
        if (escape) {
            current += '\\';
            current += c;
            escape = false;
            continue;
        }
        if (c == '\\') {
            escape = true;
            continue;
        }
        if (c == '"') {
            in_string = false;
            result.push_back(unescapeJson(current));
            continue;
        }
        current += c;
    }
    return result;
}

} // namespace

std::string formatTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count() % 1000000;

    struct tm tm_buf;
    gmtime_r(&time, &tm_buf);

    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm_buf);

    char result[64];
    std::snprintf(result, sizeof(result), "%s.%06ldZ", buf, static_cast<long>(us));
    return result;
}

std::string jobTypeToString(JobType type) {
    switch (type) {
        case JobType::Text: return "text";
        case JobType::Embed: return "embed";
        case JobType::Vision: return "vision";
        case JobType::Tts: return "tts";
        default: return "text";
    }
}

bool writeMetaJson(const std::filesystem::path& dir, const JobMeta& meta) {
    try {
        std::ostringstream json;
        json << "{\n";
        json << "  \"submitted_at\": \"" << escapeJson(meta.submitted_at) << "\",\n";
        json << "  \"mode\": \"" << escapeJson(meta.mode) << "\"";

        if (!meta.parent.empty()) {
            json << ",\n  \"parent\": \"" << escapeJson(meta.parent) << "\"";
        }

        if (!meta.tags.empty()) {
            json << ",\n  \"tags\": [";
            for (size_t i = 0; i < meta.tags.size(); ++i) {
                if (i > 0) json << ", ";
                json << "\"" << escapeJson(meta.tags[i]) << "\"";
            }
            json << "]";
        }

        if (!meta.status.empty()) {
            json << ",\n  \"completed_at\": \"" << escapeJson(meta.completed_at) << "\"";
            json << ",\n  \"duration_s\": " << std::fixed << std::setprecision(2) << meta.duration_s;
            json << ",\n  \"artifacts\": [";
            for (size_t i = 0; i < meta.artifacts.size(); ++i) {
                if (i > 0) json << ", ";
                json << "\"" << escapeJson(meta.artifacts[i]) << "\"";
            }
            json << "]";
            json << ",\n  \"status\": \"" << escapeJson(meta.status) << "\"";
        }

        json << "\n}\n";

        auto tmpPath = dir / "meta.json.tmp";
        auto finalPath = dir / "meta.json";

        {
            std::ofstream file(tmpPath, std::ios::binary);
            if (!file) return false;
            file << json.str();
            file.flush();
            if (!file.good()) return false;
        }

        std::filesystem::rename(tmpPath, finalPath);
        return true;
    } catch (...) {
        return false;
    }
}

std::optional<JobMeta> readMetaJson(const std::filesystem::path& dir) {
    try {
        auto metaPath = dir / "meta.json";
        if (!std::filesystem::exists(metaPath)) return std::nullopt;

        std::ifstream file(metaPath, std::ios::binary);
        if (!file) return std::nullopt;

        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());

        JobMeta meta;
        meta.submitted_at = extractString(content, "submitted_at");
        meta.mode = extractString(content, "mode");
        meta.parent = extractString(content, "parent");
        meta.tags = extractStringArray(content, "tags");
        meta.completed_at = extractString(content, "completed_at");
        meta.duration_s = extractDouble(content, "duration_s");
        meta.artifacts = extractStringArray(content, "artifacts");
        meta.status = extractString(content, "status");

        return meta;
    } catch (...) {
        return std::nullopt;
    }
}

} // namespace nrvnaai
