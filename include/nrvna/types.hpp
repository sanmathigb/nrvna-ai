#pragma once
#include <cstdint>
#include <string>

namespace nrvnaai {

// Core job lifecycle states.
enum class Status : std::uint8_t { Queued, Running, Done, Failed, Missing };

// Opaque job identifier (string-based for now; can evolve to strong type).
using JobId = std::string;

} // namespace nrvnaai
