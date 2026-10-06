#pragma once
// include/strata/core/dbg_probe.hpp - a debug switch a running process can flip between requests.
//
// The state probes (STATE_HASH's session read-back, DBG_NAN's per-chunk activation checks) sync the
// stream and copy device buffers, so a boot-time environment variable commits the whole session to
// their cost - on a 176k-prompt agent that is minutes per turn, enough to make production unusable.
// `dbg_probe_on("NAME")` reads a flag file instead: touch /tmp/strata_dbg/NAME (inside the
// container) while a request runs to arm the probe, remove it to disarm, and the next request end or
// the next prefill chunk (checked at most once every 2 s, so the stat itself stays free) sees the
// change.  STRATA_DBG_FORCE=1 in the environment at boot restores the old meaning of the variables:
// STRATA_<NAME> set = on, files ignored - the investigation-window switch.
//
// Usage:  if (strata::core::dbg_probe_on("STATE_HASH")) { ... }

#include <chrono>
#include <cstdlib>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <unordered_map>

namespace strata::core {

inline bool dbg_probe_on(const char* name) {
    static const bool forced = std::getenv("STRATA_DBG_FORCE") != nullptr;
    if (forced) return std::getenv(("STRATA_" + std::string(name)).c_str()) != nullptr;
    static std::mutex mx;
    static std::unordered_map<std::string, std::pair<bool, std::chrono::steady_clock::time_point>> cache;
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(mx);
    auto& entry = cache[name];
    if (now - entry.second > std::chrono::seconds(2)) {
        struct stat st;
        entry.first = stat(("/tmp/strata_dbg/" + std::string(name)).c_str(), &st) == 0;
        entry.second = now;
    }
    return entry.first;
}

}  // namespace strata::core
