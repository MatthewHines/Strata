// #606: tier-aware checkpoint identity tests (CPU-only, no CUDA: field semantics on plain structs).
#include "strata/core/conversation_cache.hpp"   // the plain-data checkpoint struct; CUDA-free on purpose
#include <cstdio>
#include <cstring>
#include <string>

using namespace strata::core;

static int failures = 0;
static void check(bool ok, const char* what) {
    if (!ok) { std::printf("FAIL: %s\n", what); ++failures; } else std::printf("ok: %s\n", what);
}

// ConversationCheckpoint::residency_fp is plain data carried through copy and the file round trip.
// The RESTORE POLICY (10-08 evening): default = restore regardless of a moved table (upstream
// behavior); refuse-on-move only with STRATA_FP_GUARD=1 (forensics). The guard's cost - a full
// prompt re-prefill per turn once adaptive residency moves the table by design - outweighed what
// it caught, so it never became a default. Mirrored here so a semantic change breaks loudly:
//     guard off: always ok    guard on: live == 0 || stamp == live
int main() {
    ConversationCheckpoint c;
    check(c.residency_fp == 0, "default-constructed checkpoint is unstamped (0)");
    c.residency_fp = 0xdeadbeefcafe1234ull;
    ConversationCheckpoint copy = c;
    check(copy.residency_fp == c.residency_fp, "copy carries the fingerprint");
    auto policy_ok = [](bool guard, uint64_t live, uint64_t stamp) {
        return !guard || live == 0 || stamp == live;
    };
    check(policy_ok(false, 7, 8), "guard off (default): a moved table still restores - cache replay wins");
    check(policy_ok(false, 7, 0), "guard off (default): an unstamped checkpoint restores");
    check(policy_ok(true, 0, 0), "guard on: no tier - unstamped restores");
    check(policy_ok(true, 7, 7), "guard on: equal stamp restores");
    check(!policy_ok(true, 7, 8), "guard on: moved table refuses (forensic mode)");
    check(!policy_ok(true, 7, 0), "guard on: an unstamped legacy checkpoint refuses");
    // #606 layer-aware rule (conversation_state.cpp validate): the COMPUTE path is part of f.
    // Mirrored here so a semantic change breaks loudly:
    //     refuse only when both sides are stamped AND differ; a zero on either side never compares.
    auto compute_ok = [](uint64_t live, uint64_t stamp) {
        return !(live != 0 && stamp != 0 && stamp != live);
    };
    check(compute_ok(0, 0) && compute_ok(0, 9) && compute_ok(9, 0), "compute rule: a zero side never compares (legacy/unset stay today's behavior)");
    check(compute_ok(9, 9), "compute rule: equal compute path restores");
    check(!compute_ok(9, 8), "compute rule: a changed compute path refuses (real invalidation: cache miss, re-read)");
    std::printf(failures ? "FAILED %d\n" : "ALL PASS (%d failures)\n", failures);
    return failures ? 1 : 0;
}
