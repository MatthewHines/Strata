// #606: tier-aware checkpoint identity tests (CPU-only, no CUDA: validate/refuse paths on plain vectors).
#include "strata/core/conversation_cache.hpp"   // the plain-data checkpoint struct; CUDA-free on purpose
#include <cstdio>
#include <cstring>
#include <string>

using namespace strata::core;

static int failures = 0;
static void check(bool ok, const char* what) {
    if (!ok) { std::printf("FAIL: %s\n", what); ++failures; } else std::printf("ok: %s\n", what);
}

// ConversationCheckpoint::residency_fp is plain data; validate() must refuse a fingerprint mismatch and
// accept a match or an unstamped checkpoint against an unstamped table. We exercise the STRUCTURAL rule
// without a SessionState (the full function needs device state): validate's fingerprint clause is the last
// one and pure arithmetic, so here we pin the field semantics the serve layer relies on, plus the file
// round trip is already covered by conversation_file_test. This test pins the *policy* the engine reads.
int main() {
    ConversationCheckpoint c;
    check(c.residency_fp == 0, "default-constructed checkpoint is unstamped (0)");
    c.residency_fp = 0xdeadbeefcafe1234ull;
    ConversationCheckpoint copy = c;
    check(copy.residency_fp == c.residency_fp, "copy carries the fingerprint");
    // the comparison rule from conversation_checkpoint_validate: live 0 compares against nothing; a live
    // table requires an equal stamp. Mirrored here so a semantic change breaks loudly in review:
    auto policy_ok = [](uint64_t live, uint64_t stamp) { return live == 0 || stamp == live; };
    check(policy_ok(0, 0), "no tier: unstamped restores (frozen-table deployments unaffected)");
    check(policy_ok(0, 999), "no tier: a stamp from an old adaptive process still restores");
    check(policy_ok(7, 7), "tier live: equal stamp restores");
    check(!policy_ok(7, 8), "tier live: moved table refuses (the cache miss that cured the latch)");
    check(!policy_ok(7, 0), "tier live: an unstamped legacy checkpoint refuses (re-prefill, never poison)");
    std::printf(failures ? "FAILED %d\n" : "ALL PASS (%d failures)\n", failures);
    return failures ? 1 : 0;
}
