#!/bin/sh
# Strata entrypoint for the UD-Q4_K_XL / 5090 / :8002 setup on this host.
#
# Model: the 4 UD-Q4_K_XL shards are pre-staged on the host and bound READ-ONLY
# at $GGUF_DIR (default /models). We pass --gguf-dir so setup.py reads them in
# place (no 111 GB re-download, no copy). Their SHA-256 is pre-verified on the
# host (a <shard>.done with "sha256 <sha>" is written next to each), so the
# container's verify_sha256 short-circuits instead of re-hashing 111 GB.
#
# Everything else (context dial $CONTEXT — model max 262144, int8 KV, the
# ~100 GiB expert RAM budget, the MTP draft layer, the expert profile) is
# setup.py's own decision with --yes. Only the ~5 GB MTP draft layer is
# fetched, at first start.
set -e
cd /opt/strata || exit 1

GGUF_DIR="${GGUF_DIR:-/models}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8002}"
# 262144 = the model's trained max (256k) — no RoPE scaling at exactly this
# value (past it setup adds yarn automatically). 2026-10-01: raised from
# 131072 (user: "that's how I'll use it" — run at model max).
CONTEXT="${CONTEXT:-262144}"

# tag = fam["tag"] + model = "unsloth-" + "UD-Q4_K_XL"; setup.py writes
# strata-{tag.lower()}.json at the repo root (/opt/strata).
TAG=unsloth-ud-q4_k_xl
CFG_DIR=/data/config
cfg="$CFG_DIR/strata-$TAG.json"
mkdir -p "$CFG_DIR"

if [ "${REINSTALL:-0}" = "1" ] || [ ! -f "$cfg" ]; then
  echo "First setup: unsloth/UD-Q4_K_XL (model at $GGUF_DIR; fetching the ~5 GB MTP layer)."
  .venv/bin/python setup.py --setup --yes \
    --family unsloth --model UD-Q4_K_XL \
    --context "$CONTEXT" \
    --gguf-dir "$GGUF_DIR" \
    --data-dir /data \
    --host "$HOST" --port "$PORT" --no-start
  cp -f "/opt/strata/strata-$TAG.json" "$cfg"
else
  # reuse the persisted config from the /data volume (skip the setup pass)
  cp -f "$cfg" "/opt/strata/strata-$TAG.json"
fi

# Host overlay (idempotent, applied on EVERY boot after the config lands):
#  - --vram-reserve-mib: keep VRAM free for the desktop (engine's expert
#    cache shrinks by that much; 3072 = user's "reserve ~3GB for the system")
#  - vision on the CPU: the XL family has no GPU-vision wiring upstream
#    (docs/UNSLOTH_Q4.md), and the CPU encoder takes zero VRAM (10-30 s per
#    picture, ~300 image tokens). mmproj lives in the persisted data volume.
# 4 CPU threads on purpose: the CPU expert kernels are the decode path;
# the encoder is a rare burst, not a steady load.
.venv/bin/python - "$cfg" <<'PY'
import json, os, sys
p = sys.argv[1]
cfg = json.load(open(p))
args = cfg["args"]
def setarg(flag, val):
    if flag in args:
        args[args.index(flag) + 1] = val
    else:
        args.extend([flag, val])   # extend, NOT +=: += would rebind args as local
setarg("--vram-reserve-mib", os.environ.get("VRAM_RESERVE_MIB", "3072"))   # 2048->3072 (2026-10-02, user): desktop breathing room back to 3 GiB
# A/B 2026-10-01 (user call): KV window to engine floor 32768->20480. Frees
# ~12k cells/layer of VRAM; --expert-cache auto should absorb it as more hot
# experts. My model says flat-to-down (QSA reads already hit 97.5% inside the
# 32k window; ~0.5 GiB freed != 'crazy amount'). User wants the measurement;
# REVERT here to 32768 if matched-depth median drops (baseline 44.8 @ 96-160k).
setarg("--kv-resident", os.environ.get("KV_RESIDENT", "20480"))
# 0.1.38 #452: --kv q4_0 reads Q4_0 K/V on tensor cores (sm_80+; measured
# +30% at 32K prompts). ADOPTED 2026-10-03 as the host default (user
# approved; expert-cache headroom matters more than the int8 floor).
# KV_MODE=int8 downgrades; k8v4 is the hybrid option. NOTE: this line wins
# over the persisted config on every boot — keep the default = the SoT value.
setarg("--kv", os.environ.get("KV_MODE", "q4_0"))
if "--vision" not in args:
    args.append("--vision")
# decode/prefill tuning (2026-10-01, user-approved 1-4):
#  suffix-draft 7: prompt-lookup drafts up to 7 after a 16-token match
#    (default 3; the eddoursul fork runs 7, ~7.5 tok/verify pass on edit traffic)
#  expert-cache-per-layer: 0.1.33, per-layer VRAM expert slots (CPU pool drain ~halved)
#  conversation-cache: park switched-away conversations in RAM with their
#    checkpoints (default off; user is single-session + occasional 1-2 extra
#    slots -> 4 GiB budget, 4 slots, engine keeps a 2.5 GiB free-RAM floor)
# A/B knobs (2026-10-02): env-overridable, defaults = current tuned values
setarg("--suffix-draft", os.environ.get("SUFFIX_DRAFT", "7"))
# A/B 2026-10-01 (user call): DROP --expert-cache-per-layer. Counter-evidence
# noted for the record: boot fills ~5047 experts / 18.8 GiB either way and
# expert hit rate logged 76-87% WITH per-layer. Decision rule: strata-perf
# median in matched depth buckets; if it drops >5% vs prior era (<32k: ~44-50,
# 160k+: ~38), restore this line. Upstream docs claim per-layer WINS at small
# slot budgets (256->2.97% shared vs 70.4%/64slots); our budget is huge, where
# global arrival-order fill may concentrate on genuinely-hot experts instead.
while "--expert-cache-per-layer" in args:
    args.remove("--expert-cache-per-layer")
setarg("--conversation-cache-mib", "4096")
# PLE n-gram table (28.8 GB IQ4_NL): `ram` = mmap + locked into RAM at start
# (2026-10-01, user: load it immediately; supersedes lazy `mmap`). Hard cost:
# 28.8 GB mlocked from ~43 GiB avail -> ~14 GiB slack stays for the desktop.
# REVERT to direct/mmap if desktop RAM pressure shows up (avail < 6 GiB).
setarg("--ple-io", "ram")
# Calibrated values (tools/calibrate.py run on THIS box, 2026-10-01 ~20:3x):
# worker sweep 7w=39.4 / 5w=40.9 / 4w=39.5 tok/s -> 5 pool workers win (+3.8%,
# single-CCD X3D: fewer GPU-feeding threads beat more); spec floor 0.70 vs
# product default 0.5; PCIe share 0.35. Rerun 'strata-udq4-run.sh calibrate'
# only on hardware/spec change (NOTE: upstream --calibrate drops into
# foreground serving after measuring — kill it once '[ok] tuned' prints).
setarg("--pool-workers", "5")
setarg("--spec-min-p", "0.70")
# Native (IQ) packs REQUIRE --spec T>=2 (+ --mtp) or the engine exits 2 at arg
# validation BEFORE CUDA init ("needs --native SHARD1, --spec T (T >= 2)"). The
# degeneration A/B's SPEC_OFF leg strips them for its own run — it must NEVER
# write that back to the persisted config (that is what broke every 0.1.36 boot
# after 2026-10-02 14:22). Re-inject the defaults so the dumped config is always
# bootable, regardless of which leg just ran.
SPEC_DEFAULTS = {"--spec": "3", "--mtp": "/data/mtp/rt"}
def ensure_spec():
    for flag, val in SPEC_DEFAULTS.items():
        if flag not in args:
            anchor = "--prefill" if "--prefill" in args else args[0]
            i = args.index(anchor)
            args[i:i] = [flag, val]
# SPEC_OFF is a NO-OP on native (IQ) packs and always was in practice: the engine
# hard-exits at arg-validation without --spec >= 2 (+ --mtp), and ensure_spec()
# below re-injects them regardless. A 2026-10-02 A/B "no-speculation" leg was in
# fact running with FULL speculation (6 checkpoints/line in its engine log) — its
# results are invalid and must not be cited. The env var is kept (harmless) so old
# scripts don't error; depth is controlled with these instead:
# 2026-10-02: 4 -> 3. The `!` degeneration (10-01/02) matches the external
# Qwen3.8-MTP verify-path corruption at draft n_max >= 4 (r/LocalLLaMA +
# vLLM #27364); n_max 3 is the community clean-sweep peak and one below the
# onset threshold. MTP depth = --spec under suffix-draft; verify window =
# spec+2 (auto). Repro not yet lab-reproducible — real agentic load is the
# test; if the loop returns, next lever is --spec 2 (SPEC_T=2).
# 2026-10-03: back UP to 4 after --spec 3 was falsified as a degen fix (loops
# persisted at spec 3 AND at zero-speculation). Engine default is 3; 4 = our
# proven-good value, MTP head confirmed VRAM-resident & exonerated. SPEC_T env
# still overrides for experiments; IQ packs need >= 2.
setarg("--spec", os.environ.get("SPEC_T", "4"))      # verify-window depth (IQ packs need >= 2)
# --mtp-max-t is NOT self-healable by ensure_spec: clear it when unset so a leg's
# value (json.dump persists it) cannot leak into the next boot. 0/absent = default.
_mtp_max_t = os.environ.get("MTP_MAX_T", "0")
if _mtp_max_t != "0":
    setarg("--mtp-max-t", _mtp_max_t)               # cap the MTP's own window depth (0 = --spec)
else:
    if "--mtp-max-t" in args:
        del args[args.index("--mtp-max-t"):args.index("--mtp-max-t") + 2]
ensure_spec()
# Calibrated 2026-10-06 (0.1.39 kernels): 0.55 = 45.8 vs 0.35 = 41.9 tok/s (+9%).
# 10-01's 0.35 was measured on 0.1.3x engines — the cluster-decode work moved the optimum.
setarg("--pcie-frac", os.environ.get("PCIE_FRAC", "0.55"))
# Bigger prefill chunks: auto caps at 8192; auto:16384 lets the expert cache
# lend bigger buffers (the fork runs 16K chunks). Prefill-only; if any OOM in
# prompt processing, drop back to plain auto.
setarg("--prefill", "auto:16384")
# Anti-degeneration sampling (2026-10-01): the engine serves GREEDY by default
# and Hermes sends no sampling params -> this model loops (engine log showed
# 928/929 suffix drafts accepted = a self-feeding repeat loop at 80 tok/s).
# These are SERVER-SIDE defaults: a request's own sampling fields still win,
# so per-request overrides are untouched. Qwen's guidance (temp 0.6, top_p
# 0.95, top_k 20; greedy is explicitly discouraged) + a mild multiplicative
# repeat penalty over a 256-token window to break loops without wrecking
# tool-call structure. Tuning: raise repetition_penalty toward 1.1 if loops
# persist; drop it toward 1.0 if tool-call JSON starts to drift.
cfg["sampling"] = {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
                   # 1.0 = rep-pen OFF (2026-10-03): penalties hypothesized as
                   # the loop driver (long-context suppression pushes mass to
                   # punctuation); our own 09-30 legs raising penalties made
                   # loops WORSE, consistent. REP_PEN env re-enables if ever.
                   "repetition_penalty": float(os.environ.get("REP_PEN", "1.0")),
                   "penalty_last_n": 256}
# #606's one-token net: 256, upstream's number (Matt 10-06, trust-upstream posture). The
# old 64 (10-04) tightened after "!" runs escaped at ~150-200 chars on 0.1.3x kernels;
# the shadow watches on 0.1.40 decide whether that tightening ever comes back.
cfg["repeat_stop_tokens"] = int(os.environ.get("REPEAT_STOP_TOKENS", "256"))
# Lanes (Matt 10-06): parallel=2 - 4 paid ~780 expert slots (~3 GiB of hot cache) to the
# batch carve and the speed dip showed; 2 keeps no-queue usability at most of the cache.
# Entrypoint default = SoT (the q4_0 lesson): a bare recreate can never silently drift it;
# PARALLEL=N env overrides per create for A/B. Solo turns keep full MTP either way.
cfg["parallel"] = int(os.environ.get("PARALLEL", "1"))  # 10-06 Matt: back to single lane - constant solo speed (batch demotion gone); the 2-lane overlap value only pays if the queue shape suits the user, and it doesn't
# Vision token budget (fixed 2026-10-02): the old cap 300 forced every image
# BELOW the documented minimum for this architecture — llama.cpp load_hparams
# warns "Qwen-VL models require at minimum 1024 image tokens to function
# correctly on grounding tasks" (issue #16842), and --max-tokens wires straight
# to cp.image_max_tokens (tools/vision/strata_vision.cpp:105). 0 = don't pass a
# cap: the mtmd chunker picks the model-native count (~1024+ per screenshot).
# CPU encode cost scales with tokens; if a picture ever hurts too much, floor
# VISION_MAX_TOKENS at 1024 — never below. threads stays 4 (burst-only encoder;
# decode needs the cores; measured 10-06: encode ~= 1 s at the current token budget
#   (the old “10-30 s per picture” note predates the token budget - do not quote it).
# GPU vision stays off: XL family has no GPU-vision
# wiring upstream and the CPU encoder holds zero VRAM.
cfg["vision"] = {"exe": "/opt/strata/engine/strata-vision",
                 "mmproj": "/data/mmproj/mmproj-Qwen3.8-Flash-Next-BF16.gguf",
                 "model": args[args.index("--native") + 1],
                 "gpu": False, "threads": 4}
if int(os.environ.get("VISION_MAX_TOKENS", "0")) > 0:
    cfg["vision"]["max_tokens"] = int(os.environ["VISION_MAX_TOKENS"])
# Display identity (Matt 10-06): the model-selector label is prettified from the
# serve model_name (GET /v1/models id). The old name led with "unsloth-ud", which
# describes the quant's origin, not the server - this box serves it from Strata,
# and the Unsloth Desktop (UD) llama.cpp fork is a DIFFERENT provider on :8888.
# Files and dirs keep the unsloth-ud tag (TAG, $CFG_DIR, packs untouched); only
# the advertised identity changes. NO `aliases` entry: /v1/models lists every
# alias as its own selector row (#297), which would re-show the old name. Stale
# callers asking the old id are still SERVED - the chat path answers any model
# string ("Other names are still served, as before", server.py model_for).
# Rollback = delete this line and recreate.
cfg["model_name"] = "qwen3.8-flash-next-q4_k_xl"
json.dump(cfg, open(p, "w"), indent=1)
print("host overlay applied: vram-reserve " + os.environ.get("VRAM_RESERVE_MIB", "3072") + ", suffix-draft 7, expert-cache-per-layer, conv-cache 4096, vision=cpu(4 threads)")
PY
cp -f "$cfg" "/opt/strata/strata-$TAG.json"

# (host overlay v7 10-04) stop-confirmation probe ON by default: an unsigned content-lane
# stop is honoured only after a confirming re-sample of the same position (serve/server.py).
# One decode of cost, prefix KV reused; STRATA_STOP_PROBE=0 to disable.
export STRATA_STOP_PROBE="${STRATA_STOP_PROBE:-1}"
# (host overlay v8 10-04) degeneration recovery (serve/server.py): the block-cycle net is
# upstream opt-in; this host has SEEN block loops on Strata (#75 class), so it ships on at
# the tested length. 0 disables. The net ends a reply whose decoded tail repeats the same
# multi-character block 9x on two re-check windows 16 tokens apart; the reply that trips a
# net restarts the engine ONCE (drops parked conversation state + adaptive tiers: the
# manual cure, automated), and a clean reply resets the trip budget. Survives a restart?
# The log line says start a fresh session (history-side). Episode probe, off by default
# because it copies every clean window's logits (one-time log line when non-finite): add
# -e STRATA_DBG_NAN=1 to the recreate command during a suspected episode.
export STRATA_DEGEN_CYCLE="${STRATA_DEGEN_CYCLE:-9}"
# (host overlay v9 10-04) the post-hoc expert-cache latch backstop (serve/server.py):
# the nets catch repetition; the episode's APERIODIC shape (the "es" pre-degeneration
# collapsing into runs) had none. A reply no net flagged that still ends at length
# with the expert cache above 95% hit is a latch the engine's state is carrying - it
# spends the same one-restart budget as the nets. Episode measured: loops 97-99.8%
# hit, healthy 37-87.5% in the same session. 0 disables.
export STRATA_DEGEN_LATCH="${STRATA_DEGEN_LATCH:-0.95}"

exec .venv/bin/python setup.py --host "$HOST" --port "$PORT"
