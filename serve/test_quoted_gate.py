"""Quoted end-of-turn gate: the stop-id decision rules.

A reserved end-of-turn id sampled MID-SENTENCE in the content lane is the model
quoting the marker, not ending the frame: the gate drops the strike and RE-SAMPLES
the position with zero injected tokens (the probe's mechanism), continuing the turn.
Nothing is spliced in any direction: a strike taught the model to echo whatever it
spliced - the word, then the shape (the scar family, 10-05) - so the context stays
exclusively the model's own text. The client sees its prose continue seamless. A stop at a sentence boundary, after a closed code span, or with no
content yet (tool-only turns) ends the request unchanged.

NEVER author a marker shape as a contiguous literal in this file: on a
marker-shaped-tokenizing server, writing one re-tokenizes into a real stop token
mid-generation. The shapes come from ByteTokenizer.SPECIALS.
"""
import contextlib
import io
import json
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve.frontend import CALL_END, THINK_END, ChatTemplate
from serve.server import DEGEN_CYCLE_REPEATS, ByteTokenizer, MockEngine, Service, serve

IM_END = ByteTokenizer.SPECIALS[1]                       # the end-of-turn shape
ZWSP = chr(0x200B)
# The old gate spliced a coined WORD into the model's context; the model learned to
# say it, and the word at frame boundaries (growing one copy per strike) was the
# visible scar. The word-splice is gone - this constant is now the FORBIDDEN seed:
# a client stream or engine prompt that contains it anywhere re-opens the scar bug.
SPLICED = " marker"                    # must appear NOWHERE in any reply or engine prompt
DEGEN_ON = DEGEN_CYCLE_REPEATS                   # the block net at its production length


class QuotedGate(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.svc = Service(MockEngine(self.tok, "x", max_context=16384), self.tok,
                           ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.svc.degen_cycle_repeats = 0      # the mock scripts are periodic filler; that is not a loop here
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def chat(self, script, **kwargs):
        self.svc.engine = MockEngine(self.tok, script, max_context=16384)
        req = {"messages": [{"role": "user", "content": "go"}],
               "reasoning_effort": "none", "max_tokens": 512, **kwargs}
        r = urllib.request.Request(self.base + "/v1/chat/completions",
                                   data=json.dumps(req).encode(),
                                   headers={"Content-Type": "application/json"})
        try:
            resp = urllib.request.urlopen(r, timeout=10)
        except urllib.error.HTTPError as e:
            resp = e
        with resp:
            return resp.status, json.loads(resp.read().decode())

    def msg(self, reply):
        return reply["choices"][0]["message"], reply["choices"][0]["finish_reason"]

    # ---- the gate fires ----------------------------------------------------
    def test_stop_after_backtick_mid_quote_continues_the_turn_seamlessly(self):
        # pass 1 = prose + open inline code + the REAL end-of-turn id (the strike);
        # pass 2 (the continuation) = the rest of the sentence + MockEngine's stop.
        # MockEngine advances its script list per generate() call.
        code, reply = self.chat(["The marker shows as `", " in the docs."])
        self.assertEqual(code, 200)
        m, finish = self.msg(reply)
        self.assertEqual(finish, "stop")
        self.assertEqual(m["content"], "The marker shows as ` in the docs.")   # the strike is
        # rewritten around invisibly: the bridge word went to the engine, never to the client
        self.assertNotIn(IM_END, m["content"])           # never the raw shape

    def test_rapid_double_quote_second_is_denied(self):
        # progress rule: a strike earned a rewrite; the very next strike came with
        # no real generation in between (degeneration signature) -> denied, the
        # request ends on that stop instead of rewriting forever.
        code, reply = self.chat(["a `", "b `"])
        self.assertEqual(code, 200)
        m, finish = self.msg(reply)
        self.assertEqual(finish, "stop")
        self.assertNotIn(SPLICED, m["content"])         # one rewrite earned, none shown
        self.assertNotIn(IM_END, m["content"])

    def test_the_strike_injects_nothing_and_continues_by_re_sampling(self):
        # Scar regression (10-05, twice-taught): every object this gate spliced into the
        # model's context - first the coined word, then the shape's own text - the model
        # learned to SAY at the boundaries where strikes land (tool-call opens): the
        # visible scar family, engine-specific because no other server injects anything.
        # A strike now re-samples the stop position with ZERO spliced tokens (the probe's
        # mechanism): the continuation prompt ends exactly at the strike position, holding
        # neither the strike id nor any material - the model has nothing new to imitate.
        code, reply = self.chat(["Note the turn boundary: `", " is text, not a frame."])
        self.assertEqual(code, 200)
        m, finish = self.msg(reply)
        self.assertEqual(finish, "stop")
        self.assertIn("is text", m["content"])              # the strike still continued the turn
        self.assertNotIn(SPLICED, json.dumps(reply))        # no coined word in any client lane
        prompt = self.svc.engine.last_prompt                # what pass 2 was fed
        self.assertNotIn(SPLICED, self.tok.decode(prompt))  # ...and nothing coined in the model's context
        self.assertFalse(self.tok.decode(prompt).endswith(IM_END))   # no shape spliced at the tail...
        self.assertTrue(self.tok.decode(prompt).endswith("`"))       # ...the prompt ends AT the strike position
        live = self.tok.encode(IM_END, parse_special=True)[0]
        self.assertEqual(prompt.count(live), 1)             # only the template's own boundary; the
        # strike id was dropped before the segment, so re-sampling rewrites the same position.

    def test_open_span_stop_splices_then_closed_span_stop_is_real_end(self):
        # Audit F3 (honest parity): the tracker counts the MODEL's stream once. A stop right
        # after an OPEN span is the model quoting (spliced, bridge word client-invisible); a
        # later stop that CLOSES the span is the real end the module doctrine always promised
        # ("after a closed code span... ends the request unchanged"). The old code counted the
        # spliced span twice, so every post-quote end became a phantom strike whose splice fed
        # the model words it never said - the cascade that wrote nested frame shapes into tool
        # arguments. One count, no fiction.
        filler1, filler2 = "word " * 15, "note " * 13
        code, reply = self.chat([filler1 + " `", filler2 + " `", " done."])
        self.assertEqual(code, 200)
        m, finish = self.msg(reply)
        self.assertEqual(finish, "stop")
        self.assertNotIn(SPLICED, m["content"])            # bridge word stays engine-side
        self.assertTrue(m["content"].startswith(filler1.strip()))
        self.assertNotIn("done.", m["content"])            # closed-span stop ends the turn

    def test_span_open_again_earns_its_splice(self):
        # Healthy quoting keeps earning rewrites: a span CLOSED and REOPENED between strikes
        # (two backticks in the continuation) leaves the span OPEN at strike two -> spliced,
        # and the continuation delivers.
        filler1, filler2 = "word " * 15, "note " * 13
        code, reply = self.chat([filler1 + " `", filler2 + " ` `", " done."])
        self.assertEqual(code, 200)
        m, finish = self.msg(reply)
        self.assertEqual(finish, "stop")
        self.assertEqual(m["content"].count(SPLICED), 0)   # rewrites earned, none shown
        self.assertIn("done.", m["content"])

    def test_stop_at_backtick_inside_open_fence_is_spliced(self):
        # prose + OPEN code fence ending on an inline backtick: backtick parity is
        # EVEN there (fence 3 + inline 1), so a parity-only rule abstains and the
        # turn dies mid-fence - the tracker must be character-level and fence-aware.
        code, reply = self.chat(["Prose:\n```py\ncode = `", " done."])
        self.assertEqual(code, 200)
        m, finish = self.msg(reply)
        self.assertEqual(finish, "stop")
        self.assertNotIn(SPLICED, m["content"])         # the gate fired (the fence turned out
                                                        # with the answer); its word stays ours
        self.assertIn("done.", m["content"])            # the answer continued

    def test_quoted_strike_inside_a_tool_call_leaves_no_word_in_the_stream(self):
        # The production scar: a stop sampled mid-tool-call was read as a quote, the bridge
        # word was held, and the continuation flushed it into the visible stream right before
        # the call's closing tag (the client saw the word then the tag). The bridge never
        # reaches a client stream in any lane now: proof of continuation is engine-side only.
        cs = chr(60) + "tool_call" + chr(62)
        ce = chr(60) + "/tool_call" + chr(62)
        open_call = (cs + chr(60) + "function=look" + chr(62)
                     + chr(60) + "parameter=q" + chr(62) + "hel")   # mid-argument when the strike hits
        code, reply = self.chat([open_call, "lo" + ce])
        self.assertEqual(code, 200)
        # whatever the turn decides (rewrite around, or honour the stop and drop the
        # half-written call), the bridge word must appear NOWHERE in the client's reply -
        # not in content, not in a call's arguments, not in an announced-but-unfinished span.
        self.assertNotIn(SPLICED, json.dumps(reply))

    def test_answer_lane_swallows_an_own_line_closer_but_keeps_a_prose_quote(self):
        # The scar, and the symmetric rule. (1) A closer on its own line in the ANSWER lane is
        # the model COMPLETING the frame shape (history taught it the pattern): frame furniture,
        # swallowed like the end-of-turn tag, so it never reaches the client. (2) A closer spelled
        # out INSIDE a line stays text - the quoted-closer rule now spans both lanes. The
        # reasoning-lane closer still closes the block (a third shape). All from THINK_END.
        one = self.chat(["Answer body." + chr(10) + THINK_END + chr(10)])
        m1 = self.msg(one[1])[0]["content"]
        self.assertNotIn(THINK_END, m1)                     # (1) swallowed
        self.assertTrue(m1.startswith("Answer body."))

        two = self.chat(["The template writes `" + THINK_END + "` inline."])
        m2 = self.msg(two[1])[0]["content"]
        self.assertIn(THINK_END, m2)                        # (2) kept: a quote, not a frame
        self.assertTrue(m2.startswith("The template writes `"))

    def test_nesting_shape_byte_streamed_leaves_no_frame_shape_anywhere(self):
        # The live cascade (10-04 23:1x): a strike at a tool boundary, a visible frame shape,
        # replay re-tokenizing it into a real frame, nested close-shapes landing in call
        # arguments. Root rule that kills the loop (audit R1): a frame shape with no frame to
        # close is furniture in the answer lane - never shown, so nothing downstream can
        # re-tokenize it. Fed ONE BYTE AT A TIME (R4): the hold window must decide a shape the
        # same way whether it arrives whole or as 9 separate deltas, and a REAL call after the
        # furniture still parses as exactly one call with clean arguments.
        from serve.frontend import OutputParser, THINK_START
        cs, ce = chr(60) + "tool_call" + chr(62), chr(60) + "/tool_call" + chr(62)
        script = ("The answer." + "\n" + THINK_END + "\n"          # furniture: stray closer
                  + "\n" + CALL_END + "\n"                          # furniture: stray call-close
                  + "\n" + THINK_START + "\n"                       # furniture: stray opener
                  + cs + chr(60) + "function=go" + chr(62)
                  + chr(60) + "parameter=q" + chr(62) + "ok"
                  + chr(60) + "/parameter" + chr(62)
                  + chr(60) + "/function" + chr(62) + ce)
        p1 = OutputParser(thinking=False, tools=[{"name": "go",
                                                  "parameters": {"type": "object",
                                                                 "properties": {"q": {"type": "string"}}}}])
        whole = [ev for ch in script for ev in p1.feed(ch)] + p1.finish()
        text = "".join(e.text for e in whole if e.text)
        calls = [e for e in whole if e.kind == "tool_call"]
        self.assertEqual(text.count(THINK_END) + text.count(CALL_END) + text.count(THINK_START), 0,
                         f"a frame shape reached the stream: {text!r}")
        self.assertIn("The answer.", text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].call.arguments, {"q": "ok"})
        # whole-string feed must decide identically (byte-boundary independence)
        p2 = OutputParser(thinking=False, tools=p1.schemas.values() and
                          [{"name": "go", "parameters": {"type": "object",
                                                         "properties": {"q": {"type": "string"}}}}])
        whole2 = [ev for ev in p2.feed(script)] + p2.finish()
        text2 = "".join(e.text for e in whole2 if e.text)
        self.assertEqual(text2, text)

    # ---- the gate stays out of the way -------------------------------------
    def test_sentence_end_stop_is_untouched(self):
        code, reply = self.chat("All done.")
        self.assertEqual(code, 200)
        m, finish = self.msg(reply)
        self.assertEqual(finish, "stop")
        self.assertEqual(m["content"], "All done.")
        self.assertNotIn(ZWSP, m["content"])

    def test_closed_inline_code_span_end_is_untouched(self):
        # even backtick count: a legit answer ending on a closed span must NOT
        # burn a rewrite or grow
        code, reply = self.chat("Use the strata-flip tool.")
        self.assertEqual(code, 200)
        m, _ = self.msg(reply)
        self.assertEqual(m["content"], "Use the strata-flip tool.")
        self.assertNotIn(ZWSP, m["content"])

    def test_empty_content_stop_is_untouched(self):
        # tool-only shape: stop with no content yet -> real end, no splice
        code, reply = self.chat("", max_tokens=8)
        self.assertEqual(code, 200)
        m, finish = self.msg(reply)
        self.assertEqual(finish, "stop")
        self.assertNotIn(ZWSP, m.get("content") or "")


if __name__ == "__main__":
    unittest.main()

class DegenRecovery(unittest.TestCase):
    """The recovery ladder above #606: a repeated BLOCK (period >= 2) ends the reply
    like the one-token run does, and the request that tripped a degeneration restarts
    the engine ONCE, so the next request re-reads the prompt without any parked
    conversation state. The reply's own text is never rewritten (doctrine): the
    recovery moves the ENGINE's state, and a loop that survives a restart is the
    client's history to fix, which the log line says plainly."""

    class FakeEngine:
        """MockEngine plus the engine's DONE line: a FRESH `last` object per pass, so the DONE block's
        same-object check (a DONE arrived only when `last` is not the request-start snapshot) stays honest."""

        def __init__(self, tok, script, hits=9, lookups=10):
            self.inner, self.restarts = MockEngine(tok, script, max_context=16384), 0
            self.last = None
            self.done = (hits, lookups)         # the DONE line's expert-cache counts
        def restart(self):
            self.restarts += 1
        def generate(self, *a, **k):
            try:
                yield from self.inner.generate(*a, **k)
            finally:      # the engine writes its DONE line when its generation is done - a new object each time
                if self.done is not None:
                    hits, lookups = self.done
                    self.last = {"generated": 0, "finish": "stop", "hits": hits, "lookups": lookups}

    def run_reply(self, script, repeat_stop=32, latch=0.0, hits=9, lookups=10,
                  max_new=3000, eng=None):
        tok = ByteTokenizer()
        if eng is None:
            eng = self.FakeEngine(tok, script, hits=hits, lookups=lookups)
        svc = Service(eng, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        svc.repeat_stop_tokens = repeat_stop
        svc.degen_cycle_repeats = DEGEN_ON
        svc.degen_latch = latch
        ids = tok.encode("hi")
        with contextlib.redirect_stdout(io.StringIO()):
            done = [x for kind, x in svc.run(ids, False, None, max_new, {}, threading.Event()) if kind == "done"][0]
        return done, eng, svc

    def test_repeated_block_ends_the_reply(self):
        done, eng, svc = self.run_reply("ab" * 400)     # period 2, far past the cycle limit
        self.assertEqual(done["finish"], "length")      # ended as a degenerate, not an answer
        self.assertLess(done["completion_tokens"], 64)  # caught before the one-token net's 64

    def test_trip_restarts_the_engine_once(self):
        done, eng, svc = self.run_reply("ab" * 400)
        self.assertEqual(eng.restarts, 1)               # parked state dropped, once

    def test_trip_prefers_flush_over_restart(self):
        # The lightweight unlatch: an engine that answers FLUSH is NOT restarted (the model
        # reload is the cost we removed; a FLUSH-less engine falls back, test below).
        class FlushEngine(DegenRecovery.FakeEngine):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.flushes = 0
            def flush(self, timeout=120.0):
                self.flushes += 1
                return {"tokens": 0, "parked": 2, "bytes": 4096}
        done, eng, svc = self.run_reply("ab" * 400, eng=FlushEngine(ByteTokenizer(), "ab" * 400))
        self.assertEqual(done["finish"], "length")      # the reply still ended as a degenerate
        self.assertEqual(eng.flushes, 1)                # unlatched the cheap way, once
        self.assertEqual(eng.restarts, 0)               # the hammer stays sheathed

    def test_flush_refusal_falls_back_to_restart(self):
        # An engine older than the FLUSH verb raises ValueError (its ERR line): the ladder
        # must still cure, the way it did before - one restart.
        class OldEngine(DegenRecovery.FakeEngine):
            def flush(self, timeout=120.0):
                raise ValueError("expected: GEN <max_new> ...")   # the engine's ERR line for an unknown verb
        done, eng, svc = self.run_reply("ab" * 400, eng=OldEngine(ByteTokenizer(), "ab" * 400))
        self.assertEqual(eng.restarts, 1)

    def test_unlatch_restart_mode_pins_the_old_behavior(self):
        # STRATA_DEGEN_UNLATCH=restart: even a FLUSH-capable engine gets the restart (the
        # operator's escape hatch).
        class FlushEngine(DegenRecovery.FakeEngine):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.flushes = 0
            def flush(self, timeout=120.0):
                self.flushes += 1
                return {"tokens": 0, "parked": 0, "bytes": 0}
        import os
        os.environ["STRATA_DEGEN_UNLATCH"] = "restart"
        try:
            done, eng, svc = self.run_reply("ab" * 400, eng=FlushEngine(ByteTokenizer(), "ab" * 400))
        finally:
            del os.environ["STRATA_DEGEN_UNLATCH"]
        self.assertEqual(eng.flushes, 0)
        self.assertEqual(eng.restarts, 1)

    def test_clean_reply_restarts_nothing(self):
        prose = ("The gate splices a marker into content. Recovery, by contrast, moves the "
                 "engine's state: a restart drops parked checkpoints and adaptive tiers alike. "
                 "A reply that loops after the restart is history to fix, not the engine's.")
        done, eng, svc = self.run_reply(prose * 2) # varied text: no block repeats 9 times
        self.assertEqual(done["finish"], "stop")
        self.assertEqual(eng.restarts, 0)

    def test_aperiodic_high_hit_length_finish_restarts_the_engine(self):
        # no repetition anywhere (128 distinct tokens: the one-token and the block nets both look
        # and see none), the reply ends at length (100-token cap), the expert cache nearly full:
        # the latch shape the episode's pre-degeneration collapsed into (the 256-token cuts sat at
        # 99.3-99.8% hit, the healthy replies of the same session topped out at 87.5%)
        script = "".join(chr(0x20 + i) for i in range(128))      # 128 distinct bytes, no token repeats
        done, eng, svc = self.run_reply(script, latch=0.95, hits=995, lookups=1000, max_new=100)
        self.assertEqual(done["finish"], "length")
        self.assertEqual(eng.restarts, 1)
        self.assertEqual(svc.degen_trips, 1)

    def test_latch_below_the_threshold_is_untouched(self):
        script = "".join(chr(0x20 + i) for i in range(128))
        done, eng, svc = self.run_reply(script, latch=0.95, hits=800, lookups=1000, max_new=100)
        self.assertEqual(done["finish"], "length")
        self.assertEqual(eng.restarts, 0)
        self.assertEqual(svc.degen_trips, 0)

    def test_short_length_finish_under_64_tokens_is_not_a_latch(self):
        # the one-token net is off: a tiny max_new answer (40 tokens) that caches 99.8% ends at
        # length, but under the 64-token floor it is a short answer, not a latch
        done, eng, svc = self.run_reply("a" * 200, repeat_stop=0, latch=0.95,
                                        hits=998, lookups=1000, max_new=40)
        self.assertEqual(done["finish"], "length")
        self.assertEqual(eng.restarts, 0)        # the 64-token floor keeps short answers out
        self.assertEqual(svc.degen_trips, 0)

    def test_stop_finish_with_high_hit_rate_is_not_a_latch(self):
        prose = ("The gate splices a marker into content. The latch, by contrast, reads the engine's "
                 "own accounting: a reply that ends at a stop token is an answer, whatever its cache.")
        done, eng, svc = self.run_reply(prose, latch=0.95, hits=999, lookups=1000)
        self.assertEqual(done["finish"], "stop")
        self.assertEqual(eng.restarts, 0)

    def test_latch_after_a_net_trip_spends_no_second_restart(self):
        # the nets and the latch share one budget: a net trip restarts and holds the budget, so a
        # latch trip on the NEXT request reports history-side without a second restart
        latch_script = "".join(chr(0x20 + i) for i in range(128))   # no repetition: only the latch sees it
        tok = ByteTokenizer()
        eng = self.FakeEngine(tok, ["a" * 256, latch_script], hits=995, lookups=1000)
        svc = Service(eng, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        svc.repeat_stop_tokens = 32              # the one-token net trips the first reply
        svc.degen_cycle_repeats = 0
        svc.degen_latch = 0.95
        ids = tok.encode("hi")
        with contextlib.redirect_stdout(io.StringIO()):
            list(svc.run(ids, False, None, 3000, {}, threading.Event()))
            list(svc.run(ids, False, None, 100, {}, threading.Event()))
        self.assertEqual(eng.restarts, 1)        # only the net's trip spent a restart
        self.assertEqual(svc.degen_trips, 2)     # net trip + latch trip: history-side

    def test_clean_reply_after_a_latch_resets_the_trip_count(self):
        script = "".join(chr(0x20 + i) for i in range(128))
        tok = ByteTokenizer()
        eng = self.FakeEngine(tok, [script, "An ordinary answer that ends cleanly, with its own words."],
                              hits=995, lookups=1000)
        svc = Service(eng, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        svc.repeat_stop_tokens = 32
        svc.degen_cycle_repeats = 0
        svc.degen_latch = 0.95
        ids = tok.encode("hi")
        with contextlib.redirect_stdout(io.StringIO()):
            list(svc.run(ids, False, None, 100, {}, threading.Event()))   # the latch trips (100 < 128: ends at length)
            list(svc.run(ids, False, None, 3000, {}, threading.Event()))  # a clean reply after it
        self.assertEqual(eng.restarts, 1)        # only the latch's trip spent a restart
        self.assertEqual(svc.degen_trips, 0)     # the clean reply cleared the count

    def test_legitimate_repetition_survives(self):
        # single-character runs under the #606 net (a setext underline), a phrase used
        # six times, and a varied tail: none of these is the 9-repeats-twice shape
        text = ("Report\n========================================\n"     # 40-char run: under the net's 64
                "ok. ok. ok. ok. ok. ok.\n" + "Each line carries its own fresh content. " * 4 + "done.")
        done, eng, svc = self.run_reply(text, repeat_stop=64) # the production net length
        self.assertEqual(done["finish"], "stop")
        self.assertEqual(eng.restarts, 0)


class ThinkQuote(unittest.TestCase):
    """A quoted closing tag written in the reasoning PROSE is the model talking about
    the marker, not ending its thinking: the parser honours the template's contract -
    the closer sits on its own line ("\\nCLOSER\\n\\n") - and treats every other
    spelling as inert text.  (The literal never appears in this file: shapes come
    from the imported THINK_END.)"""

    def parse(self, *deltas):
        from serve.frontend import OutputParser
        p = OutputParser(thinking=True)
        evs = []
        for d in deltas:
            evs += p.feed(d)
        evs += p.finish()
        return [(e.kind, e.text) for e in evs if e.kind in ("reasoning", "content")], p.state

    def test_quoted_closer_mid_sentence_is_inert_text(self):
        evs, state = self.parse("The ", THINK_END, " shape must not end a block.")
        self.assertEqual(state, "reasoning")
        self.assertEqual("".join(t for _, t in evs),
                         "The " + THINK_END + " shape must not end a block.")

    def test_real_closer_on_its_own_line_closes(self):
        evs, state = self.parse("thought\n" + THINK_END + "\n\nthe answer")
        self.assertEqual(state, "content")
        joined = "".join(t for k, t in evs)
        self.assertIn("thought", joined)
        self.assertNotIn(THINK_END, joined)

    def test_quoted_then_real_closer(self):
        evs, state = self.parse("x " + THINK_END + " y\n" + THINK_END + "\n\nz")
        self.assertEqual(state, "content")
        r = "".join(t for k, t in evs if k == "reasoning")
        self.assertEqual(r, "x " + THINK_END + " y\n")     # the newline before the real
        # closer belongs to the reasoning; the closer itself never reaches the client

    def test_undecided_closer_resolves_when_the_next_line_arrives(self):
        from serve.frontend import OutputParser
        p = OutputParser(thinking=True)
        got = p.feed("a\n" + THINK_END)                 # undecided: held, not a close
        self.assertEqual(p.state, "reasoning")
        evs = p.feed("\n\nanswer")                      # the blank line: it closes
        self.assertEqual(p.state, "content")
        self.assertTrue(any(e.kind == "content" and "answer" in e.text for e in evs))
        self.assertNotIn(THINK_END, "".join(e.text for e in evs))

    def test_undecided_closer_that_quoted_on(self):
        from serve.frontend import OutputParser
        p = OutputParser(thinking=True)
        p.feed("a\n" + THINK_END)
        evs = p.feed("b continues here.")                # not a line: it was prose
        self.assertEqual(p.state, "reasoning")
        joined = "".join(e.text for e in evs) + "".join(t for k, t in [])
        self.assertIn(THINK_END, joined)                  # inert, still visible as text

    def test_closer_at_the_very_start_closes_an_empty_block(self):
        evs, state = self.parse(THINK_END + "\n\njust the answer")
        self.assertEqual(state, "content")

class SeamProbe(unittest.TestCase):
    """The probe re-samples the position the swallowed stop token sat at.  When the
    model's continuation re-emits the punctuation the stream already ended with -
    the doubled-comma artifact - the gate clips the re-emission: exactly one copy
    of the character and at most one space survive.  A re-emitted LETTER is real
    prose continuing and is never clipped."""

    def setUp(self):
        import os
        self._saved = os.environ.get("STRATA_STOP_PROBE")
        os.environ["STRATA_STOP_PROBE"] = "1"
        self.tok = ByteTokenizer()
        self.svc = Service(MockEngine(self.tok, "x", max_context=16384), self.tok,
                           ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.svc.degen_cycle_repeats = 0
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        import os
        if self._saved is None:
            os.environ.pop("STRATA_STOP_PROBE", None)
        else:
            os.environ["STRATA_STOP_PROBE"] = self._saved
        self.httpd.shutdown()
        self.httpd.server_close()

    def chat(self, script, **kwargs):
        self.svc.engine = MockEngine(self.tok, script, max_context=16384)
        req = {"messages": [{"role": "user", "content": "go"}],
               "reasoning_effort": "none", "max_tokens": 512, **kwargs}
        r = urllib.request.Request(self.base + "/v1/chat/completions",
                                   data=json.dumps(req).encode(),
                                   headers={"Content-Type": "application/json"})
        try:
            resp = urllib.request.urlopen(r, timeout=10)
        except urllib.error.HTTPError as e:
            resp = e
        with resp:
            return json.loads(resp.read())["choices"][0]["message"].get("content") or ""

    def test_reemitted_comma_after_stop_is_clipped(self):
        # observed shape: "...is," then the stop; the re-sample emits ", and..." ->
        # without the clip the client sees ",, and"
        content = self.chat(["The value is,", ", and it moves on."])
        self.assertNotIn(",,", content)
        self.assertEqual(content, "The value is, and it moves on.")

    def test_reemitted_comma_after_comma_space_absorbs_its_space(self):
        # the stream already ends ", " : the duplicate comma AND its fresh space go
        content = self.chat(["The value is, ", ", and it moves on."])
        self.assertNotIn(",  ", content)
        self.assertNotIn(",,", content)
        self.assertEqual(content, "The value is, and it moves on.")

    def test_continuation_that_is_not_a_reemission_is_untouched(self):
        content = self.chat(["The value is,", " and it moves on."])
        self.assertEqual(content, "The value is, and it moves on.")

    def test_a_reemitted_letter_is_never_clipped(self):
        # a stop mid-word ("valu") probed and continued with the same letter: that is
        # the word continuing, not a seam artifact - clipping it would eat prose
        content = self.chat(["The valu", "e is fine, all set."])
        self.assertIn("value", content)

class StopProbe(unittest.TestCase):
    """An unsigned content stop is confirmed by a re-sample of the SAME position -
    honour it only if the model stops again.  MockEngine scripts stand in for the two
    worlds: a model that truly ended vs one mid-sentence that breaks out of the stop.
    """

    def setUp(self):
        import os
        self._saved = os.environ.get("STRATA_STOP_PROBE")
        os.environ["STRATA_STOP_PROBE"] = "1"
        self.tok = ByteTokenizer()
        self.svc = Service(MockEngine(self.tok, "x", max_context=16384), self.tok,
                           ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.svc.degen_cycle_repeats = 0      # the mock scripts are periodic filler; that is not a loop here
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        import os
        if self._saved is None:
            os.environ.pop("STRATA_STOP_PROBE", None)
        else:
            os.environ["STRATA_STOP_PROBE"] = self._saved
        self.httpd.shutdown()
        self.httpd.server_close()

    def chat(self, script, **kwargs):
        self.svc.engine = MockEngine(self.tok, script, max_context=16384)
        req = {"messages": [{"role": "user", "content": "go"}],
               "reasoning_effort": "none", "max_tokens": 512, **kwargs}
        r = urllib.request.Request(self.base + "/v1/chat/completions",
                                   data=json.dumps(req).encode(),
                                   headers={"Content-Type": "application/json"})
        try:
            resp = urllib.request.urlopen(r, timeout=10)
        except urllib.error.HTTPError as e:
            resp = e
        with resp:
            return resp.status, json.loads(resp.read().decode())

    def test_confirmed_stop_ends_unchanged(self):
        # a stop after sentence punctuation ends with NO probe - if the probe
        # had run, script 2 (" Never") would appear in the content.
        status, reply = self.chat(["The end.", " Never"])
        msg, fin = reply["choices"][0]["message"], reply["choices"][0]["finish_reason"]
        self.assertEqual(fin, "stop")
        self.assertEqual((msg.get("content") or ""), "The end.")

    def test_reopened_stop_continues_the_turn(self):
        # first stop mid-sentence, re-sample breaks out of it: the turn continues; the
        # SECOND stop (probe exhausted) ends the request
        status, reply = self.chat(["The token is ", " Actually, still going. Over."])
        content = reply["choices"][0]["message"].get("content") or ""
        self.assertIn("still going", content)
        self.assertTrue(content.startswith("The token is"))

    def test_probe_off_keeps_first_stop_final(self):
        import os
        os.environ["STRATA_STOP_PROBE"] = "0"
        status, reply = self.chat(["The end.", " Never sampled."])
        content = reply["choices"][0]["message"].get("content") or ""
        self.assertEqual(content, "The end.")

    def test_insistence_ends_after_one_thinking_splice(self):
        # INSISTENCE: a model that keeps trying to stop (unclosed thinking, empty
        # scripts = immediate stop every pass) is honoured after ONE spliced word.
        status, reply = self.chat(["", ""], reasoning_effort="low")
        reasoning = reply["choices"][0]["message"].get("reasoning_content") or ""
        # one override, then honoured: the strike word never reaches the client because
        # nothing continued after it (tombstone suppression); the block was closed cleanly
        self.assertNotIn(SPLICED, reasoning)

    def test_honest_thinking_quotes_earn_past_the_old_ceiling(self):
        # The 12/12 collapse (10-05): a COUNT ceiling ended honest citation mid-thinking.
        # Policy now: every splice needs real progress, so a long honest citation stream
        # keeps earning rewrites - runaways are impossible because each rewrite costs
        # generated tokens (max_tokens is the real end). 13 spaced citations then a closed
        # block: all thirteen splice; the answer lands clean.
        closer = chr(60) + "/think" + chr(62)
        script = [("point%d " % i) * 15 for i in range(13)]      # 90 bytes >= the progress floor
        script.append("finishing. " + closer)
        # ByteTokenizer costs a token per byte: 13 chunks + bridge words need headroom
        # the chat() default (512) does not give - the ceiling test must not die on budget.
        code, reply = self.chat(script, reasoning_effort="low", max_tokens=2048)
        self.assertEqual(code, 200)
        m = reply["choices"][0]["message"]
        reasoning = m.get("reasoning_content") or ""
        self.assertIn("point12", reasoning)             # the thirteenth strike spliced
        self.assertIn("finishing", reasoning)           # ...and the block still ran to its close
        self.assertEqual(reply["choices"][0]["finish_reason"], "stop")
        self.assertNotIn(SPLICED, json.dumps(reply))

    def test_thinking_strike_chain_without_progress_is_honoured_at_once(self):
        # The live collapse shape: strikes chaining with almost nothing between them.
        # First strike earns its rewrite; the next came with no progress -> honoured AT
        # ONCE (the old code granted eleven more), the open block closes, and the model's
        # next text arrives as the answer, not as more struck-off thought.
        script = ["a" * 30, "b" * 10, "the answer.", "not reached"]
        code, reply = self.chat(script, reasoning_effort="low")
        self.assertEqual(code, 200)
        m = reply["choices"][0]["message"]
        self.assertIn("a" * 30, m.get("reasoning_content") or "")
        # strike 2 came with ~10 tokens of progress (< 48): denied, the OPEN block closed
        # (bleed guard), and the next chunk lands as the answer - the old code granted
        # eleven more splices here instead (the live 12/12 log).
        self.assertEqual((m.get("content") or ""), "the answer.")
        self.assertNotIn("not reached", json.dumps(reply))        # its stop ended the turn
        self.assertNotIn(SPLICED, json.dumps(reply))

    def test_confirmed_mid_sentence_stop_gets_one_forced_continuation(self):
        # probe confirms (script 2 = stop only) but prose is visibly mid-sentence
        # ("The token is ") -> ONE forced spliced word; the next stop is honoured.
        status, reply = self.chat(["The token is ", "", ""])
        content = reply["choices"][0]["message"].get("content") or ""
        # the forced word did its engine-side job (escaped the collapse position), but the
        # model ended anyway - nothing continued, so the held word is dropped, not shown
        self.assertNotIn(SPLICED, content)
        self.assertEqual(content, "The token is ")

    def test_tool_only_end_is_never_probed_or_forced(self):
        # agent-loop turn: a complete tool call, no prose, stop = real end; re-sampling a
        # done call could DUPLICATE it, so it must not be probed or forced.
        cs, ce = chr(60) + "tool_call" + chr(62), chr(60) + "/tool_call" + chr(62)
        body = (cs + chr(60) + "function=look" + chr(62)
                + chr(60) + "parameter=q" + chr(62) + "hi"
                + chr(60) + "/parameter" + chr(62)
                + chr(60) + "/function" + chr(62) + ce)
        status, reply = self.chat([body, " second call", " third"])
        msg, fin = reply["choices"][0]["message"], reply["choices"][0]["finish_reason"]
        self.assertEqual(fin, "tool_calls" if msg.get("tool_calls") else "stop")
        self.assertEqual(len(msg.get("tool_calls") or []), 1)
        self.assertNotIn("second call", json.dumps(reply))

    def test_insisted_stop_in_open_thinking_closes_and_answers(self):
        # Bleed fix: a model that tries to end with the thinking block unclosed must not
        # ship a turn of raw thought and no answer - the block is closed with the wrap-up
        # (once) and the scripted answer runs. Script: thinking, stop, stop, answer.
        status, reply = self.chat(["", "", "The answer."], reasoning_effort="low")
        msg, fin = reply["choices"][0]["message"], reply["choices"][0]["finish_reason"]
        self.assertEqual(fin, "stop")
        self.assertEqual((msg.get("content") or ""), "The answer.")

    def test_deferred_re_stop_after_probe_still_forces_once(self):
        # A stray space token between the probe and the next stop clears insistence, but a
        # second stop in still-incomplete prose is the same collapse: the one forced
        # continuation must catch the DEFERRED re-stop too (observed in production).
        status, reply = self.chat(["The token is ", " ", " ", " "])
        content = reply["choices"][0]["message"].get("content") or ""
        # the deferred stop was caught and forced engine-side; the model then ended on
        # whitespace only, so the held word never proves a continuation and stays hidden.
        # (One space did legitimately stream during the probe re-sample before the hold
        # window opened - real model output, kept.)
        self.assertTrue(content.startswith("The token is"))
        self.assertNotIn(SPLICED, content)

    def test_open_tool_call_watchdog_truncates_degeneration(self):
        # A call that never closes is degeneration (live case: 20k tokens in one body).
        # With a low cap, the open-call watchdog truncates instead of streaming forever.
        import os
        saved = os.environ.get("STRATA_MAX_OPEN_CALL")
        os.environ["STRATA_MAX_OPEN_CALL"] = "12"
        try:
            cs, fn = chr(60) + "tool_call" + chr(62), chr(60) + "function=look" + chr(62)
            body = cs + fn + chr(60) + "parameter=q" + chr(62) + "runaway " * 40
            status, reply = self.chat([body])
            fin = reply["choices"][0]["finish_reason"]
            n_out = reply.get("usage", {}).get("completion_tokens", 0)
            self.assertEqual(fin, "length")
            self.assertLess(n_out, 60)        # truncated at the cap, not a 320-token stream
        finally:
            if saved is None:
                os.environ.pop("STRATA_MAX_OPEN_CALL", None)
            else:
                os.environ["STRATA_MAX_OPEN_CALL"] = saved

