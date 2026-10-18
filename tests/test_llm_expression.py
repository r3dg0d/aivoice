import json
import time

import pytest

from aivoice.expression import ExpressionCompiler
from aivoice.llm import ScriptedLLM, SpeechIntelligenceEngine, fact_check, parse_response, rule_cleanup


def reply(transformed, emotion="neutral", delivery="", confidence=0.9):
    return json.dumps({"transformed": transformed, "emotion": emotion, "delivery": delivery, "confidence": confidence})


def test_verbatim_never_calls_model():
    llm = ScriptedLLM()
    eng = SpeechIntelligenceEngine(llm, mode="verbatim")
    r = eng.transform("um I don't know")
    assert r.transformed == "um I don't know" and llm.calls == [] and not eng.enabled


def test_rule_cleanup():
    assert rule_cleanup("Um, I I think, uh, we should go") == "I think, we should go"
    assert rule_cleanup("so so the the plan") == "so the plan"


def test_cleanup_without_model_is_rule_based():
    r = SpeechIntelligenceEngine(None, mode="cleanup").transform("uh hello hello there")
    assert r.transformed == "hello there" and not r.used_llm and "rule-based" in r.fallback_reason


def test_natural_mode_uses_valid_output():
    eng = SpeechIntelligenceEngine(ScriptedLLM(reply("Wait, what exactly are you doing?", "confused", "curious")), mode="natural")
    r = eng.transform("I don't know what you're doing")
    assert r.used_llm and r.transformed == "Wait, what exactly are you doing?" and r.emotion == "confused"


def test_malformed_json_falls_back():
    eng = SpeechIntelligenceEngine(ScriptedLLM("sure! here you go: transformed text"), mode="natural")
    r = eng.transform("hello there")
    assert r.transformed == "hello there" and "malformed" in r.fallback_reason


def test_timeout_falls_back_within_budget():
    eng = SpeechIntelligenceEngine(ScriptedLLM(reply("x"), delay_ms=2000), mode="natural", timeout_ms=100)
    t0 = time.perf_counter()
    r = eng.transform("hello there")
    assert (time.perf_counter() - t0) < 0.5
    assert r.transformed == "hello there" and "exceeded" in r.fallback_reason
    assert eng.stats.timeouts == 1


def test_hallucinated_facts_rejected():
    eng = SpeechIntelligenceEngine(ScriptedLLM(reply("Meet me at 9 with Sarah.")), mode="character", character="pirate")
    r = eng.transform("meet me at 7 with Bob")
    assert r.transformed == "meet me at 7 with Bob" and "rejected" in r.fallback_reason
    assert fact_check("call Alice at 5", "ring Alice at 5") is None
    assert "dropped number" in fact_check("it costs 40 dollars", "it costs a lot")
    assert "dropped name" in fact_check("tell Marcus hi", "tell him hi")


def test_expressive_keeps_wording():
    eng = SpeechIntelligenceEngine(ScriptedLLM(reply("That is SO funny!!", "laughing", "amused")), mode="expressive")
    r = eng.transform("That's actually pretty funny.")
    assert r.transformed == "That's actually pretty funny." and r.emotion == "laughing"


def test_parse_response_tolerates_fences_and_clamps():
    d = parse_response("```json\n{\"transformed\": \"hi\", \"emotion\": \"furious\", \"confidence\": 7}\n```")
    assert d["emotion"] == "neutral" and d["confidence"] == 1.0
    with pytest.raises(ValueError):
        parse_response('{"emotion": "happy"}')


# --- expression compiler -----------------------------------------------------


def test_drama3_laugh_and_scoped_whisper():
    c = ExpressionCompiler("drama3", intensity=0.6, frequency=0.5)
    assert c.compile("That's actually pretty funny.", emotion="laughing").text == "[laughs] That's actually pretty funny."
    w = c.compile("don't tell anyone", emotion="whisper").text
    assert w == "<whisper>don't tell anyone</whisper>"


def test_drama3_validation_drops_unknown_scoped_and_balances():
    c = ExpressionCompiler("drama3")
    out = c.validate("<whisper>hi <shout>there</shout> [laughs] friend [a very long direction that goes on and on forever and ever]")
    assert "<shout>" not in out.text and out.text.endswith("</whisper>")
    assert "[laughs]" in out.text and any("forever" in r for r in out.removed)


def test_s2_only_documented_tags():
    c = ExpressionCompiler("s2", intensity=0.8)
    assert c.compile("hello", emotion="excited").text == "[excited] hello"
    assert c.compile("hello", emotion="sarcastic").text == "hello"  # not a documented S2 cue
    v = c.validate("[laughing] ok [barely holding it together] <whisper>x</whisper>")
    assert v.text == "[laughing] ok x"


def test_none_dialect_strips_everything_and_intensity_zero():
    assert ExpressionCompiler("none").compile("[laughs] hi <whisper>there</whisper>", emotion="laughing").text == "hi there"
    assert ExpressionCompiler("drama3", intensity=0.0).compile("hi", emotion="angry").text == "hi"


def test_s1_parentheses():
    assert ExpressionCompiler("s1", intensity=0.8).compile("hello", emotion="sad").text == "(sad) hello"
