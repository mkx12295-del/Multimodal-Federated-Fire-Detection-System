import json
import time
import urllib.request

from firenode.dashboard import Status, start_dashboard
from firenode.fusion import FusionEngine, State
from firenode.node import ReasoningWorker
from firenode.reasoning import Context, RuleReasoner, parse_verdict


def window(n=30, temp=(20, 20), hum=(50, 50), tvoc=(0, 0), prob=0.0, vis=None):
    lerp = lambda a, i: a[0] + (a[1] - a[0]) * i / (n - 1)  # noqa: E731
    readings = [{"Temperature": lerp(temp, i), "Humidity": lerp(hum, i),
                 "TVOC": lerp(tvoc, i), "Pressure": 1000.0} for i in range(n)]
    return Context.from_window(readings, [prob] * n, [vis] * n, "WARNING")


def test_context_summarizes_trends():
    ctx = window(temp=(20, 26), tvoc=(0, 900), prob=0.9, vis=0.7)
    assert ctx.temperature_change == 6 and ctx.tvoc_max == 900
    assert ctx.camera_available and ctx.vision_hit_fraction == 1.0
    assert "TVOC: 900.0 ppb" in ctx.to_prompt()


def test_context_without_camera():
    ctx = window()
    assert not ctx.camera_available
    assert "Camera: not available." in ctx.to_prompt()


def test_parse_verdict_handles_wrapped_json():
    v = parse_verdict('Sure! {"verdict": "fire", "confidence": 1.4, "explanation": "x", '
                      '"action": "y"} hope that helps', "test", 0.1)
    assert v.parsed and v.is_fire and v.confidence == 1.0


def test_parse_verdict_rejects_garbage_as_not_fire():
    for text in ["", "no json here", '{"verdict": "maybe"}', "{broken"]:
        v = parse_verdict(text, "test", 0.0)
        assert not v.parsed and not v.is_fire and v.confidence == 0.0


def test_rules_distinguish_fire_steam_and_quiet():
    r = RuleReasoner()
    assert r.reason(window(tvoc=(0, 900), prob=0.95)).is_fire
    assert not r.reason(window(hum=(50, 90))).is_fire
    assert not r.reason(window()).is_fire
    assert r.reason(window(vis=0.8)).is_fire  # flame seen, no gas yet


def test_escalate_only_promotes_warning():
    e = FusionEngine()
    assert e.escalate() is False and e.state == State.NORMAL
    e.update(0.6, None)
    assert e.state == State.WARNING
    assert e.escalate() is True and e.state == State.ALARM


def test_worker_runs_reasoner_off_thread_with_cooldown(tmp_path):
    w = ReasoningWorker(RuleReasoner(), cooldown_s=60, log_path=tmp_path / "inc.jsonl")
    assert w.submit(window(tvoc=(0, 900), prob=0.95))
    assert not w.submit(window())  # cooldown
    for _ in range(100):
        res = w.results()
        if res:
            break
        time.sleep(0.02)
    assert res and res[0].is_fire
    rec = json.loads((tmp_path / "inc.jsonl").read_text().splitlines()[0])
    assert rec["verdict"] == "fire" and rec["context"]["tvoc_max"] == 900


def test_dashboard_serves_status():
    status = Status()
    status.push({"state": "NORMAL", "smoothed": 0.1})
    status.update(last_verdict={"verdict": "no_fire"})
    server = start_dashboard(status, "127.0.0.1", 0)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        snap = json.loads(urllib.request.urlopen(url + "/status").read())
        assert snap["latest"]["state"] == "NORMAL" and snap["last_verdict"]["verdict"] == "no_fire"
        assert b"Fire Node" in urllib.request.urlopen(url + "/").read()
    finally:
        server.shutdown()


def test_few_shot_examples_precede_the_question():
    from firenode.reasoning import FEW_SHOT, _ChatReasoner

    r = _ChatReasoner()
    ctx = window()
    assert [m["role"] for m in r.messages(ctx)] == ["system", "user"]
    r.few_shot = True
    msgs = r.messages(ctx)
    assert len(msgs) == 2 + 2 * len(FEW_SHOT)
    assert msgs[-1]["content"] == ctx.to_prompt()
    assert {json.loads(m["content"])["verdict"] for m in msgs if m["role"] == "assistant"} \
        == {"fire", "no_fire"}
