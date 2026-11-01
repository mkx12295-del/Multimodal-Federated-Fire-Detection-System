"""Local LLM reasoning over recent sensor and camera history.

When the fusion engine enters WARNING or ALARM, the node summarizes the last
readings (trends in temperature, humidity and TVOC, model scores, camera
detections) and asks a small local LLM for a verdict, a confidence and a
plain-language explanation. Nothing leaves the device.

Safety policy: the LLM is advisory and can only escalate. A confident "fire"
verdict may promote WARNING to ALARM; the LLM can never cancel or suppress an
ALARM raised by the fusion engine.

Backends, tried in order by ``backend: auto``:
  llama_cpp     GGUF model in-process via llama-cpp-python (Raspberry Pi)
  transformers  the same GGUF file via Hugging Face transformers (desktop)
  ollama        an Ollama server on localhost
  rules         deterministic rubric, used when no LLM is available
"""
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SYSTEM_PROMPT = """You are the reasoning module of an indoor fire detection node.
You receive a summary of the last readings from a gas/temperature/humidity sensor
and a camera fire/smoke detector. Decide whether there is a real fire.

Guidance:
- Combustion raises TVOC (volatile organic compounds) sharply, often with rising temperature.
- Steam or a shower raises humidity without raising TVOC; that is not a fire.
- A camera detection with no gas or temperature change may be a false positive
  (sunlight, red objects, screens), unless detections are frequent and confident,
  which can mean an open flame before smoke reaches the sensor.
- A sensor model probability near 1 over many readings is strong evidence of fire.

Answer with JSON only, no other text:
{"verdict": "fire" or "no_fire", "confidence": number between 0 and 1,
 "explanation": one or two sentences, "action": short recommended action}"""


@dataclass
class Context:
    """Summary of the recent window that the reasoner sees."""
    samples: int
    temperature_now: float
    temperature_change: float
    humidity_now: float
    humidity_change: float
    tvoc_now: float
    tvoc_max: float
    tvoc_change: float
    sensor_prob_mean: float
    sensor_prob_now: float
    camera_available: bool
    vision_conf_max: float
    vision_hit_fraction: float
    fusion_state: str

    @classmethod
    def from_window(cls, readings, sensor_probs, vision_confs, state, vision_threshold=0.35):
        temp = np.array([r["Temperature"] for r in readings], float)
        hum = np.array([r["Humidity"] for r in readings], float)
        tvoc = np.array([r["TVOC"] for r in readings], float)
        probs = np.asarray(sensor_probs, float)
        vis = [v for v in vision_confs if v is not None]
        vis = np.asarray(vis, float)
        return cls(
            samples=len(readings),
            temperature_now=round(temp[-1], 2), temperature_change=round(temp[-1] - temp[0], 2),
            humidity_now=round(hum[-1], 2), humidity_change=round(hum[-1] - hum[0], 2),
            tvoc_now=round(tvoc[-1], 1), tvoc_max=round(tvoc.max(), 1),
            tvoc_change=round(tvoc[-1] - tvoc[0], 1),
            sensor_prob_mean=round(probs.mean(), 3), sensor_prob_now=round(probs[-1], 3),
            camera_available=bool(len(vis)),
            vision_conf_max=round(float(vis.max()), 3) if len(vis) else 0.0,
            vision_hit_fraction=round(float((vis >= vision_threshold).mean()), 3) if len(vis) else 0.0,
            fusion_state=str(state),
        )

    def to_prompt(self):
        cam = (f"Camera: max fire/smoke confidence {self.vision_conf_max}, detections in "
               f"{self.vision_hit_fraction:.0%} of frames." if self.camera_available
               else "Camera: not available.")
        return (
            f"Last {self.samples} readings (1 per second):\n"
            f"Temperature: {self.temperature_now} C (change {self.temperature_change:+} C)\n"
            f"Humidity: {self.humidity_now} % (change {self.humidity_change:+} %)\n"
            f"TVOC: {self.tvoc_now} ppb now, max {self.tvoc_max} ppb (change {self.tvoc_change:+} ppb)\n"
            f"Sensor model fire probability: now {self.sensor_prob_now}, mean {self.sensor_prob_mean}\n"
            f"{cam}\n"
            f"Fusion engine state: {self.fusion_state}"
        )


@dataclass
class Verdict:
    verdict: str
    confidence: float
    explanation: str
    action: str
    backend: str
    latency_s: float
    parsed: bool = True

    @property
    def is_fire(self):
        return self.verdict == "fire"


_JSON = re.compile(r"\{.*\}", re.S)


def parse_verdict(text, backend, latency):
    """Pull the JSON object out of a model reply; tolerate extra text around it."""
    m = _JSON.search(text or "")
    try:
        d = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        d = None
    if not isinstance(d, dict) or str(d.get("verdict", "")).lower() not in ("fire", "no_fire"):
        return Verdict("no_fire", 0.0, f"unparseable reply: {(text or '')[:120]!r}", "",
                       backend, latency, parsed=False)
    try:
        conf = min(1.0, max(0.0, float(d.get("confidence", 0.5))))
    except (TypeError, ValueError):
        conf = 0.5
    return Verdict(d["verdict"].lower(), conf, str(d.get("explanation", "")),
                   str(d.get("action", "")), backend, latency)


class RuleReasoner:
    """Deterministic rubric mirroring the system prompt; the no-LLM fallback."""
    name = "rules"

    def reason(self, ctx):
        t = time.perf_counter()
        gas = ctx.sensor_prob_mean >= 0.5 or ctx.tvoc_change > 200
        steam = ctx.humidity_change > 5 and ctx.tvoc_change < 50
        flame = ctx.camera_available and ctx.vision_hit_fraction >= 0.5
        if gas and not steam:
            v = ("fire", 0.9 if flame else 0.75, "Gas readings indicate combustion"
                 + (" and the camera confirms flame or smoke." if flame else "."),
                 "Evacuate and check the area.")
        elif flame:
            v = ("fire", 0.6, "The camera repeatedly detects flame or smoke but gas levels are "
                 "normal so far.", "Inspect the camera's field of view.")
        elif steam:
            v = ("no_fire", 0.7, "Humidity rose without VOCs, consistent with steam.",
                 "No action; keep monitoring.")
        else:
            v = ("no_fire", 0.6, "No sustained evidence of combustion.", "Keep monitoring.")
        return Verdict(*v, backend=self.name, latency_s=time.perf_counter() - t)


# Hand-written worked examples (not taken from the evaluation data), one per
# situation the guidance describes, so a small model sees both answers.
FEW_SHOT = [
    ("Last 30 readings (1 per second):\nTemperature: 27.4 C (change +4.1 C)\n"
     "Humidity: 41.0 % (change -6.0 %)\nTVOC: 1850.0 ppb now, max 1850.0 ppb (change +1790.0 ppb)\n"
     "Sensor model fire probability: now 0.98, mean 0.91\nCamera: not available.\n"
     "Fusion engine state: WARNING",
     {"verdict": "fire", "confidence": 0.9,
      "explanation": "TVOC rose by about 1800 ppb with rising temperature, typical of combustion.",
      "action": "Raise the alarm and check the area."}),
    ("Last 30 readings (1 per second):\nTemperature: 23.1 C (change +0.6 C)\n"
     "Humidity: 88.0 % (change +31.0 %)\nTVOC: 12.0 ppb now, max 15.0 ppb (change +3.0 ppb)\n"
     "Sensor model fire probability: now 0.12, mean 0.08\nCamera: not available.\n"
     "Fusion engine state: WARNING",
     {"verdict": "no_fire", "confidence": 0.85,
      "explanation": "Humidity jumped while TVOC stayed flat, which points to steam, not smoke.",
      "action": "No action; keep monitoring."}),
    ("Last 30 readings (1 per second):\nTemperature: 21.0 C (change +0.0 C)\n"
     "Humidity: 52.0 % (change -0.2 %)\nTVOC: 0.0 ppb now, max 0.0 ppb (change +0.0 ppb)\n"
     "Sensor model fire probability: now 0.03, mean 0.04\n"
     "Camera: max fire/smoke confidence 0.41, detections in 7% of frames.\n"
     "Fusion engine state: WARNING",
     {"verdict": "no_fire", "confidence": 0.8,
      "explanation": "Gas and temperature are flat and camera detections are rare and weak; "
                     "likely a false positive.",
      "action": "Keep monitoring the camera view."}),
]


class _ChatReasoner:
    max_tokens = 160
    few_shot = False

    def messages(self, ctx):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        if self.few_shot:
            for user, answer in FEW_SHOT:
                msgs += [{"role": "user", "content": user},
                         {"role": "assistant", "content": json.dumps(answer)}]
        return msgs + [{"role": "user", "content": ctx.to_prompt()}]

    def reason(self, ctx):
        t = time.perf_counter()
        text = self._chat(self.messages(ctx))
        return parse_verdict(text, self.name, time.perf_counter() - t)


class LlamaCppReasoner(_ChatReasoner):
    name = "llama_cpp"

    def __init__(self, model_path, threads=4):
        from llama_cpp import Llama

        self.llm = Llama(model_path=str(model_path), n_ctx=2048, n_threads=threads, verbose=False)

    def _chat(self, messages):
        out = self.llm.create_chat_completion(messages=messages, temperature=0.0,
                                              max_tokens=self.max_tokens,
                                              response_format={"type": "json_object"})
        return out["choices"][0]["message"]["content"]


class TransformersReasoner(_ChatReasoner):
    name = "transformers"

    def __init__(self, model_path):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        p = Path(model_path)
        self.tok = AutoTokenizer.from_pretrained(str(p.parent), gguf_file=p.name)
        self.model = AutoModelForCausalLM.from_pretrained(str(p.parent), gguf_file=p.name)

    def _chat(self, messages):
        x = self.tok.apply_chat_template(messages, add_generation_prompt=True,
                                         return_tensors="pt", return_dict=True)
        out = self.model.generate(**x, max_new_tokens=self.max_tokens, do_sample=False)
        return self.tok.decode(out[0][x["input_ids"].shape[1]:], skip_special_tokens=True)


class OllamaReasoner(_ChatReasoner):
    name = "ollama"

    def __init__(self, model="qwen2.5:0.5b", url="http://localhost:11434"):
        import urllib.request

        self.model, self.url = model, url.rstrip("/")
        urllib.request.urlopen(self.url + "/api/tags", timeout=2).read()  # fail fast if absent

    def _chat(self, messages):
        import urllib.request

        body = json.dumps({"model": self.model, "messages": messages, "stream": False,
                           "format": "json", "options": {"temperature": 0}}).encode()
        req = urllib.request.Request(self.url + "/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["message"]["content"]


def load_reasoner(cfg):
    """Build the configured reasoner, falling back down the list on failure."""
    from .config import resolve

    backend = cfg.get("backend", "auto")
    model_path = resolve(cfg.get("model_path", ""))
    few_shot = cfg.get("prompt", "zero_shot") == "few_shot"
    order = ["llama_cpp", "transformers", "ollama", "rules"] if backend == "auto" else [backend]
    for name in order:
        try:
            if name == "rules":
                return RuleReasoner()
            if name == "llama_cpp":
                r = LlamaCppReasoner(model_path, cfg.get("threads", 4))
            elif name == "transformers":
                r = TransformersReasoner(model_path)
            elif name == "ollama":
                r = OllamaReasoner(cfg.get("ollama_model", "qwen2.5:0.5b"))
            else:
                continue
            r.few_shot = few_shot
            return r
        except Exception as e:  # missing package, model file or server
            print(f"Reasoner '{name}' unavailable: {e.__class__.__name__}: {str(e)[:100]}")
    print("Falling back to rule-based reasoning.")
    return RuleReasoner()


def verdict_record(ctx, verdict):
    return {"context": asdict(ctx), **asdict(verdict)}
