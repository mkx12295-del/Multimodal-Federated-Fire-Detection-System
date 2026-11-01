"""Live status page served by the node (standard library only).

GET /            HTML page that polls /status once a second
GET /status      JSON: latest reading, state, recent history, last LLM verdict
"""
import json
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Status:
    def __init__(self, history=120):
        self._lock = threading.Lock()
        self._history = deque(maxlen=history)
        self._extra = {}

    def push(self, row):
        with self._lock:
            self._history.append(row)

    def update(self, **kw):
        with self._lock:
            self._extra.update(kw)

    def snapshot(self):
        with self._lock:
            hist = list(self._history)
            return {"latest": hist[-1] if hist else None, "history": hist, **self._extra}


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fire Node</title><style>
:root{--bg:#f6f7f9;--fg:#1d2330;--card:#fff;--mute:#667085;--ok:#2e7d32;--warn:#b26a00;--bad:#c62828}
@media (prefers-color-scheme:dark){:root{--bg:#12151b;--fg:#e6e8ec;--card:#1b2029;--mute:#98a2b3}}
body{margin:0;font:15px system-ui,sans-serif;background:var(--bg);color:var(--fg)}
main{max-width:860px;margin:auto;padding:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.card{background:var(--card);border-radius:10px;padding:12px 14px}
.k{color:var(--mute);font-size:12px}.v{font-size:22px;font-weight:600;margin-top:4px}
#state{font-size:28px;font-weight:700}.NORMAL{color:var(--ok)}.WARNING{color:var(--warn)}.ALARM{color:var(--bad)}
svg{width:100%;height:140px}
</style></head><body><main>
<h2>Fire Node</h2>
<div class="card"><div class="k">State</div><div id="state">-</div></div><br>
<div class="grid" id="tiles"></div><br>
<div class="card"><div class="k">Fused score (last 2 min)</div>
<svg viewBox="0 0 120 40" preserveAspectRatio="none"><polyline id="line" fill="none"
stroke="currentColor" stroke-width="0.6"/></svg></div><br>
<div class="card"><div class="k">Last reasoning verdict</div><div id="llm">none yet</div></div>
</main><script>
const f=(x,d=2)=>x===''||x==null?'n/a':Number(x).toFixed(d);
async function tick(){try{const s=await (await fetch('/status')).json();const l=s.latest;if(!l)return;
const st=document.getElementById('state');st.textContent=l.state;st.className=l.state;
const t=[['Temperature',f(l.Temperature,1)+' C'],['Humidity',f(l.Humidity,1)+' %'],['TVOC',f(l.TVOC,0)+' ppb'],
['Sensor prob',f(l.sensor_prob)],['Vision conf',f(l.vision_conf)],['Smoothed score',f(l.smoothed)]];
document.getElementById('tiles').innerHTML=t.map(([k,v])=>`<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');
const h=s.history;document.getElementById('line').setAttribute('points',h.map((r,i)=>`${i*120/Math.max(1,h.length-1)},${40-40*r.smoothed}`).join(' '));
if(s.last_verdict){const v=s.last_verdict;document.getElementById('llm').textContent=`${v.verdict} (${f(v.confidence)}, ${v.backend}): ${v.explanation}`}
}catch(e){}}
setInterval(tick,1000);tick();
</script></body></html>"""


def start_dashboard(status, host="0.0.0.0", port=8000):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/status":
                body, ctype = json.dumps(status.snapshot()).encode(), "application/json"
            elif self.path == "/":
                body, ctype = PAGE.encode(), "text/html; charset=utf-8"
            else:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
