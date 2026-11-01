"""Tiny webhook receiver for the Docker demo: prints alerts, lists them at GET /events."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

EVENTS = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        event = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        EVENTS.append(event)
        print(f"ALERT {event['node']}: {event['previous']} -> {event['state']}", flush=True)
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        body = json.dumps(EVENTS[-200:]).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 9000), Handler).serve_forever()
