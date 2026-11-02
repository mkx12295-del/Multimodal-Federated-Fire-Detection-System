"""Outbound alerts on state changes: generic JSON webhook and/or Telegram.

Configured through environment variables so no secrets live in the repo:
  FIRENODE_WEBHOOK_URL                   POST a JSON event to this URL
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID   send a Telegram message

Sending happens on a background thread with short timeouts; a slow or
unreachable endpoint can never delay sensing or the buzzer/relay.
"""
import json
import os
import queue
import threading
import urllib.parse
import urllib.request

from .fusion import State


class Alerter:
    def __init__(self, node_id="node", min_state=State.WARNING, webhook_url=None,
                 telegram_token=None, telegram_chat=None, timeout_s=5):
        self.node_id, self.min_state, self.timeout_s = node_id, min_state, timeout_s
        self.webhook_url = webhook_url
        self.telegram = (telegram_token, telegram_chat) if telegram_token and telegram_chat else None
        self._last = State.NORMAL
        self._q = queue.Queue(maxsize=100)
        self.sent, self.failed = 0, 0
        threading.Thread(target=self._loop, daemon=True).start()

    @classmethod
    def from_env(cls, node_id):
        a = cls(node_id, webhook_url=os.environ.get("FIRENODE_WEBHOOK_URL"),
                telegram_token=os.environ.get("TELEGRAM_BOT_TOKEN"),
                telegram_chat=os.environ.get("TELEGRAM_CHAT_ID"))
        return a if a.enabled else None

    @property
    def enabled(self):
        return bool(self.webhook_url or self.telegram)

    def update(self, state, row, verdict=None):
        """Queue an alert when the state rises to min_state or above, or clears."""
        prev, self._last = self._last, state
        if state == prev:
            return False
        rising = state > prev and state >= self.min_state
        cleared = state == State.NORMAL and prev >= self.min_state
        if not (rising or cleared):
            return False
        event = {"node": self.node_id, "state": state.name, "previous": prev.name,
                 "reading": row, "verdict": verdict}
        try:
            self._q.put_nowait(event)
        except queue.Full:
            self.failed += 1
        return True

    def _loop(self):
        while True:
            event = self._q.get()
            try:
                if self.webhook_url:
                    self._post(self.webhook_url, json.dumps(event).encode(),
                               {"Content-Type": "application/json"})
                if self.telegram:
                    token, chat = self.telegram
                    r = event["reading"] or {}
                    text = (f"[{event['node']}] {event['previous']} -> {event['state']}\n"
                            f"T={r.get('Temperature')}C H={r.get('Humidity')}% "
                            f"TVOC={r.get('TVOC')} score={r.get('smoothed')}")
                    if event["verdict"]:
                        text += f"\nLLM: {event['verdict'].get('explanation', '')}"
                    self._post(f"https://api.telegram.org/bot{token}/sendMessage",
                               urllib.parse.urlencode({"chat_id": chat, "text": text}).encode(),
                               {"Content-Type": "application/x-www-form-urlencoded"})
                self.sent += 1
            except Exception as e:  # network errors must never reach the main loop
                self.failed += 1
                print(f"[alerts] send failed: {e.__class__.__name__}: {e}")

    def _post(self, url, data, headers):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            r.read()
