#!/usr/bin/env python3
"""Mock LLM API — a tiny local server for testing 'Custom' API targets.

Run it, then point a Custom API target in PyRIT-Bench at it. It accepts any URL
path, pulls the prompt out of common request shapes, and replies in *every*
common response shape at once, so whatever Response path you choose will work:

    output.text                  (the Custom preset default)
    choices.0.message.content    (OpenAI shape)
    content.0.text               (Anthropic shape)
    response                     (Ollama shape)

Usage
-----
    python mock_llm_api.py                # http://127.0.0.1:8000
    PORT=9001 python mock_llm_api.py      # pick a port
    HOST=0.0.0.0 python mock_llm_api.py   # expose on your LAN (other machines)
    MOCK_REPLY="hello" python mock_llm_api.py    # custom canned reply
    MOCK_DELAY_MS=500 python mock_llm_api.py     # simulate latency

Auth is ignored (any key works), but every request is printed so you can
confirm your {api_key} / headers actually arrived. All replies are benign
stubs — no real model output is generated.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
REPLY = os.environ.get(
    "MOCK_REPLY",
    "This is a mock model response for testing the pipeline. No real content is generated.",
)
DELAY_MS = int(os.environ.get("MOCK_DELAY_MS", "0"))


def extract_prompt(data) -> str:
    """Best-effort: pull the user prompt out of common request shapes."""
    if not isinstance(data, dict):
        return str(data)
    for key in ("input", "prompt", "text", "query"):
        v = data.get(key)
        if isinstance(v, str):
            return v
    msgs = data.get("messages")
    if isinstance(msgs, list) and msgs:
        last = msgs[-1]
        if isinstance(last, dict):
            content = last.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):  # Anthropic-style content blocks
                parts = [b.get("text", "") for b in content if isinstance(b, dict)]
                return " ".join(p for p in parts if p)
    return json.dumps(data)


def reply_payload(reply: str) -> dict:
    """Return the reply in several shapes so any Response path resolves."""
    return {
        "output": {"text": reply},                                              # output.text
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": reply}}],       # choices.0.message.content
        "content": [{"type": "text", "text": reply}],                            # content.0.text
        "response": reply,                                                       # response
        "model": "mock-llm-1",
    }


class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            data = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            data = {"_raw": raw.decode("utf-8", "replace")}
        prompt = extract_prompt(data)
        auth = (self.headers.get("Authorization")
                or self.headers.get("x-api-key")
                or self.headers.get("api-key") or "(none)")
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"[{ts}] POST {self.path}  auth={auth!r}  prompt={prompt[:100]!r}", flush=True)
        if DELAY_MS:
            time.sleep(DELAY_MS / 1000.0)
        self._json(200, reply_payload(REPLY))

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            self._json(200, {"status": "ok", "service": "mock-llm-api"})
            return
        shown_host = HOST if HOST != "0.0.0.0" else "<this-machine-ip>"
        info = (
            "Mock LLM API is running.\n\n"
            "Add a target in PyRIT-Bench -> API -> Custom, then under Advanced set:\n"
            f"  Endpoint URL : http://{shown_host}:{PORT}/chat   (any path works)\n"
            '  Request body : {"input": "{prompt}"}\n'
            "  Response path: output.text\n"
            "  API key      : leave blank (auth is ignored)\n\n"
            "Other Response paths also work: choices.0.message.content, content.0.text, response\n"
        )
        body = info.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:  # silence default logging; we log in do_POST
        pass


def main() -> None:
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Mock LLM API listening on http://{HOST}:{PORT}  (Ctrl+C to stop)", flush=True)
    print('  Custom target -> Endpoint http://127.0.0.1:%d/chat | '
          'body {"input":"{prompt}"} | Response path output.text' % PORT, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
