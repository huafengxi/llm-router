#!/usr/bin/env python3
"""mock_upstream.py — a scriptable fake OpenAI-compatible upstream (tests only).

Used by test_router.py to prove the routing semantics WITHOUT spending a single
real Credit and without decrypting a real credential: the router under test is
pointed at these endpoints through a temporary accounts.yml whose keys are
synthetic values in a temp env file.

Behaviour is scripted per request (a queue of specs, then a default):

    kind: "ok"            200 (non-stream JSON, or SSE when the request streams)
    kind: "error"         `status` + an OpenAI-shaped error body carrying `message`
    stream: bool          force/override SSE for kind "ok"
    chunks: int           SSE content chunks before the usage chunk / [DONE]
    usage: dict|None      usage object; None => the response carries no usage
    usage_chunk: None|bool stream: None (default) = behave like a real upstream,
                          i.e. only emit the trailing usage chunk when the request
                          asked for it via stream_options.include_usage; True/False
                          forces it on/off regardless
    reject_stream_options: 400 when the request body carries stream_options
    delay: float          seconds to sleep before responding
    break_after: int      stream: send N chunks, then an SSE error event
    no_finish: bool       stream: SSE frames that carry neither content nor a
                          finish_reason, then close (the "200 + empty stream"
                          fault: bytes arrive, the client would die)
    abrupt: bool          with break_after: close without the terminating chunk
                          (the router sees an IncompleteRead mid-stream)
    quota_text: str       the error message used by break_after / kind "error"

Every request is recorded (method, path, model, stream flag, whether
stream_options was present, the LAST FOUR characters of the Authorization value
and a request counter) so tests can assert model mapping, key substitution and
which account was hit.  The mock never logs or prints a credential.

Python 3.8 syntax floor; stdlib only.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_QUOTA_TEXT = ("Allocated quota exceeded, please increase your quota "
                      "limit. For details, see: https://help.aliyun.com/")
DEFAULT_THROTTLE_TEXT = ("Requests rate limit exceeded, please try again later. "
                         "You can use `Token Plan` within 200000 requests per minute.")

OK_USAGE = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}


def spec(**kw):
    base = {"kind": "ok", "status": 200, "message": "", "code": None,
            "stream": None, "chunks": 2, "usage": dict(OK_USAGE),
            "usage_chunk": None, "reject_stream_options": False, "delay": 0.0,
            "break_after": None, "abrupt": False, "quota_text": DEFAULT_QUOTA_TEXT,
            "content": "pong", "empty_stream": False, "no_finish": False}
    base.update(kw)
    return base


def error_spec(status, message, **kw):
    out = spec(kind="error", status=status, message=message, **kw)
    return out


def exhausted_spec(status=429, message=None, **kw):
    kw.setdefault("code", "Throttling.AllocationQuota")   # real aliyun shape
    return error_spec(status, message or DEFAULT_QUOTA_TEXT, **kw)


def throttle_spec(status=429, message=None, **kw):
    kw.setdefault("code", "Throttling.RateQuota")         # real aliyun shape
    return error_spec(status, message or DEFAULT_THROTTLE_TEXT, **kw)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "mock-upstream/1"
    sys_version = ""
    mock = None                       # injected per server instance

    def log_message(self, fmt, *args):
        pass                          # the mock stays silent by design

    # ---- helpers ----

    def _read_body(self):
        length = self.headers.get("Content-Length")
        if not length:
            return b""
        try:
            n = int(length)
        except ValueError:
            return b""
        return self.rfile.read(max(0, min(n, 32 * 1024 * 1024)))

    def _record(self, body):
        mock = self.mock
        parsed = None
        if body:
            try:
                parsed = json.loads(body.decode("utf-8"))
            except Exception:
                parsed = None
        auth = self.headers.get("Authorization") or ""
        tail = auth.strip()[-4:] if auth.strip() else ""
        rec = {
            "n": 0,
            "method": self.command,
            "path": self.path,
            "model": (parsed or {}).get("model") if isinstance(parsed, dict) else None,
            "stream": bool((parsed or {}).get("stream")) if isinstance(parsed, dict) else False,
            "has_stream_options": bool(isinstance(parsed, dict)
                                       and isinstance(parsed.get("stream_options"), dict)
                                       and parsed["stream_options"].get("include_usage") is True),
            "stream_options_raw": (parsed or {}).get("stream_options")
                                  if isinstance(parsed, dict) else None,
            "auth_tail": tail,
            "auth_present": bool(auth),
            "headers": {k.lower(): v for k, v in self.headers.items()
                        if k.lower().startswith("x-mock")},
            "body_len": len(body or b""),
            "ts": time.time(),
        }
        with mock.lock:
            rec["n"] = len(mock.requests) + 1
            mock.requests.append(rec)
        return rec, parsed

    def _next_spec(self):
        mock = self.mock
        with mock.lock:
            if mock.scripts:
                return mock.scripts.pop(0)
            return dict(mock.default)

    def _send_error(self, s):
        payload = {"error": {"message": s.get("message") or "mock error",
                             "type": "mock_error", "param": None,
                             "code": s.get("code")}}
        if s.get("status") == 429 and not s.get("code"):
            payload["error"]["code"] = "Throttling.RateQuota"
        body = json.dumps(payload).encode("utf-8")
        self.send_response(s.get("status", 400))
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if s.get("retry_after") is not None:
            self.send_header("Retry-After", str(s["retry_after"]))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _completion(self, model, content, usage):
        obj = {"id": "chatcmpl-mock-%d" % int(time.time() * 1000),
               "object": "chat.completion", "created": int(time.time()),
               "model": model or "mock-model",
               "choices": [{"index": 0, "finish_reason": "stop",
                            "message": {"role": "assistant", "content": content}}]}
        if usage:
            obj["usage"] = usage
        return json.dumps(obj).encode("utf-8")

    def _sse(self, model, content, index, finish=None, usage=None):
        obj = {"id": "chatcmpl-mock-stream", "object": "chat.completion.chunk",
               "created": int(time.time()), "model": model or "mock-model",
               "choices": [{"index": index, "delta": {"content": content},
                            "finish_reason": finish}]}
        if usage is not None:
            obj["usage"] = usage
        return ("data: %s\n\n" % json.dumps(obj)).encode("utf-8")

    def _chunk(self, data):
        self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
        self.wfile.flush()

    def _send_stream(self, s, model, asked_usage=False):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        if s.get("empty_stream"):
            # 200 + headers and an immediately terminated body: the router must
            # treat "nothing arrived" as retryable (no byte reached the client)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            return
        if s.get("no_finish"):
            # 200 + SSE frames that never carry content or a finish_reason: the
            # real-world "empty stream" fault seen on 2026-09-14 (248-483 bytes
            # arrived, so the first-byte peek passes, but the client dies with
            # "Stream ended without finish_reason").  The router must detect it
            # while nothing has been written and switch accounts.
            role_only = {"id": "chatcmpl-mock-stream",
                         "object": "chat.completion.chunk",
                         "created": int(time.time()), "model": model or "mock-model",
                         "choices": [{"index": 0, "delta": {"role": "assistant"},
                                      "finish_reason": None}]}
            self._chunk(("data: %s\n\n" % json.dumps(role_only)).encode("utf-8"))
            self._chunk(b"data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            return
        chunks = int(s.get("chunks") or 0)
        break_after = s.get("break_after")
        sent = 0
        for i in range(chunks):
            if break_after is not None and sent >= int(break_after):
                break
            self._chunk(self._sse(model, "part%d " % i, 0))
            sent += 1
            if s.get("delay"):
                time.sleep(float(s["delay"]))
        if break_after is not None:
            err = {"error": {"message": s.get("quota_text") or DEFAULT_QUOTA_TEXT,
                             "type": "quota_exceeded", "code": "insufficient_quota"}}
            self._chunk(("data: %s\n\n" % json.dumps(err)).encode("utf-8"))
            if s.get("abrupt"):
                # no terminating chunk: the router sees the connection die
                try:
                    self.connection.close()
                except Exception:
                    pass
                self.close_connection = True
                return
            self._chunk(b"data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            return
        usage_chunk = s.get("usage_chunk")
        if usage_chunk is None:
            usage_chunk = bool(asked_usage)     # like a real upstream
        if s.get("usage") and usage_chunk:
            self._chunk(self._sse(model, "", 0, finish="stop", usage=s["usage"]))
        else:
            self._chunk(self._sse(model, "", 0, finish="stop"))
        self._chunk(b"data: [DONE]\n\n")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    # ---- verbs ----

    def finish(self):
        try:
            BaseHTTPRequestHandler.finish(self)
        except OSError:
            pass                          # abrupt-close scripts kill the socket

    def do_GET(self):
        path = self.path.split("?")[0]
        if path.endswith("/models"):
            # not recorded: the router serves /v1/models locally, so a recorded
            # GET here would only distort the per-account request counters
            body = json.dumps({"object": "list",
                               "data": [{"id": "mock-model", "object": "model",
                                         "owned_by": self.mock.name}]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
            return
        rec, _ = self._record(b"")
        s = self._next_spec()
        if s.get("kind") == "error":
            self._send_error(s)
            return
        body = self._completion(rec.get("model"), s.get("content", "pong"), s.get("usage"))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_POST(self):
        body = self._read_body()
        rec, parsed = self._record(body)
        s = self._next_spec()
        if s.get("delay"):
            time.sleep(float(s["delay"]))
        if s.get("reject_stream_options") and rec["has_stream_options"]:
            self._send_error(spec(kind="error", status=400,
                                  message=("Invalid parameter: `stream_options` is "
                                           "not supported by this endpoint"),
                                  code="invalid_request_error"))
            return
        if s.get("kind") == "error":
            self._send_error(s)
            return
        want_stream = s.get("stream")
        if want_stream is None:
            want_stream = bool(isinstance(parsed, dict) and parsed.get("stream"))
        model = rec.get("model")
        if want_stream:
            self._send_stream(s, model, asked_usage=rec["has_stream_options"])
            return
        payload = self._completion(model, s.get("content", "pong"), s.get("usage"))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_HEAD(self):
        self.do_GET()


class _MockServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address):
        pass        # abrupt-close scripts are expected to break the socket


class MockUpstream(object):
    """One fake upstream account. `base_url` is what accounts.yml points at."""

    def __init__(self, name="mock", default=None):
        self.name = name
        self.default = default or spec()
        self.scripts = []
        self.requests = []
        self.lock = threading.RLock()
        self._httpd = None
        self._thread = None
        self.port = None

    # ---- lifecycle ----

    def start(self):
        handler = type("BoundMockHandler", (_Handler,), {"mock": self})
        self._httpd = _MockServer(("127.0.0.1", 0), handler)
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        kwargs={"poll_interval": 0.05},
                                        daemon=True,
                                        name="mock-%s" % self.name)
        self._thread.start()
        return self

    def stop(self, timeout=10):
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    # ---- scripting / assertions ----

    @property
    def base_url(self):
        return "http://127.0.0.1:%d/v1" % self.port

    def script(self, *specs):
        with self.lock:
            self.scripts.extend(specs)
        return self

    def set_default(self, **kw):
        with self.lock:
            self.default = spec(**kw)
        return self

    def reset(self):
        with self.lock:
            self.scripts = []
            self.requests = []
            self.default = spec()

    @property
    def count(self):
        with self.lock:
            return len(self.requests)

    @property
    def models_seen(self):
        with self.lock:
            return [r["model"] for r in self.requests]

    @property
    def auth_tails(self):
        with self.lock:
            return [r["auth_tail"] for r in self.requests]

    @property
    def stream_options_seen(self):
        with self.lock:
            return [r["has_stream_options"] for r in self.requests]

    def last(self):
        with self.lock:
            return dict(self.requests[-1]) if self.requests else None


def main():                                   # manual use: python3 mock_upstream.py 9911
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9911
    handler = type("BoundMockHandler", (_Handler,),
                   {"mock": MockUpstream("manual")})
    httpd = _MockServer(("127.0.0.1", port), handler)
    print("mock upstream on http://127.0.0.1:%d/v1 (Ctrl-C to stop)" % port)
    try:
        httpd.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
