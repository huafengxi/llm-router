"""server.py — ThreadingHTTPServer, routing and the inbound auth gate.

Auth: every endpoint EXCEPT the ones listed in
`accounts.yml -> auth.exempt_paths` (default: /health) requires
`Authorization: Bearer <token>`, where the token is the value of the environment
variable named by `auth.token` and resolved through secrets.py, compared with
hmac.compare_digest.  A missing/mismatched credential yields 401
whose body never echoes the expected or the supplied value.  /health stays
unauthenticated because a liveness probe sends no credential — its payload
therefore carries liveness and a pool summary only (no key, no token, no account
names, no quota or token figures).

Python 3.8 syntax floor; stdlib only.
"""
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs

import classify
import pool as pool_mod
import proxy

SERVICE_NAME = "llm-router"
# The authoritative service version lives in the deployment's service manager,
# outside this repo. Nothing in here re-states it, so there is no second version
# face that could drift out of sync.
MAX_BODY = 64 * 1024 * 1024
CHUNK_HEADER_MAX = 65536

# response headers that must not be relayed to the client
RESPONSE_HOP = frozenset((
    "connection", "transfer-encoding", "content-length", "keep-alive",
    "proxy-authenticate", "proxy-authorization", "te", "trailer", "upgrade",
    "date", "server",
))

UNAUTHORIZED_BODY = {
    "error": {
        "type": "unauthorized",
        "message": "a valid bearer token is required for this endpoint",
    }
}


class RouterApp(object):
    """Shared per-process state handed to every handler thread."""

    def __init__(self, cfg_manager, secrets, pool, logger=None):
        self.cfgm = cfg_manager
        self.secrets = secrets
        self.pool = pool
        self.log = logger
        self.started_at = time.time()

    def cfg(self):
        return self.cfgm.get()

    def uptime(self):
        return time.time() - self.started_at


class RouterHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = SERVICE_NAME
    sys_version = ""
    timeout = None            # no client-side socket timeout: long streams must
                              # not be cut off (upstream reads have their own)

    def version_string(self):
        """`Server: llm-router` — no interpreter/version tail (and no second
        version face next to the deployment's service manager)."""
        return SERVICE_NAME

    # ---------------- plumbing ----------------

    @property
    def app(self):
        return self.server.app

    def log_message(self, fmt, *args):
        """Route the default access log through the redacting logger."""
        if self.app.log:
            self.app.log.debug("%s %s", self.address_string(), fmt % args)

    def log_error(self, fmt, *args):
        if self.app.log:
            self.app.log.warning("handler error: %s", self.app.secrets.redact(fmt % args))

    def _split(self):
        parts = urlsplit(self.path)
        path = parts.path or "/"
        if len(path) > 1 and path.endswith("/"):
            path = path.rstrip("/") or "/"
        return path, parse_qs(parts.query or "")

    def _send_json(self, obj, status=200, extra_headers=None):
        body = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
        self._send_raw(status, body, "application/json", extra_headers)

    def _send_raw(self, status, body, content_type, extra_headers):
        try:
            self.send_response(status)
            if content_type:
                self.send_header("Content-Type", content_type)
            for key, value in (extra_headers or []):
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD" and body:
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            if self.app.log:
                self.app.log.warning("client vanished while writing response: %s",
                                     type(e).__name__)

    def _chunk_start(self, status, content_type, extra_headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
        for key, value in (extra_headers or []):
            self.send_header(key, value)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _chunk_write(self, data):
        if not data:
            return
        self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
        self.wfile.flush()

    def _chunk_end(self):
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    @staticmethod
    def _relay_headers(resp):
        out = []
        for key, value in resp.getheaders():
            if key.lower() in RESPONSE_HOP:
                continue
            out.append((key, value))
        return out

    def _read_body(self):
        """Request body bytes (Content-Length or chunked), capped at MAX_BODY."""
        te = (self.headers.get("Transfer-Encoding") or "").lower()
        if "chunked" in te:
            out = b""
            while True:
                line = self.rfile.readline(CHUNK_HEADER_MAX)
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    size = int(line.split(b";")[0], 16)
                except ValueError:
                    raise ValueError("malformed chunked body")
                if size == 0:
                    while True:
                        trailer = self.rfile.readline(CHUNK_HEADER_MAX)
                        if trailer in (b"\r\n", b"\n", b""):
                            break
                    break
                out += self.rfile.read(size)
                self.rfile.read(2)
                if len(out) > MAX_BODY:
                    raise ValueError("body too large")
            return out
        raw = self.headers.get("Content-Length")
        if raw is None:
            return b""
        try:
            length = int(raw)
        except ValueError:
            raise ValueError("malformed Content-Length")
        if length < 0:
            raise ValueError("negative Content-Length")
        if length > MAX_BODY:
            raise ValueError("body too large")
        return self.rfile.read(length) if length else b""

    def _drain_body(self):
        """Discard an unread request body.

        Without this, a rejected POST leaves its body in the socket buffer and
        the keep-alive loop parses it as the next request line (observed live:
        `Bad request syntax ('{"model":...}')`).  Bounded: an over-sized body
        just closes the connection instead of being drained.
        """
        try:
            te = (self.headers.get("Transfer-Encoding") or "").lower()
            if "chunked" in te:
                self._read_body()
                return
            raw = self.headers.get("Content-Length")
            if not raw:
                return
            remaining = int(raw)
            if remaining < 0 or remaining > MAX_BODY:
                self.close_connection = True
                return
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, 65536))
                if not chunk:
                    break
                remaining -= len(chunk)
        except (ValueError, OSError):
            self.close_connection = True

    # ---------------- auth gate ----------------

    def _authorized(self, path):
        """(allowed: bool, error_body_or_None).  Fails closed."""
        cfg = self.app.cfg()
        if cfg.auth.is_exempt(path):
            return True, None
        try:
            ok = self.app.secrets.check_bearer(self.headers.get("Authorization"))
        except Exception as e:
            if self.app.log:
                self.app.log.error("AUTH_UNAVAILABLE cannot resolve the router "
                                   "token (%s): refusing every authenticated "
                                   "request", type(e).__name__)
            return False, {"error": {"type": "auth_unavailable",
                                     "message": "router token is not available"}}
        if ok:
            return True, None
        if self.app.log:
            self.app.log.warning("AUTH_REJECTED path=%s client=%s (no token "
                                 "echoed)", path, self._client())
        return False, UNAUTHORIZED_BODY

    def _client(self):
        try:
            return "%s:%s" % (self.client_address[0], self.client_address[1])
        except Exception:
            return "?"

    @staticmethod
    def _ms(started):
        """Milliseconds since `started` — the request-level timing face, which
        now lives in the log only (nothing is persisted)."""
        return int((time.time() - started) * 1000)

    # ---------------- routing ----------------

    def do_GET(self):
        self._route("GET", write_body=True)

    def do_HEAD(self):
        self._route("HEAD", write_body=False)

    def do_POST(self):
        self._route("POST", write_body=True)

    def do_PUT(self):
        self._route("PUT", write_body=True)

    def do_PATCH(self):
        self._route("PATCH", write_body=True)

    def do_DELETE(self):
        self._route("DELETE", write_body=True)

    def do_OPTIONS(self):
        path, _ = self._split()
        allowed, err = self._authorized(path)
        if not allowed:
            self._drain_body()
            self._send_json(err, 401, [("WWW-Authenticate", "Bearer")])
            return
        self._send_raw(204, b"", None, [("Allow", "GET, HEAD, POST, OPTIONS")])

    def _route(self, method, write_body=True):
        started = time.time()
        path, _query = self._split()
        allowed, err = self._authorized(path)
        if not allowed:
            self._drain_body()
            self._send_json(err, 401, [("WWW-Authenticate", "Bearer")])
            return
        try:
            if path == "/health":
                self._send_json(self.app.pool.health_view(
                    uptime_s=self.app.uptime()))
                return
            if path == "/v1/models" and method in ("GET", "HEAD"):
                self._send_json(self.app.pool.models_view())
                return
            if path == "/v1" or path == "/":
                self._send_json({"service": SERVICE_NAME,
                                 "endpoints": ["/health", "/v1/models",
                                               "/v1/chat/completions", "/v1/*"]})
                return
            if path.startswith("/v1/") or path.startswith("/v1?"):
                self._handle_proxy(method, path, started)
                return
            self._drain_body()
            self._send_json({"error": {"type": "not_found",
                                       "message": "unknown path %s" % path}}, 404)
        except Exception as e:                       # never leak a stack to the client
            detail = self.app.secrets.redact("%s: %s" % (type(e).__name__, e))
            if self.app.log:
                self.app.log.exception("INTERNAL_ERROR path=%s", path)
            try:
                self._send_json({"error": {"type": "internal_error",
                                           "message": "router internal error"}}, 500)
            except Exception:
                pass
            if self.app.log:
                self.app.log.error("INTERNAL_ERROR detail=%s", detail)

    # ---------------- upstream proxy ----------------

    def _handle_proxy(self, method, path, started):
        app = self.app
        cfg = app.pool.cfg()
        try:
            body = self._read_body()
        except ValueError as e:
            self.close_connection = True
            self._send_json({"error": {"type": "bad_request", "message": str(e)}}, 400)
            return
        model = None
        stream_requested = False
        parsed = None
        if body:
            try:
                obj = json.loads(body.decode("utf-8"))
                if isinstance(obj, dict):
                    parsed = obj
                    model = obj.get("model") if isinstance(obj.get("model"), str) else None
                    stream_requested = bool(obj.get("stream"))
            except (ValueError, UnicodeDecodeError):
                parsed = None
        accounts, skipped = app.pool.candidates(model)
        if not accounts:
            self._all_exhausted(model, skipped, started)
            return
        attempt = 0
        for acct in accounts:
            attempt += 1
            outcome = self._try_account(acct, method, path, body, parsed, model,
                                        stream_requested, cfg, attempt, skipped,
                                        started)
            if outcome in ("sent", "returned"):
                return
        # every candidate failed
        self._all_exhausted(model, skipped, started, failed=True, attempt=attempt)

    def _all_exhausted(self, model, skipped, started, failed=False, attempt=0):
        app = self.app
        cfg = app.pool.cfg()
        now = time.time()
        details = []
        for acct in cfg.ordered():
            if model and acct.upstream_model(model) is None:
                reason = "no_mapping"
                until = None
            else:
                st = app.pool.state_for(acct.name)
                reason = (skipped.get(acct.name) or st.get("blacklist_reason")
                          or st.get("status") or "failed")
                until = st.get("blacklist_until")
            details.append({"name": acct.name, "reason": reason,
                            "until": pool_mod.iso_or_none(until) if until else None,
                            "recovers_in_s": (round(until - now, 1)
                                              if until and until > now else None)})
        body = {"error": {
            "type": self._pool_error_type(details, failed),
            "message": ("every account in the pool is unavailable for model %r; "
                        "the last account of the pool IS a pay-as-you-go one — "
                        "there is no fallback outside the pool" % (model,)),
            "model": model,
            "accounts": details,
        }}
        if app.log:
            app.log.error("POOL_EXHAUSTED model=%s pool=%s accounts=%s attempt=%d "
                          "duration_ms=%d (the pool's last account is a "
                          "pay-as-you-go one; no fallback outside the pool)",
                          model, app.pool.pool_status(),
                          app.secrets.redact(json.dumps(details)), attempt,
                          self._ms(started))
        self._send_json(body, 503)

    @staticmethod
    def _pool_error_type(details, failed):
        """503 flavour: exhaustion vs. no mapping vs. generic failure."""
        relevant = [d for d in details if d["reason"] != "no_mapping"]
        if not relevant:
            return "no_account_for_model"
        if all(d["reason"] in pool_mod.EXHAUSTION_REASONS for d in relevant):
            return "all_accounts_exhausted"
        return "all_accounts_failed" if failed else "all_accounts_unavailable"

    def _try_account(self, acct, method, path, body, parsed, model,
                     stream_requested, cfg, attempt, skipped, started):
        """One candidate account.  -> 'sent' (response written) | 'next'."""
        app = self.app
        up_model = acct.upstream_model(model) if model else None
        try:
            key = app.pool.key_for(acct)
        except pool_mod.AccountKeyError:
            return "next"
        key_hint = app.secrets.mask(key)
        body_out, injected, _ = proxy.prepare_body(
            body, up_model, stream_requested, cfg.inject_stream_options)
        headers = proxy.build_headers(self.headers, key)
        noinject_tries = 0
        while True:
            try:
                conn, resp = proxy.open_upstream(
                    acct, method, self.path, headers,
                    body_out if method in ("POST", "PUT", "PATCH") else None)
            except proxy.UpstreamError as e:
                detail = app.secrets.redact(str(e))
                app.pool.mark_failure(acct, pool_mod.R_SERVER_ERROR,
                                      detail=detail[:200])
                if app.log:
                    app.log.warning("UPSTREAM_UNREACHABLE account=%s path=%s %s "
                                    "-> next account (blacklisted %ss) attempt=%d "
                                    "duration_ms=%d", acct.name, path,
                                    detail[:200], self._bl(app), attempt,
                                    self._ms(started))
                return "next"
            status = resp.status
            ctype = resp.getheader("Content-Type") or ""
            if 200 <= status < 300:
                if stream_requested and "text/event-stream" in ctype.lower():
                    return self._relay_ok_stream(acct, conn, resp, model, up_model,
                                                 attempt, started, key_hint)
                return self._relay_ok_body(acct, conn, resp, model, up_model,
                                           attempt, started, key_hint)
            err = proxy.read_error_body(resp)
            err_headers = self._relay_headers(resp)
            try:
                resp.close()
                conn.close()
            except Exception:
                pass
            err_text = err.decode("utf-8", "replace")
            # stream_options.include_usage rejected -> drop it and retry once
            if status == 400 and injected and noinject_tries == 0:
                noinject_tries += 1
                body_out, injected, _ = proxy.prepare_body(
                    body, up_model, stream_requested, inject=False)
                if app.log:
                    app.log.warning("STREAM_OPTIONS_REJECTED account=%s dropped "
                                    "stream_options.include_usage and retried "
                                    "once (the REQ ok line reports "
                                    "usage_source=missing)", acct.name)
                continue
            # `up_model` is this account's own mapping of the requested pool
            # model: when it exists, an upstream 404 means the account cannot
            # serve a model it claims to serve (account-side fault -> switch),
            # while a 404 for a request that named no model stays a path-level
            # client error (see classify.classify).
            category, evidence = classify.classify(
                status, err_text, mapped_model=up_model is not None)
            redacted = app.secrets.redact(err_text)[:200]
            if category == classify.EXHAUSTED:
                app.pool.mark_exhausted(acct, detail="%s %s" % (evidence, redacted),
                                        http_status=status)
                if app.log:
                    app.log.warning("EXHAUSTED_SWITCH account=%s status=%s %s -> "
                                    "next account (blacklisted %ss) attempt=%d "
                                    "duration_ms=%d", acct.name, status, evidence,
                                    self._bl(app, exhausted=True), attempt,
                                    self._ms(started))
                return "next"
            if category == classify.THROTTLED:
                app.pool.mark_failure(acct, pool_mod.R_THROTTLED, detail=evidence,
                                      http_status=status)
                if app.log:
                    app.log.warning("THROTTLED_SWITCH account=%s status=%s %s -> "
                                    "next account (blacklisted %ss, no "
                                    "same-account retry) attempt=%d duration_ms=%d",
                                    acct.name, status, evidence, self._bl(app),
                                    attempt, self._ms(started))
                return "next"
            if category in (classify.SERVER_ERROR, classify.UNKNOWN):
                app.pool.mark_failure(acct, pool_mod.R_SERVER_ERROR,
                                      detail=redacted, http_status=status)
                if app.log:
                    app.log.warning("UPSTREAM_ERROR account=%s status=%s %s -> next "
                                    "account (blacklisted %ss) attempt=%d "
                                    "duration_ms=%d", acct.name, status, evidence,
                                    self._bl(app), attempt, self._ms(started))
                return "next"
            if status == 401:
                # the upstream refused OUR credential: that is this account's
                # fault, so it is blacklisted and the next candidate is tried
                app.pool.mark_failure(acct, pool_mod.R_UPSTREAM_KEY_REJECTED,
                                      detail=redacted, http_status=status)
                if app.log:
                    app.log.warning("UPSTREAM_KEY_REJECTED account=%s key=%s — the "
                                    "upstream refused our credential (401); "
                                    "blacklisted %ss -> next account attempt=%d "
                                    "duration_ms=%d", acct.name, key_hint,
                                    self._bl(app), attempt, self._ms(started))
                return "next"
            # any other client_error: the same parameters would fail on every
            # account, so it is returned unchanged — no blacklist, no switch
            app.pool.record_failure(acct, redacted, status)
            if app.log:
                app.log.info("CLIENT_ERROR account=%s status=%s %s — returned to "
                             "the client unchanged (no account switch, no "
                             "blacklist) attempt=%d duration_ms=%d",
                             acct.name, status, evidence, attempt,
                             self._ms(started))
            self._send_raw(status, err, ctype or "application/json", err_headers)
            return "sent"

    @staticmethod
    def _bl(app, exhausted=False):
        """The blacklist duration this switch just applied (for log lines)."""
        cfg = app.cfg()
        return round(cfg.blacklist_exhausted if exhausted
                     else cfg.blacklist_failure)

    # ---------------- success paths ----------------

    def _relay_ok_body(self, acct, conn, resp, model, up_model, attempt, started,
                       key_hint):
        """Non-stream 2xx: buffer, forward verbatim, log the token counts."""
        app = self.app
        ctype = resp.getheader("Content-Type") or "application/json"
        headers = self._relay_headers(resp)
        try:
            data = resp.read()
        except Exception as e:
            detail = app.secrets.redact("%s: %s" % (type(e).__name__, e))
            app.pool.mark_failure(acct, pool_mod.R_SERVER_ERROR, detail=detail,
                                  http_status=resp.status)
            try:
                resp.close()
                conn.close()
            except Exception:
                pass
            self._send_json({"error": {"type": "upstream_read_error",
                                       "message": "could not read the upstream "
                                                  "response body"}}, 502)
            return "sent"
        try:
            resp.close()
            conn.close()
        except Exception:
            pass
        usage = proxy.usage_from_body(data)
        prompt, completion = proxy.usage_tokens(usage)
        source = "upstream" if usage else "missing"
        app.pool.record_ok(acct, key_hint=key_hint)
        if app.log:
            app.log.info("REQ ok account=%s model=%s upstream_model=%s status=%s "
                         "stream=false tokens=%d/%d usage_source=%s attempt=%d "
                         "duration_ms=%d client=%s",
                         acct.name, model, up_model, resp.status, prompt, completion,
                         source, attempt, self._ms(started), self._client())
        self._send_raw(resp.status, data, ctype, headers)
        return "sent"

    def _relay_ok_stream(self, acct, conn, resp, model, up_model, attempt, started,
                         key_hint):
        """Stream 2xx: peek the first chunk (still retryable), then relay raw."""
        app = self.app
        ctype = resp.getheader("Content-Type") or "text/event-stream"
        headers = self._relay_headers(resp)
        try:
            first = resp.read1(proxy.READ_SIZE)
        except Exception as e:
            detail = app.secrets.redact("%s: %s" % (type(e).__name__, e))
            app.pool.mark_failure(acct, pool_mod.R_SERVER_ERROR, detail=detail,
                                  http_status=resp.status)
            try:
                resp.close()
                conn.close()
            except Exception:
                pass
            if app.log:
                app.log.warning("STREAM_FIRST_BYTE_FAILED account=%s %s -> next "
                                "account (blacklisted %ss; nothing sent to the "
                                "client yet) attempt=%d duration_ms=%d",
                                acct.name, detail[:200], self._bl(app), attempt,
                                self._ms(started))
            return "next"
        if not first:
            app.pool.mark_failure(acct, pool_mod.R_SERVER_ERROR,
                                  detail="upstream returned an empty stream",
                                  http_status=resp.status)
            try:
                resp.close()
                conn.close()
            except Exception:
                pass
            return "next"
        # Bytes arrived, but that alone does not mean the stream is usable: an
        # upstream can send a few SSE frames (role-only delta, [DONE]) and close
        # without ever carrying content or a finish_reason.  Relayed as-is the
        # client dies with "Stream ended without finish_reason" while this router
        # logs `REQ ok` => the pool never switches.  Probe the head while it is
        # still free to do so: nothing has been written to the client yet.
        try:
            first, verdict = proxy.probe_stream_start(resp, first)
        except Exception as e:
            detail = app.secrets.redact("%s: %s" % (type(e).__name__, e))
            app.pool.mark_failure(acct, pool_mod.R_SERVER_ERROR, detail=detail,
                                  http_status=resp.status)
            try:
                resp.close()
                conn.close()
            except Exception:
                pass
            if app.log:
                app.log.warning("STREAM_FIRST_BYTE_FAILED account=%s %s -> next "
                                "account (blacklisted %ss; nothing sent to the "
                                "client yet) attempt=%d duration_ms=%d",
                                acct.name, detail[:200], self._bl(app), attempt,
                                self._ms(started))
            return "next"
        if verdict == "eof":
            app.pool.mark_failure(
                acct, pool_mod.R_SERVER_ERROR,
                detail="upstream 200 stream ended without content or "
                       "finish_reason (%d bytes)" % len(first),
                http_status=resp.status)
            try:
                resp.close()
                conn.close()
            except Exception:
                pass
            if app.log:
                app.log.warning("EMPTY_STREAM account=%s model=%s bytes=%d "
                                "saw_finish_reason=False usage_source=missing "
                                "-> next account (blacklisted %ss; nothing sent "
                                "to the client yet) attempt=%d duration_ms=%d",
                                acct.name, model, len(first), self._bl(app),
                                attempt, self._ms(started))
            return "next"
        # from here on bytes reach the client => no more account switching
        try:
            self._chunk_start(resp.status, ctype, headers)
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            if app.log:
                app.log.warning("client vanished before the stream started: %s",
                                type(e).__name__)
            try:
                resp.close()
                conn.close()
            except Exception:
                pass
            return "sent"
        usage = None
        relayed = 0
        errors = []
        broken = None
        sent = 0

        def counting_write(chunk):
            # relay_stream reports its own byte count only when it returns; if it
            # raises (read timeout, client gone) the tuple assignment never
            # happens and the log line used to claim bytes_sent=0 even though the
            # client had been served.  Count on the write side so STREAM_INTERRUPTED
            # carries the truth (that field is what tells "nothing reached the
            # client" from "the stream died half way").
            nonlocal sent
            sent += len(chunk)
            self._chunk_write(chunk)

        try:
            usage, relayed, errors = proxy.relay_stream(resp, counting_write,
                                                        first=first)
        except Exception as e:
            broken = app.secrets.redact("%s: %s" % (type(e).__name__, e))
            relayed = sent
        try:
            resp.close()
            conn.close()
        except Exception:
            pass
        prompt, completion = proxy.usage_tokens(usage)
        source = "upstream" if usage else "missing"
        stream_text = " ".join(errors)
        quota_hit = classify.has_quota_semantics(stream_text) or \
            (broken is not None and classify.has_quota_semantics(broken))
        if broken is not None or errors:
            detail = broken or ("upstream error event in stream: %s"
                                % app.secrets.redact(stream_text)[:200])
            if quota_hit:
                app.pool.mark_exhausted(acct, detail=detail, http_status=resp.status)
            else:
                # NOT blacklisted: this branch also fires when the *client*
                # vanished mid-stream (a failing write() lands here too), so the
                # account cannot be told apart from a broken peer
                app.pool.record_failure(acct, detail, resp.status)
            try:
                if broken is not None:
                    self._chunk_write(proxy.error_event(acct.name,
                                                        "upstream stream interrupted"))
                self._chunk_end()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            if app.log:
                app.log.error("STREAM_INTERRUPTED account=%s model=%s bytes_sent=%d "
                              "usage_source=%s quota_semantics=%s attempt=%d "
                              "duration_ms=%d detail=%s",
                              acct.name, model, relayed, source, quota_hit, attempt,
                              self._ms(started), detail[:200])
            return "sent"
        app.pool.record_ok(acct, key_hint=key_hint)
        try:
            self._chunk_end()          # terminate the chunked body
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            # the upstream stream was fully relayed and logged; only the
            # client went away before the terminating chunk landed
            if app.log:
                app.log.warning("CLIENT_GONE account=%s stream fully relayed "
                                "(%d bytes) but the client vanished before the "
                                "terminating chunk: %s", acct.name, relayed,
                                type(e).__name__)
        if app.log:
            app.log.info("REQ ok account=%s model=%s upstream_model=%s status=%s "
                         "stream=true tokens=%d/%d usage_source=%s bytes=%d "
                         "attempt=%d duration_ms=%d client=%s",
                         acct.name, model, up_model, resp.status, prompt, completion,
                         source, relayed, attempt, self._ms(started),
                         self._client())
        return "sent"


class RouterServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, app, handler=RouterHandler):
        self.app = app
        ThreadingHTTPServer.__init__(self, addr, handler)


def serve(app, host, port, logger=None):
    server = RouterServer((host, port), app)
    if logger:
        logger.info("LISTENING %s://%s:%d (auth: bearer token required except %s)",
                    "http", host, port,
                    ",".join(app.cfg().auth.exempt_paths))
    return server
