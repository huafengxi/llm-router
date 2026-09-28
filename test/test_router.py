#!/usr/bin/env python3
"""test_router.py — llm-router verification matrix (unittest, stdlib only).

Run (from the repo root):
    python3 -m unittest discover -s test -v
    python3 -m unittest discover -s test -v   # any python >= 3.8 (the floor)

The end-to-end cases start `router.py` as a subprocess plus N mock upstreams on
random high ports (never 9200, so a live service is untouched) and drive them
with synthetic credentials only — no real credential is read and no real quota
is spent. All waits are bounded; there is no unconditional wait anywhere.

Pool state is process memory only: the observable faces are `/v1/models`
(available / reason / until), `/health` (pool / accounts_available), the router
log and the mock request counters. Nothing is persisted, so a restart resets
every blacklist (T22).

Matrix: T1 definition order, T2 exhausted switch, T3 blacklist recovery,
T4 throttle switch, T5 classify table, T6 non-stream retry, T7 stream interrupt
(clean + abrupt), T8 stream retry before the first byte, T10 model mapping,
T12 whole pool exhausted, T13 /v1/models, T14 /health, T15 secrets faces,
T16 hot reload, T17 config validation, T18 inbound auth, T19 stream_options
injection, T20 injection 400 fallback, T21 generic passthrough,
T22 restart resets blacklists, T23 concurrency consistency, T24 retired keys
rejected, T25 upstream 401 rejected, T26 empty stream with no finish_reason
(200 + SSE frames that carry neither content nor a finish_reason => switch
accounts while nothing has been written).
"""
import http.client
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
WS = os.path.dirname(PKG)
for _p in (PKG, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import classify                       # noqa: E402
import config as config_mod           # noqa: E402
import pool as pool_mod               # noqa: E402
import proxy as proxy_mod             # noqa: E402
import secrets as secrets_mod         # noqa: E402
import mock_upstream as mock          # noqa: E402

# scratch root for the temporary accounts copies + fake credentials (gitignored,
# inside the repo so a standalone checkout never writes outside itself).
# Overridable so a caller can keep every artifact under its own scratch dir.
TMP_ROOT = os.environ.get("LLM_ROUTER_TEST_TMP") or \
    os.path.join(PKG, ".test-tmp")

# synthetic credentials — these strings must NEVER appear in a persisted face
FAKE_KEYS = {"a": "sk-fake-aaaa1111", "b": "sk-fake-bbbb2222", "c": "sk-fake-cccc3333"}
# credentials are environment variables, referenced by NAME from accounts.yml
KEY_ENV = dict((k, "FAKE_KEY_%s" % k.upper()) for k in FAKE_KEYS)   # a/b/c -> FAKE_KEY_A…
KEY_ENV_NAME = "K"          # the name unit-test configs point at
TOKEN_ENV = "FAKE_ROUTER_TOKEN"
FAKE_TOKEN = "deadbeefcafebabedeadbeefcafebabedeadbeefcafebabedeadbeefcafebabe"
FAKE_MASKS = {"a": "sk-fake-****1111", "b": "sk-fake-****2222", "c": "sk-fake-****3333"}
FAKE_TOKEN_MASK = "****babe"
MODEL = "qwen3.8-max"
CHAT = "/v1/chat/completions"

START_TIMEOUT = 25
REQ_TIMEOUT = 30


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def accounts_block(name, base_url, key_env, models, extra=""):
    """One `accounts:` list entry. The rotation order is the writing order.

    `key_env` is the NAME of an environment variable: the router reads its
    credentials from its own environment and never from a file, so a test
    injects synthetic values into the child env instead of writing key files.
    """
    lines = ["  - name: %s" % name,
             "    base_url: %s" % base_url,
             "    key: %s" % key_env]
    if extra:
        lines.append("    " + extra)
    lines.append("    models:")
    for pool_name, up_name in sorted(models.items()):
        lines.append("      %s: %s" % (pool_name, up_name))
    return "\n".join(lines)


def write_accounts(path, specs, token_env=TOKEN_ENV, defaults=None):
    """`token_env` is the NAME of the inbound-token environment variable."""
    d = {"blacklist_exhausted": "2", "blacklist_failure": "2",
         "inject": "true"}
    d.update(defaults or {})
    text = """defaults:
  blacklist_exhausted: %(blacklist_exhausted)s
  blacklist_failure: %(blacklist_failure)s
  timeout: {connect: 5, read: 30}
  inject_stream_options: %(inject)s
auth:
  token: %(token_env)s
  exempt_paths: [/health]
accounts:
%(accounts)s
""" % dict(d, token_env=token_env,
           accounts="\n".join(specs))
    with open(path, "w") as fh:
        fh.write(text)
    return path


class Response(object):
    def __init__(self, status, headers, body):
        self.status = status
        self.headers = headers
        self.body = body

    def json(self):
        return json.loads(self.body.decode("utf-8"))

    def text(self):
        return self.body.decode("utf-8", "replace")

    def __repr__(self):
        return "<Response %s %d bytes>" % (self.status, len(self.body or b""))


class Client(object):
    def __init__(self, port, token=FAKE_TOKEN):
        self.port = port
        self.token = token

    def req(self, method, path, body=None, token="__default__", headers=None,
            timeout=REQ_TIMEOUT):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        hdrs = dict(headers or {})
        tok = self.token if token == "__default__" else token
        if tok is not None:
            hdrs["Authorization"] = "Bearer " + tok
        payload = body
        if isinstance(body, (dict, list)):
            payload = json.dumps(body).encode("utf-8")
            hdrs.setdefault("Content-Type", "application/json")
        elif isinstance(body, str):
            payload = body.encode("utf-8")
        try:
            conn.request(method, path, body=payload, headers=hdrs)
            resp = conn.getresponse()
            data = resp.read()
            out = Response(resp.status, dict(resp.getheaders()), data)
        finally:
            conn.close()
        return out

    def get(self, path, **kw):
        return self.req("GET", path, **kw)

    def post(self, path, body=None, **kw):
        return self.req("POST", path, body=body, **kw)

    def chat(self, model=MODEL, stream=False, extra=None, **kw):
        body = {"model": model, "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1}
        if stream:
            body["stream"] = True
        if extra:
            body.update(extra)
        return self.post(CHAT, body, **kw)

    def stream_chat(self, model=MODEL, extra=None, deadline=20, **kw):
        """POST a streaming chat request and collect the raw chunks."""
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=deadline)
        body = {"model": model, "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1, "stream": True}
        if extra:
            body.update(extra)
        hdrs = {"Content-Type": "application/json",
                "Authorization": "Bearer " + (self.token or "")}
        hdrs.update(kw.pop("headers", None) or {})
        conn.request("POST", CHAT, body=json.dumps(body).encode("utf-8"),
                     headers=hdrs)
        resp = conn.getresponse()
        chunks = []
        status = resp.status
        headers = dict(resp.getheaders())
        if status == 200:
            end = time.time() + deadline
            while time.time() < end:
                try:
                    piece = resp.read1(65536)
                except Exception as e:
                    chunks.append(b"<<READ_ERROR %s>>" % type(e).__name__.encode())
                    break
                if not piece:
                    break
                chunks.append(piece)
        else:
            chunks.append(resp.read())
        conn.close()
        return Response(status, headers, b"".join(chunks))


class Fixture(object):
    """temp dir + synthetic credentials in the child env + N mock upstreams +
    a router subprocess."""

    def __init__(self, name="fx", accounts=None, defaults=None, n_mocks=3,
                 mock_defaults=None, log_level="INFO"):
        self.name = name
        self.accounts_override = accounts
        self.defaults = defaults
        self.n_mocks = n_mocks
        self.mock_defaults = mock_defaults or {}
        self.log_level = log_level
        self.tmp = None
        self.mocks = []
        self.proc = None
        self.port = None
        self.client = None
        self.log_path = None
        self.accounts_path = None
        self.seed = set()          # paths this fixture wrote itself
        self.run_dir = None        # WS-relative run/ dir the router must NOT create

    # ---- lifecycle ----

    def start(self):
        os.makedirs(TMP_ROOT, exist_ok=True)
        self.tmp = tempfile.mkdtemp(prefix=self.name + "-", dir=TMP_ROOT)
        self.log_path = os.path.join(self.tmp, "router.log")
        self.seed.add(self.log_path)
        # the child's credential environment: accounts.yml names these variables
        # and secrets.py reads them from os.environ (no file, no decryptor)
        self.env_vars = dict((v, FAKE_KEYS[k]) for k, v in KEY_ENV.items())
        self.env_vars[TOKEN_ENV] = FAKE_TOKEN
        self.child_env = dict(os.environ)
        self.child_env.update(self.env_vars)

        names = ["alpha", "beta", "gamma"][:self.n_mocks]
        specs = []
        for i, letter in enumerate("abc"[:self.n_mocks]):
            m = mock.MockUpstream(names[i],
                                  default=dict(self.mock_defaults.get(letter)
                                               or mock.spec()))
            m.start()
            self.mocks.append(m)
            if letter == "a":
                models = {MODEL: MODEL, "glm-5.2": "glm-5.2"}
            elif letter == "b":
                models = {MODEL: MODEL}
            else:
                models = {MODEL: "qwen/" + MODEL,
                          "claude-fable-5.1": "anthropic/claude-fable-5.1"}
            specs.append(accounts_block(names[i], m.base_url,
                                        KEY_ENV[letter], models))
        if self.accounts_override:
            specs = self.accounts_override(specs, self.mocks, self.env_vars)
        self.accounts_path = write_accounts(os.path.join(self.tmp, "accounts.yml"),
                                            specs, TOKEN_ENV, self.defaults)
        self.seed.add(self.accounts_path)
        self.port = free_port()
        cmd = [sys.executable, os.path.join(PKG, "router.py"),
               "--host", "127.0.0.1", "--port", str(self.port),
               "--accounts", self.accounts_path,
               "--log-level", self.log_level]
        self.cmd = cmd
        self._log_fh = open(self.log_path, "ab")
        self.proc = subprocess.Popen(cmd, stdout=self._log_fh,
                                     stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, env=self.child_env,
                                     start_new_session=True, cwd=self.tmp)
        self.client = Client(self.port)
        self._wait_online()
        return self

    def _wait_online(self, timeout=START_TIMEOUT):
        end = time.time() + timeout
        last = ""
        while time.time() < end:
            if self.proc.poll() is not None:
                raise RuntimeError("router exited early rc=%s log:\n%s"
                                   % (self.proc.returncode, self.log()[-3000:]))
            try:
                conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
                conn.request("GET", "/health")
                resp = conn.getresponse()
                resp.read()
                conn.close()
                return
            except Exception as e:
                last = "%s: %s" % (type(e).__name__, e)
                time.sleep(0.15)
        raise RuntimeError("router did not come online on port %d within %ss (%s)\nlog:\n%s"
                           % (self.port, timeout, last, self.log()[-3000:]))

    def stop_router(self, timeout=10):
        """Terminate ONLY the router subprocess (bounded); the mocks keep serving."""
        if self.proc is not None and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            end = time.time() + timeout
            while time.time() < end and self.proc.poll() is None:
                time.sleep(0.1)
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait(timeout=timeout)
        self.proc = None

    def restart(self):
        """Re-spawn the router on the same port (the pool state must be gone)."""
        self.stop_router()
        self.proc = subprocess.Popen(self.cmd, stdout=self._log_fh,
                                     stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, env=self.child_env,
                                     start_new_session=True, cwd=self.tmp)
        self.client = Client(self.port)
        self._wait_online()
        return self

    def stop(self, timeout=10):
        self.stop_router(timeout)
        try:
            self._log_fh.close()
        except Exception:
            pass
        for m in self.mocks:
            try:
                m.stop(timeout=5)
            except Exception:
                pass
        self.mocks = []

    def cleanup(self):
        self.stop()
        if self.tmp and os.path.isdir(self.tmp):
            shutil.rmtree(self.tmp, ignore_errors=True)

    # ---- introspection ----

    def log(self):
        try:
            with open(self.log_path, "rb") as fh:
                return fh.read().decode("utf-8", "replace")
        except OSError:
            return ""

    def grep_log(self, needle, timeout=5):
        """Bounded wait for a log line (the router logs asynchronously)."""
        end = time.time() + timeout
        while time.time() < end:
            text = self.log()
            if needle in text:
                return [l for l in text.splitlines() if needle in l]
            time.sleep(0.1)
        return []

    # ---- the state observation faces (no persistence left to read) ----

    def models_view(self):
        """GET /v1/models parsed — the authoritative per-account state face."""
        return self.client.get("/v1/models").json()

    def model_entry(self, model=MODEL):
        for entry in self.models_view()["data"]:
            if entry["id"] == model:
                return entry
        raise AssertionError("/v1/models has no entry for %r" % (model,))

    def acct_view(self, name, model=MODEL):
        """One account's annotation for `model`: {name, available, reason, ...}."""
        for a in self.model_entry(model)["accounts"]:
            if a["name"] == name:
                return a
        raise AssertionError("/v1/models entry %r has no account %r"
                             % (model, name))

    def pool_view(self):
        """GET /health parsed (unauthenticated probe face)."""
        return self.client.get("/health", token=None).json()

    def stray_files(self):
        """Files under the router cwd that this fixture did not write itself.

        The router persists nothing, so this must stay empty — it is the
        end-to-end proof that no ledger/state file reappears.
        """
        out = []
        for root, _dirs, files in os.walk(self.tmp):
            for f in files:
                p = os.path.join(root, f)
                if p not in self.seed:
                    out.append(os.path.relpath(p, self.tmp))
        return sorted(out)

    def persisted_faces(self):
        """Every byte the router could have persisted, as one string."""
        blobs = [self.log()]
        for root, _dirs, files in os.walk(self.tmp):
            for f in files:
                p = os.path.join(root, f)
                if p == self.log_path or p in self.seed:
                    continue
                try:
                    with open(p, "rb") as fh:
                        blobs.append(fh.read().decode("utf-8", "replace"))
                except OSError:
                    pass
        return "\n".join(blobs)


class RouterCase(unittest.TestCase):
    """Base class: one fixture per test, always torn down."""

    fixture_kwargs = {}

    def setUp(self):
        self.fx = Fixture(**self.fixture_kwargs).start()
        self.addCleanup(self.fx.cleanup)
        self.client = self.fx.client

    @property
    def mocks(self):
        return self.fx.mocks


# =========================== end-to-end matrix ===========================

class T01DefinitionOrder(RouterCase):
    def test_first_account_in_definition_order_serves(self):
        r = self.client.chat()
        self.assertEqual(200, r.status)
        a, b, c = self.mocks
        self.assertEqual(1, a.count)
        self.assertEqual(0, b.count)
        self.assertEqual(0, c.count)
        self.assertEqual([MODEL], a.models_seen)
        self.assertEqual([FAKE_KEYS["a"][-4:]], a.auth_tails)   # key substituted
        # the REQ ok line is the only per-request observation face left
        lines = self.fx.grep_log("REQ ok account=")
        self.assertEqual(1, len(lines))
        self.assertIn("account=alpha", lines[0])
        self.assertIn("model=%s" % MODEL, lines[0])
        self.assertIn("upstream_model=%s" % MODEL, lines[0])
        self.assertIn("usage_source=upstream", lines[0])
        self.assertIn("tokens=11/7", lines[0])
        self.assertIn("attempt=1", lines[0])
        self.assertIn("duration_ms=", lines[0])
        self.assertNotIn("credits", lines[0])
        health = self.fx.pool_view()
        self.assertEqual("ok", health["pool"])
        self.assertEqual(3, health["accounts_total"])
        self.assertEqual(3, health["accounts_available"])
        self.assertTrue(self.fx.acct_view("alpha")["available"])
        self.assertEqual("ok", self.fx.acct_view("alpha")["reason"])
        self.assertEqual([], self.fx.stray_files())   # nothing is persisted


class T02ExhaustedSwitch(RouterCase):
    # the quota tier is the LONG blacklist; pin it far above the failure tier so
    # the two tiers are distinguishable from the outside
    fixture_kwargs = {"defaults": {"blacklist_exhausted": "3600",
                                   "blacklist_failure": "5"}}

    def test_quota_error_switches_account_in_the_same_request(self):
        a, b, _c = self.mocks
        self.assertEqual(200, self.client.chat().status)
        a.script(mock.exhausted_spec(429))
        r = self.client.chat()
        self.assertEqual(200, r.status, r.text())
        self.assertEqual(2, a.count)
        self.assertEqual(1, b.count)
        self.assertIn("pong", r.text())
        view = self.fx.acct_view("alpha")
        self.assertFalse(view["available"])
        self.assertEqual("exhausted", view["reason"])
        self.assertIsNotNone(view["until"])
        health = self.fx.pool_view()
        self.assertEqual("degraded", health["pool"])
        self.assertEqual(2, health["accounts_available"])
        exh = self.fx.grep_log("ACCOUNT_EXHAUSTED")[0]
        self.assertIn("account=alpha", exh)
        self.assertIn("blacklist_s=3600", exh)     # the quota tier, not the 5s one
        self.assertIn("reason=exhausted", exh)
        ok = self.fx.grep_log("REQ ok account=")[-1]
        self.assertIn("account=beta", ok)
        self.assertIn("attempt=2", ok)             # switched inside one request
        self.assertEqual([], self.fx.stray_files())


class T03Recovery(RouterCase):
    fixture_kwargs = {"defaults": {"blacklist_exhausted": "1"}}

    def test_blacklist_expires_and_the_account_is_probed_again(self):
        a, _b, _c = self.mocks
        a.script(mock.exhausted_spec(402, "You exceeded your current quota"))
        self.assertEqual(200, self.client.chat().status)      # served by beta
        self.assertEqual("exhausted", self.fx.acct_view("alpha")["reason"])
        self.assertEqual(1, self.mocks[1].count)
        self.assertEqual(1, a.count)                          # not retried meanwhile
        time.sleep(1.6)                     # bounded: blacklist_exhausted is 1s
        self.mocks[1].reset()
        r = self.client.chat()
        self.assertEqual(200, r.status)
        self.assertEqual(2, a.count)                          # alpha probed again
        view = self.fx.acct_view("alpha")
        self.assertTrue(view["available"])
        self.assertEqual("ok", view["reason"])
        self.assertNotIn("until", view)      # a recovered account carries no deadline
        rec = self.fx.grep_log("ACCOUNT_RECOVERED")[0]
        self.assertIn("account=alpha", rec)
        self.assertIn("reason was exhausted", rec)
        self.assertIn("next request is a probe", rec)


class T04ThrottleSwitch(RouterCase):
    """A rate limit is the account's fault: short blacklist + immediate switch."""

    fixture_kwargs = {"defaults": {"blacklist_failure": "1",
                                   "blacklist_exhausted": "3600"}}

    def test_rate_limit_switches_at_once_with_no_same_account_retry(self):
        a, b, _c = self.mocks
        a.script(mock.throttle_spec(429))
        r = self.client.chat()
        self.assertEqual(200, r.status, r.text())
        self.assertEqual(1, a.count)          # NO same-account backoff retry
        self.assertEqual(1, b.count)          # switched inside the same request
        view = self.fx.acct_view("alpha")
        self.assertFalse(view["available"])
        self.assertEqual("throttled", view["reason"])
        line = self.fx.grep_log("THROTTLED_SWITCH")[0]
        self.assertIn("account=alpha", line)
        self.assertIn("blacklisted 1s", line)          # the short tier
        self.assertIn("no same-account retry", line)
        self.assertIn("attempt=1", line)
        self.assertIn("duration_ms=", line)
        bl = self.fx.grep_log("ACCOUNT_BLACKLISTED")[0]
        self.assertIn("reason=throttled", bl)
        self.assertIn("blacklist_s=1", bl)
        self.assertFalse(self.fx.grep_log("ACCOUNT_EXHAUSTED", timeout=0.3))

    def test_throttled_account_is_skipped_then_probed_after_the_deadline(self):
        a, b, _c = self.mocks
        a.script(mock.throttle_spec(429))
        self.assertEqual(200, self.client.chat().status)
        self.assertEqual(1, a.count)
        b.reset()
        self.assertEqual(200, self.client.chat().status)   # alpha still skipped
        self.assertEqual(1, a.count)                       # not even tried
        self.assertEqual(1, b.count)
        time.sleep(1.2)                    # bounded: blacklist_failure is 1s
        self.assertEqual(200, self.client.chat().status)
        self.assertEqual(2, a.count)                       # probed again
        self.assertTrue(self.fx.acct_view("alpha")["available"])


class T06NonStreamRetry(RouterCase):
    def test_server_error_switches_account(self):
        a, b, _c = self.mocks
        a.script(mock.error_spec(503, "upstream busy"))
        r = self.client.chat()
        self.assertEqual(200, r.status, r.text())
        self.assertEqual(1, a.count)
        self.assertEqual(1, b.count)
        self.assertIn("account=beta", self.fx.grep_log("REQ ok account=")[-1])
        err = self.fx.grep_log("UPSTREAM_ERROR")[0]
        self.assertIn("account=alpha", err)
        self.assertIn("status=503", err)
        self.assertIn("-> next account", err)
        # a 5xx is the transient tier: the view folds it into `throttled`
        self.assertEqual("throttled", self.fx.acct_view("alpha")["reason"])
        self.assertFalse(self.fx.acct_view("alpha")["available"])

    def test_client_error_is_returned_unchanged_without_switching(self):
        a, b, _c = self.mocks
        a.script(mock.error_spec(400, "messages.0.content is required"))
        r = self.client.chat()
        self.assertEqual(400, r.status)
        self.assertIn("messages.0.content is required", r.text())
        self.assertEqual(1, a.count)
        self.assertEqual(0, b.count)          # no pointless retry elsewhere
        line = self.fx.grep_log("CLIENT_ERROR")[0]
        self.assertIn("account=alpha", line)
        self.assertIn("status=400", line)
        self.assertIn("no account switch, no blacklist", line)
        view = self.fx.acct_view("alpha")    # a 4xx never blacklists
        self.assertTrue(view["available"])
        self.assertEqual("ok", view["reason"])
        self.assertFalse(self.fx.grep_log("ACCOUNT_BLACKLISTED", timeout=0.3))

    def test_unreachable_upstream_switches_account(self):
        _a, b, _c = self.mocks
        dead_port = free_port()                   # nothing listens there
        # rewrite alpha's base_url by hot-reloading the accounts file
        with open(self.fx.accounts_path) as fh:
            text = fh.read()
        text = text.replace(self.mocks[0].base_url,
                            "http://127.0.0.1:%d/v1" % dead_port)
        with open(self.fx.accounts_path, "w") as fh:
            fh.write(text)
        time.sleep(0.2)
        r = self.client.chat()
        self.assertEqual(200, r.status, r.text())
        self.assertEqual(1, b.count)
        line = self.fx.grep_log("UPSTREAM_UNREACHABLE")[0]
        self.assertIn("account=alpha", line)
        self.assertIn("-> next account", line)
        self.assertEqual("throttled", self.fx.acct_view("alpha")["reason"])
        self.assertTrue(self.fx.grep_log("CONFIG_RELOADED"))


class T07StreamInterrupt(RouterCase):
    def test_error_event_in_stream_is_relayed_and_no_retry_happens(self):
        a, b, _c = self.mocks
        a.script(mock.spec(chunks=2, break_after=2, abrupt=False))
        r = self.client.stream_chat()
        self.assertEqual(200, r.status)
        text = r.text()
        self.assertEqual(2, text.count('"content": "part'))
        self.assertIn("quota exceeded", text.lower())
        self.assertIn("[DONE]", text)
        self.assertEqual(1, a.count)
        self.assertEqual(0, b.count)          # bytes already reached the client
        # the interrupt carried quota semantics => the account IS blacklisted
        self.assertEqual("exhausted", self.fx.acct_view("alpha")["reason"])
        line = self.fx.grep_log("STREAM_INTERRUPTED")[0]
        self.assertIn("account=alpha", line)
        self.assertIn("quota_semantics=True", line)
        self.assertIn("bytes_sent=", line)
        self.assertTrue(self.fx.grep_log("ACCOUNT_EXHAUSTED"))
        self.assertFalse(self.fx.grep_log("REQ ok account=", timeout=0.3))

    def test_abrupt_close_appends_router_error_event_and_done(self):
        a, b, _c = self.mocks
        a.script(mock.spec(chunks=3, break_after=2, abrupt=True))
        r = self.client.stream_chat()
        self.assertEqual(200, r.status)
        text = r.text()
        self.assertIn("upstream_stream_interrupted", text)
        self.assertTrue(text.rstrip().endswith("data: [DONE]"), text[-200:])
        self.assertEqual(1, a.count)
        self.assertEqual(0, b.count)


class T08StreamRetryBeforeFirstByte(RouterCase):
    def test_exhaustion_before_any_byte_switches_account(self):
        a, b, _c = self.mocks
        a.script(mock.exhausted_spec(429))
        r = self.client.stream_chat()
        self.assertEqual(200, r.status)
        text = r.text()
        self.assertIn("[DONE]", text)
        self.assertIn("part0", text)                        # mock stream chunk
        self.assertEqual(1, a.count)
        self.assertEqual(1, b.count)
        line = self.fx.grep_log("REQ ok account=")[-1]
        self.assertIn("account=beta", line)
        self.assertIn("stream=true", line)
        self.assertIn("bytes=", line)
        self.assertEqual("exhausted", self.fx.acct_view("alpha")["reason"])

    def test_empty_stream_switches_account(self):
        a, b, _c = self.mocks
        a.script(mock.spec(empty_stream=True))
        r = self.client.stream_chat()
        self.assertEqual(200, r.status)
        self.assertIn("[DONE]", r.text())
        self.assertEqual(1, a.count)
        self.assertEqual(1, b.count)


class T10ModelMapping(RouterCase):
    def test_upstream_model_names_are_mapped_per_account(self):
        a, b, c = self.mocks
        a.script(mock.error_spec(503, "down"))
        b.script(mock.error_spec(503, "down"))
        r = self.client.chat(model=MODEL)
        self.assertEqual(200, r.status, r.text())
        self.assertEqual(["qwen/" + MODEL], c.models_seen)
        self.assertEqual([MODEL], a.models_seen)
        self.assertEqual([MODEL], b.models_seen)

    def test_model_without_mapping_is_not_offered_to_that_account(self):
        _a, _b, c = self.mocks
        r = self.client.chat(model="glm-5.2")
        self.assertEqual(200, r.status, r.text())
        self.assertEqual(0, c.count)                     # gamma has no glm-5.2
        self.assertEqual(["glm-5.2"], self.mocks[0].models_seen)

    def test_single_account_model_routes_to_it(self):
        _a, b, c = self.mocks
        r = self.client.chat(model="claude-fable-5.1")
        self.assertEqual(200, r.status, r.text())
        self.assertEqual(1, c.count)
        self.assertEqual(0, b.count)
        self.assertEqual(0, self.mocks[0].count)
        self.assertEqual(["anthropic/claude-fable-5.1"], c.models_seen)

    def test_unknown_model_reports_no_candidates(self):
        r = self.client.chat(model="no-such-model")
        self.assertEqual(503, r.status)
        body = r.json()
        self.assertEqual("no_account_for_model", body["error"]["type"])
        self.assertEqual(["no_mapping"] * 3,
                         [x["reason"] for x in body["error"]["accounts"]])


class T12PoolExhausted(RouterCase):
    def test_whole_pool_exhausted_returns_503_and_logs_an_alert(self):
        for m in self.mocks:
            m.script(mock.exhausted_spec(429))
        r = self.client.chat(model="glm-5.2")     # only alpha maps it
        self.assertEqual(503, r.status)
        body = r.json()
        self.assertEqual("all_accounts_exhausted", body["error"]["type"])
        self.assertIn("pay-as-you-go", body["error"]["message"])
        # second request: no candidate left at all
        r2 = self.client.chat(model=MODEL)
        self.assertEqual(503, r2.status)
        self.assertEqual("all_accounts_exhausted", r2.json()["error"]["type"])
        names = [x["name"] for x in r2.json()["error"]["accounts"]]
        self.assertEqual(["alpha", "beta", "gamma"], names)
        for entry in r2.json()["error"]["accounts"]:
            self.assertEqual("exhausted", entry["reason"])
            self.assertIsNotNone(entry["until"])
        alert = self.fx.grep_log("POOL_EXHAUSTED")[-1]   # the second (qwen) request
        self.assertIn("model=%s" % MODEL, alert)
        self.assertIn("pool=all_exhausted", alert)
        self.assertIn("pay-as-you-go", alert)
        self.assertNotIn("credits", alert)
        health = self.fx.pool_view()
        self.assertEqual("all_exhausted", health["pool"])
        self.assertEqual(0, health["accounts_available"])
        self.assertEqual(3, health["accounts_total"])
        # every account shows the long tier's reason through /v1/models
        for name in ("alpha", "beta", "gamma"):
            self.assertEqual("exhausted", self.fx.acct_view(name)["reason"])
        self.assertEqual([], self.fx.stray_files())


class T13ModelsEndpoint(RouterCase):
    def test_models_is_aggregated_locally_and_annotated(self):
        r = self.client.get("/v1/models")
        self.assertEqual(200, r.status)
        body = r.json()
        self.assertEqual("list", body["object"])
        ids = [m["id"] for m in body["data"]]
        self.assertEqual(["glm-5.2", MODEL, "claude-fable-5.1"], ids)
        by_id = dict((m["id"], m) for m in body["data"])
        qmax = by_id[MODEL]
        self.assertEqual(["alpha", "beta", "gamma"],
                         [a["name"] for a in qmax["accounts"]])
        self.assertEqual({"alpha": MODEL, "beta": MODEL, "gamma": "qwen/" + MODEL},
                         dict((a["name"], a["upstream_model"]) for a in qmax["accounts"]))
        self.assertTrue(all(a["available"] for a in qmax["accounts"]))
        glm = by_id["glm-5.2"]
        reasons = dict((a["name"], a["reason"]) for a in glm["accounts"])
        self.assertEqual("no_mapping", reasons["gamma"])
        self.assertFalse([a for a in glm["accounts"] if a["name"] == "gamma"][0]["available"])
        self.assertEqual(0, sum(m.count for m in self.mocks))   # never forwarded

    def test_models_marks_exhausted_accounts(self):
        self.mocks[0].script(mock.exhausted_spec(429))
        self.assertEqual(200, self.client.chat().status)
        body = self.client.get("/v1/models").json()
        qmax = dict((m["id"], m) for m in body["data"])[MODEL]
        alpha = [a for a in qmax["accounts"] if a["name"] == "alpha"][0]
        self.assertFalse(alpha["available"])
        self.assertEqual("exhausted", alpha["reason"])
        self.assertIn("until", alpha)


class T14Health(RouterCase):
    def test_health_is_open_and_carries_no_sensitive_detail(self):
        r = self.client.get("/health", token=None)
        self.assertEqual(200, r.status)
        body = r.json()
        self.assertEqual("ok", body["status"])
        self.assertEqual("ok", body["pool"])
        self.assertEqual(3, body["accounts_total"])
        self.assertEqual(3, body["accounts_available"])
        self.assertIn("uptime_s", body)
        self.assertNotIn("version", body)     # the version face is the deployment's
        self.assertEqual("llm-router", r.headers.get("Server"))
        text = r.text()
        for value in list(FAKE_KEYS.values()) + [FAKE_TOKEN]:
            self.assertNotIn(value, text)
        for name in ("alpha", "beta", "gamma"):
            self.assertNotIn(name, text)          # no account details
        for needle in ("credits", "quota", "tokens", "usage", "key_hint",
                       "authorization", "blacklist", "until", "reason"):
            self.assertNotIn(needle, text.lower())
        self.assertEqual(sorted(body.keys()),
                         ["accounts_available", "accounts_total", "pool",
                          "service", "status", "time", "uptime_s"])

    def test_health_stays_200_when_degraded(self):
        self.mocks[0].script(mock.exhausted_spec(429))
        self.assertEqual(200, self.client.chat().status)
        r = self.client.get("/health", token=None)
        self.assertEqual(200, r.status)
        body = r.json()
        self.assertEqual("degraded", body["pool"])
        self.assertEqual(2, body["accounts_available"])


class T15SecretsFaces(RouterCase):
    def test_no_plaintext_credential_in_any_persisted_or_returned_face(self):
        a, _b, _c = self.mocks
        # one request per failure mode, each with its own scripted alpha turn
        a.script(mock.exhausted_spec(429))
        self.assertEqual(200, self.client.chat().status)        # switch to beta
        time.sleep(2.2)          # the fixture's exhausted_cooldown is 2s
        a.script(mock.error_spec(400, "bad params"))
        self.assertEqual(400, self.client.chat().status)        # passthrough
        a.script(mock.error_spec(503, "boom"))
        self.assertEqual(200, self.client.chat().status)        # next account
        self.assertEqual(200, self.client.chat().status)        # plain ok
        self.assertEqual(200, self.client.stream_chat().status)  # streamed ok
        time.sleep(2.2)          # the fixture's blacklist_failure is 2s
        a.script(mock.error_spec(401, "invalid api-key"))
        self.assertEqual(200, self.client.chat().status)        # 401 => switch
        self.client.get("/v1/models")
        self.client.get("/health", token=None)
        self.client.get("/nope")
        self.client.get("/v1/models", token="wrong-token-value")
        faces = self.fx.persisted_faces()
        self.assertGreater(len(faces), 100)
        for value in list(FAKE_KEYS.values()) + [FAKE_TOKEN]:
            self.assertNotIn(value, faces)
        self.assertIn(FAKE_MASKS["a"], faces)     # the masked hint is all that is kept
        self.assertNotIn('"hi"', faces)           # no request content
        self.assertNotIn("pong", faces)           # no response content
        for value in list(FAKE_KEYS.values()) + [FAKE_TOKEN]:
            for path in ("/v1/models", "/health"):
                body = self.client.get(path, token=None).text()
                self.assertNotIn(value, body)

    def test_the_router_persists_nothing_beyond_its_log(self):
        self.assertEqual(200, self.client.chat().status)
        self.assertEqual(200, self.client.stream_chat().status)
        self.client.get("/v1/models")
        self.client.get("/health", token=None)
        self.assertEqual([], self.fx.stray_files())
        faces = self.fx.persisted_faces()
        self.assertEqual(faces, self.fx.log())   # log is the only persisted face
        self.assertIn("REQ ok", faces)


class T16HotReload(RouterCase):
    def test_new_account_is_picked_up_without_a_restart(self):
        extra = mock.MockUpstream("delta")
        extra.start()
        self.addCleanup(extra.stop)
        with open(self.fx.accounts_path) as fh:
            text = fh.read()
        # delta is appended LAST (definition order => it would be tried last for
        # any model it shares), so pickup is proven with a model name that only
        # delta maps: no order privilege is involved.
        text += accounts_block("delta", extra.base_url, KEY_ENV["a"],
                               {"delta-only": "delta-upstream"}) + "\n"
        with open(self.fx.accounts_path, "w") as fh:
            fh.write(text)
        self.assertEqual(200, self.client.get("/health", token=None).status)  # forces the reload
        self.assertTrue(self.fx.grep_log("CONFIG_RELOADED"))
        self.assertEqual(4, self.fx.pool_view()["accounts_total"])
        r = self.client.chat(model="delta-only")
        self.assertEqual(200, r.status, r.text())
        self.assertEqual(1, extra.count)          # the new account serves it
        self.assertEqual(["delta-upstream"], extra.models_seen)
        self.assertEqual(0, self.mocks[0].count)  # the shared model is untouched
        # the view annotates EVERY account of the pool, so delta shows up as
        # `no_mapping` for the models it does not carry
        names = [a["name"] for a in self.fx.model_entry(MODEL)["accounts"]]
        self.assertEqual(["alpha", "beta", "gamma", "delta"], names)
        self.assertFalse(self.fx.acct_view("delta", MODEL)["available"])
        self.assertEqual("no_mapping",
                         self.fx.acct_view("delta", MODEL)["reason"])
        self.assertEqual(["delta-only"],
                         [x["id"] for x in self.fx.models_view()["data"]
                          if x["id"] == "delta-only"])

    def test_removed_account_disappears(self):
        with open(self.fx.accounts_path) as fh:
            lines = fh.read().splitlines()
        out, skip = [], False
        for line in lines:
            if line.strip().startswith("- name: gamma"):
                skip = True
                continue
            if skip and line.startswith("  - name:"):
                skip = False
            if not skip:
                out.append(line)
        with open(self.fx.accounts_path, "w") as fh:
            fh.write("\n".join(out) + "\n")
        self.assertEqual(200, self.client.get("/health", token=None).status)  # forces the reload
        self.assertTrue(self.fx.grep_log("CONFIG_RELOADED"))
        self.assertEqual(2, self.fx.pool_view()["accounts_total"])
        self.assertEqual(2, self.fx.pool_view()["accounts_available"])
        self.assertNotIn("gamma", self.client.get("/v1/models").text())
        self.assertEqual(["alpha", "beta"],
                         [a["name"] for a in self.fx.model_entry(MODEL)["accounts"]])


class T18Auth(RouterCase):
    def test_every_authenticated_endpoint_rejects_a_missing_token(self):
        for path, method in (("/v1/models", "GET"), ("/v1/embeddings", "POST"),
                             (CHAT, "POST")):
            if method == "GET":
                r = self.client.get(path, token=None)
            elif path == CHAT:
                r = self.client.chat(token=None)
            else:
                r = self.client.post(path, {"model": MODEL}, token=None)
            self.assertEqual(401, r.status, path)
            self.assertIn("unauthorized", r.text())
            self.assertNotIn(FAKE_TOKEN, r.text())
            self.assertNotIn(FAKE_TOKEN_MASK, r.text())
            self.assertEqual("Bearer", r.headers.get("WWW-Authenticate"))

    def test_wrong_token_is_rejected_without_echoing(self):
        for bad in ("0" * 64, "x", FAKE_TOKEN[:-1] + "0", ""):
            r = self.client.get("/nope", token=bad)
            self.assertEqual(401, r.status)
            self.assertNotIn(FAKE_TOKEN, r.text())
            if bad:
                self.assertNotIn(bad, r.text())
            r2 = self.client.get("/v1/models", token=bad)
            self.assertEqual(401, r2.status)
            r3 = self.client.chat(token=bad)
            self.assertEqual(401, r3.status)
        self.assertEqual(0, sum(m.count for m in self.mocks))   # nothing forwarded

    def test_non_bearer_scheme_is_rejected(self):
        r = self.client.get("/v1/models", token=None,
                            headers={"Authorization": "Basic " + FAKE_TOKEN})
        self.assertEqual(401, r.status)

    def test_correct_token_works_on_every_authenticated_endpoint(self):
        self.assertEqual(200, self.client.get("/v1/models").status)
        self.assertEqual(200, self.client.chat().status)

    def test_unknown_path_still_requires_a_token(self):
        """The auth gate sits BEFORE routing: no token => 401, never 404."""
        self.assertEqual(401, self.client.get("/nope", token=None).status)
        self.assertEqual(404, self.client.get("/nope").status)

    def test_a_retired_endpoint_is_unknown_but_still_gated(self):
        """/usage is gone: 401 without a token, 404 with one (never 200).

        The single home of the retired-`/usage` face: assert it here only.
        """
        self.assertEqual(401, self.client.get("/usage", token=None).status)
        r = self.client.get("/usage")
        self.assertEqual(404, r.status)
        self.assertIn("error", r.json())
        self.assertEqual(404, self.client.get("/usage?format=text").status)
        self.assertEqual(404, self.client.get("/usage?days=7&account=alpha").status)
        self.assertEqual(0, sum(m.count for m in self.mocks))   # never forwarded

    def test_rejected_post_does_not_corrupt_the_keep_alive_connection(self):
        """An unread request body must not be parsed as the next request line."""
        conn = http.client.HTTPConnection("127.0.0.1", self.fx.port, timeout=10)
        try:
            body = json.dumps({"model": MODEL, "messages": []}).encode("utf-8")
            for i in range(3):
                conn.request("POST", CHAT, body=body,
                             headers={"Content-Type": "application/json"})
                r = conn.getresponse()
                r.read()
                self.assertEqual(401, r.status, "request #%d on the reused "
                                                "connection" % i)
            # and an authorized request on a fresh connection still works
            self.assertEqual(200, self.client.chat().status)
        finally:
            conn.close()

    def test_auth_rejections_are_logged_without_the_credential(self):
        self.client.get("/v1/models", token="wrong-value-1234567890")
        lines = self.fx.grep_log("AUTH_REJECTED")
        self.assertTrue(lines)
        self.assertNotIn("wrong-value-1234567890", "\n".join(lines))


class T19StreamUsageInjection(RouterCase):
    def test_stream_options_is_injected_and_tokens_are_booked(self):
        a, _b, _c = self.mocks
        a.set_default(usage={"prompt_tokens": 21, "completion_tokens": 9,
                             "total_tokens": 30})
        r = self.client.stream_chat()
        self.assertEqual(200, r.status)
        self.assertIn("[DONE]", r.text())
        self.assertEqual([True], a.stream_options_seen)        # injected
        # the REQ ok line is the only place the injected usage shows up
        lines = self.fx.grep_log("REQ ok account=")
        self.assertEqual(1, len(lines))
        self.assertIn("stream=true", lines[0])
        self.assertIn("tokens=21/9", lines[0])
        self.assertIn("usage_source=upstream", lines[0])
        self.assertIn("account=alpha", lines[0])
        self.assertNotIn("credits", lines[0])
        self.assertFalse(self.fx.grep_log("STREAM_OPTIONS_REJECTED", timeout=0.3))

    def test_client_supplied_stream_options_is_not_overwritten(self):
        a, _b, _c = self.mocks
        r = self.client.stream_chat(
            extra={"stream_options": {"include_usage": True, "foo": "bar"}})
        self.assertEqual(200, r.status)
        seen = a.last()["stream_options_raw"]
        self.assertEqual("bar", seen.get("foo"))               # passed through
        self.assertTrue(seen.get("include_usage"))

    def test_injection_can_be_disabled_by_config(self):
        fx = Fixture(name="noinject", defaults={"inject": "false"}).start()
        self.addCleanup(fx.cleanup)
        r = fx.client.stream_chat()
        self.assertEqual(200, r.status)
        self.assertEqual([False], fx.mocks[0].stream_options_seen)
        line = fx.grep_log("REQ ok account=")[-1]
        self.assertIn("usage_source=missing", line)
        self.assertIn("tokens=0/0", line)


class T20StreamOptionsRejected(RouterCase):
    def test_400_on_stream_options_drops_the_field_and_retries_once(self):
        a, b, _c = self.mocks
        a.script(mock.spec(reject_stream_options=True), mock.spec(chunks=2))
        r = self.client.stream_chat()
        self.assertEqual(200, r.status, r.text())
        self.assertEqual(2, a.count)                            # same account
        self.assertEqual(0, b.count)                            # no switch
        self.assertEqual([True, False], a.stream_options_seen)  # field dropped
        rejected = self.fx.grep_log("STREAM_OPTIONS_REJECTED")[0]
        self.assertIn("account=alpha", rejected)
        self.assertIn("usage_source=missing", rejected)
        ok = self.fx.grep_log("REQ ok account=")[-1]
        self.assertIn("account=alpha", ok)       # the retry stayed on alpha
        self.assertIn("usage_source=missing", ok)
        self.assertIn("tokens=0/0", ok)

    def test_persistent_400_is_returned_to_the_client(self):
        a, b, _c = self.mocks
        a.script(mock.spec(reject_stream_options=True),
                 mock.error_spec(400, "messages.0.role is required"))
        r = self.client.stream_chat()
        self.assertEqual(400, r.status)
        self.assertIn("messages.0.role is required", r.text())
        self.assertEqual(2, a.count)
        self.assertEqual(0, b.count)        # a 400 never switches accounts
        self.assertTrue(self.fx.grep_log("CLIENT_ERROR"))
        self.assertTrue(self.fx.acct_view("alpha")["available"])   # no blacklist
        self.assertFalse(self.fx.grep_log("REQ ok account=", timeout=0.3))


class T21GenericPassthrough(RouterCase):
    def test_other_v1_paths_are_proxied_with_the_same_account_logic(self):
        a, b, _c = self.mocks
        a.script(mock.error_spec(503, "down"))
        r = self.client.post("/v1/embeddings",
                             {"model": MODEL, "input": "hi"})
        self.assertEqual(200, r.status, r.text())
        self.assertEqual(1, a.count)
        self.assertEqual(1, b.count)
        self.assertEqual("/v1/embeddings", b.last()["path"])

    def test_query_string_is_preserved(self):
        _a, _b, _c = self.mocks
        r = self.client.post(CHAT + "?beta=true",
                             {"model": MODEL, "messages": [], "max_tokens": 1})
        self.assertEqual(200, r.status)
        self.assertIn("beta=true", self.mocks[0].last()["path"])


# ============================ unit-level tests ============================

class T05ClassifyTable(unittest.TestCase):
    def test_quota_wording_wins_over_the_status_code(self):
        cases = [
            (429, '{"error":{"message":"Allocated quota exceeded"}}', classify.EXHAUSTED),
            (402, '{"error":{"message":"You exceeded your current quota"}}', classify.EXHAUSTED),
            (403, '{"error":{"code":"insufficient_quota"}}', classify.EXHAUSTED),
            (400, '{"error":{"message":"The free tier of the model has been exhausted"}}', classify.EXHAUSTED),
            (429, '{"error":{"message":"Throttling: quota exceeded for credits"}}', classify.EXHAUSTED),
            (403, '{"error":{"message":"Access denied: 额度不足"}}', classify.EXHAUSTED),
            (429, '{"error":{"message":"Arrearage: access denied"}}', classify.EXHAUSTED),
        ]
        for status, body, want in cases:
            got, why = classify.classify(status, body)
            self.assertEqual(want, got, "status=%s body=%s -> %s (%s)"
                             % (status, body, got, why))

    def test_pure_rate_limits_are_throttled_never_exhausted(self):
        cases = [
            (429, '{"error":{"message":"Requests rate limit exceeded, please try again later"}}'),
            (429, '{"error":{"message":"Too many requests"}}'),
            (429, '{"error":{"message":"You exceeded 200000 requests per minute"}}'),
            (429, '{"error":{"message":"tokens per minute limit reached"}}'),
            (429, ''),
            (403, '{"error":{"message":"throttled by concurrency limit"}}'),
        ]
        for status, body in cases:
            got, why = classify.classify(status, body)
            self.assertEqual(classify.THROTTLED, got,
                             "status=%s body=%s -> %s (%s)" % (status, body, got, why))

    def test_aliyun_throttling_codes_are_disambiguated(self):
        # aliyun answers BOTH rate limits and spent quota with a `Throttling.*`
        # code, so the code alone must not decide the category
        rate = ('{"code":"Throttling.RateQuota","message":"Requests rate limit '
                'exceeded, please try again later."}')
        alloc = ('{"code":"Throttling.AllocationQuota","message":"Allocated '
                 'quota exceeded, please increase your quota limit."}')
        alloc_vague = ('{"code":"Throttling.AllocationQuota","message":"You are '
                       'exceeding your allocated credits limit."}')
        rate_vague = '{"code":"Throttling.RateQuota","message":"Request throttled"}'
        self.assertEqual(classify.THROTTLED, classify.classify(429, rate)[0])
        self.assertEqual(classify.EXHAUSTED, classify.classify(429, alloc)[0])
        self.assertEqual(classify.EXHAUSTED, classify.classify(429, alloc_vague)[0])
        self.assertEqual(classify.THROTTLED, classify.classify(429, rate_vague)[0])
        # an exhaustion word flips a rate-limit-looking body back to exhausted
        self.assertEqual(classify.EXHAUSTED, classify.classify(
            429, '{"message":"Requests rate limit exceeded: your credits are '
                 'exhausted"}')[0])

    def test_other_classes(self):
        self.assertEqual(classify.SERVER_ERROR, classify.classify(500, "boom")[0])
        self.assertEqual(classify.SERVER_ERROR, classify.classify(502, "bad gw")[0])
        self.assertEqual(classify.CLIENT_ERROR, classify.classify(400, "bad")[0])
        self.assertEqual(classify.CLIENT_ERROR, classify.classify(401, "no key")[0])
        self.assertEqual(classify.CLIENT_ERROR, classify.classify(404, "nope")[0])
        self.assertEqual(classify.CLIENT_ERROR, classify.classify(422, "unprocessable")[0])
        self.assertEqual(classify.CLIENT_ERROR, classify.classify(403, "forbidden")[0])
        self.assertEqual(classify.UNKNOWN, classify.classify(418, "teapot")[0])
        self.assertEqual(classify.SERVER_ERROR,
                         classify.classify_exception(socket.timeout("timed out"))[0])
        self.assertEqual(classify.SERVER_ERROR,
                         classify.classify_exception(ConnectionResetError("reset"))[0])

    def test_quota_semantics_helper(self):
        self.assertTrue(classify.has_quota_semantics("Allocated quota exceeded"))
        self.assertTrue(classify.has_quota_semantics("余额不足"))
        self.assertFalse(classify.has_quota_semantics("Too many requests"))
        self.assertFalse(classify.has_quota_semantics(""))
        self.assertFalse(classify.has_quota_semantics(None))


class T17ConfigValidation(unittest.TestCase):
    def setUp(self):
        os.makedirs(TMP_ROOT, exist_ok=True)
        self.tmp = tempfile.mkdtemp(prefix="cfg-", dir=TMP_ROOT)
        self.addCleanup(shutil.rmtree, self.tmp, True)
        # parsing is credential-free: these are NAMES, and no value is needed
        # anywhere in this test class (nothing is decrypted or even set)
        self.key_env = KEY_ENV_NAME
        self.token_env = TOKEN_ENV

    def _spec(self, **kw):
        name = kw.get("name", "one")
        return accounts_block(name,
                              kw.get("base_url", "https://example.invalid/v1"),
                              kw.get("key_env", self.key_env),
                              kw.get("models", {"m": "m"}),
                              extra=kw.get("extra", ""))

    def _load(self, specs, defaults=None):
        path = write_accounts(os.path.join(self.tmp, "a.yml"), specs,
                              self.token_env, defaults)
        return config_mod.load(path)

    def test_valid_config_loads(self):
        cfg = self._load([self._spec()])
        self.assertEqual(1, len(cfg.accounts))
        self.assertEqual("one", cfg.accounts[0].name)
        self.assertEqual(1, cfg.accounts[0].order)   # order = writing position
        self.assertEqual(["one"], [a.name for a in cfg.ordered()])
        self.assertTrue(cfg.auth.is_exempt("/health"))
        self.assertFalse(cfg.auth.is_exempt("/v1/models"))
        self.assertEqual(2.0, cfg.blacklist_exhausted)   # write_accounts' fast tiers
        self.assertEqual(2.0, cfg.blacklist_failure)
        self.assertTrue(cfg.inject_stream_options)
        self.assertEqual(["m"], cfg.model_names())

    def test_order_follows_the_writing_order(self):
        cfg = self._load([self._spec(name="a1"), self._spec(name="a2"),
                          self._spec(name="a3")])
        self.assertEqual([1, 2, 3], [a.order for a in cfg.accounts])
        self.assertEqual(["a1", "a2", "a3"], [a.name for a in cfg.ordered()])

    def test_duplicate_names_rejected(self):
        with self.assertRaises(config_mod.ConfigError):
            self._load([self._spec(), self._spec()])

    def test_retired_priority_key_is_rejected(self):
        with self.assertRaises(config_mod.ConfigError) as cm:
            self._load([self._spec(extra="priority: 10")])
        msg = str(cm.exception)
        self.assertIn("retired key", msg)
        self.assertIn("priority", msg)

    def test_retired_quota_key_is_rejected(self):
        with self.assertRaises(config_mod.ConfigError) as cm:
            self._load([self._spec(extra="quota: {credits: 100}")])
        self.assertIn("quota", str(cm.exception))

    def test_retired_credits_weights_key_is_rejected(self):
        with self.assertRaises(config_mod.ConfigError) as cm:
            self._load([self._spec(extra="credits_weights: {}")])
        self.assertIn("credits_weights", str(cm.exception))

    def test_every_retired_key_present_is_listed_in_the_error(self):
        extra = "\n    ".join("%s: 1" % k
                              for k in config_mod.REJECTED_ACCOUNT_KEYS)
        with self.assertRaises(config_mod.ConfigError) as cm:
            self._load([self._spec(extra=extra)])
        for key in config_mod.REJECTED_ACCOUNT_KEYS:
            self.assertIn(key, str(cm.exception))
        self.assertIn("writing order", str(cm.exception))

    def test_bad_blacklist_values_rejected(self):
        for defaults in ({"blacklist_exhausted": "0"},
                         {"blacklist_exhausted": "abc"},
                         {"blacklist_failure": "-1"},
                         {"blacklist_failure": "soon"}):
            with self.assertRaises(config_mod.ConfigError):
                self._load([self._spec()], defaults=defaults)

    def test_a_credential_reference_must_be_an_env_var_name(self):
        """The retired {env_file, var} mapping and anything that is not an
        identifier (a pasted literal key) are both refused at load."""
        for bad in ("{env_file: env/k.env, var: K}", FAKE_KEYS["a"], "", "K-V",
                    "9K", "K V"):
            path = os.path.join(self.tmp, "k.yml")
            with open(path, "w") as fh:
                fh.write("auth:\n  token: %s\n  exempt_paths: [/health]\n"
                         "accounts:\n  - name: one\n"
                         "    base_url: https://example.invalid/v1\n"
                         "    key: %s\n    models:\n      m: m\n"
                         % (self.token_env, bad))
            with self.assertRaises(config_mod.ConfigError, msg=repr(bad)):
                config_mod.load(path)

    def test_an_unset_credential_is_not_a_load_error(self):
        """Parsing never needs a secret in the environment: a name that is not
        set still loads (the launcher's require_env gate and secrets.resolve are
        what refuse it at runtime)."""
        cfg = self._load([self._spec(key_env="NOT_SET_ANYWHERE")])
        self.assertEqual("NOT_SET_ANYWHERE", cfg.accounts[0].key_env)
        self.assertEqual(TOKEN_ENV, cfg.auth.token_env)

    def test_bad_base_url_rejected(self):
        with self.assertRaises(config_mod.ConfigError):
            self._load([self._spec(base_url="ftp://example.invalid/v1")])

    def test_empty_models_rejected(self):
        with self.assertRaises(config_mod.ConfigError):
            self._load([self._spec(models={})])

    def test_empty_upstream_model_name_rejected(self):
        with self.assertRaises(config_mod.ConfigError):
            self._load([self._spec(models={"m": ""})])

    def test_empty_account_name_rejected(self):
        with self.assertRaises(config_mod.ConfigError):
            self._load([self._spec(name="")])

    def test_auth_token_must_be_an_env_var_name(self):
        path = os.path.join(self.tmp, "b.yml")
        with open(path, "w") as fh:
            fh.write("auth:\n  token: {env_file: env/absent.env, var: %s}\n"
                     "accounts:\n%s\n" % (TOKEN_ENV, self._spec()))
        with self.assertRaises(config_mod.ConfigError) as cm:
            config_mod.load(path)
        self.assertIn("environment variable", str(cm.exception))

    def test_manager_refuses_a_bad_reload_and_keeps_the_old_config(self):
        path = write_accounts(os.path.join(self.tmp, "c.yml"), [self._spec()],
                              self.token_env)
        mgr = config_mod.ConfigManager(path)
        self.assertEqual(1, len(mgr.get().accounts))
        time.sleep(0.01)
        with open(path, "w") as fh:
            fh.write("accounts:\n  - name: broken\n")     # no base_url/key/models
        cfg = mgr.get()
        self.assertEqual(1, len(cfg.accounts))            # old config kept
        self.assertEqual("one", cfg.accounts[0].name)
        # a second look at the same broken content does not retry forever
        self.assertEqual(1, len(mgr.get().accounts))
        time.sleep(0.01)
        write_accounts(path, [self._spec(name="two")], self.token_env)
        cfg = mgr.get()
        self.assertEqual("two", cfg.accounts[0].name)     # recovers when fixed

    def test_manager_refuses_a_config_carrying_a_retired_key(self):
        """A stale accounts.yml must not be able to change the rotation order."""
        path = write_accounts(os.path.join(self.tmp, "d.yml"), [self._spec()],
                              self.token_env)
        mgr = config_mod.ConfigManager(path)
        self.assertEqual(["one"], [a.name for a in mgr.get().accounts])
        time.sleep(0.01)
        write_accounts(path, [self._spec(name="two"),
                              self._spec(name="three", extra="priority: 1")],
                       self.token_env)
        cfg = mgr.get()
        self.assertEqual(["one"], [a.name for a in cfg.accounts])   # old kept


class T26EmptyStreamNoProgress(RouterCase):
    """200 + SSE frames carrying neither content nor a finish_reason.

    Real-world fault (2026-09-14, 5 hits in 43 min on one account): the
    upstream answers 200, sends a few hundred bytes of role-only frames and
    closes.  Relayed as-is the client dies with "Stream ended without
    finish_reason" while the router logged `REQ ok` => the pool never switched.
    The head of the stream is now probed *before* the response is committed, so
    this stays retryable.
    """

    def test_stream_without_progress_switches_account(self):
        a, b, _c = self.mocks
        a.script(mock.spec(no_finish=True))
        r = self.client.stream_chat()
        self.assertEqual(200, r.status)
        text = r.text()
        self.assertIn("[DONE]", text)
        self.assertIn("part0", text)                  # served by the next account
        self.assertEqual(1, a.count)
        self.assertEqual(1, b.count)
        line = self.fx.grep_log("EMPTY_STREAM")[-1]
        self.assertIn("account=alpha", line)
        self.assertIn("saw_finish_reason=False", line)
        self.assertIn("nothing sent to the client yet", line)
        # the internal reason is R_SERVER_ERROR (transient class => 20s blacklist);
        # /v1/models folds it to "throttled" (pool.py PUBLIC_REASON)
        self.assertEqual("throttled", self.fx.acct_view("alpha")["reason"])
        self.assertIn("account=beta", self.fx.grep_log("REQ ok account=")[-1])

    def test_finish_reason_only_stream_is_not_empty(self):
        # chunks=0 still emits the terminating finish_reason="stop" frame, so the
        # client can end its turn normally: the probe must NOT switch accounts
        # (guard against treating "no content" as "no progress").
        a, b, _c = self.mocks
        a.script(mock.spec(chunks=0))
        r = self.client.stream_chat()
        self.assertEqual(200, r.status)
        self.assertIn('"finish_reason": "stop"', r.text())
        self.assertEqual(1, a.count)
        self.assertEqual(0, b.count)
        self.assertEqual([], self.fx.grep_log("EMPTY_STREAM", timeout=1))

    def test_normal_stream_is_unaffected_by_the_probe(self):
        _a, _b, _c = self.mocks
        r = self.client.stream_chat()
        self.assertEqual(200, r.status)
        self.assertIn("part0", r.text())
        self.assertIn("[DONE]", r.text())
        self.assertEqual([], self.fx.grep_log("EMPTY_STREAM", timeout=1))


class TestSecretsUnit(unittest.TestCase):
    """Credentials are environment variables: the resolver reads a mapping."""

    def setUp(self):
        self.env = {"V": FAKE_KEYS["a"]}
        self.s = secrets_mod.Secrets(self.env)

    def test_mask_shapes(self):
        self.assertEqual("sk-sp-****wxyz", secrets_mod.mask("sk-sp-abcdefghijklmnopqrstuvwxyz"))
        self.assertEqual("sk-fake-****1111", secrets_mod.mask(FAKE_KEYS["a"]))
        self.assertEqual("****babe", secrets_mod.mask(FAKE_TOKEN))
        self.assertEqual("(empty)", secrets_mod.mask(""))

    def test_resolve_reads_the_environment(self):
        self.assertEqual(FAKE_KEYS["a"], self.s.resolve("V"))
        self.assertEqual(FAKE_KEYS["a"], self.s.resolve("V"))      # idempotent
        self.env["V"] = "sk-fake-zzzz9999"      # no cache: the next read sees it
        self.assertEqual("sk-fake-zzzz9999", self.s.resolve("V"))

    def test_unset_empty_and_nameless_are_errors(self):
        for env, name in (({"V": FAKE_KEYS["b"]}, "NOPE"),      # not set
                          ({"V": ""}, "V"),                     # set but empty
                          ({"V": FAKE_KEYS["b"]}, ""),          # no name given
                          ({"V": FAKE_KEYS["b"]}, None)):
            with self.assertRaises(secrets_mod.SecretError, msg=repr(name)):
                secrets_mod.Secrets(env).resolve(name)

    def test_check_name_shape(self):
        for good in ("V", "_V1", "DASHSCOPE_TEAMPLAN1_API_KEY"):
            self.assertTrue(secrets_mod.Secrets.check_name(good), good)
        for bad in ("", None, "sk-fake-aaaa1111", "9V", "A B", "env/k.env"):
            self.assertFalse(secrets_mod.Secrets.check_name(bad), repr(bad))

    def test_error_messages_are_secret_free(self):
        s = secrets_mod.Secrets({"V": FAKE_KEYS["c"]})
        try:
            s.resolve("NOPE")
        except secrets_mod.SecretError as e:
            self.assertNotIn(FAKE_KEYS["c"], str(e))
            self.assertIn("NOPE", str(e))       # names the variable, not a value

    def test_bearer_constant_time_compare(self):
        self.s = secrets_mod.Secrets({"V": FAKE_TOKEN})
        self.s.set_token_source("V")
        self.assertTrue(self.s.check_bearer("Bearer " + FAKE_TOKEN))
        self.assertTrue(self.s.check_bearer("bearer " + FAKE_TOKEN))
        self.assertFalse(self.s.check_bearer("Bearer " + FAKE_TOKEN[:-1]))
        self.assertFalse(self.s.check_bearer(FAKE_TOKEN))
        self.assertFalse(self.s.check_bearer(""))
        self.assertFalse(self.s.check_bearer(None))
        self.assertFalse(self.s.check_bearer("Basic " + FAKE_TOKEN))
        self.assertEqual(FAKE_TOKEN_MASK, self.s.token_hint())

    def test_bearer_without_a_source_fails_closed(self):
        with self.assertRaises(secrets_mod.SecretError):
            secrets_mod.Secrets({}).check_bearer("Bearer " + FAKE_TOKEN)

    def test_bearer_with_an_unset_token_fails_closed(self):
        s = secrets_mod.Secrets({})
        s.set_token_source(TOKEN_ENV)
        with self.assertRaises(secrets_mod.SecretError):
            s.check_bearer("Bearer " + FAKE_TOKEN)

    def test_redact_known_and_generic_shapes(self):
        self.s.resolve("V")
        self.assertEqual("key=sk-fake-****1111 here",
                         self.s.redact("key=%s here" % FAKE_KEYS["a"]))
        # unknown but key-shaped values are masked too (defence in depth)
        self.assertEqual("sk-unknown-****5678",
                         self.s.redact("sk-unknown-value-12345678"))
        self.assertIn("****", self.s.redact("Bearer abcdef0123456789"))
        self.assertNotIn("abcdef0123456789", self.s.redact("Bearer abcdef0123456789"))
        self.assertEqual("", self.s.redact(None))


class TestProxyUnit(unittest.TestCase):
    def test_upstream_path_mapping(self):
        base = "https://api.vendor.example/compatible-mode/v1"   # a base_url with a path prefix
        self.assertEqual("/compatible-mode/v1/chat/completions",
                         proxy_mod.upstream_path(base, "/v1/chat/completions"))
        self.assertEqual("/compatible-mode/v1/embeddings",
                         proxy_mod.upstream_path(base, "/v1/embeddings"))
        self.assertEqual("/compatible-mode/v1/chat/completions?a=1",
                         proxy_mod.upstream_path(base, "/v1/chat/completions?a=1"))
        self.assertEqual("/v1/chat/completions",
                         proxy_mod.upstream_path("http://h:1/v1", "/v1/chat/completions"))

    def test_prepare_body_maps_model_and_injects_once(self):
        body = json.dumps({"model": MODEL, "stream": True}).encode()
        out, injected, parsed = proxy_mod.prepare_body(body, "qwen/" + MODEL,
                                                       True, True)
        self.assertTrue(injected)
        self.assertEqual("qwen/" + MODEL, parsed["model"])
        self.assertTrue(parsed["stream_options"]["include_usage"])
        # idempotent: feeding the rewritten body back injects nothing
        out2, injected2, _ = proxy_mod.prepare_body(out, "qwen/" + MODEL, True, True)
        self.assertFalse(injected2)
        self.assertEqual(out, out2)

    def test_prepare_body_leaves_client_stream_options_alone(self):
        body = json.dumps({"model": MODEL, "stream": True,
                           "stream_options": {"include_usage": True}}).encode()
        out, injected, _ = proxy_mod.prepare_body(body, MODEL, True, True)
        self.assertFalse(injected)
        self.assertEqual(body, out)

    def test_prepare_body_non_stream_is_untouched(self):
        body = json.dumps({"model": MODEL}).encode()
        out, injected, _ = proxy_mod.prepare_body(body, MODEL, False, True)
        self.assertFalse(injected)
        self.assertNotIn(b"stream_options", out)

    def test_prepare_body_handles_garbage(self):
        out, injected, parsed = proxy_mod.prepare_body(b"not json", None, True, True)
        self.assertEqual(b"not json", out)
        self.assertFalse(injected)
        self.assertIsNone(parsed)
        out, injected, parsed = proxy_mod.prepare_body(b"", None, False, True)
        self.assertEqual(b"", out)
        self.assertFalse(injected)

    def test_header_forwarding(self):
        headers = {"Authorization": "Bearer inbound-token",
                   "Content-Type": "application/json",
                   "Accept-Encoding": "gzip", "Host": "127.0.0.1",
                   "Content-Length": "10", "Connection": "keep-alive",
                   "X-Trace": "keep-me"}
        out = proxy_mod.build_headers(headers, FAKE_KEYS["a"])
        self.assertEqual("Bearer " + FAKE_KEYS["a"], out["Authorization"])
        self.assertEqual("identity", out["Accept-Encoding"])
        self.assertNotIn("Host", out)
        self.assertNotIn("Content-Length", out)
        self.assertNotIn("Connection", out)
        self.assertEqual("application/json", out["Content-Type"])
        self.assertEqual("keep-me", out["X-Trace"])

    def test_usage_extraction(self):
        body = json.dumps({"usage": {"prompt_tokens": 3, "completion_tokens": 4}}).encode()
        self.assertEqual((3, 4), proxy_mod.usage_tokens(proxy_mod.usage_from_body(body)))
        self.assertIsNone(proxy_mod.usage_from_body(b"{}"))
        self.assertIsNone(proxy_mod.usage_from_body(b"not json"))
        self.assertEqual((0, 0), proxy_mod.usage_tokens(None))
        self.assertEqual((5, 6), proxy_mod.usage_tokens(
            {"input_tokens": 5, "output_tokens": 6}))

    def test_sse_line_parsing(self):
        line = b'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":2}}'
        usage, err = proxy_mod.sse_line_info(line)
        self.assertEqual(1, usage["prompt_tokens"])
        self.assertIsNone(err)
        self.assertEqual((None, None), proxy_mod.sse_line_info(b"data: [DONE]"))
        self.assertEqual((None, None), proxy_mod.sse_line_info(b": keepalive"))
        self.assertEqual((None, None), proxy_mod.sse_line_info(b"data: {broken"))
        _u, err = proxy_mod.sse_line_info(
            b'data: {"error":{"message":"Allocated quota exceeded"}}')
        self.assertIn("quota exceeded", err)

    def test_error_event_shape(self):
        event = proxy_mod.error_event("alpha", "boom").decode()
        self.assertIn("upstream_stream_interrupted", event)
        self.assertIn('"account": "alpha"', event)
        self.assertTrue(event.rstrip().endswith("data: [DONE]"))


class TestPoolUnit(unittest.TestCase):
    """Two-tier blacklist semantics against an injected clock (no sleeping)."""

    def setUp(self):
        os.makedirs(TMP_ROOT, exist_ok=True)
        self.tmp = tempfile.mkdtemp(prefix="pool-", dir=TMP_ROOT)
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.key_env = KEY_ENV_NAME
        self.token_env = TOKEN_ENV
        # the pool reads credentials through Secrets, which reads this mapping
        self.cred_env = {KEY_ENV_NAME: FAKE_KEYS["a"], TOKEN_ENV: FAKE_TOKEN}
        self.clock = {"t": 1_700_000_000.0}
        self.secrets = secrets_mod.Secrets(self.cred_env)

    def _pool(self, n_accounts=1, exhausted=3600, failure=60, models=None,
              specs=None):
        if specs is None:
            specs = [accounts_block("acc%d" % i, "https://example.invalid/v1",
                                    self.key_env, models or {"m": "m"})
                     for i in range(n_accounts)]
        path = write_accounts(os.path.join(self.tmp, "a.yml"), specs,
                              self.token_env,
                              {"blacklist_exhausted": str(exhausted),
                               "blacklist_failure": str(failure)})
        cfgm = config_mod.ConfigManager(path)
        return pool_mod.Pool(cfgm, self.secrets,
                             now_fn=lambda: self.clock["t"]), cfgm

    def test_a_fresh_pool_is_fully_available(self):
        pool, _cfgm = self._pool(n_accounts=2)
        self.assertEqual("ok", pool.state_for("acc0")["status"])
        self.assertIsNone(pool.state_for("acc0")["blacklist_until"])
        avail, skipped = pool.candidates("m")
        self.assertEqual(["acc0", "acc1"], [a.name for a in avail])
        self.assertEqual({}, skipped)
        self.assertEqual("ok", pool.pool_status())

    def test_key_for_keeps_only_the_masked_hint(self):
        pool, _cfgm = self._pool()
        acct = pool.candidates("m")[0][0]
        self.assertEqual(FAKE_KEYS["a"], pool.key_for(acct))
        st = pool.state_for("acc0")
        self.assertEqual(FAKE_MASKS["a"], st["key_hint"])
        self.assertEqual("ok", st["status"])
        self.assertNotIn(FAKE_KEYS["a"], json.dumps(st))

    def test_exhausted_uses_the_long_tier(self):
        pool, _cfgm = self._pool(exhausted=3600, failure=60)
        acct = pool.candidates("m")[0][0]
        pool.mark_exhausted(acct, detail="quota exceeded", http_status=429)
        st = pool.state_for("acc0")
        self.assertEqual(pool_mod.BLACKLISTED, st["status"])
        self.assertEqual(pool_mod.R_EXHAUSTED, st["blacklist_reason"])
        self.assertAlmostEqual(self.clock["t"] + 3600, st["blacklist_until"],
                               delta=1)
        self.assertEqual("quota exceeded", st["last_error"])
        self.assertEqual([], pool.candidates("m")[0])
        self.assertEqual({"acc0": "exhausted"}, pool.candidates("m")[1])
        self.assertEqual("all_exhausted", pool.pool_status())

    def test_every_failure_reason_uses_the_short_tier(self):
        for reason in (pool_mod.R_THROTTLED, pool_mod.R_SERVER_ERROR,
                       pool_mod.R_KEY_ERROR, pool_mod.R_UPSTREAM_KEY_REJECTED):
            pool, _cfgm = self._pool(n_accounts=2, exhausted=3600, failure=60)
            acct = pool.candidates("m")[0][0]
            pool.mark_failure(acct, reason, detail="d", http_status=429)
            st = pool.state_for("acc0")
            self.assertEqual(reason, st["blacklist_reason"], reason)
            self.assertAlmostEqual(self.clock["t"] + 60, st["blacklist_until"],
                                   delta=1)
            self.assertEqual({"acc0": reason}, pool.candidates("m")[1])
            self.assertEqual(["acc1"], [a.name for a in pool.candidates("m")[0]])
            self.assertEqual("degraded", pool.pool_status())
            # the 60s tier is the SHORT one: 3600s belongs to the quota tier
            self.assertLess(st["blacklist_until"] - self.clock["t"], 61)

    def test_recovery_is_lazy_and_the_next_request_is_the_probe(self):
        pool, _cfgm = self._pool(exhausted=100)
        acct = pool.candidates("m")[0][0]
        pool.mark_exhausted(acct, detail="quota")
        self.clock["t"] += 99
        self.assertFalse(pool.refresh())
        self.assertEqual(pool_mod.BLACKLISTED, pool.state_for("acc0")["status"])
        self.assertEqual([], pool.candidates("m")[0])
        self.clock["t"] += 2
        self.assertTrue(pool.refresh())
        st = pool.state_for("acc0")
        self.assertEqual(pool_mod.OK, st["status"])
        self.assertIsNone(st["blacklist_reason"])
        self.assertIsNone(st["blacklist_until"])
        self.assertEqual(1, len(pool.candidates("m")[0]))

    def test_the_tier_is_flat_a_failed_probe_is_not_doubled(self):
        pool, _cfgm = self._pool(exhausted=100)
        acct = pool.candidates("m")[0][0]
        for _ in range(3):
            pool.mark_exhausted(acct, detail="quota")
            self.assertAlmostEqual(self.clock["t"] + 100,
                                   pool.state_for("acc0")["blacklist_until"],
                                   delta=1)
            self.clock["t"] += 101
            pool.refresh()
        pool.record_ok(acct)
        self.assertEqual(pool_mod.OK, pool.state_for("acc0")["status"])

    def test_record_ok_clears_a_blacklist_at_once(self):
        pool, _cfgm = self._pool(exhausted=3600)
        acct = pool.candidates("m")[0][0]
        pool.mark_exhausted(acct, detail="quota")
        self.assertEqual([], pool.candidates("m")[0])
        pool.record_ok(acct, key_hint=FAKE_MASKS["a"])
        st = pool.state_for("acc0")
        self.assertEqual(pool_mod.OK, st["status"])
        self.assertIsNone(st["last_error"])
        self.assertEqual(FAKE_MASKS["a"], st["key_hint"])
        self.assertEqual(1, len(pool.candidates("m")[0]))

    def test_record_failure_never_blacklists(self):
        pool, _cfgm = self._pool()
        acct = pool.candidates("m")[0][0]
        pool.record_failure(acct, detail="bad params", http_status=400)
        st = pool.state_for("acc0")
        self.assertEqual(pool_mod.OK, st["status"])
        self.assertIsNone(st["blacklist_until"])
        self.assertEqual("bad params", st["last_error"])
        self.assertEqual(1, len(pool.candidates("m")[0]))
        self.assertEqual("ok", pool.pool_status())

    def test_last_error_is_redacted_and_bounded(self):
        pool, _cfgm = self._pool()
        acct = pool.candidates("m")[0][0]
        pool.mark_failure(acct, pool_mod.R_SERVER_ERROR,
                          detail="key %s leaked " % FAKE_KEYS["a"] + "x" * 900)
        err = pool.state_for("acc0")["last_error"]
        self.assertNotIn(FAKE_KEYS["a"], err)
        self.assertIn(FAKE_MASKS["a"], err)
        self.assertLessEqual(len(err), pool_mod.MAX_ERROR_LEN)

    def test_a_resolvable_key_clears_a_key_error_blacklist(self):
        pool, cfgm = self._pool()
        specs = [accounts_block("acc0", "https://example.invalid/v1",
                                "NOT_SET_ANYWHERE", {"m": "m"})]
        write_accounts(cfgm.get().path, specs, self.token_env)
        cfg2 = pool.cfgm.get()
        self.assertEqual("acc0", cfg2.accounts[0].name)
        with self.assertRaises(pool_mod.AccountKeyError):
            pool.key_for(cfg2.accounts[0])
        st = pool.state_for("acc0")
        self.assertEqual(pool_mod.R_KEY_ERROR, st["blacklist_reason"])
        self.assertEqual([], pool.candidates("m")[0])
        # the operator fixes the environment (in a deployment that means a
        # restart with the variable set; Secrets reads the mapping live)
        self.cred_env["NOT_SET_ANYWHERE"] = FAKE_KEYS["b"]
        self.assertEqual(FAKE_KEYS["b"], pool.key_for(cfg2.accounts[0]))
        st = pool.state_for("acc0")
        self.assertEqual(pool_mod.OK, st["status"])   # cleared, no deadline wait
        self.assertEqual(FAKE_MASKS["b"], st["key_hint"])

    def test_models_view_folds_the_internal_reasons(self):
        pool, cfgm = self._pool(n_accounts=3)
        accts = cfgm.get().accounts
        pool.mark_exhausted(accts[0], detail="quota")
        pool.mark_failure(accts[1], pool_mod.R_SERVER_ERROR, detail="503")
        pool.mark_failure(accts[2], pool_mod.R_UPSTREAM_KEY_REJECTED, detail="401")
        entry = pool.models_view()["data"][0]
        reasons = dict((a["name"], a["reason"]) for a in entry["accounts"])
        self.assertEqual({"acc0": "exhausted", "acc1": "throttled",
                          "acc2": "key_error"}, reasons)
        self.assertFalse(entry["available"])
        for a in entry["accounts"]:
            self.assertFalse(a["available"])
            self.assertIsNotNone(a["until"])
        # the fold table is explicit and exhaustive over the internal enum
        self.assertEqual(set(pool_mod.VIEW_REASON),
                         set((pool_mod.R_EXHAUSTED, pool_mod.R_THROTTLED,
                              pool_mod.R_SERVER_ERROR, pool_mod.R_KEY_ERROR,
                              pool_mod.R_UPSTREAM_KEY_REJECTED)))
        self.assertEqual({"exhausted", "throttled", "key_error"},
                         set(pool_mod.VIEW_REASON.values()))

    def test_models_view_marks_a_missing_mapping(self):
        specs = [accounts_block("acc0", "https://example.invalid/v1",
                                self.key_env, {"m": "m"}),
                 accounts_block("acc1", "https://example.invalid/v1",
                                self.key_env, {"other": "other"})]
        pool, _cfgm = self._pool(specs=specs)
        data = dict((x["id"], x) for x in pool.models_view()["data"])
        self.assertEqual(["m", "other"], sorted(data))
        by_name = dict((a["name"], a) for a in data["m"]["accounts"])
        self.assertEqual("no_mapping", by_name["acc1"]["reason"])
        self.assertFalse(by_name["acc1"]["available"])
        self.assertIsNone(by_name["acc1"]["upstream_model"])
        self.assertEqual("ok", by_name["acc0"]["reason"])
        self.assertTrue(data["m"]["available"])
        other = dict((a["name"], a) for a in data["other"]["accounts"])
        self.assertTrue(data["other"]["available"])   # acc1 still serves it
        self.assertEqual("ok", other["acc1"]["reason"])
        self.assertEqual("no_mapping", other["acc0"]["reason"])

    def test_health_view_has_no_sensitive_detail(self):
        pool, _cfgm = self._pool(n_accounts=2)
        view = pool.health_view(uptime_s=1.0)
        text = json.dumps(view)
        self.assertEqual("ok", view["status"])
        self.assertEqual("llm-router", view["service"])
        self.assertEqual(2, view["accounts_total"])
        self.assertEqual(2, view["accounts_available"])
        self.assertEqual(1.0, view["uptime_s"])
        self.assertNotIn("acc0", text)
        self.assertNotIn(FAKE_KEYS["a"], text)
        for needle in ("credits", "quota", "token", "key", "until", "reason",
                       "version"):
            self.assertNotIn(needle, text.lower())
        self.assertEqual(sorted(view),
                         ["accounts_available", "accounts_total", "pool",
                          "service", "status", "time", "uptime_s"])

    def test_health_view_reports_a_degraded_and_an_exhausted_pool(self):
        pool, cfgm = self._pool(n_accounts=2)
        accts = cfgm.get().accounts
        pool.mark_exhausted(accts[0], detail="quota")
        self.assertEqual("degraded", pool.health_view()["pool"])
        self.assertEqual(1, pool.health_view()["accounts_available"])
        pool.mark_exhausted(accts[1], detail="quota")
        view = pool.health_view()
        self.assertEqual("all_exhausted", view["pool"])
        self.assertEqual(0, view["accounts_available"])
        self.assertEqual(2, view["accounts_total"])


class TestEndToEndSecrets(unittest.TestCase):
    """The tracked faces: the shipped example config carries no plaintext.

    A deployment's own accounts.yml is NOT tracked in this repo (it is
    gitignored and lives next to the credentials it points at), so the pins for
    the real pool belong to that deployment, not here.
    """

    EXAMPLE = os.path.join(PKG, "accounts.example.yml")

    def test_example_accounts_yml_has_no_credential_material(self):
        with open(self.EXAMPLE) as fh:
            text = fh.read()
        self.assertNotIn("sk-", text)
        self.assertNotIn("=enc1:", text)        # no cipher value belongs here either
        for retired in ("priority:", "quota:", "credits_weights:",
                        "preemptive_ratio", "state.json", "ledger",
                        "LLM_ROUTER_ENVDEC", "LLM_ROUTER_WS", "envdec"):
            self.assertNotIn(retired, text)
        # the retired file-pointer form must not be *used* anywhere (the header
        # prose may still name it while explaining what replaced it)
        self.assertNotIn("key: {", text)
        self.assertNotIn("token: {", text)
        self.assertNotIn("127.0.0.1", text)      # no mock endpoint left behind
        self.assertNotIn("localhost", text)

    def test_example_accounts_yml_loads_and_is_the_documented_shape(self):
        """The example must load as shipped — and loading needs no credential
        anywhere, because parsing only validates variable NAMES."""
        cfg = config_mod.load(self.EXAMPLE)
        names = [a.name for a in cfg.ordered()]
        # pinned on purpose: accounts.example.yml IS the documented reference
        # shape, so editing the example must be reflected here in the same
        # change (a silent edit fails this case).
        self.assertEqual(["acme-payg", "acme-team", "fallback-metered"], names)
        self.assertEqual([1, 2, 3], [a.order for a in cfg.ordered()])
        for acct in cfg.accounts:
            self.assertTrue(acct.base_url.startswith("https://"), acct.base_url)
            self.assertNotIn("127.0.0.1", acct.base_url)
            self.assertNotIn("localhost", acct.base_url)
            # every credential is an environment-variable NAME, never a value
            self.assertTrue(secrets_mod.Secrets.check_name(acct.key_env),
                            acct.key_env)
            self.assertNotIn("sk-", acct.key_env)
        self.assertEqual(["ACME_API_KEY", "ACME_TEAM_API_KEY",
                          "ACME_METERED_API_KEY"],
                         [a.key_env for a in cfg.ordered()])
        self.assertEqual("ROUTER_TOKEN", cfg.auth.token_env)
        self.assertEqual(["/health"], cfg.auth.exempt_paths)
        self.assertTrue(cfg.inject_stream_options)
        self.assertEqual(3600.0, cfg.blacklist_exhausted)
        self.assertEqual(20.0, cfg.blacklist_failure)
        self.assertEqual(60.0, cfg.timeout.connect)
        self.assertEqual(300.0, cfg.timeout.read)
        # the metered last resort invents no pool name of its own, so dropping
        # that block would change no client-visible model name
        rest = set()
        for acct in cfg.ordered()[:-1]:
            rest |= set(acct.models)
        self.assertTrue(set(cfg.ordered()[-1].models) <= rest,
                        sorted(set(cfg.ordered()[-1].models) - rest))


# ==================== minimal-design specific cases ====================

class T22RestartResetsBlacklist(RouterCase):
    """Nothing is persisted, so a restart is the reset button."""

    fixture_kwargs = {"defaults": {"blacklist_exhausted": "3600"}}

    def test_a_second_pool_over_the_same_config_starts_clean(self):
        cfgm = config_mod.ConfigManager(self.fx.accounts_path)
        clock = {"t": time.time()}
        secrets = secrets_mod.Secrets(self.fx.env_vars)
        p1 = pool_mod.Pool(cfgm, secrets, now_fn=lambda: clock["t"])
        first = p1.candidates(MODEL)[0][0]
        self.assertEqual("alpha", first.name)
        p1.mark_exhausted(first, detail="quota")
        self.assertEqual(["beta", "gamma"],
                         [a.name for a in p1.candidates(MODEL)[0]])
        self.assertEqual({"alpha": "exhausted"}, p1.candidates(MODEL)[1])
        self.assertEqual("degraded", p1.pool_status())
        p2 = pool_mod.Pool(cfgm, secrets, now_fn=lambda: clock["t"])
        self.assertEqual("ok", p2.state_for("alpha")["status"])
        self.assertEqual(3, len(p2.candidates(MODEL)[0]))
        self.assertEqual("ok", p2.pool_status())
        self.assertEqual([], self.fx.stray_files())   # p1 wrote nothing anywhere

    def test_a_restart_clears_every_blacklist(self):
        for m in self.mocks:
            m.script(mock.exhausted_spec(429))
        r = self.client.chat()
        self.assertEqual(503, r.status)
        health = self.fx.pool_view()
        self.assertEqual("all_exhausted", health["pool"])
        self.assertEqual(0, health["accounts_available"])
        for name in ("alpha", "beta", "gamma"):
            self.assertEqual("exhausted", self.fx.acct_view(name)["reason"])
        for m in self.mocks:
            m.reset()                       # the upstreams are healthy again
        self.fx.restart()
        health = self.fx.pool_view()
        self.assertEqual("ok", health["pool"])
        self.assertEqual(3, health["accounts_available"])
        self.assertEqual(3, health["accounts_total"])
        for name in ("alpha", "beta", "gamma"):
            view = self.fx.acct_view(name)
            self.assertTrue(view["available"])
            self.assertEqual("ok", view["reason"])
        self.assertEqual(200, self.client.chat().status)
        self.assertEqual(1, self.mocks[0].count)
        self.assertTrue(self.fx.grep_log("STARTED"))
        self.assertEqual([], self.fx.stray_files())


class T23ConcurrencyConsistency(RouterCase):
    """ThreadingHTTPServer serves concurrently: the pool lock must hold."""

    def test_concurrent_pool_access_keeps_exactly_one_status_per_account(self):
        cfgm = config_mod.ConfigManager(self.fx.accounts_path)
        clock = {"t": time.time()}
        p = pool_mod.Pool(cfgm, secrets_mod.Secrets(self.fx.env_vars),
                          now_fn=lambda: clock["t"])
        names = ["alpha", "beta", "gamma"]
        errors = []

        def worker(i):
            try:
                for _ in range(30):
                    p.refresh()
                    avail, skipped = p.candidates(MODEL)
                    for a in avail:
                        p.state_for(a.name)
                    target = cfgm.get().accounts[i % 3]
                    if i % 3 == 0:
                        p.mark_exhausted(target, detail="q%d" % i)
                    elif i % 3 == 1:
                        p.mark_failure(target, pool_mod.R_THROTTLED,
                                       detail="t%d" % i)
                    else:
                        p.record_ok(target)
                    p.models_view()
                    p.health_view()
                    p.pool_status()
                    self.assertEqual(3, len(set(names) | set(skipped) |
                                            set(a.name for a in avail)))
            except Exception as e:                     # noqa: BLE001
                errors.append("%s: %s" % (type(e).__name__, e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(9)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)                          # bounded
            self.assertFalse(t.is_alive(), "pool worker hung")
        self.assertEqual([], errors)
        for name in names:
            st = p.state_for(name)
            self.assertIn(st["status"], (pool_mod.OK, pool_mod.BLACKLISTED))
            if st["status"] == pool_mod.BLACKLISTED:
                self.assertIsNotNone(st["blacklist_reason"])
                self.assertIsNotNone(st["blacklist_until"])
                self.assertGreater(st["blacklist_until"], clock["t"] - 1)
            else:
                self.assertIsNone(st["blacklist_reason"])
                self.assertIsNone(st["blacklist_until"])
        view = p.health_view()
        self.assertEqual(3, view["accounts_total"])
        self.assertTrue(0 <= view["accounts_available"] <= 3)
        self.assertIn(view["pool"], ("ok", "degraded", "all_exhausted"))

    def test_concurrent_requests_never_leave_the_view_in_a_half_state(self):
        n = 12
        out = []
        lock = threading.Lock()

        def one():
            try:
                status = self.client.chat().status
            except Exception as e:                     # noqa: BLE001
                status = "ERR %s" % type(e).__name__
            with lock:
                out.append(status)

        threads = [threading.Thread(target=one) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=REQ_TIMEOUT + 30)            # bounded
            self.assertFalse(t.is_alive(), "request worker hung")
        self.assertEqual([200] * n, sorted(out, key=str))
        self.assertEqual(n, sum(m.count for m in self.mocks))
        for entry in self.fx.models_view()["data"]:
            for a in entry["accounts"]:
                if a["available"]:
                    self.assertEqual("ok", a["reason"])
                    self.assertNotIn("until", a)
                else:
                    self.assertNotEqual("ok", a["reason"])
                    self.assertIn(a["reason"],
                                  ("exhausted", "throttled", "key_error",
                                   "no_mapping"))
        health = self.fx.pool_view()
        self.assertEqual(3, health["accounts_total"])
        self.assertIn(health["pool"], ("ok", "degraded", "all_exhausted"))
        self.assertEqual([], self.fx.stray_files())


class T24ResidualKeysRejected(RouterCase):
    """A stale accounts.yml must not be able to change the rotation order."""

    RETIRED = ("priority: 1", "quota: {credits: 100}", "credits_weights: {}")

    def test_each_retired_key_is_a_config_error(self):
        tmp = tempfile.mkdtemp(prefix="retired-", dir=TMP_ROOT)
        self.addCleanup(shutil.rmtree, tmp, True)
        self.assertEqual(("priority", "quota", "credits_weights"),
                         config_mod.REJECTED_ACCOUNT_KEYS)
        for retired in self.RETIRED:
            specs = [accounts_block("acc0", "https://example.invalid/v1",
                                    KEY_ENV_NAME, {"m": "m"}, extra=retired)]
            path = write_accounts(os.path.join(tmp, "a.yml"), specs, TOKEN_ENV)
            with self.assertRaises(config_mod.ConfigError) as cm:
                config_mod.load(path)
            self.assertIn("retired key", str(cm.exception), retired)

    def test_a_hot_reload_carrying_a_retired_key_is_refused(self):
        self.assertEqual(200, self.client.chat().status)
        with open(self.fx.accounts_path) as fh:
            good = fh.read()
        rejected = 0
        reloaded = 0
        for retired in self.RETIRED:
            bad = good.replace("  - name: beta", "  - name: beta\n    " + retired,
                               1)
            self.assertNotEqual(good, bad)
            with open(self.fx.accounts_path, "w") as fh:
                fh.write(bad)
            time.sleep(0.05)
            self.assertEqual(200, self.client.get("/health", token=None).status)
            lines = self.fx.grep_log("CONFIG_REJECTED")
            self.assertEqual(rejected + 1, len(lines), retired)
            rejected = len(lines)
            self.assertIn("retired key", lines[-1])
            health = self.fx.pool_view()
            self.assertEqual(3, health["accounts_total"])   # unchanged
            self.assertEqual(3, health["accounts_available"])
            self.assertEqual(200, self.client.chat().status)  # old cfg still serves
            with open(self.fx.accounts_path, "w") as fh:
                fh.write(good)
            time.sleep(0.05)
            self.assertEqual(200, self.client.get("/health", token=None).status)
            reloaded += 1
            self.assertGreaterEqual(len(self.fx.grep_log("CONFIG_RELOADED")),
                                    reloaded)
        self.assertEqual(3, self.fx.pool_view()["accounts_total"])
        self.assertEqual([], self.fx.stray_files())


class T25UpstreamKeyRejected(RouterCase):
    """An upstream 401 is the account's fault: blacklist + switch, never leak."""

    fixture_kwargs = {"defaults": {"blacklist_failure": "60",
                                   "blacklist_exhausted": "3600"}}
    UPSTREAM_401 = "Invalid API-key provided."

    def test_a_401_blacklists_and_switches_instead_of_being_passed_through(self):
        a, b, _c = self.mocks
        a.script(mock.error_spec(401, self.UPSTREAM_401))
        r = self.client.chat()
        self.assertEqual(200, r.status, r.text())
        self.assertEqual(1, a.count)
        self.assertEqual(1, b.count)                 # switched in the same request
        self.assertNotIn(self.UPSTREAM_401, r.text())  # never relayed to the client
        view = self.fx.acct_view("alpha")
        self.assertFalse(view["available"])
        self.assertEqual("key_error", view["reason"])  # folded credential class
        self.assertIsNotNone(view["until"])
        line = self.fx.grep_log("UPSTREAM_KEY_REJECTED")[0]
        self.assertIn("account=alpha", line)
        self.assertIn("blacklisted 60s", line)        # the short tier
        self.assertIn("-> next account", line)
        self.assertIn(FAKE_MASKS["a"], line)          # masked hint, never the key
        self.assertNotIn(FAKE_KEYS["a"], line)
        bl = self.fx.grep_log("ACCOUNT_BLACKLISTED")[0]
        self.assertIn("reason=upstream_key_rejected", bl)
        self.assertIn("blacklist_s=60", bl)
        self.assertFalse(self.fx.grep_log("ACCOUNT_EXHAUSTED", timeout=0.3))

    def test_a_401_carrying_quota_wording_still_lands_in_the_long_tier(self):
        a, _b, _c = self.mocks
        # a real arrears wording: `arrearage` is one of classify's tier-1 phrases,
        # so the WORDING wins over the 401 status code (precedence unchanged)
        a.script(mock.error_spec(401, "Access denied: arrearage, please recharge"
                                  " your account"))
        self.assertEqual(200, self.client.chat().status)
        line = self.fx.grep_log("ACCOUNT_EXHAUSTED")[0]
        self.assertIn("account=alpha", line)
        self.assertIn("blacklist_s=3600", line)       # wording beats the status
        self.assertEqual("exhausted", self.fx.acct_view("alpha")["reason"])
        self.assertFalse(self.fx.grep_log("UPSTREAM_KEY_REJECTED", timeout=0.3))

    def test_the_whole_pool_refusing_keys_yields_503_not_the_upstream_401(self):
        for m in self.mocks:
            m.script(mock.error_spec(401, self.UPSTREAM_401))
        r = self.client.chat()
        self.assertEqual(503, r.status)
        body = r.json()
        self.assertEqual("all_accounts_exhausted", body["error"]["type"])
        self.assertEqual(["upstream_key_rejected"] * 3,
                         [x["reason"] for x in body["error"]["accounts"]])
        self.assertNotIn(self.UPSTREAM_401, r.text())
        self.assertEqual(3, len(self.fx.grep_log("UPSTREAM_KEY_REJECTED")))
        self.assertEqual(0, self.fx.pool_view()["accounts_available"])
        self.assertEqual("all_exhausted", self.fx.pool_view()["pool"])

if __name__ == "__main__":
    unittest.main(verbosity=2)
