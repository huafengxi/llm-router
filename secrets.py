"""secrets.py — the single choke point for every plaintext credential.

Nothing else in this package may invoke the decryptor or hold a plaintext
API key / router token.  Everything that leaves the process (log lines, HTTP
response bodies) is passed through mask()/redact() first, so a credential can
never be transcribed in clear.

Plaintext values live only in this object's in-memory cache (TTL + mtime based).
The pool state that mask() feeds (`key_hint`) is in-memory too: the service
persists nothing, so the only durable face a credential could reach is the log.

Python 3.8 syntax floor; stdlib only.
"""
import hmac
import os
import re
import subprocess
import sys
import threading
import time

DEFAULT_TTL = 300.0          # re-decrypt at most every N seconds per (file, var)
DEC_TIMEOUT = 30             # bounded wait on the envdec.py subprocess
VALUE_RE = re.compile(r"^([A-Za-z_]\w*)=(.*)$")
# `sk-sp-`, `sk-`, `hf_`-style provider prefixes worth keeping in a mask
PREFIX_RE = re.compile(r"^((?:[A-Za-z]{2,8}[-_]){1,2})")
# defence in depth: anything that still looks like a provider key or a bearer
# credential after the known-value pass gets masked too
GENERIC_KEY_RE = re.compile(r"\b(sk-[A-Za-z0-9._-]{6,})\b")
GENERIC_BEARER_RE = re.compile(r"((?i:bearer)\s+)([A-Za-z0-9._\-]{6,})")


class SecretError(RuntimeError):
    """A credential could not be resolved (missing file, bad var, decrypt fail).

    The message is safe to log: it carries paths and return codes only, and is
    passed through redact() before it is built.
    """


def mask(value):
    """`sk-sp-****abcd` / `****abcd` — never more than the last four chars."""
    if not value:
        return "(empty)"
    value = str(value)
    tail = value[-4:]
    m = PREFIX_RE.match(value)
    prefix = ""
    if m and len(value) >= len(m.group(1)) + 8:
        prefix = m.group(1)
    return "%s****%s" % (prefix, tail)


class Secrets(object):
    def __init__(self, ws_root, python=None, dec=None, ttl=DEFAULT_TTL):
        self.ws = os.path.abspath(ws_root)
        self.python = python or sys.executable
        # Decryptor contract: `<python> <dec> <env-file>` prints the file's
        # KEY=VALUE lines with every `enc1:`-prefixed value in clear (stdout
        # only, nothing written). Default = <ws>/encrypt/envdec.py; point
        # LLM_ROUTER_ENVDEC at any script honouring that contract.
        self.dec = dec or os.environ.get("LLM_ROUTER_ENVDEC") \
            or os.path.join(self.ws, "encrypt", "envdec.py")
        self.ttl = ttl
        self._lock = threading.RLock()
        self._cache = {}          # (env_file, var) -> [expires, value, mtime]
        self._known = {}          # plaintext -> mask, for redact()
        self._token_source = None  # (env_file, var) of the inbound bearer token

    # ---------------- resolution ----------------

    def _path(self, env_file):
        return env_file if os.path.isabs(env_file) else os.path.join(self.ws, env_file)

    def _decrypt_file(self, path):
        try:
            r = subprocess.run([self.python, self.dec, path],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               timeout=DEC_TIMEOUT)
        except subprocess.TimeoutExpired:
            raise SecretError("envdec.py timed out after %ss for %s"
                              % (DEC_TIMEOUT, path))
        except OSError as e:
            raise SecretError("envdec.py could not be executed for %s: %s"
                              % (path, type(e).__name__))
        if r.returncode != 0:
            err = self.redact(r.stderr.decode("utf-8", "replace")).strip()
            raise SecretError("envdec.py failed for %s (rc=%s): %s"
                              % (path, r.returncode, err[:200]))
        return r.stdout.decode("utf-8", "replace")

    @staticmethod
    def _parse_env(text):
        """KEY=VALUE lines (optional `export `, optional matching quotes)."""
        out = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            m = VALUE_RE.match(line)
            if not m:
                continue
            v = m.group(2).strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
                v = v[1:-1]
            out[m.group(1)] = v
        return out

    def resolve(self, env_file, var):
        """Plaintext value of `var` in `env_file` (decrypted in memory, cached)."""
        if not env_file or not var:
            raise SecretError("secret source incomplete (env_file=%r var=%r)"
                              % (bool(env_file), bool(var)))
        key = (env_file, var)
        path = self._path(env_file)
        now = time.time()
        with self._lock:
            hit = self._cache.get(key)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            raise SecretError("secret env file missing: %s" % env_file)
        if hit is not None and hit[2] == mtime and hit[0] > now:
            return hit[1]
        values = self._parse_env(self._decrypt_file(path))
        if var not in values:
            raise SecretError("variable %s not found in %s (has: %d var(s))"
                              % (var, env_file, len(values)))
        value = values[var]
        if not value:
            raise SecretError("variable %s in %s is empty" % (var, env_file))
        with self._lock:
            self._cache[key] = [now + self.ttl, value, mtime]
            self._known[value] = mask(value)
        return value

    def forget(self):
        with self._lock:
            self._cache.clear()

    # ---------------- inbound bearer token ----------------

    def set_token_source(self, env_file, var):
        self._token_source = (env_file, var)

    def router_token(self):
        if not self._token_source:
            raise SecretError("router token source not configured "
                              "(accounts.yml `auth.token`)")
        return self.resolve(self._token_source[0], self._token_source[1])

    def token_hint(self):
        """Masked router token — safe for logs/README, never the value itself."""
        return mask(self.router_token())

    def check_bearer(self, header_value):
        """Constant-time verification of an inbound `Authorization` header.

        Always runs one hmac.compare_digest against the real token (no early
        return on missing/malformed input), so timing and control flow do not
        leak whether a prefix was right or how long the token is.
        """
        token = self.router_token()          # may raise SecretError
        supplied = ""
        if header_value:
            parts = str(header_value).split(None, 1)
            if len(parts) == 2 and parts[0].lower() == "bearer":
                supplied = parts[1].strip()
        return hmac.compare_digest(supplied.encode("utf-8"),
                                   token.encode("utf-8"))

    # ---------------- outbound sanitisation ----------------

    @staticmethod
    def mask(value):
        return mask(value)

    def note(self, value):
        """Register a plaintext value so redact() will scrub it anywhere."""
        if value:
            with self._lock:
                self._known[str(value)] = mask(value)

    def known(self):
        with self._lock:
            return dict(self._known)

    def redact(self, text):
        """Scrub every known credential (plus generic key/bearer shapes)."""
        if text is None:
            return ""
        out = str(text)
        with self._lock:
            items = sorted(self._known.items(), key=lambda kv: -len(kv[0]))
        for value, masked in items:
            if value and value in out:
                out = out.replace(value, masked)
        out = GENERIC_KEY_RE.sub(lambda m: mask(m.group(1)), out)
        out = GENERIC_BEARER_RE.sub(lambda m: m.group(1) + mask(m.group(2)), out)
        return out
