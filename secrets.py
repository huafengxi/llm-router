"""secrets.py — the single choke point for every plaintext credential.

Credentials are **environment variables, referenced by name**: accounts.yml says
`key: ACME_API_KEY` and this module reads `os.environ["ACME_API_KEY"]`. How a
deployment obtains those values (a plaintext env, an inline-encrypted file
decrypted at spawn, a secret store, a shell wrapper) is deliberately **not this
package's business** — it spawns no decryptor, reads no credential file and
knows no cipher format. Whoever starts the router puts the values in its
environment; `require_env`-style gates belong to that launcher, not here.

Nothing else in this package may read a credential from the environment or hold
a plaintext API key / router token. Everything that leaves the process (log
lines, HTTP response bodies) is passed through mask()/redact() first, so a
credential can never be transcribed in clear.

Plaintext values live only in this object's `_known` registry (for redaction);
the environment itself is fixed for the process lifetime, so there is nothing to
cache and nothing to invalidate — which also means **rotating a credential is a
restart**, not a hot reload.

Python 3.8 syntax floor; stdlib only.
"""
import hmac
import os
import re
import threading

# `sk-sp-`, `sk-`, `hf_`-style provider prefixes worth keeping in a mask
PREFIX_RE = re.compile(r"^((?:[A-Za-z]{2,8}[-_]){1,2})")
# defence in depth: anything that still looks like a provider key or a bearer
# credential after the known-value pass gets masked too
GENERIC_KEY_RE = re.compile(r"\b(sk-[A-Za-z0-9._-]{6,})\b")
GENERIC_BEARER_RE = re.compile(r"((?i:bearer)\s+)([A-Za-z0-9._\-]{6,})")
# what accounts.yml may name: a shell identifier, never a literal credential
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class SecretError(RuntimeError):
    """A credential could not be resolved (unset/empty variable, bad name).

    The message is safe to log: it carries variable NAMES and no value, and it is
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
    def __init__(self, env=None):
        """`env` defaults to os.environ; tests inject a dict instead."""
        self.env = os.environ if env is None else env
        self._lock = threading.RLock()
        self._known = {}          # plaintext -> mask, for redact()
        self._token_name = None   # env var name of the inbound bearer token

    # ---------------- resolution ----------------

    @staticmethod
    def check_name(name):
        """True when `name` is usable as an environment-variable name."""
        return bool(name) and bool(ENV_NAME_RE.match(str(name).strip()))

    def resolve(self, name):
        """Plaintext value of the environment variable called `name`."""
        if not name or not str(name).strip():
            raise SecretError("credential name is empty (accounts.yml points at "
                              "an environment variable by name)")
        name = str(name).strip()
        if name not in self.env:
            raise SecretError("environment variable %s is not set — the "
                              "deployment injects credentials into this process's "
                              "environment (see README «Credentials»)" % name)
        value = self.env[name]
        if value is None or str(value) == "":
            raise SecretError("environment variable %s is empty" % name)
        value = str(value)
        self.note(value)
        return value

    # ---------------- inbound bearer token ----------------

    def set_token_source(self, name):
        self._token_name = name

    def router_token(self):
        if not self._token_name:
            raise SecretError("router token source not configured "
                              "(accounts.yml `auth.token`)")
        return self.resolve(self._token_name)

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
