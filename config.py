"""config.py — accounts.yml load, validation and mtime-based hot reload.

accounts.yml is the configuration face: adding/removing an account, changing
the blacklist durations or the timeouts takes effect on the next request without
restarting the service (and without a service version bump where a deployment
keeps one — that discipline covers the service definition and the code, not
this file).

The **writing order of the `accounts:` list is the rotation order** (`order`,
1-based). There is no numeric priority field: to move an account earlier in the
rotation, move its block up.

A reload that fails validation is refused: the previous config stays in effect
and an ERROR line is logged. Blocks that still carry one of the retired keys
(`priority` / `quota` / `credits_weights`) are refused rather than silently
ignored, so a stale file cannot quietly change the rotation order.

Credentials are **names of environment variables**, never values and never file
pointers: `key: ACME_API_KEY` / `auth: {token: ROUTER_TOKEN}`. This module
validates the shape of a name (an identifier) and refuses a mapping — the
retired `{env_file:, var:}` form — so a stale file fails loudly instead of
silently reading nothing. Whether a name is actually SET is the deployment's
gate (a launcher's require_env) and the runtime's (secrets.resolve); parsing
stays credential-free, so a config check needs no secret in the environment.

Python 3.8 syntax floor; stdlib + PyYAML only.
"""
import os
import re
import threading

import yaml

# retired configuration keys: their presence means the file still describes the
# removed local-quota face, so the whole reload is refused (never ignored)
REJECTED_ACCOUNT_KEYS = ("priority", "quota", "credits_weights")
# a credential reference is an environment-variable NAME. Kept here (and not
# imported from secrets.py) on purpose: `import secrets` resolves to the stdlib
# module unless this directory is first on sys.path, and a schema check must not
# depend on that ordering.
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ConfigError(Exception):
    pass


def _num(value, default, name, minimum=None):
    if value is None:
        value = default
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ConfigError("%s must be a number, got %r" % (name, value))
    if minimum is not None and out < minimum:
        raise ConfigError("%s must be >= %s, got %s" % (name, minimum, out))
    return out


class Timeout(object):
    def __init__(self, connect=10.0, read=300.0):
        self.connect = connect
        self.read = read


class Account(object):
    def __init__(self, name, order, base_url, key_env, models,
                 timeout=None, ssl_verify=True):
        self.name = name
        self.order = order                       # 1-based position in accounts:
        self.base_url = base_url.rstrip("/")
        self.key_env = key_env                   # NAME of the env var holding it
        self.models = models                     # pool name -> upstream name
        self.timeout = timeout
        self.ssl_verify = ssl_verify

    def upstream_model(self, pool_model):
        return self.models.get(pool_model)


class Auth(object):
    def __init__(self, token_env, exempt_paths):
        self.token_env = token_env               # NAME of the env var holding it
        self.exempt_paths = exempt_paths

    def is_exempt(self, path):
        p = path or "/"
        if len(p) > 1 and p.endswith("/"):
            p = p.rstrip("/") or "/"
        return p in self.exempt_paths


class Config(object):
    def __init__(self, path, mtime, accounts, blacklist_exhausted,
                 blacklist_failure, timeout, auth, inject_stream_options=True):
        self.path = path
        self.mtime = mtime
        self.accounts = accounts
        self.by_name = dict((a.name, a) for a in accounts)
        self.blacklist_exhausted = blacklist_exhausted
        self.blacklist_failure = blacklist_failure
        self.timeout = timeout
        self.auth = auth
        self.inject_stream_options = inject_stream_options

    def ordered(self):
        """Rotation order = the writing order of the `accounts:` list."""
        return list(self.accounts)

    def model_names(self):
        """Union of pool model names, ordered by the best (lowest) order."""
        best = {}
        for acct in self.ordered():
            for name in acct.models:
                if name not in best:
                    best[name] = acct.order
        return [n for n, _ in sorted(best.items(), key=lambda kv: (kv[1], kv[0]))]


def _as_dict(value, name):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError("%s must be a mapping, got %s" % (name, type(value).__name__))
    return value


def _parse_timeout(raw, name, default):
    raw = _as_dict(raw, name)
    return Timeout(_num(raw.get("connect"), default.connect, name + ".connect", 0.1),
                   _num(raw.get("read"), default.read, name + ".read", 0.1))


def _parse_models(raw, name):
    raw = _as_dict(raw, name + ".models")
    if not raw:
        raise ConfigError("%s: models must map at least one pool model name" % name)
    out = {}
    for k, v in raw.items():
        if not isinstance(v, str) or not v.strip():
            raise ConfigError("%s: models.%s must be a non-empty upstream model "
                              "name, got %r" % (name, k, v))
        out[str(k)] = v.strip()
    return out


def _parse_account(raw, idx, defaults):
    name = "accounts[%d]" % idx
    if not isinstance(raw, dict):
        raise ConfigError("%s must be a mapping" % name)
    acc_name = raw.get("name")
    if not isinstance(acc_name, str) or not acc_name.strip():
        raise ConfigError("%s: name must be a non-empty string" % name)
    name = "account %s" % acc_name
    stale = [k for k in REJECTED_ACCOUNT_KEYS if k in raw]
    if stale:
        raise ConfigError("%s: retired key(s) %s — the rotation order is the "
                          "writing order of the accounts list and no local quota "
                          "face exists; remove them"
                          % (name, ", ".join(sorted(stale))))
    key_env = _parse_env_name(raw.get("key"), name + ".key")
    base_url = raw.get("base_url")
    if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
        raise ConfigError("%s: base_url must start with http:// or https://, got %r"
                          % (name, base_url))
    return Account(
        name=acc_name.strip(),
        order=idx + 1,                       # 1-based writing order
        base_url=base_url.strip(),
        key_env=key_env,
        models=_parse_models(raw.get("models"), name),
        timeout=_parse_timeout(raw.get("timeout"), name, defaults["timeout"]),
        ssl_verify=bool(raw.get("ssl_verify", True)),
    )


def _parse_env_name(value, name):
    """A credential reference: the NAME of an environment variable.

    A mapping is the retired `{env_file:, var:}` file-pointer form and is refused
    with a message that says what to write instead; anything that is not an
    identifier is refused as a possible literal credential (a pasted `sk-...`
    value never matches the identifier shape).
    """
    if isinstance(value, dict):
        raise ConfigError("%s must be the NAME of an environment variable (a bare "
                          "string), got a mapping with key(s) %s — the "
                          "{env_file, var} file-pointer form is retired: the "
                          "router reads credentials from its own environment and "
                          "does not know how a deployment stores them"
                          % (name, ", ".join(sorted(map(str, value)))))
    if not isinstance(value, str) or not ENV_NAME_RE.match(value.strip()):
        raise ConfigError("%s must be an environment-variable name matching "
                          "[A-Za-z_][A-Za-z0-9_]* (a credential value is never "
                          "written into this file), got %r" % (name, value))
    return value.strip()


def _parse_auth(raw):
    raw = _as_dict(raw, "auth")
    token_env = _parse_env_name(raw.get("token"), "auth.token")
    exempt = raw.get("exempt_paths") or ["/health"]
    if not isinstance(exempt, list):
        raise ConfigError("auth.exempt_paths must be a list of paths")
    paths = []
    for p in exempt:
        if not isinstance(p, str) or not p.startswith("/"):
            raise ConfigError("auth.exempt_paths entries must start with '/', got %r" % (p,))
        paths.append(p.rstrip("/") or "/")
    return Auth(token_env, paths)


def load(path):
    """Parse + validate accounts.yml into a Config (raises ConfigError)."""
    if not os.path.isfile(path):
        raise ConfigError("accounts file not found: %s" % path)
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConfigError("accounts file must be a YAML mapping: %s" % path)
    mtime = os.path.getmtime(path)

    d = _as_dict(raw.get("defaults"), "defaults")
    base_timeout = _parse_timeout(d.get("timeout"), "defaults.timeout", Timeout())
    blacklist_exhausted = _num(d.get("blacklist_exhausted"), 3600.0,
                               "defaults.blacklist_exhausted", 1.0)
    blacklist_failure = _num(d.get("blacklist_failure"), 60.0,
                             "defaults.blacklist_failure", 1.0)
    inject = bool(d.get("inject_stream_options", True))
    defaults = {"timeout": base_timeout}

    accounts_raw = raw.get("accounts")
    if not isinstance(accounts_raw, list) or not accounts_raw:
        raise ConfigError("accounts must be a non-empty list")
    accounts = [_parse_account(a, i, defaults) for i, a in enumerate(accounts_raw)]
    seen = set()
    for a in accounts:
        if a.name in seen:
            raise ConfigError("duplicate account name: %s" % a.name)
        seen.add(a.name)

    auth = _parse_auth(raw.get("auth"))
    return Config(path=path, mtime=mtime, accounts=accounts,
                  blacklist_exhausted=blacklist_exhausted,
                  blacklist_failure=blacklist_failure,
                  timeout=base_timeout, auth=auth,
                  inject_stream_options=inject)


class ConfigManager(object):
    """Holds the live Config; reloads on mtime change, refuses bad reloads."""

    def __init__(self, path, logger=None):
        self.path = os.path.abspath(path)
        self.log = logger
        self._lock = threading.RLock()
        self._cfg = load(self.path)
        self._failed_mtime = None
        if self.log:
            self.log.info("config loaded: %s (%d account(s), auth token from env "
                          "%s)", self.path, len(self._cfg.accounts),
                          self._cfg.auth.token_env)

    def get(self):
        with self._lock:
            try:
                mtime = os.path.getmtime(self.path)
            except OSError:
                return self._cfg
            if mtime == self._cfg.mtime:
                return self._cfg
            if mtime == self._failed_mtime:
                return self._cfg          # already refused this exact content
            try:
                cfg = load(self.path)
            except ConfigError as e:
                self._failed_mtime = mtime
                if self.log:
                    self.log.error("CONFIG_REJECTED hot reload refused, keeping "
                                   "previous config: %s", e)
                return self._cfg
            except Exception as e:                     # yaml parse errors & co
                self._failed_mtime = mtime
                if self.log:
                    self.log.error("CONFIG_REJECTED hot reload refused, keeping "
                                   "previous config: %s: %s", type(e).__name__, e)
                return self._cfg
            self._failed_mtime = None
            self._cfg = cfg
            if self.log:
                self.log.info("CONFIG_RELOADED %s (%d account(s))",
                              cfg.path, len(cfg.accounts))
            return cfg
