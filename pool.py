"""pool.py — the account pool state machine (all state in memory).

Candidate selection follows the **writing order** of `accounts.yml`
(`Config.ordered()`); an account leaves the rotation only when it is blacklisted,
and a blacklist has exactly two durations:

  blacklist_exhausted  quota/arrears class (classify -> exhausted)
  blacklist_failure    everything else that is the account's fault: rate limit,
                       5xx, unknown status, network error, unresolvable key, an
                       upstream 401 (refused credential) and an upstream 404 for
                       a pool model the account maps (it cannot serve what it
                       claims to serve)

Availability rules:
  blacklisted -> skipped until `blacklist_until`, with `blacklist_reason` saying
                 why (exhausted / throttled / server_error / key_error /
                 upstream_key_rejected)
  a plain client-side 4xx (400/405/413/415/422, and a 404 for a request that
                 named no model this account maps) is NOT the account's fault:
                 it never blacklists and never switches — server.py returns it
                 to the client unchanged

Recovery is lazy and request-driven, with no background thread: `refresh()`
does one time comparison per account at the start of every request, and the
first real request after a blacklist expires acts as the probe. A restart
resets everything — nothing is persisted, so debug detail lives in the log only.

`threading.RLock` guards every read and write of the state dict: the server is a
ThreadingHTTPServer, and with no persistence layer this lock is the only
correctness barrier left.

Python 3.8 syntax floor; stdlib only.
"""
import datetime
import threading
import time

MAX_ERROR_LEN = 300

OK = "ok"
BLACKLISTED = "blacklisted"

# blacklist_reason values (internal state)
R_EXHAUSTED = "exhausted"
R_THROTTLED = "throttled"
R_SERVER_ERROR = "server_error"
R_KEY_ERROR = "key_error"
R_UPSTREAM_KEY_REJECTED = "upstream_key_rejected"

# /v1/models exposes a smaller reason enum than the internal one, so the
# internal reasons are folded by class: quota stays `exhausted`, both
# credential problems become `key_error`, and the transient 60s class
# (rate limit / 5xx / network / unknown) becomes `throttled` — the exact
# recovery instant is carried alongside as `until`.
VIEW_REASON = {
    R_EXHAUSTED: "exhausted",
    R_THROTTLED: "throttled",
    R_SERVER_ERROR: "throttled",
    R_KEY_ERROR: "key_error",
    R_UPSTREAM_KEY_REJECTED: "key_error",
}

# reasons that make a whole-pool 503 an "exhausted" flavour (quota + credential:
# retrying soon is pointless) rather than a transient-failure flavour
EXHAUSTION_REASONS = (R_EXHAUSTED, R_KEY_ERROR, R_UPSTREAM_KEY_REJECTED)


def now_iso(ts=None):
    return datetime.datetime.fromtimestamp(ts if ts is not None
                                           else datetime.datetime.now().timestamp()
                                           ).astimezone().isoformat(timespec="seconds")


def iso_or_none(ts):
    if ts is None:
        return None
    return datetime.datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


class AccountKeyError(RuntimeError):
    """The account's upstream key could not be resolved."""


def _blank_state():
    return {
        "status": OK,
        "blacklist_reason": None,
        "blacklist_until": None,
        "last_error": None,
        "key_hint": None,
    }


class Pool(object):
    def __init__(self, cfg_manager, secrets, logger=None, now_fn=None):
        self.cfgm = cfg_manager
        self.secrets = secrets
        self.log = logger
        self.now = now_fn or time.time
        self._lock = threading.RLock()
        self._state = {}            # account name -> in-memory state dict

    # ---------------- config / state helpers ----------------

    def cfg(self):
        return self.cfgm.get()

    def _st(self, name):
        """State dict of one account, created on first sight.

        Takes the (reentrant) lock itself so the invariant "state is only
        touched under the lock" cannot be broken by a caller.
        """
        with self._lock:
            st = self._state.get(name)
            if st is None:
                st = _blank_state()
                self._state[name] = st
            return st

    def state_for(self, name):
        """Public read access to one account's in-memory state (for views)."""
        with self._lock:
            return dict(self._st(name))

    def _log(self, level, msg, *args):
        if self.log:
            getattr(self.log, level)(msg, *args)

    def _blacklisted(self, st, now):
        return (st.get("status") == BLACKLISTED
                and bool(st.get("blacklist_until"))
                and st["blacklist_until"] > now)

    # ---------------- lazy recovery ----------------

    def refresh(self):
        """Expire blacklists whose deadline has passed (one time comparison).

        Called at the start of every request; there is no background thread and
        nothing is persisted, so the first real request after expiry is the probe.
        """
        cfg = self.cfg()
        now = self.now()
        changed = False
        with self._lock:
            for acct in cfg.accounts:
                st = self._st(acct.name)
                until = st.get("blacklist_until")
                if st.get("status") == BLACKLISTED and until and until <= now:
                    reason = st.get("blacklist_reason")
                    st["status"] = OK
                    st["blacklist_reason"] = None
                    st["blacklist_until"] = None
                    changed = True
                    self._log("info", "ACCOUNT_RECOVERED account=%s blacklist "
                                      "expired (reason was %s), next request is "
                                      "a probe", acct.name, reason)
        return changed

    # ---------------- candidate selection ----------------

    def candidates(self, model):
        """(available accounts in rotation order, {name: reason} skipped)."""
        cfg = self.cfg()
        now = self.now()
        self.refresh()
        available = []
        skipped = {}
        with self._lock:
            for acct in cfg.ordered():
                st = self._st(acct.name)
                if model and acct.upstream_model(model) is None:
                    skipped[acct.name] = "no_mapping"
                    continue
                if self._blacklisted(st, now):
                    skipped[acct.name] = st.get("blacklist_reason") or BLACKLISTED
                    continue
                available.append(acct)
        return available, skipped

    def key_for(self, acct):
        """Plaintext upstream key (memory only); blacklists on failure."""
        try:
            value = self.secrets.resolve(acct.key_env)
        except Exception as e:
            detail = self.secrets.redact(str(e))
            self.mark_failure(acct, R_KEY_ERROR, detail=detail)
            self._log("error", "KEY_ERROR account=%s env=%s: %s",
                      acct.name, acct.key_env, detail)
            raise AccountKeyError(str(e))
        with self._lock:
            st = self._st(acct.name)
            st["key_hint"] = self.secrets.mask(value)
            if st.get("status") == BLACKLISTED and \
                    st.get("blacklist_reason") == R_KEY_ERROR:
                # a resolvable key clears that blacklist. The environment is
                # normally fixed for the process lifetime, so in practice an
                # unresolvable name is a restart-fix (the launcher's require_env
                # gate is what should have refused the start).
                st["status"] = OK
                st["blacklist_reason"] = None
                st["blacklist_until"] = None
        return value

    # ---------------- state transitions ----------------

    def mark_exhausted(self, acct, detail=None, http_status=None):
        """Quota/arrears class -> blacklist for `blacklist_exhausted` seconds."""
        seconds = self.cfg().blacklist_exhausted
        until = self._blacklist(acct, R_EXHAUSTED, seconds, detail)
        self._log("warning", "ACCOUNT_EXHAUSTED account=%s status=%s until=%s "
                             "blacklist_s=%s reason=%s evidence=%s", acct.name,
                  http_status, iso_or_none(until), round(float(seconds)),
                  R_EXHAUSTED, self.secrets.redact(detail or ""))

    def mark_failure(self, acct, reason, detail=None, http_status=None):
        """Transient / credential class -> blacklist for `blacklist_failure` seconds."""
        seconds = self.cfg().blacklist_failure
        until = self._blacklist(acct, reason, seconds, detail)
        self._log("warning", "ACCOUNT_BLACKLISTED account=%s status=%s until=%s "
                             "blacklist_s=%s reason=%s evidence=%s", acct.name,
                  http_status, iso_or_none(until), round(float(seconds)), reason,
                  self.secrets.redact(detail or ""))

    def _blacklist(self, acct, reason, seconds, detail):
        """State transition only -> the `blacklist_until` epoch (callers log)."""
        until = self.now() + float(seconds)
        with self._lock:
            st = self._st(acct.name)
            st["status"] = BLACKLISTED
            st["blacklist_reason"] = reason
            st["blacklist_until"] = until
            st["last_error"] = self.secrets.redact(detail or reason)[:MAX_ERROR_LEN]
        return until

    def record_failure(self, acct, detail=None, http_status=None):
        """A failure that is NOT the account's fault: no blacklist, last_error only."""
        with self._lock:
            st = self._st(acct.name)
            st["last_error"] = self.secrets.redact(
                detail or "error")[:MAX_ERROR_LEN]

    def record_ok(self, acct, key_hint=None):
        """A successful call clears any blacklist on that account."""
        with self._lock:
            st = self._st(acct.name)
            st["status"] = OK
            st["blacklist_reason"] = None
            st["blacklist_until"] = None
            st["last_error"] = None
            if key_hint:
                st["key_hint"] = key_hint

    # ---------------- views ----------------

    def pool_status(self):
        cfg = self.cfg()
        now = self.now()
        total = len(cfg.accounts)
        unavailable = 0
        with self._lock:
            for acct in cfg.accounts:
                if self._blacklisted(self._st(acct.name), now):
                    unavailable += 1
        if total and unavailable == total:
            return "all_exhausted"
        if unavailable:
            return "degraded"
        return "ok"

    def health_view(self, uptime_s=None):
        """Unauthenticated probe output: liveness + pool summary ONLY.

        Deliberately carries no key, no token, no account names and no usage
        figures (a liveness probe sends no credential), and no version of its
        own: the service version face belongs to the deployment's service
        manager, outside this repo.
        """
        cfg = self.cfg()
        now = self.now()
        available = 0
        with self._lock:
            for acct in cfg.accounts:
                if not self._blacklisted(self._st(acct.name), now):
                    available += 1
        out = {
            "status": "ok",
            "service": "llm-router",
            "pool": self.pool_status(),
            "accounts_total": len(cfg.accounts),
            "accounts_available": available,
            "time": iso_or_none(now),
        }
        if uptime_s is not None:
            out["uptime_s"] = round(uptime_s, 1)
        return out

    def models_view(self):
        """GET /v1/models — local aggregation, never forwarded upstream."""
        cfg = self.cfg()
        self.refresh()
        now = self.now()
        data = []
        for model in cfg.model_names():
            accounts = []
            with self._lock:
                ordered = cfg.ordered()
                states = dict((a.name, dict(self._st(a.name))) for a in ordered)
            for acct in ordered:
                up = acct.upstream_model(model)
                if up is None:
                    accounts.append({"name": acct.name, "available": False,
                                     "upstream_model": None, "reason": "no_mapping"})
                    continue
                st = states[acct.name]
                if self._blacklisted(st, now):
                    reason = st.get("blacklist_reason") or BLACKLISTED
                    entry = {"name": acct.name, "available": False,
                             "upstream_model": up,
                             "reason": VIEW_REASON.get(reason, "throttled"),
                             "until": iso_or_none(st.get("blacklist_until"))}
                else:
                    entry = {"name": acct.name, "available": True,
                             "upstream_model": up, "reason": "ok"}
                accounts.append(entry)
            data.append({
                "id": model,
                "object": "model",
                "created": int(now),
                "owned_by": "llm-router",
                "accounts": accounts,
                "available": any(a["available"] for a in accounts),
            })
        return {"object": "list", "data": data}
