"""classify.py — (HTTP status, upstream error text) -> retry semantics.

Pure functions, no I/O.  The status code is NEVER the sole criterion: quota
exhaustion is decided by the error wording first, and 402/429/403 only count as
exhaustion when the body also carries quota semantics — otherwise a plain
per-minute throttle would be booked as the long quota-exhaustion class.

Categories (the caller, server.py, turns them into a blacklist duration and a
switch decision; this module only classifies):
  exhausted     account quota is spent, or the account is not entitled to a
                model it maps -> the long blacklist, switch account
  throttled     rate limit -> the short blacklist, switch account
  server_error  5xx / network / timeout, and an upstream 404 for a pool model
                this account maps -> the short blacklist, switch account
  client_error  4xx parameter & auth errors -> the caller decides: a 401 (the
                upstream refused our key) is the account's fault and gets the
                short blacklist + a switch, every other 4xx is returned to the
                client unchanged and never blacklists.  A 404 lands here only
                when the caller did NOT map a pool model for this account, i.e.
                the request named a path (or no model at all) that the upstream
                does not serve — that is the request's own shape, so trying
                another account would repeat it
  unknown       anything else -> treated like server_error

Two 4xx classes are split by ROUTING CONTEXT, not by body wording: the router
only offers a request to an account whose `models:` covers the pool name, so a
404 — or a 403 denying entitlement to that very model — from that upstream means
the account cannot serve a model it claims to serve (its mapping drifted, or its
plan changed / lapsed) — an account-side fault.  Leaving them on the client side
instead pins the whole pool to the one account that answers them: no switch, no
blacklist, every request for that pool name lost while the healthy seats idle.
The two classes differ only in how long the state lasts: a drifted mapping is
fixed by a config reload (short tier), a lapsed entitlement lasts until the plan
changes again (long tier, like spent quota).

Python 3.8 syntax floor; stdlib only.
"""

# --- quota-exhaustion wording (matched case-insensitively, in this order) ---
EXHAUSTED_PHRASES = (
    "allocated quota exceeded",
    "free allocated quota exceeded",
    "you exceeded your current quota",
    "exceeded your current plans",
    "the free tier of the model has been exhausted",
    "free tier of the model has been exhausted",
    "insufficient_quota",
    "quota exhausted",
    "quota exceeded",
    "credits exhausted",
    "insufficient balance",
    "insufficient credits",
    "arrearage",
    "额度不足",
    "额度已用尽",
    "额度耗尽",
    "余额不足",
    "欠费",
)

# --- the double-condition tokens for status 402/429/403 ---
QUOTA_TOKENS = ("quota", "credits", "credit", "额度", "arrearage", "欠费")

# --- rate-limit wording, tier 1 (specific enough to outrank the 402/429/403 +
# --- quota-token double condition below)
# Aliyun answers BOTH per-minute rate limits and spent quota with a
# `Throttling.*Quota` code, so the bare word "quota" in a body cannot decide
# anything on its own: the specific code identifiers and the explicit
# per-minute wording are matched first, and only a body that carries quota
# semantics WITHOUT any of them falls through to the double condition.
THROTTLE_STRONG_PHRASES = (
    "requests rate limit exceeded",
    "requests per minute",
    "tokens per minute",
    "rate limit exceeded",
    "too many requests",
    "limit_requests",
    "throttling.ratequota",
    "ratequota",
    "请求过于频繁",
    "限流",
)

# --- rate-limit wording, tier 2 (weak hints: consulted only after the quota
# --- double condition, because "throttl" also appears in exhaustion codes)
THROTTLE_WEAK_PHRASES = (
    "throttl",
    "rate limit",
    "rate-limit",
    "ratelimit",
    "concurrency",
)

THROTTLE_PHRASES = THROTTLE_STRONG_PHRASES + THROTTLE_WEAK_PHRASES

# words that flip a rate-limit-looking body back to exhaustion
EXHAUSTED_WORDS = (
    "exhausted", "depleted", "used up", "arrearage",
    "欠费", "余额不足", "用尽", "耗尽",
)

EXHAUSTED_STATUS = (402, 429, 403)
CLIENT_ERROR_STATUS = (400, 401, 403, 404, 405, 413, 415, 422)

# --- model-entitlement denial: the seat's plan does not cover a model it maps.
# Consulted ONLY together with the caller's routing context (`mapped_model`) and
# never for a 401 (that class is about our credential, not about the plan), so a
# request-level 403 keeps its client-side meaning.  Wording collected from a real
# upstream; the family is about entitlement, not about a spent balance, so no
# quota token appears in it.
MODEL_ACCESS_PHRASES = (
    "unpurchased",
    "access to model denied",
    "eligible for using the model",
    "没有使用该模型的权限",
    "模型未开通",
)

EXHAUSTED = "exhausted"
THROTTLED = "throttled"
SERVER_ERROR = "server_error"
CLIENT_ERROR = "client_error"
UNKNOWN = "unknown"

# network-level exception types that mean "this account could not be reached"
NETWORK_HINTS = ("timeout", "timed out", "refused", "reset", "eof", "broken",
                 "unreachable", "name or service not known", "temporarily",
                 "ssl", "certificate", "incomplete", "closed")


def _low(text):
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            text = text.decode("utf-8", "replace")
        except AttributeError:
            text = str(text)
    return text.lower()


def has_quota_semantics(text):
    """True when the text carries quota-exhaustion wording (any status)."""
    low = _low(text)
    if not low:
        return False
    return _has(low, EXHAUSTED_PHRASES) is not None


def exhausted_text_match(text):
    """The matched phrase (for logs), or None."""
    return _has(_low(text), EXHAUSTED_PHRASES)


def _has(text, phrases):
    for phrase in phrases:
        if phrase in text:
            return phrase
    return None


def classify(status, body, mapped_model=False):
    """-> (category, evidence).  `evidence` is a short, log-safe string.

    `mapped_model` is routing context from the caller: True when this account's
    `models:` mapping covers the pool model of the request (the account claimed
    to serve it), False when the request named no model / the account maps none.
    It decides two classes: a 404 (mapping drift -> server_error) and a
    model-entitlement denial (plan lapse -> exhausted).

    Precedence (the status code alone never decides):
      1 explicit quota-exhaustion wording            -> exhausted
      2 explicit rate-limit wording, no exhaustion word -> throttled
      3 status 402/429/403 + a quota token           -> exhausted
      4 weak rate-limit hint                         -> throttled
      5 bare 429                                     -> throttled
      6 5xx                                          -> server_error
      7 404 + mapped_model                           -> server_error
      8 entitlement-denial wording + mapped_model,
        not a 401                                    -> exhausted
      9 other 4xx / anything else
    """
    low = _low(body)
    hit = _has(low, EXHAUSTED_PHRASES)
    if hit:
        return EXHAUSTED, "phrase=%r" % hit
    hit = _has(low, THROTTLE_STRONG_PHRASES)
    if hit and not _has(low, EXHAUSTED_WORDS):
        return THROTTLED, "phrase=%r" % hit
    try:
        code = int(status)
    except (TypeError, ValueError):
        code = 0
    if code in EXHAUSTED_STATUS:
        token = _has(low, QUOTA_TOKENS)
        if token:
            return EXHAUSTED, "status=%d token=%r" % (code, token)
    hit = _has(low, THROTTLE_WEAK_PHRASES)
    if hit and not _has(low, EXHAUSTED_WORDS):
        return THROTTLED, "phrase=%r" % hit
    if code == 429:
        return THROTTLED, "status=429 (no quota semantics)"
    if code >= 500:
        return SERVER_ERROR, "status=%d" % code
    if code == 404 and mapped_model:
        return SERVER_ERROR, ("mapped_model=True (this account maps the pool "
                              "model, so the upstream 404 is its own fault)")
    if code != 401 and mapped_model:
        denied = _has(low, MODEL_ACCESS_PHRASES)
        if denied:
            return EXHAUSTED, ("status=%d mapped_model=True phrase=%r (this "
                               "account maps the pool model, so the upstream "
                               "denies its own entitlement to it)"
                               % (code, denied))
    if code in CLIENT_ERROR_STATUS:
        return CLIENT_ERROR, "status=%d" % code
    if code:
        return UNKNOWN, "status=%d" % code
    return UNKNOWN, "no status"


def classify_exception(exc):
    """Network/timeout failures -> (category, evidence).  Never `exhausted`."""
    text = "%s: %s" % (type(exc).__name__, exc)
    low = _low(text)
    if has_quota_semantics(low):        # e.g. an SSL body is never quota text
        return EXHAUSTED, text[:200]
    for hint in NETWORK_HINTS:
        if hint in low:
            return SERVER_ERROR, text[:200]
    return SERVER_ERROR, text[:200]
