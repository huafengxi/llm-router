#!/usr/bin/env python3
"""probe_models.py — per-account upstream model availability probe.

Maintenance tool, read-only: a standalone entry point that is never imported by
router.py / server.py and never touches the router's listening port.  Its only
input is the account table (accounts.yml, read-only -- this script never writes
it), so changing it is not a service-code change: no restart and no service
version bump.

When to run it: before repointing a role name (an `accounts[].models:` value) at
a different upstream model, and after an account's quota recovers and its
mappings need topping up.  Accounts rarely expose the same set of upstream model
names, so a blind repoint silently drops an account out of the candidate set
(`/v1/models` reports `reason: no_mapping` and nothing warns).

Read-only against the upstreams of the pool.  Credentials: like router.py, this
probe reads them **from its own environment by name** (accounts.yml `key:` is an
environment-variable name) -- it spawns no decryptor and reads no credential
file, so run it with the deployment's credentials injected, e.g.

    set -a; eval "$(<the deployment's decrypt-or-export step>)"; set +a
    python3 probe_models.py one ACCT MODEL

The key lives only in a python str and is injected into an Authorization header
built in memory.  It is NEVER passed as argv, NEVER written to any file, NEVER
printed (every persisted/echoed body goes through scrub()).

Usage:
  probe_models.py models [acct[,acct...]]    GET {base_url}/models
  probe_models.py probe PLAN.json            chat/completions probes per plan
                                             ({account: [model, ...]})
  probe_models.py one ACCOUNT MODEL          single ad-hoc probe cell

ACCOUNTS below is DERIVED from accounts.yml (the single source of the
pool: name / base_url / key env-var name, list order = rotation order) by
_load_accounts() -- a minimal line scanner, because this script is stdlib-only
and must not grow a PyYAML dependency.  Adding a seat to accounts.yml therefore
makes it probeable with no edit here (no second table to keep in sync).  The
scanner is deliberately LOUD: an unreadable file, a missing `accounts:` key, a
zero-account result, an account missing name/base_url/key, a duplicate
name, or an unexpected indentation all exit non-zero with the reason -- it never
returns a shorter table, because a silently missing account would make that seat
unprobeable and read as "no such account".

stdlib only; python 3.8 syntax floor.  Serial, >=1.1s between calls.
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("LLM_ROUTER_PROBE_OUT") or os.path.join(HERE, ".probe-out")

ACCOUNTS_YML = os.environ.get("LLM_ROUTER_ACCOUNTS") or os.path.join(HERE, "accounts.yml")
# `key: <ENV_VAR_NAME>` -- the only credential shape accounts.yml allows (a mapping
# is the retired file-pointer form and a literal value is refused by config.py
# before it gets here).
KEY_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _loud(path, lineno, why):
    """Fail loudly: never fall back to a shorter (or empty) account table."""
    raise SystemExit("probe_models.py: cannot derive ACCOUNTS from %s:%d -- %s"
                     % (path, lineno, why))


def _scalar(text):
    """Drop a trailing `# ...` comment (only after whitespace) and the quotes."""
    cut, i = text, 0
    while True:
        j = cut.find("#", i)
        if j < 0:
            break
        if j == 0 or cut[j - 1] in " \t":
            cut = cut[:j]
            break
        i = j + 1
    return cut.strip().strip('"').strip("'")


def _load_accounts(path):
    """-> [(name, base_url, key_env), ...] in accounts.yml list order.

    Minimal scanner for the three fields this probe needs; expects the file's
    present shape (top-level `accounts:`, account items at 2 spaces, their
    `base_url:` / `key:` at 4 spaces).  Any other shape exits non-zero.
    """
    try:
        with open(path) as fh:
            lines = fh.read().splitlines()
    except OSError as exc:
        raise SystemExit("probe_models.py: cannot read %s: %s" % (path, exc))
    out, cur, in_accounts, seen_marker, items = [], None, False, False, 0
    dash_name_re = re.compile(r"^\s*-\s*name:")
    for lineno, raw in enumerate(lines, 1):
        body = raw.strip()
        if not body or body.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if indent == 0:                       # a top-level key ends `accounts:`
            in_accounts = body.startswith("accounts:")
            seen_marker = seen_marker or in_accounts
            cur = None
            continue
        if not in_accounts:
            continue
        if dash_name_re.match(raw):
            items += 1                      # any-indent `- name:` = one account item
        if indent == 2 and body.startswith("- name:"):
            name = _scalar(body[len("- name:"):])
            if not name:
                _loud(path, lineno, "empty account name")
            cur = {"name": name, "line": lineno,
                   "base_url": None, "key": None}
            out.append(cur)
            continue
        if cur is None:
            _loud(path, lineno, "account field before any `- name:` line")
        if indent != 4:
            continue                          # models: mappings and wrapped comments
        if body.startswith("base_url:"):
            cur["base_url"] = _scalar(body[len("base_url:"):])
        elif body.startswith("key:"):
            cur["key"] = _scalar(body[len("key:"):])
            if not KEY_NAME_RE.match(cur["key"] or ""):
                _loud(path, lineno,
                      "key: is not an environment-variable name ([A-Za-z_][A-Za-z0-9_]*)")
    if not seen_marker:
        _loud(path, 0, "no top-level `accounts:` key found")
    if not out:
        _loud(path, 0, "`accounts:` holds no account block")
    if items != len(out):
        _loud(path, 0, "parsed %d of %d `- name:` items (unexpected indentation?)"
                     % (len(out), items))
    names = set()
    for cur in out:
        for field in ("base_url", "key"):
            if not cur[field]:
                _loud(path, cur["line"], "account %s has no %s" % (cur["name"], field))
        if cur["name"] in names:
            _loud(path, cur["line"], "duplicate account name %s" % cur["name"])
        names.add(cur["name"])
    return [(a["name"], a["base_url"], a["key"]) for a in out]


ACCOUNTS = _load_accounts(ACCOUNTS_YML)
BY_NAME = {a[0]: a for a in ACCOUNTS}

SLEEP = 1.1          # >= 1s between calls (task cost discipline)
TIMEOUT = 45

# --- quota / throttle wording, mirroring llm-router/classify.py (read-only ref) ---
sys.path.insert(0, HERE)
try:
    import classify as ref_classify          # noqa: E402
    EXHAUSTED_PHRASES = ref_classify.EXHAUSTED_PHRASES
    QUOTA_TOKENS = ref_classify.QUOTA_TOKENS
    THROTTLE_STRONG = ref_classify.THROTTLE_STRONG_PHRASES
    EXHAUSTED_WORDS = ref_classify.EXHAUSTED_WORDS
except Exception:                            # pragma: no cover - fallback
    EXHAUSTED_PHRASES = ("quota exceeded", "insufficient_quota", "arrearage",
                         "额度不足", "欠费", "余额不足")
    QUOTA_TOKENS = ("quota", "credits", "credit", "额度", "arrearage", "欠费")
    THROTTLE_STRONG = ("rate limit exceeded", "too many requests", "requests per minute",
                       "限流", "请求过于频繁")
    EXHAUSTED_WORDS = ("exhausted", "depleted", "used up", "arrearage", "欠费", "耗尽")

NO_MODEL_HINTS = ("model not found", "model_not_found", "invalid model", "invalid_model",
                  "does not exist", "not exist", "no such model", "unknown model",
                  "unsupported model", "模型不存在", "不支持的模型", "not supported",
                  "invalidparameter", "invalid parameter", "invalid_parameter")
AUTH_HINTS = ("invalid api key", "invalid_api_key", "invalidapikey", "incorrect api key",
              "unauthorized", "unauthenticated", "permission denied", "access denied",
              "no permission", "无权", "鉴权失败")
# --- model-level entitlement wording (bucket `not_purchased`, mirroring a real
# --- upstream body: 403 {"code":"AccessDenied.Unpurchased", "message":"Access to
# --- model denied. Please make sure you are eligible for using the model."}).
# This is a fact about ONE model on this seat, not an account credential fault
# and not spent quota => it gets its own bucket and never skips the account.
NOT_PURCHASED_HINTS = ("accessdenied.unpurchased", "unpurchased", "not purchased",
                       "hasn't purchased", "haven't purchased", "has not been purchased",
                       "eligible for using the model", "未购买", "未开通")


def get_key(name):
    """Value of the environment variable `name`. Nothing written, nothing echoed."""
    value = os.environ.get(name)
    if not value:
        raise SystemExit("credential env var %s is not set (or empty) -- inject the "
                         "deployment's credentials into this probe's environment"
                         % name)
    return value


def scrub(text, key):
    """Strip any accidental key occurrence and cap the excerpt length."""
    if text is None:
        return ""
    text = str(text)
    if key:
        text = text.replace(key, "<key-redacted>")
    return " ".join(text.split())


def request(url, key, body=None):
    """-> (status, text). Network errors -> (0, 'EXC ...')."""
    data = None
    headers = {"Authorization": "Bearer " + key, "Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        try:
            payload = exc.read().decode("utf-8", "replace")
        except Exception as exc2:             # pragma: no cover
            payload = "<unreadable body: %s>" % exc2
        return exc.code, payload
    except Exception as exc:
        return 0, "EXC %s: %s" % (type(exc).__name__, exc)


def bucket(status, body):
    """Task judgement archetypes. Status alone never decides (classify.py rule)."""
    low = (body or "").lower()
    if 200 <= status < 300:
        return "ok"
    for phrase in EXHAUSTED_PHRASES:
        if phrase in low:
            return "exhausted"
    for phrase in THROTTLE_STRONG:
        if phrase in low and not any(w in low for w in EXHAUSTED_WORDS):
            return "throttled"
    if status in (402, 429, 403) and any(t in low for t in QUOTA_TOKENS):
        return "exhausted"
    if status == 429:
        return "throttled"
    if status != 401 and any(h in low for h in NOT_PURCHASED_HINTS):
        return "not_purchased"      # 403 AccessDenied.Unpurchased: model-level,
                                    # NOT an account fault => no account-level skip
    if status in (401, 403) or any(h in low for h in AUTH_HINTS):
        return "auth"
    if any(h in low for h in NO_MODEL_HINTS):
        return "no_model"
    if status in (404, 405):          # 404 on /chat/completions = unknown route/model
        return "no_model"
    return "other"                    # 400/422/5xx without model wording: not a verdict


def append_result(rec):
    os.makedirs(OUT, exist_ok=True)
    """Persist immediately (a truncated run still leaves its evidence)."""
    with open(os.path.join(OUT, "results.jsonl"), "a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def do_models(names):
    for name in names:
        acc, base, key_env = BY_NAME[name]
        key = get_key(key_env)
        status, body = request(base.rstrip("/") + "/models", key)
        ids = []
        try:
            data = json.loads(body)
            seq = data.get("data", []) if isinstance(data, dict) else data
            for item in seq or []:
                if isinstance(item, dict) and item.get("id"):
                    ids.append(item["id"])
                elif isinstance(item, str):
                    ids.append(item)
        except Exception:
            pass
        rec = {"account": name, "kind": "models", "model": None, "status": status,
               "bucket": "ok" if 200 <= status < 300 else "other",
               "n_ids": len(ids), "ids": ids,
               "evidence": scrub(body, key)[:400] if not ids else ""}
        append_result(rec)
        with open(os.path.join(OUT, "catalog_%s.json" % name), "w") as fh:
            json.dump({"account": name, "status": status, "n_ids": len(ids), "ids": ids,
                       "body_excerpt": scrub(body, key)[:600]}, fh,
                      ensure_ascii=False, indent=1)
        sys.stdout.write("== %s status=%s n_ids=%d\n" % (name, status, len(ids)))
        for mid in ids:
            sys.stdout.write("   %s\n" % mid)
        if not ids:
            sys.stdout.write("   <no ids> %s\n" % rec["evidence"][:300])
        sys.stdout.flush()
        time.sleep(SLEEP)


def do_probe(plan):
    total = 0
    for name, models in plan.items():
        acc, base, key_env = BY_NAME[name]
        key = get_key(key_env)
        skipped = None
        for model in models:
            if skipped:
                rec = {"account": name, "kind": "chat", "model": model, "status": None,
                       "bucket": "not_probed",
                       "evidence": "skipped: account already %s this run" % skipped}
                append_result(rec)
                sys.stdout.write("%-10s %-30s -- not_probed (%s)\n" % (name, model, skipped))
                sys.stdout.flush()
                continue
            body = {"model": model, "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1, "stream": False}
            status, text = request(base.rstrip("/") + "/chat/completions", key, body)
            bkt = bucket(status, text)
            rec = {"account": name, "kind": "chat", "model": model, "status": status,
                   "bucket": bkt, "evidence": scrub(text, key)[:400]}
            append_result(rec)
            total += 1
            sys.stdout.write("%-10s %-30s status=%-5s %s\n" % (name, model, status, bkt))
            sys.stdout.flush()
            # account-level skip: only the classes that are a fact about the
            # ACCOUNT (spent quota / rate limit / refused credential).  Per-model
            # facts -- not_purchased, no_model, other, ok -- never skip the rest.
            if bkt in ("exhausted", "throttled", "auth"):
                skipped = bkt           # no retry, skip the rest of this account
            time.sleep(SLEEP)
        sys.stdout.write("# chat calls so far: %d\n" % total)
        sys.stdout.flush()
    sys.stdout.write("TOTAL chat calls this run: %d\n" % total)


USAGE = """usage:
  probe_models.py models [acct[,acct...]]   GET {base_url}/models（目录面；不给名字 = ACCOUNTS 里全部已声明账号）
  probe_models.py probe PLAN.json           按 plan 逐账号 chat/completions 实探（max_tokens=1）
  probe_models.py one ACCT MODEL            单笔实探
账号名 = %s
凭据 = 本进程环境里按名取（accounts.yml 的 `key:` 就是环境变量名；不 spawn 解密器、
不读凭据文件 ⇒ 跑之前要先把部署面的凭据注入环境）；key 只进 str、不进 argv、
不落盘、不打印。产物落 %s（自动建目录，gitignored）。

注意：本探针把模型名原样发给上游，不套用 accounts.yml 的 `models:` 映射 ⇒
用池角色档名（accounts.yml 里 `models:` 的那些别名）裸探恒得上游 404 `model_not_found`，
那不代表席位缺陷；要探角色档请用它在 `models:` 里映射到的上游名。
归类（bucket）取值 = ok / exhausted / throttled / auth / not_purchased /
no_model / other / not_probed；其中 exhausted·throttled·auth 是账号级事实 ⇒
触发「该账号其余模型记 not_probed」的 skip，其余各类是模型级事实、不 skip。
"""


def main():
    if len(sys.argv) < 2:
        # 缺省不默认跑全量 models：那是 5 次真实上游调用（成本面），必须显式给 mode
        sys.stdout.write(USAGE % (", ".join(a[0] for a in ACCOUNTS), OUT))
        return
    mode = sys.argv[1]
    if mode == "models":
        names = (sys.argv[2].split(",") if len(sys.argv) > 2
                 else [a[0] for a in ACCOUNTS])
        for n in names:
            if n not in BY_NAME:
                raise SystemExit("unknown account " + n)
        do_models(names)
    elif mode == "probe":
        with open(sys.argv[2]) as fh:
            plan = json.load(fh)
        for n in plan:
            if n not in BY_NAME:
                raise SystemExit("unknown account " + n)
        do_probe(plan)
    elif mode == "one":
        name, model = sys.argv[2], sys.argv[3]
        acc, base, key_env = BY_NAME[name]
        key = get_key(key_env)
        body = {"model": model, "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1, "stream": False}
        status, text = request(base.rstrip("/") + "/chat/completions", key, body)
        rec = {"account": name, "kind": "chat", "model": model, "status": status,
               "bucket": bucket(status, text), "evidence": scrub(text, key)[:400]}
        append_result(rec)
        sys.stdout.write(json.dumps(rec, ensure_ascii=False)[:500] + "\n")
    else:
        raise SystemExit("mode must be models|probe|one")


if __name__ == "__main__":
    main()
