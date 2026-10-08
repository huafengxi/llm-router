# llm-router

An OpenAI-compatible reverse proxy that aggregates several **independently
quota'd** upstream accounts behind one endpoint: a request walks the account
list in its writing order, and when the upstream signals that the current
account is at fault (spent quota, a lapsed entitlement to a model it maps, rate
limit, 5xx, refused credential, empty stream, a 404 for a model it maps) the
**same request** is retried on the next account. Clients see one
base URL and one model namespace; the pool absorbs the account-level failures.

Protocol passthrough only — no protocol translation, no request rewriting
except the model-name mapping and the `stream_options` injection described
below. All pool state is process memory: nothing is persisted, so a restart
resets every blacklist.

Python 3.8+ syntax floor, stdlib + PyYAML only.

## Quick start

```bash
pip install pyyaml
cp accounts.example.yml accounts.yml      # then edit: base_url, models, key env-var names
export ROUTER_TOKEN=... ACME_API_KEY=... ACME_TEAM_API_KEY=... ACME_METERED_API_KEY=...
python3 router.py --host 127.0.0.1 --port 9200
curl -s localhost:9200/health | python3 -m json.tool
curl -s localhost:9200/v1/chat/completions \
     -H "Authorization: Bearer $ROUTER_TOKEN" -H 'Content-Type: application/json' \
     -d '{"model":"fast","messages":[{"role":"user","content":"hi"}]}'
```

`router.py` resolves every path from its own location (never from the cwd), so a
service manager needs no working-directory setting.

## Configuration

`accounts.yml` is the single configuration face and is **hot-reloaded on mtime
change**: adding/removing an account, reordering the rotation, retuning the
blacklist windows or the timeouts all take effect without a restart. A reload
that fails validation is **refused** — the previous configuration stays in
effect and a `CONFIG_REJECTED` line is logged. `accounts.example.yml` is the
annotated reference shape.

Structure: `defaults:` (two blacklist windows, `timeout: {connect, read}`,
`inject_stream_options`), `auth:` (inbound bearer token env-var name + exempt
paths), `accounts:` (the pool, in rotation order). Per-account keys: `name`,
`base_url`, `key: <ENV_VAR_NAME>`, `models: {<pool name>: <upstream name>}`,
optional `timeout:` / `ssl_verify:`.

**Rotation order = writing order.** `order` is derived from the list position
(1-based); there is no numeric field to edit and no local quota face. Move a
block up to try that account earlier. The retired keys `priority` / `quota` /
`credits_weights` are rejected outright, so a stale config file cannot quietly
change the rotation order.

**Model names are mapped, not aliased.** `models:` maps the name clients ask for
to the name *that* upstream expects; an account is a candidate for a pool name
only if it maps it (otherwise `/v1/models` reports `reason: no_mapping`). The
same upstream model may be pointed at by any number of pool names — that is how
role-style names (`planner`, `executor`, `utility`, …) are expressed: they are
ordinary pool names, there is no separate alias mechanism.

## Credentials

**No credential ever appears in this file, and this repo never reads one from
disk.** A credential is an **environment variable, referenced by name**
(`key: ACME_API_KEY`, `auth: {token: ROUTER_TOKEN}`) and `secrets.py` — the
single choke point — looks it up in the router's own environment. How a
deployment obtains those values is deliberately not this package's business: it
spawns no decryptor, knows no cipher format and opens no credential file, so a
plaintext env, an inline-encrypted file decrypted by the launcher, a secret
store or a shell wrapper all work unchanged.

A reference that is not an identifier is refused at load: the retired
`{env_file:, var:}` mapping fails with a message that names the new form, and
anything else (a pasted `sk-…` value) fails the identifier shape check — so a
literal credential cannot be committed by accident.

Two consequences of reading the environment instead of a file:

- **Inject at spawn, and gate the start there.** The router logs a masked hint
  per account at startup (`ACCOUNT … key=sk-****abcd`) and a `KEY_ERROR` line for
  a name it cannot resolve, but the fail-loud gate belongs to the launcher (a
  `require_env`-style list): a service must not come up without its credentials.
- **Rotating a credential is a restart, not a hot reload.** The environment is
  fixed for the process lifetime, so there is no TTL cache and nothing to
  invalidate; editing `accounts.yml` stays hot-reloaded, editing a *value* does
  not.

Plaintext values live only in the process environment and in `secrets.py`'s
redaction registry; everything that leaves the process (log lines, response
bodies) passes through `mask()`/`redact()` first. After the refactor the router
spawns **no subprocess at all**, so no child inherits the credentials either.

| Environment variable | Meaning | Default |
|---|---|---|
| `LLM_ROUTER_ACCOUNTS` | pool configuration path | `<repo>/accounts.yml` |
| `LLM_ROUTER_PROBE_OUT` | `probe_models.py` artifact directory | `<repo>/.probe-out` |
| `LLM_ROUTER_TEST_TMP` | test scratch root | `<repo>/.test-tmp` |

…plus one variable per credential named in `accounts.yml` (`probe_models.py`
reads the same names from its own environment).

CLI: `--host` (default `127.0.0.1` — loopback only), `--port` (9200),
`--accounts`, `--log-level`.

## Inbound auth

Every endpoint except `auth.exempt_paths` (default `/health`) requires
`Authorization: Bearer <token>`, compared in constant time
(`hmac.compare_digest`); a missing, malformed or mismatched credential yields
**401** whose body never echoes the expected or the supplied value. The gate runs
**before** routing, so an unauthenticated request to any path — including a
nonexistent one — is 401, not 404.

`/health` stays unauthenticated so a liveness probe needs no credential; its
payload is therefore limited to liveness plus a pool summary (no key, no token,
no account names, no usage figures). Binding a non-loopback address makes the
router a credential-swapping proxy for everything that can reach the port: keep
the token strong and treat its disclosure as a rotation event.

## HTTP face

| Method & path | Semantics |
|---|---|
| `POST /v1/chat/completions` | main path: pick candidates from the body's `model`, map the model name, pass through; `stream: true` relays SSE |
| `GET /v1/models` | **aggregated locally** (not forwarded): the union of pool model names, each annotated per account with `available` / `upstream_model` / `reason` (`ok`\|`exhausted`\|`throttled`\|`key_error`\|`no_mapping`) and, while blacklisted, `until` |
| `GET /health` | liveness + pool summary (probe semantics: 200 even when the whole pool is unavailable — the state is in the body's `pool` field) |
| other `/v1/*` | generic passthrough with the same account selection (embeddings etc. do not 404); nothing rewritten except the model mapping and `stream_options` |
| anything else | 404 + JSON error (401 first when unauthenticated) |

The `reason` values above are a **folded** public enumeration: the two
credential classes (unresolvable key variable / upstream refusing our key) both
surface as `key_error`, and the short-window transient class (throttle, 5xx, network
error, unknown status) surfaces as `throttled` with the exact recovery instant
in `until`. The folding table is an explicit constant (`pool.py::VIEW_REASON`),
not a string coincidence.

## Account switching, blacklists, recovery

Per request: `body.model` → the accounts mapping it are the candidates → try
them in ascending `order`, skipping any still inside its blacklist window.

| Upstream outcome | internal reason | blacklist window | action for this request | log tag |
|---|---|---|---|---|
| 2xx | — | cleared | return to client | `REQ ok` |
| **quota exhausted** (semantic phrase family, or 402/429/403 *plus* quota semantics) | `exhausted` | `blacklist_exhausted` | **switch to the next candidate immediately** | `ACCOUNT_EXHAUSTED` + `EXHAUSTED_SWITCH` |
| **rate limited** (rate limit / rpm / tpm / too many requests, no quota semantics) | `throttled` | `blacklist_failure` | switch immediately (no same-account backoff) | `THROTTLED_SWITCH` |
| 5xx / unknown status | `server_error` | `blacklist_failure` | switch | `UPSTREAM_ERROR` |
| connection error / timeout | `server_error` | `blacklist_failure` | switch | `UPSTREAM_UNREACHABLE` |
| upstream **401** (our credential refused) | `upstream_key_rejected` | `blacklist_failure` | switch (the upstream 401 body is never leaked; if every account refuses, 503 `all_accounts_exhausted`) | `UPSTREAM_KEY_REJECTED` |
| key undecryptable | `key_error` | `blacklist_failure` | switch | `KEY_ERROR` |
| upstream **404** for a pool model **this account maps** (its mapping drifted, or its upstream plan changed) | `server_error` | `blacklist_failure` | switch — the account cannot serve what it claims to serve | `UPSTREAM_ERROR` (evidence `mapped_model=True`) |
| upstream **denial of entitlement** to a pool model **this account maps** (an `AccessDenied.Unpurchased`-family 403: the plan lapsed or never covered it) | `exhausted` | `blacklist_exhausted` | switch — the account's own plan state, which changes on the plan's clock, not on ours | `ACCOUNT_EXHAUSTED` + `EXHAUSTED_SWITCH` (evidence `mapped_model=True phrase=…`) |
| **2xx stream that never carries content nor `finish_reason`** (0 bytes, or role-only frames + `[DONE]`) | `server_error` | `blacklist_failure` | **switch** — decided by probing the stream head before the response is committed (`proxy.probe_stream_start`, 64KB cap); once a byte is written, no switch | `EMPTY_STREAM` / `STREAM_FIRST_BYTE_FAILED` |
| other 4xx (400/403/405/413/415/422, and a 404 — or an entitlement denial — for a request that named **no** model this account maps; a bare 402 without quota semantics falls to `unknown` and is handled like a 5xx) | — | **never blacklisted** (`last_error` only) | return to the client as-is, no switch (retrying would repeat the same error) | `CLIENT_ERROR` |

**A status code alone is never the judgement.** Some upstreams answer both rate
limiting and quota exhaustion with the same `Throttling.*Quota` family, so the
response body's semantics are read first and the status code is only used as a
second condition (table = `classify.py`; its `EXHAUSTED_PHRASES` / `QUOTA_TOKENS`
/ `MODEL_ACCESS_PHRASES` constants collect real upstream wording and are pinned
case-by-case in `test/test_router.py::T05ClassifyTable`).

**A 404 and an entitlement denial are split by routing context, not by
wording.** The router only offers a request to an account whose `models:` covers
the pool name, so a 404 from that upstream is the account's own fault — it cannot
serve what it claims to serve — and the request switches. The same holds when the
upstream denies entitlement to that very model (an `AccessDenied.Unpurchased`
family 403, which a plan boundary can produce after the spent-quota 429s stop):
the seat's plan state is the account's fault, so it takes the **long** exhaustion
tier — a plan changes on its own clock, and re-probing it every request would
only re-lose requests. A 404 (or such a 403) for a request that named no model is
about the request's own shape and goes back to the client unchanged, because
another account would answer it the same way; so does any other 403, which stays a
client error, and a 401, which stays the credential class whatever its body says.
Judging these apart by the upstream's phrasing alone would leave every wording not
yet invented on the client side: no switch and no blacklist, so the one account
answering it keeps receiving every request for that pool name and the rest of the
pool is never tried (`classify.classify(..., mapped_model=…)`, pinned by
`test/test_router.py::T27Upstream404` and `::T28Upstream403Entitlement`).

**Recovery is lazy and request-driven — there is no background thread.** Each
request starts with one "expired ⇒ back to `ok`" pass (`ACCOUNT_RECOVERED`); the
first real request after the deadline is the probe. Success clears the state,
failure re-blacklists per that attempt's judgement (no exponential backoff).
**Restart = full reset** (nothing is persisted), which is the shortest manual
remedy; to reset a single account, wait for its `until`.

**Whole pool unavailable**: `503` + `{"error":{"type": …, "accounts":[{name,
reason, until, recovers_in_s}]}}` + a `POOL_EXHAUSTED` ERROR line. `type` is
`all_accounts_exhausted` when every reason is quota/credential class,
`no_account_for_model` when all are `no_mapping`, otherwise `all_accounts_failed`
/ `all_accounts_unavailable` depending on whether a real failure was seen.

**Streaming idempotence boundary** = whether a byte has been written to the
client. Not written (upstream non-2xx, failure before the first byte, empty
stream) ⇒ switching accounts and retrying is safe. Written ⇒ no retry: an
`{"error":{"type":"upstream_stream_interrupted","account":…}}` event is appended
to the SSE stream followed by `data: [DONE]` (never a fake success, never a
silent truncation), logged as `STREAM_INTERRUPTED`. If the interrupted body
carries quota semantics the account is still marked `exhausted` for later
requests; a non-quota interruption blacklists **nothing**, because that branch
covers both "upstream stopped" and "client went away" and cannot tell them
apart.

### `timeout` reach (read before tuning)

`connect` is **not** just the TCP/TLS handshake: the connection is created with
`timeout=connect` and `getresponse()` still runs under it; only afterwards is the
socket switched to `read`. So `connect` bounds handshake + sending the request
body + waiting for the response headers (i.e. time-to-first-byte), and `read`
bounds the streamed body only. A TTFB overrun is reported by Python as
`The read operation timed out` even though the value that fired was `connect` —
expect that misleading wording in the log; it is classified as a connection-level
timeout (`server_error` + the short window + switch).

`duration_ms` in `REQ ok` is the **whole request** (entry to relay complete,
stream included), not the TTFB, so its percentiles cannot be used to size these
two values. The log carries no TTFB field.

## Streamed usage (`stream_options.include_usage`)

Before forwarding a streaming request the router injects
`stream_options: {"include_usage": true}` unless the client already set it, so
the upstream reports real usage in its final chunk — otherwise streamed requests
would log `tokens=0/0`. This is the only body rewrite besides the model mapping.
If the upstream answers **400** because of that field, it is dropped and the
**same account** is retried once (that request logs `usage_source=missing`, tag
`STREAM_OPTIONS_REJECTED`; a second 400 is returned to the client as-is).
Disable with `defaults.inject_stream_options: false` (hot reload).

Without usage the router logs `usage_source=missing` and does **no** local
token estimation (no tokenizer dependency, no invented numbers). It keeps no
usage totals and persists nothing, so the `REQ ok` line is the only observation
point — grep the log if you need cross-account accounting.

## Log reference

One line per event on stdout (redirect it wherever your service manager
collects logs). Every line is scrubbed by `secrets.py::redact`, which masks
known credential values plus anything still shaped like a provider key or a
bearer credential.

| Line | Meaning |
|---|---|
| `STARTED` / `LISTENING` / `SIGNAL` / `STOPPED` / `BIND_FAILED` | lifecycle; `STARTED` carries `accounts=` in rotation order, `LISTENING` the bound URL and the exempt paths, `SIGNAL` a graceful shutdown beginning, `BIND_FAILED` (ERROR) a taken port |
| `REQ ok account=… model=… status=… tokens=<in>/<out> usage_source=… attempt=… duration_ms=… client=…` | one served request (streamed ones add `bytes=` = bytes written to the client) |
| `ACCOUNT_EXHAUSTED` / `ACCOUNT_BLACKLISTED` / `ACCOUNT_RECOVERED` | blacklist and reset facts (`until` / `blacklist_s` / `reason` / masked evidence) |
| `EXHAUSTED_SWITCH` / `THROTTLED_SWITCH` / `UPSTREAM_ERROR` / `UPSTREAM_UNREACHABLE` / `UPSTREAM_KEY_REJECTED` / `KEY_ERROR` / `CLIENT_ERROR` | the switching judgement per class (all with `attempt=` / `duration_ms=`). A quota switch logs **two** lines — the pool-side fact and the server-side switch — so grep both tags |
| `AUTH` / `ACCOUNT` | startup credential warm-up: the inbound token (masked hint) and one line per account (`order` / `base_url` / masked key / model count); an undecryptable key logs `KEY_ERROR` (WARN) while the others keep serving |
| `STREAM_INTERRUPTED` | a stream broken **after** bytes were written (`bytes_sent=` / `usage_source=` / `quota_semantics=`) |
| `EMPTY_STREAM` / `STREAM_FIRST_BYTE_FAILED` | a streamed failure **before** any byte ⇒ switch candidate |
| `CLIENT_GONE` | the upstream stream was fully relayed and `REQ ok` logged, and the client disappeared before the closing chunk (WARN) — not an account fault, nothing blacklisted |
| `INTERNAL_ERROR` | an uncaught handler exception (ERROR, two lines: traceback + masked detail); the client only gets `500 {"error":{"type":"internal_error"}}` |
| `POOL_EXHAUSTED` | whole pool unavailable (ERROR — the line to alert on) |
| `CONFIG_RELOADED` / `CONFIG_REJECTED` | hot reload accepted (with account count) / refused, previous config kept |
| `AUTH_REJECTED` / `AUTH_UNAVAILABLE` | inbound auth refused (credentials never echoed) / the router's own token could not be resolved |

## Maintenance tool: `probe_models.py`

A read-only probe of each account's upstream (never imported by the router,
never touches its port):

```bash
python3 probe_models.py models [acct[,acct…]]   # GET {base_url}/models per account
python3 probe_models.py probe PLAN.json         # chat/completions probes, max_tokens=1
python3 probe_models.py one ACCT MODEL          # one ad-hoc probe
```

Run it **before** repointing a pool name at a different upstream model, and after
a quota recovery: accounts rarely expose the same upstream names, and a blind
repoint silently drops an account out of that name's candidate set. Its account
table is derived from `accounts.yml` by a loud minimal line scanner (stdlib only,
no PyYAML) — a seat added there becomes probeable with no edit here. Buckets:
`ok` / `exhausted` / `throttled` / `auth` / `not_purchased` / `no_model` /
`other` / `not_probed`; the first three are account-level facts and skip that
account's remaining models, the rest are per-model facts and do not. Model names
are sent verbatim, so probing a pool-side alias yields a 404 `model_not_found`
that says nothing about the seat. Credentials are read from this probe's own
environment (same names as `accounts.yml`) and never reach argv, a file or the
output (every persisted body goes through `scrub()`); artifacts land in
`LLM_ROUTER_PROBE_OUT`.

## Tests

```bash
python3 -m unittest discover -s test -v          # full matrix, any python >= 3.8
python3 test/mock_upstream.py 9911               # one mock upstream, by hand
```

The end-to-end cases start `router.py` as a subprocess against N mock upstreams
on random high ports (never the production port) with synthetic credentials
only: no real key is decrypted and no real quota is spent. All waits are bounded.
Scratch lands in `LLM_ROUTER_TEST_TMP` (default `<repo>/.test-tmp`, gitignored)
and is removed per case. Mock behaviour scripts are documented in the header of
`test/mock_upstream.py` (ok / error / exhausted / throttle / 401 / stream break /
empty stream / reject_stream_options …).

## Troubleshooting

| Symptom | Where to look |
|---|---|
| every request 401 | is the client sending `Authorization: Bearer …`? An `AUTH_UNAVAILABLE` line means the router's own token could not be resolved |
| an account is never selected | `GET /v1/models` → that entry's `available` / `reason` / `until`; `KEY_ERROR` = its env file or variable name; `UPSTREAM_KEY_REJECTED` = the upstream refuses that key |
| config edits do not take effect | `CONFIG_REJECTED` gives the validation reason (the old config is still running); `CONFIG_RELOADED` = accepted, with the account count |
| reset a blacklist | restart (all state is in memory), or wait for that account's `until` |
| `UPSTREAM_ERROR … status=404 mapped_model=True` repeats for one account | that upstream no longer serves the model this account maps it to (plan change, rename): the pool now routes around it for `blacklist_failure` seconds at a time — unmap the name in `accounts.yml` (hot reload), and probe with `probe_models.py` before mapping it again |
| `ACCOUNT_EXHAUSTED … status=403 … mapped_model=True phrase='unpurchased'` | that account's plan does not (or no longer) cover the model it maps: the pool routes around it for `blacklist_exhausted` seconds at a time. Fix = renew the plan, or unmap the name in `accounts.yml` (hot reload) and re-probe with `probe_models.py` before mapping it again |
| clients all die on their first turn while the pool looks healthy | a client-error class was relayed instead of switched: read the `CLIENT_ERROR` lines' account and status, then check whether that upstream wording belongs to an account-side class the table above does not know yet (`classify.py` is the table, `test/test_router.py::T05ClassifyTable` pins it case by case) |
| 503 `all_accounts_exhausted` | read the `POOL_EXHAUSTED` line's per-account reasons, then decide: wait for recovery, fix a key, or add an account |
| port taken | `BIND_FAILED`; `--port` overrides |

## Non-goals

No protocol translation (non-OpenAI upstreams need an adapter in front), no
usage accounting or persistence, no cross-process pool state (run one instance;
a restart is the reset button), no local token estimation, no out-of-pool
fallback (make the last account a metered one instead), no cost-based or
latency-based routing (the order is the policy).

## Files

| File | Role |
|---|---|
| `router.py` | entry point: CLI, logging, signal handling |
| `server.py` | ThreadingHTTPServer, routing, inbound auth gate |
| `config.py` | `accounts.yml` parse/validate + mtime hot reload |
| `pool.py` | candidate selection, blacklists, lazy recovery, `/v1/models` + `/health` views |
| `classify.py` | upstream outcome → reason judgement table (the only place wording is matched) |
| `proxy.py` | upstream connection, streaming relay, empty-stream probe |
| `secrets.py` | the single credential choke point: environment lookup by name, `mask()`/`redact()` |
| `probe_models.py` | read-only per-account upstream probe (maintenance tool) |
| `accounts.example.yml` | annotated reference configuration |
| `test/` | verification matrix + mock upstream |
