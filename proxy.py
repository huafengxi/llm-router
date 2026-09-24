"""proxy.py — upstream requests (http.client), SSE relay, usage extraction.

Request-body rewriting is limited to exactly two things:
  1. `model`   pool name -> the account's upstream model name
  2. `stream_options.include_usage` injected when the client asked for a stream
     and did not set it itself, so the `REQ ok` log line can report real token
     counts.  If the upstream answers 400 to that field, server.py retries the
     same account once without it and logs that request as usage_source=missing.

Everything else (headers except the hop-by-hop/auth set, body bytes) is passed
through untouched — protocol passthrough, no protocol translation.

Python 3.8 syntax floor; stdlib only.
"""
import http.client
import json
import ssl
from urllib.parse import urlsplit

READ_SIZE = 65536
MAX_ERROR_BODY = 262144          # bounded read of an upstream error body
MAX_PARSE_BUFFER = 1 << 20       # bounded SSE parse buffer (relay is raw)

# client headers that must not be forwarded (Authorization is replaced with the
# selected account's real key, Accept-Encoding is forced to identity so a stream
# can be relayed chunk by chunk)
HOP_BY_HOP = frozenset((
    "authorization", "host", "content-length", "accept-encoding", "connection",
    "transfer-encoding", "keep-alive", "proxy-authenticate",
    "proxy-authorization", "te", "trailer", "upgrade",
))


class UpstreamError(Exception):
    """Connection/timeout/EOF level failure (never a quota verdict)."""


def upstream_path(base_url, request_path):
    """`/v1/chat/completions` + base_url -> `<base path>/chat/completions`."""
    parts = urlsplit(base_url)
    base = (parts.path or "").rstrip("/")
    path = request_path or "/"
    query = ""
    if "?" in path:
        path, query = path.split("?", 1)
        query = "?" + query
    if path == "/v1":
        path = "/"
    elif path.startswith("/v1/"):
        path = path[len("/v1"):]
    if not path.startswith("/"):
        path = "/" + path
    return base + path + query


def build_headers(client_headers, api_key):
    """Forwarded header set: client headers minus hop-by-hop/auth, plus ours."""
    out = {}
    for key, value in client_headers.items():
        lk = key.lower()
        if lk in HOP_BY_HOP:
            continue
        out[key] = value
    out["Authorization"] = "Bearer %s" % api_key
    out["Accept-Encoding"] = "identity"
    return out


def prepare_body(body_bytes, upstream_model=None, stream=False, inject=True):
    """-> (body_bytes, injected, parsed_or_None).

    `injected` is True only when this call added stream_options.include_usage
    (i.e. when a 400 fallback that drops it is worth trying).
    """
    parsed = None
    if body_bytes:
        try:
            parsed = json.loads(body_bytes.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            parsed = None
    if not isinstance(parsed, dict):
        return body_bytes, False, parsed
    changed = False
    if upstream_model and parsed.get("model") != upstream_model:
        parsed["model"] = upstream_model
        changed = True
    injected = False
    if stream and inject:
        opts = parsed.get("stream_options")
        if not isinstance(opts, dict):
            opts = {}
        if opts.get("include_usage") is not True:
            opts = dict(opts)
            opts["include_usage"] = True
            parsed["stream_options"] = opts
            injected = True
            changed = True
    if not changed:
        return body_bytes, False, parsed
    return json.dumps(parsed, ensure_ascii=False).encode("utf-8"), injected, parsed


def _connection(acct):
    parts = urlsplit(acct.base_url)
    host = parts.hostname
    if not host:
        raise UpstreamError("bad base_url for account %s" % acct.name)
    scheme = (parts.scheme or "https").lower()
    port = parts.port or (443 if scheme == "https" else 80)
    if scheme == "https":
        ctx = ssl.create_default_context()
        if not acct.ssl_verify:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        return http.client.HTTPSConnection(host, port, timeout=acct.timeout.connect,
                                           context=ctx)
    return http.client.HTTPConnection(host, port, timeout=acct.timeout.connect)


def open_upstream(acct, method, request_path, headers, body=None):
    """Connect + send; -> (conn, response).  Read timeout applied afterwards."""
    conn = _connection(acct)
    path = upstream_path(acct.base_url, request_path)
    try:
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
    except Exception as e:
        try:
            conn.close()
        except Exception:
            pass
        raise UpstreamError("%s: %s" % (type(e).__name__, e))
    try:
        if conn.sock is not None:
            conn.sock.settimeout(acct.timeout.read)
    except OSError:
        pass
    return conn, resp


def read_error_body(resp):
    try:
        return resp.read(MAX_ERROR_BODY) or b""
    except Exception as e:
        return ("(error body unreadable: %s: %s)"
                % (type(e).__name__, e)).encode("utf-8")


def usage_from_body(body):
    """Non-stream usage object from a JSON response body (or None)."""
    if not body:
        return None
    try:
        obj = json.loads(body.decode("utf-8", "replace"))
    except (ValueError, UnicodeDecodeError):
        return None
    if isinstance(obj, dict):
        usage = obj.get("usage")
        if isinstance(usage, dict) and usage:
            return usage
    return None


def _sse_payload(line):
    if isinstance(line, bytes):
        line = line.decode("utf-8", "replace")
    line = line.strip().rstrip("\r")
    if not line.startswith("data:"):
        return None
    payload = line[len("data:"):].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        return json.loads(payload)
    except ValueError:
        return None


def sse_line_info(line):
    """-> (usage_or_None, error_text_or_None) for one SSE `data:` line."""
    obj = _sse_payload(line)
    if not isinstance(obj, dict):
        return None, None
    usage = obj.get("usage")
    if not isinstance(usage, dict) or not usage:
        usage = None
    err = obj.get("error")
    text = None
    if isinstance(err, dict):
        text = json.dumps(err, ensure_ascii=False)[:2000]
    elif isinstance(err, str) and err:
        text = err[:2000]
    return usage, text


# ---- "200 + empty SSE stream" probe -------------------------------------
# An upstream can answer 200, send a few SSE frames and close without ever
# carrying content or a non-null finish_reason.  Relayed as-is the client dies
# with "Stream ended without finish_reason" while the router logged `REQ ok`,
# so the pool never switched accounts (5 hits in 43 min on 2026-09-14).  The
# head of the stream is therefore buffered *before* the response is committed:
# while nothing has been written, switching accounts is still safe.
PROBE_CAP = 65536                 # buffered head bytes before giving up on the probe


def sse_line_progress(line):
    """-> True when one SSE `data:` line proves the stream is producing output.

    Progress = a non-null `finish_reason`, any content / reasoning / tool_calls
    delta, a completions-shaped `text` field, or a usage block.  Role-only
    deltas, keep-alive comments, `[DONE]` and unparsable payloads are NOT
    progress: a stream made only of those is the empty-stream fault.
    """
    obj = _sse_payload(line)
    if not isinstance(obj, dict):
        return False
    if obj.get("usage"):
        return True
    choices = obj.get("choices")
    if not isinstance(choices, list):
        return False
    for ch in choices:
        if not isinstance(ch, dict):
            continue
        if ch.get("finish_reason") is not None:
            return True
        if ch.get("text"):
            return True
        delta = ch.get("delta")
        if isinstance(delta, dict) and (delta.get("content")
                                        or delta.get("reasoning_content")
                                        or delta.get("tool_calls")):
            return True
    return False


def probe_stream_start(resp, first, cap=PROBE_CAP):
    """Buffer the head of an SSE stream until it proves useful (or gives up).

    -> (buffered_bytes, verdict); verdict is one of
      "progress"  a frame carried content / finish_reason / usage => relay it
      "eof"       the upstream closed without ever showing progress => the
                  empty-stream fault; nothing reached the client, so the caller
                  may switch accounts
      "cap"       more than `cap` bytes buffered with no verdict => relay it
                  (bounded memory; behaviour identical to before this probe)
    Read errors propagate — the caller already handles a failed read as
    STREAM_FIRST_BYTE_FAILED (also retryable, nothing sent yet).
    """
    buf = first or b""
    rest = b""

    def scan(data):
        nonlocal rest
        rest += data
        while b"\n" in rest:
            line, rest = rest.split(b"\n", 1)
            if sse_line_progress(line):
                return True
        return False

    if scan(buf):
        return buf, "progress"
    while len(buf) < cap:
        chunk = resp.read1(READ_SIZE)
        if not chunk:
            if rest and sse_line_progress(rest):
                return buf, "progress"
            return buf, "eof"
        buf += chunk
        if scan(chunk):
            return buf, "progress"
    return buf, "cap"


def relay_stream(resp, write, first=b"", max_errors=8):
    """Relay raw SSE bytes to `write` as they arrive.

    -> (usage_or_None, bytes_relayed, [error texts seen in the stream])
    Any exception from resp.read1()/write() propagates to the caller, which
    decides whether anything has already been sent to the client.
    """
    usage = None
    errors = []
    relayed = 0
    buf = b""

    def feed(chunk):
        nonlocal usage, buf
        if chunk:
            write(chunk)
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            u, e = sse_line_info(line)
            if u:
                usage = u
            if e and len(errors) < max_errors:
                errors.append(e)
        if len(buf) > MAX_PARSE_BUFFER:
            buf = buf[-MAX_PARSE_BUFFER:]

    if first:
        feed(first)
        relayed += len(first)
    while True:
        chunk = resp.read1(READ_SIZE)
        if not chunk:
            break
        feed(chunk)
        relayed += len(chunk)
    if buf:
        u, e = sse_line_info(buf)
        if u:
            usage = u
        if e and len(errors) < max_errors:
            errors.append(e)
    return usage, relayed, errors


def usage_tokens(usage):
    """(prompt_tokens, completion_tokens) from an OpenAI usage object."""
    if not isinstance(usage, dict):
        return 0, 0
    def pick(*names):
        for n in names:
            v = usage.get(n)
            if isinstance(v, int) and v >= 0:
                return v
            if isinstance(v, float) and v >= 0:
                return int(v)
        return 0
    return (pick("prompt_tokens", "input_tokens"),
            pick("completion_tokens", "output_tokens"))


def error_event(account_name, detail):
    """SSE error event + terminator, appended when a stream breaks mid-flight."""
    payload = {"error": {"type": "upstream_stream_interrupted",
                         "message": detail, "account": account_name}}
    return ("data: %s\n\ndata: [DONE]\n\n"
            % json.dumps(payload, ensure_ascii=False)).encode("utf-8")
