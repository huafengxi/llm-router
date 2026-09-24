#!/usr/bin/env python3
"""router.py — llm-router entry point (CLI + logging + signal handling).

Aggregates several independently-quota'd OpenAI-compatible accounts behind one
local endpoint: account selection in the writing order of accounts.yml,
automatic switch to the next account when the upstream reports exhaustion or
another account-level fault, and inbound bearer-token auth. All state is
in-memory — nothing is persisted, debug detail goes to the log only.

  python3 router.py --host 127.0.0.1 --port 9200 --accounts accounts.yml

Paths are resolved from this file's own location and from LLM_ROUTER_WS (the
credential base, default: this repo's parent directory), never from the cwd, so
a service manager needs no `cwd` setting.

Python 3.8 syntax floor; stdlib + PyYAML only.
"""
import argparse
import logging
import os
import signal
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
# Credential base: relative `env_file:` paths in accounts.yml resolve against it
# and the decryptor is looked up as <WS>/encrypt/envdec.py (override with
# LLM_ROUTER_ENVDEC). Default = this repo's parent directory, which is the
# in-workspace layout <ws>/llm-router/; a standalone checkout sets LLM_ROUTER_WS.
WS = os.environ.get("LLM_ROUTER_WS") or os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import config as config_mod          # noqa: E402
import pool as pool_mod              # noqa: E402
import secrets as secrets_mod        # noqa: E402  (local module, not stdlib)
import server as server_mod          # noqa: E402

if not hasattr(secrets_mod, "Secrets"):
    sys.exit("FATAL: `import secrets` resolved to %r instead of %s — the local "
             "choke-point module must shadow the stdlib one; check sys.path"
             % (getattr(secrets_mod, "__file__", "?"), os.path.join(HERE, "secrets.py")))


class RedactFormatter(logging.Formatter):
    """Every log line is scrubbed of known credentials on its way out."""

    def __init__(self, fmt, secrets):
        logging.Formatter.__init__(self, fmt)
        self.secrets = secrets

    def format(self, record):
        return self.secrets.redact(logging.Formatter.format(self, record))


def build_logger(secrets, level):
    logger = logging.getLogger("llm-router")
    logger.setLevel(level)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(RedactFormatter(
        "%(asctime)s %(levelname)s %(message)s", secrets))
    logger.addHandler(handler)
    return logger


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="llm-router",
        description="local OpenAI-compatible multi-account router "
                    "(quota-exhaustion aware)")
    p.add_argument("--host", default="127.0.0.1",
                   help="bind address (default 127.0.0.1 — loopback only; the "
                        "inbound bearer token is what actually isolates us from "
                        "other users on this host)")
    p.add_argument("--port", type=int, default=9200, help="bind port (default 9200)")
    p.add_argument("--accounts",
                   default=os.environ.get("LLM_ROUTER_ACCOUNTS")
                   or os.path.join(HERE, "accounts.yml"),
                   help="accounts.yml path (hot-reloaded on mtime change; "
                        "default $LLM_ROUTER_ACCOUNTS or <repo>/accounts.yml)")
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    p.add_argument("--secret-ttl", type=float, default=secrets_mod.DEFAULT_TTL,
                   help="seconds a decrypted credential is cached (default %(default)s)")
    return p.parse_args(argv)


def warm_credentials(pool, cfg, secrets, logger):
    """Resolve every credential once at startup: fail loudly, log masked only."""
    try:
        hint = secrets.token_hint()
        logger.info("AUTH inbound bearer token loaded from %s (var %s, hint %s)",
                    cfg.auth.token_env_file, cfg.auth.token_var, hint)
    except Exception as e:
        logger.error("AUTH_UNAVAILABLE cannot load the inbound router token from "
                     "%s: %s — every authenticated endpoint will answer 500 "
                     "until this is fixed", cfg.auth.token_env_file,
                     secrets.redact(str(e)))
    for acct in cfg.accounts:
        try:
            key = pool.key_for(acct)
            logger.info("ACCOUNT account=%s order=%s base_url=%s key=%s models=%d",
                        acct.name, acct.order, acct.base_url, secrets.mask(key),
                        len(acct.models))
        except Exception as e:
            logger.warning("KEY_ERROR account=%s env_file=%s var=%s: %s (the other "
                           "accounts still serve)", acct.name, acct.key_env_file,
                           acct.key_var, secrets.redact(str(e)))


def main(argv=None):
    args = parse_args(argv)
    secrets = secrets_mod.Secrets(WS, ttl=args.secret_ttl)
    logger = build_logger(secrets, getattr(logging, args.log_level))
    cfgm = config_mod.ConfigManager(args.accounts, WS, logger=logger)
    cfg = cfgm.get()
    secrets.set_token_source(cfg.auth.token_env_file, cfg.auth.token_var)

    pool = pool_mod.Pool(cfgm, secrets, logger=logger)
    warm_credentials(pool, cfg, secrets, logger)

    app = server_mod.RouterApp(cfgm, secrets, pool, logger=logger)
    try:
        httpd = server_mod.serve(app, args.host, args.port, logger=logger)
    except OSError as e:
        logger.error("BIND_FAILED %s:%s — %s (is another process holding the "
                     "port? `--port` overrides)", args.host, args.port, e)
        return 2

    stopping = {"flag": False}

    def _stop(signum, _frame):
        if stopping["flag"]:
            return
        stopping["flag"] = True
        logger.info("SIGNAL %s received — shutting down", signum)
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    logger.info("STARTED llm-router host=%s port=%s accounts=%s "
                "auth=bearer(exempt:%s) (no version face here by design)",
                args.host,
                args.port, ",".join(a.name for a in cfg.ordered()),
                ",".join(cfg.auth.exempt_paths))
    try:
        httpd.serve_forever(poll_interval=0.5)
    finally:
        httpd.server_close()
        logger.info("STOPPED llm-router (in-memory state discarded)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
