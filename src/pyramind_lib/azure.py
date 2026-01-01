"""Azure OpenAI client wrapper with disk cache, token tracking, and proper
retry/backoff so transient rate limits never become silent skips."""

from __future__ import annotations

import hashlib
import json
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import requests

_CATEGORIES = ("embedding", "scoring", "completion", "judging")

# ── retry policy ──────────────────────────────────────────────────────────
# 429 (rate limit): we WAIT, never skip. Backoff schedule is long because
# Azure PAYG quota is per-minute and only resets when the rolling window
# slides forward. Capped at 600s/attempt to prevent one bad call from
# stalling forever; 12 attempts × cap = up to ~75 min before giving up.
_MAX_429_ATTEMPTS = 12
_BACKOFF_BASE_429 = 5.0     # seconds
_BACKOFF_CAP_429 = 600.0    # 10 min per attempt cap

# Connection / 5xx errors: shorter backoff, fewer attempts. These are
# usually transient network issues that clear in seconds.
_MAX_TRANSIENT_ATTEMPTS = 8
_BACKOFF_BASE_TRANSIENT = 2.0
_BACKOFF_CAP_TRANSIENT = 60.0


@dataclass
class TokenTracker:
    """Counts tokens spent and calls made, per call category."""

    embedding: int = 0
    scoring: int = 0
    completion: int = 0
    judging: int = 0
    calls: dict = field(default_factory=lambda: {c: 0 for c in _CATEGORIES})

    def reset(self) -> None:
        self.embedding = 0
        self.scoring = 0
        self.completion = 0
        self.judging = 0
        self.calls = {c: 0 for c in _CATEGORIES}

    def total_tokens(self) -> int:
        return self.embedding + self.scoring + self.completion + self.judging


def _hash_key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _parse_retry_after(resp) -> Optional[float]:
    """Return the server-suggested wait in seconds, if any."""
    if resp is None:
        return None
    h = resp.headers or {}
    for k in ("Retry-After", "retry-after", "x-ratelimit-reset-requests",
              "x-ratelimit-reset-tokens"):
        v = h.get(k)
        if not v:
            continue
        try:
            return max(0.0, float(v))
        except (ValueError, TypeError):
            continue
    return None


def _backoff_seconds(attempt: int, base: float, cap: float) -> float:
    """Exponential backoff with jitter. attempt is 0-indexed."""
    raw = base * (2 ** attempt)
    raw = min(cap, raw)
    return raw * (0.5 + random.random() * 0.5)  # 0.5×–1.0× jitter


@dataclass
class AzureClient:
    """Azure OpenAI wrapper with on-disk JSON cache and token tracking."""

    api_key: str
    endpoint: str
    api_version: str
    deployment: str
    embedding_deployment: str
    cache_dir: Union[Path, str]
    timeout_seconds: float = 60.0
    tracker: TokenTracker = field(default_factory=TokenTracker)

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        for sub in _CATEGORIES:
            (self.cache_dir / sub).mkdir(parents=True, exist_ok=True)
        if not self.endpoint.endswith("/"):
            self.endpoint += "/"
        # Lock so concurrent threads don't blow up the rate limiter.
        self._429_lock = threading.Lock()
        self._429_until: float = 0.0  # epoch — all calls block past this

    # ── caching helpers ────────────────────────────────────────────────────

    def _cache_path(self, category: str, key: str) -> Path:
        return Path(self.cache_dir) / category / f"{key}.json"

    def _read_cache(self, category: str, key: str):
        p = self._cache_path(category, key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            p.unlink(missing_ok=True)
            return None

    def _write_cache(self, category: str, key: str, data) -> None:
        self._cache_path(category, key).write_text(json.dumps(data))

    # ── shared rate-limit gate ────────────────────────────────────────────
    # When ANY caller sees a 429, it sets `_429_until` to (now + wait). All
    # other callers (other threads, other engine sub-tasks) check this gate
    # before issuing their next request, so the whole client pauses
    # together instead of N callers each separately hammering 429s.

    def _wait_if_throttled(self) -> None:
        with self._429_lock:
            now = time.time()
            wait = self._429_until - now
        if wait > 0:
            time.sleep(wait)

    def _set_throttle_until(self, wait_s: float) -> None:
        with self._429_lock:
            target = time.time() + wait_s
            if target > self._429_until:
                self._429_until = target

    def _post_with_retry(self, url: str, body: dict, *, label: str) -> dict:
        """POST with proper retry semantics. Returns the parsed JSON body on
        success. Raises RuntimeError if every retry exhausted; raises
        immediately on 400 (bad input / content filter — not retryable)."""
        attempt_429 = 0
        attempt_transient = 0
        last_err: Optional[str] = None

        while True:
            self._wait_if_throttled()
            try:
                resp = requests.post(
                    url,
                    headers={"Content-Type": "application/json",
                             "api-key": self.api_key},
                    json=body,
                    timeout=self.timeout_seconds,
                )
            except requests.RequestException as e:
                # Network / DNS / connection-pool / timeout. Backoff and retry.
                last_err = f"{type(e).__name__}: {str(e)[:140]}"
                if attempt_transient >= _MAX_TRANSIENT_ATTEMPTS:
                    raise RuntimeError(f"transient-exhausted: {last_err}")
                wait = _backoff_seconds(
                    attempt_transient, _BACKOFF_BASE_TRANSIENT, _BACKOFF_CAP_TRANSIENT)
                print(f"[azure] {label} transient retry "
                      f"{attempt_transient + 1}/{_MAX_TRANSIENT_ATTEMPTS} "
                      f"after {wait:.1f}s — {last_err}",
                      file=sys.stderr, flush=True)
                time.sleep(wait)
                attempt_transient += 1
                continue

            if resp.ok:
                return resp.json()

            # 400 = bad input / content filter / oversized. Not retryable —
            # caller decides how to handle (sentinel for completion, zero
            # vec for embedding).
            if resp.status_code == 400:
                raise RuntimeError(f"400 {resp.text[:200]}")

            # 429 = rate limit. WAIT and retry, never skip.
            if resp.status_code == 429:
                if attempt_429 >= _MAX_429_ATTEMPTS:
                    raise RuntimeError(
                        f"429-exhausted after {_MAX_429_ATTEMPTS} attempts")
                server_wait = _parse_retry_after(resp)
                wait = server_wait if server_wait is not None else \
                    _backoff_seconds(
                        attempt_429, _BACKOFF_BASE_429, _BACKOFF_CAP_429)
                # Set the shared throttle gate so other concurrent callers
                # don't immediately re-hit the same 429.
                self._set_throttle_until(wait)
                print(f"[azure] {label} 429 retry "
                      f"{attempt_429 + 1}/{_MAX_429_ATTEMPTS} after {wait:.1f}s "
                      f"({'server' if server_wait is not None else 'backoff'})",
                      file=sys.stderr, flush=True)
                time.sleep(wait)
                attempt_429 += 1
                continue

            # Other 5xx / 408 → treat as transient.
            if resp.status_code >= 500 or resp.status_code == 408:
                last_err = f"{resp.status_code} {resp.text[:160]}"
                if attempt_transient >= _MAX_TRANSIENT_ATTEMPTS:
                    raise RuntimeError(f"5xx-exhausted: {last_err}")
                wait = _backoff_seconds(
                    attempt_transient, _BACKOFF_BASE_TRANSIENT, _BACKOFF_CAP_TRANSIENT)
                print(f"[azure] {label} {resp.status_code} retry "
                      f"{attempt_transient + 1}/{_MAX_TRANSIENT_ATTEMPTS} "
                      f"after {wait:.1f}s",
                      file=sys.stderr, flush=True)
                time.sleep(wait)
                attempt_transient += 1
                continue

            # Anything else (401, 403, 404, etc.) — surface immediately.
            raise RuntimeError(f"{resp.status_code} {resp.text[:200]}")

    # ── embedding ──────────────────────────────────────────────────────────

    def embed(self, text: str, model: Optional[str] = None) -> list:
        model = model or self.embedding_deployment
        key = _hash_key(model, text)
        cached = self._read_cache("embedding", key)
        if cached is not None:
            return cached

        url = f"{self.endpoint}openai/deployments/{model}/embeddings?api-version={self.api_version}"
        try:
            data = self._post_with_retry(url, {"input": text}, label="EMBED")
        except RuntimeError as e:
            # 400 / exhaustion. Only path that returns the zero-vec sentinel.
            print(f"[azure] EMBED-SKIP {str(e)[:160]} text={text[:80]!r}",
                  file=sys.stderr, flush=True)
            zero = [0.0] * 3072
            self._write_cache("embedding", key, zero)
            return zero

        vec = data["data"][0]["embedding"]
        self.tracker.embedding += int(data.get("usage", {}).get("total_tokens", 0))
        self.tracker.calls["embedding"] += 1
        self._write_cache("embedding", key, vec)
        return vec

    # ── completion ─────────────────────────────────────────────────────────

    def complete(
        self,
        prompt: str,
        system: str,
        category: str = "completion",
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> str:
        if category not in _CATEGORIES:
            raise ValueError(f"Unknown category: {category}. Use one of {_CATEGORIES}")

        key = _hash_key(category, system, prompt, str(temperature), str(max_tokens))
        cached = self._read_cache(category, key)
        if cached is not None:
            return cached["content"]

        url = f"{self.endpoint}openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"
        body = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            data = self._post_with_retry(url, body, label="COMPLETE")
        except RuntimeError as e:
            # 400 (content filter) or exhaustion. Sentinel path.
            print(f"[azure] COMPLETE-SKIP {str(e)[:160]}",
                  file=sys.stderr, flush=True)
            sentinel = "I do not have enough information to answer."
            self._write_cache(category, key, {"content": sentinel})
            return sentinel

        content = (data.get("choices", [{}])[0].get("message", {}) or {}).get("content") or ""
        tokens = int(data.get("usage", {}).get("total_tokens", 0))
        if category == "scoring":
            self.tracker.scoring += tokens
        elif category == "judging":
            self.tracker.judging += tokens
        else:
            self.tracker.completion += tokens
        self.tracker.calls[category] += 1
        self._write_cache(category, key, {"content": content})
        return content
