"""Talking to the AI service. You bring your own key, and choose the service and model in the dashboard
(Settings, AI) or in .env. What is saved from the dashboard is kept in ai.json in the data folder and wins.

    PEAKCUT_AI_PROVIDER   gemini | openai | xai | groq | openrouter | anthropic | ollama | custom
    PEAKCUT_AI_MODELS     one or more model ids, comma separated, tried in this order
    PEAKCUT_AI_KEY        your key (or PEAKCUT_AI_KEY_FILE, a file holding it)
    PEAKCUT_AI_BASE_URL   only for "custom" (any service that speaks the OpenAI chat format), or to override

Every service is asked for the same thing: JSON that follows a small schema. Gemini takes the schema directly.
The others are told the schema in the prompt and asked for a JSON object, which works with any chat model."""
import hashlib
import json
import os
import re
import threading
import time

import httpx

from . import config

# name -> (label shown on the pages, address, how it is spoken to, where to get a key, an example model id)
PROVIDERS = {
    "gemini": ("Gemini", "https://generativelanguage.googleapis.com/v1beta/models", "gemini", "https://aistudio.google.com/apikey", "gemini-2.5-flash"),
    "openai": ("OpenAI", "https://api.openai.com/v1", "openai", "https://platform.openai.com/api-keys", "gpt-4o-mini"),
    "xai": ("Grok", "https://api.x.ai/v1", "openai", "https://console.x.ai", "grok-4"),
    "groq": ("Groq", "https://api.groq.com/openai/v1", "openai", "https://console.groq.com/keys", "llama-3.3-70b-versatile"),
    "openrouter": ("OpenRouter", "https://openrouter.ai/api/v1", "openai", "https://openrouter.ai/keys", "google/gemini-2.5-flash"),
    "anthropic": ("Claude", "https://api.anthropic.com/v1", "anthropic", "https://console.anthropic.com", ""),
    "ollama": ("Ollama (on this machine)", "http://127.0.0.1:11434/v1", "openai", "https://ollama.com", "llama3.1"),
    "custom": ("Custom (OpenAI-compatible)", "", "openai", "", ""),
}
ALIASES = {"google": "gemini", "grok": "xai", "claude": "anthropic", "local": "ollama"}
NO_KEY = {"ollama", "custom"}       # a model running on your own machine needs no key

# ---------------------------------------------------------------- the settings in use
_lock = threading.Lock()
_saved = {"at": None, "data": {}}


def _file():
    return config.DATA_DIR / "ai.json"


def _parse_keys(raw) -> list:
    """Extract individual API keys from string, list, or comma/newline/semicolon-separated text."""
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        items = raw
    elif isinstance(raw, str):
        items = re.split(r"[\r\n,;]+", raw)
    else:
        items = [str(raw)]
    res = []
    for item in items:
        k = str(item).strip().strip('"').strip("'")
        if k and k not in res:
            res.append(k)
    return res


# ---------------------------------------------------------------- key pool & dynamic failover
class KeyPool:
    """Manages multiple API keys: load balancing in parallel across keys,
    tracking health, and instant automatic failover when a key hits a rate limit."""

    def __init__(self):
        self._lock = threading.Lock()
        self._index = 0
        self._cooldowns = {}       # key -> timestamp until which it is in cooldown
        self._active = {}          # key -> count of currently active calls
        self._failures = {}        # key -> count of consecutive rate limits
        self._last_used = {}       # key -> timestamp when last acquired
        self._stats = self._load_stats()

    def _stats_file(self):
        return config.DATA_DIR / "ai_key_stats.json"

    def _load_stats(self) -> dict:
        try:
            f = self._stats_file()
            if f.exists():
                return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    def _save_stats(self):
        try:
            f = self._stats_file()
            tmp = f.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._stats), encoding="utf-8")
            tmp.replace(f)
        except Exception:
            pass

    def _key_id(self, key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

    def get_key_stat(self, key: str) -> dict:
        with self._lock:
            kid = self._key_id(key)
            return dict(self._stats.get(kid, {}))

    def record_success(self, key: str, tokens: int = 0, latency_ms: int = 0):
        if not key:
            return
        with self._lock:
            self._failures.pop(key, None)
            self._cooldowns.pop(key, None)
            kid = self._key_id(key)
            st = self._stats.setdefault(kid, {"calls": 0, "tokens": 0, "success": 0, "failures": 0})
            st["calls"] = st.get("calls", 0) + 1
            st["success"] = st.get("success", 0) + 1
            st["tokens"] = st.get("tokens", 0) + tokens
            st["last_used"] = time.time()
            st["last_status"] = "ok"
            st["last_error"] = ""
            st["latency_ms"] = latency_ms
            self._save_stats()

    def record_failure(self, key: str, error_msg: str, is_rate_limit: bool = True, retry_after: float = 60.0):
        if not key:
            return
        with self._lock:
            self._failures[key] = self._failures.get(key, 0) + 1
            duration = max(5.0, min(float(retry_after or 60.0), 300.0))
            if is_rate_limit:
                self._cooldowns[key] = time.time() + duration
            else:
                # Invalid or errored keys get an extended cooldown so they do not interrupt working keys
                self._cooldowns[key] = time.time() + 3600.0
            kid = self._key_id(key)
            st = self._stats.setdefault(kid, {"calls": 0, "tokens": 0, "success": 0, "failures": 0})
            st["calls"] = st.get("calls", 0) + 1
            st["failures"] = st.get("failures", 0) + 1
            st["last_used"] = time.time()
            st["last_status"] = "cooldown" if is_rate_limit else "error"
            st["last_error"] = str(error_msg or "")[:150]
            self._save_stats()

    def acquire(self, available_keys: list, exclude: set = None) -> tuple:
        """Choose the best key from available_keys that is not in exclude.
        Prioritizes:
          1. Healthy keys (cooldown timestamp passed)
          2. Least active in-flight requests (for true parallel load balancing)
          3. Round-robin rotation among ties
        If all keys are in cooldown, falls back to the key whose cooldown expires earliest.
        Returns (key, is_cooling_down)."""
        if not available_keys:
            return "", False

        exclude = exclude or set()
        candidates = [k for k in available_keys if k and k not in exclude]
        if not candidates:
            candidates = [k for k in available_keys if k]
            if not candidates:
                return "", False

        with self._lock:
            now = time.time()
            # Clean up expired cooldowns
            expired = [k for k, expire_t in self._cooldowns.items() if expire_t <= now]
            for k in expired:
                del self._cooldowns[k]

            # Prioritize healthy keys that are not cooling down and have no hard errors
            healthy = [k for k in candidates if k not in self._cooldowns and self._stats.get(self._key_id(k), {}).get("last_status") != "error"]
            if not healthy:
                # If all have errors/cooldown, fall back to any candidate not cooling down
                healthy = [k for k in candidates if k not in self._cooldowns]

            if healthy:
                self._index += 1
                idx = self._index
                # Sort by active count ascending, then by round-robin offset
                healthy_sorted = sorted(
                    healthy,
                    key=lambda k: (self._active.get(k, 0), (healthy.index(k) - idx) % len(healthy))
                )
                chosen = healthy_sorted[0]
                self._active[chosen] = self._active.get(chosen, 0) + 1
                self._last_used[chosen] = now
                return chosen, False

            # If all candidates are currently in cooldown:
            earliest = min(candidates, key=lambda k: self._cooldowns.get(k, 0))
            self._active[earliest] = self._active.get(earliest, 0) + 1
            self._last_used[earliest] = now
            return earliest, True

    def release(self, key: str):
        if not key:
            return
        with self._lock:
            cur = self._active.get(key, 0)
            if cur > 1:
                self._active[key] = cur - 1
            else:
                self._active.pop(key, None)

    def mark_success(self, key: str):
        self.record_success(key)

    def mark_rate_limited(self, key: str, retry_after: float = 60.0):
        self.record_failure(key, "Rate limit / quota exceeded", is_rate_limit=True, retry_after=retry_after)

    def status(self, pool_keys: list) -> dict:
        with self._lock:
            now = time.time()
            in_cd = sum(1 for k in pool_keys if self._cooldowns.get(k, 0) > now)
            active = sum(self._active.get(k, 0) for k in pool_keys)
            return {
                "total": len(pool_keys),
                "healthy": max(0, len(pool_keys) - in_cd),
                "in_cooldown": in_cd,
                "active_calls": active,
            }


_pool = KeyPool()


def _from_dashboard() -> dict:
    """What was saved from the dashboard, read again only when the file changes."""
    f = _file()
    try:
        at = f.stat().st_mtime_ns
    except OSError:
        at = 0
    if _saved["at"] != at:
        try:
            data = json.loads(f.read_text(encoding="utf-8")) if at else {}
        except (OSError, ValueError):
            data = {}
        _saved.update(at=at, data=data if isinstance(data, dict) else {})
    return _saved["data"]


def current() -> dict:
    """Provider, models, address and keys: what the dashboard saved, or else what .env says."""
    d = _from_dashboard()
    p = str(d.get("provider") or config.AI_PROVIDER or "").strip().lower()
    if d.get("provider"):
        raw = d.get("keys") or d.get("key") or ""
        all_keys = _parse_keys(raw)
        raw_key = d.get("key") or (", ".join(all_keys) if all_keys else "")
    else:
        all_keys = _parse_keys(getattr(config, "ai_keys", lambda: [])())
        raw_key = config.ai_key()
        if not all_keys and raw_key:
            all_keys = _parse_keys(raw_key)

    primary_key = all_keys[0] if all_keys else (str(raw_key).strip() if raw_key else "")
    return {
        "provider": ALIASES.get(p, p),
        "models": list(d["models"]) if d.get("provider") else list(config.AI_MODELS),
        "base_url": (str(d.get("base_url") or "") if d.get("provider") else config.AI_BASE_URL).rstrip("/"),
        "key": primary_key,
        "keys": all_keys,
        "raw_key": raw_key,
    }


def provider() -> str:
    return current()["provider"]


def models() -> list:
    return current()["models"]


def key() -> str:
    return current()["key"]


def keys() -> list:
    return current()["keys"]


def label() -> str:
    return PROVIDERS.get(provider(), ("",))[0]


def base_url() -> str:
    c = current()
    return c["base_url"] or PROVIDERS.get(c["provider"], ("", ""))[1]


def ready() -> bool:
    c = current()
    return bool(c["provider"] in PROVIDERS and c["models"] and base_url() and (c["keys"] or c["key"] or c["provider"] in NO_KEY))


def key_details() -> list:
    """Detailed health and usage information for each configured key."""
    ks = keys()
    res = []
    now = time.time()
    for idx, k in enumerate(ks):
        kid = hashlib.sha256(k.encode("utf-8")).hexdigest()[:16]
        hint = k[-4:] if len(k) >= 4 else "key"
        prefix = (k[:4] + "..." + k[-4:]) if len(k) >= 8 else (k[:2] + "..." + k[-2:] if len(k) >= 4 else "***")
        st = _pool.get_key_stat(k)
        cd = _pool._cooldowns.get(k, 0)
        in_cd = cd > now
        rem_cd = int(cd - now) if in_cd else 0

        if in_cd:
            status = "cooldown"
        elif st.get("last_status"):
            status = st["last_status"]
        else:
            status = "ready"

        res.append({
            "index": idx,
            "id": kid,
            "hint": hint,
            "masked": prefix,
            "status": status,
            "in_cooldown": in_cd,
            "cooldown_remaining": rem_cd,
            "calls": st.get("calls", 0),
            "tokens": st.get("tokens", 0),
            "last_latency_ms": st.get("latency_ms", 0),
            "last_error": st.get("last_error", ""),
            "last_used": st.get("last_used", 0),
        })
    return res


def public() -> dict:
    """What the settings page may see. The keys themselves never leave the server: only whether there are any,
    how many, and their hints so the owner can recognize them."""
    c = current()
    ks = c["keys"]
    if len(ks) <= 1:
        hint = ks[0][-4:] if ks and len(ks[0]) >= 4 else ""
    else:
        hint = f"{len(ks)} keys (ends in " + ", ".join(k[-4:] for k in ks if len(k) >= 4) + ")"
    return {
        "provider": c["provider"],
        "models": c["models"],
        "base_url": c["base_url"],
        "has_key": bool(ks),
        "key_hint": hint,
        "key_count": len(ks),
        "key_hints": [k[-4:] for k in ks if len(k) >= 4],
        "key_details": key_details(),
        "pool_status": _pool.status(ks),
        "ready": ready(),
        "label": label(),
        "saved_here": bool(_from_dashboard().get("provider")),
        "providers": [
            {"id": k, "label": v[0], "address": v[1], "needs_key": k not in NO_KEY, "key_url": v[3], "example": v[4]}
            for k, v in PROVIDERS.items()
        ],
    }


def save(provider_id: str, model_ids, address: str = "", new_key=None) -> dict:
    """Save the choice made in the dashboard. `new_key` None keeps the key(s) already there; an empty string removes them."""
    p = ALIASES.get(str(provider_id or "").strip().lower(), str(provider_id or "").strip().lower())
    if p not in PROVIDERS:
        raise ValueError("Choose one of the AI services in the list.")
    if isinstance(model_ids, str):
        model_ids = model_ids.split(",")
    ms = [str(m).strip()[:120] for m in (model_ids or []) if str(m).strip()][:5]
    if not ms:
        raise ValueError("Enter the model to use.")
    if any(not re.fullmatch(r"[\w.\-:/@]+", m) for m in ms):
        raise ValueError("A model id has letters, digits and . - : / only.")
    address = str(address or "").strip().rstrip("/")[:300]
    if address and not re.fullmatch(r"https?://[^\s]+", address):
        raise ValueError("The address has to start with http:// or https://")
    if p == "custom" and not address:
        raise ValueError("A custom service needs its address, for example http://127.0.0.1:1234/v1")
    old = current()
    if new_key is None and old["provider"] == p:
        k_list = old["keys"]
        k_str = old["raw_key"]
    else:
        k_list = _parse_keys(new_key)
        k_str = ", ".join(k_list) if len(k_list) > 1 else (k_list[0] if k_list else "")
    if not k_list and p not in NO_KEY:
        raise ValueError("Paste the key for this service.")
    with _lock:
        f = _file()
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps({"provider": p, "models": ms, "base_url": address, "key": k_str, "keys": k_list}), encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)        # the key is only for whoever runs the server
        except OSError:
            pass
        tmp.replace(f)
        _saved["at"] = None
    return public()


def list_models(provider_id=None, address=None, with_key=None) -> list:
    """The model ids a service offers, so the settings page can show a list to pick from."""
    c = current()
    p = ALIASES.get(str(provider_id or c["provider"]).lower(), str(provider_id or c["provider"]).lower())
    if p not in PROVIDERS:
        raise ValueError("Choose one of the AI services in the list.")
    candidate_keys = _parse_keys(with_key) if with_key else (c["keys"] if c["provider"] == p else [])
    if not candidate_keys:
        cand = str(with_key or "").strip() or (c["key"] if c["provider"] == p else "")
        if cand:
            candidate_keys = [cand]
    base = str(address or "").strip().rstrip("/") or (c["base_url"] if c["provider"] == p else "") or PROVIDERS[p][1]
    if not base:
        raise ValueError("Enter the service's address first.")
    how = PROVIDERS[p][2]

    last_err = None
    for cand_k in (candidate_keys or [""]):
        try:
            if how == "gemini":
                r = httpx.get(base, params={"pageSize": 200}, headers={"x-goog-api-key": cand_k} if cand_k else {}, timeout=20)
            elif how == "anthropic":
                r = httpx.get(f"{base}/models", params={"limit": 200}, headers={"x-api-key": cand_k, "anthropic-version": "2023-06-01"} if cand_k else {}, timeout=20)
            else:
                r = httpx.get(f"{base}/models", headers={"Authorization": f"Bearer {cand_k}"} if cand_k else {}, timeout=20)
            if r.status_code == 200:
                data = r.json()
                if how == "gemini":
                    found = [m.get("name", "").split("/")[-1] for m in data.get("models", [])
                             if "generateContent" in (m.get("supportedGenerationMethods") or [])]
                else:
                    found = [m.get("id", "") for m in data.get("data", []) if isinstance(m, dict)]
                return sorted({m for m in found if m})[:400]
            last_err = AIError(f"HTTP {r.status_code} {r.text[:200].replace(cand_k or chr(0), '<key>')}")
        except Exception as e:
            last_err = e
    if last_err:
        raise last_err
    return []


def test() -> dict:
    """One tiny request with the settings in use, to prove the key(s), the address and the model all work."""
    ms = models()
    if not ready():
        raise ValueError("Save a service, a model and a key first.")
    ks = keys()
    out, sent, wrote = call(ms[0], 'Answer with a JSON array that holds the single string "ok".', {"type": "ARRAY", "items": {"type": "STRING"}})
    res = {"ok": True, "model": ms[0], "tokens": sent + wrote, "answer": str(out)[:80]}
    if len(ks) > 1:
        res["note"] = f"{len(ks)} keys configured with parallel load balancing and auto-failover."
        res["key_count"] = len(ks)
    return res


def test_key(index: int) -> dict:
    """Test a single key by its index in keys()."""
    ks = keys()
    if index < 0 or index >= len(ks):
        raise ValueError(f"Key index {index} is out of range.")
    k = ks[index]
    ms = models()
    if not ms:
        raise ValueError("Enter the model to use first.")

    start_t = time.time()
    p = provider()
    how = PROVIDERS[p][2]
    dispatcher = {"gemini": _gemini, "openai": _openai, "anthropic": _anthropic}[how]

    try:
        out, sent, wrote = dispatcher(ms[0], 'Answer with a JSON array that holds the single string "ok".', {"type": "ARRAY", "items": {"type": "STRING"}}, key=k)
        latency = int((time.time() - start_t) * 1000)
        _pool.record_success(k, tokens=sent + wrote, latency_ms=latency)
        return {"ok": True, "index": index, "status": "ok", "latency_ms": latency, "tokens": sent + wrote, "error": ""}
    except Exception as e:
        latency = int((time.time() - start_t) * 1000)
        err_str = str(e)
        for cand in ks:
            if cand:
                err_str = err_str.replace(cand, "<key>")
        is_rl = _is_rate_limit(e)
        _pool.record_failure(k, err_str, is_rate_limit=is_rl, retry_after=_extract_retry_after(e))
        return {"ok": False, "index": index, "status": "cooldown" if is_rl else "error", "latency_ms": latency, "tokens": 0, "error": err_str}


def test_all_keys() -> list:
    """Test all keys in keys() and return status of each."""
    ks = keys()
    return [test_key(idx) for idx in range(len(ks))]


def add_key(new_key: str) -> dict:
    """Add one or more keys to the existing list."""
    to_add = _parse_keys(new_key)
    if not to_add:
        raise ValueError("Enter the key to add.")
    cur_keys = keys()
    updated = list(cur_keys)
    added_count = 0
    for k in to_add:
        if k not in updated:
            updated.append(k)
            added_count += 1
    if added_count == 0:
        return public()
    c = current()
    return save(c["provider"], c["models"], c["base_url"], new_key=updated)


def delete_key(index: int) -> dict:
    """Delete a key by its index."""
    cur_keys = keys()
    if index < 0 or index >= len(cur_keys):
        raise ValueError(f"Key index {index} is out of range.")
    updated = [k for i, k in enumerate(cur_keys) if i != index]
    c = current()
    return save(c["provider"], c["models"], c["base_url"], new_key=updated if updated else "")


def update_key(index: int, new_key: str) -> dict:
    """Update a specific key at index."""
    cur_keys = keys()
    if index < 0 or index >= len(cur_keys):
        raise ValueError(f"Key index {index} is out of range.")
    clean = _parse_keys(new_key)
    if not clean:
        raise ValueError("Enter a valid replacement key.")
    updated = list(cur_keys)
    updated[index] = clean[0]
    c = current()
    return save(c["provider"], c["models"], c["base_url"], new_key=updated)


class AIError(RuntimeError):
    pass


def _fail(r, model, use_key=None):
    text = r.text[:300]
    all_k = set(keys())
    if use_key:
        all_k.add(use_key)
    primary = key()
    if primary:
        all_k.add(primary)
    for k in all_k:
        if k:
            text = text.replace(k, "<key>")
    err = AIError(f"HTTP {r.status_code} {text}")
    err.status_code = r.status_code
    err.headers = getattr(r, "headers", {})
    raise err


def _is_rate_limit(e: Exception) -> bool:
    code = getattr(e, "status_code", None)
    if code in (429, 500, 502, 503, 504):
        return True
    s = str(e).lower()
    return any(term in s for term in (
        "http 429", "http 500", "http 502", "http 503", "http 504",
        "resource_exhausted", "resourceexhausted", "quota",
        "rate limit", "rate_limit", "too many requests", "overloaded",
        "temporarily unavailable", "try again later"
    ))


def _extract_retry_after(e: Exception) -> float:
    headers = getattr(e, "headers", None) or {}
    val = headers.get("retry-after") or headers.get("Retry-After")
    if val:
        try:
            sec = float(val)
            if 0 < sec < 300:
                return sec
        except ValueError:
            pass
    return 60.0


# ---------------------------------------------------------------- the schema, said in words for chat models
def _json_schema(s):
    """Gemini's schema spelling (ARRAY, OBJECT...) as ordinary JSON Schema."""
    if isinstance(s, dict):
        return {k: (v.lower() if k == "type" and isinstance(v, str) else _json_schema(v)) for k, v in s.items()}
    if isinstance(s, list):
        return [_json_schema(x) for x in s]
    return s


def _wrapped(prompt, schema):
    return (prompt + "\n\nAnswer with JSON only, no other text and no code fences: one object with a single key "
            "\"items\", whose value follows this JSON Schema:\n" + json.dumps(_json_schema(schema)))


def _parse(text):
    """The JSON in an answer, even if the model wrapped it in a code fence or a sentence."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        out = json.loads(text)
    except ValueError:
        m = re.search(r"[\[{].*[\]}]", text, re.S)
        if not m:
            raise AIError("The AI did not answer with JSON.")
        out = json.loads(m.group(0))
    if isinstance(out, dict) and "items" in out:
        out = out["items"]
    return out


# ---------------------------------------------------------------- one call
def call(model: str, prompt: str, schema: dict, key: str = None):
    """Ask one model. Returns (the parsed JSON answer, tokens sent, tokens written back).
    When multiple API keys are configured, distributes requests dynamically in parallel
    and switches immediately with zero delay if a key hits rate limits."""
    p = provider()
    how = PROVIDERS[p][2]
    dispatcher = {"gemini": _gemini, "openai": _openai, "anthropic": _anthropic}[how]

    if key is not None:
        return dispatcher(model, prompt, schema, key)

    all_keys = keys()
    if not all_keys:
        if p in NO_KEY:
            return dispatcher(model, prompt, schema, "")
        raise ValueError(f"No API key configured for {label()}.")

    if len(all_keys) == 1:
        chosen = all_keys[0]
        _pool.acquire([chosen])
        t0 = time.time()
        try:
            res = dispatcher(model, prompt, schema, chosen)
            latency = int((time.time() - t0) * 1000)
            tokens = (res[1] + res[2]) if len(res) >= 3 else 0
            _pool.record_success(chosen, tokens=tokens, latency_ms=latency)
            return res
        except Exception as e:
            _pool.record_failure(chosen, str(e), is_rate_limit=_is_rate_limit(e), retry_after=_extract_retry_after(e))
            raise
        finally:
            _pool.release(chosen)

    tried = set()
    last_err = None
    max_tries = min(len(all_keys) * 2, 8)

    for _attempt in range(max_tries):
        chosen, is_cooling = _pool.acquire(all_keys, exclude=tried)
        if not chosen:
            break
        tried.add(chosen)

        if is_cooling:
            now = time.time()
            wait_s = _pool._cooldowns.get(chosen, 0) - now
            if wait_s > 0:
                if wait_s <= 1.0:
                    time.sleep(wait_s)
                elif len(tried) >= len(all_keys):
                    _pool.release(chosen)
                    raise AIError(f"HTTP 429 All {len(all_keys)} API keys are temporarily rate-limited. Next available in {int(wait_s)}s.")

        t0 = time.time()
        try:
            res = dispatcher(model, prompt, schema, chosen)
            latency = int((time.time() - t0) * 1000)
            tokens = (res[1] + res[2]) if len(res) >= 3 else 0
            _pool.record_success(chosen, tokens=tokens, latency_ms=latency)
            return res
        except Exception as e:
            last_err = e
            is_rl = _is_rate_limit(e)
            retry_after = _extract_retry_after(e)
            _pool.record_failure(chosen, str(e), is_rate_limit=is_rl, retry_after=retry_after)
            # Failover immediately to next key if untried keys remain in the pool
            if len(tried) < len(all_keys):
                continue
            raise
        finally:
            _pool.release(chosen)

    if last_err:
        raise last_err
    raise AIError("All API keys failed to answer.")


def _gemini(model, prompt, schema, key=None):
    use_key = key or current()["key"]
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.4, "responseMimeType": "application/json", "responseSchema": schema}}
    r = httpx.post(f"{base_url()}/{model}:generateContent", json=body, headers={"x-goog-api-key": use_key}, timeout=120)
    if r.status_code != 200:
        _fail(r, model, use_key)
    data = r.json()
    text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
    u = data.get("usageMetadata", {})
    sent = int(u.get("promptTokenCount") or 0)
    return json.loads(text), sent, max(0, int(u.get("totalTokenCount") or 0) - sent)   # the answer, and any thinking billed with it


def _openai(model, prompt, schema, key=None):
    use_key = key or current()["key"]
    headers = {"Authorization": f"Bearer {use_key}"} if use_key else {}
    body = {"model": model, "messages": [{"role": "user", "content": _wrapped(prompt, schema)}]}
    # Most services accept these two extras; a few models refuse one of them, so they are dropped on a refusal.
    extras = [{"temperature": 0.4, "response_format": {"type": "json_object"}}, {"response_format": {"type": "json_object"}}, {}]
    r = None
    for extra in extras:
        r = httpx.post(f"{base_url()}/chat/completions", json={**body, **extra}, headers=headers, timeout=180)
        if r.status_code not in (400, 422):
            break
    if r.status_code != 200:
        _fail(r, model, use_key)
    data = r.json()
    u = data.get("usage") or {}
    return _parse(data["choices"][0]["message"].get("content")), int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)


def _anthropic(model, prompt, schema, key=None):
    use_key = key or current()["key"]
    body = {"model": model, "max_tokens": 8192, "messages": [{"role": "user", "content": _wrapped(prompt, schema)}]}
    r = httpx.post(f"{base_url()}/messages", json=body, timeout=180,
                   headers={"x-api-key": use_key, "anthropic-version": "2023-06-01"})
    if r.status_code != 200:
        _fail(r, model, use_key)
    data = r.json()
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    u = data.get("usage") or {}
    return _parse(text), int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)
