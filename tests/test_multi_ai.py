"""Tests for multiple AI keys: parallel load balancing, automatic zero-delay failover on rate limits,
cooldown tracking, and key masking."""
import json
import os
import sys
import tempfile
import threading
import time

os.environ["PEAKCUT_DATA"] = tempfile.mkdtemp(prefix="pc-multi-ai-")
os.environ["DATABASE_URL"] = ""
os.environ.update(PEAKCUT_AI_PROVIDER="gemini", PEAKCUT_AI_MODELS="gemini-2.5-flash", PEAKCUT_AI_KEY="", PEAKCUT_AI_BASE_URL="")
sys.path.insert(0, ".")

from studio import ai, brain, config, db  # noqa: E402

db.init()
fails = []


def check(name, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {extra}" if extra and not ok else ""))
    ok or fails.append(name)


class Reply:
    def __init__(self, status, data, headers=None):
        self.status_code, self._data, self.text = status, data, json.dumps(data)
        self.headers = headers or {}

    def json(self):
        return self._data


# 1. Test parsing of multiple keys
k_multi = "key-alpha, key-beta\nkey-gamma; key-delta"
parsed = ai._parse_keys(k_multi)
check("parse multiple keys separated by comma, newline, semicolon", parsed == ["key-alpha", "key-beta", "key-gamma", "key-delta"], str(parsed))

# 2. Test saving multiple keys
ai.save("gemini", ["gemini-2.5-flash"], new_key=k_multi)
check("keys() returns all 4 keys", ai.keys() == ["key-alpha", "key-beta", "key-gamma", "key-delta"])
check("key() returns primary key", ai.key() == "key-alpha")

pub = ai.public()
check("public() reports correct key_count", pub["key_count"] == 4, str(pub["key_count"]))
check("public() has_key is True", pub["has_key"] is True)
check("public() pool_status has 4 healthy keys", pub["pool_status"]["healthy"] == 4, str(pub["pool_status"]))

# 3. Test parallel load balancing across keys
calls_made = []
call_lock = threading.Lock()


def mock_post_parallel(url, json=None, headers=None, timeout=None, **_k):
    key = (headers or {}).get("x-goog-api-key")
    with call_lock:
        calls_made.append(key)
    # Return valid Gemini response
    return Reply(200, {
        "candidates": [{"content": {"parts": [{"text": '["ok"]'}]}}],
        "usageMetadata": {"promptTokenCount": 10, "totalTokenCount": 15}
    })


ai.httpx.post = mock_post_parallel
calls_made.clear()

threads = []
for _ in range(4):
    t = threading.Thread(target=lambda: ai.call("gemini-2.5-flash", "test prompt", brain.SCHEMA))
    threads.append(t)
for t in threads:
    t.start()
for t in threads:
    t.join()

check("4 calls executed across all keys", set(calls_made) == {"key-alpha", "key-beta", "key-gamma", "key-delta"}, str(calls_made))

# 4. Test instant zero-delay failover on rate limit (429)
# Suppose key-alpha returns 429 ResourceExhausted, but key-beta succeeds
rate_limited_keys = {"key-alpha"}
seen_failover = []


def mock_post_failover(url, json=None, headers=None, timeout=None, **_k):
    key = (headers or {}).get("x-goog-api-key")
    seen_failover.append(key)
    if key in rate_limited_keys:
        return Reply(429, {
            "error": {
                "code": 429,
                "message": f"Resource has been exhausted on key {key}",
                "status": "RESOURCE_EXHAUSTED"
            }
        })
    return Reply(200, {
        "candidates": [{"content": {"parts": [{"text": '["success_from_backup"]'}]}}],
        "usageMetadata": {"promptTokenCount": 10, "totalTokenCount": 20}
    })


ai.httpx.post = mock_post_failover
seen_failover.clear()

# Reset pool cooldowns and index so key-alpha is tried first
ai._pool._cooldowns.clear()
ai._pool._active.clear()
ai._pool._index = -1

start_time = time.time()
res, s, w = ai.call("gemini-2.5-flash", "test failover", brain.SCHEMA)
elapsed = time.time() - start_time

check("failover succeeded without raising error", res == ["success_from_backup"], str(res))
check("failover switched from key-alpha to next key", seen_failover[0] == "key-alpha" and seen_failover[1] != "key-alpha", str(seen_failover))
check("failover was immediate (< 0.5s, no 4s sleep)", elapsed < 0.5, f"elapsed: {elapsed:.3f}s")
check("key-alpha marked in cooldown", "key-alpha" in ai._pool._cooldowns)

# Next call should automatically avoid key-alpha because it is in cooldown
seen_failover.clear()
res2, _, _ = ai.call("gemini-2.5-flash", "next call", brain.SCHEMA)
check("subsequent call skips key in cooldown", seen_failover[0] != "key-alpha", str(seen_failover))

# 5. Test key masking in error messages
rate_limited_keys.update(["key-beta", "key-gamma", "key-delta"])  # all keys rate limited
try:
    ai.call("gemini-2.5-flash", "test error masking", brain.SCHEMA)
    check("raises when all keys fail", False)
except Exception as e:
    err_str = str(e)
    check("no raw api key leaked in error message", all(k not in err_str for k in ["key-alpha", "key-beta", "key-gamma", "key-delta"]), err_str)

# 6. Test key details, add, update, delete, and test_key
rate_limited_keys.clear()  # keys healthy now
kd = ai.key_details()
check("key_details() returns 4 keys", len(kd) == 4, str(len(kd)))
check("key_details() includes masked string", "..." in kd[0]["masked"])
check("key_details() includes calls count", "calls" in kd[0])

# Test single key test
tk = ai.test_key(0)
check("test_key(0) succeeded", tk["ok"] is True and tk["status"] == "ok")

# Test add key
ai.add_key("key-epsilon")
check("add_key added 5th key", len(ai.keys()) == 5 and ai.keys()[-1] == "key-epsilon")

# Test update key
ai.update_key(4, "key-omega")
check("update_key updated 5th key", ai.keys()[4] == "key-omega")

# Test delete key
ai.delete_key(4)
check("delete_key removed 5th key", len(ai.keys()) == 4 and "key-omega" not in ai.keys())

print("\n" + ("ALL PASSED" if not fails else f"{len(fails)} failed"))
import shutil
db.engine.dispose()
shutil.rmtree(os.environ["PEAKCUT_DATA"], ignore_errors=True)
sys.exit(1 if fails else 0)
