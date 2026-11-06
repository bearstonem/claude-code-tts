"""End-to-end tests: the hooks, the daemon, per-session voices and the CLI.

Everything runs as it does under Claude Code (hooks are separate processes that
start a detached daemon), but on its own port with its own config and state
directory, so a real install on the same machine is left alone. Machines
without an audio device still pass: the daemon accepts and synthesizes speech,
and only playing it fails.
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tts_common  # noqa: E402
import tts_hook  # noqa: E402

PORT = 7461
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tts")
    (tmp / "config.json").write_text(json.dumps({"port": PORT}), encoding="utf-8")
    environ = dict(
        os.environ,
        CLAUDE_TTS_CONFIG=str(tmp / "config.json"),
        CLAUDE_TTS_STATE_DIR=str(tmp / "state"),
    )
    yield environ
    call("/shutdown", {})


def call(path: str, payload: dict | None = None):
    data = None if payload is None else json.dumps(payload).encode()
    try:
        with OPENER.open(f"http://127.0.0.1:{PORT}{path}", data=data, timeout=2) as resp:
            return json.loads(resp.read())
    except OSError:
        return None


def wait_for(check, timeout: float = 180) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.25)
    return False


def ready() -> bool:
    return bool((call("/health") or {}).get("ready"))


def run(env, script: str, *args: str, data: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / script), *args],
        input=json.dumps(data or {}),
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )


def hook(env, command: str, data: dict) -> None:
    result = run(env, "tts_hook.py", command, data=data)
    assert result.returncode == 0 and result.stdout == "", result


def sessions(env) -> dict:
    path = Path(env["CLAUDE_TTS_STATE_DIR"]) / "sessions.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def hook_log(env) -> str:
    path = Path(env["CLAUDE_TTS_STATE_DIR"]) / "hook.log"
    return path.read_text(encoding="utf-8") if path.exists() else ""


# Units


def test_voice_list_matches_paradee():
    from paradee import VOICES

    assert tts_common.VOICES == VOICES[tts_common.PARADEE_VERSION]


def test_speaks_only_the_summary():
    reply = "Long technical answer.\n\n<!-- TTS_SUMMARY\nAll the tests pass.\nTTS_SUMMARY -->"
    assert tts_hook.to_speech(reply).strip() == "All the tests pass."


@pytest.mark.parametrize("voice", tts_common.VOICES)
def test_synthesizes_every_voice(voice):
    from paradee import SAMPLE_RATE, Paradee

    model, config = tts_common.download_model(gpu=False)
    audio = Paradee(model_path=model, config_path=config)("Hello from the tests.", voice=voice)
    assert audio.dtype == np.float32 and np.isfinite(audio).all()
    assert 0.5 < len(audio) / SAMPLE_RATE < 5


# The whole flow, in order: each test builds on the daemon and sessions before it


def test_session_start_launches_the_daemon(env):
    hook(env, "warm", {"session_id": "session-one", "cwd": "/work/one"})
    assert wait_for(ready), "daemon never became ready"
    health = call("/health")
    assert health["voices"] == tts_common.VOICES
    assert health["providers"][0] == "CPUExecutionProvider"


def test_sessions_get_different_voices(env):
    hook(env, "warm", {"session_id": "session-two", "cwd": "/work/two"})
    claims = sessions(env)
    assert claims["session-one"]["voice"] == "af_heart"
    assert claims["session-two"]["voice"] == "bm_fable"
    assert claims["session-two"]["project"] == "two"


def test_stop_hook_speaks_in_the_session_voice(env):
    message = "Done.\n<!-- TTS_SUMMARY\nSession two is speaking.\nTTS_SUMMARY -->"
    hook(env, "stop", {"session_id": "session-two", "last_assistant_message": message})
    log = hook_log(env)
    assert "reply (24 chars, bm_fable): {'ok': True, 'skipped': False}" in log, log


def test_parallel_session_starts_all_claim_a_voice(env):
    ids = [f"parallel-{i}" for i in range(6)]
    procs = [
        subprocess.Popen(
            [sys.executable, str(ROOT / "tts_hook.py"), "warm"],
            stdin=subprocess.PIPE,
            env=env,
            text=True,
        )
        for _ in ids
    ]
    for proc, sid in zip(procs, ids):
        proc.communicate(json.dumps({"session_id": sid, "cwd": f"/work/{sid}"}), timeout=60)
    claims = sessions(env)
    assert set(ids) <= set(claims)
    # 8 sessions over 4 voices: two each
    counts = {v: [c["voice"] for c in claims.values()].count(v) for v in tts_common.VOICES}
    assert set(counts.values()) == {2}, counts
    for sid in ids:
        hook(env, "end", {"session_id": sid})
    assert not set(ids) & set(sessions(env))


def test_off_unloads_and_keeps_hooks_quiet(env):
    result = run(env, "tts_cli.py", "off")
    assert result.returncode == 0, result
    assert wait_for(lambda: call("/health") is None, 15), "daemon still running after off"
    hook(env, "stop", {"session_id": "session-one", "last_assistant_message": "Quiet."})
    hook(env, "warm", {"session_id": "session-three"})
    time.sleep(2)
    assert call("/health") is None, "a hook started the daemon while TTS was off"


def test_on_brings_it_back(env):
    result = run(env, "tts_cli.py", "on")
    assert result.returncode == 0 and "TTS is on" in result.stdout, result
    assert wait_for(ready), "daemon did not come back after on"
    status = run(env, "tts_cli.py", "status")
    assert "TTS is on" in status.stdout and "running on CPU" in status.stdout, status


def test_session_end_releases_the_voice(env):
    hook(env, "end", {"session_id": "session-one"})
    hook(env, "end", {"session_id": "session-two"})
    assert sessions(env) == {}
