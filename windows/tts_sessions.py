"""One voice per Claude Code session, so you can tell concurrent sessions apart by ear.

Every hook gets the session's id from Claude Code. With "voice_per_session" on,
a session's first hook claims the voice from the rotation ([voice] followed by
"session_voices") that the fewest other live sessions are using, and keeps it
until SessionEnd. Claims live in %LOCALAPPDATA%\\claude-code-tts\\sessions.json,
shared by every session's hooks under a file lock.
"""

import json
import msvcrt
import os
import time
from contextlib import contextmanager
from pathlib import Path

from tts_common import LOG_DIR

SESSIONS_PATH = LOG_DIR / "sessions.json"
LOCK_PATH = LOG_DIR / "sessions.lock"
# A session that crashed never sends SessionEnd; let its voice go after this long idle
IDLE_SECONDS = 12 * 3600


def rotation(cfg: dict) -> list[str]:
    """The voices sessions take in turn, the configured voice first."""
    return [cfg["voice"]] + [v for v in cfg["session_voices"] if v != cfg["voice"]]


@contextmanager
def _locked():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fd = os.open(LOCK_PATH, os.O_RDWR | os.O_CREAT)
    try:
        msvcrt.locking(fd, msvcrt.LK_LOCK, 1)  # retries for up to 10 s, then OSError
        try:
            yield
        finally:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    finally:
        os.close(fd)


def _read() -> dict:
    try:
        sessions = json.loads(SESSIONS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}
    now = time.time()
    return {k: v for k, v in sessions.items() if now - v.get("seen", 0) < IDLE_SECONDS}


def _write(sessions: dict) -> None:
    tmp = SESSIONS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(sessions, indent=2) + "\n", encoding="utf-8")
    tmp.replace(SESSIONS_PATH)


def active() -> dict:
    """{session id: {"voice", "project", "seen"}} for the sessions still alive."""
    return _read()


def voice_for(cfg: dict, session: str, cwd: str | None) -> str:
    """The session's voice, claiming the least used one if it has none yet."""
    pool = rotation(cfg)
    with _locked():
        sessions = _read()
        entry = sessions.get(session)
        if not entry or entry.get("voice") not in pool:
            taken = [v["voice"] for k, v in sessions.items() if k != session]
            # Fewest sessions first; on a tie, the earlier voice in the rotation
            entry = {"voice": min(pool, key=lambda v: (taken.count(v), pool.index(v)))}
        entry["seen"] = time.time()
        entry["project"] = Path(cwd).name if cwd else entry.get("project", "")
        sessions[session] = entry
        _write(sessions)
    return entry["voice"]


def forget(session: str) -> None:
    """Release the session's voice (SessionEnd)."""
    with _locked():
        sessions = _read()
        if sessions.pop(session, None) is not None:
            _write(sessions)
