"""Claude Code hooks that drive the Kokoro TTS daemon (Windows port of ../hooks/*.sh).

  tts_hook.py warm       SessionStart      start the daemon early so the model is loaded
  tts_hook.py stop       Stop              speak the reply's TTS_SUMMARY, or the whole reply
  tts_hook.py pretool    PreToolUse        speak the text Claude wrote before a tool call
  tts_hook.py interrupt  UserPromptSubmit  silence playback when a new prompt is sent
  tts_hook.py end        SessionEnd        silence playback

Hooks never print to stdout (SessionStart and UserPromptSubmit output would be added
to Claude's context) and always exit 0, so a TTS problem can never block Claude Code.
While "enabled" is false in config.json (`claude-tts off`) they speak nothing.
"""

import json
import re
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

from tts_common import HERE, MAX_LOG_BYTES, REPO, load_config, log_path
from tts_common import log as _log

sys.path.insert(0, str(REPO / "scripts"))
from strip_markdown import strip_markdown  # noqa: E402

STARTED_AT = time.time()
CFG = load_config()
BASE = f"http://127.0.0.1:{CFG['port']}"
# Never route localhost through a proxy configured in the environment or registry
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
# The closing TTS_SUMMARY is optional: CLAUDE.md can only describe the markers in
# words, so accept the closer with or without it
SUMMARY = re.compile(r"<!--\s*TTS_SUMMARY(.*?)(?:TTS_SUMMARY\s*)?-->", re.DOTALL)
# strip_markdown drops POSIX paths; Windows ones would be spelled out
WINDOWS_PATH = re.compile(r"`?\b[A-Za-z]:[\\/][^\s`]*`?")
TRANSCRIPT_TAIL_BYTES = 4_000_000


def log(message: str) -> None:
    _log("hook", message)


def request(path: str, payload: dict | None = None, timeout: float = 2.0):
    """POST payload (GET when None); the parsed reply, or None if the daemon is down."""
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with OPENER.open(req, timeout=timeout) as resp:
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as error:
        log(f"{path} -> HTTP {error.code}: {error.read()[:200]!r}")
        return {"error": error.code}
    except (OSError, ValueError):
        return None


def start_daemon() -> None:
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    exe = pythonw if pythonw.exists() else Path(sys.executable)
    path = log_path("daemon")
    try:
        if path.exists() and path.stat().st_size > MAX_LOG_BYTES:
            path.replace(path.with_suffix(".log.old"))
    except OSError:
        pass
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    error = None
    with path.open("ab") as out:
        # Break away from Claude Code's job object, if it has one, so the daemon
        # outlives this hook; not every job allows that, so fall back without it
        for extra in (subprocess.CREATE_BREAKAWAY_FROM_JOB, 0):
            try:
                subprocess.Popen(
                    [str(exe), str(HERE / "tts_daemon.py")],
                    cwd=HERE,
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=out,
                    creationflags=flags | extra,
                    close_fds=True,
                )
                log("started daemon")
                return
            except OSError as e:
                error = e
    log(f"could not start daemon: {error!r}")


def wait_ready(timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        health = request("/health", timeout=1)
        if health and health.get("ready"):
            return True
        time.sleep(0.25)
    return False


def restart_daemon(timeout: float = 90) -> bool:
    """Stop a running daemon, start a fresh one and wait until its model is loaded."""
    if request("/shutdown", {}, timeout=2) is not None:
        # Wait for the old daemon to release the port, or the new one cannot bind
        deadline = time.monotonic() + 10
        while request("/health", timeout=0.5) is not None and time.monotonic() < deadline:
            time.sleep(0.2)
    start_daemon()
    return wait_ready(timeout)


def speak(text: str, narration: bool = False, voice: str | None = None) -> None:
    # Read the switch again: `claude-tts off` may have run while this hook waited
    # for the transcript, and starting the daemon now would undo it
    if not load_config()["enabled"]:
        return
    if request("/health", timeout=1) is None:
        start_daemon()
    if not wait_ready(90):
        log("daemon did not become ready, see daemon.log")
        return
    payload = {
        "text": text[: CFG["max_chars"]],
        "voice": voice or CFG["voice"],
        "speed": CFG["speed"],
        # Lets the daemon drop this request if a newer one or a stop overtook it
        "requested_at": STARTED_AT,
        "narration": narration,
    }
    reply = request("/speak", payload, timeout=10)
    log(f"{'narration' if narration else 'reply'} ({len(payload['text'])} chars): {reply}")


def to_speech(text: str) -> str:
    summaries = SUMMARY.findall(text)
    if summaries:
        text = summaries[-1]
    return strip_markdown(WINDOWS_PATH.sub(" ", text))


def transcript_entries(transcript: str) -> list[dict]:
    """User and assistant entries near the end of the transcript, oldest first."""
    try:
        with open(transcript, "rb") as fp:
            fp.seek(0, 2)
            fp.seek(max(0, fp.tell() - TRANSCRIPT_TAIL_BYTES))
            lines = fp.read().splitlines()
    except OSError:
        return []
    entries = []
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:  # the cut-off first line, or one still being written
            continue
        if entry.get("type") in ("user", "assistant") and not entry.get("isSidechain"):
            entries.append(entry)
    return entries


def blocks(entry: dict) -> list[dict]:
    content = (entry.get("message") or {}).get("content")
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def text_before_tool(transcript: str, tool_use_id: str) -> str | None:
    """Text of the assistant message that made this tool call, written before it.

    None while the call has not reached the transcript yet, "" when there is
    nothing to say: no text, or this is not the message's first tool call.
    """
    messages = [e for e in transcript_entries(transcript) if e["type"] == "assistant"]
    for i in range(len(messages) - 1, -1, -1):
        if any(b.get("id") == tool_use_id for b in blocks(messages[i])):
            break
    else:
        return None
    # Each content block of a message is its own transcript line
    def message_id(entry: dict):
        return (entry.get("message") or {}).get("id")

    start = i
    while start > 0 and message_id(messages[start - 1]) == message_id(messages[i]):
        start -= 1
    texts = []
    for message in messages[start : i + 1]:
        for block in blocks(message):
            if block.get("type") == "tool_use":
                return " ".join(texts) if block.get("id") == tool_use_id else ""
            if block.get("type") == "text":
                texts.append(block.get("text") or "")
    return ""


def last_reply(transcript: str) -> str:
    """Fallback for Claude Code versions without last_assistant_message."""
    texts: list[str] = []
    for entry in reversed(transcript_entries(transcript)):
        if entry["type"] == "user" or any(b.get("type") == "tool_use" for b in blocks(entry)):
            break
        texts[:0] = [b.get("text") or "" for b in blocks(entry) if b.get("type") == "text"]
    return "\n".join(texts)


def on_stop(data: dict) -> None:
    text = data.get("last_assistant_message")
    if not isinstance(text, str) or not text.strip():
        time.sleep(1)  # the transcript is written asynchronously
        text = last_reply(data.get("transcript_path") or "")
    speech = to_speech(text)
    if speech:
        speak(speech)


def on_pretool(data: dict) -> None:
    # Subagent tool calls are not in the main transcript, and would be chatty anyway
    if not CFG["narrate_tool_use"] or data.get("agent_id"):
        return
    transcript, tool_use_id = data.get("transcript_path"), data.get("tool_use_id")
    if not transcript or not tool_use_id:
        return
    for _ in range(30):  # the transcript is written asynchronously and may lag
        text = text_before_tool(transcript, tool_use_id)
        if text is not None:
            break
        time.sleep(0.1)
    else:
        log(f"{tool_use_id} did not reach the transcript")
        return
    if "TTS_SUMMARY" in text:  # leave the summary to the Stop hook
        return
    speech = to_speech(text)
    if len(speech) > 10:
        speak(speech, narration=True)


def on_warm(data: dict) -> None:
    if request("/health", timeout=1) is None:
        start_daemon()


def on_silence(data: dict) -> None:
    request("/stop", {}, timeout=1)


HANDLERS = {
    "warm": on_warm,
    "stop": on_stop,
    "pretool": on_pretool,
    "interrupt": on_silence,
    "end": on_silence,
}
# Hooks that would start the daemon or speak; skipped while TTS is turned off
SPEAKING = {"warm", "stop", "pretool"}


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    handler = HANDLERS.get(command)
    if handler is None:
        log(f"unknown command {command!r}")
        return 0
    if command in SPEAKING and not CFG["enabled"]:
        return 0
    try:
        raw = sys.stdin.buffer.read() if sys.stdin else b""
        handler(json.loads(raw or b"{}"))
    except Exception:
        log(f"{command} failed:\n{traceback.format_exc()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
