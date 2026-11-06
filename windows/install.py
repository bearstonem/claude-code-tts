"""Install the Windows/CUDA TTS hooks for Claude Code.

    uv run --project windows windows/install.py [--gpu 4090] [--voice af_sky]

Pins the daemon to the named GPU, downloads the Kokoro model if needed, merges
the hooks into ~/.claude/settings.json (backing it up first) and appends the
TTS_SUMMARY instructions to ~/.claude/CLAUDE.md. Safe to re-run.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from tts_common import CONFIG_PATH, DEFAULTS, HERE, MODEL_DIR

CLAUDE_DIR = Path.home() / ".claude"
MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
HOOK_EVENTS = {
    # event: (hook command, extra hook settings)
    "SessionStart": ("warm", {"async": True, "timeout": 10}),
    "UserPromptSubmit": ("interrupt", {"timeout": 5}),
    "PreToolUse": ("pretool", {"async": True, "timeout": 30}),
    "Stop": ("stop", {"async": True, "timeout": 120}),
    "SessionEnd": ("end", {"timeout": 5}),
}
# install.sh's text shows the markers as a literal HTML comment, but Claude Code
# strips HTML comments from CLAUDE.md before the model reads it, so the format
# never arrives. Describe the markers in words instead (and drop the ASCII rule).
CLAUDE_MD_SECTION = re.compile(r"\n*## TTS Summary Instructions\n.*?(?=\n## |\Z)", re.DOTALL)
CLAUDE_MD_BLOCK = """
## TTS Summary Instructions

Your replies are read aloud by a text-to-speech hook. At the END of EVERY response,
add a short spoken summary wrapped in an HTML comment so it stays out of the way on screen:

- First line: an HTML comment opener immediately followed by the word TTS_SUMMARY
- Then 1-2 plain sentences saying what was accomplished, like you're talking to someone
- Last line: the word TTS_SUMMARY followed by an HTML comment closer

(The literal markers are not written out here because HTML comments are removed
from this file before you read it.)

Keep the summary conversational and avoid:
- URLs (say "a link" instead)
- File paths (say "the configuration file" instead)
- Technical constants or variable names
- Code syntax

The technical response should come BEFORE this summary.
"""


def find_gpu(name: str) -> str:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    gpus = [[part.strip() for part in line.split(",")] for line in out.splitlines() if line]
    for index, gpu_name, uuid in gpus:
        if name.lower() in gpu_name.lower():
            print(f"GPU: {gpu_name} (nvidia-smi index {index}, {uuid})")
            return uuid
    sys.exit(f"No GPU matching {name!r}; found {[g[1] for g in gpus]}")


def download_models() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("kokoro-v1.0.onnx", "voices-v1.0.bin"):
        target = MODEL_DIR / name
        if target.exists():
            continue
        print(f"Downloading {name}...")
        part = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(MODEL_URL + name) as resp, part.open("wb") as fp:
            shutil.copyfileobj(resp, fp)
        part.replace(target)


def write_config(gpu_uuid: str, voice: str | None) -> None:
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    cfg["cuda_device"] = gpu_uuid
    if voice:
        cfg["voice"] = voice
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"Config: {CONFIG_PATH}")


def shell_path(path: Path) -> str:
    # Forward slashes work in Git Bash and PowerShell; quoting is only needed for
    # spaces, and a quoted first word would not run under PowerShell
    text = path.as_posix()
    return f'"{text}"' if " " in text else text


def install_hooks() -> None:
    settings_path = CLAUDE_DIR / "settings.json"
    settings = {}
    if settings_path.exists():
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        # Keep the first backup: on a re-run it is still the pre-TTS original
        backup = settings_path.with_name("settings.json.bak-tts")
        if not backup.exists():
            shutil.copy2(settings_path, backup)
            print(f"Backed up settings to {backup}")
    python = shell_path(HERE / ".venv" / "Scripts" / "python.exe")
    hook = shell_path(HERE / "tts_hook.py")
    hooks = settings.setdefault("hooks", {})
    for event, (command, extra) in HOOK_EVENTS.items():
        groups = []
        # Drop earlier installs of these hooks, Windows or upstream bash, keep the rest
        for group in hooks.get(event, []):
            kept = [
                h
                for h in group.get("hooks", [])
                if "tts_hook.py" not in h.get("command", "")
                and "/.claude/hooks/tts-" not in h.get("command", "")
            ]
            if kept:
                groups.append({**group, "hooks": kept})
        entry = {"type": "command", "command": f"{python} {hook} {command}", **extra}
        groups.append({"hooks": [entry]})
        hooks[event] = groups
    settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print(f"Hooks: {settings_path}")


def install_claude_md() -> None:
    path = CLAUDE_DIR / "CLAUDE.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    # Replace an earlier version of the section, ours or install.sh's
    rest = CLAUDE_MD_SECTION.sub("", existing).rstrip()
    text = (rest or "# Global Claude Code Instructions") + "\n" + CLAUDE_MD_BLOCK
    if text != existing:
        path.write_text(text, encoding="utf-8")
    print(f"TTS instructions: {path}")


def restart_daemon_and_test() -> None:
    sys.path.insert(0, str(HERE))
    import tts_hook

    if tts_hook.request("/shutdown", {}, timeout=2) is not None:
        time.sleep(1)  # let the old daemon release the port
    tts_hook.start_daemon()
    if not tts_hook.wait_ready(90):
        sys.exit("Daemon did not start; see %LOCALAPPDATA%\\claude-code-tts\\daemon.log")
    print(f"Daemon: {tts_hook.request('/health')}")
    tts_hook.speak("Text to speech is set up on the forty ninety.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gpu", default="4090", help="substring of the GPU name")
    parser.add_argument("--voice", help="Kokoro voice, e.g. af_sky, af_heart, bm_george")
    args = parser.parse_args()
    gpu_uuid = find_gpu(args.gpu)
    download_models()
    write_config(gpu_uuid, args.voice)
    install_hooks()
    install_claude_md()
    restart_daemon_and_test()
    print("Done. Restart Claude Code sessions to load the hooks.")


if __name__ == "__main__":
    main()
