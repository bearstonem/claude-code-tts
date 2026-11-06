"""Install the Windows TTS hooks for Claude Code.

    uv run --project windows windows/install.py [--engine paradee] [--gpu 4090] [--voice af_heart]

Downloads the model for the chosen engine (Kokoro pinned to the named GPU, or
Paradee on the CPU), writes config.json, merges the hooks into
~/.claude/settings.json (backing it up first), appends the TTS_SUMMARY
instructions to ~/.claude/CLAUDE.md and puts a `claude-tts` command in
~/.local/bin. Safe to re-run; without --engine it keeps the configured one.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from tts_common import (
    CONFIG_PATH,
    DEFAULT_VOICES,
    DEFAULTS,
    ENGINES,
    HERE,
    MODEL_DIR,
    read_config_file,
    save_config,
)

CLAUDE_DIR = Path.home() / ".claude"
BIN_DIR = Path.home() / ".local" / "bin"  # uv puts its own tools here, so it is usually on PATH
MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
PARADEE_VERSION = "1.2"
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


def download_kokoro() -> None:
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


def download_paradee() -> dict:
    """Fetch Paradee's int8 model (8.5 MB) into the Hugging Face cache; its paths for config.json.

    With the paths in the config the daemon starts without asking the Hub anything.
    """
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from huggingface_hub import hf_hub_download
    from paradee.tts import REPOS

    repo, revision = REPOS[PARADEE_VERSION], f"v{PARADEE_VERSION}"
    print(f"Fetching {repo} ({revision})...")
    return {
        "paradee_model": hf_hub_download(repo, "onnx/paradee_int8.onnx", revision=revision),
        "paradee_config": hf_hub_download(repo, "config.json", revision=revision),
    }


def engine_voices(engine: str, cfg: dict) -> list[str] | None:
    """The voices an engine has, or None if its files are not there to ask."""
    if engine == "paradee":
        from paradee import VOICES

        return VOICES[PARADEE_VERSION]
    if not Path(cfg["voices"]).exists():
        return None
    import numpy as np

    return sorted(np.load(cfg["voices"]).keys())


def fit_voice(engine: str, voice: str, cfg: dict) -> str:
    """voice if the engine has it, else the engine's default (Paradee has only four)."""
    voices = engine_voices(engine, cfg)
    if voices is None or voice in voices:
        return voice
    fallback = DEFAULT_VOICES[engine]
    print(f"{engine} has no voice {voice!r}; using {fallback!r} (it has {', '.join(voices)})")
    return fallback


def write_config(changes: dict) -> None:
    # Fill in every default so new settings show up in the file for editing
    save_config({**DEFAULTS, **read_config_file(), **changes})
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


def install_cli() -> None:
    """`claude-tts` for cmd and PowerShell (.cmd) and for Git Bash (no extension)."""
    python = HERE / ".venv" / "Scripts" / "python.exe"
    cli = HERE / "tts_cli.py"
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    (BIN_DIR / "claude-tts.cmd").write_text(f'@"{python}" "{cli}" %*\r\n', encoding="utf-8")
    sh = f'#!/bin/sh\nexec "{python.as_posix()}" "{cli.as_posix()}" "$@"\n'
    (BIN_DIR / "claude-tts").write_bytes(sh.encode())
    print(f"Command: {BIN_DIR / 'claude-tts'} (on, off, status, engine, voice, restart)")
    on_path = any(
        Path(entry).resolve() == BIN_DIR.resolve()
        for entry in os.environ.get("PATH", "").split(os.pathsep)
        if entry
    )
    if not on_path:
        print(f"  {BIN_DIR} is not on PATH; add it to run claude-tts from anywhere")


def restart_daemon_and_test(engine: str) -> None:
    sys.path.insert(0, str(HERE))
    import tts_hook

    if not tts_hook.restart_daemon(90):
        sys.exit("Daemon did not start; see %LOCALAPPDATA%\\claude-code-tts\\daemon.log")
    print(f"Daemon: {tts_hook.request('/health')}")
    tts_hook.speak(f"Text to speech is set up with {engine.capitalize()}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--engine",
        choices=ENGINES,
        help="kokoro (Kokoro-82M on a CUDA GPU) or paradee (Paradee-8M on the CPU); "
        "default: the configured engine",
    )
    parser.add_argument("--gpu", default="4090", help="Kokoro: substring of the GPU name")
    parser.add_argument("--voice", help="e.g. af_heart, bf_isabella, am_puck, bm_fable")
    args = parser.parse_args()
    cfg = {**DEFAULTS, **read_config_file()}
    engine = args.engine or cfg["engine"]
    changes: dict = {"engine": engine}
    if engine == "kokoro":
        changes["cuda_device"] = find_gpu(args.gpu)
        download_kokoro()
    else:
        changes.update(download_paradee())
    changes["voice"] = fit_voice(engine, args.voice or cfg["voice"], cfg)
    write_config(changes)
    install_hooks()
    install_claude_md()
    install_cli()
    restart_daemon_and_test(engine)
    print("Done. Restart Claude Code sessions to load the hooks.")


if __name__ == "__main__":
    main()
