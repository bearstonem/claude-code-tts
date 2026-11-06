"""Install the Claude Code text-to-speech hooks (Windows, Linux, macOS).

    uv run install.py                   # Paradee on the CPU
    uv run install.py --voice bm_fable  # pick the main voice
    uv run install.py --gpu [NAME]      # optional: an NVIDIA GPU (NAME: part of its name)
    uv run install.py --cpu             # back to the CPU
    uv run install.py --uninstall

Downloads Paradee-8M, writes config.json, merges the hooks into
~/.claude/settings.json (backing it up first), adds the TTS_SUMMARY instructions
to ~/.claude/CLAUDE.md and puts a `claude-tts` command in ~/.local/bin, then
starts the daemon and says a test sentence. Safe to re-run; without --gpu or
--cpu it keeps the configured device.
"""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from tts_common import (
    CONFIG_PATH,
    DEFAULTS,
    GPU_VENV,
    HERE,
    VENV,
    VOICES,
    download_model,
    read_config_file,
    venv_python,
)

CLAUDE_DIR = Path.home() / ".claude"
BIN_DIR = Path.home() / ".local" / "bin"  # uv puts its own tools here, so it is usually on PATH
HOOK_EVENTS = {
    # event: (hook command, extra hook settings)
    "SessionStart": ("warm", {"async": True, "timeout": 10}),
    "UserPromptSubmit": ("interrupt", {"timeout": 5}),
    "PreToolUse": ("pretool", {"async": True, "timeout": 30}),
    "Stop": ("stop", {"async": True, "timeout": 120}),
    "SessionEnd": ("end", {"timeout": 5}),
}
# Claude Code strips HTML comments from CLAUDE.md before the model reads it, so
# the summary markers are described in words rather than shown
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
PORTAUDIO_HELP = """PortAudio, which plays the audio, is not installed. Install it, then re-run:
  Debian/Ubuntu: sudo apt install libportaudio2
  Fedora:        sudo dnf install portaudio
  Arch:          sudo pacman -S portaudio"""


def check_audio() -> None:
    try:
        import sounddevice as sd

        sd.query_devices(kind="output")
    except OSError:  # Linux without the PortAudio library (Windows and macOS bundle it)
        sys.exit(PORTAUDIO_HELP)
    except Exception as error:  # e.g. no sound server running yet, or no output device
        print(f"Warning: no audio output right now ({error}).")
        print("  Speech stays silent until there is one; nothing needs re-installing.")


def find_gpu(name: str | None) -> str:
    """UUID of the first NVIDIA GPU whose name contains name (any GPU when empty)."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    gpus = [[part.strip() for part in line.split(",")] for line in out.splitlines() if line]
    for index, gpu_name, uuid in gpus:
        if not name or name.lower() in gpu_name.lower():
            print(f"GPU: {gpu_name} (nvidia-smi index {index}, {uuid})")
            return uuid
    sys.exit(f"No GPU matching {name!r}; found {[g[1] for g in gpus]}")


def setup_gpu_runtime() -> None:
    """Create gpu/.venv with onnxruntime-gpu and check that CUDA works in it."""
    uv = shutil.which("uv")
    if not uv:
        sys.exit("--gpu installs the GPU runtime with uv, which is not on PATH.")
    if not shutil.which("nvidia-smi"):
        sys.exit("--gpu needs an NVIDIA GPU and its driver (nvidia-smi was not found).")
    print("Setting up the GPU runtime in gpu/ (about 1 GB of CUDA libraries the first time)...")
    # `uv run` points VIRTUAL_ENV at the main environment; the GPU one is separate
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    subprocess.run([uv, "sync", "--quiet", "--project", str(HERE / "gpu")], check=True, env=env)
    probe = subprocess.run(
        [
            str(venv_python(GPU_VENV)),
            "-c",
            "import onnxruntime as rt; getattr(rt, 'preload_dlls', lambda: None)(); "
            "print(rt.get_available_providers())",
        ],
        capture_output=True,
        text=True,
    )
    if "CUDAExecutionProvider" not in probe.stdout:
        sys.exit(f"CUDA does not work in the GPU runtime:\n{probe.stdout}{probe.stderr}")


def write_config(changes: dict) -> None:
    # Every setting, in the documented order; settings older versions had are dropped
    cfg = {**DEFAULTS, **read_config_file(), **changes}
    cfg = {key: cfg[key] for key in DEFAULTS}
    tmp = CONFIG_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    tmp.replace(CONFIG_PATH)
    print(f"Config: {CONFIG_PATH}")


def hook_python() -> Path:
    python = venv_python(VENV)
    return python if python.exists() else Path(sys.executable)


def shell_path(path: Path) -> str:
    if os.name != "nt":
        return shlex.quote(str(path))
    # Forward slashes work in Git Bash and PowerShell; quoting is only needed for
    # spaces, and a quoted first word would not run under PowerShell
    text = path.as_posix()
    return f'"{text}"' if " " in text else text


def is_ours(hook: dict) -> bool:
    # This project's hook, or one from the original bash version in ~/.claude/hooks
    command = hook.get("command", "")
    return "tts_hook.py" in command or "/.claude/hooks/tts-" in command


def edit_hooks(add: bool) -> None:
    """Drop earlier installs of these hooks, keep everyone else's, and add ours."""
    settings_path = CLAUDE_DIR / "settings.json"
    settings = {}
    if settings_path.exists():
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        # Keep the first backup: on a re-run it is still the original
        backup = settings_path.with_name("settings.json.bak-tts")
        if add and not backup.exists():
            shutil.copy2(settings_path, backup)
            print(f"Backed up settings to {backup}")
    elif not add:
        return
    hooks = settings.setdefault("hooks", {})
    command_base = f"{shell_path(hook_python())} {shell_path(HERE / 'tts_hook.py')}"
    for event, (command, extra) in HOOK_EVENTS.items():
        groups = []
        for group in hooks.get(event, []):
            kept = [h for h in group.get("hooks", []) if not is_ours(h)]
            if kept:
                groups.append({**group, "hooks": kept})
        if add:
            entry = {"type": "command", "command": f"{command_base} {command}", **extra}
            groups.append({"hooks": [entry]})
        if groups:
            hooks[event] = groups
        else:
            hooks.pop(event, None)
    if not hooks:
        settings.pop("hooks")
    CLAUDE_DIR.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print(f"Hooks: {settings_path}")


def edit_claude_md(add: bool) -> None:
    path = CLAUDE_DIR / "CLAUDE.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    rest = CLAUDE_MD_SECTION.sub("", existing).rstrip()
    if add:
        text = (rest or "# Global Claude Code Instructions") + "\n" + CLAUDE_MD_BLOCK
    else:
        text = rest + "\n" if rest else ""
    if text != existing:
        CLAUDE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(f"TTS instructions: {path}")


def install_cli() -> None:
    """`claude-tts`: a shell script, plus a .cmd for cmd and PowerShell on Windows."""
    python, cli = hook_python(), HERE / "tts_cli.py"
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    script = BIN_DIR / "claude-tts"
    py, cl = (python.as_posix(), cli.as_posix()) if os.name == "nt" else (str(python), str(cli))
    script.write_bytes(f'#!/bin/sh\nexec "{py}" "{cl}" "$@"\n'.encode())
    script.chmod(0o755)
    if os.name == "nt":
        (BIN_DIR / "claude-tts.cmd").write_text(f'@"{python}" "{cli}" %*\r\n', encoding="utf-8")
    print(f"Command: {script} (on, off, status, voice, sessions, restart)")
    on_path = any(
        Path(entry).expanduser().resolve() == BIN_DIR.resolve()
        for entry in os.environ.get("PATH", "").split(os.pathsep)
        if entry
    )
    if not on_path:
        print(f"  {BIN_DIR} is not on PATH; add it to run claude-tts from anywhere")


def restart_daemon_and_test(gpu: bool) -> None:
    import tts_hook

    if not tts_hook.restart_daemon(120):
        sys.exit(f"The daemon did not start; see {tts_hook.log_path('daemon')}")
    print(f"Daemon: {tts_hook.request('/health')}")
    tts_hook.speak("Text to speech is set up" + (" on the GPU." if gpu else "."))


def uninstall() -> None:
    import tts_hook

    tts_hook.request("/shutdown", {}, timeout=2)
    edit_hooks(add=False)
    edit_claude_md(add=False)
    for name in ("claude-tts", "claude-tts.cmd"):
        (BIN_DIR / name).unlink(missing_ok=True)
    print("Uninstalled. The model stays in the Hugging Face cache (~/.cache/huggingface).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    where = parser.add_mutually_exclusive_group()
    where.add_argument(
        "--gpu",
        nargs="?",
        const="",
        metavar="NAME",
        help="run on an NVIDIA GPU (optional; sets up gpu/). NAME picks one by part of its "
        "name, e.g. 4090; default the first",
    )
    where.add_argument("--cpu", action="store_true", help="run on the CPU (the default)")
    where.add_argument("--uninstall", action="store_true", help="remove the hooks and command")
    parser.add_argument("--voice", choices=VOICES, help="the main voice (default af_heart)")
    args = parser.parse_args()
    if args.uninstall:
        return uninstall()

    check_audio()
    cfg = {**DEFAULTS, **read_config_file()}
    gpu = args.gpu is not None or (not args.cpu and cfg["device"] == "cuda")
    changes: dict = {"device": "cuda" if gpu else "cpu"}
    if gpu:
        setup_gpu_runtime()
        if args.gpu is not None or not cfg["cuda_device"]:
            changes["cuda_device"] = find_gpu(args.gpu)
    if args.voice:
        changes["voice"] = args.voice
    print("Fetching Paradee-8M v1.2 from Hugging Face...")
    changes["model"], changes["model_config"] = download_model(gpu)
    write_config(changes)
    edit_hooks(add=True)
    edit_claude_md(add=True)
    install_cli()
    restart_daemon_and_test(gpu)
    print("Done. New Claude Code sessions will speak; restart open ones to load the hooks.")


if __name__ == "__main__":
    main()
