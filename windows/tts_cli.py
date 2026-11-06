"""claude-tts: turn Claude Code's speech on and off, and pick the engine and voice.

  claude-tts off            stop talking now and stay quiet (also unloads the model)
  claude-tts on             talk again
  claude-tts toggle         flip between on and off
  claude-tts status         what is configured and what the daemon is running (default)
  claude-tts engine [NAME]  show or switch the engine: kokoro (GPU) or paradee (CPU)
  claude-tts voice [NAME]   list the voices, or switch to one and hear it
  claude-tts restart        restart the daemon, e.g. after editing config.json

Changes apply to every Claude Code session at once; nothing needs restarting.
Inside Claude Code, run them with a leading "!", e.g. `! claude-tts off`.
"""

import argparse
import sys
import textwrap
from pathlib import Path

import install
import tts_hook as hook
from tts_common import CONFIG_PATH, ENGINES, LOG_DIR, load_config, save_config

DAEMON_LOG = LOG_DIR / "daemon.log"


def fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def health() -> dict | None:
    reply = hook.request("/health", timeout=1)
    return reply if reply and "ready" in reply else None


def cmd_on(args) -> int:
    save_config({"enabled": True})
    cfg = load_config()
    if health() is None:
        hook.start_daemon()  # loads in the background; the next reply waits for it
    print(f"TTS is on ({cfg['engine']}, voice {cfg['voice']}).")
    return 0


def cmd_off(args) -> int:
    save_config({"enabled": False})
    hook.request("/stop", {}, timeout=1)
    # Unload the model too (frees the GPU memory Kokoro holds); `on` brings it back
    hook.request("/shutdown", {}, timeout=1)
    print("TTS is off. `claude-tts on` turns it back on.")
    return 0


def cmd_toggle(args) -> int:
    return cmd_off(args) if load_config()["enabled"] else cmd_on(args)


def cmd_status(args) -> int:
    cfg = load_config()
    print(f"TTS is {'on' if cfg['enabled'] else 'off'}")
    print(f"engine  {cfg['engine']}, voice {cfg['voice']}, speed {cfg['speed']}")
    state = health()
    if state is None:
        when = " (starts with the next reply)" if cfg["enabled"] else ""
        print(f"daemon  not running{when}")
    elif not state["ready"]:
        print(f"daemon  pid {state['pid']}, loading {state.get('engine', '')}")
    else:
        where = (state.get("providers") or ["?"])[0].replace("ExecutionProvider", "")
        speaking = ", speaking" if state.get("speaking") else ""
        print(f"daemon  pid {state['pid']}, {state.get('engine')} on {where}{speaking}")
        if state.get("engine") != cfg["engine"]:
            print(f"        config.json says {cfg['engine']}; `claude-tts restart` switches")
    print(f"config  {CONFIG_PATH}")
    return 0


def cmd_engine(args) -> int:
    cfg = load_config()
    if not args.name:
        print(f"{cfg['engine']} (choose from {', '.join(ENGINES)})")
        return 0
    name = args.name
    changes: dict = {"engine": name}
    if name == "kokoro" and not Path(cfg["model"]).exists():
        return fail(f"Kokoro is not downloaded; run: uv run --project {install.HERE} "
                    f"{install.HERE / 'install.py'} --engine kokoro")
    if name == "paradee" and not (cfg["paradee_model"] and Path(cfg["paradee_model"]).exists()):
        changes.update(install.download_paradee())
    changes["voice"] = install.fit_voice(name, cfg["voice"], cfg)
    save_config(changes)
    if not cfg["enabled"]:
        print(f"Engine: {name}. TTS is off; `claude-tts on` to hear it.")
        return 0
    print(f"Engine: {name}. Restarting the daemon...")
    if not hook.restart_daemon(90):
        return fail(f"The daemon did not start; see {DAEMON_LOG}")
    hook.speak(f"Now speaking with {name.capitalize()}.", voice=changes["voice"])
    return 0


def known_voices(cfg: dict) -> list[str] | None:
    state = health()
    if state and state["ready"] and state.get("engine") == cfg["engine"]:
        return state["voices"]
    return install.engine_voices(cfg["engine"], cfg)


def cmd_voice(args) -> int:
    cfg = load_config()
    voices = known_voices(cfg)
    if not args.name:
        print(f"{cfg['voice']} ({cfg['engine']})")
        if voices:
            print(textwrap.fill(", ".join(voices), width=78, initial_indent="  ",
                                subsequent_indent="  "))
        return 0
    if voices is not None and args.name not in voices:
        return fail(f"{cfg['engine']} has no voice {args.name!r}; it has {', '.join(voices)}")
    save_config({"voice": args.name})
    print(f"Voice: {args.name}")
    if cfg["enabled"]:
        # af_heart -> "Heart", bm_fable -> "Fable"
        hook.speak(f"Hi, I'm {args.name.split('_', 1)[-1].capitalize()}.", voice=args.name)
    return 0


def cmd_restart(args) -> int:
    if not hook.restart_daemon(90):
        return fail(f"The daemon did not start; see {DAEMON_LOG}")
    return cmd_status(args)


COMMANDS = {
    "on": (cmd_on, "talk again"),
    "off": (cmd_off, "stop talking now and stay quiet"),
    "toggle": (cmd_toggle, "flip between on and off"),
    "status": (cmd_status, "show what is configured and running"),
    "engine": (cmd_engine, "show or switch the engine"),
    "voice": (cmd_voice, "list the voices, or switch to one"),
    "restart": (cmd_restart, "restart the daemon"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="claude-tts",
        description=__doc__.split("\n\n")[0],
        epilog=__doc__.split("\n\n", 1)[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="command")
    for name, (_, help_text) in COMMANDS.items():
        cmd = sub.add_parser(name, help=help_text)
        if name == "engine":
            cmd.add_argument("name", nargs="?", choices=ENGINES)
        elif name == "voice":
            cmd.add_argument("name", nargs="?")
    args = parser.parse_args(argv)
    handler = COMMANDS[args.command or "status"][0]
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
