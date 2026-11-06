"""claude-tts: turn Claude Code's speech on and off, and pick its voice.

  claude-tts off            stop talking now and stay quiet (also unloads the model)
  claude-tts on             talk again
  claude-tts toggle         flip between on and off
  claude-tts status         what is configured and what the daemon is doing (default)
  claude-tts voice [NAME]   list the voices, or switch to one and hear it
  claude-tts sessions [on|off]
                            give each Claude Code session its own voice (the
                            default), or list the sessions and their voices
  claude-tts restart        restart the daemon, e.g. after editing config.json

Changes apply to every Claude Code session at once; nothing needs restarting.
Inside Claude Code, run them with a leading "!", e.g. `! claude-tts off`.
To run on an NVIDIA GPU instead of the CPU: `uv run install.py --gpu`.
"""

import argparse
import sys
import time

import tts_hook as hook
import tts_sessions
from tts_common import CONFIG_PATH, LOG_DIR, VOICES, load_config, save_config

DAEMON_LOG = LOG_DIR / "daemon.log"


def fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def health() -> dict | None:
    reply = hook.request("/health", timeout=1)
    return reply if reply and "ready" in reply else None


def device(cfg: dict) -> str:
    return "GPU" if cfg["device"] == "cuda" else "CPU"


def cmd_on(args) -> int:
    save_config({"enabled": True})
    cfg = load_config()
    if health() is None:
        hook.start_daemon()  # loads in the background; the next reply waits for it
    print(f"TTS is on (voice {cfg['voice']}, {device(cfg)}).")
    return 0


def cmd_off(args) -> int:
    save_config({"enabled": False})
    hook.request("/stop", {}, timeout=1)
    # Unload the model too, freeing its memory; `on` brings it back
    hook.request("/shutdown", {}, timeout=1)
    print("TTS is off. `claude-tts on` turns it back on.")
    return 0


def cmd_toggle(args) -> int:
    return cmd_off(args) if load_config()["enabled"] else cmd_on(args)


def cmd_status(args) -> int:
    cfg = load_config()
    print(f"TTS is {'on' if cfg['enabled'] else 'off'}")
    print(f"voice   {cfg['voice']}, speed {cfg['speed']}, on the {device(cfg)}")
    if cfg["voice_per_session"]:
        print(f"voices  one per session: {', '.join(tts_sessions.rotation(cfg))}")
    state = health()
    if state is None:
        when = " (starts with the next reply)" if cfg["enabled"] else ""
        print(f"daemon  not running{when}")
    elif not state["ready"]:
        print(f"daemon  pid {state['pid']}, loading the model")
    else:
        where = (state.get("providers") or ["?"])[0].replace("ExecutionProvider", "")
        speaking = ", speaking" if state.get("speaking") else ""
        print(f"daemon  pid {state['pid']}, running on {where}{speaking}")
        if state.get("device") != cfg["device"]:
            print(f"        config.json says {cfg['device']}; `claude-tts restart` switches")
    print(f"config  {CONFIG_PATH}")
    return 0


def cmd_voice(args) -> int:
    cfg = load_config()
    if not args.name:
        print(f"{cfg['voice']} (voices: {', '.join(VOICES)})")
        return 0
    save_config({"voice": args.name})
    print(f"Voice: {args.name}")
    if cfg["voice_per_session"]:
        print("  (first in line for new sessions; running sessions keep theirs)")
    if cfg["enabled"]:
        # af_heart -> "Heart", bm_fable -> "Fable"
        hook.speak(f"Hi, I'm {args.name.split('_', 1)[-1].capitalize()}.", voice=args.name)
    return 0


def ago(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes // 60}h ago" if minutes >= 60 else f"{minutes}m ago"


def cmd_sessions(args) -> int:
    if args.state:
        save_config({"voice_per_session": args.state == "on"})
    cfg = load_config()
    if not cfg["voice_per_session"]:
        print(f"Every session speaks as {cfg['voice']}. `claude-tts sessions on` "
              "gives each its own voice.")
        return 0
    print(f"Each session gets its own voice, in turn: {', '.join(tts_sessions.rotation(cfg))}")
    print("(set the list with \"session_voices\" in config.json)")
    sessions = tts_sessions.active()
    if args.state == "on":
        print("Running sessions pick theirs at their next reply.")
    elif sessions:
        now = time.time()
        for sid, entry in sorted(sessions.items(), key=lambda kv: kv[1]["seen"]):
            name = entry.get("project") or "?"
            print(f"  {entry['voice']:<12} {name:<28} {sid[:8]}  {ago(now - entry['seen'])}")
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
    "voice": (cmd_voice, "list the voices, or switch to one"),
    "sessions": (cmd_sessions, "one voice per Claude Code session (on/off), or list them"),
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
        if name == "voice":
            cmd.add_argument("name", nargs="?", choices=VOICES)
        elif name == "sessions":
            cmd.add_argument("state", nargs="?", choices=("on", "off"))
    args = parser.parse_args(argv)
    handler = COMMANDS[args.command or "status"][0]
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
