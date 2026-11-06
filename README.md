# Claude Code TTS

Claude Code reads its replies aloud as you work. Speech comes from
[Paradee-8M](https://www.sahilmahendrakar.com/writings/paradee-1-2), an
8-million-parameter text-to-speech model (8.5 MB) that runs on the CPU, about
30 times faster than real time on one core of a desktop CPU. No GPU needed; an
NVIDIA GPU is optional.

Works on Windows, Linux and macOS.

## Features

- **Speaks a short summary of each reply**: Claude ends every response with a
  hidden spoken summary, so you hear what was done, not the code.
- **Narrates as it works**: the text Claude writes before running a tool is read out too.
- **A voice per session**: with several Claude Code sessions open, each speaks
  in its own voice, so you can tell them apart by ear.
- **Stops when you move on**: a new prompt silences that session's speech.
- **`claude-tts on` / `claude-tts off`**: mute and unmute every session at once.
- **Fast and local**: the model stays loaded in a small background process, so
  speech starts within a fraction of a second, and nothing leaves your machine.
- **Clean speech**: Markdown, code, links and file paths are stripped before speaking.
- **Audio ducking (macOS)**: lowers Apple Music's volume while Claude speaks.

## Requirements

- [Claude Code](https://claude.com/claude-code)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Linux only: PortAudio, for playing audio
  - Debian/Ubuntu: `sudo apt install libportaudio2`
  - Fedora: `sudo dnf install portaudio`
  - Arch: `sudo pacman -S portaudio`

Windows (x64), Linux (x64 and ARM64) and macOS (Apple Silicon) are supported.

## Install

```
git clone https://github.com/bearstonem/claude-code-tts
cd claude-code-tts
uv run install.py
```

`uv run` creates the environment (Python 3.12 included, about 400 MB) and the
installer:

1. downloads Paradee from Hugging Face,
2. writes `config.json`,
3. adds the hooks to `~/.claude/settings.json` (backing it up to
   `settings.json.bak-tts` first),
4. adds the spoken-summary instructions to `~/.claude/CLAUDE.md`,
5. puts the `claude-tts` command in `~/.local/bin`,
6. starts the speech daemon and says "Text to speech is set up."

Start a new Claude Code session and it will talk. Re-running the installer is safe.

Options: `--voice bm_fable` picks the main voice, and `--gpu` runs on an NVIDIA
GPU (see below).

## Use

```
claude-tts off            # stop talking now and stay quiet (also unloads the model)
claude-tts on             # talk again
claude-tts toggle
claude-tts status
claude-tts voice          # list the voices
claude-tts voice bm_fable # switch and hear a sample
claude-tts sessions       # which session has which voice
claude-tts sessions off   # one voice for every session
claude-tts restart        # after editing config.json
```

These apply to every Claude Code session at once. Inside Claude Code, prefix
them with `!` (for example `! claude-tts off`) to run them without a model turn.

## Voices

| Voice | Accent |
| --- | --- |
| `af_heart` | American, female (default) |
| `am_puck` | American, male |
| `bf_isabella` | British, female |
| `bm_fable` | British, male |

### A voice per session

Each Claude Code session takes the voice that the fewest other open sessions
are using. The first session gets your main `voice`, then `bm_fable`,
`am_puck` and `bf_isabella` (set `session_voices` in `config.json` to change
the list). A session keeps its voice until it ends, and sessions that crashed
let theirs go after 12 idle hours. `claude-tts sessions` lists them by project
folder; `claude-tts sessions off` gives every session the main voice.

## GPU (optional)

Paradee is small enough that the CPU is plenty; on an RTX 4090 it runs only
about a quarter faster than on one CPU core, since most of the time goes to
turning text into phonemes. To move the work to an NVIDIA GPU anyway:

```
uv run install.py --gpu        # the first GPU
uv run install.py --gpu 4090   # the GPU whose name contains "4090"
uv run install.py --cpu        # back to the CPU
```

`--gpu` sets up a separate environment in `gpu/` with the CUDA runtime (about
1 GB, as pip packages; no CUDA toolkit needed) and pins the daemon to that GPU.
It needs an NVIDIA driver and works on Windows and Linux.

## Where it works

The hooks run wherever Claude Code runs on your own machine:

- the `claude` command in a terminal,
- the **Claude desktop app's Code tab** (local sessions),
- the VS Code and JetBrains extensions.

All of them read `~/.claude/settings.json` and `~/.claude/CLAUDE.md`. It does
not work in cloud sessions (claude.ai/code or a desktop session in a cloud
environment), which do not read your local settings, or in ordinary Claude
chats, which have no hooks. For a desktop session over SSH, install it on the
remote machine; the audio then plays there.

## How it works

- **`tts_daemon.py`** loads Paradee once and listens on `127.0.0.1:7457`. It
  plays speech on the default output device, re-checking it for every reply so
  switching to headphones works. A new reply replaces the one playing.
- **`tts_hook.py`** is the Claude Code hook for SessionStart (start the daemon
  early), PreToolUse (narrate), Stop (speak the summary), UserPromptSubmit
  (silence) and SessionEnd (silence and release the voice). Hooks never write
  to stdout and always exit 0, so a speech problem cannot block Claude.
- **`tts_sessions.py`** hands out the per-session voices.
- **`tts_cli.py`** is the `claude-tts` command; **`install.py`** the installer.
- **`scripts/strip_markdown.py`** turns Markdown into speakable text.

Claude writes the spoken summary at the end of each reply inside an HTML
comment marked `TTS_SUMMARY`, so it does not show on screen. When a reply has
none, the whole reply is read.

## Configure

Settings live in `config.json` next to the installer. Run `claude-tts restart`
after editing it.

| Setting | Default | |
| --- | --- | --- |
| `enabled` | `true` | `claude-tts on` / `off` |
| `voice` | `af_heart` | the main voice |
| `voice_per_session` | `true` | a voice per Claude Code session |
| `session_voices` | all four | the voices sessions take in turn |
| `speed` | `1.0` | 0.5 to 2.0 |
| `volume` | `1.0` | |
| `output_device` | `null` | part of a device name; `null` follows the system default |
| `narrate_tool_use` | `true` | read the text written before tool calls |
| `device` | `cpu` | `cuda` after `install.py --gpu` |
| `cuda_device` | `null` | GPU UUID or nvidia-smi index |
| `threads` | `1` | CPU threads for synthesis |
| `duck_music` | `true` | macOS: lower Apple Music while speaking |
| `port` | `7457` | |

`CLAUDE_TTS_VOICE` and `CLAUDE_TTS_SPEED` in the `env` block of
`~/.claude/settings.json` override the voice and speed.

## Troubleshoot

- `claude-tts status` shows whether speech is on and what the daemon is doing.
- Logs (`hook.log`, `daemon.log`): `%LOCALAPPDATA%\claude-code-tts` on Windows,
  `~/.local/state/claude-code-tts` on Linux and macOS.
- No sound on Linux: install PortAudio (see Requirements) and re-run the installer.
- Speech in the wrong place: set `output_device` to part of the device's name.
- Claude does not end replies with a summary: start a new session so it reads
  the updated `~/.claude/CLAUDE.md`.

## Uninstall

```
uv run install.py --uninstall
```

This removes the hooks, the CLAUDE.md section and the `claude-tts` command, and
stops the daemon. Then delete this folder. The model stays in the Hugging Face
cache (`~/.cache/huggingface`).

## Development

```
uv sync
uv run ruff check .
uv run pytest
```

The tests run the real hooks and daemon on a separate port with their own
config, so they do not touch an installed copy. CI runs them on Windows, Linux
(x64 and ARM64) and macOS.

## Credits

- [Paradee](https://github.com/sahilmahendrakar/paradee) by Sahil Mahendrakar
  ([model](https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.2),
  [write-up](https://www.sahilmahendrakar.com/writings/paradee-1-2)), Apache-2.0.
- Forked from [ktaletsk/claude-code-tts](https://github.com/ktaletsk/claude-code-tts),
  which started the idea of speaking Claude Code's replies with hooks.

## License

ISC, see [LICENSE](LICENSE).
