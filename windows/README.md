# Windows port

The upstream hooks are macOS bash scripts that start a fresh `kokoro-tts` CPU
process for every reply. This port keeps their behaviour (speak the
`TTS_SUMMARY`, narrate the text before tool calls, stop on a new prompt) but
runs the model in a small daemon that stays loaded, with a choice of two engines:

| Engine | Model | Runs on | Voices |
| --- | --- | --- | --- |
| `kokoro` | [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (kokoro-onnx) | one CUDA GPU | 54 |
| `paradee` | [Paradee-8M v1.2](https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.2), distilled from Kokoro | CPU, about 30x real time | `af_heart`, `bf_isabella`, `am_puck`, `bm_fable` |

Paradee is a tenth of Kokoro's size and needs no GPU, at a small cost in
naturalness ([write-up](https://www.sahilmahendrakar.com/writings/paradee-1-2)).

- `tts_daemon.py` loads the configured engine, listens on `127.0.0.1:7457` and
  plays speech on the default output device. It re-checks the default device
  for every utterance, so switching to a headset works.
- `tts_hook.py` is the Claude Code hook (`warm`, `stop`, `pretool`, `interrupt`,
  `end`). It starts the daemon if needed and sends it text over HTTP.
- `tts_cli.py` is the `claude-tts` command: on/off, engine, voice.
- `install.py` downloads the model, writes `config.json`, merges the hooks into
  `~/.claude/settings.json` and puts `claude-tts` in `~/.local/bin`.

## Install

```
uv sync --project windows --managed-python
uv run --project windows windows/install.py --engine paradee
```

or `--engine kokoro --gpu 4090` for Kokoro on the GPU whose name contains
`4090`. Then start a new Claude Code session.

## Turn it on and off

```
claude-tts off            # stop talking now and stay quiet (also unloads the model)
claude-tts on             # talk again
claude-tts toggle
claude-tts status
claude-tts engine paradee # or kokoro; restarts the daemon
claude-tts voice          # list voices
claude-tts voice bm_fable # switch and hear a sample
claude-tts restart        # after editing config.json
```

These apply to every Claude Code session at once. Inside Claude Code, prefix
them with `!` (e.g. `! claude-tts off`) to run them without a model turn.

## Configure

`windows/config.json`: `enabled`, `engine`, `voice`, `speed`, `volume`,
`output_device` (part of a device name, or `null` for the Windows default),
`narrate_tool_use`, `cuda_device` (Kokoro's GPU UUID), and the model paths.
Run `claude-tts restart` after editing it. If the voice is not one the engine
has, the daemon uses `af_sky` (Kokoro) or `af_heart` (Paradee).

`KOKORO_VOICE` and `KOKORO_SPEED` in the `env` block of `settings.json`
override the voice and speed without a restart.

## Troubleshoot

- Logs: `%LOCALAPPDATA%\claude-code-tts\hook.log` and `daemon.log`
- Status: `claude-tts status`, or `curl http://127.0.0.1:7457/health`
- Which GPU (Kokoro): `nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name --format=csv`
- Uninstall: restore `~/.claude/settings.json.bak-tts`, remove the TTS section
  from `~/.claude/CLAUDE.md`, delete `~/.local/bin/claude-tts*` and run
  `claude-tts off` first.

Paradee comes from its GitHub repo, pinned in `pyproject.toml`; it is not on
PyPI. Its phonemizer (misaki) needs spaCy's `en_core_web_sm`, which is pinned
there too because misaki would otherwise try to `pip install` it at runtime.

Audio ducking (`scripts/audio-duck.sh`) is macOS-only and is not used here.
