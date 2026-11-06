# Windows / CUDA port

The upstream hooks are macOS bash scripts that start a fresh `kokoro-tts` CPU
process for every reply. This port keeps their behaviour (speak the
`TTS_SUMMARY`, narrate the text before tool calls, stop on a new prompt) but
runs Kokoro in a small daemon that stays loaded on one GPU:

- `tts_daemon.py` loads the model on the GPU named in `config.json`, listens on
  `127.0.0.1:7457` and plays speech on the default output device. It re-checks
  the default device for every utterance, so switching to a headset works.
- `tts_hook.py` is the Claude Code hook (`warm`, `stop`, `pretool`, `interrupt`,
  `end`). It starts the daemon if needed and sends it text over HTTP.
- `install.py` finds the GPU, downloads the model, writes `config.json` and
  merges the hooks into `~/.claude/settings.json`.

## Install

```
uv sync --project windows --managed-python
uv run --project windows windows/install.py --gpu 4090
```

Then start a new Claude Code session.

## Configure

Edit `windows/config.json`: `voice` (e.g. `af_heart`, `bf_emma`, `am_michael`),
`speed`, `volume`, `output_device` (part of a device name, or `null` for the
Windows default), `narrate_tool_use`, `cuda_device` (GPU UUID). Restart the
daemon after editing:

```
curl -X POST http://127.0.0.1:7457/shutdown
```

The next hook starts it again. `KOKORO_VOICE` and `KOKORO_SPEED` in the `env`
block of `settings.json` override the voice and speed without a restart.

## Troubleshoot

- Logs: `%LOCALAPPDATA%\claude-code-tts\hook.log` and `daemon.log`
- Status: `curl http://127.0.0.1:7457/health` (look for `CUDAExecutionProvider`)
- Which GPU: `nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name --format=csv`
- Uninstall: restore `~/.claude/settings.json.bak-tts`, remove the TTS section
  from `~/.claude/CLAUDE.md`, and POST `/shutdown`.

Audio ducking (`scripts/audio-duck.sh`) is macOS-only and is not used here.
