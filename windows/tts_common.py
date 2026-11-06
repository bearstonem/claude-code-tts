"""Settings, paths and logging shared by the TTS daemon, the hooks and the installer."""

import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CONFIG_PATH = HERE / "config.json"
LOG_DIR = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "claude-code-tts"
MODEL_DIR = Path.home() / ".local" / "share" / "kokoro-tts"

DEFAULTS = {
    "port": 7457,
    # A GPU UUID (stable) or an nvidia-smi index; install.py fills in the 4090's UUID
    "cuda_device": "0",
    "model": str(MODEL_DIR / "kokoro-v1.0.onnx"),
    "voices": str(MODEL_DIR / "voices-v1.0.bin"),
    "voice": "af_sky",
    "speed": 1.0,
    "volume": 1.0,
    # Substring of an output device name; null follows the Windows default device
    "output_device": None,
    "max_chars": 5000,
    "narrate_tool_use": True,
}

MAX_LOG_BYTES = 1_000_000


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    try:
        cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    except FileNotFoundError:
        pass
    # Same variables as the upstream bash hooks, settable under "env" in settings.json
    if os.environ.get("KOKORO_VOICE"):
        cfg["voice"] = os.environ["KOKORO_VOICE"]
    if os.environ.get("KOKORO_SPEED"):
        cfg["speed"] = float(os.environ["KOKORO_SPEED"])
    return cfg


def log_path(name: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / f"{name}.log"


def log(name: str, message: str) -> None:
    path = log_path(name)
    try:
        if path.exists() and path.stat().st_size > MAX_LOG_BYTES:
            path.replace(path.with_suffix(".log.old"))
        with path.open("a", encoding="utf-8") as fp:
            fp.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    except OSError:
        pass
