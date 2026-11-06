"""Settings, paths and logging shared by the TTS daemon, the hooks, the CLI and the installer."""

import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CONFIG_PATH = HERE / "config.json"
LOG_DIR = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "claude-code-tts"
MODEL_DIR = Path.home() / ".local" / "share" / "kokoro-tts"

# kokoro: Kokoro-82M on a CUDA GPU. paradee: Paradee-8M (distilled from Kokoro) on the CPU.
ENGINES = ("kokoro", "paradee")
# Used when the configured voice is not one the engine has; Paradee has only four voices
DEFAULT_VOICES = {"kokoro": "af_sky", "paradee": "af_heart"}

DEFAULTS = {
    # false keeps the hooks quiet; `claude-tts off` and `claude-tts on` flip it
    "enabled": True,
    "engine": "kokoro",
    "port": 7457,
    # Kokoro only. A GPU UUID (stable) or an nvidia-smi index; install.py fills in the UUID
    "cuda_device": "0",
    "model": str(MODEL_DIR / "kokoro-v1.0.onnx"),
    "voices": str(MODEL_DIR / "voices-v1.0.bin"),
    # Paradee only. Local model files (install.py fills these in); null downloads from Hugging Face
    "paradee_model": None,
    "paradee_config": None,
    "voice": "af_sky",
    # Give each concurrent Claude Code session its own voice: the first gets "voice",
    # later ones the least used of "session_voices" (`claude-tts sessions on`)
    "voice_per_session": False,
    # Voices both engines have: American and British, female and male
    "session_voices": ["af_heart", "bm_fable", "am_puck", "bf_isabella"],
    "speed": 1.0,
    "volume": 1.0,
    # Substring of an output device name; null follows the Windows default device
    "output_device": None,
    "max_chars": 5000,
    "narrate_tool_use": True,
}

MAX_LOG_BYTES = 1_000_000


def read_config_file() -> dict:
    """config.json as written, without defaults or environment overrides."""
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    try:
        cfg.update(read_config_file())
    except ValueError as error:
        # A typo in a hand-edited config must not crash every hook
        log("hook", f"ignoring {CONFIG_PATH}: {error}")
    # Same variables as the upstream bash hooks, settable under "env" in settings.json
    if os.environ.get("KOKORO_VOICE"):
        cfg["voice"] = os.environ["KOKORO_VOICE"]
    if os.environ.get("KOKORO_SPEED"):
        cfg["speed"] = float(os.environ["KOKORO_SPEED"])
    return cfg


def save_config(changes: dict) -> dict:
    """Merge changes into config.json and return the file's new contents.

    Writes a temporary file and renames it over the old one, so a hook that
    reads the config at the same moment never sees half a file.
    """
    cfg = read_config_file()
    cfg.update(changes)
    tmp = CONFIG_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    tmp.replace(CONFIG_PATH)
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
