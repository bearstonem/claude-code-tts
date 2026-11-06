"""Settings, paths and logging shared by the TTS daemon, the hooks, the CLI and the installer."""

import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Both can be pointed elsewhere, which the tests use to stay out of a real install
CONFIG_PATH = Path(os.environ.get("CLAUDE_TTS_CONFIG") or HERE / "config.json")


def _state_dir() -> Path:
    if os.environ.get("CLAUDE_TTS_STATE_DIR"):
        return Path(os.environ["CLAUDE_TTS_STATE_DIR"])
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
    else:
        base = os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state"
    return Path(base) / "claude-code-tts"


# Logs and the per-session voice claims
LOG_DIR = _state_dir()
VENV = HERE / ".venv"
# Optional NVIDIA runtime (install.py --gpu); see gpu/pyproject.toml
GPU_VENV = HERE / "gpu" / ".venv"

PARADEE_VERSION = "1.2"
# Paradee 1.2's voices (paradee.VOICES["1.2"]), here so the hooks need not import it
VOICES = ["af_heart", "bf_isabella", "am_puck", "bm_fable"]

DEFAULTS = {
    # false keeps the hooks quiet; `claude-tts off` and `claude-tts on` flip it
    "enabled": True,
    "port": 7457,
    "voice": "af_heart",
    # Each concurrent Claude Code session speaks in its own voice: the first gets
    # "voice", later ones the least used of "session_voices"
    "voice_per_session": True,
    "session_voices": ["af_heart", "bm_fable", "am_puck", "bf_isabella"],
    "speed": 1.0,
    "volume": 1.0,
    # Substring of an output device name; null follows the system default device
    "output_device": None,
    "max_chars": 5000,
    "narrate_tool_use": True,
    # "cpu", or "cuda" for an NVIDIA GPU after `install.py --gpu`
    "device": "cpu",
    # With "cuda": a GPU UUID (stable) or an nvidia-smi index; null for the first GPU
    "cuda_device": None,
    # CPU threads for synthesis; one already runs many times faster than real time
    "threads": 1,
    # Local model files (install.py fills these in); null downloads from Hugging Face
    "model": None,
    "model_config": None,
    # macOS: lower Apple Music's volume while speaking (scripts/audio-duck.sh)
    "duck_music": True,
}

MAX_LOG_BYTES = 1_000_000


def venv_python(venv: Path, windowless: bool = False) -> Path:
    """The interpreter of a venv; on Windows pythonw.exe runs without a console window."""
    if os.name == "nt":
        return venv / "Scripts" / ("pythonw.exe" if windowless else "python.exe")
    return venv / "bin" / "python"


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
    # Settable under "env" in ~/.claude/settings.json
    if os.environ.get("CLAUDE_TTS_VOICE"):
        cfg["voice"] = os.environ["CLAUDE_TTS_VOICE"]
    if os.environ.get("CLAUDE_TTS_SPEED"):
        cfg["speed"] = float(os.environ["CLAUDE_TTS_SPEED"])
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


def download_model(gpu: bool) -> tuple[str, str]:
    """Paradee's model and config from Hugging Face (cached after the first time).

    The CPU uses the 8.5 MB int8 model, which Paradee builds for it; the GPU the
    35 MB fp32 one.
    """
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from huggingface_hub import hf_hub_download
    from paradee.tts import REPOS

    repo, revision = REPOS[PARADEE_VERSION], f"v{PARADEE_VERSION}"
    name = "onnx/paradee.onnx" if gpu else "onnx/paradee_int8.onnx"
    return (
        hf_hub_download(repo, name, revision=revision),
        hf_hub_download(repo, "config.json", revision=revision),
    )


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
