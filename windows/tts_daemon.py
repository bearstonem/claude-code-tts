"""TTS daemon for Claude Code: Kokoro on one CUDA GPU, or Paradee on the CPU.

Loads the model once, keeps it resident and plays speech on the output device.
config.json's "engine" picks the model. tts_hook.py talks to it over localhost:

  POST /speak     {"text": ..., "voice": ..., "speed": ...}  replaces whatever is playing
  POST /stop      silences playback
  POST /shutdown  exits the daemon (the next hook starts a fresh one)
  GET  /health    readiness, engine, voices, execution providers, pinned device
"""

import json
import os
import queue
import re
import socket
import socketserver
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

from tts_common import DEFAULT_VOICES, ENGINES, load_config

CFG = load_config()
ENGINE = CFG["engine"] if CFG["engine"] in ENGINES else "kokoro"

if ENGINE == "kokoro":
    # Pin before onnxruntime touches CUDA. PCI bus order makes indices match nvidia-smi
    # (CUDA's default is fastest-first, which would put the 5090 at 0); UUIDs need no order.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CFG["cuda_device"])
# Windows without Developer Mode cannot symlink; the Hub cache copies instead, which is fine
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np  # noqa: E402
import sounddevice as sd  # noqa: E402

SAMPLE_RATE = 24000  # Kokoro and Paradee both produce 24 kHz mono
BLOCK = SAMPLE_RATE // 10  # write audio in 100 ms blocks so /stop takes effect quickly
GAP = np.zeros(SAMPLE_RATE // 4, dtype=np.float32)  # pause after each synthesized chunk
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# Kokoro voice names start with a letter for their language
LANGS = {
    "a": "en-us",
    "b": "en-gb",
    "e": "es",
    "f": "fr-fr",
    "h": "hi",
    "i": "it",
    "j": "ja",
    "p": "pt-br",
    "z": "cmn",
}


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", file=sys.stderr, flush=True)


@dataclass
class Engine:
    name: str
    # (text, voice, speed) -> float32 audio at SAMPLE_RATE
    create: Callable[[str, str, float], np.ndarray]
    voices: list[str]
    providers: list[str]

    def pick_voice(self, voice: str) -> str:
        if voice in self.voices:
            return voice
        fallback = DEFAULT_VOICES[self.name]
        return fallback if fallback in self.voices else self.voices[0]


def load_kokoro() -> Engine:
    import onnxruntime as rt
    from kokoro_onnx import Kokoro

    rt.preload_dlls()  # CUDA and cuDNN DLLs from the nvidia-* wheels
    rt.set_default_logger_severity(3)
    options = rt.SessionOptions()
    options.log_severity_level = 3
    session = rt.InferenceSession(
        CFG["model"],
        options,
        providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
    )
    if session.get_providers()[0] != "CUDAExecutionProvider":
        log(f"WARNING: CUDA unavailable, running on {session.get_providers()}")
    kokoro = Kokoro.from_session(session, CFG["voices"])

    def create(text: str, voice: str, speed: float) -> np.ndarray:
        audio, _ = kokoro.create(text, voice=voice, speed=speed, lang=LANGS.get(voice[0], "en-us"))
        return audio

    return Engine("kokoro", create, kokoro.get_voices(), session.get_providers())


def load_paradee() -> Engine:
    # Paradee runs its int8 model on onnxruntime's CPU provider; it is about 30x
    # faster than real time there, so it needs no GPU
    from paradee import Paradee

    model = Paradee(model_path=CFG["paradee_model"], config_path=CFG["paradee_config"])

    def create(text: str, voice: str, speed: float) -> np.ndarray:
        # Paradee picks American or British phonemes from the voice name itself
        return model(text, speed=speed, voice=voice)

    return Engine("paradee", create, list(model.voices), model.session.get_providers())


def load_model() -> Engine:
    engine = load_paradee() if ENGINE == "paradee" else load_kokoro()
    # The first run allocates memory and picks kernels (Kokoro) or loads the
    # phonemizer for the voice's accent (Paradee); do it before real speech
    engine.create("Warm up.", engine.pick_voice(CFG["voice"]), 1.0)
    return engine


def chunks(text: str, first: int = 150, rest: int = 400) -> list[str]:
    """Split at sentence ends: a short first chunk so speech starts fast, longer after."""
    out: list[str] = []
    current = ""
    for sentence in SENTENCE_END.split(text.strip()):
        limit = rest if out else first
        if current and len(current) + 1 + len(sentence) > limit:
            out.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence
    if current:
        out.append(current)
    return out


def output_device() -> int | None:
    """Re-scan devices so a change of default output (headset, VR) is picked up."""
    sd._terminate()
    sd._initialize()
    wanted = CFG.get("output_device")
    if wanted:
        for index, device in enumerate(sd.query_devices()):
            name = device["name"].lower()
            if device["max_output_channels"] > 0 and wanted.lower() in name:
                return index
        log(f"no output device matches {wanted!r}, using the default")
    return None


class Speaker:
    """Plays one utterance at a time; a newer request or a stop cancels the current one."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self.generation = 0
        self.speaking = False
        self.last_speak = 0.0
        self.last_stop = 0.0
        self.lock = threading.Lock()
        self.play_lock = threading.Lock()
        self.synth_lock = threading.Lock()

    def speak(
        self,
        text: str,
        voice: str,
        speed: float,
        requested_at: float,
        narration: bool,
    ) -> bool:
        """Start speaking unless the request was overtaken while its hook ran.

        Hooks run in the background and can arrive out of order. A reply is
        dropped only by a stop sent after it (the user moved on); narration of a
        tool call is also dropped by any newer speech, so it never talks over the
        final reply.
        """
        with self.lock:
            overtaken = max(self.last_stop, self.last_speak if narration else 0.0)
            if requested_at < overtaken:
                return False
            self.last_speak = max(self.last_speak, requested_at)
            self.generation += 1
            generation = self.generation
        threading.Thread(
            target=self._play, args=(generation, text, voice, speed), daemon=True
        ).start()
        return True

    def stop(self) -> None:
        with self.lock:
            self.last_stop = time.time()
            self.generation += 1

    def _live(self, generation: int) -> bool:
        return generation == self.generation

    def _synthesize(self, generation, text, voice, speed, out: queue.Queue) -> None:
        for chunk in chunks(text):
            if not self._live(generation):
                break
            try:
                with self.synth_lock:
                    audio = self.engine.create(chunk, voice, speed)
            except Exception as error:  # e.g. a chunk that yields no phonemes
                log(f"synthesis failed for {chunk[:60]!r}: {error}")
                continue
            out.put(np.clip(audio.astype(np.float32) * CFG["volume"], -1.0, 1.0))
        out.put(None)

    def _play(self, generation, text, voice, speed) -> None:
        # Synthesis starts right away, overlapping the previous utterance winding down
        audio_chunks: queue.Queue = queue.Queue()
        threading.Thread(
            target=self._synthesize,
            args=(generation, text, voice, speed, audio_chunks),
            daemon=True,
        ).start()
        with self.play_lock:
            if not self._live(generation):
                return
            try:
                with sd.OutputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                    device=output_device(),
                ) as stream:
                    self.speaking = True
                    while (audio := audio_chunks.get()) is not None:
                        for start in range(0, len(audio), BLOCK):
                            if not self._live(generation):
                                stream.abort()
                                return
                            stream.write(audio[start : start + BLOCK])
                        stream.write(GAP)
            except Exception as error:
                log(f"playback failed: {error!r}")
            finally:
                self.speaking = False


class State:
    speaker: Speaker | None = None
    ready = threading.Event()
    server: "Server | None" = None
    unknown_voices: set[str] = set()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/health":
            return self._reply(404, {"error": "not found"})
        engine = State.speaker.engine if State.speaker else None
        self._reply(
            200,
            {
                "ready": State.ready.is_set(),
                "speaking": bool(State.speaker and State.speaker.speaking),
                "engine": ENGINE,
                "voices": engine.voices if engine else [],
                "providers": engine.providers if engine else [],
                "cuda_device": CFG["cuda_device"] if ENGINE == "kokoro" else None,
                "pid": os.getpid(),
            },
        )

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return self._reply(400, {"error": "invalid JSON"})

        if self.path == "/stop":
            if State.speaker:
                State.speaker.stop()
            return self._reply(200, {"ok": True})
        if self.path == "/shutdown":
            self._reply(200, {"ok": True})
            threading.Thread(target=State.server.shutdown, daemon=True).start()
            return
        if self.path != "/speak":
            return self._reply(404, {"error": "not found"})

        if not State.ready.wait(120):
            return self._reply(503, {"error": "model is not loaded"})
        text = str(body.get("text") or "").strip()
        if not text:
            return self._reply(400, {"error": "empty text"})
        wanted = str(body.get("voice") or CFG["voice"])
        # A voice the engine lacks (say af_sky after switching to Paradee) falls
        # back to its default rather than going silent
        voice = State.speaker.engine.pick_voice(wanted)
        if voice != wanted and wanted not in State.unknown_voices:
            State.unknown_voices.add(wanted)
            log(f"{ENGINE} has no voice {wanted!r}, using {voice!r}")
        speed = min(max(float(body.get("speed") or CFG["speed"]), 0.5), 2.0)
        accepted = State.speaker.speak(
            text,
            voice,
            speed,
            requested_at=float(body.get("requested_at") or time.time()),
            narration=bool(body.get("narration")),
        )
        self._reply(202 if accepted else 200, {"ok": True, "skipped": not accepted})

    def _reply(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        pass


class Server(ThreadingHTTPServer):
    daemon_threads = True
    # On Windows SO_REUSEADDR lets a second daemon bind the same port; the bind
    # must fail instead so that only one instance ever runs.
    allow_reuse_address = False

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        # Skip HTTPServer.server_bind, whose getfqdn() lookup can stall for seconds
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


def load_in_background() -> None:
    started = time.monotonic()
    try:
        engine = load_model()
    except Exception as error:
        log(f"{ENGINE} model load failed: {error!r}")
        State.server.shutdown()
        return
    State.speaker = Speaker(engine)
    State.ready.set()
    log(f"{ENGINE} ready in {time.monotonic() - started:.1f}s, providers {engine.providers}")


def main() -> int:
    port = int(CFG["port"])
    try:
        State.server = Server(("127.0.0.1", port), Handler)
    except OSError as error:
        log(f"port {port} unavailable ({error}); another daemon is probably running")
        return 1
    device = f"CUDA device {CFG['cuda_device']}" if ENGINE == "kokoro" else "CPU"
    log(f"pid {os.getpid()} listening on 127.0.0.1:{port}, {ENGINE} on {device}")
    threading.Thread(target=load_in_background, daemon=True).start()
    try:
        State.server.serve_forever()
    except KeyboardInterrupt:
        pass
    log("shutting down")
    return 0 if State.ready.is_set() else 1


if __name__ == "__main__":
    sys.exit(main())
