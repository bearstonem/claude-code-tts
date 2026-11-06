"""Kokoro TTS daemon for Claude Code, pinned to one CUDA GPU.

Loads the model once, keeps it resident on the GPU and plays speech on the
output device. tts_hook.py talks to it over localhost:

  POST /speak     {"text": ..., "voice": ..., "speed": ...}  replaces whatever is playing
  POST /stop      silences playback
  POST /shutdown  exits the daemon (the next hook starts a fresh one)
  GET  /health    readiness, execution providers, pinned device
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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tts_common import load_config

CFG = load_config()

# Pin before onnxruntime touches CUDA. PCI bus order makes indices match nvidia-smi
# (CUDA's default is fastest-first, which would put the 5090 at 0); UUIDs need no order.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(CFG["cuda_device"])

import numpy as np  # noqa: E402
import onnxruntime as rt  # noqa: E402
import sounddevice as sd  # noqa: E402
from kokoro_onnx import Kokoro  # noqa: E402
from kokoro_onnx.config import SAMPLE_RATE  # noqa: E402

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


def load_model() -> Kokoro:
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
    # The first run allocates GPU memory and picks kernels; do it before real speech
    warmup_voice = CFG["voice"] if CFG["voice"] in kokoro.voices else "af_sky"
    kokoro.create("Warm up.", voice=warmup_voice)
    return kokoro


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

    def __init__(self, kokoro: Kokoro):
        self.kokoro = kokoro
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
        lang = LANGS.get(voice[0], "en-us")
        for chunk in chunks(text):
            if not self._live(generation):
                break
            try:
                with self.synth_lock:
                    audio, _ = self.kokoro.create(
                        chunk, voice=voice, speed=speed, lang=lang
                    )
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
    providers: list[str] = []
    ready = threading.Event()
    server: "Server | None" = None


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/health":
            return self._reply(404, {"error": "not found"})
        self._reply(
            200,
            {
                "ready": State.ready.is_set(),
                "speaking": bool(State.speaker and State.speaker.speaking),
                "providers": State.providers,
                "cuda_device": CFG["cuda_device"],
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
        voice = str(body.get("voice") or CFG["voice"])
        if not text:
            return self._reply(400, {"error": "empty text"})
        if voice not in State.speaker.kokoro.voices:
            return self._reply(400, {"error": f"unknown voice {voice!r}"})
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
    try:
        kokoro = load_model()
    except Exception as error:
        log(f"model load failed: {error!r}")
        State.server.shutdown()
        return
    State.speaker = Speaker(kokoro)
    State.providers = kokoro.sess.get_providers()
    State.ready.set()
    log(f"ready, providers {State.providers}")


def main() -> int:
    port = int(CFG["port"])
    try:
        State.server = Server(("127.0.0.1", port), Handler)
    except OSError as error:
        log(f"port {port} unavailable ({error}); another daemon is probably running")
        return 1
    log(f"pid {os.getpid()} listening on 127.0.0.1:{port}, CUDA device {CFG['cuda_device']}")
    threading.Thread(target=load_in_background, daemon=True).start()
    try:
        State.server.serve_forever()
    except KeyboardInterrupt:
        pass
    log("shutting down")
    return 0 if State.ready.is_set() else 1


if __name__ == "__main__":
    sys.exit(main())
