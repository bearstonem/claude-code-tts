"""Text-to-speech daemon for Claude Code, speaking with Paradee-8M.

Loads the model once, keeps it resident and plays speech on the output device.
It runs on the CPU; with "device": "cuda" (install.py --gpu) on an NVIDIA GPU.
tts_hook.py talks to it over localhost:

  POST /speak     {"text": ..., "voice": ..., "speed": ..., "session": ...}
                  replaces whatever is playing
  POST /stop      silences playback; with {"session": ...} only that session's
  POST /shutdown  exits the daemon (the next hook starts a fresh one)
  GET  /health    readiness, voices, execution providers, device
"""

import faulthandler
import json
import os
import queue
import re
import socket
import socketserver
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tts_common import HERE, VOICES, download_model, load_config

# stderr is daemon.log: a crash in native code (PortAudio, onnxruntime) leaves a
# trace there, and so does a start-up that hangs (cancelled once the model is ready)
faulthandler.enable()
STARTUP_WATCHDOG = 90
faulthandler.dump_traceback_later(STARTUP_WATCHDOG)

CFG = load_config()
CUDA = CFG["device"] == "cuda"

if CUDA:
    # Pin before onnxruntime touches CUDA. PCI bus order makes indices match
    # nvidia-smi (CUDA's default is fastest first); UUIDs need no order.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if CFG["cuda_device"] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(CFG["cuda_device"])
# Windows without Developer Mode cannot symlink; the Hub cache copies instead, which is fine
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np  # noqa: E402
from paradee import SAMPLE_RATE, Paradee  # noqa: E402

sd = None  # sounddevice, imported by audio()

BLOCK = SAMPLE_RATE // 10  # write audio in 100 ms blocks so /stop takes effect quickly
GAP = np.zeros(SAMPLE_RATE // 4, dtype=np.float32)  # pause after each synthesized chunk
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
DUCK_SCRIPT = HERE / "scripts" / "audio-duck.sh"
DUCK = sys.platform == "darwin" and bool(CFG["duck_music"]) and DUCK_SCRIPT.exists()


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", file=sys.stderr, flush=True)


class Model:
    """Paradee, loaded once; create() turns text into float32 audio at SAMPLE_RATE."""

    def __init__(self):
        if CFG["model"] and CFG["model_config"]:
            model_path, config_path = CFG["model"], CFG["model_config"]
        else:
            model_path, config_path = download_model(gpu=CUDA)
        self.paradee = Paradee(
            model_path=model_path, config_path=config_path, threads=int(CFG["threads"])
        )
        if CUDA:
            self._use_cuda(model_path)
        self.voices: list[str] = list(self.paradee.voices)
        self.providers: list[str] = self.paradee.session.get_providers()

    def _use_cuda(self, model_path: str) -> None:
        """Swap Paradee's CPU-only session for a CUDA one, built the same way."""
        import onnxruntime as rt

        # Paradee adds the per-token durations as a graph output when it loads;
        # its _run() reads them, so the CUDA session needs the same graph
        from paradee.tts import DURATIONS, _with_output

        if hasattr(rt, "preload_dlls"):
            rt.preload_dlls()  # CUDA and cuDNN libraries from the nvidia-* wheels
        rt.set_default_logger_severity(3)
        options = rt.SessionOptions()
        options.log_severity_level = 3
        with open(model_path, "rb") as fp:
            graph = _with_output(fp.read(), DURATIONS)
        self.paradee.session = rt.InferenceSession(
            graph,
            options,
            providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
        )
        if self.paradee.session.get_providers()[0] != "CUDAExecutionProvider":
            log(f"WARNING: CUDA unavailable, running on {self.paradee.session.get_providers()}")

    def create(self, text: str, voice: str, speed: float) -> np.ndarray:
        # Paradee picks American or British phonemes from the voice name itself
        return self.paradee(text, speed=speed, voice=voice)

    def pick_voice(self, voice: str) -> str:
        return voice if voice in self.voices else VOICES[0]


def load_model() -> Model:
    model = Model()
    # The first run loads the phonemizer for the voice's accent and, on a GPU,
    # allocates memory and picks kernels; do it before real speech
    model.create("Warm up.", model.pick_voice(CFG["voice"]), 1.0)
    return model


def duck(action: str) -> None:
    """macOS: lower Apple Music while speaking ("duck"), then put it back ("restore")."""
    if DUCK:
        try:
            subprocess.run(["bash", str(DUCK_SCRIPT), action], timeout=5, check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            log(f"audio-duck {action} failed: {error!r}")


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


def audio():
    """sounddevice with PortAudio freshly initialized, so a new default output
    (headset, VR) is picked up.

    It is imported here rather than at start-up because PortAudio's start-up fails
    while no sound server runs (PulseAudio on a headless Linux box, or a desktop
    session not up yet); the daemon then still loads, and speaks once sound works.
    A failed import is not cached, so the next utterance simply tries again.
    """
    global sd
    if sd is None:
        import sounddevice

        sd = sounddevice
        return sd
    try:
        sd._terminate()
    except sd.PortAudioError:  # the last initialization failed; nothing to end
        pass
    sd._initialize()
    return sd


def output_device() -> int | None:
    sd = audio()
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

    def __init__(self, model: Model):
        self.model = model
        self.generation = 0
        self.speaking = False
        self.last_speak = 0.0
        self.last_stop = 0.0  # a stop for everyone (`claude-tts off`)
        self.session_stops: dict[str, float] = {}  # a stop from one session's hook
        self.session = ""  # the session whose speech is playing, or played last
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
        session: str = "",
    ) -> bool:
        """Start speaking unless the request was overtaken while its hook ran.

        Hooks run in the background and can arrive out of order. A reply is
        dropped only by a stop sent after it, by its own session or for all (the
        user moved on); narration of a tool call is also dropped by any newer
        speech, so it never talks over a reply.
        """
        with self.lock:
            overtaken = max(
                self.last_stop,
                self.session_stops.get(session, 0.0),
                self.last_speak if narration else 0.0,
            )
            if requested_at < overtaken:
                return False
            self.last_speak = max(self.last_speak, requested_at)
            self.session = session
            self.generation += 1
            generation = self.generation
        threading.Thread(
            target=self._play, args=(generation, text, voice, speed), daemon=True
        ).start()
        return True

    def stop(self, session: str | None = None) -> None:
        """Silence everything, or with a session only that session's speech."""
        with self.lock:
            if session is None:
                self.last_stop = time.time()
                self.generation += 1
                return
            self.session_stops[session] = time.time()
            if session == self.session:
                self.generation += 1

    def _live(self, generation: int) -> bool:
        return generation == self.generation

    def _synthesize(self, generation, text, voice, speed, out: queue.Queue) -> None:
        for chunk in chunks(text):
            if not self._live(generation):
                break
            try:
                with self.synth_lock:
                    audio = self.model.create(chunk, voice, speed)
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
                device = output_device()  # also (re)initializes PortAudio
                with sd.OutputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                    device=device,
                ) as stream:
                    self.speaking = True
                    duck("duck")
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
                if self.speaking:
                    duck("restore")
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
        model = State.speaker.model if State.speaker else None
        self._reply(
            200,
            {
                "ready": State.ready.is_set(),
                "speaking": bool(State.speaker and State.speaker.speaking),
                "device": "cuda" if CUDA else "cpu",
                "cuda_device": CFG["cuda_device"] if CUDA else None,
                "voices": model.voices if model else [],
                "providers": model.providers if model else [],
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
                State.speaker.stop(str(body["session"]) if body.get("session") else None)
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
        # An unknown voice (a typo in config.json) falls back rather than going silent
        voice = State.speaker.model.pick_voice(wanted)
        if voice != wanted and wanted not in State.unknown_voices:
            State.unknown_voices.add(wanted)
            log(f"no voice {wanted!r}, using {voice!r}")
        speed = min(max(float(body.get("speed") or CFG["speed"]), 0.5), 2.0)
        accepted = State.speaker.speak(
            text,
            voice,
            speed,
            requested_at=float(body.get("requested_at") or time.time()),
            narration=bool(body.get("narration")),
            session=str(body.get("session") or ""),
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
    # Only one daemon may ever run, so a second bind of the port must fail. On
    # Windows SO_REUSEADDR would let it succeed (SO_EXCLUSIVEADDRUSE is used
    # instead); on Linux and macOS it only allows rebinding over connections in
    # TIME_WAIT, which a restart needs.
    allow_reuse_address = os.name != "nt"

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        # Skip HTTPServer.server_bind, whose getfqdn() lookup can stall for seconds
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


def load_in_background() -> None:
    started = time.monotonic()
    try:
        model = load_model()
    except Exception as error:
        log(f"model load failed: {error!r}")
        State.server.shutdown()
        return
    State.speaker = Speaker(model)
    State.ready.set()
    faulthandler.cancel_dump_traceback_later()
    try:
        audio()
    except Exception as error:  # speech is synthesized anyway; playback retries each time
        log(f"no audio yet: {error!r}")
    log(f"ready in {time.monotonic() - started:.1f}s, providers {model.providers}")


def main() -> int:
    port = int(CFG["port"])
    try:
        State.server = Server(("127.0.0.1", port), Handler)
    except OSError as error:
        log(f"port {port} unavailable ({error}); another daemon is probably running")
        return 1
    device = f"CUDA device {CFG['cuda_device'] or 0}" if CUDA else "the CPU"
    log(f"pid {os.getpid()} listening on 127.0.0.1:{port}, Paradee on {device}")
    threading.Thread(target=load_in_background, daemon=True).start()
    try:
        State.server.serve_forever()
    except KeyboardInterrupt:
        pass
    log("shutting down")
    return 0 if State.ready.is_set() else 1


if __name__ == "__main__":
    sys.exit(main())
