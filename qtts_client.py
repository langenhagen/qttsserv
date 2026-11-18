#!/usr/bin/env python3
"""Qwen3-TTS client that talks to a running qtts_server.py instead of loading
the model itself.

Reads text from a file, a pipe, or the clipboard, sends it to the server, and
plays the speech while the server is still generating it. Starts the server in
the background if it is not running; the server then keeps running after this
client exits. The server is never restarted automatically after a failure.

The voice is cloned from the reference recording --voice. The optional
--transcript holds the exact words spoken in that recording; with it, the model
copies rhythm and accent as well as the timbre.

A new qtts call cancels the running one: starting qtts while another one still
speaks stops the other one, and so does --stop-playback. Each client stays
connected to the server until its playback ends, so that the server can tell it
to stop.

Usage:

    qtts_client.py FILE --voice me.wav
        speak the text file in the voice of the recording me.wav
    qtts_client.py --clipboard --voice me.wav
        speak the text from the clipboard
    echo "hi" | qtts_client.py --voice me.wav
        speak text piped to stdin
    qtts_client.py FILE --voice me.wav --transcript me.txt
        clone the voice more closely with the words spoken in me.wav
    qtts_client.py FILE --voice me.wav --language german
        speak the text file as German instead of detecting the language
    qtts_client.py FILE --voice me.wav --temperature 0.7
        speak the text file steadier than the default 1.3; higher values sound livelier
    qtts_client.py FILE --voice me.wav --seed 42
        speak the text file the same way every time
    qtts_client.py FILE --voice me.wav --model-size 1.7B --precision bf16
        speak with the larger model at full precision instead of the default
        0.6B at 4 bits; the server swaps out its loaded model
    qtts_client.py FILE --voice me.wav --prebuffer 2.3
        start playing after 2.3 s of audio are ready instead of at once (the
        default); potentially avoids short pauses on slow machines
    qtts_client.py --stop-playback
        cancel the running qtts call, so it stops speaking; the server keeps running
    qtts_client.py --stop-server
        stop the running server, freeing its memory

    qtts_client.py --help
        show help
"""

import argparse
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qtts_protocol import (
    DEFAULT_MODEL_SIZE,
    DEFAULT_PRECISION,
    LANGUAGES,
    LOG_PATH,
    MODEL_SIZES,
    PRECISIONS,
    SOCKET_PATH,
    ensure_state_dir,
    model_id,
    read_frame,
    send_frame,
)

if TYPE_CHECKING:
    import io
    from collections.abc import Callable

# What the reader thread hands the main thread: a frame, the reader's exit on
# an error, or None once playback has ended.
type _Event = tuple[dict[str, Any], bytes] | SystemExit | None

SERVER_SCRIPT = Path(__file__).resolve().parent / "qtts_server.py"
# First start downloads the model if it is not cached yet.
STARTUP_TIMEOUT_S = 300
STARTUP_POLL_INTERVAL_S = 0.1
TRAILING_SILENCE_S = 0.3
FLOAT32_BYTES = 4
# Seconds of audio to collect before playback starts; see --prebuffer.
DEFAULT_PREBUFFER_S = 0.0


def _clipboard_read_command() -> list[str]:
    """Return the OS-appropriate command that prints the clipboard to stdout."""
    if sys.platform == "darwin":
        return ["pbpaste"]
    if shutil.which("wl-paste"):
        return ["wl-paste", "--no-newline"]
    if shutil.which("xclip"):
        return ["xclip", "-selection", "clipboard", "-o"]
    if shutil.which("xsel"):
        return ["xsel", "--clipboard", "--output"]
    msg = "No clipboard utility found (need pbpaste, wl-paste, xclip, or xsel)"
    raise RuntimeError(msg)


def read_clipboard() -> str:
    """Return the text currently stored in the clipboard."""
    return subprocess.check_output(_clipboard_read_command(), text=True)


def _connect() -> socket.socket:
    """Connect to the server socket; raise OSError if no server listens."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(str(SOCKET_PATH))
    except OSError:
        sock.close()
        raise
    return sock


def _start_server(model: str) -> subprocess.Popen[bytes]:
    """Start the server with model detached from this terminal, logging to LOG_PATH."""
    ensure_state_dir()
    print(f"qtts: starting server, log: {LOG_PATH}", file=sys.stderr)
    with LOG_PATH.open("ab") as log_file:
        return subprocess.Popen(
            [sys.executable, str(SERVER_SCRIPT), "--model", model],
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,  # Ctrl-C in this terminal must not reach the server
        )


def connect_or_start_server(model: str) -> socket.socket:
    """Connect to the running server, starting it with model first if necessary."""
    with suppress(FileNotFoundError, ConnectionRefusedError):
        return _connect()

    server_process = _start_server(model)
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while True:
        with suppress(FileNotFoundError, ConnectionRefusedError):
            return _connect()
        # Exit code 0 means another server won the start race; keep waiting for it.
        returncode = server_process.poll()
        if returncode not in {None, 0}:
            sys.exit(f"qtts: server exited with code {returncode}, see {LOG_PATH}")
        if time.monotonic() > deadline:
            sys.exit(f"qtts: server not ready in {STARTUP_TIMEOUT_S} s, see {LOG_PATH}")
        time.sleep(STARTUP_POLL_INTERVAL_S)


def _send_command(command: str) -> bool:
    """Send command to the running server and wait for its reply.

    Return False, without starting a server, if none is running.
    """
    try:
        sock = _connect()
    except FileNotFoundError, ConnectionRefusedError:
        print("qtts: server is not running", file=sys.stderr)
        return False
    with sock, sock.makefile("rb") as rfile:
        send_frame(sock, {"command": command})
        read_frame(rfile)
    return True


def stop_server() -> None:
    """Ask the running server to exit; do nothing if none is running."""
    if _send_command("stop"):
        print("qtts: server stopped", file=sys.stderr)


def stop_playback() -> None:
    """Cancel the running qtts call, if any; the server keeps running."""
    if _send_command("stop_playback"):
        print("qtts: playback stopped", file=sys.stderr)


def _next_frame(rfile: io.BufferedIOBase) -> tuple[dict[str, Any], bytes]:
    """Return the next frame from the server; exit on an error or a hang-up."""
    frame = read_frame(rfile)
    if frame is None:
        sys.exit(f"qtts: server hung up mid-request, see {LOG_PATH}")
    if frame[0]["type"] == "error":
        sys.exit(f"qtts: server error: {frame[0]['message']}")
    return frame


def _read_frames(rfile: io.BufferedReader, events: queue.Queue[_Event]) -> None:
    """Move the server's frames into the queue until "stopped" or the end.

    After "done" the connection stays open while the audio still plays, so that
    the server can still send "stopped"; the connection ending after "done" is
    the normal end, not an error.
    """
    done = False
    try:
        while True:
            if done and not rfile.peek(1):
                return
            frame = _next_frame(rfile)
            events.put(frame)
            if frame[0]["type"] == "done":
                done = True
            elif frame[0]["type"] == "stopped":
                return
    except SystemExit as exit_request:  # _next_frame exits on an error or a hang-up
        events.put(exit_request)


class _Player:
    """Play mono float32 PCM pieces as they arrive, from a PortAudio callback.

    The callback runs on PortAudio's own thread, so the main thread stays free
    to take in frames and can stop playback at any moment with abort().
    Playback starts once prebuffer_s seconds of audio are buffered, or at
    finish(). While the server falls behind, the callback plays silence.
    """

    def __init__(
        self,
        sample_rate: int,
        prebuffer_s: float,
        on_finished: Callable[[], None],
    ) -> None:
        """Open the output stream; on_finished runs once playback has ended."""
        import sounddevice as sd  # noqa: PLC0415 - lazy: loads PortAudio

        self._callback_stop = sd.CallbackStop
        self._sample_rate = sample_rate
        self._prebuffer_bytes = int(sample_rate * prebuffer_s) * FLOAT32_BYTES
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._finishing = False
        self._started = False
        self._stream = sd.RawOutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
            finished_callback=on_finished,
        )

    # Any: sounddevice's cffi buffer and structs.
    def _callback(
        self,
        outdata: Any,  # noqa: ANN401
        _frames: int,
        _time: Any,  # noqa: ANN401
        _status: Any,  # noqa: ANN401
    ) -> None:
        """Fill outdata from the buffer, with silence for what is missing."""
        with self._lock:
            size = min(len(outdata), len(self._buffer))
            outdata[:size] = self._buffer[:size]
            del self._buffer[:size]
            drained = self._finishing and not self._buffer
        outdata[size:] = bytes(len(outdata) - size)
        if drained:
            raise self._callback_stop

    def _start_once(self) -> None:
        """Start the stream unless it has started already."""
        if not self._started:
            self._started = True
            self._stream.start()

    def add(self, audio: bytes) -> None:
        """Buffer an audio piece; start playing once enough is buffered."""
        with self._lock:
            self._buffer += audio
            ready = len(self._buffer) >= self._prebuffer_bytes
        if ready:
            self._start_once()

    def finish(self) -> None:
        """Play what is buffered, then stop; on_finished runs after that."""
        with self._lock:
            # Without trailing silence, the last word ends abruptly.
            silence_frames = int(self._sample_rate * TRAILING_SILENCE_S)
            self._buffer += bytes(silence_frames * FLOAT32_BYTES)
            self._finishing = True
        self._start_once()

    def abort(self) -> None:
        """Stop playing at once, dropping what is still buffered."""
        self._stream.abort()

    def close(self) -> None:
        """Close the output stream, aborting it if it still plays."""
        self._stream.close()


def _play(events: queue.Queue[_Event], player: _Player) -> None:
    """Hand the frames to the player until playback ends or the server stops it."""
    while True:
        event = events.get()
        if event is None:
            return
        if isinstance(event, SystemExit):
            raise event
        header, audio = event
        if header["type"] == "chunk":
            print(header["index"])
            print(header["text"])
        elif header["type"] == "audio":
            player.add(audio)
        elif header["type"] == "done":
            player.finish()
        elif header["type"] == "stopped":
            player.abort()
            print("qtts: canceled by another qtts call", file=sys.stderr)
            return


def speak(request: dict[str, Any], prebuffer_s: float) -> None:
    """Send the speak request to the server and play the audio pieces as they arrive.

    A reader thread takes the frames off the socket into a queue, and the main
    thread hands them to a _Player, so the server never waits for playback. The
    first prebuffer_s seconds of audio are held back before playback starts, to
    bridge the pauses in which the server generates slower than the speakers
    play. The connection stays open until playback ends, so that the server can
    still send "stopped" when a new request cancels this one.
    """
    with (
        connect_or_start_server(request["model"]) as sock,
        sock.makefile("rb") as rfile,
    ):
        send_frame(sock, request)
        hello = _next_frame(rfile)[0]
        # A new request can cancel this one while it still waits for the model.
        if hello["type"] == "stopped":
            print("qtts: canceled by another qtts call", file=sys.stderr)
            return
        print(f"model: {hello['model']}")
        events: queue.Queue[_Event] = queue.Queue()
        reader = threading.Thread(
            target=_read_frames,
            args=(rfile, events),
            daemon=True,
        )
        reader.start()
        player = _Player(hello["sample_rate"], prebuffer_s, lambda: events.put(None))
        try:
            _play(events, player)
        finally:
            player.close()
            # Hanging up ends the reader thread's read and tells the server that
            # this request is through; the server may have hung up already.
            with suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
            reader.join()


def main() -> None:
    """Send the text from a file, clipboard, or stdin pipe to the server and play it."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("file", nargs="?", help="text file to speak")
    parser.add_argument(
        "--clipboard",
        "-c",
        action="store_true",
        help="speak text from the clipboard",
    )
    parser.add_argument(
        "--voice",
        "-v",
        type=Path,
        metavar="AUDIO_FILE",
        help="reference recording of the voice to clone; required for speaking",
    )
    parser.add_argument(
        "--transcript",
        "-t",
        type=Path,
        metavar="TEXT_FILE",
        help="exact words spoken in the --voice recording; optional",
    )
    parser.add_argument(
        "--language",
        "-l",
        default="auto",
        choices=LANGUAGES,
        help="language of the text; auto detects it (default: %(default)s)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.3,
        help="randomness of the delivery: lower is steadier, higher livelier "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="candidate sounds per step (default: %(default)s)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="share of likely sounds to pick from (default: %(default)s)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.5,
        help="discourages repeated sounds; mlx-audio raises it to at least 1.5 "
        "with --transcript (default: %(default)s)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=300,
        help="max tokens per chunk; 12.5 tokens are 1 s of audio "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=100,
        help="max characters per chunk (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for reproducible speech; 0 for random (default: %(default)s)",
    )
    parser.add_argument(
        "--model-size",
        choices=MODEL_SIZES,
        default=DEFAULT_MODEL_SIZE,
        help="model size; 0.6B is smaller and faster, 1.7B clones a voice more "
        "faithfully (default: %(default)s)",
    )
    parser.add_argument(
        "--precision",
        choices=PRECISIONS,
        default=DEFAULT_PRECISION,
        help="bits of the model weights, fewer is smaller and faster; bf16 is "
        "full precision (default: %(default)s)",
    )
    parser.add_argument(
        "--prebuffer",
        type=float,
        default=DEFAULT_PREBUFFER_S,
        metavar="SECONDS",
        help="seconds of audio to collect before playback starts; more hides "
        "the pauses of a server slower than real time, 0 starts at once "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--stop-playback",
        action="store_true",
        help="cancel the running qtts call and exit; the server keeps running",
    )
    parser.add_argument(
        "--stop-server",
        action="store_true",
        help="stop the running qtts server and exit",
    )
    args = parser.parse_args()

    if args.stop_server:
        stop_server()
        return
    if args.stop_playback:
        stop_playback()
        return

    if args.voice is None:
        parser.error("--voice is required")
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    elif args.clipboard:
        text = read_clipboard()
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        parser.print_usage()
        sys.exit(2)
    ref_text = args.transcript.read_text().strip() if args.transcript else None

    speak(
        {
            "command": "speak",
            "text": text,
            "model": model_id(args.model_size, args.precision),
            "language": args.language,
            # Absolute, because the server runs in another working directory.
            "ref_audio": str(args.voice.expanduser().resolve(strict=True)),
            "ref_text": ref_text,
            "sampling": {
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty,
                "max_tokens": args.max_tokens,
            },
            "max_chars": args.max_chars,
            "seed": args.seed,
        },
        args.prebuffer,
    )


if __name__ == "__main__":
    with suppress(KeyboardInterrupt):
        main()
