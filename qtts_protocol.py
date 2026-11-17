"""Shared file locations and message framing for the Qwen3-TTS server and client.

The server listens on a Unix domain socket. Every message in both directions
is one frame:

    4 bytes     big-endian length of the JSON header
    N bytes     JSON header (UTF-8)
    M bytes     raw payload; M is the header's "audio_bytes" value (default 0)

Requests (client -> server), one frame per connection:

    {"command": "speak", "text": "...", "model": "mlx-community/...",
     "language": "auto", "ref_audio": "/abs/path.wav", "ref_text": "..." or null,
     "sampling": {"temperature": 1.3, "top_k": 50, "top_p": 1.0,
                  "repetition_penalty": 1.5, "max_tokens": 300},
     "max_chars": 100, "seed": 0}
    {"command": "stop"}

Responses (server -> client) to "speak":

    {"type": "hello", "model": "mlx-community/...", "sample_rate": 24000}
    {"type": "chunk", "index": 0, "text": "..."}     a text chunk starts
    {"type": "audio", "audio_bytes": M} followed by M bytes of mono float32 PCM
    ... more audio frames, then more chunks ...
    {"type": "done"}

or {"type": "error", "message": "..."} instead, e.g. if the model fails to load.

Responses to "stop": {"type": "done"}, then the server exits.
"""

import json
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import io
    import socket

# The 4-bit quantized Base model: the smallest and fastest; clones a voice
# from a reference recording.
DEFAULT_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit"

# Languages of Qwen3-TTS; "auto" detects the language from the text.
LANGUAGES = [
    "auto",
    "english",
    "german",
    "chinese",
    "french",
    "italian",
    "japanese",
    "korean",
    "portuguese",
    "russian",
    "spanish",
]

STATE_DIR = Path.home() / ".cache" / "qttsserv"
SOCKET_PATH = STATE_DIR / "qttsserv.sock"
LOCK_PATH = STATE_DIR / "qttsserv.lock"
LOG_PATH = STATE_DIR / "qttsserv.log"

HEADER_LENGTH = struct.Struct(">I")


def ensure_state_dir() -> None:
    """Create the state directory, readable by the current user only."""
    STATE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)


def send_frame(sock: socket.socket, header: dict[str, Any], audio: bytes = b"") -> None:
    """Send one frame with the given JSON header and optional audio payload."""
    if audio:
        header = header | {"audio_bytes": len(audio)}
    header_bytes = json.dumps(header).encode()
    sock.sendall(HEADER_LENGTH.pack(len(header_bytes)) + header_bytes + audio)


def _exactly(data: bytes, expected_length: int) -> bytes:
    """Return data, or raise if the connection ended in the middle of a frame."""
    if len(data) != expected_length:
        msg = f"connection closed mid-frame ({len(data)} of {expected_length} bytes)"
        raise ConnectionError(msg)
    return data


def read_frame(rfile: io.BufferedIOBase) -> tuple[dict[str, Any], bytes] | None:
    """Read one frame; return (header, audio), or None if the peer closed the
    connection cleanly before the frame started.
    """
    length_bytes = rfile.read(HEADER_LENGTH.size)
    if not length_bytes:
        return None
    (header_length,) = HEADER_LENGTH.unpack(_exactly(length_bytes, HEADER_LENGTH.size))
    header = json.loads(_exactly(rfile.read(header_length), header_length))
    audio_length = header.get("audio_bytes", 0)
    audio = _exactly(rfile.read(audio_length), audio_length) if audio_length else b""
    return header, audio
