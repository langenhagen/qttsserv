#!/usr/bin/env python3
"""Keep a Qwen3-TTS model loaded and stream speech to qtts_client.py.

Loads one model with mlx-audio on the Apple Silicon GPU, then serves requests
on a Unix domain socket, one at a time. Each request clones the voice of its
reference recording, optionally with the recording's transcript. See
qtts_protocol.py for the socket location and the message format.

Each request names its model. If that differs from the loaded one, the server
drops the loaded model from memory first and then loads the requested one, so
only one model is ever in memory.

The server splits the text into chunks (see chunking.py) and generates each
chunk in streaming mode: mlx-audio yields about every 2 s of audio while it is
still generating the rest of the chunk, and the server sends each piece on
right away.

A lock file ensures that at most one server per user runs. A second server
exits with code 0 right away, without loading a model.

Usage:

    qtts_server.py                  load the default model, then serve until stopped
    qtts_server.py --model ID       load the model ID (Hugging Face repo or local path)

Stop the server with `qtts_client.py --stop-server`, SIGTERM, or Ctrl-C.
"""

import argparse
import fcntl
import gc
import logging
import signal
import socketserver
from contextlib import suppress
from typing import IO, Any, cast

import chunking
from qtts_protocol import (
    DEFAULT_MODEL,
    LOCK_PATH,
    LOG_PATH,
    SOCKET_PATH,
    ensure_state_dir,
    read_frame,
    send_frame,
)

logger = logging.getLogger("qttsserv")


def _release_cached_memory() -> None:
    """Return MLX's buffer cache to the system.

    On Apple Silicon the GPU shares the system RAM, so an idle server should not
    hold on to buffers from the last long text.
    """
    import mlx.core as mx  # noqa: PLC0415 - loaded by main() already

    mx.clear_cache()
    logger.info("MLX memory in use: %.2f GB", mx.get_active_memory() / 1e9)


# Any: an mlx-audio model, whose class depends on the model ID.
def _load_model(model_id: str) -> Any:  # noqa: ANN401 - see above
    """Load a Qwen3-TTS model with mlx-audio and warm it up."""
    # Importing MLX and mlx-audio takes seconds.
    from mlx_audio.tts.utils import load  # noqa: PLC0415 - see comment above

    logger.info("loading model %s", model_id)
    model = load(model_id)
    # Compile the GPU kernels once, so that the first real request is as fast
    # as all later ones.
    for _ in model.generate(text="Hello.", split_pattern=""):
        pass
    _release_cached_memory()
    logger.info("loaded model %s", model_id)
    return model


class SpeakHandler(socketserver.StreamRequestHandler):
    """Handle one request: stream the speech of a text back, or stop."""

    def _speak(self, server: QttsServer, request: dict[str, Any]) -> None:
        """Generate the text chunk by chunk and send each audio piece once ready."""
        import mlx.core as mx  # noqa: PLC0415 - loaded with mlx-audio already
        import numpy as np  # noqa: PLC0415 - see above

        text = request["text"]
        logger.info(
            "speak: %d characters, model %s, language %s, voice %s, sampling %s, "
            "max chars %d, seed %d",
            len(text),
            request["model"],
            request["language"],
            request["ref_audio"],
            request["sampling"],
            request["max_chars"],
            request["seed"],
        )
        try:
            model = server.model_for(request["model"])
        # Catch any load error, e.g. a mistyped model ID, to report it to the client.
        except Exception as error:  # pylint: disable=broad-exception-caught
            logger.exception("loading model %s failed", request["model"])
            send_frame(self.request, {"type": "error", "message": str(error)})
            return
        if request["seed"] != 0:
            mx.random.seed(request["seed"])
        hello = {"type": "hello", "model": server.model_id}
        send_frame(self.request, hello | {"sample_rate": model.sample_rate})
        for index, chunk in enumerate(
            chunking.split_chunks(text, request["max_chars"]),
        ):
            send_frame(self.request, {"type": "chunk", "index": index, "text": chunk})
            # split_pattern="" because chunking.py split the text already.
            results = model.generate(
                text=chunk,
                lang_code=request["language"],
                ref_audio=request["ref_audio"],
                ref_text=request["ref_text"],
                split_pattern="",
                stream=True,
                **request["sampling"],
            )
            for result in results:
                audio = np.array(result.audio, dtype=np.float32).tobytes()
                send_frame(self.request, {"type": "audio"}, audio)
        send_frame(self.request, {"type": "done"})
        logger.info("speak: done")

    def handle(self) -> None:
        """Read the request frame and dispatch on its command."""
        server = cast("QttsServer", self.server)
        frame = read_frame(self.rfile)
        if frame is None:
            return
        request, _ = frame
        if request["command"] == "stop":
            logger.info("stop requested")
            server.stop_requested = True
            send_frame(self.request, {"type": "done"})
            return
        try:
            self._speak(server, request)
        except BrokenPipeError, ConnectionResetError:
            logger.info("client disconnected, generation aborted")
        finally:
            _release_cached_memory()


class QttsServer(socketserver.UnixStreamServer):
    """Unix socket server that holds the one loaded Qwen3-TTS model."""

    # Any: see _load_model.
    def __init__(self, model_id: str, model: Any) -> None:  # noqa: ANN401
        """Bind the socket and keep the model for the request handler."""
        super().__init__(str(SOCKET_PATH), SpeakHandler)
        self.model_id: str | None = model_id
        self.model = model
        self.stop_requested = False

    def model_for(self, model_id: str) -> Any:  # noqa: ANN401 - see __init__
        """Return the model model_id, replacing the loaded model if it differs."""
        if model_id != self.model_id:
            logger.info("dropping model %s", self.model_id)
            # Drop the only reference before loading, so that the two models
            # are never in memory at the same time. Reset model_id first, so
            # that a failed load is retried on the next request.
            self.model_id = None
            self.model = None
            gc.collect()
            _release_cached_memory()
            self.model = _load_model(model_id)
            self.model_id = model_id
        return self.model


def _acquire_single_instance_lock() -> IO[str] | None:
    """Return the locked lock file, or None if another server holds the lock.

    The caller must keep the returned file object open for the server's lifetime.
    """
    lock_file = LOCK_PATH.open("w")  # stays open to hold the lock
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        return None
    return lock_file


def _raise_keyboard_interrupt(_signum: int, _frame: object) -> None:
    """Turn SIGTERM into the same clean shutdown path as Ctrl-C."""
    raise KeyboardInterrupt


def main() -> None:
    """Load the model once, then serve requests until stopped."""
    parser = argparse.ArgumentParser(description="Qwen3-TTS server for qtts_client.py")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="model to load first, Hugging Face repo or local path "
        "(default: %(default)s)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ensure_state_dir()
    lock_file = _acquire_single_instance_lock()
    if lock_file is None:
        logger.info("another qttsserv server is already running; exiting")
        return

    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    logger.info("starting (log file for autostarted servers: %s)", LOG_PATH)
    # A socket file left over from a crashed server blocks bind(). Holding the
    # lock guarantees that no live server uses it.
    SOCKET_PATH.unlink(missing_ok=True)
    # Pass the model straight in: a local variable here would keep it in memory
    # after QttsServer.model_for() swapped it out.
    server = QttsServer(args.model, _load_model(args.model))
    logger.info("ready on %s", SOCKET_PATH)
    try:
        while not server.stop_requested:
            server.handle_request()
    except KeyboardInterrupt:
        logger.info("interrupted")
    finally:
        server.server_close()
        SOCKET_PATH.unlink(missing_ok=True)
        lock_file.close()
        logger.info("stopped")


if __name__ == "__main__":
    with suppress(KeyboardInterrupt):
        main()
