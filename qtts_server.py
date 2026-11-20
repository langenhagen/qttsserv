#!/usr/bin/env python3
"""Keep a Qwen3-TTS model loaded and stream speech to qtts_client.py.

Loads one model with mlx-audio on the Apple Silicon GPU, then serves requests
on a Unix domain socket with asyncio, so any number of clients can be connected
at once. A request with a reference recording clones that voice, optionally
using the recording's transcript; without one, the Base model speaks in a
random voice. See qtts_protocol.py for the socket location and the message
format.

A new "speak" request cancels the running one. When it comes in, the server
sends {"type": "stopped"} to the running "speak" client, whether that one is
still generating or already only playing, and that client stops its playback
at once. A canceled request that is still generating stops at its next audio
piece, so that the new one gets the model next. Clients keep their connection
open until their own playback ends, so that a new request can still reach them.
A "stop_playback" request cancels the running one the same way, without
starting a new one.

All MLX work, i.e. loading, generating, and freeing memory, runs on one worker
thread. So the GPU work never blocks the event loop, and only one request
generates at a time.

Each request names its model. If that differs from the loaded one, the server
drops the loaded model from memory first and then loads the requested one, so
only one model is ever in memory.

The client may send a request's text in pieces. The server splits the text it
has into chunks (see chunking.py) and generates each chunk as soon as its
boundaries are final, in streaming mode: mlx-audio yields about every 2 s of
audio while it is still generating the rest of the chunk, and the server sends
each piece on right away.

A lock file ensures that at most one server per user runs. A second server
exits with code 0 right away, without loading a model.

Usage:

    qtts_server.py                  load the default model, then serve until stopped
    qtts_server.py --model ID       load the model ID (Hugging Face repo or local path)

Stop the server with `qtts_client.py --stop-server`, SIGTERM, or Ctrl-C.
"""

import argparse
import asyncio
import fcntl
import gc
import json
import logging
import signal
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from typing import IO, TYPE_CHECKING, Any

import chunking
from qtts_protocol import (
    DEFAULT_MODEL,
    HEADER_LENGTH,
    LOCK_PATH,
    LOG_PATH,
    SOCKET_PATH,
    encode_frame,
    ensure_state_dir,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

logger = logging.getLogger("qttsserv")


def _release_cached_memory() -> None:
    """Return MLX's buffer cache to the system.

    On Apple Silicon the GPU shares the system RAM, so an idle server should not
    hold on to buffers from the last long text.
    """
    import mlx.core as mx  # noqa: PLC0415 - loaded by _load_model() already

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


def _seed(seed: int) -> None:
    """Make the next generations reproducible."""
    import mlx.core as mx  # noqa: PLC0415 - loaded by _load_model() already

    mx.random.seed(seed)


# Any: see _load_model.
def _start_chunk(
    model: Any,  # noqa: ANN401
    chunk: str,
    request: dict[str, Any],
) -> Iterator[Any]:
    """Start generating one chunk in streaming mode; return its result iterator."""
    # split_pattern="" because chunking.py split the text already.
    return model.generate(
        text=chunk,
        lang_code=request["language"],
        ref_audio=request["ref_audio"],
        ref_text=request["ref_text"],
        split_pattern="",
        stream=True,
        **request["sampling"],
    )


def _next_audio(results: Iterator[Any]) -> bytes | None:
    """Generate the next audio piece as mono float32 PCM; None after the last one."""
    import numpy as np  # noqa: PLC0415 - loaded with mlx-audio already

    result = next(results, None)
    if result is None:
        return None
    return np.array(result.audio, dtype=np.float32).tobytes()


async def _read_request(reader: asyncio.StreamReader) -> dict[str, Any] | None:
    """Read the client's one request frame; None if it hung up before sending it."""
    try:
        length_bytes = await reader.readexactly(HEADER_LENGTH.size)
    except asyncio.IncompleteReadError as error:
        if error.partial:
            raise
        return None
    (header_length,) = HEADER_LENGTH.unpack(length_bytes)
    return json.loads(await reader.readexactly(header_length))


@dataclass(eq=False)
class _Session:
    """One connected "speak" request; canceled once a new one comes in."""

    writer: asyncio.StreamWriter
    canceled: bool = False
    text_queue: asyncio.Queue[str | None] | None = None


class QttsServer:
    """Holds the one loaded Qwen3-TTS model and the connected "speak" requests."""

    # Any: see _load_model.
    def __init__(
        self,
        model_id: str,
        model: Any,  # noqa: ANN401
        executor: ThreadPoolExecutor,
        stop_requested: asyncio.Event,
    ) -> None:
        """Keep the model, the MLX worker thread, and the stop signal."""
        self.model_id: str | None = model_id
        self.model = model
        self.executor = executor
        self.stop_requested = stop_requested
        # One model on one GPU: one request generates at a time.
        self.generation_lock = asyncio.Lock()
        self.sessions: set[_Session] = set()

    # Any: whatever function returns.
    async def run_mlx(
        self,
        function: Callable[..., Any],
        *args: Any,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        """Run function(*args) on the MLX worker thread and return its result."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, function, *args)

    def model_for(self, model_id: str) -> Any:  # noqa: ANN401 - see __init__
        """Return the model model_id, replacing the loaded model if it differs.

        Runs on the MLX worker thread.
        """
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

    async def handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Read the client's request and dispatch on its command."""
        try:
            request = await _read_request(reader)
            if request is None:
                return
            if request["command"] == "stop":
                logger.info("stop requested")
                writer.write(encode_frame({"type": "done"}))
                await writer.drain()
                self.stop_requested.set()
                return
            if request["command"] == "stop_playback":
                logger.info("stop playback requested")
                self._cancel_all()
                writer.write(encode_frame({"type": "done"}))
                await writer.drain()
                return
            await self._serve_speak(request, reader, writer)
        except ConnectionError:
            logger.info("client disconnected, generation aborted")
        finally:
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()

    def _cancel_all(self) -> None:
        """Cancel the running requests and tell their clients to stop playing.

        Each request cancels its predecessor, so at most one is running; the
        loop still covers one that is canceled but not yet disconnected.
        """
        for session in self.sessions:
            if session.canceled:
                continue
            session.canceled = True
            # Unblock a generator still waiting for the next piece of text.
            if session.text_queue is not None:
                session.text_queue.put_nowait(None)
            if not session.writer.is_closing():
                logger.info("canceling the running request")
                session.writer.write(encode_frame({"type": "stopped"}))

    async def _collect_text(
        self,
        reader: asyncio.StreamReader,
        text_queue: asyncio.Queue[str | None],
    ) -> None:
        """Queue the request's text frames, then wait for the client to hang up.

        A None sentinel ends the text. After that the client keeps the
        connection open while its audio plays, so a new request can still
        cancel this one; further frames from this client are ignored.
        """
        text_ended = False
        try:
            while True:
                frame = await _read_request(reader)
                if frame is None:
                    return
                if text_ended:
                    continue
                if frame.get("command") == "text":
                    await text_queue.put(frame["text"])
                elif frame.get("command") == "finish":
                    text_ended = True
                    await text_queue.put(None)
        except asyncio.IncompleteReadError:
            pass
        finally:
            if not text_ended:
                await text_queue.put(None)

    async def _serve_speak(
        self,
        request: dict[str, Any],
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Cancel the running request, generate this one, and wait for its playback."""
        session = _Session(writer)
        self._cancel_all()
        self.sessions.add(session)
        text_queue: asyncio.Queue[str | None] = asyncio.Queue()
        session.text_queue = text_queue
        collector = asyncio.create_task(self._collect_text(reader, text_queue))
        try:
            async with self.generation_lock:
                # Another request may have come in while this one waited.
                if session.canceled:
                    return
                await self._generate(session, request, text_queue)
            # The client plays on after "done" and hangs up when it is through;
            # until then, a new request can still cancel it.
            await collector
        finally:
            collector.cancel()
            with suppress(asyncio.CancelledError):
                await collector
            self.sessions.discard(session)

    async def _send(
        self,
        session: _Session,
        header: dict[str, Any],
        audio: bytes = b"",
    ) -> None:
        """Send one frame to the session's client; raise if it hung up."""
        session.writer.write(encode_frame(header, audio))
        await session.writer.drain()

    async def _emit_chunk(
        self,
        session: _Session,
        model: Any,  # noqa: ANN401 - see _load_model
        request: dict[str, Any],
        index: int,
        chunk: str,
    ) -> bool:
        """Stream one chunk's audio; return False if the session was canceled."""
        if session.canceled:
            logger.info("speak: canceled")
            return False
        await self._send(session, {"type": "chunk", "index": index, "text": chunk})
        results = await self.run_mlx(_start_chunk, model, chunk, request)
        try:
            while (audio := await self.run_mlx(_next_audio, results)) is not None:
                if session.canceled:
                    logger.info("speak: canceled")
                    return False
                await self._send(session, {"type": "audio"}, audio)
        finally:
            await self.run_mlx(results.close)
        return True

    async def _generate(
        self,
        session: _Session,
        request: dict[str, Any],
        text_queue: asyncio.Queue[str | None],
    ) -> None:
        """Generate the text chunk by chunk and send each audio piece once ready."""
        logger.info(
            "speak: model %s, language %s, voice %s, sampling %s, max chars %d, "
            "seed %d",
            request["model"],
            request["language"],
            request["ref_audio"],
            request["sampling"],
            request["max_chars"],
            request["seed"],
        )
        try:
            try:
                model = await self.run_mlx(self.model_for, request["model"])
            # Report any load error, e.g. a mistyped model ID, to the client.
            except Exception as error:  # pylint: disable=broad-exception-caught
                logger.exception("loading model %s failed", request["model"])
                await self._send(session, {"type": "error", "message": str(error)})
                return
            if request["seed"] != 0:
                await self.run_mlx(_seed, request["seed"])
            hello = {"type": "hello", "model": self.model_id}
            await self._send(session, hello | {"sample_rate": model.sample_rate})
            chunker = chunking.IncrementalChunks(request["max_chars"])
            index = 0
            while True:
                piece = await text_queue.get()
                if piece is None:
                    break
                for chunk in chunker.feed(piece):
                    if not await self._emit_chunk(
                        session,
                        model,
                        request,
                        index,
                        chunk,
                    ):
                        return
                    index += 1
            for chunk in chunker.flush():
                if not await self._emit_chunk(session, model, request, index, chunk):
                    return
                index += 1
            if session.canceled:
                return
            await self._send(session, {"type": "done"})
            logger.info("speak: done (%d chunks)", index)
        finally:
            await self.run_mlx(_release_cached_memory)


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


async def _serve(model_id: str) -> None:
    """Load the model, then serve clients until a "stop" request, SIGTERM, or Ctrl-C."""
    loop = asyncio.get_running_loop()
    stop_requested = asyncio.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop_requested.set)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx") as executor:
        # Pass the model straight in: a local variable here would keep it in
        # memory after QttsServer.model_for() swapped it out.
        server = QttsServer(
            model_id,
            await loop.run_in_executor(executor, _load_model, model_id),
            executor,
            stop_requested,
        )
        unix_server = await asyncio.start_unix_server(
            server.handle_client,
            path=str(SOCKET_PATH),
        )
        logger.info("ready on %s", SOCKET_PATH)
        await stop_requested.wait()
        logger.info("stopping")
        unix_server.close()
        unix_server.close_clients()
        await unix_server.wait_closed()


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

    logger.info("starting (log file for autostarted servers: %s)", LOG_PATH)
    # A socket file left over from a crashed server blocks bind(). Holding the
    # lock guarantees that no live server uses it.
    SOCKET_PATH.unlink(missing_ok=True)
    try:
        asyncio.run(_serve(args.model))
    finally:
        SOCKET_PATH.unlink(missing_ok=True)
        lock_file.close()
        logger.info("stopped")


if __name__ == "__main__":
    # Ctrl-C before the event loop installs its signal handlers.
    with suppress(KeyboardInterrupt):
        main()
