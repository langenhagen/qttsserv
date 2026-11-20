# Qttsserv

A Qwen3-TTS server and CLI client for quick and responsive TTS on macOS.

Keeps one [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) model loaded in a background process and streams its speech
to the speakers while it is still generating.

- `qtts_server.py` loads one model with
  [mlx-audio](https://github.com/Blaizzy/mlx-audio) and serves any number of
  clients, but generates for one request at a time. A request for another
  model drops the loaded model from memory first.
- `qtts_client.py` reads the text from a file, the clipboard, or stdin, and
  plays the audio as it arrives. Ctrl-C stops playback and generation.
- `qtts.sh` runs the client through the project venv; link it into `PATH` as
  `qtts`.
- `qtts_protocol.py` holds the socket location and the message format.
- `chunking.py` splits the text into chunks; copied from my
  [qwen3-tts-hello](https://github.com/langenhagen/experiments-and-tutorials/blob/master/Python/qwen3-tts-hello/chunking.py).

If not already running, the first CLI invocation starts the server.

## Setup

```bash
uv sync
sudo ln -sf "$PWD/qtts.sh" /usr/local/bin/qtts  # link the CLI so you can use it everywhere
```

## Usage

```bash
qtts --help
qtts FILE                                           # speak a text file in random voice
qtts FILE --voice me.wav                            # speak a text file in the cloned voice of me.wav
qtts FILE --voice me.wav --transcript me.txt        # speak a text file in a more accurately cloned voice
qtts FILE --voice-design "A deep, calm male voice"  # describe a new voice (1.7B only)
qtts --clipboard --voice me.wav                     # speak the clipboard
printf 'One. Two.\n\nThree.' | qtts --voice me.wav  # stream piped text as it arrives
qtts FILE --voice me.wav --language german          # set the language instead of auto-detecting it
qtts FILE --voice me.wav --temperature 0.7          # steadier delivery; default 1.3, higher is livelier
qtts FILE --voice me.wav --seed 42                  # same delivery on every run; default 0 is random
qtts FILE --voice me.wav --model-size 0.6B --precision bf16  # smaller model, full precision
qtts FILE --voice me.wav --prebuffer 2.3            # prebuffer to avoid pauses on slow machines
qtts FILE --voice me.wav --verbose                  # also show model, settings, and chunk numbers
qtts --stop-playback                                # silence the running qtts call
qtts --stop-server                                  # stop the server and free its memory
```

`--model-size` is `1.7B` (default) or `0.6B`, and `--precision` is `4` (default), `5`,
`6`, `8`, or `bf16` for full precision; fewer bits are smaller and faster. The
default,
[mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit),
holds about 2.4 GB of RAM while idle.

`--voice-design` loads the `VoiceDesign` model, which speaks in the voice you
describe in words instead of cloning a reference. It needs `--model-size 1.7B`;
there is no 0.6B VoiceDesign model. It cannot be combined with `--voice` or
`--transcript`.

## How It Works

Client and server talk over a Unix domain socket: a file-system path that only
the current user can open, with no TCP port and no network exposure. The
socket, the lock file, and the server log `qttsserv.log` live in
`~/.cache/qttsserv/`.

If no server listens, the client starts one detached from the terminal and
waits until the model is loaded. A lock file guarantees at most one server per
user, so concurrent clients never load the model twice. Nothing restarts the
server after a crash; the next `qtts` call starts a fresh one.

The server splits the text into chunks and generates each chunk in streaming
mode. It sends about every 2 s of audio to the client while it still generates
the rest of the chunk, so playback starts before the whole text is done. If the
client disconnects, for example on Ctrl-C, the server aborts the generation.

When stdin is a pipe, the client sends text to the server as the producer
writes it. The server speaks each chunk as soon as its boundaries are final,
so a slow producer does not delay the text already received. Chunk boundaries
are the same as for a text file: blank lines, sentence ends, and `--max-chars`.

A new `qtts` call cancels the running one. Each client stays connected until
its playback ends. When a new request comes in, the server tells the running
client to stop.

## License

See [LICENSE](LICENSE) file.
