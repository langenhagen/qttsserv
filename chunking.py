"""Split a text into chunks of at most max_chars characters for text-to-speech.

Copied from qwen3-tts-hello's chunking.py, without its command-line interface.

Blank lines separate paragraphs; a chunk never crosses one. Within a
paragraph, line breaks become spaces, so hard-wrapped lines are joined. Then,
until the paragraph is used up: take the first max_chars characters and cut
after the last sentence end in them (". ", "? ", "! ", also before closing
quotes or brackets like `." ` or `!) `). Without a sentence end, cut after the
last "; ", else after the last ", ", else at the last space, else at
max_chars. Once the rest of the paragraph fits, it is the last chunk. So short
sentences are packed together greedily, in order.
"""

import itertools
import re

# Places to cut, best first; each pattern ends in the space to cut at.
CUT_PATTERNS = (
    # Sentence end, also before closing quotes or brackets, e.g. `." ` or `!) `
    re.compile(r"[.!?][\"'\u201c\u201d\u2018\u2019\u00ab\u00bb)\]]* "),
    re.compile(r"; "),
    re.compile(r", "),
    re.compile(r" "),
)


def find_cut(window: str) -> int:
    """Return the index of the space after the best cut place, or -1."""
    for pattern in CUT_PATTERNS:
        matches = list(pattern.finditer(window))
        if matches:
            return matches[-1].end() - 1
    return -1


def split_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """Cut one paragraph into chunks; see the module docstring."""
    rest = " ".join(paragraph.split())
    chunks = []
    while len(rest) > max_chars:
        cut = find_cut(rest[: max_chars + 1])
        if cut != -1:
            after_space = cut + 1
            chunk, rest = rest[:cut], rest[after_space:]
        else:
            chunk, rest = rest[:max_chars], rest[max_chars:]
        chunks.append(chunk)
    if rest:
        chunks.append(rest)
    return chunks


def split_chunks(text: str, max_chars: int) -> list[str]:
    """Return the chunks of all paragraphs of the text in one list."""
    return list(
        itertools.chain.from_iterable(
            split_paragraph(paragraph, max_chars)
            for paragraph in re.split(r"\n\s*\n", text)
            if paragraph.strip()
        ),
    )
