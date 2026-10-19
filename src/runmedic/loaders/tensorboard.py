"""Reads scalar summaries from TensorBoard event files without depending on tensorflow/tensorboard.

Event files are TFRecord streams of `Event` protobufs. We only need a tiny part of the schema:

    Event   { double wall_time = 1; int64 step = 2; Summary summary = 5; }
    Summary { repeated Value value = 1; }
    Value   { string tag = 1; float simple_value = 2; TensorProto tensor = 8; }
    TensorProto { int32 dtype = 1; bytes tensor_content = 4; repeated float float_val = 5; repeated double double_val = 6; }
"""

from __future__ import annotations

import struct
from pathlib import Path

from runmedic.model import Run

DT_FLOAT, DT_DOUBLE = 1, 2


def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7


def _fields(buf: bytes):
    """Yields (field_number, wire_type, value) for a serialized protobuf message."""
    pos, end = 0, len(buf)
    while pos < end:
        key, pos = _varint(buf, pos)
        num, wire = key >> 3, key & 7
        if wire == 0:
            val, pos = _varint(buf, pos)
        elif wire == 1:
            val, pos = buf[pos : pos + 8], pos + 8
        elif wire == 2:
            size, pos = _varint(buf, pos)
            val, pos = buf[pos : pos + size], pos + size
        elif wire == 5:
            val, pos = buf[pos : pos + 4], pos + 4
        else:  # groups are not used in these messages
            return
        yield num, wire, val


def _tensor_scalar(buf: bytes) -> float | None:
    dtype, content, floats, doubles = 0, b"", [], []
    for num, wire, val in _fields(buf):
        if num == 1:
            dtype = val
        elif num == 4:
            content = val
        elif num == 5:
            floats += struct.unpack(f"<{len(val) // 4}f", val) if wire == 2 else struct.unpack("<f", val)
        elif num == 6:
            doubles += struct.unpack(f"<{len(val) // 8}d", val) if wire == 2 else struct.unpack("<d", val)
    if dtype == DT_FLOAT:
        if floats:
            return floats[0] if len(floats) == 1 else None
        if len(content) == 4:
            return struct.unpack("<f", content)[0]
    if dtype == DT_DOUBLE:
        if doubles:
            return doubles[0] if len(doubles) == 1 else None
        if len(content) == 8:
            return struct.unpack("<d", content)[0]
    return None


def _value(buf: bytes) -> tuple[str, float] | None:
    tag, value = None, None
    for num, _, val in _fields(buf):
        if num == 1:
            tag = val.decode("utf-8", "replace")
        elif num == 2:
            value = struct.unpack("<f", val)[0]
        elif num == 8:
            value = _tensor_scalar(val)
    if tag is None or value is None:
        return None
    return tag, value


def _records(path: Path):
    data = path.read_bytes()
    pos = 0
    while pos + 12 <= len(data):
        (length,) = struct.unpack("<Q", data[pos : pos + 8])
        start = pos + 12
        if start + length + 4 > len(data):  # truncated tail of a run that is still writing
            break
        yield data[start : start + length]
        pos = start + length + 4


def read_events(path: Path, run: Run) -> None:
    for record in _records(path):
        step, summary = 0, None
        for num, _, val in _fields(record):
            if num == 2:
                step = val
            elif num == 5:
                summary = val
        if summary is None:
            continue
        for num, _, val in _fields(summary):
            if num != 1:
                continue
            parsed = _value(val)
            if parsed:
                run.get(parsed[0]).add(step, parsed[1])


def load_dir(directory: Path) -> Run:
    run = Run(name=directory.name, source="tensorboard", path=str(directory))
    for f in sorted(directory.glob("events.out.tfevents*")):
        read_events(f, run)
    return run.finalize()
