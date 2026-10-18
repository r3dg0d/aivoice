"""Incremental decoder for streamed WAV / raw PCM responses.

fish-speech's streaming mode sends a WAV header first and then raw PCM frames,
with the header's size fields set to placeholders. This parser accepts that, a
complete WAV, or headerless PCM at a known rate, and yields float32 blocks as
soon as whole samples are available - no waiting for the response to finish.
"""

from __future__ import annotations

import struct

import numpy as np


class WavStreamError(ValueError):
    pass


class WavStreamDecoder:
    def __init__(self, *, raw_pcm_rate: int | None = None, channels: int = 1) -> None:
        self._buf = bytearray()
        self.sample_rate: int | None = raw_pcm_rate
        self.channels = channels
        self.bits = 16
        self.fmt_code = 1
        self._in_data = raw_pcm_rate is not None
        self._data_left: int | None = None

    @property
    def ready(self) -> bool:
        return self._in_data

    def _parse_header(self) -> None:
        b = self._buf
        if len(b) < 12:
            return
        if b[:4] != b"RIFF" or b[8:12] != b"WAVE":
            raise WavStreamError("response is not WAV (expected RIFF/WAVE header)")
        pos = 12
        while True:
            if len(b) < pos + 8:
                return
            cid = bytes(b[pos : pos + 4])
            size = struct.unpack("<I", b[pos + 4 : pos + 8])[0]
            if cid == b"fmt ":
                if len(b) < pos + 8 + 16:
                    return
                code, ch, rate, _, _, bits = struct.unpack("<HHIIHH", b[pos + 8 : pos + 24])
                if code == 0xFFFE and len(b) >= pos + 8 + 26:  # WAVE_FORMAT_EXTENSIBLE
                    code = struct.unpack("<H", b[pos + 32 : pos + 34])[0] if len(b) >= pos + 34 else 1
                self.fmt_code, self.channels, self.sample_rate, self.bits = code, ch, rate, bits
                if (code, bits) not in ((1, 16), (1, 24), (1, 32), (3, 32)):
                    raise WavStreamError(f"unsupported WAV encoding (format {code}, {bits}-bit)")
                pos += 8 + size + (size & 1)
            elif cid == b"data":
                if self.sample_rate is None:
                    raise WavStreamError("WAV data chunk before fmt chunk")
                del b[: pos + 8]
                # streaming servers write 0 or 0xFFFFFFFF here: treat as "until EOF"
                self._data_left = None if size in (0, 0xFFFFFFFF) else size
                self._in_data = True
                return
            else:
                if len(b) < pos + 8 + size:
                    return
                pos += 8 + size + (size & 1)

    def feed(self, data: bytes) -> np.ndarray:
        """Append bytes; return any complete samples as mono float32 (may be empty)."""
        self._buf += data
        if not self._in_data:
            self._parse_header()
            if not self._in_data:
                return np.zeros(0, np.float32)
        frame = (self.bits // 8) * self.channels
        avail = len(self._buf)
        if self._data_left is not None:
            avail = min(avail, self._data_left)
        n = (avail // frame) * frame
        if n == 0:
            return np.zeros(0, np.float32)
        raw = bytes(self._buf[:n])
        del self._buf[:n]
        if self._data_left is not None:
            self._data_left -= n
        return self._decode(raw)

    def _decode(self, raw: bytes) -> np.ndarray:
        if self.fmt_code == 3:
            x = np.frombuffer(raw, dtype="<f4")
        elif self.bits == 16:
            x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif self.bits == 32:
            x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        else:  # 24-bit
            b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
            i = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
            i = np.where(i & 0x800000, i - 0x1000000, i)
            x = i.astype(np.float32) / 8388608.0
        if self.channels > 1:
            x = x.reshape(-1, self.channels).mean(axis=1)
        return x.astype(np.float32, copy=False)
