"""RealtimeEngine: microphone -> VAD -> streaming ASR -> stabilizer -> (LLM) -> expression -> TTS -> player.

Threads (all daemon, all talking through bounded queues):

``listen``  reads capture blocks, runs VAD/endpointing and ASR, commits stable
            text. The capture callback itself only enqueues, so a slow ASR
            pass delays recognition but never loses or blocks microphone audio.
``transform`` optional LLM rewrite + expression compilation per segment.
``synth``   streams TTS audio into the player, segment after segment.
``status``  writes a JSON snapshot for ``aivoice experimental status``.

Each utterance gets a uid and an :class:`UtteranceTimeline`; every stage stamps
it with ``time.monotonic()``. Speculative mode synthesizes committed (stable)
phrases while the user is still talking; otherwise synthesis starts at the
endpoint with the final transcript.
"""

from __future__ import annotations

import enum
import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..asr.base import ASRBackend, streaming_view
from ..audio.playback import Player, PlaybackEvent
from ..audio.resample import StreamResampler
from ..expression import ExpressionCompiler
from ..stabilize import Commit, TranscriptStabilizer
from ..tts.base import TTSBackend
from ..vad import Endpointer
from .timeline import SegmentTiming, TimelineStore, UtteranceTimeline, new_uid


class EngineState(str, enum.Enum):
    IDLE = "IDLE"
    STARTING = "STARTING"
    LISTENING = "LISTENING"
    CAPTURING = "CAPTURING"
    FINALIZING = "FINALIZING"
    MUTED = "MUTED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


@dataclass
class EngineOptions:
    speculative: bool = True
    agreement: int = 2
    min_commit_words: int = 3
    commit_on_punctuation: bool = True
    min_confidence: float | None = None
    barge_in: str = "queue"  # queue | interrupt | duck
    duck_db: float = -12.0
    max_backlog_ms: int = 4000
    input_mode: str = "vad"  # vad | ptt
    echo_gate: bool = True
    monitor: bool = False  # output audible to the mic (speakers / loopback)
    barge_threshold: float = 0.9
    queue_size: int = 8
    output_rate: int | None = None  # None: decided by the TTS warmup
    drain_on_eof: bool = True
    status_path: Path | None = None


@dataclass
class _Job:
    uid: str
    seg: SegmentTiming
    key: str
    cancel: threading.Event = field(default_factory=threading.Event)
    emotion: str | None = None
    delivery: str | None = None


def _put_drop_oldest(q: queue.Queue, item: Any, on_drop: Callable[[Any], None]) -> None:
    while True:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            try:
                old = q.get_nowait()
                on_drop(old)
            except queue.Empty:
                pass


class RealtimeEngine:
    def __init__(
        self,
        *,
        capture: Any,
        asr: ASRBackend,
        endpointer: Endpointer,
        tts: TTSBackend,
        driver_factory: Callable[[Player], Any],
        llm: Any | None = None,
        compiler: ExpressionCompiler | None = None,
        voice: Any | None = None,
        options: EngineOptions | None = None,
        prebuffer_ms: int = 60,
        gain_db: float = 0.0,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.capture = capture
        self.asr = streaming_view(asr)
        self.endpointer = endpointer
        self.tts = tts
        self.driver_factory = driver_factory
        self.llm = llm
        self.voice = voice
        self.opt = options or EngineOptions()
        self.compiler = compiler or ExpressionCompiler(getattr(tts.capabilities(), "dialect", "none"))
        self.prebuffer_ms = prebuffer_ms
        self.gain_db = gain_db
        self.on_event = on_event
        self.timelines = TimelineStore()
        self.state = EngineState.IDLE
        self.player: Player | None = None
        self.driver: Any = None
        self._resampler: StreamResampler | None = None
        self._stab = TranscriptStabilizer(
            agreement=self.opt.agreement,
            min_commit_words=self.opt.min_commit_words,
            commit_on_punctuation=self.opt.commit_on_punctuation,
            min_confidence=self.opt.min_confidence,
            speculative=self._speculative_allowed(),
        )
        self._tq: queue.Queue = queue.Queue(maxsize=self.opt.queue_size)
        self._sq: queue.Queue = queue.Queue(maxsize=self.opt.queue_size)
        self._jobs: dict[str, _Job] = {}
        self._jobs_lock = threading.Lock()
        self._running = threading.Event()
        self._threads: list[threading.Thread] = []
        self._utt: UtteranceTimeline | None = None
        self._seq = 0
        self._muted = False
        self._ptt = False
        self._pending_asr: ASRBackend | None = None
        self.dropped_segments = 0
        self.last_partial = ""
        self.last_final = ""
        self.last_spoken = ""
        self.errors: list[str] = []
        self.input_level = 0.0
        self.t_started: float | None = None
        self.warmup_ms: float | None = None

    # --- setup -----------------------------------------------------------

    def _speculative_allowed(self) -> bool:
        if not self.opt.speculative:
            return False
        # An LLM rewrite needs the whole sentence: speculative fragments would be rewritten out of context.
        if self.llm is not None and self.llm.enabled and getattr(self.llm, "backend", None) is not None:
            return self.llm.mode == "verbatim"
        return True

    def _emit(self, kind: str, **data: Any) -> None:
        if self.on_event:
            try:
                self.on_event(kind, data)
            except Exception:  # noqa: BLE001
                pass

    def prepare(self, *, warmup: bool = True) -> None:
        self.state = EngineState.STARTING
        if not self.asr.loaded:
            self.asr.load_model()
        rate = self.opt.output_rate
        if warmup:
            t0 = time.perf_counter()
            try:
                for ch in self.tts.synthesize("Ready.", voice=self.voice):
                    rate = rate or ch.sample_rate
                    self.warmup_ms = (time.perf_counter() - t0) * 1000
                    break
            except Exception as e:  # noqa: BLE001
                self.errors.append(f"TTS warmup failed: {e}")
                self._emit("error", stage="tts-warmup", error=str(e))
        rate = rate or 48000
        self.player = Player(rate, prebuffer_ms=self.prebuffer_ms, gain_db=self.gain_db, on_event=self._on_playback)
        self.driver = self.driver_factory(self.player)

    def start(self, *, warmup: bool = True) -> None:
        if self.player is None:
            self.prepare(warmup=warmup)
        self._running.set()
        self.driver.start()
        self.capture.start()
        self.t_started = time.monotonic()
        for name, fn in (
            ("aivoice-listen", self._listen_loop),
            ("aivoice-transform", self._transform_loop),
            ("aivoice-synth", self._synth_loop),
        ):
            t = threading.Thread(target=fn, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        if self.opt.status_path:
            t = threading.Thread(target=self._status_loop, name="aivoice-status", daemon=True)
            t.start()
            self._threads.append(t)
        self.state = EngineState.MUTED if self._muted else EngineState.LISTENING
        self._emit("state", state=self.state.value)

    # --- controls --------------------------------------------------------

    def mute(self, on: bool = True) -> None:
        self._muted = on
        if on and self._utt is not None:
            self._finish(time.monotonic(), forced=True)
        self.state = EngineState.MUTED if on else EngineState.LISTENING
        self._emit("state", state=self.state.value)

    def toggle_mute(self) -> bool:
        self.mute(not self._muted)
        return self._muted

    def set_ptt(self, pressed: bool) -> None:
        self._ptt = pressed

    def interrupt(self, reason: str = "interrupted") -> int:
        """Cancel every pending and playing segment (barge-in / emergency stop)."""
        n = 0
        with self._jobs_lock:
            jobs = list(self._jobs.values())
        for j in jobs:
            if not j.cancel.is_set():
                j.cancel.set()
                j.seg.cancelled = j.seg.cancelled or reason
                n += 1
        for q in (self._tq, self._sq):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        if self.player:
            self.player.cancel(None)
        self._emit("interrupt", reason=reason, cancelled=n)
        return n

    def emergency_stop(self) -> None:
        self.interrupt("emergency stop")
        self.mute(True)

    def switch_asr(self, backend: ASRBackend) -> None:
        """Load *backend* now; it replaces the current one at the next utterance boundary."""
        def load():
            try:
                b = streaming_view(backend)
                if not b.loaded:
                    b.load_model()
                self._pending_asr = b
                self._emit("asr-ready", backend=b.get_capabilities().name)
            except Exception as e:  # noqa: BLE001
                self.errors.append(f"ASR switch failed: {e}")
                self._emit("error", stage="asr-switch", error=str(e))

        threading.Thread(target=load, name="aivoice-asr-switch", daemon=True).start()

    def stop(self, *, timeout: float = 3.0) -> None:
        if self.state in (EngineState.STOPPED, EngineState.IDLE) and not self._running.is_set():
            return
        self.state = EngineState.STOPPING
        self._running.clear()
        for t in self._threads:
            t.join(timeout=timeout)
        self._threads.clear()
        try:
            self.capture.stop()
        finally:
            if self.driver:
                self.driver.stop()
        self.state = EngineState.STOPPED
        self._write_status()
        self._emit("state", state=self.state.value)

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Until capture hit EOF and everything queued has played (file-driven runs)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            busy = (
                not getattr(self.capture, "eof", False)
                or self._utt is not None
                or not self._tq.empty()
                or not self._sq.empty()
                or bool(self._jobs)
                or (self.player is not None and self.player.pending)
            )
            if not busy:
                return True
            time.sleep(0.01)
        return False

    # --- listen thread ---------------------------------------------------

    def _listen_loop(self) -> None:
        try:
            while self._running.is_set():
                item = self.capture.read(0.05)
                if item is None:
                    if getattr(self.capture, "eof", False) and self._utt is not None:
                        self._finish(time.monotonic(), forced=True)
                    continue
                t, block = item
                self.input_level = max(self.input_level * 0.8, float(np.abs(block).max()) if block.size else 0.0)
                if self._utt is None and self._pending_asr is not None:
                    old, self.asr, self._pending_asr = self.asr, self._pending_asr, None
                    try:
                        old.unload_model()
                    except Exception:  # noqa: BLE001
                        pass
                if self._muted:
                    continue
                if self.opt.input_mode == "ptt":
                    self._ptt_block(t, block)
                    continue
                playing = self.player is not None and self.player.is_playing
                base = self.endpointer.on
                if self.opt.echo_gate and self.opt.monitor and playing:
                    self.endpointer.on = max(base, self.opt.barge_threshold)
                try:
                    events = self.endpointer.process(block, t)
                finally:
                    self.endpointer.on = base
                for ev in events:
                    if ev.kind == "start":
                        self._begin(t, ev.audio, forced=ev.forced)
                    elif ev.kind == "audio":
                        self._feed(ev.audio)
                    elif ev.kind == "end":
                        self._finish(t, forced=ev.forced)
        except Exception as e:  # noqa: BLE001
            self.state = EngineState.ERROR
            self.errors.append(f"listen loop crashed: {e!r}")
            self._emit("error", stage="listen", error=repr(e))

    def _ptt_block(self, t: float, block: np.ndarray) -> None:
        if self._ptt and self._utt is None:
            self._begin(t, np.zeros(0, np.float32))
        if self._utt is not None:
            if self._ptt:
                self._feed(block)
            else:
                self._finish(t, forced=False)

    def _begin(self, t: float, preroll: np.ndarray, *, forced: bool = False) -> None:
        if self._utt is not None:
            self._finish(t, forced=True)
        if not forced:
            busy = bool(self._jobs) or (self.player is not None and self.player.is_playing)
            if busy and self.opt.barge_in == "interrupt":
                self.interrupt("barge-in")
            elif busy and self.opt.barge_in == "duck" and self.player:
                self.player.duck(True, self.opt.duck_db)
        uid = new_uid()
        self._utt = self.timelines.add(
            UtteranceTimeline(
                uid=uid,
                asr_backend=self.asr.get_capabilities().name,
                t_speech_start=t - preroll.size / 16000,
                silence_ms=float(getattr(self.endpointer, "min_silence_ms", 0)),
                forced_split=forced,
            )
        )
        self._seq = 0
        self._stab.reset()
        self._stab.speculative = self._speculative_allowed()
        try:
            self.asr.reset_stream()
        except Exception:  # noqa: BLE001
            pass
        self.state = EngineState.CAPTURING
        self._emit("utterance-start", uid=uid)
        if preroll.size:
            self._feed(preroll)

    def _feed(self, audio: np.ndarray) -> None:
        utt = self._utt
        if utt is None or not audio.size:
            return
        try:
            ups = self.asr.transcribe_stream(audio, 16000)
        except Exception as e:  # noqa: BLE001
            utt.errors.append(f"asr: {e}")
            self._emit("error", stage="asr", error=str(e), uid=utt.uid)
            return
        now = time.monotonic()
        for u in ups:
            if utt.t_first_partial is None:
                utt.t_first_partial = now
            utt.partials.append((now, u.revision, u.text))
            self.last_partial = u.text
            self._emit("partial", uid=utt.uid, text=u.text, revision=u.revision)
            conf = [w.confidence for w in u.words if w.confidence is not None] if u.words else None
            for c in self._stab.update(u.text, confidences=conf or None, revision=u.revision):
                self._commit(utt, c, now)

    def _finish(self, t: float, *, forced: bool) -> None:
        utt = self._utt
        if utt is None:
            return
        self.state = EngineState.FINALIZING
        utt.t_endpoint = t
        if forced:
            utt.silence_ms = 0.0
        t0 = time.monotonic()
        try:
            final = self.asr.finalize_utterance()
            text = final.text
        except Exception as e:  # noqa: BLE001
            utt.errors.append(f"asr-final: {e}")
            text = self._stab.latest
        now = time.monotonic()
        utt.asr_final_ms = (now - t0) * 1000
        utt.t_final = now
        utt.final_text = text
        self.last_final = text
        self._emit("final", uid=utt.uid, text=text)
        c = self._stab.finalize(text)
        if c is not None:
            self._commit(utt, c, now)
        self._utt = None
        if self.opt.barge_in == "duck" and self.player:
            self.player.duck(False)
        self.state = EngineState.MUTED if self._muted else EngineState.LISTENING

    def _commit(self, utt: UtteranceTimeline, c: Commit, now: float) -> None:
        self._seq += 1
        seg = SegmentTiming(seq=self._seq, text=c.text, final=c.final, t_commit=now)
        utt.segments.append(seg)
        utt.queue_depth = self._tq.qsize() + self._sq.qsize()
        job = _Job(uid=utt.uid, seg=seg, key=f"{utt.uid}#{seg.seq}")
        with self._jobs_lock:
            self._jobs[job.key] = job
        _put_drop_oldest(self._tq, job, self._drop)
        self._emit("commit", uid=utt.uid, seq=seg.seq, text=c.text, final=c.final)

    def _drop(self, job: _Job, reason: str = "queue full") -> None:
        job.cancel.set()
        job.seg.cancelled = job.seg.cancelled or reason
        self.dropped_segments += 1
        with self._jobs_lock:
            self._jobs.pop(job.key, None)

    # --- transform thread ------------------------------------------------

    def _transform_loop(self) -> None:
        while self._running.is_set():
            try:
                job: _Job = self._tq.get(timeout=0.05)
            except queue.Empty:
                continue
            if job.cancel.is_set():
                self._finish_job(job)
                continue
            text = job.seg.text
            emotion = delivery = None
            if self.llm is not None and self.llm.enabled:
                job.seg.t_llm_start = time.monotonic()
                try:
                    r = self.llm.transform(text)
                    text, emotion, delivery = r.transformed, r.emotion, r.delivery
                    job.seg.llm_fallback = r.fallback_reason
                except Exception as e:  # noqa: BLE001
                    job.seg.llm_fallback = f"error: {e}"
                job.seg.t_llm_end = time.monotonic()
            compiled = self.compiler.compile(text, emotion=emotion, delivery=delivery)
            job.seg.tts_text = compiled.text
            _put_drop_oldest(self._sq, job, self._drop)

    # --- synth thread ----------------------------------------------------

    def _synth_loop(self) -> None:
        while self._running.is_set():
            try:
                job: _Job = self._sq.get(timeout=0.05)
            except queue.Empty:
                continue
            seg = job.seg
            if not job.cancel.is_set() and (time.monotonic() - seg.t_commit) * 1000 > self.opt.max_backlog_ms:
                self._drop(job, "backlog")
            if job.cancel.is_set() or not (seg.tts_text or "").strip():
                self._finish_job(job)
                continue
            seg.t_tts_request = time.monotonic()
            assert self.player is not None
            try:
                for ch in self.tts.synthesize(seg.tts_text, voice=self.voice, cancel=job.cancel):
                    if job.cancel.is_set():
                        break
                    if seg.t_tts_first_chunk is None:
                        seg.t_tts_first_chunk = time.monotonic()
                    pcm = ch.pcm
                    if ch.sample_rate != self.player.sample_rate:
                        if self._resampler is None or self._resampler.sr_in != ch.sample_rate:
                            self._resampler = StreamResampler(ch.sample_rate, self.player.sample_rate)
                        pcm = self._resampler.process(pcm)
                    seg.audio_s += pcm.size / self.player.sample_rate
                    self.player.write(job.key, pcm)
            except Exception as e:  # noqa: BLE001
                seg.error = str(e)
                self.errors.append(f"tts: {e}")
                self._emit("error", stage="tts", error=str(e), uid=job.uid)
            if job.cancel.is_set():
                self.player.cancel(job.key)
                self._finish_job(job)
            else:
                self.player.end(job.key)
                self.last_spoken = seg.tts_text or ""
                if seg.audio_s == 0:
                    self._finish_job(job)

    def _finish_job(self, job: _Job) -> None:
        with self._jobs_lock:
            self._jobs.pop(job.key, None)

    def _on_playback(self, ev: PlaybackEvent) -> None:  # audio thread: dict lookups only
        job = self._jobs.get(ev.uid)
        if job is None:
            return
        if ev.kind == "first_audio":
            job.seg.t_first_played = ev.t
        elif ev.kind == "done":
            job.seg.t_done = ev.t
            self._jobs.pop(ev.uid, None)
        elif ev.kind == "cancelled":
            job.seg.cancelled = job.seg.cancelled or "cancelled"
            self._jobs.pop(ev.uid, None)

    # --- status ----------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        rep = self.timelines.report()
        caps = self.asr.get_capabilities()
        return {
            "pid": os.getpid(),
            "state": self.state.value,
            "muted": self._muted,
            "uptime_s": (time.monotonic() - self.t_started) if self.t_started else 0,
            "asr": {"backend": caps.name, "streaming": caps.streaming_method, "notes": caps.notes},
            "tts": self.tts.capabilities().to_dict(),
            "llm": {"mode": getattr(self.llm, "mode", "off") if self.llm else "off",
                    "backend": getattr(getattr(self.llm, "backend", None), "name", None)},
            "voice": getattr(self.voice, "name", None),
            "queues": {"transform": self._tq.qsize(), "synth": self._sq.qsize(), "jobs": len(self._jobs)},
            "player": {
                "rate": self.player.sample_rate if self.player else None,
                "buffered_ms": self.player.buffered_ms() if self.player else 0,
                "underruns": self.player.underruns if self.player else 0,
                "peak": self.player.peak if self.player else 0,
            },
            "capture": {"dropped_blocks": getattr(self.capture, "dropped", 0),
                        "overflows": getattr(self.capture, "overflows", 0),
                        "level": self.input_level},
            "dropped_segments": self.dropped_segments,
            "last": {"partial": self.last_partial, "final": self.last_final, "spoken": self.last_spoken},
            "latency": rep,
            "warmup_ms": self.warmup_ms,
            "errors": self.errors[-10:],
            "updated": time.time(),
        }

    def _write_status(self) -> None:
        p = self.opt.status_path
        if not p:
            return
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.snapshot(), default=str) + "\n", encoding="utf-8")
            os.replace(tmp, p)
        except OSError:
            pass

    def _status_loop(self) -> None:
        while self._running.is_set():
            self._write_status()
            time.sleep(0.5)
