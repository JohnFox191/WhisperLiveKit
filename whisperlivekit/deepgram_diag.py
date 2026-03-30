"""Diagnostic instrumentation for the Deepgram-compat endpoint.

Hooks into AudioProcessor to dump audio, trace VAC decisions, and detect
race conditions. Enable by setting DEEPGRAM_DIAG=1 environment variable.

Produces:
  - /tmp/wlk_diag/ffmpeg_out.raw   — raw 16kHz s16le PCM from FFmpeg stdout
  - /tmp/wlk_diag/enqueued.raw     — PCM that actually reached the transcription queue
  - /tmp/wlk_diag/vac_log.jsonl    — per-chunk VAC decisions
  - /tmp/wlk_diag/race_log.txt     — race condition traces
  - /tmp/wlk_diag/summary.txt      — final summary
"""

import json
import logging
import os
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

DIAG_DIR = Path(os.environ.get("WLK_DIAG_DIR", str(Path(tempfile.gettempdir()) / "wlk_diag")))
_enabled = os.environ.get("DEEPGRAM_DIAG", "0") == "1"


def is_enabled() -> bool:
    return _enabled


class DiagSession:
    """Per-session diagnostic collector."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.dir = DIAG_DIR / session_id
        self.dir.mkdir(parents=True, exist_ok=True)

        self._ffmpeg_out = open(self.dir / "ffmpeg_out.raw", "wb")
        self._enqueued = open(self.dir / "enqueued.raw", "wb")
        self._vac_log = open(self.dir / "vac_log.jsonl", "w", encoding="utf-8")
        self._race_log = open(self.dir / "race_log.txt", "w", encoding="utf-8")

        self.stats = {
            "ffmpeg_bytes": 0,
            "enqueued_bytes": 0,
            "enqueued_chunks": 0,
            "dropped_by_vac": 0,
            "dropped_by_silence": 0,
            "vac_speech_start": 0,
            "vac_speech_end": 0,
            "vac_no_decision": 0,
            "process_audio_calls": 0,
            "handle_pcm_calls": 0,
            "total_input_bytes": 0,
        }
        self._start = time.monotonic()
        self._last_process_audio_time = 0.0
        self._last_handle_pcm_time = 0.0
        self._stopping_at: Optional[float] = None
        self._sentinel_at: Optional[float] = None
        self._ffmpeg_stop_at: Optional[float] = None

        print(f"[DIAG] Session {session_id} started, writing to {self.dir}")

    def on_ffmpeg_output(self, chunk: bytes):
        """Called when ffmpeg_stdout_reader gets data."""
        self._ffmpeg_out.write(chunk)
        self.stats["ffmpeg_bytes"] += len(chunk)

    def on_audio_enqueued(self, pcm_array: np.ndarray):
        """Called when audio reaches _enqueue_active_audio."""
        pcm_bytes = (pcm_array * 32768.0).astype(np.int16).tobytes()
        self._enqueued.write(pcm_bytes)
        self.stats["enqueued_bytes"] += len(pcm_bytes)
        self.stats["enqueued_chunks"] += 1

    def on_vac_decision(self, chunk_samples: int, total_samples: int,
                        rms: float, vac_result: Any, current_silence: bool,
                        was_enqueued: bool):
        """Called after VAC processes a chunk."""
        entry = {
            "t": round(time.monotonic() - self._start, 3),
            "chunk_samples": chunk_samples,
            "total_samples": total_samples,
            "rms": round(float(rms), 6),
            "vac_result": str(vac_result) if vac_result is not None else None,
            "has_start": "start" in vac_result if isinstance(vac_result, dict) else False,
            "has_end": "end" in vac_result if isinstance(vac_result, dict) else False,
            "in_silence": current_silence,
            "enqueued": was_enqueued,
        }
        self._vac_log.write(json.dumps(entry) + "\n")

        if vac_result is not None and isinstance(vac_result, dict):
            if "start" in vac_result:
                self.stats["vac_speech_start"] += 1
            if "end" in vac_result:
                self.stats["vac_speech_end"] += 1
        else:
            self.stats["vac_no_decision"] += 1

        if not was_enqueued:
            if current_silence:
                self.stats["dropped_by_silence"] += 1
            else:
                self.stats["dropped_by_vac"] += 1

    def on_process_audio(self, data_len: int, is_stop: bool):
        """Called at the start of process_audio."""
        now = time.monotonic()
        self.stats["process_audio_calls"] += 1
        if data_len:
            self.stats["total_input_bytes"] += data_len
        self._last_process_audio_time = now
        if is_stop:
            self._stopping_at = now

    def on_handle_pcm(self, buf_len: int, threshold: int):
        """Called at the start of handle_pcm_data."""
        now = time.monotonic()
        self.stats["handle_pcm_calls"] += 1
        self._last_handle_pcm_time = now
        if buf_len < threshold:
            self._race_log.write(
                f"[{now - self._start:.3f}] handle_pcm: buf={buf_len}B < threshold={threshold}B, skipping\n"
            )

    def on_sentinel_sent(self):
        """Called when SENTINEL is put in the transcription queue."""
        self._sentinel_at = time.monotonic()

    def on_ffmpeg_stopped(self):
        """Called when FFmpeg manager is stopped."""
        self._ffmpeg_stop_at = time.monotonic()

    def on_race_event(self, msg: str):
        """Log a potential race condition."""
        t = time.monotonic() - self._start
        self._race_log.write(f"[{t:.3f}] {msg}\n")
        print(f"[DIAG RACE] [{t:.3f}] {msg}")

    def finish(self):
        """Write summary and close files."""
        elapsed = time.monotonic() - self._start

        # Check for race conditions in shutdown sequence
        if self._stopping_at and self._sentinel_at and self._ffmpeg_stop_at:
            sentinel_to_stop = self._ffmpeg_stop_at - self._sentinel_at
            if sentinel_to_stop < 0.1:
                self.on_race_event(
                    f"SENTINEL→FFmpeg stop gap only {sentinel_to_stop:.3f}s — "
                    f"FFmpeg may have been killed before reader drained stdout"
                )

        ffmpeg_sec = self.stats["ffmpeg_bytes"] / 32000 if self.stats["ffmpeg_bytes"] else 0
        enqueued_sec = self.stats["enqueued_bytes"] / 32000 if self.stats["enqueued_bytes"] else 0
        input_sec = self.stats["total_input_bytes"] / 96000 if self.stats["total_input_bytes"] else 0  # 48kHz s16le

        summary = [
            f"Session: {self.session_id}",
            f"Duration: {elapsed:.1f}s",
            f"",
            f"Audio flow:",
            f"  Input from client:     {self.stats['total_input_bytes']:>10d} bytes ({input_sec:.1f}s @ 48kHz)",
            f"  FFmpeg output:         {self.stats['ffmpeg_bytes']:>10d} bytes ({ffmpeg_sec:.1f}s @ 16kHz)",
            f"  Enqueued to model:     {self.stats['enqueued_bytes']:>10d} bytes ({enqueued_sec:.1f}s @ 16kHz)",
            f"  Enqueued chunks:       {self.stats['enqueued_chunks']}",
            f"  Dropped (in silence):  {self.stats['dropped_by_silence']}",
            f"  Dropped (by VAC):      {self.stats['dropped_by_vac']}",
            f"",
            f"VAC decisions:",
            f"  Speech start:          {self.stats['vac_speech_start']}",
            f"  Speech end:            {self.stats['vac_speech_end']}",
            f"  No decision (pass):    {self.stats['vac_no_decision']}",
            f"",
            f"Pipeline calls:",
            f"  process_audio:         {self.stats['process_audio_calls']}",
            f"  handle_pcm_data:       {self.stats['handle_pcm_calls']}",
            f"",
            f"Shutdown timing:",
            f"  Stopping at:           {(self._stopping_at or 0) - self._start:.3f}s",
            f"  Sentinel at:           {(self._sentinel_at or 0) - self._start:.3f}s",
            f"  FFmpeg stop at:        {(self._ffmpeg_stop_at or 0) - self._start:.3f}s",
        ]

        summary_text = "\n".join(summary)
        with open(self.dir / "summary.txt", "w") as f:
            f.write(summary_text)

        print(f"\n[DIAG] === SESSION SUMMARY ===\n{summary_text}\n[DIAG] === END ===\n")

        self._ffmpeg_out.close()
        self._enqueued.close()
        self._vac_log.close()
        self._race_log.close()

        # Convert raw dumps to WAV for easy playback
        for name in ("ffmpeg_out", "enqueued"):
            raw_path = self.dir / f"{name}.raw"
            wav_path = self.dir / f"{name}.wav"
            try:
                raw_data = raw_path.read_bytes()
                if raw_data:
                    with wave.open(str(wav_path), "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(16000)
                        wf.writeframes(raw_data)
            except Exception as e:
                print(f"[DIAG] Failed to convert {name}.raw to WAV: {e}")

        print(f"[DIAG] Audio dumps at: {self.dir}")
        print(f"[DIAG]   ffmpeg_out.wav  - full audio from FFmpeg")
        print(f"[DIAG]   enqueued.wav    - audio that reached the model")
