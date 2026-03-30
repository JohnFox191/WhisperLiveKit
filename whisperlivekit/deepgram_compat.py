"""Deepgram-compatible WebSocket endpoint for WhisperLiveKit.

Provides a /v1/listen endpoint that speaks the Deepgram Live Transcription
protocol, enabling drop-in compatibility with Deepgram client SDKs.

Protocol mapping:
  - Client sends binary audio frames -> forwarded to AudioProcessor
  - Client sends JSON control messages (KeepAlive, CloseStream, Finalize)
  - Server sends Results, Metadata, UtteranceEnd messages

Differences from Deepgram:
  - No authentication required (self-hosted)
  - Word-level timestamps approximate (interpolated from segment boundaries)
  - Confidence scores not available (set to 0.0)
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

# Maps Deepgram encoding names to FFmpeg input format arguments.
# When encoding is specified, the client sends raw headerless audio and we
# must tell FFmpeg exactly what format to expect via -f / -acodec / -ar / -ac.
# When encoding is omitted, audio is containerized and FFmpeg auto-detects.
_ENCODING_TO_FFMPEG = {
    "linear16": {"format": "s16le", "codec": "pcm_s16le"},
    "linear32": {"format": "f32le", "codec": "pcm_f32le"},
    "mulaw": {"format": "mulaw", "codec": "pcm_mulaw"},
    "alaw": {"format": "alaw", "codec": "pcm_alaw"},
    "flac": {"format": "flac", "codec": None},
    "opus": {"format": "ogg", "codec": None},
    "ogg-opus": {"format": "ogg", "codec": None},
    "speex": {"format": "ogg", "codec": None},
}


def _build_ffmpeg_input_args(encoding: Optional[str], sample_rate: int,
                             channels: int) -> list:
    """Build FFmpeg input format arguments for the given Deepgram encoding."""
    if not encoding:
        return []
    spec = _ENCODING_TO_FFMPEG.get(encoding)
    if not spec:
        return []
    args = ["-f", spec["format"]]
    if spec["codec"]:
        args += ["-acodec", spec["codec"]]
    args += ["-ar", str(sample_rate), "-ac", str(channels)]
    return args


def _parse_time_str(time_str: str) -> float:
    """Parse 'H:MM:SS.cc' to seconds."""
    parts = time_str.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


def _line_to_words(line: dict) -> list:
    """Convert a line dict to Deepgram-style word objects."""
    text = line.get("text", "")
    if not text or not text.strip():
        return []

    start = _parse_time_str(line.get("start", "0:00:00"))
    end = _parse_time_str(line.get("end", "0:00:00"))
    speaker = line.get("speaker", 0)
    if speaker == -2:
        return []

    words = text.split()
    if not words:
        return []

    duration = end - start
    step = duration / max(len(words), 1)

    return [
        {
            "word": w,
            "start": round(start + i * step, 3),
            "end": round(start + (i + 1) * step, 3),
            "confidence": 0.0,
            "punctuated_word": w,
            "speaker": speaker if speaker > 0 else 0,
        }
        for i, w in enumerate(words)
    ]


def _lines_to_result(lines: list, is_final: bool, speech_final: bool,
                     start_time: float = 0.0) -> dict:
    """Convert FrontData lines to a Deepgram Results message."""
    all_words = []
    full_text_parts = []

    for line in lines:
        if line.get("speaker") == -2:
            continue
        words = _line_to_words(line)
        all_words.extend(words)
        text = line.get("text", "")
        if text and text.strip():
            full_text_parts.append(text.strip())

    transcript = " ".join(full_text_parts)

    if all_words:
        seg_start = all_words[0]["start"]
        seg_end = all_words[-1]["end"]
        duration = seg_end - seg_start
    else:
        seg_start = start_time
        seg_end = start_time
        duration = 0.0

    return {
        "type": "Results",
        "channel_index": [0, 1],
        "duration": round(duration, 3),
        "start": round(seg_start, 3),
        "is_final": is_final,
        "speech_final": speech_final,
        "channel": {
            "alternatives": [
                {
                    "transcript": transcript,
                    "confidence": 0.0,
                    "words": all_words,
                }
            ]
        },
    }


class DeepgramAdapter:
    """Adapts WhisperLiveKit's FrontData stream to Deepgram's protocol."""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.request_id = str(uuid.uuid4())
        self._prev_n_lines = 0
        self._sent_lines = 0
        self._last_word_end = 0.0
        self._speech_started_sent = False
        self._vad_events = False

    async def send_metadata(self, config):
        """Send initial Metadata message."""
        backend = getattr(config, "backend", "whisper") if config else "whisper"
        msg = {
            "type": "Metadata",
            "request_id": self.request_id,
            "sha256": "",
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "duration": 0,
            "channels": 1,
            "models": [backend],
            "model_info": {
                backend: {
                    "name": backend,
                    "version": "whisperlivekit",
                }
            },
        }
        await self.websocket.send_json(msg)

    async def process_update(self, front_data_dict: dict):
        """Convert a FrontData dict into Deepgram messages and send them."""
        lines = front_data_dict.get("lines", [])
        buffer = front_data_dict.get("buffer_transcription", "")

        speech_lines = [l for l in lines if l.get("speaker", 0) != -2]
        n_speech = len(speech_lines)

        # Detect new committed lines -> emit as is_final=true results
        if n_speech > self._sent_lines:
            new_lines = speech_lines[self._sent_lines:]
            result = _lines_to_result(new_lines, is_final=True, speech_final=True)
            await self.websocket.send_json(result)

            if result["channel"]["alternatives"][0]["words"]:
                self._last_word_end = result["channel"]["alternatives"][0]["words"][-1]["end"]

            self._sent_lines = n_speech

        # Emit buffer as interim result (is_final=false)
        elif buffer and buffer.strip():
            if self._vad_events and not self._speech_started_sent:
                await self.websocket.send_json({
                    "type": "SpeechStarted",
                    "channel_index": [0],
                    "timestamp": 0.0,
                })
                self._speech_started_sent = True

            interim = {
                "type": "Results",
                "channel_index": [0, 1],
                "duration": 0.0,
                "start": self._last_word_end,
                "is_final": False,
                "speech_final": False,
                "channel": {
                    "alternatives": [
                        {
                            "transcript": buffer.strip(),
                            "confidence": 0.0,
                            "words": [],
                        }
                    ]
                },
            }
            await self.websocket.send_json(interim)

        # Detect silence -> emit UtteranceEnd
        silence_lines = [l for l in lines if l.get("speaker") == -2]
        if silence_lines and n_speech > 0:
            for sil in silence_lines:
                sil_start = _parse_time_str(sil.get("start", "0:00:00"))
                if sil_start >= self._last_word_end:
                    await self.websocket.send_json({
                        "type": "UtteranceEnd",
                        "channel": [0, 1],
                        "last_word_end": round(self._last_word_end, 3),
                    })
                    self._speech_started_sent = False
                    break


async def handle_deepgram_websocket(websocket: WebSocket, transcription_engine, config):
    """Handle a Deepgram-compatible WebSocket session."""
    from whisperlivekit.audio_processor import AudioProcessor
    from whisperlivekit.deepgram_diag import DiagSession, is_enabled as diag_enabled
    from whisperlivekit.ffmpeg_manager import FFmpegManager

    # Parse Deepgram query parameters
    params = websocket.query_params
    language = params.get("language", None)
    encoding = params.get("encoding", None)
    sample_rate = int(params.get("sample_rate", "16000"))
    channels = int(params.get("channels", "1"))
    vad_events = params.get("vad_events", "false").lower() == "true"

    print(f"[DEEPGRAM-COMPAT] params: encoding={encoding}, sample_rate={sample_rate}, channels={channels}, language={language}")
    logger.info(
        "Deepgram-compat params: encoding=%s, sample_rate=%d, channels=%d, language=%s",
        encoding, sample_rate, channels, language,
    )

    audio_processor = AudioProcessor(
        transcription_engine=transcription_engine,
        language=language,
    )

    # Attach diagnostic session if enabled (DEEPGRAM_DIAG=1)
    diag = None
    if diag_enabled():
        diag = DiagSession(f"dg_{int(time.time())}")
        audio_processor._diag = diag

    # When the client declares a raw encoding, replace the AudioProcessor's
    # FFmpegManager with one that has explicit input format flags so FFmpeg
    # knows what the raw byte stream contains (sample rate, codec, channels).
    input_args = _build_ffmpeg_input_args(encoding, sample_rate, channels)
    print(f"[DEEPGRAM-COMPAT] input_args={input_args}, has_ffmpeg={audio_processor.ffmpeg_manager is not None}")
    if input_args and audio_processor.ffmpeg_manager is not None:
        print(f"[DEEPGRAM-COMPAT] Replacing FFmpegManager with: {input_args}")
        old_callback = audio_processor.ffmpeg_manager.on_error_callback
        audio_processor.ffmpeg_manager = FFmpegManager(
            sample_rate=audio_processor.sample_rate,
            channels=audio_processor.channels,
            input_format_args=input_args,
        )
        audio_processor.ffmpeg_manager.on_error_callback = old_callback


    # Echo back the "token" subprotocol if the client requested it.
    # Per RFC 6455 S4.2.2, the client MUST fail the connection if it sent
    # Sec-WebSocket-Protocol but the server didn't select one.
    requested_protocols = websocket.headers.get("sec-websocket-protocol", "")
    subprotocol = "token" if "token" in requested_protocols else None
    await websocket.accept(subprotocol=subprotocol)
    logger.info("Deepgram-compat WebSocket opened (subprotocol=%s)", subprotocol)

    adapter = DeepgramAdapter(websocket)
    adapter._vad_events = vad_events

    # Send metadata
    await adapter.send_metadata(config)

    results_generator = await audio_processor.create_tasks()

    # Debug counters
    _dbg = {"chunks_in": 0, "bytes_in": 0, "results_out": 0}

    # Results consumer
    async def handle_results():
        try:
            async for response in results_generator:
                rd = response.to_dict()
                _dbg["results_out"] += 1
                lines = rd.get("lines", [])
                buf = rd.get("buffer_transcription", "")
                if _dbg["results_out"] <= 5 or _dbg["results_out"] % 20 == 0:
                    print(f"[DEEPGRAM-COMPAT] result #{_dbg['results_out']}: "
                          f"lines={len(lines)}, buffer='{buf[:60]}', "
                          f"pcm_buf={len(audio_processor.pcm_buffer)}B, "
                          f"total_samples={audio_processor.total_pcm_samples}")
                await adapter.process_update(rd)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.exception(f"Deepgram compat results error: {e}")

    results_task = asyncio.create_task(handle_results())

    # Audio / control message consumer
    try:
        while True:
            try:
                message = await asyncio.wait_for(
                    websocket.receive(), timeout=30.0,
                )
            except asyncio.TimeoutError:
                break

            if "bytes" in message:
                data = message["bytes"]
                if data:
                    _dbg["chunks_in"] += 1
                    _dbg["bytes_in"] += len(data)
                    if _dbg["chunks_in"] <= 3 or _dbg["chunks_in"] % 100 == 0:
                        ffm = audio_processor.ffmpeg_manager
                        ffm_state = ffm.state.value if ffm else "N/A"
                        print(f"[DEEPGRAM-COMPAT] chunk #{_dbg['chunks_in']}: "
                              f"{len(data)}B (total {_dbg['bytes_in']}B), "
                              f"ffmpeg={ffm_state}, "
                              f"pcm_buf={len(audio_processor.pcm_buffer)}B, "
                              f"results_so_far={_dbg['results_out']}")
                    await audio_processor.process_audio(data)
                else:
                    await audio_processor.process_audio(b"")
                    break
            elif "text" in message:
                try:
                    ctrl = json.loads(message["text"])
                    msg_type = ctrl.get("type", "")

                    if msg_type == "CloseStream":
                        await audio_processor.process_audio(b"")
                        break
                    elif msg_type == "Finalize":
                        await audio_processor.process_audio(b"")
                        results_generator = await audio_processor.create_tasks()
                    elif msg_type == "KeepAlive":
                        pass
                    else:
                        logger.debug("Unknown Deepgram control message: %s", msg_type)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON control message")
            else:
                break

    except WebSocketDisconnect:
        logger.info("Deepgram-compat WebSocket disconnected")
    except Exception as e:
        logger.error(f"Deepgram-compat error: {e}", exc_info=True)
    finally:
        if not results_task.done():
            results_task.cancel()
        try:
            await results_task
        except (asyncio.CancelledError, Exception):
            pass
        await audio_processor.cleanup()
        if diag:
            diag.finish()
        logger.info("Deepgram-compat WebSocket cleaned up")
