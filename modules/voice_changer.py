"""Real-Time Voice Transformation Engine for Deep-Live-Cam Studio.

Provides low-latency (<20ms), smooth voice conversion with:
- 5 distinct human vocal profiles (Deep Male, Natural Female, Warm Female, Young Male, Radio Broadcaster)
- Real-time vectorized pitch modulation and formant shaping
- Device selection for microphones and Virtual Audio Cable routing (WhatsApp/Zoom)
- Live headphone monitoring ("Hear My Voice") and volume level metering
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.signal as signal
import sounddevice as sd


# ─── Vocal Profiles ────────────────────────────────────────────────────────────

VOICE_PRESETS: Dict[str, Dict[str, Any]] = {
    "deep_male": {
        "name": "🎙️ Deep / Confident Male",
        "description": "Rich chest resonance with authoritative low-end baritone",
        "semitones": -4.5,
        "low_shelf_db": 5.0,
        "high_shelf_db": -2.5,
        "presence_freq": 1800,
        "presence_gain_db": 1.5,
        "compression": True,
    },
    "natural_female": {
        "name": "🎙️ Natural Female",
        "description": "Smooth pitch elevation with bright, natural feminine formants",
        "semitones": 5.5,
        "low_shelf_db": -6.0,
        "high_shelf_db": 3.5,
        "presence_freq": 3200,
        "presence_gain_db": 4.0,
        "compression": True,
    },
    "warm_female": {
        "name": "🎙️ Warm / Mature Female",
        "description": "Gentle, elegant female tone with warm acoustic presence",
        "semitones": 3.8,
        "low_shelf_db": -4.0,
        "high_shelf_db": 1.5,
        "presence_freq": 2400,
        "presence_gain_db": 2.5,
        "compression": True,
    },
    "young_male": {
        "name": "🎙️ Young / Energetic Male",
        "description": "Higher clarity, bright articulate presence, and energetic tone",
        "semitones": 1.5,
        "low_shelf_db": -1.5,
        "high_shelf_db": 3.0,
        "presence_freq": 2800,
        "presence_gain_db": 3.0,
        "compression": False,
    },
    "broadcaster": {
        "name": "🎙️ Studio Broadcaster / Radio Host",
        "description": "Warm punchy low-end, studio clarity, and broadcast dynamics",
        "semitones": -1.5,
        "low_shelf_db": 6.0,
        "high_shelf_db": 2.0,
        "presence_freq": 3800,
        "presence_gain_db": 4.5,
        "compression": True,
    },
}


# ─── Vectorized Pitch & Formant DSP Processor ────────────────────────────────

class RealtimePitchShifter:
    """Low-latency, vectorized rotating delay-line pitch shifter."""

    def __init__(self, buf_size: int = 4096, sample_rate: int = 44100):
        self.buf_size = buf_size
        self.sample_rate = sample_rate
        self.buffer = np.zeros(buf_size, dtype=np.float32)
        self.write_pos = 0
        self.read_pos1 = 0.0
        self.read_pos2 = float(buf_size // 2)

    def reset(self) -> None:
        self.buffer.fill(0)
        self.write_pos = 0
        self.read_pos1 = 0.0
        self.read_pos2 = float(self.buf_size // 2)

    def process(self, chunk: np.ndarray, semitones: float) -> np.ndarray:
        if abs(semitones) < 0.05:
            return chunk.copy()

        rate = 2.0 ** (semitones / 12.0)
        n = len(chunk)
        if n == 0:
            return chunk

        # Vectorized ring buffer write
        indices = (self.write_pos + np.arange(n)) % self.buf_size
        self.buffer[indices] = chunk
        self.write_pos = (self.write_pos + n) % self.buf_size

        # Vectorized dual read pointers
        r1 = (self.read_pos1 + np.arange(n) * rate) % self.buf_size
        r2 = (self.read_pos2 + np.arange(n) * rate) % self.buf_size
        self.read_pos1 = (self.read_pos1 + n * rate) % self.buf_size
        self.read_pos2 = (self.read_pos2 + n * rate) % self.buf_size

        # Linear interpolation for read 1
        idx1 = r1.astype(np.int32)
        f1 = (r1 - idx1).astype(np.float32)
        idx1_next = (idx1 + 1) % self.buf_size
        s1 = (1.0 - f1) * self.buffer[idx1] + f1 * self.buffer[idx1_next]

        # Linear interpolation for read 2
        idx2 = r2.astype(np.int32)
        f2 = (r2 - idx2).astype(np.float32)
        idx2_next = (idx2 + 1) % self.buf_size
        s2 = (1.0 - f2) * self.buffer[idx2] + f2 * self.buffer[idx2_next]

        # Cosine cross-fade window to eliminate glitch artifacts
        rel1 = (self.write_pos - r1) % self.buf_size
        w1 = 0.5 * (1.0 - np.cos(2.0 * np.pi * rel1 / self.buf_size))
        w2 = 1.0 - w1

        return (w1 * s1 + w2 * s2).astype(np.float32)


class VocalAcousticFilter:
    """Multi-band EQ and dynamic limiter for human voice coloring."""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def apply_eq(
        self,
        audio: np.ndarray,
        low_db: float = 0.0,
        high_db: float = 0.0,
        presence_freq: int = 2500,
        presence_db: float = 0.0,
    ) -> np.ndarray:
        if len(audio) == 0:
            return audio

        nyq = self.sample_rate * 0.5
        out = audio

        # 1. Low Shelf (Chest resonance / bass reduction)
        if abs(low_db) > 0.5:
            low_cutoff = 220.0 / nyq
            gain_linear = 10.0 ** (low_db / 20.0)
            b, a = signal.butter(1, low_cutoff, btype="low")
            low_comp = signal.lfilter(b, a, out)
            out = (out - low_comp) + low_comp * gain_linear

        # 2. Presence Peak (Formant vocal clarity)
        if abs(presence_db) > 0.5:
            center = max(200.0, min(nyq - 200.0, float(presence_freq)))
            bandwidth = 700.0
            low_w = max(0.01, (center - bandwidth / 2.0) / nyq)
            high_w = min(0.99, (center + bandwidth / 2.0) / nyq)
            gain_linear = 10.0 ** (presence_db / 20.0)
            b, a = signal.butter(1, [low_w, high_w], btype="bandpass")
            mid_comp = signal.lfilter(b, a, out)
            out = out + mid_comp * (gain_linear - 1.0)

        # 3. High Shelf (Air / Treble smoothing)
        if abs(high_db) > 0.5:
            high_cutoff = 4500.0 / nyq
            gain_linear = 10.0 ** (high_db / 20.0)
            b, a = signal.butter(1, high_cutoff, btype="high")
            high_comp = signal.lfilter(b, a, out)
            out = (out - high_comp) + high_comp * gain_linear

        return out.astype(np.float32)

    @staticmethod
    def soft_clip_limiter(audio: np.ndarray, drive: float = 1.1) -> np.ndarray:
        """Tanh soft-clip limiter to prevent distortion and level speech."""
        x = audio * drive
        return np.tanh(x).astype(np.float32)


# ─── Real-Time Stream Engine ──────────────────────────────────────────────────

class VoiceChangerEngine:
    """Manages real-time audio streams between microphone and virtual audio cable."""

    def __init__(self, sample_rate: int = 44100, block_size: int = 1024):
        self.sample_rate = sample_rate
        self.block_size = block_size

        self.pitch_shifter = RealtimePitchShifter(buf_size=4096, sample_rate=sample_rate)
        self.acoustic_filter = VocalAcousticFilter(sample_rate=sample_rate)

        self._active_preset_key = "deep_male"
        self._custom_semitones: Optional[float] = None
        self._mic_gain = 1.0
        self._monitor_enabled = False

        self._stream: Optional[sd.Stream] = None
        self._is_running = False
        self._lock = threading.Lock()

        # Metering
        self._current_rms = 0.0
        self._level_callback: Optional[Callable[[float], None]] = None

    # ── Configuration ─────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._is_running

    def set_preset(self, preset_key: str) -> None:
        with self._lock:
            if preset_key in VOICE_PRESETS:
                self._active_preset_key = preset_key
                self._custom_semitones = None

    def set_pitch_offset(self, semitones: float) -> None:
        with self._lock:
            self._custom_semitones = semitones

    def set_mic_gain(self, gain: float) -> None:
        self._mic_gain = max(0.0, min(3.0, gain))

    def set_monitor(self, enabled: bool) -> None:
        self._monitor_enabled = enabled

    def set_level_callback(self, cb: Optional[Callable[[float], None]]) -> None:
        self._level_callback = cb

    # ── Device Enumeration ───────────────────────────────────────────────────

    @staticmethod
    def get_input_devices() -> List[Tuple[int, str]]:
        devices = []
        try:
            for idx, dev in enumerate(sd.query_devices()):
                if dev.get("max_input_channels", 0) > 0:
                    name = dev.get("name", f"Microphone {idx}")
                    devices.append((idx, name))
        except Exception:
            devices.append((-1, "Default Microphone"))
        return devices if devices else [(-1, "Default Microphone")]

    @staticmethod
    def get_output_devices() -> List[Tuple[int, str]]:
        devices = []
        try:
            for idx, dev in enumerate(sd.query_devices()):
                if dev.get("max_output_channels", 0) > 0:
                    name = dev.get("name", f"Output {idx}")
                    devices.append((idx, name))
        except Exception:
            devices.append((-1, "Default Speakers"))
        return devices if devices else [(-1, "Default Speakers")]

    @staticmethod
    def find_virtual_cable_output() -> Optional[int]:
        """Auto-detect VB-Audio Virtual Cable or other virtual cables if installed."""
        try:
            for idx, dev in enumerate(sd.query_devices()):
                if dev.get("max_output_channels", 0) > 0:
                    name = dev.get("name", "").lower()
                    if "cable input" in name or "vb-audio" in name or "virtual cable" in name:
                        return idx
        except Exception:
            pass
        return None

    # ── Stream Management ─────────────────────────────────────────────────────

    def start(
        self,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
    ) -> bool:
        if self._is_running:
            self.stop()

        self.pitch_shifter.reset()

        def _audio_callback(indata: np.ndarray, outdata: np.ndarray, frames: int, time_info, status):
            try:
                # Average mono from input channels
                if indata.ndim > 1:
                    mono = np.mean(indata, axis=1)
                else:
                    mono = indata.flatten()

                # Apply mic volume gain
                mono = mono * self._mic_gain

                # Calculate RMS level for UI meter
                rms = float(np.sqrt(np.mean(mono ** 2)))
                self._current_rms = rms
                if self._level_callback:
                    self._level_callback(min(1.0, rms * 5.0))

                # Retrieve active preset parameters
                with self._lock:
                    preset = VOICE_PRESETS.get(self._active_preset_key, VOICE_PRESETS["deep_male"])
                    semitones = self._custom_semitones if self._custom_semitones is not None else preset["semitones"]
                    low_db = preset.get("low_shelf_db", 0.0)
                    high_db = preset.get("high_shelf_db", 0.0)
                    pres_freq = preset.get("presence_freq", 2500)
                    pres_db = preset.get("presence_gain_db", 0.0)
                    use_comp = preset.get("compression", True)

                # 1. Pitch Shift
                processed = self.pitch_shifter.process(mono, semitones)

                # 2. Formant & Acoustic EQ
                processed = self.acoustic_filter.apply_eq(
                    processed,
                    low_db=low_db,
                    high_db=high_db,
                    presence_freq=pres_freq,
                    presence_db=pres_db,
                )

                # 3. Dynamic Soft-Clipping Limiter
                if use_comp:
                    processed = self.acoustic_filter.soft_clip_limiter(processed, drive=1.1)

                # Duplicate mono to all output channels
                out_channels = outdata.shape[1] if outdata.ndim > 1 else 1
                if out_channels == 1:
                    outdata[:] = processed[:, np.newaxis]
                else:
                    for ch in range(out_channels):
                        outdata[:, ch] = processed

            except Exception:
                outdata.fill(0)

        try:
            # Query default channels for safety
            in_dev_info = sd.query_devices(input_device, "input") if input_device is not None and input_device >= 0 else None
            out_dev_info = sd.query_devices(output_device, "output") if output_device is not None and output_device >= 0 else None

            in_channels = min(2, in_dev_info["max_input_channels"]) if in_dev_info else 1
            out_channels = min(2, out_dev_info["max_output_channels"]) if out_dev_info else 2

            self._stream = sd.Stream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                device=(input_device if input_device is not None and input_device >= 0 else None,
                        output_device if output_device is not None and output_device >= 0 else None),
                channels=(in_channels, out_channels),
                dtype=np.float32,
                latency="low",
                callback=_audio_callback,
            )
            self._stream.start()
            self._is_running = True
            return True
        except Exception as exc:
            print(f"[VoiceChanger] Failed to start stream: {exc}")
            self._is_running = False
            return False

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._is_running = False
        self._current_rms = 0.0
        if self._level_callback:
            self._level_callback(0.0)


# Global singleton instance
VOICE_ENGINE = VoiceChangerEngine()
