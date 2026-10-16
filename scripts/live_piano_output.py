"""Optional local SF2 piano playback of strict-live note events.

Rendering runs in the sounddevice output callback. Recognition never waits for
synthesis; an unbounded SimpleQueue contains only tiny MIDI control events.
"""
from __future__ import annotations

import queue
import threading
import numpy as np
import sys

from scripts.live_audio_io import open_output_stream, parse_device


class PianoOutput:
    def __init__(self, soundfont, output_device=None, gain=0.35, samplerate=44100, blocksize=128):
        try:
            import fluidsynth
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "Piano output requires pyfluidsynth and native FluidSynth DLL "
                "(libfluidsynth on PATH)."
            ) from exc
        self.commands = queue.SimpleQueue()
        self.active = set()
        self.closed = False
        self.errors = 0
        self.consecutive_errors = 0
        self.last_render_error = None
        self.output_status_count = 0
        self.last_output_status = None
        self.callback_count = 0
        self.max_control_events_in_callback = 0
        self.same_callback_on_off_pairs = 0
        self._lock = threading.Lock()
        self.synth = fluidsynth.Synth(samplerate=samplerate, gain=gain)
        self.stream = None
        self.audio_info = None
        try:
            sfid = self.synth.sfload(str(soundfont))
            if sfid < 0:
                raise RuntimeError(f"Could not load piano SoundFont: {soundfont}")
            # General MIDI bank 0, program 0 = acoustic grand piano.
            self.synth.program_select(0, sfid, 0, 0)

            def callback(outdata, frames, _timing, status):
                if status:
                    # Do not raise in an audio callback: capture dropout data.
                    self.output_status_count += 1
                    self.last_output_status = str(status)
                self.callback_count += 1
                midi_count = 0
                new_on_in_this_callback = set()
                try:
                    while True:
                        try:
                            kind, pitch, velocity = self.commands.get_nowait()
                        except queue.Empty:
                            break
                        midi_count += 1
                        if kind == "on":
                            new_on_in_this_callback.add(pitch)
                            self.synth.noteon(0, pitch, velocity)
                        elif kind == "off":
                            if pitch in new_on_in_this_callback:
                                # MIDI on/off were both consumed before any
                                # samples were rendered; attack may be silent.
                                self.same_callback_on_off_pairs += 1
                                new_on_in_this_callback.discard(pitch)
                            self.synth.noteoff(0, pitch)
                        elif kind == "panic":
                            self.synth.cc(0, 123, 0)
                            self.synth.cc(0, 120, 0)
                    self.max_control_events_in_callback = max(
                        self.max_control_events_in_callback, midi_count)
                    samples = np.asarray(self.synth.get_samples(frames))
                    if samples.size != frames * 2:
                        raise RuntimeError("Unexpected stereo sample count")
                    if np.issubdtype(samples.dtype, np.integer):
                        values = samples.astype(np.float32) / 32768.0
                    else:
                        values = samples.astype(np.float32, copy=False)
                    outdata[:] = values.reshape(frames, 2)
                    self.consecutive_errors = 0
                except Exception as exc:
                    self.errors += 1
                    self.consecutive_errors += 1
                    self.last_render_error = repr(exc)
                    outdata.fill(0.0)

            self.stream, self.audio_info = open_output_stream(
                preferred_device=parse_device(output_device),
                samplerate=samplerate,
                blocksize=blocksize,
                channels=2,
                dtype="float32",
                latency="low",
                callback=callback,
            )
            print(
                f"Piano audio output: {self.audio_info.host_api} / "
                f"{self.audio_info.device_name} [{self.audio_info.device_index}] / "
                f"negotiated latency={self.audio_info.latency_ms:.2f} ms",
                file=sys.stderr,
            )
        except Exception:
            self.close()
            raise

    def send(self, event):
        kind = event["type"]
        pitch = int(event["pitch"])
        if not 0 <= pitch <= 127:
            return
        with self._lock:
            if self.closed:
                return
            if kind == "note_on":
                velocity = int(event.get("velocity", 100))
                self.active.add(pitch)
                self.commands.put(("on", pitch, max(1, min(127, velocity))))
            elif kind == "note_off":
                self.active.discard(pitch)
                self.commands.put(("off", pitch, 0))

    def diagnostics(self):
        """Passive output callback quality counters (no causal model changes)."""
        return dict(
            callback_count=self.callback_count,
            output_status_count=self.output_status_count,
            last_output_status=self.last_output_status,
            max_control_events_in_callback=self.max_control_events_in_callback,
            same_callback_on_off_pairs=self.same_callback_on_off_pairs,
            synthesis_error_count=self.errors,
            last_render_error=self.last_render_error,
            reported_output_latency_ms=(
                self.audio_info.latency_ms if self.audio_info is not None else None),
        )

    def health_error(self):
        if self.consecutive_errors >= 3:
            return f"FluidSynth failed to render piano audio: {self.last_render_error}"
        if self.stream is not None and not self.closed and not bool(getattr(self.stream, "active", True)):
            return "Piano audio output stream stopped unexpectedly"
        return None

    def close(self):
        with self._lock:
            if self.closed:
                return
            self.closed = True
            self.active.clear()
        if self.stream is not None:
            try:
                self.stream.stop()
            finally:
                self.stream.close()
        self.synth.delete()
