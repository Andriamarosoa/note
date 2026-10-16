"""Optional local SF2 piano playback of strict-live note events.

Rendering runs in the sounddevice output callback. Recognition never waits for
synthesis; an unbounded SimpleQueue contains only tiny MIDI control events.
"""
from __future__ import annotations

import queue
import threading
import numpy as np


class PianoOutput:
    def __init__(self, soundfont, output_device=None, gain=0.35, samplerate=44100, blocksize=128):
        import sounddevice as sd
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
        self._lock = threading.Lock()
        self.synth = fluidsynth.Synth(samplerate=samplerate, gain=gain)
        self.stream = None
        try:
            sfid = self.synth.sfload(str(soundfont))
            if sfid < 0:
                raise RuntimeError(f"Could not load piano SoundFont: {soundfont}")
            # General MIDI bank 0, program 0 = acoustic grand piano.
            self.synth.program_select(0, sfid, 0, 0)

            def callback(outdata, frames, _timing, status):
                try:
                    while True:
                        try:
                            kind, pitch, velocity = self.commands.get_nowait()
                        except queue.Empty:
                            break
                        if kind == "on":
                            self.synth.noteon(0, pitch, velocity)
                        elif kind == "off":
                            self.synth.noteoff(0, pitch)
                        elif kind == "panic":
                            self.synth.cc(0, 123, 0)
                            self.synth.cc(0, 120, 0)
                    samples = np.asarray(self.synth.get_samples(frames))
                    if samples.size != frames * 2:
                        raise RuntimeError("Unexpected stereo sample count")
                    if np.issubdtype(samples.dtype, np.integer):
                        values = samples.astype(np.float32) / 32768.0
                    else:
                        values = samples.astype(np.float32, copy=False)
                    outdata[:] = values.reshape(frames, 2)
                except Exception:
                    self.errors += 1
                    outdata.fill(0.0)

            selected = (
                int(output_device) if isinstance(output_device, str)
                and output_device.isdigit() else output_device
            )
            self.stream = sd.OutputStream(
                samplerate=samplerate, blocksize=blocksize, channels=2,
                dtype="float32", latency="low", device=selected, callback=callback
            )
            self.stream.start()
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
