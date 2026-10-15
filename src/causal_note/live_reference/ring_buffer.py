from __future__ import annotations

import numpy as np


class MonoRingBuffer:
    """Fixed-size mono float32 circular buffer.

    The audio callback writes new samples into the ring buffer.
    The inference thread can request the most recent N samples.

    No new array is allocated by ``copy_latest_into``.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")

        self.capacity = int(capacity)
        self._buffer = np.zeros(self.capacity, dtype=np.float32)
        self._write_index = 0
        self._samples_written = 0

    @property
    def samples_written(self) -> int:
        return self._samples_written

    @property
    def available(self) -> int:
        return min(self._samples_written, self.capacity)

    @property
    def is_full(self) -> bool:
        return self._samples_written >= self.capacity

    def reset(self) -> None:
        self._buffer.fill(0.0)
        self._write_index = 0
        self._samples_written = 0

    def write(self, samples: np.ndarray) -> None:
        """Append mono samples to the circular buffer."""
        values = np.asarray(samples, dtype=np.float32).reshape(-1)
        count = int(values.size)

        if count == 0:
            return

        # Only the newest capacity samples can survive.
        if count >= self.capacity:
            self._buffer[:] = values[-self.capacity :]
            self._write_index = 0
            self._samples_written += count
            return

        first_count = min(count, self.capacity - self._write_index)
        self._buffer[self._write_index : self._write_index + first_count] = values[:first_count]

        remaining = count - first_count
        if remaining:
            self._buffer[:remaining] = values[first_count:]

        self._write_index = (self._write_index + count) % self.capacity
        self._samples_written += count

    def copy_latest_into(self, destination: np.ndarray, length: int | None = None) -> np.ndarray:
        """Copy the newest samples into a preallocated float32 array.

        If fewer samples are available, the beginning is zero padded.
        """
        output = np.asarray(destination)
        if output.dtype != np.float32:
            raise TypeError("destination must use dtype float32")
        if output.ndim != 1:
            raise ValueError("destination must be one-dimensional")

        requested = output.size if length is None else int(length)
        if requested <= 0 or requested > output.size:
            raise ValueError("length must be in [1, destination.size]")
        if requested > self.capacity:
            raise ValueError("requested length exceeds ring buffer capacity")

        view = output[:requested]
        view.fill(0.0)

        available = min(requested, self.available)
        if available == 0:
            return view

        start = (self._write_index - available) % self.capacity
        target_start = requested - available

        if start + available <= self.capacity:
            view[target_start:] = self._buffer[start : start + available]
        else:
            first_count = self.capacity - start
            view[target_start : target_start + first_count] = self._buffer[start:]
            view[target_start + first_count :] = self._buffer[: available - first_count]

        return view
