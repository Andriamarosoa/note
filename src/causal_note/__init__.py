"""Temporal onset/offset detection and exact frame renewal."""

from .audio_buffer import (
    AudioDiscardedError,
    AudioBufferError,
    CausalAudioBuffer,
    FrameAvailability,
    InsufficientAudioError,
    NonContiguousAudioError,
)
from .detector import (
    ActiveEvent,
    BoundaryCandidate,
    BoundaryEvent,
    BoundaryScoreChunk,
    BoundaryScorePredictor,
    BoundaryType,
    FRAME_SIZE,
    LiveBoundaryPeakDecoder,
    LiveBoundaryScoreDecoder,
    LiveEnergyDetector,
    LiveEventTracker,
    LiveModelDetector,
    SAMPLE_RATE,
)
from .guitarset import (
    ALLOWED_PLAYERS,
    BoundarySlots,
    GuitarSetError,
    GuitarSetFormatError,
    GuitarSetTrack,
    NoteBoundary,
    index_guitarset,
    load_boundary_slots,
)
from .keras_predictor import KerasBoundaryPredictor
from .neural_model import (
    EVENT_SLOTS,
    build_causal_boundary_model,
    calculate_receptive_field,
)
from .pipeline import LiveChunkResult, LiveOnsetOffsetPipeline
from .scheduler import (
    RestartFrame,
    RestartScheduler,
    SchedulerError,
    UnknownEventError,
)

__all__ = [
    "AudioBufferError",
    "AudioDiscardedError",
    "ActiveEvent",
    "ALLOWED_PLAYERS",
    "BoundaryCandidate",
    "BoundaryEvent",
    "BoundaryScoreChunk",
    "BoundaryScorePredictor",
    "BoundarySlots",
    "BoundaryType",
    "CausalAudioBuffer",
    "FRAME_SIZE",
    "FrameAvailability",
    "GuitarSetError",
    "GuitarSetFormatError",
    "GuitarSetTrack",
    "InsufficientAudioError",
    "KerasBoundaryPredictor",
    "LiveBoundaryPeakDecoder",
    "LiveBoundaryScoreDecoder",
    "LiveChunkResult",
    "LiveEnergyDetector",
    "LiveEventTracker",
    "LiveModelDetector",
    "LiveOnsetOffsetPipeline",
    "NoteBoundary",
    "NonContiguousAudioError",
    "RestartFrame",
    "RestartScheduler",
    "SAMPLE_RATE",
    "SchedulerError",
    "UnknownEventError",
    "EVENT_SLOTS",
    "build_causal_boundary_model",
    "calculate_receptive_field",
    "index_guitarset",
    "load_boundary_slots",
]
