"""Emit associated onset/offset events from a WAV or a raw PCM16 stream."""

import argparse
from array import array
import json
from pathlib import Path
import sys
import wave

from causal_note.detector import LiveModelDetector
from causal_note.keras_predictor import KerasBoundaryPredictor
from causal_note.pipeline import LiveOnsetOffsetPipeline


SAMPLE_RATE = 44_100


def decode_pcm16(raw: bytes):
    if len(raw) % 2:
        raise ValueError("PCM16 input ended with an incomplete sample")
    values = array("h")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return tuple(value / 32768.0 for value in values)


def emit(pipeline, samples, start_sample):
    result = pipeline.process_chunk(samples, start_sample=start_sample)
    for event in result.events:
        print(
            f"{event.kind.value}({event.event_id}, {event.sample})",
            flush=True,
        )
    # The continuous causal model has already analysed these exact samples.
    # Acknowledge ready renewal frames so scheduler state remains bounded.
    for frame in result.ready_frames:
        pipeline.complete_frame(frame.frame_start)
    return start_sample + len(samples)


def finalize(pipeline):
    """Emit terminal control offsets exactly once after source EOF."""

    result = pipeline.finalize_stream()
    for event in result.events:
        print(
            f"{event.kind.value}({event.event_id}, {event.sample})",
            flush=True,
        )
    for frame in result.ready_frames:
        pipeline.complete_frame(frame.frame_start)
    return result


def stream_wav(path, pipeline, chunk_size):
    position = 0
    with wave.open(str(path), "rb") as audio:
        if audio.getnchannels() != 1:
            raise ValueError("live detector requires mono audio")
        if audio.getsampwidth() != 2:
            raise ValueError("live detector requires PCM16 audio")
        if audio.getframerate() != SAMPLE_RATE:
            raise ValueError("live detector requires audio at 44100 Hz")
        while True:
            raw = audio.readframes(chunk_size)
            if not raw:
                break
            position = emit(pipeline, decode_pcm16(raw), position)
    finalize(pipeline)


def stream_stdin(pipeline, chunk_size):
    position = 0
    byte_count = chunk_size * 2
    while True:
        raw = sys.stdin.buffer.read(byte_count)
        if not raw:
            break
        position = emit(pipeline, decode_pcm16(raw), position)
    finalize(pipeline)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Causal live onset/offset output with opaque event IDs"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--metadata",
        type=Path,
        help="training metadata JSON (defaults to MODEL.metadata.json)",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wav", type=Path)
    source.add_argument(
        "--stdin-pcm16",
        action="store_true",
        help="read mono 44100 Hz little-endian PCM16 from stdin",
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--onset-threshold", type=float, default=0.5)
    parser.add_argument("--offset-threshold", type=float, default=0.5)
    parser.add_argument("--onset-release-threshold", type=float)
    parser.add_argument("--offset-release-threshold", type=float)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be > 0")
    metadata_path = args.metadata or args.model.with_suffix(".metadata.json")
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    receptive_field = metadata.get("receptive_field")
    if receptive_field is None:
        receptive_field = metadata.get("model", {}).get(
            "receptive_field_samples"
        )
    predictor = KerasBoundaryPredictor.from_path(
        str(args.model),
        receptive_field=receptive_field,
    )
    predictor.warm_up(args.chunk_size)
    detector = LiveModelDetector(
        predictor,
        onset_threshold=args.onset_threshold,
        offset_threshold=args.offset_threshold,
        onset_release_threshold=args.onset_release_threshold,
        offset_release_threshold=args.offset_release_threshold,
    )
    pipeline = LiveOnsetOffsetPipeline(detector)
    if args.wav is not None:
        stream_wav(args.wav, pipeline, args.chunk_size)
    else:
        stream_stdin(pipeline, args.chunk_size)


if __name__ == "__main__":
    main()
