"""Emit V8 associated onset/offset events from WAV or raw PCM16 stdin."""
import argparse
from array import array
from pathlib import Path
import sys
import wave

from causal_note.v8_predictor import V8KerasPredictor
from causal_note.v8_runtime import AnonymousEventAssociator, V8BoundaryDecoder


SAMPLE_RATE = 44_100


def decode_pcm16(raw: bytes):
    if len(raw) % 2:
        raise ValueError("PCM16 input ended with an incomplete sample")
    values = array("h")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return tuple(value / 32768.0 for value in values)


class V8LivePipeline:
    def __init__(
        self,
        predictor,
        *,
        onset_threshold: float,
        offset_threshold: float,
        onset_release_threshold=None,
        offset_release_threshold=None,
    ) -> None:
        self.predictor = predictor
        self.decoder = V8BoundaryDecoder(
            onset_threshold=onset_threshold,
            offset_threshold=offset_threshold,
            onset_release_threshold=onset_release_threshold,
            offset_release_threshold=offset_release_threshold,
        )
        self.associator = AnonymousEventAssociator()
        self.next_sample = 0

    def process_chunk(self, samples):
        scores = self.predictor.predict_chunk(
            samples,
            start_sample=self.next_sample,
        )
        boundaries = self.decoder.process_chunk(scores)
        events = self.associator.process(boundaries)
        self.next_sample += len(samples)
        return events

    def finalize_stream(self):
        return self.associator.finalize_stream(self.next_sample)


def emit(events):
    for event in events:
        print(
            f"{event.kind.value}({event.event_id}, {event.sample})",
            flush=True,
        )


def stream_wav(path: Path, pipeline: V8LivePipeline, chunk_size: int):
    with wave.open(str(path), "rb") as audio:
        if audio.getnchannels() != 1:
            raise ValueError("V8 live detector requires mono audio")
        if audio.getsampwidth() != 2:
            raise ValueError("V8 live detector requires PCM16 audio")
        if audio.getframerate() != SAMPLE_RATE:
            raise ValueError("V8 live detector requires 44100 Hz audio")
        while True:
            raw = audio.readframes(chunk_size)
            if not raw:
                break
            emit(pipeline.process_chunk(decode_pcm16(raw)))
    emit(pipeline.finalize_stream())


def stream_stdin(pipeline: V8LivePipeline, chunk_size: int):
    byte_count = chunk_size * 2
    while True:
        raw = sys.stdin.buffer.read(byte_count)
        if not raw:
            break
        emit(pipeline.process_chunk(decode_pcm16(raw)))
    emit(pipeline.finalize_stream())


def create_argument_parser():
    parser = argparse.ArgumentParser(
        description="V8 causal anonymous onset/offset output with opaque IDs"
    )
    parser.add_argument("--model", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wav", type=Path)
    source.add_argument("--stdin-pcm16", action="store_true")
    parser.add_argument("--receptive-field", type=int, default=4093)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--onset-threshold", type=float, default=0.5)
    parser.add_argument("--offset-threshold", type=float, default=0.5)
    parser.add_argument("--onset-release-threshold", type=float)
    parser.add_argument("--offset-release-threshold", type=float)
    return parser


def main(argv=None):
    args = create_argument_parser().parse_args(argv)
    if args.chunk_size <= 0 or args.receptive_field <= 0:
        raise ValueError("chunk-size and receptive-field must be > 0")
    predictor = V8KerasPredictor.from_path(
        str(args.model),
        receptive_field=args.receptive_field,
    )
    predictor.warm_up(args.chunk_size)
    predictor.reset()
    pipeline = V8LivePipeline(
        predictor,
        onset_threshold=args.onset_threshold,
        offset_threshold=args.offset_threshold,
        onset_release_threshold=args.onset_release_threshold,
        offset_release_threshold=args.offset_release_threshold,
    )
    if args.wav is not None:
        stream_wav(args.wav, pipeline, args.chunk_size)
    else:
        stream_stdin(pipeline, args.chunk_size)


if __name__ == "__main__":
    main()
