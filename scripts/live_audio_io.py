"""NOTE audio I/O transport, adapted from Andriamarosoa/midi/src/product/audio_io.py.

Device probing and WASAPI/WDM-KS selection only; NOTE pitch models untouched.
Adaptation: diagnostics go to stderr to keep NOTE JSONL stdout machine-readable.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import threading
import sys
from typing import Any, Callable


def _diagnostic(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)


KERNEL_STREAMING_TARGET_LATENCY_S = 0.002


@dataclass(frozen=True)
class AudioStreamInfo:
    direction: str
    device_index: int | None
    device_name: str
    host_api: str
    native_sample_rate: int
    target_sample_rate: int
    block_size: int
    latency_ms: float
    wasapi_shared: bool
    automatic_selection: bool
    kernel_streaming: bool = False
    requested_latency_ms: float | None = None

    @property
    def conversion_active(self) -> bool:
        return (
            self.wasapi_shared
            and self.native_sample_rate != self.target_sample_rate
        )

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["conversion_active"] = self.conversion_active
        return values


@dataclass(frozen=True)
class _StreamCandidate:
    device_index: int | None
    wasapi_shared: bool
    kernel_streaming: bool
    requested_latency: str | float


def parse_device(value: str | int | None):
    if value is None or (
        isinstance(value, str) and value.strip().casefold() == "auto"
    ):
        return None
    if isinstance(value, int):
        return value
    stripped = value.strip()
    return int(stripped) if stripped.lstrip("-").isdigit() else stripped


def _channel_key(direction: str) -> str:
    if direction not in {"input", "output"}:
        raise ValueError(f"Direction audio invalide: {direction}")
    return f"max_{direction}_channels"


def _default_index(sd, direction: str) -> int | None:
    position = 0 if direction == "input" else 1
    try:
        index = int(sd.default.device[position])
        return index if index >= 0 else None
    except Exception:
        try:
            info = sd.query_devices(kind=direction)
            index = info.get("index")
            return None if index is None else int(index)
        except Exception:
            return None


def _resolve_index(sd, value, direction: str) -> int | None:
    selected = parse_device(value)
    if selected is None:
        return _default_index(sd, direction)
    if isinstance(selected, int):
        info = sd.query_devices(selected)
        if int(info[_channel_key(direction)]) < 1:
            raise ValueError(
                f"Le peripherique audio {selected} ne fournit aucune voie {direction}."
            )
        return selected

    # A numeric PortAudio index is only meaningful for one enumeration. On
    # Windows WDM-KS probing, indices can differ between launches and after
    # stream initialization. Allow a stable host-API qualified endpoint name:
    #     WASAPI:External Microphone
    #     MME:External Microphone
    # Host-qualified lookup must never fall through to an unrelated endpoint.
    host_filter = None
    if ":" in selected:
        maybe_host, name = selected.split(":", 1)
        maybe_host = maybe_host.strip().casefold()
        aliases = {
            "wasapi": "WASAPI",
            "mme": "MME",
            "directsound": "DirectSound",
            "wdm-ks": "WDM-KS",
        }
        if maybe_host in aliases:
            host_filter = aliases[maybe_host].casefold()
            selected = name.strip()
            if not selected:
                raise ValueError("A host-qualified device must include an endpoint name")
    else:
        try:
            info = sd.query_devices(selected, kind=direction)
            if "index" in info:
                return int(info["index"])
        except Exception:
            pass

    needle = selected.casefold()
    matches = [
        index
        for index, info in enumerate(sd.query_devices())
        if int(info[_channel_key(direction)]) > 0
        and needle in str(info["name"]).casefold()
        and (host_filter is None or host_filter in str(
            sd.query_hostapis()[int(info["hostapi"])]["name"]
        ).casefold())
    ]
    if len(matches) != 1:
        qualifier = f"{host_filter}:" if host_filter else ""
        raise ValueError(
            f"Peripherique audio ambigu ou absent: {qualifier}{selected!r}; "
            f"correspondances={matches}; utilisez WASAPI:nom-du-microphone"
        )
    return matches[0]


def _host_index(sd, name: str) -> int | None:
    for index, host in enumerate(sd.query_hostapis()):
        if name.upper() in str(host["name"]).upper():
            return index
    return None


def _wasapi_host_index(sd) -> int | None:
    return _host_index(sd, "WASAPI")


def _wdm_ks_host_index(sd) -> int | None:
    return _host_index(sd, "WDM-KS")


def _same_endpoint_name(first: str, second: str) -> bool:
    left = " ".join(first.casefold().split())
    right = " ".join(second.casefold().split())
    return left == right or left.startswith(right) or right.startswith(left)


def resolve_wasapi_device(
    direction: str,
    reference_device=None,
    *,
    _sd=None,
) -> int | None:
    """Resolve the WASAPI counterpart of an input/output endpoint by name."""
    if _sd is None:
        import sounddevice as _sd

    host_index = _wasapi_host_index(_sd)
    if host_index is None:
        return None
    reference_index = _resolve_index(_sd, reference_device, direction)
    reference_name = ""
    if reference_index is not None:
        reference_name = str(_sd.query_devices(reference_index)["name"]).rstrip()

    candidates: list[int] = []
    for index, info in enumerate(_sd.query_devices()):
        if int(info["hostapi"]) != host_index:
            continue
        if int(info[_channel_key(direction)]) < 1:
            continue
        if reference_name and _same_endpoint_name(str(info["name"]), reference_name):
            candidates.append(index)
    if candidates:
        return candidates[0]

    try:
        key = f"default_{direction}_device"
        fallback = int(_sd.query_hostapis()[host_index].get(key, -1))
        return fallback if fallback >= 0 else None
    except Exception:
        return None


def _endpoint_tokens(value: str) -> set[str]:
    normalized = "".join(
        character.casefold() if character.isalnum() else " "
        for character in value
    )
    ignored = {"device", "driver", "windows", "primary", "sound"}
    return {
        token for token in normalized.split()
        if len(token) >= 3 and token not in ignored
    }


def resolve_wdm_ks_device(
    direction: str,
    reference_device=None,
    *,
    samplerate: int,
    _sd=None,
) -> int | None:
    """Find the native-rate WDM-KS endpoint matching the system default."""
    if _sd is None:
        import sounddevice as _sd

    host_index = _wdm_ks_host_index(_sd)
    if host_index is None:
        return None
    reference_index = _resolve_index(_sd, reference_device, direction)
    if reference_index is None:
        return None
    reference_name = str(_sd.query_devices(reference_index)["name"])
    reference_tokens = _endpoint_tokens(reference_name)

    ranked: list[tuple[int, int]] = []
    for index, info in enumerate(_sd.query_devices()):
        if int(info["hostapi"]) != host_index:
            continue
        if int(info[_channel_key(direction)]) < 1:
            continue
        native_rate = int(round(float(info["default_samplerate"])))
        if native_rate != int(samplerate):
            continue
        common_tokens = reference_tokens & _endpoint_tokens(str(info["name"]))
        if len(common_tokens) < 2:
            continue
        ranked.append((len(common_tokens), index))
    if not ranked:
        return None
    best_score = max(score for score, _ in ranked)
    best_indices = [index for score, index in ranked if score == best_score]
    if len(best_indices) == 1:
        return best_indices[0]
    try:
        key = f"default_{direction}_device"
        host_default = int(_sd.query_hostapis()[host_index].get(key, -1))
    except Exception:
        host_default = -1
    return host_default if host_default in best_indices else None


def _device_details(sd, index: int | None) -> tuple[str, str, int]:
    if index is None:
        return "defaut systeme", "default", 0
    info = sd.query_devices(index)
    host_index = int(info["hostapi"])
    host_name = str(sd.query_hostapis()[host_index]["name"])
    return (
        str(info["name"]),
        host_name,
        int(round(float(info["default_samplerate"]))),
    )


def _latency_ms(stream, direction: str) -> float:
    latency = stream.latency
    if isinstance(latency, (tuple, list)):
        latency = latency[0 if direction == "input" else 1]
    return float(latency) * 1000.0


def _open_stream(
    direction: str,
    *,
    preferred_device=None,
    samplerate: int,
    blocksize: int,
    channels: int,
    dtype: str,
    latency: str | float,
    callback: Callable,
    prefer_wasapi: bool = True,
    prefer_kernel_streaming: bool = True,
    _sd=None,
):
    if _sd is None:
        import sounddevice as _sd

    explicit = parse_device(preferred_device) is not None
    requested_index = _resolve_index(_sd, preferred_device, direction)
    wasapi_host_index = _wasapi_host_index(_sd)
    wdm_ks_host_index = _wdm_ks_host_index(_sd)

    def device_uses_host(device_index, host_index) -> bool:
        return bool(
            device_index is not None
            and host_index is not None
            and int(_sd.query_devices(device_index)["hostapi"]) == host_index
        )

    selected_is_wasapi = bool(
        device_uses_host(requested_index, wasapi_host_index)
    )
    selected_is_kernel_streaming = bool(
        device_uses_host(requested_index, wdm_ks_host_index)
    )

    candidates: list[_StreamCandidate] = []

    def append_candidate(candidate: _StreamCandidate) -> None:
        actual_index = (
            requested_index
            if candidate.device_index is None else candidate.device_index
        )
        for existing in candidates:
            existing_index = (
                requested_index
                if existing.device_index is None else existing.device_index
            )
            if existing_index == actual_index:
                return
        candidates.append(candidate)

    if explicit:
        requested_latency = (
            KERNEL_STREAMING_TARGET_LATENCY_S
            if selected_is_kernel_streaming and latency == "low"
            else latency
        )
        append_candidate(_StreamCandidate(
            requested_index,
            selected_is_wasapi,
            selected_is_kernel_streaming,
            requested_latency,
        ))
    else:
        kernel_streaming = (
            resolve_wdm_ks_device(
                direction,
                None,
                samplerate=samplerate,
                _sd=_sd,
            )
            if prefer_kernel_streaming else None
        )
        if kernel_streaming is not None:
            append_candidate(_StreamCandidate(
                kernel_streaming,
                False,
                True,
                KERNEL_STREAMING_TARGET_LATENCY_S,
            ))
        wasapi = (
            resolve_wasapi_device(direction, None, _sd=_sd)
            if prefer_wasapi else None
        )
        if wasapi is not None:
            append_candidate(_StreamCandidate(
                wasapi,
                True,
                False,
                latency,
            ))
        default_latency = (
            KERNEL_STREAMING_TARGET_LATENCY_S
            if selected_is_kernel_streaming and latency == "low"
            else latency
        )
        append_candidate(_StreamCandidate(
            None,
            selected_is_wasapi,
            selected_is_kernel_streaming,
            default_latency,
        ))

    stream_type = _sd.InputStream if direction == "input" else _sd.OutputStream

    def stream_kwargs(candidate: _StreamCandidate) -> dict[str, Any]:
        values: dict[str, Any] = {
            "device": candidate.device_index,
            "channels": int(channels),
            "samplerate": int(samplerate),
            "blocksize": int(blocksize),
            "dtype": dtype,
            "latency": candidate.requested_latency,
            "callback": callback,
        }
        if candidate.wasapi_shared:
            values["extra_settings"] = _sd.WasapiSettings(
                exclusive=False,
                auto_convert=True,
            )
        return values

    # PortAudio's advertised device defaults are misleading once a concrete
    # block size/rate is requested (on the reference machine MME advertises
    # 90 ms but negotiates 17.41 ms). Start a short callback probe with the
    # actual configuration, then try the lowest healthy negotiated latency.
    if not explicit and len(candidates) > 1:
        scored: list[tuple[float, int, _StreamCandidate]] = []
        for order, candidate in enumerate(candidates):
            probe = None
            callbacks = 0
            status_events = 0
            invalid_blocks = 0
            ready = threading.Event()

            def probe_callback(buffer, frames, _time_info, status):
                nonlocal callbacks, status_events, invalid_blocks
                callbacks += 1
                status_events += int(bool(status))
                invalid_blocks += int(int(frames) != int(blocksize))
                if direction == "output":
                    buffer.fill(0.0)
                if callbacks >= 16:
                    ready.set()

            try:
                probe_values = stream_kwargs(candidate)
                probe_values["callback"] = probe_callback
                probe = stream_type(**probe_values)
                probe.start()
                timeout_s = max(0.5, 32.0 * blocksize / float(samplerate))
                completed = ready.wait(timeout_s)
                score = (
                    _latency_ms(probe, direction)
                    if completed and status_events == 0 and invalid_blocks == 0
                    else float("inf")
                )
            except Exception:
                score = float("inf")
            finally:
                if probe is not None:
                    try:
                        probe.stop()
                    except Exception:
                        pass
                    try:
                        probe.close()
                    except Exception:
                        pass
            scored.append((score, order, candidate))
        scored.sort(key=lambda item: (item[0], item[1]))
        candidates = [item[2] for item in scored]
        summary = []
        for score, _, candidate in scored:
            actual_index = (
                _default_index(_sd, direction)
                if candidate.device_index is None else candidate.device_index
            )
            _, host_name, _ = _device_details(_sd, actual_index)
            value = "indisponible" if score == float("inf") else f"{score:.2f}ms"
            mode = (
                " low-latency" if candidate.kernel_streaming
                else " shared" if candidate.wasapi_shared else ""
            )
            summary.append(f"{host_name}{mode}={value}")
        _diagnostic(f"Selection {direction} automatique: " + ", ".join(summary))

    last_error: Exception | None = None
    for candidate in candidates:
        kwargs = stream_kwargs(candidate)
        stream = None
        try:
            stream = stream_type(**kwargs)
            stream.start()
            actual_index = candidate.device_index
            if actual_index is None:
                actual_index = _default_index(_sd, direction)
            name, host_name, native_rate = _device_details(_sd, actual_index)
            requested_latency_ms = (
                float(candidate.requested_latency) * 1000.0
                if isinstance(candidate.requested_latency, (int, float))
                else None
            )
            info = AudioStreamInfo(
                direction=direction,
                device_index=actual_index,
                device_name=name,
                host_api=host_name,
                native_sample_rate=native_rate,
                target_sample_rate=int(samplerate),
                block_size=int(blocksize),
                latency_ms=_latency_ms(stream, direction),
                wasapi_shared=candidate.wasapi_shared,
                automatic_selection=not explicit,
                kernel_streaming=candidate.kernel_streaming,
                requested_latency_ms=requested_latency_ms,
            )
            conversion = (
                f", conversion {native_rate}->{samplerate} Hz"
                if info.conversion_active else ""
            )
            label = "Entree" if direction == "input" else "Sortie"
            mode = (
                " low-latency" if candidate.kernel_streaming
                else " shared" if candidate.wasapi_shared else ""
            )
            _diagnostic(
                f"{label} audio: {host_name}{mode}, {name} [{actual_index}]"
                f"{conversion}, latence={info.latency_ms:.2f}ms"
            )
            return stream, info
        except Exception as exc:
            last_error = exc
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            if explicit:
                raise
            if candidate.device_index is not None:
                _, host_name, _ = _device_details(
                    _sd, candidate.device_index
                )
                _diagnostic(
                    f"{direction.capitalize()} {host_name} indisponible ({exc}); "
                    "essai du candidat suivant."
                )

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Aucun peripherique audio {direction} disponible.")


def open_input_stream(**kwargs):
    # Automatic WDM-KS input is deliberately disabled.  On common Realtek
    # hardware the matching kernel pin bypasses the endpoint microphone
    # gain/boost: it opens cleanly at very low latency but delivers an almost
    # silent signal.  An explicitly selected WDM-KS device is still honoured,
    # while automatic input selection compares WASAPI with the system device.
    # Output WDM-KS remains enabled because it does not have this capture-gain
    # ambiguity.
    kwargs.setdefault("prefer_kernel_streaming", False)
    return _open_stream("input", **kwargs)


def open_output_stream(**kwargs):
    return _open_stream("output", **kwargs)


@contextmanager
def managed_input_stream(**kwargs):
    stream, info = open_input_stream(**kwargs)
    try:
        yield stream, info
    finally:
        try:
            stream.stop()
        finally:
            stream.close()


def audio_devices(direction: str, *, _sd=None) -> list[dict[str, Any]]:
    if _sd is None:
        import sounddevice as _sd
    rows = []
    default_index = _default_index(_sd, direction)
    for index, info in enumerate(_sd.query_devices()):
        if int(info[_channel_key(direction)]) < 1:
            continue
        host_name = str(_sd.query_hostapis()[int(info["hostapi"])]["name"])
        latency_key = f"default_low_{direction}_latency"
        rows.append({
            "index": index,
            "name": str(info["name"]),
            "host_api": host_name,
            "native_sample_rate": int(round(float(info["default_samplerate"]))),
            "low_latency_ms": float(info.get(latency_key, 0.0)) * 1000.0,
            "default": index == default_index,
        })
    return rows
