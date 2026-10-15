"""Single discoverable entrypoint for the legacy Live and offline-protected mode.

The legacy Live algorithms are delegated verbatim, not rewritten or silently
replaced by the noncausal 5-second S422 model. 'live-midi' is an event bridge
fed by EXTERNAL precomputed live K and pitches, not raw-WAV-to-S80 inference.
"""
from __future__ import annotations
import argparse
import json

from causal_note.live_k_midi_bridge import load_frozen_profile
from scripts.stream_k_midi_live import DEFAULT_PROFILE


def main(argv=None):
    parser=argparse.ArgumentParser(allow_abbrev=False,
        description="note protected runtimes: legacy causal live or optional K-first MIDI event stream"
    )
    parser.add_argument(
        "--mode",required=True,
        choices=("live-boundaries","live-v8","live-midi","native-notes","recent-notes","strict-live","status"),
        help="existing Live modes are delegated without changes; offline best is pinned, not live"
    )
    parser.add_argument("--status-profile",default=str(DEFAULT_PROFILE),
        help="versioned protection manifest for status (event sidecar has its own flag)")
    ns, forwarded=parser.parse_known_args(argv)
    if ns.mode=="status":
        if forwarded:
            parser.error("status does not accept stream or inference flags")
        from pathlib import Path
        record=load_frozen_profile(Path(ns.status_profile))
        print(json.dumps({
            "status":"ORIGINAL_LIVE_AND_BEST_OFFLINE_REFERENCES_PROTECTED",
            "legacy_live_untouched":record["legacy_live_untouched"],
            "original_live_cli":["scripts.detect_live","scripts.detect_v8_live"],
            "S80_ExactK_reference":record["original_S80_ExactK_reference"],
            "S402_MIDI_pitch_reference":record["original_active_MIDI_pitch_reference"],
            "S408_Offset_reference":record["original_active_Offset_reference"],
            "best_offline_native":record["best_offline_native_challenger"],
            "offline_exactK_global":record["offline_development_exact_K_global"]["accuracy"],
            "offline_exactK_poly":record["offline_development_exact_K_poly"]["accuracy"],
            "offline_full_note_F1":record["offline_native_full_note"]["full_note_F1"],
            "live_accuracy_certified":False,
            "S80_causal_live_frontend_currently_integrated":False,
            "native_pitch_stream":"scripts.live_native_notes (requires upstream S80 events/K)",
            "recent_native_stream":"scripts.live_recent_notes (S569/S578/S585, external V88/BP required)",
            "autonomous_native_stream":"scripts.live_strict_notes (experimental S613 causal heads, no sidecars)",
            "S422_noncausal_lookahead_ms":5000,
        },indent=2,sort_keys=True))
        return
    if ns.mode=="live-boundaries":
        # SAME legacy CLI and detector; all arguments forwarded unchanged.
        from scripts import detect_live
        return detect_live.main(forwarded)
    if ns.mode=="live-v8":
        from scripts import detect_v8_live
        return detect_v8_live.main(forwarded)
    if ns.mode=="live-midi":
        from scripts import stream_k_midi_live
        return stream_k_midi_live.main(forwarded)
    if ns.mode=="native-notes":
        from scripts import live_native_notes
        return live_native_notes.main(forwarded)
    if ns.mode=="recent-notes":
        from scripts import live_recent_notes
        return live_recent_notes.main(forwarded)
    if ns.mode=="strict-live":
        from scripts import live_strict_notes
        return live_strict_notes.main(forwarded)
    raise AssertionError("unsupported mode")


if __name__=="__main__":
    main()
