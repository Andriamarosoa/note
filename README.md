# OFFSET only — preserve original S708/H0 score

This Git branch contains **only the offset historical scorer and signed replay**. All five other axis implementations and experiments have been removed from the Git tree, not from their own reference branches.

Reference on 80 signed FIT80 GuitarSet tracks: **69.8862289642% conditional offset accuracy**. Original S708 neural inference historically ran on 1024-sample decisions. This branch only revalidates the exact original signed H0 events against the original ground truth, checking event-level raw counters for **80/80** tracks and matching the S704 evaluation protocol. It is **NOT** a stand-alone PCM256 neural detector or evidence that Onset/Offset now run every 256 samples. No GT labels are used to change original events.

Only score output and validation; no public MIDI, no model edits, no added hold/lookahead, no false promotional claims.
