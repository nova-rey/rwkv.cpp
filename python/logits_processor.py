"""Definition of logits processor used for generation."""

import time
import json
import os

import miditok
import numpy as np
import torch
from miditok import TokSequence
from transformers import LogitsProcessor


def _decoded_token_names(tokenizer, token_id: int) -> tuple[str, ...]:
    """Decode one bundled BPE id into its underlying MMM token names."""
    sequence = TokSequence(ids=[int(token_id)], are_ids_encoded=True)
    tokenizer.decode_token_ids(sequence)
    return tuple(sequence.tokens or ())


def decoded_token_names(tokenizer, token_id: int) -> tuple[str, ...]:
    """Public tokenizer-semantic decode helper for generation code and tests."""
    return _decoded_token_names(tokenizer, token_id)


def semantic_bar_count_after_fill(
    input_ids,
    tokenizer,
    n_attribute_controls: int,
    infill_type: str = "bar",
) -> int:
    """Count generated bar boundaries without depending on one BPE encoding.

    ``MMM`` can encode ``Bar_None`` standalone, as a compound
    ``Bar_None + TimeSig`` token, or as a split sequence.  Counting only the
    canonical ``Bar_None`` id therefore under-counts valid generated output.
    The prompt's initial bar/time-signature scaffold and first attribute packet
    are skipped semantically; only boundaries after that scaffold are counted.
    """
    ids = [int(token_id) for token_id in input_ids]
    marker_name = "FillBar_Start" if infill_type == "bar" else "Infill_Track"
    marker_id = tokenizer.vocab[marker_name]
    marker_positions = [i for i, token_id in enumerate(ids) if token_id == marker_id]
    if not marker_positions:
        raise ValueError(f"{marker_name} marker is absent from the generation sequence")

    cursor = marker_positions[-1] + 1
    if infill_type == "bar":
        structural_seen = set()
        while cursor < len(ids) and not {"Bar_None", "TimeSig_4/4"}.issubset(structural_seen):
            structural_seen.update(decoded_token_names(tokenizer, ids[cursor]))
            cursor += 1
    else:
        # Track infill has a program token before any optional controls, not a
        # bar/time-signature scaffold.
        if cursor < len(ids):
            cursor += 1

    controls_seen = 0
    while cursor < len(ids) and controls_seen < n_attribute_controls:
        controls_seen += sum(
            name.startswith("AC")
            for name in decoded_token_names(tokenizer, ids[cursor])
        )
        cursor += 1

    return sum(
        name == "Bar_None"
        for token_id in ids[cursor:]
        for name in decoded_token_names(tokenizer, token_id)
    )


def _semantic_token_ids(tokenizer, names: tuple[str, ...]) -> set[int]:
    """Return ids whose single-token decode contains all requested names."""
    wanted = set(names)
    return {
        token_id
        for token_id in range(tokenizer.vocab_size)
        if wanted.issubset(_decoded_token_names(tokenizer, token_id))
    }


def _bar_time_token_ids(tokenizer) -> set[int]:
    """Return noncanonical 4/4 BPE ids for an immediately repeated bar boundary.

    The released model is trained on the bundled 4/4 grammar. Other compound
    ``Bar_None + TimeSig_*`` tokens are legitimate time-signature structures and
    must not be masked merely because they share the ``Bar_None`` prefix.
    Multiple BPE ids can encode the same canonical MMM sequence (665 and 797 in
    the bundled tokenizer). Preserve the canonical encoding produced by the
    tokenizer and constrain only alternate encodings; masking both would remove
    valid support from the released model.
    """
    equivalent_ids = {
        token_id
        for token_id in range(tokenizer.vocab_size)
        if (
            _decoded_token_names(tokenizer, token_id)
            == ("Bar_None", "TimeSig_4/4")
        )
    }
    canonical = TokSequence(tokens=["Bar_None", "TimeSig_4/4"], ids=[], are_ids_encoded=False)
    try:
        tokenizer.encode_token_ids(canonical)
    except Exception:
        return set()
    return equivalent_ids - set(canonical.ids)


def canonical_structural_token_replacements(tokenizer) -> dict[int, int]:
    """Map noncanonical 4/4 compound encodings to the tokenizer's canonical ID."""
    canonical = TokSequence(tokens=["Bar_None", "TimeSig_4/4"], ids=[], are_ids_encoded=False)
    try:
        tokenizer.encode_token_ids(canonical)
    except Exception:
        return {}
    if len(canonical.ids) != 1:
        return {}
    canonical_id = int(canonical.ids[0])
    return {token_id: canonical_id for token_id in _bar_time_token_ids(tokenizer)}


def _empty_decode_token_ids(tokenizer) -> set[int]:
    """Return bundled BPE ids with no valid MMM decode (e.g. token 663)."""
    return {
        token_id
        for token_id in range(tokenizer.vocab_size)
        if not _decoded_token_names(tokenizer, token_id)
    }


def semantic_constraint_token_ids(tokenizer, track_start_token_id: int, track_end_token_id: int) -> dict[str, set[int]]:
    """Build tokenizer-aware structural token constraint sets."""
    return {
        "DISALLOW_TRACK_START": {int(track_start_token_id)},
        "DISALLOW_TRACK_END": {int(track_end_token_id)},
        "DISALLOW_INFIL_TRACK": _semantic_token_ids(tokenizer, ("Infill_Track",)),
        "DISALLOW_FILLBAR_END": _semantic_token_ids(tokenizer, ("FillBar_End",)),
        "DISALLOW_PAD": _semantic_token_ids(tokenizer, ("PAD_None",)),
        "DISALLOW_COMPOUND_BAR_TIME": _bar_time_token_ids(tokenizer),
        "DISALLOW_EMPTY_BPE": _empty_decode_token_ids(tokenizer),
    }


class StopLogitsProcessor(LogitsProcessor):
    """

    Custom ``transformers.LogitsProcessor`` implementation.

    Allows stopping generation when enough content to infill bars is generated.

    :param bar_start_token_id: ID of the token indicating the start of a bar.
    :param n_bars_to_infill: number of bars to be infilled in this generation step.
    :param eos_token_id: ID of the EOS (end of sequence) token. If the number
    of bars reaches `max_bars`, the EOS token will be forced to stop generation.

    """

    n_bars_to_infill: int = 0  # This should change at every generation
    # step as we may need to infill a different number of bars at each step
    n_attribute_controls: int = 0  # Number of attribute controls to skip
    # when decoding using BPE
    infill_type: str = None

    def __init__(
        self,
        bar_start_token_id: int,
        eos_token_id: int,
        track_start_token_id: int,
        track_end_token_id: int,
        tokenizer: miditok.MusicTokenizer
    ) -> None:
        self.bar_start_token_id = bar_start_token_id
        self.eos_token_id = eos_token_id
        self.track_start_token_id = track_start_token_id
        self.track_end_token_id = track_end_token_id
        self.tokenizer = tokenizer
        self.total_time = 0
        self.constraint_token_ids = semantic_constraint_token_ids(
            tokenizer, track_start_token_id, track_end_token_id
        )

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        To handle proper infilling generation content.

        Assert that the right number of bars are generated
        for infilling and that the generation is stopped when all
        the bars are generated.

        :param input_ids: ids of the input sequence of tokens
        :param scores: pre-softmax sampling probabilities
        :return: output tokens prediction scores
        """
        start_time = time.time()

        generated_tokens = TokSequence(are_ids_encoded=True)

        input_ids = np.asarray(input_ids, dtype=np.int64).reshape(-1)

        if self.infill_type == "bar":
            fill_positions = np.flatnonzero(
                input_ids == self.tokenizer.vocab["FillBar_Start"]
            )
        elif self.infill_type == "track":
            fill_positions = np.flatnonzero(
                input_ids == self.tokenizer.vocab["Infill_Track"]
            )
        else:
            raise ValueError(f"Unsupported infill type: {self.infill_type!r}")

        if len(fill_positions) == 0:
            raise ValueError("Infill marker is absent from the generation sequence")
        fill_start_idx = int(fill_positions[-1])

        n_bar_none = semantic_bar_count_after_fill(
            input_ids,
            self.tokenizer,
            self.n_attribute_controls,
            self.infill_type,
        )

        penalty = float("inf")

        # The prompt already opens the first target bar with the
        # ``Bar_None + TimeSig`` scaffold.  A generated ``Bar_None`` therefore
        # starts the next target bar; for N requested bars we need N-1 such
        # transitions, while still requiring one transition for a one-bar
        # infill so that the model emits a complete bar boundary before EOS.
        required_boundaries = max(1, int(self.n_bars_to_infill) - 1)
        completed = n_bar_none >= required_boundaries

        trace_path = os.getenv("MIDI_RWKV_STOP_TRACE")
        if trace_path:
            with open(trace_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "sequence_length": len(input_ids),
                    "n_bar_none": int(n_bar_none),
                    "requested": int(self.n_bars_to_infill),
                    "required_boundaries": int(required_boundaries),
                    "completed": bool(completed),
                    "tail": [
                        list(decoded_token_names(self.tokenizer, int(token_id)))
                        for token_id in input_ids[-32:]
                    ],
                }, default=int) + "\n")

        # Don't sample an EOS token until all bars are generated. Completion
        # handling is applied after semantic masks below so the EOS token is not
        # immediately masked by DISALLOW_FILLBAR_END.
        if not completed:
            scores[:, self.eos_token_id] = -penalty

        end_time = time.time()
        self.total_time += end_time - start_time

        for rule, token_ids in self.constraint_token_ids.items():
            for token_id in token_ids:
                if token_id < scores.shape[-1]:
                    scores[:, token_id] = -penalty

        if completed:
            scores[:, :] = -penalty
            scores[:, self.eos_token_id] = 0.0

        return scores
