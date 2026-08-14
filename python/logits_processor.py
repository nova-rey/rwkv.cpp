"""Definition of logits processor used for generation."""

import time

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

        n_bar_none = 0
        if fill_start_idx + self.n_attribute_controls + 1 < len(input_ids):
            generated_tokens.ids = input_ids[
                fill_start_idx + self.n_attribute_controls + 1 :
            ].tolist()
            self.tokenizer.decode_token_ids(generated_tokens)

            n_bar_none = len(
                np.where(
                    np.array(generated_tokens.ids) == self.tokenizer.vocab["Bar_None"]
                )[0]
            )

        penalty = float("inf")

        completed = n_bar_none > self.n_bars_to_infill

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
