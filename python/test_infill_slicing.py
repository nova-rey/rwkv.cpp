import sys

sys.path.insert(0, ".")

from inference import (  # noqa: E402
    _ensure_generated_bar_prefix,
    _extract_generated_token_ids,
    _replace_decoded_token_span,
    _trim_generated_bar_stream,
)


class FakeTokenizer:
    vocab = {
        "Bar_None": 9,
        "TimeSig_4/4": 14,
        "FillBar_Start": 5,
        "Infill_Track": 4,
        "FillBar_End": 6,
        "Track_End": 8,
    }

    decoded = {
        1: ("Bar_None", "TimeSig_4/4"),
        2: ("Bar_None",),
        3: ("TimeSig_4/4",),
        5: ("FillBar_Start",),
        6: ("FillBar_End",),
        10: ("ACBarNoteDensity_1",),
        11: ("ACBarNoteDensity_8",),
        12: ("Pitch_60",),
        13: ("Bar_None",),
        14: ("TimeSig_4/4",),
        15: ("Track_End",),
    }

    def decode_token_ids(self, sequence):
        sequence.tokens = list(self.decoded.get(sequence.ids[0], ()))

    def _ids_to_tokens(self, ids):
        return [name for token_id in ids for name in self.decoded.get(token_id, ())]


def subset(controls):
    return (0, 1, [controls], "bar")


def test_extract_preserves_final_bar_and_stops_at_semantic_eos():
    output = [99, 5, 1, 10, 13, 2, 6, 12]
    assert _extract_generated_token_ids(output, FakeTokenizer(), subset(["ACBarNoteDensity_1"])) == [13, 2]


def test_extract_handles_split_scaffold_tokens():
    output = [5, 2, 3, 10, 13, 6]
    assert _extract_generated_token_ids(output, FakeTokenizer(), subset(["ACBarNoteDensity_1"])) == [13]


def test_extract_uses_last_fill_marker_and_does_not_use_last_token_as_eos():
    output = [5, 1, 10, 13, 6, 5, 1, 10, 13, 12, 6, 12]
    assert _extract_generated_token_ids(output, FakeTokenizer(), subset(["ACBarNoteDensity_1"])) == [13, 12]


def test_compound_first_bar_is_not_prefixed_twice():
    tokenizer = FakeTokenizer()
    assert _ensure_generated_bar_prefix([1, 12], tokenizer) == [1, 12]
    assert _ensure_generated_bar_prefix([12], tokenizer) == [9, 14, 12]


def test_decoded_span_replacement_preserves_semantic_context():
    class Sequence:
        are_ids_encoded = False

        def __init__(self, ids, tokens):
            self.ids = list(ids)
            self.tokens = list(tokens)

    target = Sequence(
        ["Track_Start", "Bar_0", "Pitch_A", "Bar_1", "Pitch_B", "Bar_2", "Track_End"],
        ["Track_Start", "Bar_None", "Pitch_60", "Bar_None", "Pitch_61", "Bar_None", "Track_End"],
    )
    generated = Sequence(
        ["Bar_None", "Pitch_X", "Bar_None", "Pitch_Y"],
        ["Bar_None", "Pitch_70", "Bar_None", "Pitch_71"],
    )
    _replace_decoded_token_span(target, 3, 5, generated)
    assert target.tokens == [
        "Track_Start", "Bar_None", "Pitch_60", "Bar_None", "Pitch_70",
        "Bar_None", "Pitch_71", "Bar_None", "Track_End",
    ]
    assert target.ids == [
        "Track_Start", "Bar_0", "Pitch_A", "Bar_None", "Pitch_X",
        "Bar_None", "Pitch_Y", "Bar_2", "Track_End",
    ]


def test_terminal_bar_boundary_is_not_inserted_into_requested_span():
    ids = [9, 14, 20, 9, 14, 21]
    tokens = ["Bar_None", "TimeSig_4/4", "Pitch_60", "Bar_None", "TimeSig_4/4", "Pitch_61"]
    trimmed_ids, trimmed_tokens = _trim_generated_bar_stream(ids, tokens, 1)
    assert trimmed_ids == [9, 14, 20]
    assert trimmed_tokens == ["Bar_None", "TimeSig_4/4", "Pitch_60"]


def test_two_bar_stream_keeps_two_bar_starts_and_drops_only_third():
    ids = [9, 14, 20, 9, 14, 21, 9, 14, 22]
    tokens = ["Bar_None", "TimeSig_4/4", "Pitch_60", "Bar_None", "TimeSig_4/4", "Pitch_61", "Bar_None", "TimeSig_4/4", "Pitch_62"]
    _, trimmed_tokens = _trim_generated_bar_stream(ids, tokens, 2)
    assert trimmed_tokens == tokens[:6]
