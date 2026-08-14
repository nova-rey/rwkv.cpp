import sys

sys.path.insert(0, ".")

from inference import _ensure_generated_bar_prefix, _extract_generated_token_ids  # noqa: E402


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
