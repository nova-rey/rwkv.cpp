import sys

import torch

sys.path.insert(0, ".")

from logits_processor import (  # noqa: E402
    StopLogitsProcessor,
    canonical_structural_token_replacements,
    semantic_bar_count_after_fill,
    semantic_constraint_token_ids,
)
from rwkv_cpp.cpp_model import apply_repetition_penalty, semantic_bar_count  # noqa: E402
from rwkv_cpp.cpp_model import CppModelConfig  # noqa: E402


class FakeTokenizer:
    vocab_size = 11
    vocab = {
        "Bar_None": 9,
        "FillBar_Start": 5,
        "Infill_Track": 4,
        "FillBar_End": 6,
        "PAD_None": 0,
    }

    decoded = {
        0: ("PAD_None",),
        4: ("Infill_Track",),
        5: ("FillBar_Start",),
        6: ("FillBar_End",),
        7: ("Track_Start",),
        8: ("Track_End",),
        9: ("Bar_None",),
        # Both IDs are equivalent compound BPE encodings in the bundled tokenizer.
        1: ("Bar_None", "TimeSig_4/4"),
        2: ("Bar_None", "TimeSig_4/4"),
        10: ("Bar_None", "TimeSig_3/4"),
    }

    def decode_token_ids(self, sequence):
        sequence.tokens = list(self.decoded.get(sequence.ids[0], ()))

    def encode_token_ids(self, sequence):
        sequence.ids = [1 if sequence.tokens == ["Bar_None", "TimeSig_4/4"] else 10]


def test_structural_masks_are_tokenizer_semantic():
    constraints = semantic_constraint_token_ids(FakeTokenizer(), 7, 8)
    assert constraints["DISALLOW_COMPOUND_BAR_TIME"] == {2}
    assert 1 not in constraints["DISALLOW_COMPOUND_BAR_TIME"]
    assert 10 not in constraints["DISALLOW_COMPOUND_BAR_TIME"]
    assert constraints["DISALLOW_TRACK_START"] == {7}
    assert constraints["DISALLOW_TRACK_END"] == {8}
    assert constraints["DISALLOW_EMPTY_BPE"] == {3}

    assert canonical_structural_token_replacements(FakeTokenizer()) == {2: 1}
    assert constraints["DISALLOW_EMPTY_BPE"] != {663}


def test_repetition_penalty_sign_behavior():
    scores = torch.tensor([[2.0, -2.0, 0.5, -0.5]])
    apply_repetition_penalty(scores, {0, 1, 2, 3}, 2.0)
    assert torch.allclose(scores, torch.tensor([[1.0, -4.0, 0.25, -1.0]]))


def test_stop_processor_accepts_python_lists_and_applies_semantic_masks():
    processor = StopLogitsProcessor(9, 6, 7, 8, FakeTokenizer())
    processor.infill_type = "bar"
    processor.n_bars_to_infill = 1
    processor.n_attribute_controls = 0

    # The production generator passes a Python list here, not a rank-2 tensor.
    scores = torch.zeros(1, FakeTokenizer.vocab_size)
    masked = processor([5, 9], scores)

    assert torch.isneginf(masked[0, 6])  # FillBar_End
    assert torch.isneginf(masked[0, 7])  # Track_Start
    assert torch.isneginf(masked[0, 8])  # Track_End
    assert not torch.isneginf(masked[0, 1])  # canonical compound encoding remains valid
    assert torch.isneginf(masked[0, 2])  # noncanonical equivalent is constrained
    assert torch.isneginf(masked[0, 3])  # invalid/empty BPE token

    finished = processor([5, 1, 9, 9], scores.clone())
    assert finished[0, 6].item() == 0.0  # completed infill must force FillBar_End
    assert torch.isneginf(finished[0, 1])


def test_bar_count_handles_compound_and_split_bar_encodings():
    tokenizer = FakeTokenizer()
    tokenizer.vocab_size = 16
    tokenizer.decoded.update({
        11: ("ACBarNoteDensity_1",),
        12: ("TimeSig_4/4",),
        13: ("Bar_None",),
        14: ("Bar_None", "TimeSig_3/4"),
        15: (),
    })
    # FillBar_Start, one compound prompt bar, one AC packet, then two
    # generated boundaries using different valid BPE layouts.
    sequence = [5, 1, 11, 13, 12, 14, 6]
    assert semantic_bar_count_after_fill(
        sequence, tokenizer, n_attribute_controls=1, infill_type="bar"
    ) == 2


def test_runtime_bar_index_counts_compound_bpe_boundaries():
    tokenizer = FakeTokenizer()
    tokenizer.vocab_size = 16
    tokenizer.decoded.update({
        11: ("TimeSig_4/4",),
        12: ("Bar_None",),
        13: ("Bar_None", "TimeSig_4/4"),
    })
    assert semantic_bar_count(tokenizer, [5, 13, 12, 11]) == 2


def test_stop_processor_completes_at_requested_bar_count():
    tokenizer = FakeTokenizer()
    tokenizer.vocab_size = 16
    tokenizer.decoded.update({11: ("ACBarNoteDensity_1",), 13: ("Bar_None",)})
    processor = StopLogitsProcessor(9, 6, 7, 8, tokenizer)
    processor.infill_type = "bar"
    processor.n_bars_to_infill = 1
    processor.n_attribute_controls = 1
    scores = torch.zeros(1, tokenizer.vocab_size)
    finished = processor([5, 1, 11, 13], scores)
    assert finished[0, tokenizer.vocab["FillBar_End"]].item() == 0.0


def test_stop_processor_counts_prompt_scaffold_as_first_target_bar():
    tokenizer = FakeTokenizer()
    tokenizer.vocab_size = 16
    tokenizer.decoded.update({11: ("ACBarNoteDensity_1",), 13: ("Bar_None",)})
    processor = StopLogitsProcessor(9, 6, 7, 8, tokenizer)
    processor.infill_type = "bar"
    processor.n_bars_to_infill = 2
    processor.n_attribute_controls = 1
    scores = torch.zeros(1, tokenizer.vocab_size)
    # The scaffold opens bar 0; one generated boundary starts bar 1 and
    # therefore completes a two-bar request.
    finished = processor([5, 1, 11, 13], scores)
    assert finished[0, tokenizer.vocab["FillBar_End"]].item() == 0.0


def test_cpp_model_config_accepts_thread_count(monkeypatch):
    monkeypatch.setenv("MIDI_RWKV_THREADS", "4")
    config = CppModelConfig("model.bin")
    assert config.thread_count == 4
