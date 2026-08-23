from __future__ import annotations

import numpy as np

from language_analysis.study import holm, root_signals, train_indices


def test_training_indices_are_deterministic_and_exclude_held_out_rows() -> None:
    first = train_indices(101, 42)
    second = train_indices(101, 42)
    assert np.array_equal(first, second)
    assert len(first) == round(101 * 0.7)
    assert len(np.unique(first)) == len(first)
    assert set(first).isdisjoint(set(range(101)) - set(first))


def test_exact_root_signals_identify_unary_and_parity_structure() -> None:
    x = np.asarray(
        [[(mask >> bit) & 1 for bit in range(4)] for mask in range(16)],
        dtype=np.uint8,
    )
    unary = root_signals(x, x[:, 0])
    assert unary["unary_signal"] == 1.0

    parity_labels = x[:, 0] ^ x[:, 1] ^ x[:, 2]
    parity = root_signals(x, parity_labels)
    assert parity["affine_signal"] == 1.0
    assert parity["unary_signal"] == 0.0


def test_holm_correction_is_monotone_in_sorted_p_value_order() -> None:
    raw = [0.04, 0.001, 0.02, 0.5]
    adjusted = holm(raw)
    ordered = sorted(range(len(raw)), key=raw.__getitem__)
    ordered_adjusted = [adjusted[index] for index in ordered]
    assert ordered_adjusted == sorted(ordered_adjusted)
    assert all(corrected >= original for corrected, original in zip(adjusted, raw))
