import math

import pytest

from accord.structure import (compare, dotbracket_to_pairs, is_dotbracket, mcc,
                              pairing_disagreement)


def test_pairs_include_pseudoknot_brackets():
    assert dotbracket_to_pairs("((..[[..))..]]") == {(0, 9), (1, 8), (4, 13), (5, 12)}


def test_is_dotbracket():
    assert is_dotbracket("((..))")
    assert not is_dotbracket("((..)")
    assert not is_dotbracket("((xx))")
    assert not is_dotbracket("")


def test_identical_structures_score_one():
    c = compare("((..))", "((..))")
    assert c.mcc == 1.0 and c.exact_match


def test_unpaired_structures_score_one():
    assert mcc("......", "......") == 1.0


def test_hand_computed_value():
    # reference pairs {(0,5)}, prediction pairs {(0,5),(1,4)}: tp=1, fp=1, fn=0, tn=13
    c = compare("((..))", "(....)")
    assert (c.tp, c.fp, c.fn, c.tn) == (1, 1, 0, 13)
    assert c.mcc == pytest.approx(13 / math.sqrt(2 * 1 * 14 * 13))


def test_wrong_partners_are_not_credited():
    # Every position is paired in both structures, but no pair is shared.
    # A per-position convention would score this as perfect; base-pair MCC must not.
    assert mcc("()()", "(())") < 0.5


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        compare("((..))", "(...)")


def test_pairing_disagreement_bounds():
    assert pairing_disagreement(["((..))", "((..))", "((..))"]) == 0.0
    assert pairing_disagreement(["((..))", "......"]) == pytest.approx(2 / 3 * 0.5)
    with pytest.raises(ValueError):
        pairing_disagreement(["((..))"])
