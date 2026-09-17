"""Property-based tests for bharat_units.numerals.

These run BEFORE any C1 item is generated, and nothing downstream is allowed to start until this
file is green.
"""
from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bharat_units.errors import ParseError
from bharat_units.numerals import (
    format_indian_grouping,
    from_words,
    parse_indian_number,
    scale_factor,
    to_scale,
    to_words,
)

# ---------------------------------------------------------------------------
# Digit grouping: round-trip and the "3-then-2s" invariant
# ---------------------------------------------------------------------------

@given(st.integers(min_value=0, max_value=10**15))
@settings(max_examples=500)
def test_format_parse_round_trip(n):
    assert parse_indian_number(format_indian_grouping(n)) == Decimal(n)


@given(st.integers(min_value=0, max_value=10**15))
@settings(max_examples=500)
def test_grouping_invariant(n):
    """The RIGHTMOST group is always exactly 3 digits (when there's more than one group);
    the leftmost group is 1-2 digits; any middle groups are exactly 2 digits. A single-group
    number (<=999) is 1-3 digits with no comma at all."""
    grouped = format_indian_grouping(n)
    groups = grouped.split(",")
    if len(groups) == 1:
        assert 1 <= len(groups[0]) <= 3
    else:
        assert len(groups[-1]) == 3
        assert 1 <= len(groups[0]) <= 2
        for g in groups[1:-1]:
            assert len(g) == 2


@given(st.integers(min_value=-10**12, max_value=10**12))
@settings(max_examples=300)
def test_format_handles_sign(n):
    grouped = format_indian_grouping(n)
    assert grouped.startswith("-") == (n < 0)
    assert parse_indian_number(grouped) == Decimal(n)


# ---------------------------------------------------------------------------
# Scale conversion: round-trip and transitivity
# ---------------------------------------------------------------------------

SCALE_UNITS = ["thousand", "lakh", "crore", "arab", "million", "billion", "trillion"]


@given(
    st.decimals(min_value="0.01", max_value="1000000", allow_nan=False, allow_infinity=False, places=4),
    st.sampled_from(SCALE_UNITS),
    st.sampled_from(SCALE_UNITS),
)
@settings(max_examples=300)
def test_to_scale_round_trip(value, unit_a, unit_b):
    converted = to_scale(value, unit_a, unit_b)
    back = to_scale(converted, unit_b, unit_a)
    assert abs(back - value) < Decimal("1e-10")


@given(
    st.decimals(min_value="0.01", max_value="1000", allow_nan=False, allow_infinity=False, places=4),
)
@settings(max_examples=200)
def test_to_scale_transitive_lakh_crore_arab(value):
    direct = to_scale(value, "lakh", "arab")
    via_crore = to_scale(to_scale(value, "lakh", "crore"), "crore", "arab")
    assert abs(direct - via_crore) < Decimal("1e-10")


def test_scale_factor_lakh_crore_compound_equals_trillion():
    """The 'X lakh crore' idiom must equal 'X trillion' - the benchmark's own headline example."""
    assert scale_factor("lakh", "crore") == scale_factor("trillion")


# ---------------------------------------------------------------------------
# Word form: round-trip
# ---------------------------------------------------------------------------

@given(st.integers(min_value=1, max_value=99_99_99_999))  # up to just under 100 crore (arab boundary)
@settings(max_examples=500)
def test_words_round_trip(n):
    assert from_words(to_words(n)) == n


def test_words_zero():
    assert to_words(0) == "zero"
    assert from_words("zero") == 0


# ---------------------------------------------------------------------------
# Adversarial corpus: real-world strings, parse-or-explicitly-fail, never a silent misparse
# ---------------------------------------------------------------------------

ADVERSARIAL_PARSE_CASES = [
    ("₹1.2 lakh crore", Decimal("1.2") * 10**12),
    ("Rs 4,500 crore", Decimal("4500") * 10**7),
    ("12.5 cr", Decimal("12.5") * 10**7),
    ("sava crore", Decimal("1.25") * 10**7),
    ("Rs. 5 cr 40 lakh", Decimal(5) * 10**7 + Decimal(40) * 10**5),
    ("₹1,20,000", Decimal("120000")),
    ("45 lakhs", Decimal(45) * 10**5),
    ("dedh lakh", Decimal("1.5") * 10**5),
    ("paune lakh", Decimal("0.75") * 10**5),
    ("one and a half crore", Decimal("1.5") * 10**7),
]


@pytest.mark.parametrize("text,expected", ADVERSARIAL_PARSE_CASES)
def test_adversarial_corpus_parses_correctly(text, expected):
    assert parse_indian_number(text) == expected


def test_bare_english_fraction_phrase_parses_as_a_plain_number():
    """'one and a half' with no unit attached unambiguously means 1.5 - a valid parse, not garbage."""
    assert parse_indian_number("one and a half") == Decimal("1.5")


GARBAGE_INPUTS = ["", "five bananas", "lakh lakh lakh", "crore 5", "sava"]


@pytest.mark.parametrize("text", GARBAGE_INPUTS)
def test_garbage_raises_not_silently_misparsed(text):
    with pytest.raises(ParseError):
        parse_indian_number(text)


# ---------------------------------------------------------------------------
# Edge cases the property tests above don't reach (fractional grouping, error paths,
# negative word-form, the shankh ceiling) - added to close the coverage gate, but each one
# is a real behaviour worth pinning down, not filler.
# ---------------------------------------------------------------------------

@given(
    st.integers(min_value=0, max_value=10**10),
    st.integers(min_value=1, max_value=99),
)
@settings(max_examples=200)
def test_format_handles_fractional_part(int_part, frac_hundredths):
    """A value with a fractional part still round-trips through format->parse, and the grouped
    integer portion (before the decimal point) still obeys the grouping invariant."""
    from decimal import Decimal as D
    value = D(int_part) + D(frac_hundredths) / D(100)
    grouped = format_indian_grouping(value)
    assert "." in grouped
    int_portion = grouped.split(".")[0]
    assert format_indian_grouping(int_part) == int_portion
    assert parse_indian_number(grouped) == value


def test_format_fractional_exact_example():
    from decimal import Decimal as D
    assert format_indian_grouping(D("120000.5")) == "1,20,000.5"
    assert format_indian_grouping(D("100.25")) == "100.25"


def test_number_with_no_trailing_scale_word_raises():
    with pytest.raises(ParseError):
        parse_indian_number("5 crore 10")


def test_number_followed_by_a_non_scale_word_raises():
    with pytest.raises(ParseError):
        parse_indian_number("5 bananas")


def test_colloquial_modifier_alone_with_no_scale_word_raises():
    with pytest.raises(ParseError):
        parse_indian_number("sava")


def test_to_words_rejects_unsupported_language():
    with pytest.raises(ValueError):
        to_words(100, lang="hi")


def test_to_words_beyond_shankh_raises():
    with pytest.raises(ValueError):
        to_words(10**19)


def test_from_words_negative_round_trip():
    assert from_words(to_words(-12345)) == -12345
    assert to_words(-12345).startswith("minus ")


def test_from_words_unrecognized_word_raises():
    with pytest.raises(ParseError):
        from_words("one bazillion")


def test_from_words_empty_raises():
    with pytest.raises(ParseError):
        from_words("")
