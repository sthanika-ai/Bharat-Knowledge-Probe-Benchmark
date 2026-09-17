"""Property-based tests for bharat_units.mass.

These run BEFORE any C2 item is generated.
"""
from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bharat_units.errors import AmbiguousUnitError, ScaleError
from bharat_units.mass import ConversionResult, convert_traditional, rate_convert, to_mass_scale, tonne_from_scaled
from bharat_units.numerals import scale_factor

MASS_UNITS = ["gram", "kilogram", "quintal", "tonne", "pound"]


# ---------------------------------------------------------------------------
# to_mass_scale: round-trip, transitivity, exact definitional ratios
# ---------------------------------------------------------------------------

@given(
    st.decimals(min_value="0.01", max_value="100000", allow_nan=False, allow_infinity=False, places=4),
    st.sampled_from(MASS_UNITS), st.sampled_from(MASS_UNITS),
)
@settings(max_examples=300)
def test_to_mass_scale_round_trip(value, unit_a, unit_b):
    converted = to_mass_scale(value, unit_a, unit_b)
    back = to_mass_scale(converted, unit_b, unit_a)
    assert abs(back - value) < Decimal("1e-9")


def test_quintal_is_exactly_100kg():
    assert to_mass_scale(1, "quintal", "kilogram") == Decimal(100)


def test_tonne_is_exactly_10_quintal():
    assert to_mass_scale(1, "tonne", "quintal") == Decimal(10)


# ---------------------------------------------------------------------------
# rate_convert: the mirror-image ratio, and the classic "invert by mistake" trap
# ---------------------------------------------------------------------------

@given(
    st.decimals(min_value="0.01", max_value="100000", allow_nan=False, allow_infinity=False, places=2),
)
@settings(max_examples=200)
def test_rate_convert_round_trip(value):
    converted = rate_convert(value, "quintal", "kilogram")
    back = rate_convert(converted, "kilogram", "quintal")
    assert abs(back - value) < Decimal("1e-9")


def test_rate_convert_is_inverse_ratio_of_to_mass_scale():
    """The defining property: rate_convert and to_mass_scale use INVERTED ratios - this is the one
    thing every quintal-subcategory item is designed to catch a model getting backwards."""
    q = to_mass_scale(1, "quintal", "kilogram")  # 100 - "how many kg in 1 quintal"
    r = rate_convert(1, "quintal", "kilogram")  # 0.01 - "Rs.1/quintal is what Rs/kg"
    assert q * r == Decimal(1)  # exact inverses


def test_msp_quintal_to_kg_worked_example():
    assert rate_convert(2000, "quintal", "kilogram") == Decimal(20)


# ---------------------------------------------------------------------------
# tonne_from_scaled: composition with numerals.scale_factor (cross-category reuse)
# ---------------------------------------------------------------------------

def test_tonne_from_scaled_matches_scale_factor_directly():
    assert tonne_from_scaled(1, "lakh") == scale_factor("lakh")
    assert tonne_from_scaled(1, "million") == scale_factor("million")


def test_lmt_mmt_worked_example():
    # 3539.59 LMT and 357.73 MMT were reported for the same estimate round (different vintages,
    # not required to match exactly) - but the UNIT RELATIONSHIP is exact: 1 million = 10 lakh.
    assert scale_factor("million") == 10 * scale_factor("lakh")


def test_tonne_from_scaled_noop_with_no_scale_word():
    assert tonne_from_scaled(42) == Decimal(42)


# ---------------------------------------------------------------------------
# convert_traditional: AmbiguousUnitError contract
# ---------------------------------------------------------------------------

def test_maund_without_region_raises():
    with pytest.raises(AmbiguousUnitError):
        convert_traditional(1, "maund", "kilogram")


def test_seer_without_variant_raises():
    with pytest.raises(AmbiguousUnitError):
        convert_traditional(1, "seer", "gram")


@pytest.mark.parametrize("strict", [True, False])
def test_candy_always_raises_regardless_of_strict(strict):
    with pytest.raises(AmbiguousUnitError):
        convert_traditional(1, "candy", "kilogram", strict=strict)


@pytest.mark.parametrize("strict", [True, False])
def test_peti_always_raises_regardless_of_strict(strict):
    with pytest.raises(AmbiguousUnitError):
        convert_traditional(1, "peti", "gram", strict=strict)


def test_unrecognized_unit_raises_scale_error():
    with pytest.raises(ScaleError):
        convert_traditional(1, "furlong", "gram")


def test_to_mass_scale_unrecognized_to_unit_raises_scale_error():
    with pytest.raises(ScaleError):
        to_mass_scale(1, "quintal", "furlong")


def test_strict_false_disambiguates_and_reports_the_assumption():
    """With multiple valued rows and strict=False, a best-effort pick is made - but it must show
    its work: assumed/warnings/confidence, never a silent, unlabeled scalar."""
    r = convert_traditional(1, "seer", "gram", strict=False)
    assert isinstance(r, ConversionResult)
    assert r.assumed.get("variant") is not None
    assert len(r.warnings) >= 1
    assert "variant" in r.warnings[0]


@pytest.mark.parametrize("region,expected_kg", [
    ("Bengal", Decimal("37.324")),
    ("Punjab", Decimal("36.740")),
    ("standard", Decimal("37.324")),
])
def test_maund_regions_give_distinct_values(region, expected_kg):
    r = convert_traditional(1, "maund", "kilogram", region=region)
    assert r.value == expected_kg


def test_maund_bengal_and_punjab_are_genuinely_different():
    bengal = convert_traditional(1, "maund", "kilogram", region="Bengal").value
    punjab = convert_traditional(1, "maund", "kilogram", region="Punjab").value
    assert bengal != punjab


@pytest.mark.parametrize("variant,expected_g", [
    ("trade_customary", Decimal("933.1")),
    ("statutory_1956", Decimal("933.104304")),
])
def test_seer_variant_param_selects_correct_registry_row(variant, expected_g):
    r = convert_traditional(1, "seer", "gram", variant=variant)
    assert r.value == expected_g


def test_seer_trade_and_statutory_agree_to_within_rounding():
    # CORRECTION: this used to assert
    # trade_customary != statutory_1956 on the theory that the 1956 Act defined a seer as
    # 1.25 kg. That figure was itself wrong - the Act defines the seer as 80 tola = 0.933104304
    # kg, which agrees with the trade-customary seer to within rounding, not a materially
    # different legal figure.
    trade = convert_traditional(1, "seer", "gram", variant="trade_customary").value
    statutory = convert_traditional(1, "seer", "gram", variant="statutory_1956").value
    assert abs(trade - statutory) < Decimal("1")


def test_result_carries_confidence_and_source():
    r = convert_traditional(1, "maund", "kilogram", region="Bengal")
    assert r.confidence in ("high", "medium", "low", "unverified")
    assert r.source is not None


def test_disambiguated_call_carries_no_spurious_assumed_warning():
    r = convert_traditional(1, "seer", "gram", variant="trade_customary")
    assert r.assumed == {}
    assert r.warnings == ()


# ---------------------------------------------------------------------------
# The tola/chhatak/pav/ratti chain - the C2 analogue of the numerals round-trip tests
# ---------------------------------------------------------------------------

def test_tola_chhatak_pav_chain():
    tola = convert_traditional(1, "tola", "gram").value
    chhatak = convert_traditional(1, "chhatak", "gram").value
    pav = convert_traditional(1, "pav", "gram").value
    assert abs(5 * tola - chhatak) < Decimal("0.01")
    assert abs(4 * chhatak - pav) < Decimal("0.01")


def test_ratti_sunari_pakki_ratio():
    sunari = convert_traditional(1, "ratti", "gram", variant="sunari").value
    pakki = convert_traditional(1, "ratti", "gram", variant="pakki").value
    assert abs(pakki / sunari - Decimal("1.5")) < Decimal("0.01")


def test_cotton_bale_is_170kg():
    r = convert_traditional(1, "cotton_bale", "kilogram")
    assert r.value == Decimal(170)
    assert r.confidence == "high"


def test_jute_bale_is_medium_confidence_not_high():
    """Jute's 180kg figure should NOT be promoted to high confidence on the sourcing found so far
    (only an exam-answer-key mention, no ministry/board citation)."""
    r = convert_traditional(1, "jute_bale", "kilogram")
    assert r.value == Decimal(180)
    assert r.confidence == "medium"


# ---------------------------------------------------------------------------
# Adversarial-style: real MSP/production phrasings should compute correctly end to end
# ---------------------------------------------------------------------------

def test_adversarial_msp_wheat_per_quintal():
    assert rate_convert(2275, "quintal", "kilogram") == Decimal("22.75")


def test_adversarial_foodgrain_million_tonnes():
    assert tonne_from_scaled(Decimal("330"), "million") == Decimal("330000000")
