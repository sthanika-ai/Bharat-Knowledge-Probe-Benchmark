"""Property-based tests for bharat_units.fiscal.

These run BEFORE any C5 item is generated.
"""
from datetime import date, timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bharat_units.errors import ParseError
from bharat_units.fiscal import (
    ay_of_fy,
    fy_bounds,
    fy_of,
    fy_of_ay,
    ist_utc_offset,
    parse_fy_label,
    quarter_bounds,
    quarter_of,
)

# ---------------------------------------------------------------------------
# The single highest-yield trap in the whole benchmark, asserted directly and precisely:
# FY24 and FY2024-25 are DIFFERENT, adjacent years.
# ---------------------------------------------------------------------------

def test_fy24_short_form_means_year_ending_2024():
    start, end = fy_bounds("FY24")
    assert (start, end) == (date(2023, 4, 1), date(2024, 3, 31))


def test_fy2024_25_long_form_means_year_starting_2024():
    start, end = fy_bounds("FY2024-25")
    assert (start, end) == (date(2024, 4, 1), date(2025, 3, 31))


def test_fy24_and_fy2024_25_are_different_adjacent_years():
    """The collision itself, asserted rather than left implicit: these are NOT the same year."""
    assert fy_bounds("FY24") != fy_bounds("FY2024-25")
    assert fy_bounds("FY24")[1] == date(2024, 3, 31)
    assert fy_bounds("FY2024-25")[0] == date(2024, 4, 1)
    # exactly one year (to the day) apart at the seam
    assert fy_bounds("FY2024-25")[0] - fy_bounds("FY24")[1] == timedelta(days=1)


def test_bare_4digit_matches_bare_2digit():
    assert parse_fy_label("FY24") == parse_fy_label("FY2024") == parse_fy_label(2024) == parse_fy_label(24)


# ---------------------------------------------------------------------------
# parse_fy_label: accepted forms and malformed-input rejection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,expected", [
    ("FY24", (2023, 2024)),
    ("FY2024-25", (2024, 2025)),
    ("2024-25", (2024, 2025)),
    ("FY24-25", (2024, 2025)),
    ("FY 2024-25", (2024, 2025)),
    ("FY2024-2025", (2024, 2025)),
    (2024, (2023, 2024)),
    ("24", (2023, 2024)),
])
def test_parse_fy_label_accepted_forms(label, expected):
    assert parse_fy_label(label) == expected


@pytest.mark.parametrize("label", ["FY2024-26", "FY2024-23", "not a year", "", "FY2024-25-26"])
def test_parse_fy_label_rejects_malformed(label):
    with pytest.raises(ParseError):
        parse_fy_label(label)


# ---------------------------------------------------------------------------
# fy_of: round trip against fy_bounds, including the exact rollover day (31 Mar / 1 Apr)
# ---------------------------------------------------------------------------

@given(st.dates(min_value=date(2000, 1, 1), max_value=date(2099, 12, 31)))
@settings(max_examples=300)
def test_fy_of_round_trips_through_fy_bounds(d):
    label = fy_of(d)
    start, end = fy_bounds(label)
    assert start <= d <= end


def test_fy_of_rollover_day():
    assert fy_of(date(2024, 3, 31)) == "FY2023-24"
    assert fy_of(date(2024, 4, 1)) == "FY2024-25"


# ---------------------------------------------------------------------------
# FY <-> AY mapping
# ---------------------------------------------------------------------------

def test_ay_of_fy_worked_example():
    assert ay_of_fy("FY2024-25") == "AY2025-26"


def test_fy_of_ay_worked_example():
    assert fy_of_ay("AY2025-26") == "FY2024-25"


@given(st.integers(min_value=2000, max_value=2098))
@settings(max_examples=100)
def test_ay_fy_round_trip(start_year):
    fy_label = f"FY{start_year}-{(start_year + 1) % 100:02d}"
    assert fy_of_ay(ay_of_fy(fy_label)) == fy_label


def test_ay_of_fy24_short_form_uses_the_correct_earlier_year():
    """Stacks both traps: FY24 (not FY2024-25) income is assessed in AY2024-25, one year
    earlier than FY2024-25's AY2025-26."""
    assert ay_of_fy("FY24") == "AY2024-25"


# ---------------------------------------------------------------------------
# Quarters: FY-quarter definition, and the calendar-quarter-vs-FY-quarter trap
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("month,expected_q", [(4, 1), (5, 1), (6, 1), (7, 2), (8, 2), (9, 2),
                                                (10, 3), (11, 3), (12, 3), (1, 4), (2, 4), (3, 4)])
def test_quarter_of_all_months(month, expected_q):
    year = 2024 if month >= 4 else 2025
    assert quarter_of(date(year, month, 15)) == expected_q


def test_january_is_fy_q4_not_calendar_q1():
    """The explicit mixed-basis trap: January is FY-Q4, NOT calendar-Q1, even though
    it's the same calendar month people default to associating with 'Q1'."""
    assert quarter_of(date(2025, 1, 15)) == 4


def test_quarter_bounds_all_four_of_a_fy():
    assert quarter_bounds("FY2024-25", 1) == (date(2024, 4, 1), date(2024, 6, 30))
    assert quarter_bounds("FY2024-25", 2) == (date(2024, 7, 1), date(2024, 9, 30))
    assert quarter_bounds("FY2024-25", 3) == (date(2024, 10, 1), date(2024, 12, 31))
    assert quarter_bounds("FY2024-25", 4) == (date(2025, 1, 1), date(2025, 3, 31))


def test_quarter_bounds_rejects_bad_quarter_number():
    with pytest.raises(ValueError):
        quarter_bounds("FY2024-25", 5)


@given(st.integers(min_value=2000, max_value=2098), st.integers(min_value=1, max_value=4))
@settings(max_examples=200)
def test_quarter_bounds_within_fy_bounds(start_year, q):
    fy_label = f"FY{start_year}-{(start_year + 1) % 100:02d}"
    fy_start, fy_end = fy_bounds(fy_label)
    q_start, q_end = quarter_bounds(fy_label, q)
    assert fy_start <= q_start <= q_end <= fy_end


# ---------------------------------------------------------------------------
# IST offset - sourced from the registry, not a hand-typed literal
# ---------------------------------------------------------------------------

def test_ist_offset_is_5h30m_no_dst():
    assert ist_utc_offset() == timedelta(hours=5, minutes=30)


def test_ist_offset_rejects_malformed_registry_value(monkeypatch):
    """Defends against a corrupted/hand-edited registry row, not just current-data happy path."""
    import bharat_units.fiscal as fiscal_module
    from bharat_units._registry_loader import FiscalFact

    bad_fact = FiscalFact(
        id="FF015", fact_id="ist_offset", description="bad", value="garbage",
        effective_note="", source_url=None, confidence="high", volatile=False,
    )
    monkeypatch.setattr(fiscal_module, "load_fiscal_facts", lambda: {"ist_offset": bad_fact})
    with pytest.raises(ParseError):
        fiscal_module.ist_utc_offset()
