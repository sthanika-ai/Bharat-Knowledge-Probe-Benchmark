"""Property-based tests for bharat_units.seasons and fiscal.marketing_year_bounds.

These run BEFORE any C4 item is generated.
"""
from datetime import date

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bharat_units._registry_loader import load_msp_policy_facts
from bharat_units.errors import AmbiguousUnitError, ScaleError
from bharat_units.fiscal import marketing_year_bounds
from bharat_units.seasons import (
    _month_in_range,
    active_seasons,
    crop_season,
    default_row,
    has_fixed_season,
    harvest_window,
    season_window,
    sowing_window,
)

MARKETING_YEAR_TYPES = ["crop_year", "sugar_season", "cotton_season", "oil_year",
                         "kharif_marketing_season", "rabi_marketing_season"]


# ---------------------------------------------------------------------------
# season_window / active_seasons: the season-level (not per-crop) facts
# ---------------------------------------------------------------------------

def test_kharif_rabi_zaid_windows():
    kharif = season_window("kharif")
    assert (kharif.sowing_start_month, kharif.sowing_end_month) == (6, 7)
    assert (kharif.harvest_start_month, kharif.harvest_end_month) == (9, 10)
    rabi = season_window("rabi")
    assert (rabi.sowing_start_month, rabi.sowing_end_month) == (10, 12)
    assert (rabi.harvest_start_month, rabi.harvest_end_month) == (3, 4)
    zaid = season_window("zaid")
    assert (zaid.sowing_start_month, zaid.sowing_end_month) == (3, 6)


def test_season_window_unrecognized_raises():
    with pytest.raises(ScaleError):
        season_window("monsoon")


@pytest.mark.parametrize("month,expected", [
    (6, ["kharif", "zaid"]),  # kharif sowing start + zaid's Mar-Jun tail end overlap
    (7, ["kharif"]),
    (10, ["kharif", "rabi"]),  # kharif harvest tail + rabi sowing start overlap
    (3, ["rabi", "zaid"]),     # rabi harvest + zaid start overlap
])
def test_active_seasons_overlap(month, expected):
    assert active_seasons(month) == sorted(expected)


def test_active_seasons_not_assumed_mutually_exclusive():
    """The explicit trap this function exists to test: a model assuming exactly one season is
    active per month is wrong for at least one real month (October)."""
    assert len(active_seasons(10)) > 1


def test_active_seasons_rejects_bad_month():
    with pytest.raises(ValueError):
        active_seasons(13)
    with pytest.raises(ValueError):
        active_seasons(0)


def test_month_in_range_wrap_around():
    """No current season_windows.csv row wraps around year-end, but paddy's regional rabi/summer
    rows (Nov-Feb) plausibly could reuse this helper later - exercised directly here rather than
    left as an untested "kept correct" claim."""
    assert _month_in_range(12, 11, 2)  # December, within an Nov-Feb window
    assert _month_in_range(1, 11, 2)   # January, within an Nov-Feb window
    assert not _month_in_range(6, 11, 2)  # June, outside an Nov-Feb window


# ---------------------------------------------------------------------------
# crop_season: the three exception mechanisms, each asserted directly
# ---------------------------------------------------------------------------

def test_paddy_has_a_default_plus_seven_regional_rows():
    rows = crop_season("paddy")
    assert len(rows) == 8  # 1 default (kharif) + 7 season_reassignment (WB/AS/OD/AP/KL/TN/BR)
    mechanisms = {r.mechanism for r in rows}
    assert mechanisms == {"default", "season_reassignment"}


def test_paddy_default_is_kharif_but_wb_region_is_rabi():
    assert default_row("paddy").season == "kharif"
    wb_row = crop_season("paddy", region="WB")[0]
    assert wb_row.season == "rabi"
    assert wb_row.mechanism == "season_reassignment"
    assert wb_row.regional_name == "boro"


def test_boro_is_shared_by_wb_and_assam():
    assert crop_season("paddy", region="WB")[0].regional_name == "boro"
    assert crop_season("paddy", region="AS")[0].regional_name == "boro"


def test_cotton_sowing_split_is_same_season_different_months():
    """The mechanism-(b) contrast with paddy's mechanism-(a): cotton's two region rows share the
    SAME season value (never reclassified), unlike paddy's genuine season_reassignment."""
    rows = [r for r in crop_season("cotton") if r.mechanism == "sowing_split"]
    assert len(rows) == 2
    seasons = {r.season for r in rows}
    assert seasons == {"kharif"}  # never reclassified
    windows = {(r.sowing_start_month, r.sowing_end_month) for r in rows}
    assert len(windows) == 2  # but genuinely different sowing windows


def test_sugarcane_has_no_default_row_and_is_no_fixed_season():
    """Mechanism (c): sugarcane doesn't fit kharif/rabi/zaid at all - asserted directly, the C4
    analogue of C5's FY24-vs-FY2024-25 collision assertion and C2's AmbiguousUnitError contract."""
    rows = crop_season("sugarcane")
    assert len(rows) == 2
    assert all(r.mechanism == "no_fixed_season" for r in rows)
    assert not has_fixed_season("sugarcane")
    with pytest.raises(AmbiguousUnitError):
        default_row("sugarcane")


def test_sugarcane_eksali_and_adsali_have_different_durations():
    rows = {r.regional_name: r for r in crop_season("sugarcane")}
    assert rows["eksali"].duration_months == 12
    assert rows["adsali"].duration_months == 18
    assert rows["eksali"].duration_months != rows["adsali"].duration_months


def test_has_fixed_season_true_for_ordinary_crops():
    assert has_fixed_season("wheat")
    assert has_fixed_season("paddy")
    assert has_fixed_season("cotton")


def test_crop_season_unrecognized_crop_raises():
    with pytest.raises(ScaleError):
        crop_season("quinoa")


def test_sowing_window_and_harvest_window_default_vs_region():
    assert sowing_window("wheat") == (10, 12)
    assert harvest_window("wheat") == (3, 4)
    assert sowing_window("paddy") == (6, 7)  # default (national kharif)
    assert sowing_window("paddy", region="OD") == (11, 2)  # dalua (rabi/summer)


# ---------------------------------------------------------------------------
# marketing_year_bounds: the six agricultural marketing years
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("year_type,label,expected", [
    ("crop_year", "2023-24", (date(2023, 7, 1), date(2024, 6, 30))),
    ("sugar_season", "2024-25", (date(2024, 10, 1), date(2025, 9, 30))),
    ("cotton_season", "2024-25", (date(2024, 10, 1), date(2025, 9, 30))),
    ("oil_year", "2024-25", (date(2024, 11, 1), date(2025, 10, 31))),
    ("kharif_marketing_season", "2023-24", (date(2023, 10, 1), date(2024, 9, 30))),
    ("rabi_marketing_season", "2024-25", (date(2024, 4, 1), date(2025, 3, 31))),
])
def test_marketing_year_bounds_worked_examples(year_type, label, expected):
    assert marketing_year_bounds(year_type, label) == expected


def test_marketing_year_bounds_strips_kms_rms_prefix():
    assert marketing_year_bounds("kharif_marketing_season", "KMS 2023-24") == \
        marketing_year_bounds("kharif_marketing_season", "2023-24")
    assert marketing_year_bounds("rabi_marketing_season", "RMS2024-25") == \
        marketing_year_bounds("rabi_marketing_season", "2024-25")


def test_sugar_season_and_cotton_season_share_the_same_span_but_are_distinct_series():
    """Same Oct-Sep month-anchor, but tracked as genuinely separate statistical series - the
    marketing_statistical_years subcategory's 'same span, different year_type' contrast item."""
    assert marketing_year_bounds("sugar_season", "2024-25") == \
        marketing_year_bounds("cotton_season", "2024-25")


def test_kms_and_rms_do_not_cover_the_same_12_months():
    kms_start, kms_end = marketing_year_bounds("kharif_marketing_season", "2023-24")
    rms_start, rms_end = marketing_year_bounds("rabi_marketing_season", "2023-24")
    assert (kms_start, kms_end) != (rms_start, rms_end)


def test_marketing_year_bounds_unrecognized_type_raises():
    with pytest.raises(ScaleError):
        marketing_year_bounds("monsoon_year", "2024-25")


@given(st.sampled_from(MARKETING_YEAR_TYPES), st.integers(min_value=2015, max_value=2098))
@settings(max_examples=100)
def test_marketing_year_bounds_start_before_end(year_type, start_year):
    label = f"{start_year}-{(start_year + 1) % 100:02d}"
    start, end = marketing_year_bounds(year_type, label)
    assert start < end


# ---------------------------------------------------------------------------
# msp_policy_facts.csv - the CACP/CCEA/Advance-Estimates registry backing policy_timing
# ---------------------------------------------------------------------------

def test_msp_policy_facts_cacp_recommends_ccea_approves():
    facts = load_msp_policy_facts()
    assert facts["cacp_role"].value == "recommends MSP"
    assert facts["ccea_role"].value == "approves MSP"


def test_msp_policy_facts_dated_examples_are_volatile():
    facts = load_msp_policy_facts()
    assert facts["kharif_msp_2025_26_approval_date"].value == "2025-05-28"
    assert facts["rabi_msp_2025_26_approval_date"].value == "2024-10-16"
    assert facts["kharif_msp_2025_26_approval_date"].volatile is True
    assert facts["rabi_msp_2025_26_approval_date"].volatile is True


def test_msp_policy_facts_advance_estimate_schedule():
    facts = load_msp_policy_facts()
    assert facts["ae_schedule_1st"].value == "September"
    assert facts["ae_schedule_2nd"].value == "February"
