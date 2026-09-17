"""Crop-season lookups: the kharif/rabi/zaid season windows themselves, and per-crop calendar data.

A crop's calendar can carry three distinct exception mechanisms:
  - "default": the crop's one national season/window.
  - "season_reassignment": a crop normally in one season is genuinely grown as a DIFFERENT season's
    crop in specific states, under its own regional name (paddy's boro/dalua/dalwa/navarai/punja/
    garma rabi/summer rice).
  - "sowing_split": the crop stays in the SAME season everywhere, but the sowing window itself splits
    by region/irrigation (cotton's north-zone-irrigated vs central/south-zone-monsoon sowing).
  - "no_fixed_season": the crop's cycle is defined by planting month + duration, not by kharif/rabi/
    zaid at all (sugarcane's eksali/adsali).

crop_season() returns every matching row rather than silently collapsing to one, mirroring
mass.convert_traditional's posture on maund/seer/candy - a caller asking a bare question about a
crop with more than one mechanism gets the full picture, not a silently-picked default.
"""
from __future__ import annotations

from bharat_units._registry_loader import CropCalendarRow, load_crop_calendar, load_season_windows
from bharat_units.errors import AmbiguousUnitError, ScaleError


def season_window(season: str):
    """The sowing/harvest month range for kharif/rabi/zaid itself (not a specific crop)."""
    windows = load_season_windows()
    if season not in windows:
        raise ScaleError(f"unrecognized season {season!r}")
    return windows[season]


def _month_in_range(month: int, start: int, end: int) -> bool:
    if start <= end:
        return start <= month <= end
    return month >= start or month <= end  # wrap-around, not currently exercised but kept correct


def active_seasons(month: int) -> list[str]:
    """Which season(s) have EITHER sowing OR harvest active in the given calendar month (1-12).
    Deliberately returns a list, not a single season: a month can have one crop's harvest tailing
    off and a different crop's sowing beginning (e.g. October: kharif harvest + rabi sowing start) -
    a caller assuming seasons are mutually exclusive by month is exactly the trap this function's
    callers (the season_definitions subcategory) test for.
    """
    if not 1 <= month <= 12:
        raise ValueError(f"month must be 1-12, got {month!r}")
    windows = load_season_windows()
    active = [
        season for season, w in windows.items()
        if _month_in_range(month, w.sowing_start_month, w.sowing_end_month)
        or _month_in_range(month, w.harvest_start_month, w.harvest_end_month)
    ]
    return sorted(active)


def crop_season(crop: str, region: str | None = None) -> list[CropCalendarRow]:
    """Every crop_calendar.csv row matching `crop` (optionally filtered to a region_scope). Returns
    a list, never silently picks one - see the module docstring's three mechanisms."""
    rows = [r for r in load_crop_calendar() if r.crop == crop]
    if not rows:
        raise ScaleError(f"unrecognized crop {crop!r}")
    if region is not None:
        rows = [r for r in rows if r.region_scope == region]
    return rows


def default_row(crop: str) -> CropCalendarRow:
    """The national default (mechanism == 'default') row for `crop`. Raises AmbiguousUnitError, not
    ScaleError, when the crop is real but has no single default row (sugarcane: only no_fixed_season
    rows exist) - mirroring convert_traditional's candy/peti posture: there is no defensible "the"
    season/value, not merely an unrecognized name.
    """
    rows = [r for r in crop_season(crop) if r.mechanism == "default"]
    if not rows:
        raise AmbiguousUnitError(
            f"{crop!r} has no single default season - see its mechanism-specific rows "
            "(crop_season() with no filter returns all of them)"
        )
    return rows[0]


def has_fixed_season(crop: str) -> bool:
    """False iff EVERY row for `crop` is no_fixed_season (sugarcane) - i.e. the crop genuinely does
    not fit the kharif/rabi/zaid model at all, which is itself the fact under test."""
    return any(r.mechanism != "no_fixed_season" for r in crop_season(crop))


def sowing_window(crop: str, region: str | None = None) -> tuple[int | None, int | None]:
    row = crop_season(crop, region=region)[0] if region is not None else default_row(crop)
    return row.sowing_start_month, row.sowing_end_month


def harvest_window(crop: str, region: str | None = None) -> tuple[int | None, int | None]:
    row = crop_season(crop, region=region)[0] if region is not None else default_row(crop)
    return row.harvest_start_month, row.harvest_end_month
