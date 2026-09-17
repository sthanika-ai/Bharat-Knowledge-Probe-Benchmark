"""Indian fiscal-year, assessment-year, and quarter arithmetic.

Unlike area.py/mass.py, there is no AmbiguousUnitError-style registry here - every fact in this
module is a single, national, legally-defined convention with no regional variance - but there IS
a genuine, well-attested labeling collision this module must get exactly right:

  - Short form "FYnn" is named by the year the FY ENDS in: FY24 = 1 Apr 2023 - 31 Mar 2024.
  - Long form "FYyyyy-yy" is named start-year-end-year: FY2024-25 = 1 Apr 2024 - 31 Mar 2025.
  - FY24 and FY2024-25 are DIFFERENT, adjacent years - not two spellings of the same year. Getting
    this backwards (silently expanding "FY24" to "FY2024-25") produces a wrong-by-one-year answer
    that looks entirely plausible - arguably the highest-yield trap in the whole benchmark.

parse_fy_label() is the shared primitive every other function composes with, mirroring
numerals.scale_factor() being the shared primitive C1/C2 both call into.
"""
from __future__ import annotations

import re
from datetime import date, timedelta

from bharat_units._registry_loader import load_fiscal_facts, load_marketing_years
from bharat_units.errors import ParseError, ScaleError

_FY_PREFIX_RE = re.compile(r"^\s*FY\s*", re.IGNORECASE)
_AY_PREFIX_RE = re.compile(r"^\s*AY\s*", re.IGNORECASE)
_ALPHA_PREFIX_RE = re.compile(r"^\s*[A-Za-z]+\s*")
_BARE_RE = re.compile(r"^(\d{2}|\d{4})\s*$")
_RANGE_RE = re.compile(r"^(\d{2}|\d{4})\s*-\s*(\d{2}|\d{4})\s*$")
_MONTH_END_DAY = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

_QUARTER_MONTHS = {1: (4, 6), 2: (7, 9), 3: (10, 12), 4: (1, 3)}
_QUARTER_END_DAY = {6: 30, 9: 30, 12: 31, 3: 31}


def _normalize_year(digits: str) -> int:
    # This benchmark's timeframe is entirely 21st century; a 2-digit year is always 2000+n.
    return int(digits) if len(digits) == 4 else 2000 + int(digits)


def _reconcile_end_year(start_year: int, end_raw: str) -> int:
    if len(end_raw) == 4:
        return int(end_raw)
    candidate = (start_year // 100) * 100 + int(end_raw)
    if candidate <= start_year:  # century rollover, e.g. start=2099, end="00"
        candidate += 100
    return candidate


def parse_fy_label(label: str | int) -> tuple[int, int]:
    """Parse an FY (or, after stripping an 'AY' prefix by the caller, an AY) label into
    (start_year, end_year), both full 4-digit calendar years.

    Accepts: "FY24", "FY2024", 24, 2024 (bare - end year), and "FY2024-25", "2024-25", "FY24-25"
    (range - start_year-end_year, validated to be exactly one year apart). Raises ParseError on
    anything else, or on a range whose two halves aren't consecutive years (e.g. "FY2024-26") -
    refuses to silently guess which end was meant, same posture as AmbiguousUnitError elsewhere in
    this library.
    """
    s = _FY_PREFIX_RE.sub("", str(label).strip())
    m_range = _RANGE_RE.match(s)
    if m_range:
        start_year = _normalize_year(m_range.group(1))
        end_year = _reconcile_end_year(start_year, m_range.group(2))
        if end_year != start_year + 1:
            raise ParseError(
                f"malformed FY range {label!r}: end year {end_year} is not start year {start_year} + 1"
            )
        return start_year, end_year
    m_bare = _BARE_RE.match(s)
    if m_bare:
        end_year = _normalize_year(m_bare.group(1))
        return end_year - 1, end_year
    raise ParseError(f"unrecognized FY label {label!r}")


def fy_bounds(label: str | int) -> tuple[date, date]:
    """"FY24" | "FY2024-25" | "2024-25" -> (date(start,4,1), date(end,3,31))."""
    start_year, end_year = parse_fy_label(label)
    return date(start_year, 4, 1), date(end_year, 3, 31)


def fy_of(d: date) -> str:
    """A calendar date -> the canonical long-form "FYyyyy-yy" label of the FY containing it."""
    start_year = d.year if d.month >= 4 else d.year - 1
    end_year = start_year + 1
    return f"FY{start_year}-{end_year % 100:02d}"


def ay_of_fy(fy_label: str | int) -> str:
    """"FY2024-25" -> "AY2025-26" (the assessment year immediately following the FY)."""
    _, fy_end_year = parse_fy_label(fy_label)
    ay_start_year = fy_end_year
    ay_end_year = ay_start_year + 1
    return f"AY{ay_start_year}-{ay_end_year % 100:02d}"


def fy_of_ay(ay_label: str | int) -> str:
    """"AY2025-26" -> "FY2024-25" (inverse of ay_of_fy)."""
    s = _AY_PREFIX_RE.sub("", str(ay_label).strip())
    ay_start_year, _ = parse_fy_label(s)
    fy_start_year = ay_start_year - 1
    fy_end_year = ay_start_year
    return f"FY{fy_start_year}-{fy_end_year % 100:02d}"


def quarter_of(d: date) -> int:
    """A calendar date -> its FY quarter (1-4). FY-Q1 = Apr-Jun ... FY-Q4 = Jan-Mar.

    Deliberately NOT the calendar-year quarter: quarter_of(date(2025,1,15)) is FY-Q4 (Jan-Mar of
    FY2024-25), even though January is calendar-Q1 - see the mixed_basis_reasoning subcategory.
    """
    month = d.month
    if 4 <= month <= 6:
        return 1
    if 7 <= month <= 9:
        return 2
    if 10 <= month <= 12:
        return 3
    return 4


def quarter_bounds(fy_label: str | int, q: int) -> tuple[date, date]:
    """The (start, end) dates of FY quarter q (1-4) of the given FY."""
    if q not in (1, 2, 3, 4):
        raise ValueError(f"quarter must be 1-4, got {q!r}")
    start_year, end_year = parse_fy_label(fy_label)
    start_month, end_month = _QUARTER_MONTHS[q]
    year = end_year if q == 4 else start_year  # Jan-Mar of a "FY2024-25" falls in calendar 2025
    start = date(year, start_month, 1)
    end = date(year, end_month, _QUARTER_END_DAY[end_month])
    return start, end


def ist_utc_offset() -> timedelta:
    """India Standard Time's fixed offset from UTC (+05:30, no daylight saving), parsed from the
    sourced registry fact rather than hand-typed as a literal."""
    facts = load_fiscal_facts()
    value = facts["ist_offset"].value  # "UTC+05:30"
    m = re.match(r"UTC([+-])(\d{2}):(\d{2})", value)
    if not m:
        raise ParseError(f"unrecognized ist_offset registry value {value!r}")
    sign, hh, mm = m.group(1), int(m.group(2)), int(m.group(3))
    delta = timedelta(hours=hh, minutes=mm)
    return -delta if sign == "-" else delta


def marketing_year_bounds(year_type: str, label: str | int) -> tuple[date, date]:
    """One generic, registry-driven function for all six agricultural marketing years
    (crop_year/sugar_season/cotton_season/oil_year/kharif_marketing_season/rabi_marketing_season),
    rather than six bespoke functions - they all share the exact same "named year, anchored to a
    start/end month" shape.

    `label` accepts the same bare/range forms as parse_fy_label ("2024-25", 24, "2024"), with any
    alphabetic prefix ("KMS", "RMS") stripped first the same permissive way "FY"/"AY" are stripped
    elsewhere in this module - reusing parse_fy_label's parser directly rather than reimplementing
    it, since KMS/RMS labels are, in every source checked, always long-form "yyyy-yy" with no
    attested short-form usage to disambiguate (unlike FY - see Sec.0.2's note on this difference).
    """
    years = load_marketing_years()
    if year_type not in years:
        raise ScaleError(f"unrecognized marketing year_type {year_type!r}")
    spec = years[year_type]
    stripped = _ALPHA_PREFIX_RE.sub("", str(label).strip())
    start_year, end_year = parse_fy_label(stripped)
    start = date(start_year, spec.start_month, 1)
    end_year_actual = end_year if spec.end_month < spec.start_month else start_year
    end = date(end_year_actual, spec.end_month, _MONTH_END_DAY[spec.end_month])
    return start, end
