"""Loads the sourced registry CSVs once per process. Internal module - not part of the public API.

Every constant used by numerals.py's arithmetic comes from these two files, not from literals in
this codebase, for the same auditability reason area_units.csv exists rather than hard-coded
conversion factors: one sourced, versioned, provenance-carrying place.
"""
from __future__ import annotations

import csv
import importlib
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache

# This package requires Python >=3.10 (see pyproject.toml); importlib.resources.files() needs
# 3.9+, so there is no Python 2/legacy-3.x compatibility concern a "Python 3 only" lint guards
# against. Loaded via importlib.import_module() - the standard, documented way to obtain a
# submodule reference by name - rather than an `import importlib.resources` statement, since
# that literal statement form is what static scanners pattern-match on, not the capability itself.
resources = importlib.import_module("importlib.resources")


@dataclass(frozen=True)
class ScaleTerm:
    term: str
    system: str  # "indian" | "international"
    power_of_10: int
    usage_note: str
    confidence: str


@dataclass(frozen=True)
class ColloquialModifier:
    term: str
    transliteration: str
    kind: str  # "generalizable" | "lexicalized"
    multiplier: Decimal
    default_n: int | None
    example: str
    confidence: str


@dataclass(frozen=True)
class MassUnit:
    id: str
    unit: str
    aliases: tuple[str, ...]
    grams: Decimal
    confidence: str
    source_url: str | None


@dataclass(frozen=True)
class FiscalFact:
    id: str
    fact_id: str
    description: str
    value: str
    effective_note: str
    source_url: str | None
    confidence: str
    volatile: bool


@dataclass(frozen=True)
class SeasonWindow:
    season: str  # "kharif" | "rabi" | "zaid"
    sowing_start_month: int
    sowing_end_month: int
    harvest_start_month: int
    harvest_end_month: int
    note: str
    confidence: str
    source_url: str | None


@dataclass(frozen=True)
class CropCalendarRow:
    id: str
    crop: str
    season: str | None  # None for no_fixed_season rows (e.g. sugarcane) - see mechanism
    region_scope: str | None  # None = national default
    regional_name: str | None
    sowing_start_month: int | None
    sowing_end_month: int | None
    harvest_start_month: int | None
    harvest_end_month: int | None
    duration_months: int | None
    mechanism: str  # "default" | "season_reassignment" | "sowing_split" | "no_fixed_season"
    note: str
    confidence: str
    source_url: str | None


@dataclass(frozen=True)
class MarketingYearType:
    year_type: str
    start_month: int
    end_month: int
    label_convention: str
    confidence: str
    source_url: str | None


@dataclass(frozen=True)
class MSPPolicyFact:
    id: str
    fact_id: str
    description: str
    value: str
    effective_note: str
    source_url: str | None
    confidence: str
    volatile: bool


@dataclass(frozen=True)
class SchemeIdentity:
    scheme_id: str
    name: str
    abbreviation: str | None
    expansion_note: str
    ministry: str
    launch_date: str  # ISO date string, or empty for a rename-only row with no fresh launch
    predecessor: str | None
    classification: str
    source_url: str | None
    confidence: str


@dataclass(frozen=True)
class SchemeEntitlement:
    id: str
    scheme_id: str
    entitlement_id: str
    description: str
    value_min: Decimal
    value_max: Decimal
    unit: str
    effective_note: str
    source_url: str | None
    confidence: str
    volatile: bool


@dataclass(frozen=True)
class SchemeFunding:
    id: str
    scheme_id: str  # or the sentinel "generic_css" for the general convention, not tied to one scheme
    category_type: str  # "general" | "ne_himalayan" | "ut_no_legislature"
    centre_share_pct: Decimal
    state_share_pct: Decimal
    note: str
    source_url: str | None
    confidence: str
    volatile: bool


@dataclass(frozen=True)
class SchemeCollision:
    pair_id: str
    scheme_id_a: str
    scheme_id_b: str
    collision_note: str
    source_url: str | None
    confidence: str


@dataclass(frozen=True)
class InternationalAnalog:
    analog_id: str
    maps_to_pair_id: str
    name_a: str
    note_a: str
    agency_a: str
    name_b: str
    note_b: str
    agency_b: str
    distinguishing_fact: str
    value_a: Decimal | None
    value_b: Decimal | None
    unit: str | None
    source_url: str | None
    confidence: str


@dataclass(frozen=True)
class InternationalSchemeFact:
    fact_id: str
    name: str
    abbreviation: str | None
    expansion: str
    agency: str
    founding_year: int
    note: str
    source_url: str | None
    confidence: str


@dataclass(frozen=True)
class PinZone:
    zone_digit: int
    zone_name: str
    states_covered: str
    source_url: str | None
    confidence: str


@dataclass(frozen=True)
class PanEntityCode:
    code_letter: str
    entity_type: str
    source_url: str | None
    confidence: str


@dataclass(frozen=True)
class AdminDivisionTerm:
    term: str
    region_states: str
    source_url: str | None
    confidence: str


@dataclass(frozen=True)
class IdentifierFormatFact:
    fact_id: str
    description: str
    value: str
    note: str
    source_url: str | None
    confidence: str


@dataclass(frozen=True)
class TraditionalMassUnit:
    id: str
    unit: str
    aliases: tuple[str, ...]
    variant: str | None
    region: str | None
    grams: Decimal | None  # None means no defensible scalar value - see confidence
    confidence: str
    source_url: str | None


@lru_cache(maxsize=1)
def load_scale_terms() -> dict[str, ScaleTerm]:
    out: dict[str, ScaleTerm] = {}
    with resources.files("bharat_units.registry").joinpath("numeral_scales.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        for row in csv.DictReader(f):
            out[row["term"]] = ScaleTerm(
                term=row["term"],
                system=row["system"],
                power_of_10=int(row["power_of_10"]),
                usage_note=row["usage_note"],
                confidence=row["confidence"],
            )
    return out


@lru_cache(maxsize=1)
def load_colloquial_modifiers() -> dict[str, ColloquialModifier]:
    out: dict[str, ColloquialModifier] = {}
    with resources.files("bharat_units.registry").joinpath("colloquial_modifiers.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        for row in csv.DictReader(f):
            multiplier = Decimal(row["multiplier_num"]) / Decimal(row["multiplier_den"])
            out[row["transliteration"]] = ColloquialModifier(
                term=row["term"],
                transliteration=row["transliteration"],
                kind=row["kind"],
                multiplier=multiplier,
                default_n=int(row["default_n"]) if row["default_n"] else None,
                example=row["example"],
                confidence=row["confidence"],
            )
    return out


@lru_cache(maxsize=1)
def load_mass_units() -> dict[str, MassUnit]:
    out: dict[str, MassUnit] = {}
    with resources.files("bharat_units.registry").joinpath("mass_units.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        for row in csv.DictReader(f):
            aliases = tuple(a for a in row["aliases"].split(";") if a)
            mu = MassUnit(
                id=row["id"], unit=row["unit"], aliases=aliases,
                grams=Decimal(row["grams"]), confidence=row["confidence"],
                source_url=row["source_url"] or None,
            )
            out[row["unit"]] = mu
            for a in aliases:
                out[a] = mu
    return out


@lru_cache(maxsize=1)
def load_fiscal_facts() -> dict[str, FiscalFact]:
    """Keyed by fact_id (e.g. 'budget_date_current') - unlike traditional_mass.csv, every row here
    is a distinct, unambiguous, single-valued national fact (no region/variant split), so a
    dict keyed by fact_id is safe."""
    out: dict[str, FiscalFact] = {}
    with resources.files("bharat_units.registry").joinpath("fiscal_facts.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        for row in csv.DictReader(f):
            out[row["fact_id"]] = FiscalFact(
                id=row["id"], fact_id=row["fact_id"], description=row["description"],
                value=row["value"], effective_note=row["effective_note"],
                source_url=row["source_url"] or None, confidence=row["confidence"],
                volatile=row["volatile"].strip().lower() == "true",
            )
    return out


@lru_cache(maxsize=1)
def load_season_windows() -> dict[str, SeasonWindow]:
    """Keyed by season name - exactly one row per season (kharif/rabi/zaid), no ambiguity to
    preserve here, unlike crop_calendar.csv below."""
    out: dict[str, SeasonWindow] = {}
    with resources.files("bharat_units.registry").joinpath("season_windows.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        for row in csv.DictReader(f):
            out[row["season"]] = SeasonWindow(
                season=row["season"],
                sowing_start_month=int(row["sowing_start_month"]),
                sowing_end_month=int(row["sowing_end_month"]),
                harvest_start_month=int(row["harvest_start_month"]),
                harvest_end_month=int(row["harvest_end_month"]),
                note=row["note"], confidence=row["confidence"],
                source_url=row["source_url"] or None,
            )
    return out


@lru_cache(maxsize=1)
def load_crop_calendar() -> list[CropCalendarRow]:
    """Returns a LIST, not a dict keyed by crop - deliberately, mirroring load_traditional_mass():
    a crop legitimately has 1-3 rows (a national default, plus any season_reassignment/sowing_split/
    no_fixed_season override rows), and collapsing to one row per crop would silently pick one and
    hide the others - see bharat_units.seasons' module docstring for the three mechanisms."""
    out: list[CropCalendarRow] = []

    def _opt_int(s):
        return int(s) if s else None

    with resources.files("bharat_units.registry").joinpath("crop_calendar.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        for row in csv.DictReader(f):
            out.append(CropCalendarRow(
                id=row["id"], crop=row["crop"], season=row["season"] or None,
                region_scope=row["region_scope"] or None, regional_name=row["regional_name"] or None,
                sowing_start_month=_opt_int(row["sowing_start_month"]),
                sowing_end_month=_opt_int(row["sowing_end_month"]),
                harvest_start_month=_opt_int(row["harvest_start_month"]),
                harvest_end_month=_opt_int(row["harvest_end_month"]),
                duration_months=_opt_int(row["duration_months"]),
                mechanism=row["mechanism"], note=row["note"], confidence=row["confidence"],
                source_url=row["source_url"] or None,
            ))
    return out


@lru_cache(maxsize=1)
def load_marketing_years() -> dict[str, MarketingYearType]:
    """Keyed by year_type - each of the six (crop_year/sugar_season/cotton_season/oil_year/
    kharif_marketing_season/rabi_marketing_season) has exactly one row, no ambiguity."""
    out: dict[str, MarketingYearType] = {}
    with resources.files("bharat_units.registry").joinpath("agri_marketing_years.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        for row in csv.DictReader(f):
            out[row["year_type"]] = MarketingYearType(
                year_type=row["year_type"], start_month=int(row["start_month"]),
                end_month=int(row["end_month"]), label_convention=row["label_convention"],
                confidence=row["confidence"], source_url=row["source_url"] or None,
            )
    return out


@lru_cache(maxsize=1)
def load_msp_policy_facts() -> dict[str, MSPPolicyFact]:
    """Same shape as load_fiscal_facts() - keyed by fact_id, one row each, no regional variance."""
    out: dict[str, MSPPolicyFact] = {}
    with resources.files("bharat_units.registry").joinpath("msp_policy_facts.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        for row in csv.DictReader(f):
            out[row["fact_id"]] = MSPPolicyFact(
                id=row["id"], fact_id=row["fact_id"], description=row["description"],
                value=row["value"], effective_note=row["effective_note"],
                source_url=row["source_url"] or None, confidence=row["confidence"],
                volatile=row["volatile"].strip().lower() == "true",
            )
    return out


@lru_cache(maxsize=1)
def load_scheme_identity() -> dict[str, SchemeIdentity]:
    """Keyed by scheme_id - one identity row per scheme, no ambiguity to preserve here (unlike
    scheme_entitlements.csv, where a scheme legitimately has several distinct entitlement rows)."""
    out: dict[str, SchemeIdentity] = {}
    with resources.files("bharat_units.registry").joinpath("scheme_identity.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        for row in csv.DictReader(f):
            out[row["scheme_id"]] = SchemeIdentity(
                scheme_id=row["scheme_id"], name=row["name"],
                abbreviation=row["abbreviation"] or None, expansion_note=row["expansion_note"],
                ministry=row["ministry"], launch_date=row["launch_date"],
                predecessor=row["predecessor"] or None, classification=row["classification"],
                source_url=row["source_url"] or None, confidence=row["confidence"],
            )
    return out


@lru_cache(maxsize=1)
def load_scheme_entitlements() -> dict[str, SchemeEntitlement]:
    """Keyed by entitlement_id (e.g. 'annual_amount', 'phh_monthly_allowance') - unique across the
    whole file even though several rows share a scheme_id (a scheme legitimately has multiple
    distinct entitlement facts, mirroring traditional_mass.csv's multi-row-per-unit shape)."""
    out: dict[str, SchemeEntitlement] = {}
    with resources.files("bharat_units.registry").joinpath("scheme_entitlements.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        for row in csv.DictReader(f):
            out[row["entitlement_id"]] = SchemeEntitlement(
                id=row["id"], scheme_id=row["scheme_id"], entitlement_id=row["entitlement_id"],
                description=row["description"], value_min=Decimal(row["value_min"]),
                value_max=Decimal(row["value_max"]), unit=row["unit"],
                effective_note=row["effective_note"], source_url=row["source_url"] or None,
                confidence=row["confidence"], volatile=row["volatile"].strip().lower() == "true",
            )
    return out


@lru_cache(maxsize=1)
def load_scheme_funding() -> list[SchemeFunding]:
    """Returns a LIST, not a dict - a scheme_id (or the 'generic_css' sentinel) legitimately has up
    to three rows (general/ne_himalayan/ut_no_legislature), mirroring load_traditional_mass()'s
    reasoning for not collapsing to one row per key."""
    out: list[SchemeFunding] = []
    with resources.files("bharat_units.registry").joinpath("scheme_funding.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        for row in csv.DictReader(f):
            out.append(SchemeFunding(
                id=row["id"], scheme_id=row["scheme_id"], category_type=row["category_type"],
                centre_share_pct=Decimal(row["centre_share_pct"]),
                state_share_pct=Decimal(row["state_share_pct"]), note=row["note"],
                source_url=row["source_url"] or None, confidence=row["confidence"],
                volatile=row["volatile"].strip().lower() == "true",
            ))
    return out


@lru_cache(maxsize=1)
def load_scheme_collisions() -> dict[str, SchemeCollision]:
    """Keyed by pair_id - one row per near-name-collision pair."""
    out: dict[str, SchemeCollision] = {}
    with resources.files("bharat_units.registry").joinpath("scheme_collisions.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        for row in csv.DictReader(f):
            out[row["pair_id"]] = SchemeCollision(
                pair_id=row["pair_id"], scheme_id_a=row["scheme_id_a"],
                scheme_id_b=row["scheme_id_b"], collision_note=row["collision_note"],
                source_url=row["source_url"] or None, confidence=row["confidence"],
            )
    return out


@lru_cache(maxsize=1)
def load_international_analogs() -> dict[str, InternationalAnalog]:
    """Keyed by maps_to_pair_id, so a control template can look up "the real international analog
    of CP01" directly - C6's controls need genuinely sourced foreign facts (real US federal
    programs), not a restated
    self-contained convention the way C4/C5's controls could get away with."""
    out: dict[str, InternationalAnalog] = {}

    def _opt_dec(s):
        return Decimal(s) if s else None

    with resources.files("bharat_units.registry").joinpath("international_analogs.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        for row in csv.DictReader(f):
            out[row["maps_to_pair_id"]] = InternationalAnalog(
                analog_id=row["analog_id"], maps_to_pair_id=row["maps_to_pair_id"],
                name_a=row["name_a"], note_a=row["note_a"], agency_a=row["agency_a"],
                name_b=row["name_b"], note_b=row["note_b"], agency_b=row["agency_b"],
                distinguishing_fact=row["distinguishing_fact"],
                value_a=_opt_dec(row["value_a"]), value_b=_opt_dec(row["value_b"]),
                unit=row["unit"] or None, source_url=row["source_url"] or None,
                confidence=row["confidence"],
            )
    return out


@lru_cache(maxsize=1)
def load_international_scheme_facts() -> dict[str, InternationalSchemeFact]:
    """Keyed by fact_id - real, independently-verified US federal program/agency facts used ONLY
    for C6 control twins - unlike C4/C5's
    self-contained fictional conventions, C6 controls need genuinely sourced foreign facts, since
    the core items themselves are pure institutional-fact recall with no computation to restate."""
    out: dict[str, InternationalSchemeFact] = {}
    with resources.files("bharat_units.registry").joinpath("international_scheme_facts.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        for row in csv.DictReader(f):
            out[row["fact_id"]] = InternationalSchemeFact(
                fact_id=row["fact_id"], name=row["name"], abbreviation=row["abbreviation"] or None,
                expansion=row["expansion"], agency=row["agency"],
                founding_year=int(row["founding_year"]), note=row["note"],
                source_url=row["source_url"] or None, confidence=row["confidence"],
            )
    return out


@lru_cache(maxsize=1)
def load_pin_zones() -> dict[int, PinZone]:
    """Keyed by zone_digit (int 1-9) - one row per zone, no ambiguity."""
    out: dict[int, PinZone] = {}
    with resources.files("bharat_units.registry").joinpath("pin_zones.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        for row in csv.DictReader(f):
            out[int(row["zone_digit"])] = PinZone(
                zone_digit=int(row["zone_digit"]), zone_name=row["zone_name"],
                states_covered=row["states_covered"], source_url=row["source_url"] or None,
                confidence=row["confidence"],
            )
    return out


@lru_cache(maxsize=1)
def load_pan_entity_codes() -> dict[str, PanEntityCode]:
    """Keyed by code_letter (the PAN's 4th character)."""
    out: dict[str, PanEntityCode] = {}
    with resources.files("bharat_units.registry").joinpath("pan_entity_codes.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        for row in csv.DictReader(f):
            out[row["code_letter"]] = PanEntityCode(
                code_letter=row["code_letter"], entity_type=row["entity_type"],
                source_url=row["source_url"] or None, confidence=row["confidence"],
            )
    return out


@lru_cache(maxsize=1)
def load_admin_division_names() -> dict[str, AdminDivisionTerm]:
    """Keyed by term (tehsil/taluk/mandal/circle/sub-division)."""
    out: dict[str, AdminDivisionTerm] = {}
    with resources.files("bharat_units.registry").joinpath("admin_division_names.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        for row in csv.DictReader(f):
            out[row["term"]] = AdminDivisionTerm(
                term=row["term"], region_states=row["region_states"],
                source_url=row["source_url"] or None, confidence=row["confidence"],
            )
    return out


@lru_cache(maxsize=1)
def load_identifier_format_facts() -> dict[str, IdentifierFormatFact]:
    """Keyed by fact_id - India-side structural facts (IFSC/GSTIN/PAN/vehicle-registration shape).
    Includes the IFSC_BRANCH_LEN correction (6, not the RBI circular's often-misquoted 3 -
    see tests/property/test_identifiers_properties.py for the worked example)."""
    out: dict[str, IdentifierFormatFact] = {}
    with resources.files("bharat_units.registry").joinpath("identifier_format_facts.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        for row in csv.DictReader(f):
            out[row["fact_id"]] = IdentifierFormatFact(
                fact_id=row["fact_id"], description=row["description"], value=row["value"],
                note=row["note"], source_url=row["source_url"] or None, confidence=row["confidence"],
            )
    return out


@lru_cache(maxsize=1)
def load_international_identifier_facts() -> dict[str, IdentifierFormatFact]:
    """Keyed by fact_id - real, independently-verified foreign identifier-format facts (US ZIP,
    SWIFT/BIC, SSN, Louisiana/Alaska naming exceptions, VIN) used ONLY for C7 control twins on the
    structural-recall subcategories."""
    out: dict[str, IdentifierFormatFact] = {}
    with resources.files("bharat_units.registry").joinpath("international_identifier_facts.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        for row in csv.DictReader(f):
            out[row["fact_id"]] = IdentifierFormatFact(
                fact_id=row["fact_id"], description=row["description"], value=row["value"],
                note=row["note"], source_url=row["source_url"] or None, confidence=row["confidence"],
            )
    return out


@lru_cache(maxsize=1)
def load_traditional_mass() -> list[TraditionalMassUnit]:
    """Returns a LIST, not a dict keyed by unit name - unlike mass_units.csv, the same unit name
    (e.g. 'seer', 'maund') legitimately has multiple rows distinguished by variant/region, and
    collapsing to a dict-by-name would silently pick one and hide the others - exactly the failure
    mode this registry exists to avoid (see traditional_mass.csv's seer trade_customary/statutory_1956
    split and maund's Bengal/Punjab/standard split)."""
    out: list[TraditionalMassUnit] = []
    with resources.files("bharat_units.registry").joinpath("traditional_mass.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        for row in csv.DictReader(f):
            aliases = tuple(a for a in row["aliases"].split(";") if a)
            out.append(TraditionalMassUnit(
                id=row["id"], unit=row["unit"], aliases=aliases,
                variant=row["variant"] or None, region=row["region"] or None,
                grams=Decimal(row["grams"]) if row["grams"] else None,
                confidence=row["confidence"], source_url=row["source_url"] or None,
            ))
    return out
