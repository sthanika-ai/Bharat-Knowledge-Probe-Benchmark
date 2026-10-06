"""Mass conversion: quintal/tonne/kg arithmetic and traditional Indian mass units.

This module imports bharat_units.numerals.scale_factor directly for the tonne_ambiguity
subcategory - "lakh tonne" / MMT is a scale-word composed with a mass unit, not new arithmetic.

Design rule: a unit with more than one attested value refuses to guess. convert_traditional()
raises AmbiguousUnitError for maund without region=, seer without variant=, and unconditionally
for candy/peti (no defensible scalar value exists in the registry at all).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from bharat_units._registry_loader import load_mass_units, load_traditional_mass
from bharat_units.errors import AmbiguousUnitError, ScaleError
from bharat_units.numerals import scale_factor


def _grams_per(unit: str) -> Decimal:
    units = load_mass_units()
    if unit not in units:
        raise ScaleError(f"unrecognized mass unit {unit!r}")
    return units[unit].grams


def to_mass_scale(value: Decimal | int | float | str, from_unit: str, to_unit: str) -> Decimal:
    """Convert a QUANTITY of mass from one unit to another.
    to_mass_scale(5, "quintal", "kilogram") -> Decimal(500)  (5 quintal = 500 kg)
    """
    value = Decimal(str(value))
    return value * _grams_per(from_unit) / _grams_per(to_unit)


def rate_convert(value: Decimal | int | float | str, from_unit: str, to_unit: str) -> Decimal:
    """Convert a RATE per unit mass (e.g. rupees per quintal) to a rate per a different unit mass.

    This is the mirror-image ratio of to_mass_scale, not the same formula: a price of X per
    LARGER unit is a SMALLER price per smaller unit, so the grams ratio is inverted relative to a
    plain quantity conversion. rate_convert(2000, "quintal", "kilogram") -> 20 (Rs.2000/quintal is
    Rs.20/kg, since 1 quintal = 100 kg). This is exactly the "MSP quoted in Rs/quintal -> Rs/kg"
    arithmetic the quintal subcategory tests, and the classic error is applying to_mass_scale's
    ratio here instead (multiplying instead of dividing, or vice versa).
    """
    value = Decimal(str(value))
    return value * _grams_per(to_unit) / _grams_per(from_unit)


def tonne_from_scaled(value: Decimal | int | float | str, scale_word: str | None = None) -> Decimal:
    """Compose an Indian/international scale word with 'tonne' - the tonne_ambiguity subcategory's
    core operation. tonne_from_scaled(3.2, "lakh") -> 3.2 lakh tonnes, as plain tonnes.
    tonne_from_scaled(357.73, "million") -> 357.73 million tonnes, as plain tonnes.
    With scale_word=None, value is already in plain tonnes (a no-op, useful for uniform call sites).
    """
    value = Decimal(str(value))
    if scale_word is None:
        return value
    return value * scale_factor(scale_word)


@dataclass(frozen=True)
class ConversionResult:
    value: Decimal
    unit: str
    assumed: dict = field(default_factory=dict)
    confidence: str = "high"
    warnings: tuple[str, ...] = ()
    source: str | None = None


def _find_traditional(unit: str, variant: str | None, region: str | None):
    rows = [r for r in load_traditional_mass() if unit in (r.unit, *r.aliases)]
    if not rows:
        raise ScaleError(f"unrecognized traditional mass unit {unit!r}")
    if variant is not None:
        rows = [r for r in rows if r.variant == variant]
    if region is not None:
        rows = [r for r in rows if r.region == region]
    return rows


def convert_traditional(
    value: Decimal | int | float | str,
    from_unit: str,
    to_unit: str = "gram",
    variant: str | None = None,
    region: str | None = None,
    strict: bool = True,
) -> ConversionResult:
    """Convert a quantity of a traditional Indian mass unit to a standard unit (default: grams).

    Refuses to guess, mirroring area.py's planned convert_area contract:
      - candy, peti: ALWAYS raises AmbiguousUnitError - no defensible scalar value exists at all,
        regardless of variant=/region=/strict=.
      - maund: raises unless region= is given (Bengal/Punjab/standard are all attested, materially
        different values).
      - seer: raises unless variant= is given ('trade_customary' 0.9331 kg or 'statutory_1956'
        0.933104304 kg, i.e. 80 tola under the Standards of Weights and Measures Act, 1956 /
        Act No. 89 of 1956 - the two variants agree to within rounding; an earlier, incorrect
        1.25 kg figure for the statutory value has been corrected).
      - Pass strict=False to get a best-effort ConversionResult instead of raising - it always
        carries `assumed`, `confidence`, and `warnings` so a caller can't launder the uncertainty
        away silently. strict=False still raises for candy/peti - there is no "best effort" available
        when the registry itself has no scalar row to fall back to.
    """
    rows = _find_traditional(from_unit, variant, region)
    valued = [r for r in rows if r.grams is not None]

    if not valued:
        # No row has a scalar value at all (candy, peti). strict=
        # only controls whether an AMBIGUOUS set of valued rows gets a best-effort pick below;
        # when there is nothing to fall back to, strict=False can't manufacture one, so this
        # raises unconditionally rather than special-casing by unit name (which wouldn't cover a
        # future no-value unit added to the registry under some other name).
        reasons = {r.confidence for r in rows}
        raise AmbiguousUnitError(
            f"{from_unit!r} has no defensible scalar value"
            + (f" (variant={variant!r})" if variant else "")
            + (f" (region={region!r})" if region else "")
            + f" - registry confidence: {reasons or 'unresolvable'}"
        )

    if len(valued) > 1 and strict:
        distinct_values = {r.grams for r in valued}
        if len(distinct_values) > 1:
            options = ", ".join(f"{r.variant or r.region}={r.grams}g" for r in valued)
            raise AmbiguousUnitError(
                f"{from_unit!r} resolves to {len(distinct_values)} distinct values ({options}); "
                f"pass variant= or region= to disambiguate, or strict=False for a best-effort pick"
            )

    row = valued[0]
    value_dec = Decimal(str(value))
    grams_total = value_dec * row.grams
    result_value = grams_total / _grams_per(to_unit) if to_unit != "gram" else grams_total

    assumed = {}
    warnings = []
    if variant is None and row.variant:
        assumed["variant"] = row.variant
        warnings.append(f"assumed variant={row.variant!r}; other variants exist")
    if region is None and row.region:
        assumed["region"] = row.region
        warnings.append(f"assumed region={row.region!r}; other regions exist")

    return ConversionResult(
        value=result_value, unit=to_unit, assumed=assumed,
        confidence=row.confidence, warnings=tuple(warnings), source=row.source_url,
    )
