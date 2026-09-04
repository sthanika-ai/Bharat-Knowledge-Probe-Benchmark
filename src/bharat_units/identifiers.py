"""Structural/format identifiers: PIN code, IFSC, PAN, GSTIN, vehicle registration, Aadhaar.

Two of these (GSTIN, Aadhaar) involve REAL COMPUTABLE CHECKSUM ALGORITHMS - the Verhoeff algorithm
and a Luhn-mod-36 variant - unlike the other structural-identifier categories, which needed none.

The Verhoeff tables below are transcribed VERBATIM from the canonical dihedral-group-D5
construction (cross-checked against multiple independent sources) - a single wrong table entry
would be the worst kind of bug here (silently wrong for some, not all, inputs). Both
verhoeff_check_digit and verhoeff_validate are independently verified in
tests/property/test_identifiers_properties.py against the classic worked example (2363).

The GSTIN check-digit algorithm's exact weighting/direction was NOT taken from prose description
alone (sources are inconsistent on this point) - it was determined empirically by testing against
a real, independently-cited well-formed GSTIN
(27AABCU9603R1ZN) until the implementation reproduced the correct check character 'N'. The
confirmed convention: characters are processed LEFT TO RIGHT, with alternating weights of 1 then 2
starting with 1 on the leftmost (first) character.
"""
from __future__ import annotations

from dataclasses import dataclass

from bharat_units._registry_loader import (
    load_admin_division_names,
    load_pan_entity_codes,
    load_pin_zones,
)
from bharat_units.errors import ParseError

# ---------------------------------------------------------------------------
# Verhoeff algorithm (Aadhaar's 12th-digit checksum)
# ---------------------------------------------------------------------------

_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]
_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def _require_digits(s: str, label: str) -> None:
    if not s.isdigit():
        raise ParseError(f"{label} must be all digits, got {s!r}")


def verhoeff_validate(number: str) -> bool:
    """True iff `number`'s last digit is the correct Verhoeff check digit for the rest."""
    _require_digits(number, "verhoeff number")
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c == 0


def verhoeff_check_digit(body: str) -> str:
    """The single Verhoeff check digit for `body` (the number WITHOUT its check digit) -
    appending it to `body` makes verhoeff_validate(body + digit) True.
    """
    _require_digits(body, "verhoeff body")
    c = 0
    for i, ch in enumerate(reversed(body)):
        c = _D[c][_P[(i + 1) % 8][int(ch)]]
    return str(_INV[c])


# ---------------------------------------------------------------------------
# GSTIN Luhn-mod-36 check character
# ---------------------------------------------------------------------------

_GSTIN_CODE = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _gstin_char_value(ch: str) -> int:
    try:
        return _GSTIN_CODE.index(ch.upper())
    except ValueError:
        raise ParseError(f"unrecognized GSTIN character {ch!r} (must be 0-9 or A-Z)") from None


def gstin_check_digit(gstin_14: str) -> str:
    """The 15th (check) character for a 14-character GSTIN body, characters 1-14.

    Empirically confirmed convention (module docstring): left-to-right, alternating weights
    1, 2, 1, 2, ... starting with 1 on the leftmost character; each weighted value is reduced
    (floor-div-36 + mod-36) before summing; the check character is whichever of 0-35 makes the
    total (including the check value) a multiple of 36.
    """
    if len(gstin_14) != 14:
        raise ParseError(f"GSTIN body must be exactly 14 characters, got {len(gstin_14)}")
    total = 0
    factor = 1
    for ch in gstin_14:
        v = _gstin_char_value(ch) * factor
        v = (v // 36) + (v % 36)
        total += v
        factor = 2 if factor == 1 else 1
    check_value = (36 - (total % 36)) % 36
    return _GSTIN_CODE[check_value]


def gstin_validate(gstin_15: str) -> bool:
    """True iff `gstin_15`'s 15th character is the correct check character for the first 14."""
    if len(gstin_15) != 15:
        raise ParseError(f"GSTIN must be exactly 15 characters, got {len(gstin_15)}")
    return gstin_check_digit(gstin_15[:14]) == gstin_15[14].upper()


# ---------------------------------------------------------------------------
# Structural parsers: PIN code, IFSC, PAN, vehicle registration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PinCodeInfo:
    zone_digit: int
    zone_name: str
    states_covered: str


def parse_pin_code(pin: str):
    if len(pin) != 6 or not pin.isdigit():
        raise ParseError(f"PIN code must be exactly 6 digits, got {pin!r}")
    zones = load_pin_zones()
    zone_digit = int(pin[0])
    if zone_digit not in zones:
        raise ParseError(f"unrecognized PIN zone digit {zone_digit!r}")
    z = zones[zone_digit]
    return PinCodeInfo(zone_digit=zone_digit, zone_name=z.zone_name, states_covered=z.states_covered)


@dataclass(frozen=True)
class IfscInfo:
    bank_code: str
    reserved_char: str
    branch_code: str


def parse_ifsc(ifsc: str):
    if len(ifsc) != 11:
        raise ParseError(f"IFSC must be exactly 11 characters, got {len(ifsc)}")
    bank_code, reserved_char, branch_code = ifsc[:4], ifsc[4], ifsc[5:]
    if not bank_code.isalpha():
        raise ParseError(f"IFSC bank code (first 4 chars) must be alphabetic, got {bank_code!r}")
    if reserved_char != "0":
        raise ParseError(f"IFSC's 5th character must be the reserved '0', got {reserved_char!r}")
    return IfscInfo(bank_code=bank_code, reserved_char=reserved_char, branch_code=branch_code)


@dataclass(frozen=True)
class PanInfo:
    series: str
    entity_type_code: str
    entity_type_name: str
    surname_initial: str
    numeric_series: str
    check_char: str


def parse_pan(pan: str):
    if len(pan) != 10:
        raise ParseError(f"PAN must be exactly 10 characters, got {len(pan)}")
    series, entity_code, surname_initial = pan[0:3], pan[3], pan[4]
    numeric_series, check_char = pan[5:9], pan[9]
    if not series.isalpha() or not entity_code.isalpha() or not surname_initial.isalpha():
        raise ParseError(f"PAN's first 5 characters must be alphabetic, got {pan[:5]!r}")
    if not numeric_series.isdigit():
        raise ParseError(f"PAN's numeric series (positions 6-9) must be digits, got {numeric_series!r}")
    codes = load_pan_entity_codes()
    if entity_code not in codes:
        raise ParseError(f"unrecognized PAN entity-type code {entity_code!r}")
    return PanInfo(
        series=series, entity_type_code=entity_code, entity_type_name=codes[entity_code].entity_type,
        surname_initial=surname_initial, numeric_series=numeric_series, check_char=check_char,
    )


@dataclass(frozen=True)
class VehicleRegInfo:
    state_code: str
    rto_code: str
    series: str
    number: str


def parse_vehicle_registration(reg: str):
    if len(reg) < 8 or len(reg) > 10:
        raise ParseError(f"vehicle registration must be 8-10 characters, got {len(reg)}")
    state_code, rto_code = reg[0:2], reg[2:4]
    if not state_code.isalpha():
        raise ParseError(f"vehicle registration state code must be alphabetic, got {state_code!r}")
    if not rto_code.isdigit():
        raise ParseError(f"vehicle registration RTO code must be numeric, got {rto_code!r}")
    number = reg[-4:]
    series = reg[4:-4]
    if not number.isdigit():
        raise ParseError(f"vehicle registration serial number must be 4 digits, got {number!r}")
    if series and (not series.isalpha() or "O" in series.upper() or "I" in series.upper()):
        raise ParseError(f"vehicle registration series {series!r} must be alphabetic and exclude O/I")
    return VehicleRegInfo(state_code=state_code, rto_code=rto_code, series=series, number=number)


def admin_division_term(state_or_region: str) -> str:
    """Given a state/region description, find the matching admin_division_names.csv term - a
    simple substring lookup over the registry's region_states field, used by the generator/controls
    rather than a general-purpose function (the registry is small and free-text, not a lookup key)."""
    terms = load_admin_division_names()
    for term, row in terms.items():
        if state_or_region.lower() in row.region_states.lower():
            return term
    raise ParseError(f"no admin_division_names.csv row mentions {state_or_region!r}")
