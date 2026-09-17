"""Property-based tests for bharat_units.identifiers.

These run BEFORE any C7 item is generated, and specifically validate the Verhoeff tables and the
empirically-determined GSTIN check-digit convention against known-correct real-world examples,
not just internal consistency.
"""
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bharat_units._registry_loader import (
    load_identifier_format_facts,
    load_international_identifier_facts,
)
from bharat_units.errors import ParseError
from bharat_units.identifiers import (
    admin_division_term,
    gstin_check_digit,
    gstin_validate,
    parse_ifsc,
    parse_pan,
    parse_pin_code,
    parse_vehicle_registration,
    verhoeff_check_digit,
    verhoeff_validate,
)

# ---------------------------------------------------------------------------
# Verhoeff: the classic worked example, plus round-trip properties
# ---------------------------------------------------------------------------

def test_verhoeff_classic_worked_example():
    """The textbook Verhoeff example: body '236', check digit '3', full number '2363' valid -
    verified by hand against the canonical tables before this test was written (see the module
    docstring)."""
    assert verhoeff_check_digit("236") == "3"
    assert verhoeff_validate("2363") is True


def test_verhoeff_detects_single_digit_substitution():
    assert verhoeff_validate("2364") is False  # last digit changed 3->4


def test_verhoeff_detects_adjacent_transposition():
    """The property Verhoeff is specifically chosen for over a simple sum-based checksum -
    swapping two adjacent digits must be caught."""
    assert verhoeff_validate("2363") is True
    assert verhoeff_validate("2633") is False  # first two digits of '2363' transposed


def test_verhoeff_rejects_non_digit_input():
    with pytest.raises(ParseError):
        verhoeff_validate("23a3")
    with pytest.raises(ParseError):
        verhoeff_check_digit("23a")


@given(st.text(alphabet="0123456789", min_size=11, max_size=11))
@settings(max_examples=200)
def test_verhoeff_check_digit_round_trips(body):
    check = verhoeff_check_digit(body)
    assert verhoeff_validate(body + check) is True


# ---------------------------------------------------------------------------
# GSTIN: the real worked example (27AABCU9603R1ZN), plus round-trip properties
# ---------------------------------------------------------------------------

def test_gstin_real_worked_example():
    """27AABCU9603R1ZN is a real, independently-cited well-formed GSTIN (Maharashtra state code
    27) - the exact weighting/direction convention was determined empirically against this
    example, not assumed from prose alone."""
    assert gstin_check_digit("27AABCU9603R1Z") == "N"
    assert gstin_validate("27AABCU9603R1ZN") is True


def test_gstin_detects_altered_check_char():
    assert gstin_validate("27AABCU9603R1ZA") is False


def test_gstin_detects_altered_body_char():
    assert gstin_validate("28AABCU9603R1ZN") is False  # state code changed 27->28


def test_gstin_rejects_wrong_length():
    with pytest.raises(ParseError):
        gstin_check_digit("27AABCU9603R1")  # 13 chars, not 14
    with pytest.raises(ParseError):
        gstin_validate("27AABCU9603R1Z")  # 14 chars, not 15


def test_gstin_rejects_unrecognized_character():
    with pytest.raises(ParseError):
        gstin_check_digit("27AABCU9603R1@")


@given(st.text(alphabet="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=14, max_size=14))
@settings(max_examples=200)
def test_gstin_check_digit_round_trips(body14):
    check = gstin_check_digit(body14)
    assert gstin_validate(body14 + check) is True


# ---------------------------------------------------------------------------
# Structural parsers
# ---------------------------------------------------------------------------

def test_parse_pin_code_delhi_example():
    info = parse_pin_code("110001")
    assert info.zone_digit == 1
    assert "Delhi" in info.states_covered


def test_parse_pin_code_army_zone():
    info = parse_pin_code("900123")
    assert info.zone_digit == 9
    assert "Army" in info.zone_name


def test_parse_pin_code_rejects_bad_length():
    with pytest.raises(ParseError):
        parse_pin_code("1100")


def test_parse_pin_code_rejects_zero_zone_digit():
    """0 is not a valid PIN zone - zones are 1-9 (8 regional + 1 Army Postal)."""
    with pytest.raises(ParseError):
        parse_pin_code("012345")


def test_parse_ifsc_shape():
    info = parse_ifsc("HDFC0001234")
    assert info.bank_code == "HDFC"
    assert info.reserved_char == "0"
    assert info.branch_code == "001234"
    assert len(info.branch_code) == 6  # IFSC's true branch-code length, verified below


def test_parse_ifsc_rejects_nonzero_fifth_char():
    with pytest.raises(ParseError):
        parse_ifsc("HDFC1001234")


def test_parse_ifsc_rejects_wrong_length():
    with pytest.raises(ParseError):
        parse_ifsc("HDFC000123")  # 10 chars


def test_parse_ifsc_rejects_non_alpha_bank_code():
    with pytest.raises(ParseError):
        parse_ifsc("HD1C0001234")


def test_parse_pan_individual():
    info = parse_pan("ABCPD1234E")
    assert info.entity_type_code == "P"
    assert info.entity_type_name == "Individual (Person)"
    assert info.surname_initial == "D"


def test_parse_pan_company():
    info = parse_pan("ABCCD1234E")
    assert info.entity_type_code == "C"
    assert info.entity_type_name == "Company"


def test_parse_pan_rejects_unrecognized_entity_code():
    with pytest.raises(ParseError):
        parse_pan("ABCXD1234E")  # 'X' is not a recognized entity-type code


def test_parse_pan_rejects_wrong_length():
    with pytest.raises(ParseError):
        parse_pan("ABCPD123E")  # 9 chars


def test_parse_pan_rejects_non_alpha_prefix():
    with pytest.raises(ParseError):
        parse_pan("AB1PD1234E")


def test_parse_pan_rejects_non_digit_numeric_series():
    with pytest.raises(ParseError):
        parse_pan("ABCPD12A4E")


def test_parse_vehicle_registration_delhi():
    info = parse_vehicle_registration("DL01AB1234")
    assert info.state_code == "DL"
    assert info.rto_code == "01"
    assert info.series == "AB"
    assert info.number == "1234"


def test_parse_vehicle_registration_no_series():
    info = parse_vehicle_registration("MH121234")
    assert info.state_code == "MH"
    assert info.rto_code == "12"
    assert info.series == ""
    assert info.number == "1234"


def test_parse_vehicle_registration_rejects_o_and_i_in_series():
    with pytest.raises(ParseError):
        parse_vehicle_registration("DL01OI1234")


def test_parse_vehicle_registration_rejects_bad_length():
    with pytest.raises(ParseError):
        parse_vehicle_registration("D")  # far too short
    with pytest.raises(ParseError):
        parse_vehicle_registration("DL01AB123456")  # too long (12 chars)


def test_parse_vehicle_registration_rejects_non_alpha_state_code():
    with pytest.raises(ParseError):
        parse_vehicle_registration("D101AB1234")


def test_parse_vehicle_registration_rejects_non_digit_rto_code():
    with pytest.raises(ParseError):
        parse_vehicle_registration("DLABAB1234")


def test_parse_vehicle_registration_rejects_non_digit_serial():
    with pytest.raises(ParseError):
        parse_vehicle_registration("DL01AB12AB")


def test_admin_division_term_lookup():
    assert admin_division_term("Andhra Pradesh") == "mandal"
    assert admin_division_term("Uttar Pradesh") == "tehsil"
    assert admin_division_term("Tamil Nadu") == "taluk"


def test_admin_division_term_unrecognized_raises():
    with pytest.raises(ParseError):
        admin_division_term("Atlantis")


# ---------------------------------------------------------------------------
# identifier_format_facts.csv / international_identifier_facts.csv - the registries backing both
# the core-item generator and generate_controls.py's C7 templates
# ---------------------------------------------------------------------------

def test_identifier_format_facts_ifsc_branch_length_correction():
    """IFSC's branch-code portion is 6 characters, not the commonly misquoted 3
    (4+1+6=11 matches IFSC's known total length; 4+1+3=8 does not)."""
    facts = load_identifier_format_facts()
    assert facts["IFSC_BRANCH_LEN"].value == "6"
    assert facts["IFSC_TOTAL_LEN"].value == "11"


def test_identifier_format_facts_pan_surname_position():
    assert load_identifier_format_facts()["PAN_SURNAME_POSITION"].value == "5"


def test_identifier_format_facts_gstin_length():
    assert load_identifier_format_facts()["GSTIN_TOTAL_LEN"].value == "15"


def test_international_identifier_facts_swift_bic_and_ssn():
    facts = load_international_identifier_facts()
    assert facts["SWIFT_BIC_TOTAL_LEN"].value == "11"
    assert facts["SSN_TOTAL_LEN"].value == "9"
    assert facts["US_COUNTY_LOUISIANA"].value == "parish"
    assert facts["US_COUNTY_ALASKA"].value == "borough"
