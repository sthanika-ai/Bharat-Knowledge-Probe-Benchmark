"""Tests for the C6 scheme registries (scheme_identity/entitlements/funding/collisions,
international_analogs). C6 has no bharat_units computation module to test - gold values are
direct registry lookups, so these tests verify the registries themselves load correctly and
carry the key verified facts, mirroring how test_seasons_properties.py tests
load_msp_policy_facts() directly for C4's equivalent case.
"""
from decimal import Decimal

from bharat_units._registry_loader import (
    load_international_analogs,
    load_international_scheme_facts,
    load_scheme_collisions,
    load_scheme_entitlements,
    load_scheme_funding,
    load_scheme_identity,
)

# ---------------------------------------------------------------------------
# scheme_identity.csv
# ---------------------------------------------------------------------------

def test_mgnrega_renamed_from_nrega():
    s = load_scheme_identity()["S01"]
    assert s.abbreviation == "MGNREGA"
    assert s.launch_date == "2009-10-02"
    assert "NREGA" in s.predecessor


def test_pmjay_launch_date_and_ministry():
    s = load_scheme_identity()["S02"]
    assert s.abbreviation == "PM-JAY"
    assert s.launch_date == "2018-09-23"
    assert s.ministry == "Health and Family Welfare"


def test_pmay_g_predecessor_is_indira_awaas_yojana():
    s = load_scheme_identity()["S03"]
    assert s.predecessor == "Indira Awaas Yojana"
    assert s.launch_date == "2016-04-01"


def test_pm_kisan_is_central_sector_not_centrally_sponsored():
    s = load_scheme_identity()["S05"]
    assert s.classification == "Central Sector (100% Centre-funded)"


def test_ujjwala_and_ujjawala_are_different_ministries():
    """The collision itself, asserted directly at the identity level."""
    pmuy = load_scheme_identity()["S12"]
    ujjawala = load_scheme_identity()["S13"]
    assert pmuy.ministry == "Petroleum and Natural Gas"
    assert ujjawala.ministry == "Women and Child Development"
    assert pmuy.ministry != ujjawala.ministry


def test_stand_up_india_and_startup_india_launch_dates_are_close_but_different():
    stand_up = load_scheme_identity()["S14"]
    startup = load_scheme_identity()["S15"]
    assert stand_up.launch_date == "2016-04-05"
    assert startup.launch_date == "2016-01-16"
    assert stand_up.launch_date != startup.launch_date


def test_day_nrlm_and_day_nulm_share_the_day_prefix_but_different_ministries():
    nrlm = load_scheme_identity()["S16"]
    nulm = load_scheme_identity()["S17"]
    assert "Rural Livelihoods" in nrlm.name
    assert "Urban Livelihoods" in nulm.name
    assert nrlm.ministry == "Rural Development"
    assert nulm.ministry == "Housing and Urban Affairs"


def test_all_scheme_identity_rows_have_a_source():
    for s in load_scheme_identity().values():
        assert s.source_url, f"{s.scheme_id} ({s.name}) has no source_url"


# ---------------------------------------------------------------------------
# scheme_entitlements.csv
# ---------------------------------------------------------------------------

def test_pm_kisan_amount_and_installments():
    e = load_scheme_entitlements()
    assert e["annual_amount"].value_min == e["annual_amount"].value_max == Decimal(6000)
    assert e["installment_count"].value_min == Decimal(3)
    assert e["installment_amount"].value_min == Decimal(2000)


def test_pm_jay_coverage_amount():
    e = load_scheme_entitlements()["coverage_amount"]
    assert e.value_min == e.value_max == Decimal(500000)


def test_nfsa_phh_and_aay_entitlements():
    e = load_scheme_entitlements()
    assert e["phh_monthly_allowance"].value_min == Decimal(5)
    assert e["aay_monthly_allowance"].value_min == Decimal(35)


def test_nfsa_aay_entitlement_flags_the_pending_2026_amendment():
    """The volatility risk must survive into the registry row itself, not just design notes -
    a reviewer reading this row alone should see the risk."""
    e = load_scheme_entitlements()["aay_monthly_allowance"]
    assert "2026" in e.effective_note
    assert e.volatile is True


def test_mgnrega_days_guarantee():
    assert load_scheme_entitlements()["days_guarantee"].value_min == Decimal(100)


def test_pmfby_premium_rates():
    e = load_scheme_entitlements()
    assert e["premium_kharif"].value_min == Decimal(2)
    assert e["premium_rabi"].value_min == Decimal("1.5")
    assert e["premium_commercial_horticultural"].value_min == Decimal(5)


def test_mudra_tiers_are_contiguous_and_increasing():
    e = load_scheme_entitlements()
    shishu, kishor, tarun, tarun_plus = (e["shishu_tier"], e["kishor_tier"], e["tarun_tier"],
                                         e["tarun_plus_tier"])
    assert shishu.value_max == kishor.value_min
    assert kishor.value_max == tarun.value_min
    assert tarun.value_max == tarun_plus.value_min
    assert tarun_plus.value_max == Decimal(2000000)


def test_tarun_plus_is_the_newest_tier():
    e = load_scheme_entitlements()["tarun_plus_tier"]
    assert "2024" in e.effective_note


# ---------------------------------------------------------------------------
# scheme_funding.csv
# ---------------------------------------------------------------------------

def test_generic_css_funding_pattern():
    rows = {r.category_type: r for r in load_scheme_funding() if r.scheme_id == "generic_css"}
    assert rows["general"].centre_share_pct == Decimal(60)
    assert rows["ne_himalayan"].centre_share_pct == Decimal(90)
    assert rows["ut_no_legislature"].centre_share_pct == Decimal(100)


def test_pmjay_general_share_matches_the_generic_css_pattern_but_pmfby_does_not():
    """The 90:10 NE/Himalayan split recurs everywhere, but the "general" split is NOT one universal
    number: PM-JAY's is the generic CSS 60:40, while PMFBY's own crop-insurance-premium subsidy
    split is 50:50 for general states - a real, scheme-specific exception worth testing precisely
    rather than assuming every scheme follows the same generic ratio."""
    funding = load_scheme_funding()
    pmfby_general = next(r for r in funding if r.scheme_id == "S04" and r.category_type == "general")
    pmjay_general = next(r for r in funding if r.scheme_id == "S02" and r.category_type == "general")
    generic_general = next(r for r in funding if r.scheme_id == "generic_css" and r.category_type == "general")
    assert pmjay_general.centre_share_pct == generic_general.centre_share_pct == Decimal(60)
    assert pmfby_general.centre_share_pct == Decimal(50)
    assert pmfby_general.centre_share_pct != generic_general.centre_share_pct


def test_pmfby_and_generic_css_share_the_same_ne_himalayan_ratio():
    """Unlike the general-category split, the NE/Himalayan 90:10 ratio genuinely IS shared across
    PMFBY and the generic convention - the one part of the pattern that does generalize."""
    funding = load_scheme_funding()
    pmfby_ne = next(r for r in funding if r.scheme_id == "S04" and r.category_type == "ne_himalayan")
    generic_ne = next(r for r in funding if r.scheme_id == "generic_css" and r.category_type == "ne_himalayan")
    assert pmfby_ne.centre_share_pct == generic_ne.centre_share_pct == Decimal(90)


def test_centre_and_state_shares_sum_to_100_for_every_row():
    for r in load_scheme_funding():
        assert r.centre_share_pct + r.state_share_pct == Decimal(100), r.id


# ---------------------------------------------------------------------------
# scheme_collisions.csv + international_analogs.csv
# ---------------------------------------------------------------------------

def test_every_collision_pair_resolves_to_two_real_schemes():
    identity = load_scheme_identity()
    for pair in load_scheme_collisions().values():
        assert pair.scheme_id_a in identity
        assert pair.scheme_id_b in identity


def test_every_collision_pair_has_an_international_analog():
    collisions = load_scheme_collisions()
    analogs = load_international_analogs()
    for pair_id in collisions:
        assert pair_id in analogs, f"{pair_id} has no international analog for control-twin use"


def test_international_scheme_facts_load_and_carry_verified_founding_years():
    """Used only by generate_controls.py's C6 templates - never referenced from the core item
    generator, so it needs its own direct test rather than relying on incidental coverage from the
    core-item generation path."""
    facts = load_international_scheme_facts()
    assert facts["IF01"].abbreviation == "SNAP"
    assert facts["IF01"].founding_year == 1964
    assert facts["IF02"].abbreviation == "TANF"
    assert facts["IF02"].founding_year == 1996
    assert facts["IF03"].name == "Medicare"
    assert facts["IF03"].founding_year == 1965
    assert facts["IF04"].name == "Medicaid"
    assert facts["IF04"].founding_year == 1965
    assert facts["IF05"].abbreviation == "SBA"
    assert facts["IF05"].founding_year == 1953


def test_sba_loan_analog_has_distinct_real_dollar_caps():
    """The one analog pair carrying numeric entitlement-style facts (loan caps), verified this
    session against US SBA sources - not invented placeholder numbers."""
    ia = load_international_analogs()["CP02"]
    assert ia.value_a == Decimal(5000000)
    assert ia.value_b == Decimal(20000000)
    assert ia.value_a != ia.value_b
