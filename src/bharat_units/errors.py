"""Shared exception hierarchy for bharat_units.

Design rule: the library refuses to guess. A malformed or ambiguous input raises one of these,
never a silently wrong number. A silent misparse is a more severe defect than a raised exception.
"""


class BharatUnitsError(Exception):
    """Base class for every error this package raises on purpose."""


class ParseError(BharatUnitsError):
    """Raised when a numeral string cannot be parsed with confidence.

    Never raised for input that's merely unusual - only for input where guessing would risk a
    silent wrong answer (unrecognized scale word, ambiguous compound expression, empty input, etc).
    """


class ScaleError(BharatUnitsError):
    """Raised for an unknown or mismatched magnitude scale term (e.g. a typo'd 'crores' variant
    not in the alias table, or an attempt to convert between an Indian-system and
    international-system unit without going through a common base - which to_scale() actually
    supports, so this mostly fires on genuinely unrecognized terms)."""


class AmbiguousUnitError(BharatUnitsError):
    """Raised when a unit has more than one attested value and the caller hasn't disambiguated
    (e.g. maund without a region=, or the bare seer without specifying trade_customary vs
    statutory_1956) - the library refuses to silently pick one convention over another. Also
    raised (always, with no disambiguating argument that would help) for units with NO defensible
    scalar value at all (candy, peti)."""
