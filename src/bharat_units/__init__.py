"""bharat_units — Indian numeral, weight, land-unit, and calendar normalizer.

Reference implementation for the Bharat Knowledge Probe (BKP-500) benchmark: the same library
the dataset's gold answers were computed against, covering Indian numeral magnitudes/digit
grouping, traditional mass/land units, the fiscal year, crop seasons, and structural identifiers
(PAN/GSTIN/IFSC/PIN). See the submodules (numerals, mass, fiscal, seasons, identifiers) for the
per-domain design rationale.
"""

from bharat_units.errors import BharatUnitsError, ParseError, ScaleError

__all__ = ["BharatUnitsError", "ParseError", "ScaleError"]
__version__ = "0.1.0"
