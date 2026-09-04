"""Indian numeral parsing, formatting, scale conversion, and word-form conversion.

All scale and colloquial-modifier constants are loaded from the sourced CSVs in registry/ -
nothing here is a hand-typed magic number.

Public API: parse_indian_number, format_indian_grouping, scale_factor, to_scale, to_words, from_words.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation, getcontext

from bharat_units._registry_loader import load_colloquial_modifiers, load_scale_terms
from bharat_units.errors import ParseError, ScaleError

getcontext().prec = 50  # scale/colloquial ratios are all exact in base 10; ample headroom, no rounding

# ---------------------------------------------------------------------------
# Scale terms: aliases (plurals, abbreviations) -> canonical term in numeral_scales.csv
# ---------------------------------------------------------------------------

_SCALE_ALIASES: dict[str, str] = {
    "hundred": "hundred", "hundreds": "hundred",
    "thousand": "thousand", "thousands": "thousand", "k": "thousand",
    "lakh": "lakh", "lakhs": "lakh", "lac": "lakh", "lacs": "lakh",
    "crore": "crore", "crores": "crore", "cr": "crore",
    "arab": "arab", "arabs": "arab",
    "kharab": "kharab", "kharabs": "kharab",
    "neel": "neel", "nil": "neel",
    "padma": "padma", "padmas": "padma",
    "shankh": "shankh", "shankhs": "shankh",
    "million": "million", "millions": "million", "mn": "million",
    "billion": "billion", "billions": "billion", "bn": "billion",
    "trillion": "trillion", "trillions": "trillion", "tn": "trillion",
    "quadrillion": "quadrillion", "quadrillions": "quadrillion",
    "quintillion": "quintillion", "quintillions": "quintillion",
    "sextillion": "sextillion", "sextillions": "sextillion",
}

_ENGLISH_FRACTION_PHRASES: dict[str, Decimal] = {
    "one and a half": Decimal("1.5"),
    "one and a quarter": Decimal("1.25"),
    "two and a half": Decimal("2.5"),
    "three quarters of a": Decimal("0.75"),
    "three quarters": Decimal("0.75"),
    "quarter of a": Decimal("0.25"),
    "half a": Decimal("0.5"),
}

_CURRENCY_RE = re.compile(r"(?:₹|rs\.?|inr|rupees)\s*", re.IGNORECASE)
_PLAIN_NUMBER_RE = re.compile(r"^-?\d+(\.\d+)?$")

# Alternate transliterations attested for the same colloquial_modifiers.csv row (see that CSV's
# own source_note on CM04: "Both 'dhai' and 'adhai' transliterations are in use.").
_COLLOQUIAL_ALIASES: dict[str, str] = {"adhai": "dhai"}


def _resolve_scale_alias(term: str) -> str:
    canonical = _SCALE_ALIASES.get(term.lower().strip())
    if canonical is None:
        raise ScaleError(f"unrecognized scale term {term!r}")
    return canonical


def scale_factor(*terms: str) -> Decimal:
    """Product of the scale factors for one or more scale terms.

    scale_factor('lakh') = 10**5. scale_factor('lakh', 'crore') = 10**12 - the "X lakh crore"
    compound idiom used for GDP/budget-scale figures (e.g. "1.2 lakh crore" = 1.2e12).
    """
    scales = load_scale_terms()
    total_power = 0
    for t in terms:
        canonical = _resolve_scale_alias(t)
        total_power += scales[canonical].power_of_10
    return Decimal(10) ** total_power


def to_scale(value: Decimal | int | float | str, from_unit: str, to_unit: str) -> Decimal:
    """Convert `value` (denominated in `from_unit`) into `to_unit`.

    to_scale(Decimal("1.2"), "lakh", "crore") -> Decimal("0.012")
    to_scale(Decimal("120"), "crore", "million") -> Decimal("1200")
    """
    value = Decimal(str(value))
    return value * scale_factor(from_unit) / scale_factor(to_unit)


# ---------------------------------------------------------------------------
# format_indian_grouping
# ---------------------------------------------------------------------------

def format_indian_grouping(n: Decimal | int | float | str) -> str:
    """Render `n` with Indian digit grouping: the rightmost 3 digits together, then pairs of 2
    working left from there. 1234567 -> "12,34,567". 100000 -> "1,00,000". 100 -> "100".
    """
    d = Decimal(str(n))
    sign = "-" if d < 0 else ""
    d = abs(d)
    int_part = int(d)
    frac_part = d - int_part

    s = str(int_part)
    if len(s) <= 3:
        grouped = s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        groups: list[str] = []
        while rest:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        grouped = ",".join(groups + [last3])

    if frac_part:
        frac_str = format(frac_part, "f")[2:]  # drop the leading "0."
        grouped += "." + frac_str
    return sign + grouped


# ---------------------------------------------------------------------------
# parse_indian_number
# ---------------------------------------------------------------------------

def parse_indian_number(s: str) -> Decimal:
    """Parse an Indian-context numeral string into a Decimal.

    Handles, in order of attempt:
      1. A plain grouped/ungrouped number, with optional currency prefix: "₹1,20,000", "1,234,567".
      2. An English fraction phrase + scale word: "one and a half crore".
      3. A colloquial Hindi modifier + scale word: "sava lakh", "dedh crore", "paune lakh", "adhai crore".
      4. One or more "<number> <scale-word>" chunks, additive across chunks ("5 cr 40 lakh"),
         multiplicative across CONSECUTIVE scale words with no number between them
         ("1.2 lakh crore" = 1.2 * lakh * crore).

    Raises ParseError rather than guessing when the input doesn't match a known pattern - see
    errors.py for why a raised exception beats a silent misparse.
    """
    original = s
    cleaned = _CURRENCY_RE.sub("", s.strip().lower())
    cleaned = cleaned.replace(",", "").strip()
    if not cleaned:
        raise ParseError(f"empty input: {original!r}")

    if _PLAIN_NUMBER_RE.match(cleaned):
        try:
            return Decimal(cleaned)
        except InvalidOperation as e:
            raise ParseError(f"could not parse plain number from {original!r}") from e

    # English fraction phrase, longest match first, e.g. "one and a half crore"
    for phrase, mult in sorted(_ENGLISH_FRACTION_PHRASES.items(), key=lambda kv: -len(kv[0])):
        if cleaned.startswith(phrase):
            remainder = cleaned[len(phrase):].strip()
            if not remainder:
                return mult
            terms = remainder.split()
            return mult * scale_factor(*terms)

    tokens = cleaned.replace("rs.", "").split()
    if not tokens:
        raise ParseError(f"could not parse {original!r}")

    modifiers = load_colloquial_modifiers()

    # Colloquial modifier as the leading token, e.g. "sava lakh", "dedh crore"
    leading = _COLLOQUIAL_ALIASES.get(tokens[0], tokens[0])
    if leading in modifiers:
        mod = modifiers[leading]
        remainder_tokens = tokens[1:]
        if not remainder_tokens:
            raise ParseError(f"colloquial modifier {tokens[0]!r} with no scale word in {original!r}")
        return mod.multiplier * scale_factor(*remainder_tokens)

    # General additive/multiplicative chunk scan: <number> <scale> [<scale> ...] [<number> <scale> ...]
    total = Decimal(0)
    i = 0
    matched_any = False
    while i < len(tokens):
        num_tok = tokens[i]
        if not _PLAIN_NUMBER_RE.match(num_tok):
            raise ParseError(f"expected a number at token {i} ({num_tok!r}) in {original!r}")
        value = Decimal(num_tok)
        i += 1
        if i >= len(tokens):
            raise ParseError(f"number {num_tok!r} with no scale word in {original!r}")
        # consume one or more consecutive scale words multiplicatively ("lakh crore" compound)
        scale_terms: list[str] = []
        while i < len(tokens):
            try:
                scale_terms.append(_resolve_scale_alias(tokens[i]))
            except ScaleError:
                break
            i += 1
        if not scale_terms:
            raise ParseError(f"expected a scale word after {value} in {original!r}")
        total += value * scale_factor(*scale_terms)
        matched_any = True

    if not matched_any:
        raise ParseError(f"could not parse {original!r}")
    return total


# ---------------------------------------------------------------------------
# to_words / from_words  (Indian-English word form)
# ---------------------------------------------------------------------------

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALE_WORDS_ASCENDING = ["thousand", "lakh", "crore", "arab", "kharab", "neel", "padma", "shankh"]


def _two_digit_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")


def _three_digit_words(n: int) -> str:
    hundreds, rest = divmod(n, 100)
    parts = []
    if hundreds:
        parts.append(f"{_ONES[hundreds]} hundred")
    if rest:
        parts.append(_two_digit_words(rest))
    return " ".join(parts)


def to_words(n: int, lang: str = "en_in") -> str:
    """Convert an integer to Indian-English word form: 12345678 ->
    "one crore twenty-three lakh forty-five thousand six hundred seventy-eight".

    Supported up through the crore/arab range that generate_c1_items.py actually exercises; the
    algorithm is scale-agnostic (it just walks 2-digit groups) so it extends to kharab/neel/padma/
    shankh for free, but those have not been separately verified against a worked example.
    """
    if lang != "en_in":
        raise ValueError(f"unsupported lang {lang!r}; only 'en_in' is implemented")
    if n == 0:
        return "zero"
    sign = "minus " if n < 0 else ""
    n = abs(n)
    s = str(n)

    if len(s) <= 3:
        last3 = int(s)
        groups: list[str] = []
    else:
        last3 = int(s[-3:])
        rest = s[:-3]
        groups = []
        while rest:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]

    parts = []
    for i, grp_str in enumerate(reversed(groups)):
        val = int(grp_str)
        if val:
            if i >= len(_SCALE_WORDS_ASCENDING):
                raise ValueError(f"{n} exceeds the supported scale ladder (beyond shankh)")
            parts.append(f"{_two_digit_words(val)} {_SCALE_WORDS_ASCENDING[i]}")
    parts.reverse()
    if last3:
        parts.append(_three_digit_words(last3))
    return sign + " ".join(parts)


_WORD_VALUES: dict[str, int] = {w: i for i, w in enumerate(_ONES)}
_WORD_VALUES.update({t: v * 10 for v, t in enumerate(_TENS) if t})
_SCALE_WORD_VALUES: dict[str, int] = {}


def _scale_word_value(word: str) -> int | None:
    global _SCALE_WORD_VALUES
    if not _SCALE_WORD_VALUES:
        scales = load_scale_terms()
        for alias, canonical in _SCALE_ALIASES.items():
            if canonical in scales and canonical != "hundred":
                _SCALE_WORD_VALUES[alias] = 10 ** scales[canonical].power_of_10
        _SCALE_WORD_VALUES["hundred"] = 100
        _SCALE_WORD_VALUES["hundreds"] = 100
    return _SCALE_WORD_VALUES.get(word)


def from_words(s: str) -> int:
    """Inverse of to_words: parse an Indian-English number-word phrase back to an integer.
    This is the reliably-gradable direction for the word_form_output subcategory - a model asked
    to parse words into a number can be graded exactly, while asking it to spell a number out in
    words invites too many equally-valid phrasings to grade deterministically.
    """
    cleaned = s.strip().lower().replace("-", " ").replace(",", " ")
    tokens = [t for t in cleaned.split() if t not in ("and",)]
    if not tokens:
        raise ParseError(f"empty input: {s!r}")

    sign = 1
    if tokens[0] in ("minus", "negative"):
        sign = -1
        tokens = tokens[1:]

    total = 0
    current = 0
    for tok in tokens:
        if tok == "zero" and len(tokens) == 1:
            return 0
        if tok in _WORD_VALUES:
            current += _WORD_VALUES[tok]
            continue
        scale_val = _scale_word_value(tok)
        if scale_val is not None:
            if scale_val == 100:
                current = (current or 1) * 100
            else:
                total += (current or 1) * scale_val
                current = 0
            continue
        raise ParseError(f"unrecognized word {tok!r} in {s!r}")
    total += current
    return sign * total
