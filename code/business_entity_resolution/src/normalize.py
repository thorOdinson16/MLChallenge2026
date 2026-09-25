"""Text normalization shared by blocking, features, and evaluation.

Design rules:
- Language/country agnostic: no hard-coded {US, India} logic.
- Only uses the provided data (no external lookups).
"""
import re
import unicodedata

# Token-level abbreviation expansion (applied after lowercasing/accent strip).
ABBR = {
    # generic address
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "hwy": "highway", "pkwy": "parkway", "ct": "court",
    "pl": "place", "sq": "square", "ter": "terrace", "cir": "circle", "trl": "trail",
    "bldg": "building", "flr": "floor", "fl": "floor", "opp": "opposite", "nr": "near",
    "apt": "apartment", "ste": "suite", "no": "number",
    "saint": "street",  # noise pattern seen in data: ST -> SAINT
    # compass
    "n": "north", "s": "south", "e": "east", "w": "west",
    # french
    "r": "rue", "imp": "impasse", "bd": "boulevard", "pl.": "place", "all": "allee",
    "chem": "chemin", "rte": "route", "fg": "faubourg",
    # business
    "co": "company", "corp": "corporation", "inc": "incorporated", "ltd": "limited",
    "pvt": "private", "prv": "private", "llc": "llc", "lp": "lp",
}

# Legal-form tokens: separated from the core name.
LEGAL = {
    "private", "limited", "corporation", "company", "incorporated", "llc", "llp", "lp",
    "pllc", "pc", "inc", "corp", "ltd", "pvt", "co", "plc", "the",
    # france
    "sa", "sarl", "sas", "sasu", "eurl", "sci", "ei", "snc", "scop", "association",
    # india extra
    "opc", "pl", "pte",
}

# Very common address filler tokens that carry little identity.
ADDR_STOP = {
    "road", "street", "avenue", "drive", "lane", "boulevard", "court", "place",
    "number", "floor", "unit", "apartment", "suite", "the", "of", "and", "near",
    "opposite", "bis", "ter", "rue", "de", "la", "le", "les", "du", "des", "d", "l",
    "north", "south", "east", "west", "nagar", "new", "city", "county",
}

_WS = re.compile(r"\s+")
_NONALNUM = re.compile(r"[^a-z0-9]+")
_DOMAIN = re.compile(r"\b([a-z0-9\-]+)\.(com|net|org|in|co|io|fr|us)\b")
_UNIT = re.compile(r"\b(unit|apt|apartment|suite|ste|fl|floor|flr|room|rm)\s*#?\s*[a-z0-9\-]{1,5}\b")
_HASHNUM = re.compile(r"#\s*[a-z0-9\-]{1,6}\b")


def strip_accents(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def _base(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = strip_accents(s.lower())
    s = s.replace("&", " and ")
    return s


def norm_name(s: str):
    """Returns (core_name, legal_tokens).
    core_name: lowercase, accent-stripped, abbreviation-expanded, legal forms removed,
    domain-style names turned into words (fortunefinance.com -> fortunefinance).
    """
    s = _base(s)
    s = _DOMAIN.sub(lambda m: m.group(1).replace("-", " "), s)
    s = _NONALNUM.sub(" ", s)
    toks = [ABBR.get(t, t) for t in s.split()]
    legal = [t for t in toks if t in LEGAL]
    core = [t for t in toks if t not in LEGAL]
    if not core:  # name consisted only of legal-form words
        core = toks
    return " ".join(core), " ".join(legal)


def norm_addr(s: str) -> str:
    """Address normalization: strip units, expand abbreviations, keep digits."""
    s = _base(s)
    s = _UNIT.sub(" ", s)
    s = _HASHNUM.sub(" ", s)
    s = re.sub(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)", " ", s)  # split 78a -> 78 a
    s = _NONALNUM.sub(" ", s)
    toks = [ABBR.get(t, t) for t in s.split()]
    return " ".join(toks)


def addr_tokens(s: str):
    """Identity-bearing address tokens (norm_addr output minus filler)."""
    return [t for t in norm_addr(s).split() if t not in ADDR_STOP and len(t) > 1]


def house_numbers(s: str):
    """Set of leading-style house numbers found in an address (digits only, leading zeros stripped)."""
    nums = re.findall(r"\d+", _base(s))
    return {n.lstrip("0") or "0" for n in nums}


def compact(s: str) -> str:
    """Alphanumeric-only form, for acronym / concatenated-name comparison."""
    return _NONALNUM.sub("", _base(s))


def acronym(core_name: str) -> str:
    return "".join(t[0] for t in core_name.split() if t)