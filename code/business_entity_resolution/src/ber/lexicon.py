"""Hand-written normalisation lexicons (no external data; general language knowledge only).

Country-specific maps are keyed by the raw `country` label; unknown labels fall back to GENERIC,
so a new country degrades gracefully instead of breaking.
"""

# ---------------------------------------------------------------- names
# token -> canonical token (applied after lower-casing, accent stripping, dot removal)
NAME_CANON = {
    "private": "pvt", "pvte": "pvt", "limited": "ltd", "ltd": "ltd", "ltd.": "ltd",
    "incorporated": "inc", "incorporation": "inc", "corporation": "corp", "company": "co", "compagnie": "co",
    "cie": "co", "companies": "cos", "and": "and", "et": "and", "und": "and",
    "center": "centre", "centr": "centre", "services": "service", "svcs": "service", "svc": "service",
    "intl": "international", "int": "international", "mfg": "manufacturing", "assoc": "associates",
    "assocs": "associates", "bros": "brothers", "dept": "department", "tech": "technology",
    "technologies": "technology", "technolgies": "technology", "natl": "national", "mgmt": "management",
    "grp": "group", "hldgs": "holdings", "hldg": "holdings", "ent": "enterprises", "entp": "enterprises",
    "etablissements": "etablissement", "ets": "etablissement", "st": "saint", "ste": "sainte", "mt": "mount",
    "ft": "fort", "sr": "senior", "jr": "junior", "ctr": "centre",
}
# legal-form vocabulary (after NAME_CANON)
LEGAL = {
    "inc", "llc", "ltd", "pvt", "corp", "co", "lp", "llp", "pc", "pllc", "plc", "pa", "lc", "public",
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "ei", "scop", "scp", "gie", "selarl", "selas", "sca",
    "gmbh", "ag", "bv", "nv", "tbk", "pty", "opc", "lllp", "ltda", "srl", "spa", "kg", "oy", "ab",
}
# tokens that carry no identity information for the core name
NAME_STOP = {
    "the", "and", "of", "de", "du", "des", "la", "le", "les", "l", "d", "a", "an", "et", "en", "au", "aux",
    "m", "s", "ms", "mr", "mrs", "smt", "shri", "shree", "sri", "dr", "messrs", "kumari", "prof",
    "india", "indian", "france", "usa", "us", "dba", "ta", "aka",
}
DBA_PATTERN = r"\s(?:d\s?b\s?a|d/b/a|t/a|trading as|a\s?k\s?a|doing business as)\s"
URL_PATTERN = r"^\s*(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:com|net|org|in|co\.in|co|biz|info|fr|us|io|org\.in|net\.in)\s*$"

# ---------------------------------------------------------------- addresses
ADDR_GENERIC = {
    "st": "street", "str": "street", "rd": "road", "dr": "drive", "drv": "drive", "ave": "avenue",
    "av": "avenue", "avn": "avenue", "ln": "lane", "ct": "court", "crt": "court", "cir": "circle",
    "pl": "place", "blvd": "boulevard", "bd": "boulevard", "pkwy": "parkway", "pky": "parkway",
    "hwy": "highway", "trl": "trail", "ter": "terrace", "terr": "terrace", "sq": "square", "pt": "point",
    "mt": "mount", "ft": "fort", "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast",
    "nw": "northwest", "se": "southeast", "sw": "southwest", "cres": "crescent", "expy": "expressway",
    "fwy": "freeway", "xing": "crossing", "hts": "heights", "jct": "junction", "mtn": "mountain",
    "spgs": "springs", "sprgs": "springs", "vly": "valley", "vlg": "village", "aly": "alley", "cv": "cove",
    "pass": "passage", "psge": "passage", "rte": "route", "mkt": "market",
    # India
    "nr": "near", "opp": "opposite", "sec": "sector", "sect": "sector", "ph": "phase", "blk": "block",
    "bldg": "building", "bldng": "building", "flr": "floor", "apts": "apartments", "appt": "apartment",
    "extn": "extension", "ext": "extension", "indl": "industrial", "ind": "industrial", "estt": "estate",
    "soc": "society", "chs": "society", "mkt.": "market", "cmplx": "complex", "compl": "complex",
    "gf": "ground", "ff": "first", "sf": "second", "tf": "third",
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
    "bangalore": "bengaluru", "bombay": "mumbai", "calcutta": "kolkata", "madras": "chennai",
    "gurgaon": "gurugram", "trivandrum": "thiruvananthapuram", "poona": "pune",
    "saint": "saint", "ste": "suite",
}
ADDR_FR = {
    **{k: v for k, v in ADDR_GENERIC.items() if k not in {"st", "ste", "dr", "s", "e", "n", "w", "pl", "ct"}},
    "r": "rue", "ru": "rue", "av": "avenue", "ave": "avenue", "bd": "boulevard", "bld": "boulevard",
    "boul": "boulevard", "bvd": "boulevard", "all": "allee", "pl": "place", "imp": "impasse",
    "ch": "chemin", "chem": "chemin", "rte": "route", "sq": "square", "pass": "passage", "qu": "quai",
    "crs": "cours", "fg": "faubourg", "fbg": "faubourg", "res": "residence", "lot": "lotissement",
    "st": "saint", "ste": "sainte", "zi": "industrial", "za": "activite", "zac": "activite",
    "prom": "promenade", "sent": "sentier", "hameau": "hameau", "cite": "cite", "pte": "porte",
}
ADDR_MAPS = {"France": ADDR_FR}

# unit / label words dropped entirely (their numbers are kept)
ADDR_STOP = {
    "unit", "apt", "apartment", "suite", "ste", "room", "rm", "no", "nos", "num", "number", "hno", "dno",
    "sno", "door", "house", "plot", "flat", "shop", "office", "h", "d", "the", "of", "and", "near", "opposite",
    "behind", "beside", "next", "to", "c", "o", "co", "via", "po", "p", "box", "pmb", "dist", "district",
    "tq", "taluk", "taluka", "tehsil", "mandal", "cdp", "township", "twp", "town", "city", "county",
    "village", "vill", "at", "post", "null", "none", "na", "nan", "de", "du", "des", "la", "le", "les", "l",
    "bis", "ter", "et", "en", "sur", "lieu", "dit", "cedex",
    "st", "nd", "rd", "th",  # ordinal suffix leftovers
}

# multi-word state/region names -> one token (applied to the raw string, word-bounded)
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc", "puerto rico": "pr",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br", "chhattisgarh": "cg",
    "chattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh",
    "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od",
    "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "tamilnadu": "tn",
    "telangana": "tg", "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk",
    "west bengal": "wb", "delhi": "dl", "nct of delhi": "dl", "jammu and kashmir": "jk",
    "jammu & kashmir": "jk", "chandigarh": "ch", "puducherry": "py", "pondicherry": "py", "ladakh": "la",
    "dadra and nagar haveli": "dn", "daman and diu": "dd", "lakshadweep": "ld",
    "andaman and nicobar islands": "an", "ts": "tg", "or": "od", "ct": "cg", "ut": "uk", "tl": "tg",
}
# French regions and departments -> region code (a record may carry either level)
FR_STATES = {
    "hauts-de-france": "hdf", "hauts de france": "hdf", "nord": "hdf", "pas-de-calais": "hdf",
    "pas de calais": "hdf", "aisne": "hdf", "oise": "hdf", "somme": "hdf",
    "nouvelle-aquitaine": "naq", "nouvelle aquitaine": "naq", "gironde": "naq", "landes": "naq",
    "dordogne": "naq", "charente": "naq", "charente-maritime": "naq", "pyrenees-atlantiques": "naq",
    "lot-et-garonne": "naq", "vienne": "naq", "haute-vienne": "naq", "correze": "naq", "creuse": "naq",
    "deux-sevres": "naq",
    "pays de la loire": "pdl", "pays-de-la-loire": "pdl", "loire-atlantique": "pdl", "loire atlantique": "pdl",
    "maine-et-loire": "pdl", "vendee": "pdl", "sarthe": "pdl", "mayenne": "pdl",
    "ile-de-france": "idf", "ile de france": "idf", "paris": "idf",
    "auvergne-rhone-alpes": "ara", "provence-alpes-cote d'azur": "pac", "occitanie": "occ",
    "grand est": "ges", "bretagne": "bre", "normandie": "nor", "bourgogne-franche-comte": "bfc",
    "centre-val de loire": "cvl", "corse": "cor",
}
STATE_MAPS = {"US": US_STATES, "India": IN_STATES, "France": FR_STATES}
