"""Build <team>_submission.zip in the structure required by the organisers.

    python -m ber.package <team_name>
"""
import sys
import zipfile
from pathlib import Path

from . import config

CODE = Path(__file__).resolve().parents[2]          # code/business_entity_resolution
PROJECT = CODE.parents[1]


def build(team: str) -> Path:
    out = PROJECT / "submissions" / f"{team}_submission.zip"
    out.parent.mkdir(exist_ok=True)
    files = [(config.OUT / "matching_results.tsv", "output/matching_results.tsv"),
             (config.OUT / "candidate_pairs.tsv", "output/candidate_pairs.tsv"),
             (PROJECT / "Documentation_template.md", "Documentation_template.md")]
    for p in sorted(CODE.rglob("*")):
        rel = p.relative_to(CODE)
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc" and (not p.name.startswith("_") or p.name == "__init__.py"):
            files.append((p, "code/business_entity_resolution/" + rel.as_posix()))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for src, arc in files:
            z.write(src, arc)
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
        print(f"{out}: {out.stat().st_size / 2 ** 20:.1f} MB, {len(z.namelist())} files")
    return out


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "team")
