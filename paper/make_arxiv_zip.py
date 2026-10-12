"""Build paper/arxiv_source.zip for upload to arXiv (forward-slash paths, .bbl included).

    tectonic -X compile main.tex --keep-intermediates    # in paper/, writes main.bbl
    python paper/make_arxiv_zip.py
"""

import zipfile
from pathlib import Path

PAPER = Path(__file__).resolve().parent
PATTERNS = ("main.tex", "main.bbl", "numbers.tex", "references.bib", "sec_*.tex", "tab_*.tex", "ex_*.tex",
            "figures/*.pdf")

files = sorted({f for p in PATTERNS for f in PAPER.glob(p)})
if not (PAPER / "main.bbl").exists():
    raise SystemExit("main.bbl missing: run tectonic with --keep-intermediates first")
with zipfile.ZipFile(PAPER / "arxiv_source.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, f.relative_to(PAPER).as_posix())
print("\n".join(f.relative_to(PAPER).as_posix() for f in files))
