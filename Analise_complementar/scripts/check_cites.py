#!/usr/bin/env python3
"""Verifica citações órfãs e bibitems não citados no conference_resumido.tex"""
import re
from pathlib import Path

P = Path(__file__).resolve().parents[2] / "Artigo" / "conference_resumido.tex"
s = P.read_text(encoding="utf-8")

cite_re = re.compile(r"\\cite\{([^}]+)\}")
bib_re = re.compile(r"\\bibitem\{([^}]+)\}")

cites = set()
for m in cite_re.finditer(s):
    for k in m.group(1).split(","):
        cites.add(k.strip())

bibs = set(bib_re.findall(s))

print("Cites sem bibitem:", sorted(cites - bibs))
print("Bibitems nao citados:", sorted(bibs - cites))
print(f"Total cites: {len(cites)}, total bibitems: {len(bibs)}")
