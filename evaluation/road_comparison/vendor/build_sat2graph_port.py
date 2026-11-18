"""Rebuild the Python 3 port from the pinned, unmodified Sat2Graph files."""

from pathlib import Path
from lib2to3.refactor import RefactoringTool, get_fixers_from_package
import re


ROOT = Path(__file__).parent / "sat2graph"
fixer = RefactoringTool(get_fixers_from_package("lib2to3.fixes"))
for name in ("graph", "topo"):
    upstream = (ROOT / f"{name}.upstream.py.txt").read_text(encoding="utf-8")
    converted = str(fixer.refactor_string(upstream + "\n", f"{name}.py"))
    if name == "topo":
        converted = converted.replace("from sets import Set", "Set = set")
        converted = converted.replace("import showTOPO", "from . import no_render as showTOPO")
        converted = converted.replace("import graph as splfy", "from . import graph as splfy")
        converted = converted.replace("from . import showTOPO", "from . import no_render as showTOPO")
    # Python 2 dict.keys() gave a list. In Python 3, list(keys()) membership
    # makes graph construction quadratic; dict membership is identical.
    converted = re.sub(r"(not in|in) list\((\w+(?:\.\w+)*)\.keys\(\)\)",
                       r"\1 \2", converted)
    if name == "topo":
        converted = converted.replace("visitedNodes = []", "visitedNodes = set()")
        converted = converted.replace("visitedNodes.append(", "visitedNodes.add(")
    (ROOT / f"{name}.py").write_text(converted, encoding="utf-8")
