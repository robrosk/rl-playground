"""Traceability: every step id defined in docs/ must be tagged somewhere in the code."""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
STEP = re.compile(r"\*\*\[([A-Z0-9]+-\d+)\]\*\*")


def documented_ids():
    ids = set()
    for page in (ROOT / "docs").rglob("*.md"):
        ids |= set(STEP.findall(page.read_text()))
    return ids


def test_every_documented_step_is_tagged_in_code():
    code = "\n".join(p.read_text() for p in (ROOT / "rl_playground").rglob("*.py"))
    missing = sorted(i for i in documented_ids() if f"[{i}]" not in code)
    assert not missing, f"documented but not tagged in code: {missing}"
