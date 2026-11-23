#!/usr/bin/env python3
"""
Bump the version in symintegral/__init__.py.

Usage:

    scripts/bump_version.py patch     0.1.0 -> 0.1.1
    scripts/bump_version.py minor     0.1.0 -> 0.2.0
    scripts/bump_version.py major     0.1.0 -> 1.0.0
    scripts/bump_version.py 0.3.0     set an explicit version

The package metadata (pyproject.toml) and the docs read the version from
there, so this is the only place to change. Commit afterwards, then run
scripts/release.sh.
"""

import re
import sys
from pathlib import Path

INIT = Path(__file__).resolve().parent.parent / "symintegral" / "__init__.py"
PATTERN = re.compile(r'^__version__ = "(\d+)\.(\d+)\.(\d+)"$', re.M)


def bumped(current, part):
    """The new version string for `part` ("major", "minor", "patch" or X.Y.Z)."""
    major, minor, patch = current
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    if re.fullmatch(r"\d+\.\d+\.\d+", part):
        if tuple(map(int, part.split("."))) <= current:
            sys.exit(f"error: {part} is not newer than {'.'.join(map(str, current))}")
        return part
    sys.exit(f"error: expected major, minor, patch or X.Y.Z, got {part!r}")


def main():
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        sys.exit(__doc__.strip())
    text = INIT.read_text()
    match = PATTERN.search(text)
    if match is None:
        sys.exit(f"error: no __version__ = \"X.Y.Z\" line in {INIT}")
    current = tuple(int(g) for g in match.groups())
    new = bumped(current, sys.argv[1])
    INIT.write_text(PATTERN.sub(f'__version__ = "{new}"', text, count=1))
    print(f"{'.'.join(map(str, current))} -> {new}  ({INIT.relative_to(INIT.parent.parent)})")


if __name__ == "__main__":
    main()
