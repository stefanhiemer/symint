#!/usr/bin/env bash
# Build symintegral and upload it to TestPyPI, test the install from
# there, then upload it to PyPI.
#
# Usage (from anywhere; the active Python needs the [pypi] and [test]
# extras, e.g. pip install -e ".[pypi,test]"):
#
#   scripts/release.sh             TestPyPI, install test, then PyPI
#   scripts/release.sh --test-only stop after the TestPyPI install test
#   scripts/release.sh --skip-test-pypi   upload to PyPI directly
#   scripts/release.sh --dry-run   all checks and the build, no upload
#
# Credentials come from ~/.pypirc (sections [testpypi] and [pypi]).
# A version can be uploaded only once per index: bump __version__ in
# symintegral/__init__.py before each release.

set -euo pipefail

TEST_ONLY=0
SKIP_TEST_PYPI=0
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --test-only)      TEST_ONLY=1 ;;
        --skip-test-pypi) SKIP_TEST_PYPI=1 ;;
        --dry-run)        DRY_RUN=1 ;;
        -h|--help)        sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PYTHON="${PYTHON:-python}"

step() { printf '\n=== %s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

VERSION="$(sed -n 's/^__version__ = "\(.*\)"/\1/p' symintegral/__init__.py)"
[ -n "$VERSION" ] || fail "could not read __version__ from symintegral/__init__.py"
echo "symintegral $VERSION"

# --- checks ---------------------------------------------------------------

step "Checking the working tree"
if [ -n "$(git status --porcelain)" ]; then
    git status --short
    if [ "$DRY_RUN" = 1 ]; then
        echo "(dry run: continuing despite uncommitted changes)"
    else
        fail "uncommitted changes; commit them first, so the release matches a commit"
    fi
fi

# 200 = the version exists, 404 = it is free
version_exists() {  # $1: https://pypi.org or https://test.pypi.org
    [ "$(curl -s -o /dev/null -w '%{http_code}' "$1/pypi/symintegral/$VERSION/json")" = 200 ]
}
step "Checking that $VERSION is not uploaded yet"
if [ "$SKIP_TEST_PYPI" = 0 ] && version_exists https://test.pypi.org; then
    fail "$VERSION is already on TestPyPI; bump __version__ (or use --skip-test-pypi)"
fi
if [ "$TEST_ONLY" = 0 ] && version_exists https://pypi.org; then
    fail "$VERSION is already on PyPI; bump __version__"
fi
echo "ok"

step "Running the tests"
"$PYTHON" -m pytest -q --disable-warnings

# --- build ----------------------------------------------------------------

step "Building"
rm -rf dist/
"$PYTHON" -m build
"$PYTHON" -m twine check --strict dist/*

if [ "$DRY_RUN" = 1 ]; then
    step "Dry run: built dist/ without uploading"
    ls -l dist/
    exit 0
fi

# --- TestPyPI -------------------------------------------------------------

if [ "$SKIP_TEST_PYPI" = 0 ]; then
    step "Uploading to TestPyPI"
    "$PYTHON" -m twine upload --repository testpypi dist/*

    step "Testing the install from TestPyPI in a fresh environment"
    TMP="$(mktemp -d)"
    trap 'rm -rf "$TMP"' EXIT
    "$PYTHON" -m venv "$TMP/venv"
    VPY="$TMP/venv/bin/python"
    # dependencies from the real PyPI (on TestPyPI anyone can upload
    # packages named like them), only symintegral itself from TestPyPI
    "$VPY" -m pip install -q sympy "passagemath-symbolics[maxima,fricas]" \
        passagemath-repl numpy scipy
    # the new version can take a minute to appear in the index
    for attempt in 1 2 3 4 5 6; do
        if "$VPY" -m pip install -q --no-deps --no-cache-dir \
               -i https://test.pypi.org/simple/ "symintegral==$VERSION"; then
            break
        fi
        [ "$attempt" = 6 ] && fail "symintegral $VERSION not installable from TestPyPI"
        echo "not in the index yet, retrying in 20 s ..."
        sleep 20
    done
    # run outside the repository, so the installed copy is imported
    (cd "$TMP" && "$VPY" - <<'EOF'
import sympy as sp
import symintegral
assert "site-packages" in symintegral.__file__, symintegral.__file__
x = sp.Symbol("x", real=True)
res = symintegral.integrate(sp.Integral(sp.exp(-x**2) * sp.cos(x), (x, 0, sp.oo)))
assert sp.simplify(res - sp.sqrt(sp.pi) * sp.exp(-sp.Rational(1, 4)) / 2) == 0, res
print("installed", symintegral.__version__, "from", symintegral.__file__)
print("Maxima check ok:", res)
EOF
    )
    echo "TestPyPI page: https://test.pypi.org/project/symintegral/$VERSION/"
fi

if [ "$TEST_ONLY" = 1 ]; then
    step "Stopping after TestPyPI (--test-only)"
    exit 0
fi

# --- PyPI -----------------------------------------------------------------

step "Uploading symintegral $VERSION to PyPI"
read -r -p "Upload to the real PyPI now? This cannot be undone. [y/N] " answer
[ "$answer" = y ] || [ "$answer" = Y ] || fail "aborted before the PyPI upload"
"$PYTHON" -m twine upload dist/*
echo "PyPI page: https://pypi.org/project/symintegral/$VERSION/"
echo "Consider tagging the release: git tag v$VERSION && git push origin v$VERSION"
