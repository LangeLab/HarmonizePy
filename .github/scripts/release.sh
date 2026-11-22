#!/usr/bin/env bash
# Release checks and packages for HarmonizePy.
#
#   release.sh version              print the version, after checking pyproject.toml, __init__.py, and CITATION.cff agree
#   release.sh notes [TAG]          print the CHANGELOG.md section of the version (TAG must be vVERSION)
#   release.sh package OUTPUT_DIR   build the wheel and sdist, check them, smoke-test the wheel, copy both to OUTPUT_DIR
#
# The sdist must hold only files git tracks, and the wheel is installed into a fresh environment outside the
# checkout and run there. Work directories are made under RUNNER_TEMP (or /var/tmp) and left in place.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

version=$(sed -n 's/^version = "\([^"]*\)"$/\1/p' pyproject.toml)
[[ "$version" =~ ^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]] ||
    die "pyproject.toml must declare one major.minor.patch version"
init_version=$(sed -n 's/^__version__ = "\([^"]*\)"$/\1/p' src/harmonizepy/__init__.py)
[[ "$init_version" == "$version" ]] || die "__init__.py has version '$init_version', pyproject.toml '$version'"
cff_version=$(sed -n 's/^version: \(.*\)$/\1/p' CITATION.cff)
[[ "$cff_version" == "$version" ]] || die "CITATION.cff has version '$cff_version', pyproject.toml '$version'"

notes() {
    awk -v version="$version" '
        /^## / {
            if (found) exit
            found = ($2 == "[" version "]")
            next
        }
        found {
            if ($0 ~ /[^[:space:]]/) nonempty = 1
            if (nonempty) print
        }
        END { if (!found || !nonempty) exit 1 }
    ' CHANGELOG.md || die "CHANGELOG.md needs a nonempty [$version] section"
}

package() {
    local output="$1" work dist wheel sdist untracked name="harmonizepy-$version"
    work=$(mktemp -d "${RUNNER_TEMP:-/var/tmp}/harmonizepy-release.XXXXXX")
    dist="$work/dist"
    uv build --out-dir "$dist" > "$work/build.log" 2>&1 || { cat "$work/build.log" >&2; die "uv build failed"; }
    wheel="$dist/$name-py3-none-any.whl"
    sdist="$dist/$name.tar.gz"
    [[ -s "$wheel" && -s "$sdist" ]] || die "expected $name-py3-none-any.whl and $name.tar.gz, found: $(ls "$dist")"
    [[ "$(find "$dist" -maxdepth 1 -type f \( -name '*.whl' -o -name '*.tar.gz' \) | wc -l)" -eq 2 ]] ||
        die "expected one wheel and one sdist in the build output: $(ls "$dist")"

    tar -tzf "$sdist" | sed -n "s,^$name/,,p" | grep -v -e '/$' -e '^PKG-INFO$' | LC_ALL=C sort > "$work/sdist-files"
    git ls-files | LC_ALL=C sort > "$work/tracked-files"
    untracked=$(LC_ALL=C comm -23 "$work/sdist-files" "$work/tracked-files")
    [[ -z "$untracked" ]] || die "the sdist holds files git does not track: $untracked"

    uv venv --quiet --python 3.12 "$work/venv"
    uv pip install --quiet --python "$work/venv/bin/python" "$wheel"
    (
        cd "$work"
        export PATH="$work/venv/bin:/usr/bin:/bin"
        [[ "$(harmonizepy --version)" == "harmonizepy $version" ]] || die "--version is not 'harmonizepy $version'"
        python - <<'PY'
import numpy as np
import pandas as pd

import harmonizepy

rng = np.random.default_rng(0)
batch = np.repeat([1, 2, 3], 4)
values = rng.normal(20, 1, (30, 1)) + rng.normal(0, 0.3, (30, 12)) + np.array([0.0, 1.0, -1.0])[batch - 1]
values[:2, batch == 3] = np.nan
samples = [f"s{i}" for i in range(12)]
data = pd.DataFrame(values, index=[f"f{i}" for i in range(30)], columns=samples)
data.to_csv("data.tsv", sep="\t")
pd.DataFrame({"ID": samples, "sample": range(1, 13), "batch": batch}).to_csv("batch.csv", index=False)
for algorithm, mode in (("ComBat", 1), ("ComBat", 3), ("limma", 1)):
    out, report = harmonizepy.harmonize(data, "batch.csv", algorithm=algorithm, combat_mode=mode, return_report=True)
    assert out.shape == data.shape and report["status"].eq("corrected").all(), (algorithm, mode)
    assert out.iloc[:2, 8:].isna().all().all() and np.isfinite(out.iloc[2:]).all().all()
PY
        harmonizepy data.tsv batch.csv -o corrected.tsv --report report.tsv --no-log -q
        [[ "$(wc -l < corrected.tsv)" -eq 31 && "$(wc -l < report.tsv)" -eq 31 ]] || die "the CLI run wrote the wrong number of rows"
    )
    mkdir -p "$output"
    cp "$wheel" "$sdist" "$output/"
    printf 'tested %s\n' "$output/$(basename "$wheel")" "$output/$(basename "$sdist")"
}

case "${1:-}" in
    version)
        printf '%s\n' "$version"
        ;;
    notes)
        tag="${2:-v$version}"
        [[ "$tag" == "v$version" ]] || die "tag $tag does not match version v$version"
        notes
        ;;
    package)
        [[ $# -eq 2 ]] || die "usage: release.sh package OUTPUT_DIR"
        package "$2"
        ;;
    *)
        printf 'usage: release.sh version | notes [TAG] | package OUTPUT_DIR\n' >&2
        exit 2
        ;;
esac
