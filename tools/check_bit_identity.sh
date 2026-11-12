#!/usr/bin/env bash
# Bit-for-bit comparison of the current solver with a baseline commit on the
# bundled Hinterkante case (legacy mesh format).
#
# Usage:
#   tools/check_bit_identity.sh [--baseline COMMIT] [--bin PATH] [--steps N] [--keep]
#
#   --baseline COMMIT  commit to compare with (default: the root commit of the repository)
#   --bin PATH         current solver executable (default: build/HydraMa2D)
#   --steps N          time steps per run (default: 200)
#   --keep             keep the work directory
#
# The baseline is built in a temporary git worktree (CMake, Release, no tests).
# Every configuration runs both executables on a fresh copy of the case and
# compares
#   * results/hinterkante.var (all fields, 17 significant digits) and, for
#     turbulent runs, results/hinterkante.turb.var, byte for byte (cmp);
#   * the whole run log, line for line (every pressure residual and
#     iteration count), except the wall time ("Time Taken"), the case
#     directory (it differs by construction) and the informational lines the
#     current version adds (banner "Mesh formats: ...", "Mesh quality: ...").
# Exit status 0 if every configuration is identical.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
baseline=""
bin="$here/build/HydraMa2D"
steps=200
keep=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --baseline) baseline="$2"; shift 2 ;;
        --bin) bin="$2"; shift 2 ;;
        --steps) steps="$2"; shift 2 ;;
        --keep) keep=1; shift ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ -x "$bin" ]] || { echo "solver executable not found: $bin (build first, or pass --bin)" >&2; exit 2; }
bin="$(cd "$(dirname "$bin")" && pwd)/$(basename "$bin")"
[[ -n "$baseline" ]] || baseline="$(git -C "$here" rev-list --max-parents=0 HEAD | tail -n 1)"

work="$(mktemp -d "${TMPDIR:-/tmp}/hydrama2d-identity.XXXXXX")"
cleanup() {
    git -C "$here" worktree remove --force "$work/baseline" >/dev/null 2>&1 || true
    [[ $keep == 1 ]] && echo "work directory kept: $work" || rm -rf "$work"
}
trap cleanup EXIT

echo "Building baseline $(git -C "$here" rev-parse --short "$baseline") ..."
git -C "$here" worktree add --detach "$work/baseline" "$baseline" >/dev/null 2>&1
cmake -S "$work/baseline" -B "$work/baseline-build" -DCMAKE_BUILD_TYPE=Release -DHYDRAMA2D_BUILD_TESTS=OFF \
      -DHYDRAMA2D_WITH_CGNS=OFF >"$work/baseline-configure.log" 2>&1
cmake --build "$work/baseline-build" -j"$(nproc 2>/dev/null || echo 2)" >"$work/baseline-build.log" 2>&1
base_bin="$work/baseline-build/HydraMa2D"

# set_line FILE INDEX VALUE: replace the INDEX-th (0-based) non-empty line.
set_line() {
    awk -v idx="$2" -v val="$3" '
        { line = $0; sub(/\r$/, "", line) }
        line ~ /[^ \t]/ { if (n == idx) { print val; n++; next } n++ }
        { print }' "$1" >"$1.tmp" && mv "$1.tmp" "$1"
}

# Control file value lines (0-based non-empty line index): number of steps 5,
# transient 8, output every N steps 17, dtfixed 23, fixed dt 26.
# Solver file: relative tolerance 6, absolute tolerance 9; settings after line 10.
prepare() { # prepare DIR CONFIG
    local dir="$1" config="$2"
    rm -rf "$dir" && mkdir -p "$dir"
    cp -r "$here/Hinterkante.dev.Re100/constant" "$here/Hinterkante.dev.Re100/system" "$dir/"
    mkdir -p "$dir/results"
    local control="$dir/system/hinterkante.control" solver="$dir/system/hinterkante.solver"
    set_line "$control" 5 "$steps"
    set_line "$control" 17 "$steps"
    settings=""
    case "$config" in
        legacy) ;;
        steady-local-dt) set_line "$control" 8 0; set_line "$control" 23 0 ;;
        fixed-dt) set_line "$control" 23 1; set_line "$control" 26 "2.5d-03" ;;
        converged-pressure) set_line "$solver" 6 "1.0d-10"; set_line "$solver" 9 "1.0d-14" ;;
        ssor-rcm) settings=$'preconditioner = ssor\noperator_format = csr\nreordering = rcm' ;;
        ic0-rcm) settings=$'preconditioner = ic0\noperator_format = csr\nreordering = rcm' ;;
        ilu0-rcm) settings=$'preconditioner = ilu0\noperator_format = csr\nreordering = rcm' ;;
        bicgstab-ilu0) settings=$'method = bicgstab\npreconditioner = ilu0\noperator_format = csr\nreordering = rcm' ;;
        gmres-ilu0) settings=$'method = gmres\npreconditioner = ilu0\noperator_format = csr\nreordering = rcm' ;;
        sa|k-omega-sst|k-epsilon|launder-sharma-k-epsilon)
            local model="$config"; [[ $config == sa ]] && model=spalart-allmaras
            printf 'model %s\n' "$model" >"$dir/system/hinterkante.turbulence" ;;
        *) echo "unknown configuration $config" >&2; exit 2 ;;
    esac
    if [[ -n "$settings" ]]; then
        # keep the ten positional non-empty lines, then the settings
        awk '{ line = $0; sub(/\r$/, "", line) } line ~ /[^ \t]/ { n++ } { print } n == 10 { exit }' "$solver" >"$solver.tmp"
        printf '%s\n' "$settings" >>"$solver.tmp" && mv "$solver.tmp" "$solver"
    fi
}

filter_log() {
    grep -v -e "Time Taken" -e "^Mesh quality: " -e "^  Mesh formats: " -e "^ Case Directory: " "$1" || true
}

configs=(legacy steady-local-dt fixed-dt converged-pressure ssor-rcm ic0-rcm ilu0-rcm bicgstab-ilu0 gmres-ilu0
         sa k-omega-sst k-epsilon launder-sharma-k-epsilon)
failed=0
printf '%-26s %s\n' "configuration" "result"
for config in "${configs[@]}"; do
    for side in baseline current; do
        exe="$bin"; [[ $side == baseline ]] && exe="$base_bin"
        prepare "$work/$side-$config" "$config"
        "$exe" "$work/$side-$config" >"$work/$side-$config.log" 2>&1 || true
    done
    result="IDENTICAL"
    for f in hinterkante.var hinterkante.turb.var; do
        b="$work/baseline-$config/results/$f" c="$work/current-$config/results/$f"
        if [[ -e "$b" || -e "$c" ]]; then
            cmp -s "$b" "$c" || result="DIFFERENT ($f)"
        fi
    done
    [[ -e "$work/baseline-$config/results/hinterkante.var" ]] || result="NO RESULTS"
    if [[ $result == IDENTICAL ]] && ! diff <(filter_log "$work/baseline-$config.log") \
                                            <(filter_log "$work/current-$config.log") >"$work/$config.diff"; then
        result="DIFFERENT (log, see $config.diff)"
        keep=1
    fi
    [[ $result == IDENTICAL ]] || failed=1
    printf '%-26s %s\n' "$config" "$result"
done
exit $failed
