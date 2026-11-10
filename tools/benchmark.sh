#!/usr/bin/env bash
# Benchmark the pressure solver configurations on the bundled Hinterkante case.
#
# Usage:
#   tools/benchmark.sh [options] [--] ["key = value" ...]
#       Runs one configuration; the "key = value" arguments are written as the
#       optional settings block of <case>.solver (see README, "Pressure solver
#       options"). Without settings: the legacy PCG.
#   tools/benchmark.sh --suite [options]
#       Runs the comparison table of the README (legacy, and ssor / ic0 / ilu0
#       with RCM, each with preconditioner_reuse = rebuild and = rescale),
#       interleaved, best of --repeats runs.
#
# Options:
#   --bin PATH        solver executable (default: build/HydraMa2D)
#   --steps N         time steps (default: 200)
#   --repeats N       runs per configuration, the best wall time is reported (default: 5)
#   --rel R           relative tolerance of the pressure solve (default: bundled 0.1)
#   --abs A           absolute tolerance (default: bundled 1.0d-06)
#   --fixed-dt DT     fixed time step DT (dtfixed = 1) instead of the global CFL step
#
# Output per configuration: best wall time ("Time Taken" of the solver),
# total pressure iterations, and the setup statistics of the pressure solver.
# Each run works in a private temporary copy of the case.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case_src="$here/Hinterkante.dev.Re100"
bin="$here/build/HydraMa2D"
steps=200
repeats=5
rel=""
abs=""
fixed_dt=""
suite=0
settings=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bin) bin="$2"; shift 2 ;;
        --steps) steps="$2"; shift 2 ;;
        --repeats) repeats="$2"; shift 2 ;;
        --rel) rel="$2"; shift 2 ;;
        --abs) abs="$2"; shift 2 ;;
        --fixed-dt) fixed_dt="$2"; shift 2 ;;
        --suite) suite=1; shift ;;
        --) shift; settings+=("$@"); break ;;
        -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
        *) settings+=("$1"); shift ;;
    esac
done

[[ -x "$bin" ]] || { echo "solver executable not found: $bin (build first, or pass --bin)" >&2; exit 1; }

work="$(mktemp -d "${TMPDIR:-/tmp}/hydrama2d-bench.XXXXXX")"
trap 'rm -rf "$work"' EXIT

# set_line FILE INDEX VALUE: replace the INDEX-th (0-based) non-empty line.
set_line() {
    awk -v idx="$2" -v val="$3" '
        { line = $0; sub(/\r$/, "", line) }
        line ~ /[^ \t]/ { if (n == idx) { print val; n++; next } n++ }
        { print }' "$1" > "$1.tmp" && mv "$1.tmp" "$1"
}

# keep_positional FILE: drop everything after the 10th non-empty line.
keep_positional() {
    awk '{ line = $0; sub(/\r$/, "", line) } line ~ /[^ \t]/ { n++ } { print } n == 10 { exit }' "$1" > "$1.tmp" &&
        mv "$1.tmp" "$1"
}

# prepare DIR SETTING...: copy of the case with the benchmark settings.
prepare() {
    local dir="$1"; shift
    rm -rf "$dir"
    mkdir -p "$dir"
    cp -r "$case_src/constant" "$case_src/system" "$dir/"
    mkdir -p "$dir/results"
    local control="$dir/system/hinterkante.control" solver="$dir/system/hinterkante.solver"
    set_line "$control" 5 "$steps"   # number of time steps
    set_line "$control" 17 "$steps"  # write output at the last step only
    set_line "$control" 29 "$steps"  # log at the last step (cumulative iterations)
    if [[ -n "$fixed_dt" ]]; then
        set_line "$control" 23 1
        set_line "$control" 26 "$fixed_dt"
    fi
    [[ -n "$rel" ]] && set_line "$solver" 6 "$rel"
    [[ -n "$abs" ]] && set_line "$solver" 9 "$abs"
    keep_positional "$solver"
    local s
    for s in "$@"; do printf '%s\n' "$s" >> "$solver"; done
}

# run_once DIR -> prints "seconds iterations|setup line"
run_once() {
    local log
    log="$("$bin" "$1" 2>&1)" || { echo "solver failed:" >&2; echo "$log" >&2; exit 1; }
    local t it st
    t="$(awk '/Time Taken/ { print $4 }' <<< "$log")"
    it="$(awk '/IterCumulative:/ { v = $NF } END { print v }' <<< "$log")"
    st="$(sed -n 's/^ Pressure solver setup: //p' <<< "$log")"
    printf '%s %s|%s\n' "$t" "$it" "$st"
}

min() { awk -v a="$1" -v b="$2" 'BEGIN { print (a < b) ? a : b }'; }

echo "HydraMa2D pressure solver benchmark: $steps steps, best of $repeats" \
     "(rel ${rel:-0.1}, abs ${abs:-1.0d-06}${fixed_dt:+, fixed dt $fixed_dt})"

if [[ $suite -eq 0 ]]; then
    prepare "$work/case" "${settings[@]+"${settings[@]}"}"
    best=1e30; info=""
    for ((r = 0; r < repeats; ++r)); do
        out="$(run_once "$work/case")"
        best="$(min "$best" "${out%% *}")"; info="${out#* }"
    done
    label="${settings[*]:-legacy PCG (defaults)}"
    printf '%-60s %7.3f s  iterations %6s  %s\n' "$label" "$best" "${info%%|*}" "${info#*|}"
    exit 0
fi

names=(legacy ssor-rebuild ssor-rescale ic0-rebuild ic0-rescale ilu0-rebuild ilu0-rescale)
declare -A opts=(
    [legacy]=""
    [ssor-rebuild]="preconditioner = ssor|reordering = rcm|preconditioner_reuse = rebuild"
    [ssor-rescale]="preconditioner = ssor|reordering = rcm|preconditioner_reuse = rescale"
    [ic0-rebuild]="preconditioner = ic0|reordering = rcm|preconditioner_reuse = rebuild"
    [ic0-rescale]="preconditioner = ic0|reordering = rcm|preconditioner_reuse = rescale"
    [ilu0-rebuild]="preconditioner = ilu0|reordering = rcm|preconditioner_reuse = rebuild"
    [ilu0-rescale]="preconditioner = ilu0|reordering = rcm|preconditioner_reuse = rescale"
)
declare -A best info
for n in "${names[@]}"; do
    IFS='|' read -r -a s <<< "${opts[$n]}"
    prepare "$work/$n" "${s[@]+"${s[@]}"}"
    best[$n]=1e30
done
for ((r = 0; r < repeats; ++r)); do # interleaved: machine noise affects all alike
    for n in "${names[@]}"; do
        out="$(run_once "$work/$n")"
        best[$n]="$(min "${best[$n]}" "${out%% *}")"; info[$n]="${out#* }"
    done
done
for n in "${names[@]}"; do
    printf '%-14s %7.3f s  iterations %6s  %s\n' "$n" "${best[$n]}" "${info[$n]%%|*}" "${info[$n]#*|}"
done
