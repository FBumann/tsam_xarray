# Benchmarks

Performance suite for `aggregate()`, built on
[pytest-benchmem](https://github.com/fluxopt/pytest-benchmem) (pytest-benchmark
plus memray peak-memory columns). Not part of the regular test suite — CI runs
it via `.github/workflows/benchmarks.yaml`, which posts a head-vs-base delta
table on every PR and uploads an interactive plot artifact.

Three layers (see the module docstring in `test_bench_aggregate.py`):
wrapper-stage micro-benchmarks with many rounds (the only code this repo's
PRs can regress — ~90% of an end-to-end run is inside tsam), config cost
ratios at reduced size, and a few full-pipeline sentinels at production
size. Cases are a deterministic grid, not random: `benchmem compare`
matches runs by test ID, so IDs must be stable across runs for baselines
to work.

## Run locally

```bash
uv run --with "pytest-benchmem[plot]" pytest benchmarks/ \
    -o addopts="" -p no:xdist --benchmark-only --benchmark-memory
```

(`-o addopts="" -p no:xdist` because the project's xdist config auto-disables
benchmarks.)

## Large tier

Production-sized cases (`test_large_*`, several seconds and hundreds of MiB
peak each) are skipped by default — in CI too. Run them locally before
dependency bumps or performance work:

```bash
uv run --with "pytest-benchmem[plot]" pytest benchmarks/ --large -k large \
    -o addopts="" -p no:xdist --benchmark-only --benchmark-memory
```

## Baselines

Save a named baseline (stored in `.benchmarks/`, gitignored — machine-specific):

```bash
... --benchmark-save=baseline
```

Compare a later run against it, failing on >10% mean regressions:

```bash
... --benchmark-compare=0001 --benchmark-compare-fail=mean:10%
```

## Reports and plots

```bash
uv run --with "pytest-benchmem[plot]" benchmem compare a.json b.json --diff
uv run --with "pytest-benchmem[plot]" benchmem plot a.json b.json -o plot.html --open
```

`benchmem flamegraph <test-id>` renders a memory flamegraph when the run was
saved with `--benchmark-memory-profile`.
