# Benchmarks

Performance suite for `aggregate()`, built on
[pytest-benchmem](https://github.com/fluxopt/pytest-benchmem) (pytest-benchmark
plus memray peak-memory columns). Not part of the regular test suite — CI runs
it via `.github/workflows/benchmarks.yaml`, which posts a head-vs-base delta
table on every PR and uploads an interactive plot artifact.

The suite focuses on the production config — hierarchical clustering,
`Distribution` representations, `include_period_sums=False`, extremes — and
scales columns, slices, and days under it. Cases are a deterministic grid, not
random: `benchmem compare` matches runs by test ID, so IDs must be stable
across runs for baselines to work.

## Run locally

```bash
uv run --with "pytest-benchmem[plot]" pytest benchmarks/ \
    -o addopts="" -p no:xdist --benchmark-only --benchmark-memory
```

(`-o addopts="" -p no:xdist` because the project's xdist config auto-disables
benchmarks.)

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
