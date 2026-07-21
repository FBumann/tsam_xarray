# Changelog

## [0.6.2](https://github.com/FBumann/tsam_xarray/compare/v0.6.1...v0.6.2) (2026-07-21)


### Features

* first-class original-vs-reconstructed comparison on AggregationResult ([#93](https://github.com/FBumann/tsam_xarray/issues/93)) ([b3fd174](https://github.com/FBumann/tsam_xarray/commit/b3fd174cdbe379eb1de586aba5bfb75c0e4d21bb))


### Bug Fixes

* align reconstructed coordinate order with input (+ comparison recipe) ([#97](https://github.com/FBumann/tsam_xarray/issues/97)) ([afd6818](https://github.com/FBumann/tsam_xarray/commit/afd6818715704822bacfad6edfa5c6595e9f8fc4))

## [0.6.1](https://github.com/FBumann/tsam_xarray/compare/v0.6.0...v0.6.1) (2026-07-09)


### Features

* add cluster_on to select which coordinates drive clustering ([#89](https://github.com/FBumann/tsam_xarray/issues/89)) ([ddff73b](https://github.com/FBumann/tsam_xarray/commit/ddff73b24243e53d5a182b9e8c794e2b57a8d1b7))
* validate extremes options against cluster_on ([#91](https://github.com/FBumann/tsam_xarray/issues/91)) ([2fa5a91](https://github.com/FBumann/tsam_xarray/commit/2fa5a91c5222f8b6dd6932d9572dec62e90f595b))

## [0.6.0](https://github.com/FBumann/tsam_xarray/compare/v0.5.2...v0.6.0) (2026-05-27)


### ⚠ BREAKING CHANGES

* **ClusteringResult:** `time_coords` attribute and the `_time_coords_to_dict` / `_time_coords_from_dict` helpers have been removed. The time index now lives inside the tsam `ClusteringResult` payload (`time_index`) and flows through `aggregate` / `disaggregate` natively. Pre-0.6 JSONs are still loadable — the legacy outer `time_coords` field is forwarded to the inner `time_index` with a `DeprecationWarning`. Re-save to silence the warning. ([#83](https://github.com/FBumann/tsam_xarray/pull/83))


### Refactors

* Reuse tsam 3.4's `DatetimeIndex` round-trip in `disaggregate`, dropping the parallel `time_coords` field, the compact serialization helpers, and the manual `MultiIndex` truncation in `_disaggregate_single`. ([#83](https://github.com/FBumann/tsam_xarray/pull/83))


### Dependencies

* Bump minimum `tsam` to `>=3.4.0` (required for the `time_index` round-trip above).
* Bump `googleapis/release-please-action` from 4 to 5. ([#82](https://github.com/FBumann/tsam_xarray/pull/82))
* Bump `dependabot/fetch-metadata` from 2 to 3. ([#81](https://github.com/FBumann/tsam_xarray/pull/81))

## [0.5.2](https://github.com/FBumann/tsam_xarray/compare/v0.5.1...v0.5.2) (2026-04-01)


### Features

* compact time_coords serialization in ClusteringResult JSON ([#79](https://github.com/FBumann/tsam_xarray/issues/79)) ([bac9fd1](https://github.com/FBumann/tsam_xarray/commit/bac9fd17ee28a48fd6f51ce16fea0df883cccb99))

## [0.5.1](https://github.com/FBumann/tsam_xarray/compare/v0.5.0...v0.5.1) (2026-03-31)


### Features

* add to_dict/from_dict on ClusteringResult ([#75](https://github.com/FBumann/tsam_xarray/issues/75)) ([24723a8](https://github.com/FBumann/tsam_xarray/commit/24723a82b0daa9eacdaa98e2ee300b9e44697bd6))

## [0.5.0](https://github.com/FBumann/tsam_xarray/compare/v0.4.0...v0.5.0) (2026-03-31)


### ⚠ BREAKING CHANGES

* find_best_combination renamed to grid_search. Old name still works but emits FutureWarning.

### Features

* rename grid_search, add timesteps param, update notebook ([#73](https://github.com/FBumann/tsam_xarray/issues/73)) ([c54c321](https://github.com/FBumann/tsam_xarray/commit/c54c3216991060a4e9d500ac51757467bceaa8e8))

## [0.4.0](https://github.com/FBumann/tsam_xarray/compare/v0.3.1...v0.4.0) (2026-03-31)


### ⚠ BREAKING CHANGES

* AccuracyMetrics.weighted_rmse, weighted_mae, weighted_rmse_duration changed from float to xr.DataArray.

### Features

* make weighted accuracy metrics per-slice DataArrays ([#71](https://github.com/FBumann/tsam_xarray/issues/71)) ([5540913](https://github.com/FBumann/tsam_xarray/commit/5540913ce2cb9dd95c440250d6390426fa977f4b))

## [0.3.1](https://github.com/FBumann/tsam_xarray/compare/v0.3.0...v0.3.1) (2026-03-31)


### Features

* add cluster_centers, segment_assignments, segment_centers to ClusteringResult ([#68](https://github.com/FBumann/tsam_xarray/issues/68)) ([8b0087e](https://github.com/FBumann/tsam_xarray/commit/8b0087e55b888130e329032d26c795fe33cd63cc))

## [0.3.0](https://github.com/FBumann/tsam_xarray/compare/v0.2.0...v0.3.0) (2026-03-30)


### ⚠ BREAKING CHANGES

* ClusteringInfo renamed to ClusteringResult. ClusteringInfo remains as a backwards-compatible alias.
* ClusteringInfo renamed to ClusteringResult. ClusteringInfo remains as a backwards-compatible alias.

### Features

* rename ClusteringInfo to ClusteringResult with cached DataArray properties ([#64](https://github.com/FBumann/tsam_xarray/issues/64)) ([5db89d0](https://github.com/FBumann/tsam_xarray/commit/5db89d0e2eb92b0b3664c2757b353d3107cb6e88))
* store weighted accuracy metrics and compact repr ([#65](https://github.com/FBumann/tsam_xarray/issues/65)) ([5327ecd](https://github.com/FBumann/tsam_xarray/commit/5327ecd79b5b35b5ffad3ef97e45db422d469f61))

## [0.2.0](https://github.com/FBumann/tsam_xarray/compare/v0.1.1...v0.2.0) (2026-03-30)


### ⚠ BREAKING CHANGES

* add ClusteringInfo.disaggregate() (requires tsam >=3.3.0) ([#62](https://github.com/FBumann/tsam_xarray/issues/62))

### Features

* add ClusteringInfo.disaggregate() (requires tsam &gt;=3.3.0) ([#62](https://github.com/FBumann/tsam_xarray/issues/62)) ([945210a](https://github.com/FBumann/tsam_xarray/commit/945210a17e2069f7657515d2e7ffd2a6a5f61c74))

## [0.1.1](https://github.com/FBumann/tsam_xarray/compare/v0.1.0...v0.1.1) (2026-03-27)


### Features

* add Python 3.11 support ([#60](https://github.com/FBumann/tsam_xarray/issues/60)) ([7434a52](https://github.com/FBumann/tsam_xarray/commit/7434a52ee90b279214ac25f894910343c220cc90))

## [0.1.0](https://github.com/FBumann/tsam_xarray/compare/v0.0.4...v0.1.0) (2026-03-27)

Initial release of tsam_xarray — lightweight xarray wrapper for tsam time series aggregation.


### Features

* find_optimal_combination with cross-slice RMSE ([#50](https://github.com/FBumann/tsam_xarray/issues/50))
* clustering IO and apply() ([#35](https://github.com/FBumann/tsam_xarray/issues/35)) ([2068e12](https://github.com/FBumann/tsam_xarray/commit/2068e123892d81008d8fbfa6396a4e6dd51e7f2d))
* dict-based weights API ([#31](https://github.com/FBumann/tsam_xarray/issues/31)) ([1141f61](https://github.com/FBumann/tsam_xarray/commit/1141f6113448f151f87ac65950aba087a479da1b))
* find_optimal_combination with cross-slice RMSE ([#50](https://github.com/FBumann/tsam_xarray/issues/50)) ([ae4f281](https://github.com/FBumann/tsam_xarray/commit/ae4f281ada28a01b8d8d3a3be759658b98bd7ede))
* implement aggregate() API with stack_dims and slice_dims ([#9](https://github.com/FBumann/tsam_xarray/issues/9)) ([dc1070c](https://github.com/FBumann/tsam_xarray/commit/dc1070c2aec930214d1896c2f86782e0e5301add))
* input data validation ([#32](https://github.com/FBumann/tsam_xarray/issues/32)) ([69fbc51](https://github.com/FBumann/tsam_xarray/commit/69fbc5119a5a683803a1cfba8e00507e4c94cefb))
* per-dimension weight mapping for multi-dim cluster_dim ([#26](https://github.com/FBumann/tsam_xarray/issues/26)) ([18e62f5](https://github.com/FBumann/tsam_xarray/commit/18e62f555ffe9bcf99c5156101cc903347625b38))
* segment_durations as DataArray and disaggregate() method ([#28](https://github.com/FBumann/tsam_xarray/issues/28)) ([9358696](https://github.com/FBumann/tsam_xarray/commit/9358696b81011fe7708a51cd71962a9f00e27e02))
* validate consistent cluster counts across slices ([#44](https://github.com/FBumann/tsam_xarray/issues/44)) ([0b98ea2](https://github.com/FBumann/tsam_xarray/commit/0b98ea2c3cc3e9cfcc9b9cc23be8e0eb1a5464c8))


### Bug Fixes

* allow 1D DataArray clustering with cluster_dim=() ([#38](https://github.com/FBumann/tsam_xarray/issues/38)) ([9a1a46b](https://github.com/FBumann/tsam_xarray/commit/9a1a46b0d51b63c4f062d5c5871deffd9215626d)), closes [#36](https://github.com/FBumann/tsam_xarray/issues/36)
* configure release-please for 0.0.1-alpha prerelease ([#29](https://github.com/FBumann/tsam_xarray/issues/29)) ([8036b30](https://github.com/FBumann/tsam_xarray/commit/8036b30041a2c6df6329481ec36cd62efb8e92bd))
* correct release-please option name ([#57](https://github.com/FBumann/tsam_xarray/issues/57)) ([2e385bc](https://github.com/FBumann/tsam_xarray/commit/2e385bce54a629534dab4827c8723b23cadc9ce5))
* remove alpha suffix from release-please manifest ([#54](https://github.com/FBumann/tsam_xarray/issues/54)) ([9615ec1](https://github.com/FBumann/tsam_xarray/commit/9615ec1b42f7d874ec0a6263534a50fa88f81e27))
* replace remaining my-package placeholder in docs/index.md ([e4f5dc0](https://github.com/FBumann/tsam_xarray/commit/e4f5dc0229d24b75828f3ba02a85db3417c17ef5))

## [0.0.4-alpha.0](https://github.com/FBumann/tsam_xarray/compare/v0.0.3-alpha.0...v0.0.4-alpha.0) (2026-03-25)


### Features

* validate consistent cluster counts across slices ([#44](https://github.com/FBumann/tsam_xarray/issues/44)) ([0b98ea2](https://github.com/FBumann/tsam_xarray/commit/0b98ea2c3cc3e9cfcc9b9cc23be8e0eb1a5464c8))

## [0.0.3-alpha.0](https://github.com/FBumann/tsam_xarray/compare/v0.0.2-alpha.0...v0.0.3-alpha.0) (2026-03-25)


### Features

* clustering IO and apply() ([#35](https://github.com/FBumann/tsam_xarray/issues/35)) ([2068e12](https://github.com/FBumann/tsam_xarray/commit/2068e123892d81008d8fbfa6396a4e6dd51e7f2d))


### Bug Fixes

* allow 1D DataArray clustering with cluster_dim=() ([#38](https://github.com/FBumann/tsam_xarray/issues/38)) ([9a1a46b](https://github.com/FBumann/tsam_xarray/commit/9a1a46b0d51b63c4f062d5c5871deffd9215626d)), closes [#36](https://github.com/FBumann/tsam_xarray/issues/36)

## [0.0.2-alpha.0](https://github.com/FBumann/tsam_xarray/compare/v0.0.1-alpha.0...v0.0.2-alpha.0) (2026-03-25)


### Features

* input data validation ([#32](https://github.com/FBumann/tsam_xarray/issues/32)) ([69fbc51](https://github.com/FBumann/tsam_xarray/commit/69fbc5119a5a683803a1cfba8e00507e4c94cefb))

## 0.0.1-alpha.0 (2026-03-25)


### Features

* dict-based weights API ([#31](https://github.com/FBumann/tsam_xarray/issues/31)) ([1141f61](https://github.com/FBumann/tsam_xarray/commit/1141f6113448f151f87ac65950aba087a479da1b))
* implement aggregate() API with stack_dims and slice_dims ([#9](https://github.com/FBumann/tsam_xarray/issues/9)) ([dc1070c](https://github.com/FBumann/tsam_xarray/commit/dc1070c2aec930214d1896c2f86782e0e5301add))
* per-dimension weight mapping for multi-dim cluster_dim ([#26](https://github.com/FBumann/tsam_xarray/issues/26)) ([18e62f5](https://github.com/FBumann/tsam_xarray/commit/18e62f555ffe9bcf99c5156101cc903347625b38))
* segment_durations as DataArray and disaggregate() method ([#28](https://github.com/FBumann/tsam_xarray/issues/28)) ([9358696](https://github.com/FBumann/tsam_xarray/commit/9358696b81011fe7708a51cd71962a9f00e27e02))


### Bug Fixes

* configure release-please for 0.0.1-alpha prerelease ([#29](https://github.com/FBumann/tsam_xarray/issues/29)) ([8036b30](https://github.com/FBumann/tsam_xarray/commit/8036b30041a2c6df6329481ec36cd62efb8e92bd))
* replace remaining my-package placeholder in docs/index.md ([e4f5dc0](https://github.com/FBumann/tsam_xarray/commit/e4f5dc0229d24b75828f3ba02a85db3417c17ef5))

## Changelog
