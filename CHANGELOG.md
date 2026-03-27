# Changelog

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
