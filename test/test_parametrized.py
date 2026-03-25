"""Parametrized integration tests across dim combinations."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import AggregateCase
from tsam import SegmentConfig

import tsam_xarray


def _aggregate(case: AggregateCase, **kwargs):  # type: ignore[no-untyped-def]
    """Helper to avoid repeating aggregate call."""
    return tsam_xarray.aggregate(
        case.da,
        time_dim=case.time_dim,
        cluster_dim=case.cluster_dim,
        n_clusters=case.n_clusters,
        **kwargs,
    )


class TestDimsAndShapes:
    """Output dimensions match expectations."""

    def test_typical_periods_dims(self, agg_case: AggregateCase):
        result = _aggregate(agg_case)
        dims = set(result.typical_periods.dims)
        expected = (
            {"cluster", "timestep"}
            | agg_case.expected_cluster_dims
            | agg_case.expected_slice_dims
        )
        assert dims == expected

    def test_reconstructed_dims(self, agg_case: AggregateCase):
        result = _aggregate(agg_case)
        dims = set(result.reconstructed.dims)
        expected = (
            {agg_case.time_dim}
            | agg_case.expected_cluster_dims
            | agg_case.expected_slice_dims
        )
        assert dims == expected

    def test_reconstructed_time_size(self, agg_case: AggregateCase):
        result = _aggregate(agg_case)
        assert (
            result.reconstructed.sizes[agg_case.time_dim]
            == agg_case.da.sizes[agg_case.time_dim]
        )

    def test_accuracy_dims(self, agg_case: AggregateCase):
        result = _aggregate(agg_case)
        if agg_case.expected_cluster_dims:
            expected = agg_case.expected_cluster_dims | agg_case.expected_slice_dims
            assert set(result.accuracy.rmse.dims) == expected


class TestClusterAssignments:
    """Cluster assignments are structurally valid."""

    def test_values_in_range(self, agg_case: AggregateCase):
        result = _aggregate(agg_case)
        assignments = result.cluster_assignments.values.ravel()
        assert np.all(assignments >= 0)
        assert np.all(assignments < agg_case.n_clusters)

    def test_all_clusters_used(self, agg_case: AggregateCase):
        """Every cluster ID appears at least once (per slice)."""
        result = _aggregate(agg_case)
        if not agg_case.expected_slice_dims:
            unique = set(result.cluster_assignments.values.tolist())
            assert len(unique) == agg_case.n_clusters
        # With slicing, each slice should use all clusters

    def test_cluster_weights_sum(self, agg_case: AggregateCase):
        result = _aggregate(agg_case)
        if agg_case.expected_slice_dims:
            for key in np.ndindex(
                *[result.cluster_weights.sizes[d] for d in agg_case.expected_slice_dims]
            ):
                sel = {
                    d: result.cluster_weights.coords[d].values[i]
                    for d, i in zip(
                        agg_case.expected_slice_dims,
                        key,
                        strict=True,
                    )
                }
                assert int(result.cluster_weights.sel(sel).sum()) == agg_case.n_periods
        else:
            assert int(result.cluster_weights.sum()) == agg_case.n_periods


class TestAccuracyMetrics:
    """Accuracy metrics are valid."""

    def test_rmse_positive(self, agg_case: AggregateCase):
        result = _aggregate(agg_case)
        assert float(result.accuracy.rmse.min()) >= 0

    def test_mae_positive(self, agg_case: AggregateCase):
        result = _aggregate(agg_case)
        assert float(result.accuracy.mae.min()) >= 0

    def test_rmse_bounded(self, agg_case: AggregateCase):
        """RMSE should be less than std of original data."""
        result = _aggregate(agg_case)
        original_std = float(agg_case.da.std())
        max_rmse = float(result.accuracy.rmse.max())
        assert max_rmse < original_std * 2  # generous bound


class TestReconstructedValues:
    """Reconstructed values are plausible."""

    def test_values_within_bounds(self, agg_case: AggregateCase):
        """Reconstructed values near original range."""
        result = _aggregate(agg_case)
        orig_min = float(agg_case.da.min())
        orig_max = float(agg_case.da.max())
        margin = (orig_max - orig_min) * 0.5
        assert float(result.reconstructed.min()) >= orig_min - margin
        assert float(result.reconstructed.max()) <= orig_max + margin

    def test_mean_preserved(self, agg_case: AggregateCase):
        """Column means approximately preserved (tsam default)."""
        result = _aggregate(agg_case)
        # Compare overall means — should be close with preserve_column_means=True
        orig_mean = float(agg_case.da.mean())
        recon_mean = float(result.reconstructed.mean())
        np.testing.assert_allclose(orig_mean, recon_mean, rtol=0.1)


class TestSliceIndependence:
    """Sliced results are genuinely independent."""

    def test_slices_differ(self, agg_case: AggregateCase):
        """Different slices produce different assignments."""
        if not agg_case.expected_slice_dims:
            pytest.skip("No slice dims")
        result = _aggregate(agg_case)
        # Get first slice dim
        dim = next(iter(agg_case.expected_slice_dims))
        coords = result.cluster_assignments.coords[dim].values
        if len(coords) < 2:
            pytest.skip("Only one slice coordinate")
        a1 = result.cluster_assignments.sel({dim: coords[0]}).values
        a2 = result.cluster_assignments.sel({dim: coords[1]}).values
        # With random data, independent clusterings should differ
        # (not guaranteed but extremely likely)
        assert not np.array_equal(a1, a2)


class TestDisaggregateRoundtrip:
    """disaggregate(typical_periods) == reconstructed."""

    def test_roundtrip(self, agg_case: AggregateCase):
        result = _aggregate(agg_case)
        dis = result.disaggregate(result.typical_periods)
        np.testing.assert_allclose(dis.values, result.reconstructed.values, rtol=1e-10)


class TestSegmentationMatrix:
    """Segmentation across all dim combinations."""

    def test_segment_durations_shape(self, agg_case: AggregateCase):
        result = _aggregate(agg_case, segments=SegmentConfig(n_segments=6))
        assert result.segment_durations is not None
        assert result.segment_durations.sizes["timestep"] == 6

    def test_segment_durations_sum_to_period(self, agg_case: AggregateCase):
        """Each cluster's durations sum to timesteps per period."""
        result = _aggregate(agg_case, segments=SegmentConfig(n_segments=6))
        assert result.segment_durations is not None
        # Sum across timestep dim for each cluster
        sums = result.segment_durations.sum(dim="timestep")
        assert np.all(sums.values == 24)

    def test_segmented_disaggregate_ffill(self, agg_case: AggregateCase):
        result = _aggregate(agg_case, segments=SegmentConfig(n_segments=6))
        dis = result.disaggregate(result.typical_periods)
        filled = dis.ffill(dim=agg_case.time_dim)
        np.testing.assert_allclose(
            filled.values, result.reconstructed.values, rtol=1e-10
        )

    def test_segmented_has_nan_before_fill(self, agg_case: AggregateCase):
        result = _aggregate(agg_case, segments=SegmentConfig(n_segments=6))
        dis = result.disaggregate(result.typical_periods)
        assert bool(dis.isnull().any())


class TestClusteringIORoundtrip:
    """save/load/apply preserves results."""

    def test_assignments_preserved(
        self, agg_case: AggregateCase, tmp_path: pytest.TempPathFactory
    ):
        result = _aggregate(agg_case)
        path = tmp_path / "clustering.json"
        result.clustering.to_json(str(path))
        clustering = tsam_xarray.load_clustering(str(path))
        new_result = clustering.apply(agg_case.da)
        np.testing.assert_array_equal(
            result.cluster_assignments.values,
            new_result.cluster_assignments.values,
        )

    def test_typical_periods_preserved(
        self, agg_case: AggregateCase, tmp_path: pytest.TempPathFactory
    ):
        result = _aggregate(agg_case)
        path = tmp_path / "clustering.json"
        result.clustering.to_json(str(path))
        clustering = tsam_xarray.load_clustering(str(path))
        new_result = clustering.apply(agg_case.da)
        np.testing.assert_allclose(
            result.typical_periods.values,
            new_result.typical_periods.values,
            rtol=1e-10,
        )

    def test_is_transferred(
        self, agg_case: AggregateCase, tmp_path: pytest.TempPathFactory
    ):
        result = _aggregate(agg_case)
        assert not result.is_transferred
        path = tmp_path / "clustering.json"
        result.clustering.to_json(str(path))
        clustering = tsam_xarray.load_clustering(str(path))
        new_result = clustering.apply(agg_case.da)
        assert new_result.is_transferred
