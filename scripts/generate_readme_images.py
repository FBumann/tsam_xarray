"""Generate images for README.md."""

import xarray_plotly  # noqa: F401

import tsam_xarray
from tsam_xarray._sample_data import sample_energy_data


def generate_input_plot() -> None:
    """Multi-dimensional input data plot."""
    da = sample_energy_data(n_days=30).sel(scenario="low")
    fig = da.plotly.line(x="time", color="variable", facet_col="region")
    fig.update_layout(
        height=250,
        width=850,
        margin=dict(t=40, b=25, l=50, r=20),
        template="plotly_white",
        font=dict(size=11),
        legend=dict(orientation="h", y=-0.15),
        title_text=("Input: 3 variables x 3 regions x 2 scenarios x 720 hours"),
        title_x=0.5,
        title_font_size=13,
    )
    fig.update_xaxes(tickformat="%b %d")
    fig.update_traces(line_width=0.8)
    fig.write_image("docs/assets/multi-dim-input.png", scale=2)
    print("Saved docs/assets/multi-dim-input.png")


def generate_metrics_plot() -> None:
    """Per-column RMSE heatmap across all dimensions."""
    da = sample_energy_data(n_days=30)
    r = tsam_xarray.aggregate(
        da,
        time_dim="time",
        cluster_dim="variable",
        n_clusters=4,
    )
    fig = r.accuracy.rmse.plotly.imshow(
        x="variable",
        y="region",
        facet_col="scenario",
        text_auto=".2f",
        color_continuous_scale="YlOrRd",
    )
    fig.update_layout(
        height=280,
        width=650,
        margin=dict(t=40, b=20, l=60, r=20),
        template="plotly_white",
        font=dict(size=12),
        title_text=("Per-column RMSE — faceted by scenario (independent clustering)"),
        title_x=0.5,
        title_font_size=13,
    )
    fig.write_image("docs/assets/multi-dim-metrics.png", scale=2)
    print("Saved docs/assets/multi-dim-metrics.png")


if __name__ == "__main__":
    generate_input_plot()
    generate_metrics_plot()
