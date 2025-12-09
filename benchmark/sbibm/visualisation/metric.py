from typing import Any, Dict, List, Optional

import altair as alt
import deneb as den
import pandas as pd
import seaborn as sns

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any


def fig_metric_seaborn(
    df: pd.DataFrame,
    metric: str,
    title: Optional[str] = None,
    title_dx: int = 0,
    width: Optional[int] = None,
    height: Optional[int] = None,
    labels: bool = True,
    keywords: Dict[str, Any] = {},
    style: Dict[str, Any] = {},
    default_color: str = "#000000",
    colors_dict: Dict[str, Any] = {},
    config: Optional[str] = None,
):
    # ---- Color handling (same logic as original) ----
    palette = {}
    for algorithm in df.algorithm.unique():
        stripped = algorithm.strip()
        palette[algorithm] = colors_dict.get(stripped, default_color)

    # ---- Default sizes ----
    n_cols = len(df.algorithm.unique())

    if config == "manuscript":
        plot_width = (700 / n_cols) if width is None else width
        plot_height = 65 if height is None else height
        font_size = 12
    elif config == "streamlit":
        plot_width = width
        plot_height = height
        font_size = 16
    else:
        plot_width = width or 6
        plot_height = height or 4
        font_size = 12

    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=font_size / 12)

    # ---- Facet grid (equivalent to column="algorithm:N") ----
    g = sns.FacetGrid(
        df.sort_values("algorithm"),
        col="algorithm",
        sharey=True,
        height=plot_height / 100 if config == "manuscript" else plot_height,
        aspect=(plot_width / plot_height) if plot_width and plot_height else 1.5,
    )

    g.map_dataframe(
        sns.lineplot,
        x="num_simulations",
        y=metric,
        estimator="mean",
        errorbar=("ci", 95),
        marker="o",
        palette=palette,
        hue=None,
        **keywords
    )

    # ---- Axis logic ----
    for ax in g.axes.flat:
        # X as ordinal look (like Altair :O)
        ax.set_xticks(sorted(df["num_simulations"].unique()))
        ax.tick_params(axis="x", rotation=270)

        # Y axis label logic
        if metric == "MMD":
            ax.set_ylabel("MMD²")
        else:
            ax.set_ylabel(metric)

        # C2ST limits
        if "C2ST" in metric:
            ax.set_ylim(0.5, 1.0)

        # RT special handling
        if metric == "RT":
            ax.set_yscale("log")
            ax.set_ylim(0.001, 1000.0)
            ax.set_yticks([0.001, 0.01, 0.1, 1, 10, 100, 1000])

        # Grid + spines to match Altair look
        ax.grid(True, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ---- Remove legend (similar to original behavior) ----
    g.add_legend = False

    # ---- Column titles ----
    if labels:
        g.set_titles(col_template="{col_name}")
    else:
        for ax in g.axes.flat:
            ax.set_title("")

    # ---- Global title ----
    if title is not None:
        plt.subplots_adjust(top=0.85)
        plt.suptitle(title, x=0.5 + (title_dx / 100))

    return g




def fig_metric(
    df: pd.DataFrame,
    metric: str,
    title: Optional[str] = None,
    title_dx: int = 0,
    width: Optional[int] = None,
    height: Optional[int] = None,
    labels: bool = True,
    keywords: Dict[str, Any] = {},
    style: Dict[str, Any] = {},
    default_color: str = "#000000",
    colors_dict: Dict[str, Any] = {},
    config: Optional[str] = None,
):
    """Plots metrics

    Args:
        df: Dataframe which at least has columns `algorithm`, `num_simulations` and
            a column titled accordingly to `metric`.
        metric: Metric to plot, should be a column in `df`.
        title: Title for plot
        title_dx: x-direction offset for title
        labels: Whether to plot labels
        seed: Seed
        width: Width
        height: Height
        default_color: Default color of samples
        colors_dict: Dictionary of colors
        config: Optional string to load predefined config
        style: Optional dictionary for `den.set_style`
        keywords: Optional dictionary passed on to `den.lineplot`

    Returns:
        Chart

    Note:
        Due to an open issue on vega-lite, it is difficult to sort columns in
        non-alphabetical fashion (i.e., ordered by algorithm name). As a workaround,
        consider prepending an algorithm with a space to have it listed first, e.g.
        `df.loc[df["algorithm"] == "REJ-ABC", "algorithm"] = " REJ-ABC"`.
        See also: https://github.com/vega/vega-lite/issues/5366/
    """
    colors = {}
    for algorithm in df.algorithm.unique():
        algorithm_stripped = algorithm.strip()
        if algorithm_stripped not in colors_dict:
            colors[algorithm] = default_color
        else:
            colors[algorithm] = colors_dict[algorithm_stripped]

    keywords["column_labels"] = labels
    keywords["color"] = den.colorscale(colors, shorthand="algorithm:N")

    x_axis_kwargs = {}
    y_axis_kwargs = {}

    if config == "manuscript":
        keywords["width"] = 700 / len(df.algorithm.unique()) if width is None else width
        keywords["height"] = 65 if height is None else height
        style["font_size"] = 12
        style["font_family"] = "Inter"
        y_axis_kwargs = {"minExtent": 60}

    if config == "streamlit":
        keywords["width"] = None if width is None else width
        keywords["height"] = None if height is None else height
        style["font_size"] = 16

    keywords["limits"] = None
    keywords["log_y"] = False
    keywords["y_axis"] = alt.Axis(title=metric)

    if "MMD" in metric:
        keywords["y_axis"] = alt.Axis(title="MMD²")

    if "C2ST" in metric:
        keywords["limits"] = [0.5, 1.0]

    if metric == "RT":
        keywords["log_y"] = True
        keywords["limits"] = [0.001, 1000.0]
        keywords["y_axis"] = alt.Axis(
            values=[0.001, 0.01, 0.1, 0.0, 1.0, 10.0, 100.0, 1000.0]
        )

    alt.themes.enable("default")

    den.set_style(
        extra={
            "config": {
                "axisX": {
                    "grid": False,
                    "labelAngle": 270,
                    "domain": False,
                    "domainWidth": 0,
                    "ticks": True,
                    "tickWidth": 0,
                    "minExtent": 0,
                    **x_axis_kwargs,
                },
                "axisY": {
                    "domain": False,
                    "domainWidth": 0,
                    "ticks": True,
                    "tickWidth": 0,
                    "grid": True,
                    "titlePadding": 0,
                    "tickCount": 6.0,
                    **y_axis_kwargs,
                },
            }
        },
        **style,
    )

    chart = den.lineplot(
        df.sort_values("algorithm"),
        x="num_simulations:O",
        y=f"{metric}:Q",
        error_extent="ci",
        column="algorithm:N",
        independent_y=False,
        row_title="",
        column_title="Number of Simulations",
        title_orient="bottom",
        **keywords,
    )

    chart = chart.configure_point(size=50).configure_line(size=1.5)

    if title is not None:
        chart = chart.properties(title={"text": [title],}).configure_title(
            offset=10,
            orient="top",
            anchor="middle",
            dx=title_dx,
        )

    return chart
