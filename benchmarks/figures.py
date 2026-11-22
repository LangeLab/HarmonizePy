"""Benchmark figures, light and dark, from the JSON the benchmark runs write.

    uv run --group figures python -m benchmarks.figures

reads the result files under ``benchmarks/results/`` at the names the
commands in ``benchmarks/RESULTS.md`` give them, and writes each figure as an
SVG and a PNG (200 dpi on the theme's page color, 256 colors, no dithering)
to ``benchmarks/figures/``, with a copy of the PNG in ``wiki/assets/``:

- ``01-speed``: wall time of both tools per dataset and engine (parity pass).
- ``02-memory``: peak memory of both tools in a fresh process (memory_peak).
- ``03-scaling``: wall time against features and samples (scaling).
- ``04-memory-scaling``: peak memory against the input size (scaling).
- ``05-blocking``: what ``block=2`` saves in time and costs in kept values (scaling).
- ``06-outcomes``: what each tool returns for every input feature (outcomes).

A figure whose result file is missing is skipped.

Colors: HarmonizePy as the subject (teal on the light page, Python yellow on
the dark one) against a recessive gray for HarmonizR; the engines as three hues
(limma, parametric ComBat, non-parametric ComBat). Every pair was checked for
color vision deficiency separation and contrast on both page colors.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter, NullLocator
from PIL import Image

plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "harmonizepy"})

# GitHub's page colors, with the HarmonizePy logo teal as the accent.
THEMES: dict[str, dict[str, str]] = {
    "light": {
        "bg": "#ffffff",
        "ink": "#1f2328",
        "ink2": "#59636e",
        "muted": "#818b98",
        "grid": "#d1d9e0",
        "region": "#e6f4f2",
        "subject": "#0A7A6F",
        "other": "#8c959f",
        "limma": "#2a78d6",
        "parametric": "#eb6834",
        "nonparametric": "#1baf7a",
    },
    "dark": {
        "bg": "#0d1117",
        "ink": "#f0f6fc",
        "ink2": "#9198a1",
        "muted": "#656c76",
        "grid": "#3d444d",
        "region": "#2a2410",
        "subject": "#FFD43B",
        "other": "#6e7681",
        "limma": "#3987e5",
        "parametric": "#d95926",
        "nonparametric": "#199e70",
    },
}
SUBJECT, OTHER = "HarmonizePy", "HarmonizR"
DATASETS = ("small", "medium", "murine", "dia", "scp_small")
DATASET_LABEL = {
    "small": "small",
    "medium": "medium",
    "murine": "murine",
    "dia": "dia",
    "scp_small": "scp_small",
    "large": "large",
    "scp_large": "scp_large",
}
# The engines, in legend order: (algorithm, mode) -> (label, color key, dashed).
ENGINES: dict[tuple[str, int | None], tuple[str, str, bool]] = {
    ("limma", None): ("limma", "limma", False),
    ("ComBat", 1): ("ComBat 1, parametric", "parametric", False),
    ("ComBat", 2): ("ComBat 2, parametric, location only", "parametric", True),
    ("ComBat", 3): ("ComBat 3, non-parametric", "nonparametric", False),
    ("ComBat", 4): ("ComBat 4, non-parametric, location only", "nonparametric", True),
}
MARKER = {
    ("limma", None): "o",
    ("ComBat", 1): "s",
    ("ComBat", 2): "s",
    ("ComBat", 3): "D",
    ("ComBat", 4): "D",
}


def times(r: float) -> str:
    """A ratio as a short multiple: 0.62x, 1.4x, 24x, 310x."""
    if r >= 100:
        return f"{r:.0f}x"
    if r >= 10:
        return f"{r:.0f}x"
    if r >= 0.995:
        return f"{r:.1f}x"
    return f"{r:.2g}x"


def seconds(s: float) -> str:
    """A wall time: 21 ms, 0.47 s, 5.0 s, 312 s."""
    if s < 0.1:
        return f"{s * 1000:.0f} ms"
    if s < 10:
        return f"{s:.2g} s" if s < 1 else f"{s:.1f} s"
    return f"{s:.0f} s"


def log_axis(axis: Any, fmt: Any, subs: tuple[float, ...] = (1.0, 2.0, 5.0)) -> None:
    axis.set_major_locator(LogLocator(base=10, subs=subs))
    axis.set_major_formatter(FuncFormatter(fmt))
    axis.set_minor_formatter(NullFormatter())
    axis.set_minor_locator(NullLocator())


def count_fmt(v: float, _: Any = None) -> str:
    for div, suffix in ((1e6, "M"), (1e3, "k")):
        if v >= div:
            return f"{v / div:g}{suffix}"
    return f"{v:g}"


def decades(axis: Any) -> None:
    """Every third power of ten, as 10^-15 ... 1, without minor ticks."""
    axis.set_major_locator(FixedLocator([10.0**k for k in range(-15, 1, 3)]))
    axis.set_major_formatter(
        FuncFormatter(lambda v, _: "1" if v == 1 else f"$10^{{{math.log10(v):.0f}}}$")
    )
    axis.set_minor_locator(NullLocator())


def spread_log(values: list[float], gap: float) -> list[float]:
    """Label heights for points at `values` (log scale), at least `gap` decades apart, in input order."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = [0.0] * len(values)
    last = -math.inf
    for i in order:
        y = max(math.log10(values[i]), last + gap)
        placed[i] = 10**y
        last = y
    return placed


def engine_key(scenario: dict[str, Any]) -> tuple[str, int | None]:
    return (
        scenario["algorithm"],
        scenario.get("combat_mode") if scenario["algorithm"] == "ComBat" else None,
    )


RESULTS = "benchmarks/results"


class Figures:
    def __init__(
        self,
        parity: dict[str, Any],
        scaling: dict[str, Any] | None,
        outcomes: dict[str, Any] | None,
        out: Path,
        memory_peak: dict[str, Any] | None = None,
        wiki: Path | None = None,
    ) -> None:
        self.memory_peak = memory_peak
        self.wiki = wiki
        self.parity = parity["results"]
        self.scaling = scaling
        self.outcomes = outcomes
        self.system = parity["system"]
        self.datasets = parity["datasets"]
        self.out = out
        out.mkdir(parents=True, exist_ok=True)

    # ---- shared layout ---------------------------------------------------------

    def axes(
        self, theme: str, ncols: int, width: float, height: float, sharey: bool = False
    ) -> tuple[Any, list[Any]]:
        t = THEMES[theme]
        fig, axes = plt.subplots(1, ncols, figsize=(width, height), sharey=sharey, squeeze=False)
        fig.patch.set_facecolor(t["bg"])
        for ax in axes[0]:
            ax.set_facecolor(t["bg"])
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(t["muted"])
            ax.tick_params(colors=t["ink2"], labelsize=8.5, length=3)
            ax.grid(True, which="major", color=t["grid"], linewidth=0.6, alpha=0.8)
            ax.set_axisbelow(True)
        return fig, list(axes[0])

    def panel(self, ax: Any, theme: str, title: str, xlabel: str, ylabel: str = "") -> None:
        t = THEMES[theme]
        ax.set_title(title, loc="left", fontsize=10, color=t["ink"], fontweight="bold", pad=8)
        ax.set_xlabel(xlabel, fontsize=8.5, color=t["ink2"])
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=8.5, color=t["ink2"])

    def finish(
        self,
        fig: Any,
        stem: str,
        theme: str,
        title: str,
        subtitle: str,
        handles: list[Any] | None = None,
        ncol: int | None = None,
    ) -> None:
        """Title and italic note centered on top, legend centered below, tight bounds."""
        t = THEMES[theme]
        fig.tight_layout(w_pad=2.0)
        boxes = [ax.get_tightbbox() for ax in fig.axes]
        fh = fig.bbox.height
        top = max(b.y1 for b in boxes) / fh
        bottom = min(b.y0 for b in boxes) / fh
        inch = 1 / fig.get_figheight()
        lines = textwrap.wrap(subtitle, int(fig.get_figwidth() * 15))
        fig.text(
            0.5,
            top + 0.16 * inch,
            "\n".join(lines),
            ha="center",
            va="bottom",
            fontsize=9,
            color=t["ink2"],
            style="italic",
            linespacing=1.45,
        )
        fig.text(
            0.5,
            top + (0.26 + 0.2 * len(lines)) * inch,
            title,
            ha="center",
            va="bottom",
            fontsize=13,
            color=t["ink"],
            fontweight="bold",
        )
        if handles:
            fig.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, bottom - 0.06 * inch),
                ncol=ncol or min(len(handles), 5),
                frameon=False,
                fontsize=8.5,
                labelcolor=t["ink"],
                handlelength=2.2,
                columnspacing=1.5,
                borderaxespad=0,
            )
        path = self.out / f"{stem}-{theme}.svg"
        fig.savefig(path, format="svg", facecolor=t["bg"], bbox_inches="tight", pad_inches=0.2)
        # The README's copy: a flat chart needs few colors, so a 256-color
        # palette without dithering keeps every edge and keeps the file small.
        buf = io.BytesIO()
        fig.savefig(
            buf, format="png", dpi=200, facecolor=t["bg"], bbox_inches="tight", pad_inches=0.2
        )
        image = (
            Image.open(buf)
            .convert("RGB")
            .quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        )
        image.save(path.with_suffix(".png"), format="png", optimize=True)
        if self.wiki is not None:
            # The wiki names its images by subject: 01-speed-light.png -> benchmark-speed-light.png.
            image.save(self.wiki / f"benchmark-{stem[3:]}-{theme}.png", format="png", optimize=True)
        plt.close(fig)

    def engine_handles(self, theme: str, engines: list[tuple[str, int | None]]) -> list[Any]:
        t = THEMES[theme]
        handles = []
        for key in engines:
            label, color, dashed = ENGINES[key]
            handles.append(
                Line2D(
                    [],
                    [],
                    color=t[color],
                    linewidth=1.8,
                    linestyle=(0, (4, 2)) if dashed else "-",
                    marker=MARKER[key],
                    markersize=6,
                    markerfacecolor=t["bg"] if dashed else t[color],
                    markeredgecolor=t[color],
                    markeredgewidth=1.4,
                    label=label,
                )
            )
        return handles

    # ---- 01, 02: speed and memory against HarmonizR ----------------------------------------

    PANELS = (("limma", None), ("ComBat", 1), ("ComBat", 3))
    PANEL_TITLES = ("limma", "ComBat, parametric (mode 1)", "ComBat, non-parametric (mode 3)")

    def paired(
        self,
        theme: str,
        stem: str,
        values: dict[tuple[str, Any], tuple[float, float]],
        *,
        fmt: Any,
        gap: Any,
        ticks: list[float],
        xlim: tuple[float, float],
        xlabel: str,
        title: str,
        note: str,
        baselines: tuple[float, float] | None = None,
    ) -> None:
        """Both tools per dataset on one log axis, one panel per engine: the shared headline layout."""
        t = THEMES[theme]
        meta = self.datasets
        order = sorted(DATASETS, key=lambda d: meta[d]["features"] * meta[d]["samples"])
        fig, axes = self.axes(theme, 3, 12, 0.62 * len(order) + 1.3, sharey=True)
        ys = list(range(len(order)))[::-1]
        # A page-colored box behind each label, so grid and idle lines pass behind the text.
        label_box = {"boxstyle": "square,pad=0.15", "facecolor": t["bg"], "edgecolor": "none"}
        for ax, key, panel_title in zip(axes, self.PANELS, self.PANEL_TITLES, strict=True):
            if baselines:
                for value, color in zip(baselines, (t["subject"], t["other"]), strict=True):
                    ax.axvline(value, color=color, linewidth=1.0, linestyle=(0, (1, 2)), zorder=0)
            for y, ds in zip(ys, order, strict=True):
                py, r = values[(ds, key)]
                ax.plot(
                    [py, r],
                    [y, y],
                    color=t["region"],
                    linewidth=7,
                    solid_capstyle="round",
                    zorder=1,
                )
                ax.plot(
                    r,
                    y,
                    "o",
                    markersize=7.5,
                    markerfacecolor=t["bg"],
                    markeredgecolor=t["other"],
                    markeredgewidth=1.6,
                    zorder=3,
                )
                ax.plot(py, y, "o", markersize=7.5, color=t["subject"], zorder=4)
                left, right = (py, r) if py <= r else (r, py)
                ax.annotate(
                    fmt(left),
                    (left, y),
                    xytext=(-8, 0),
                    textcoords="offset points",
                    bbox=label_box,
                    ha="right",
                    va="center",
                    fontsize=7.8,
                    color=t["ink"] if left == py else t["ink2"],
                    fontweight="bold" if left == py else "normal",
                )
                ax.annotate(
                    fmt(right),
                    (right, y),
                    xytext=(8, 0),
                    textcoords="offset points",
                    bbox=label_box,
                    ha="left",
                    va="center",
                    fontsize=7.8,
                    color=t["ink"] if right == py else t["ink2"],
                    fontweight="bold" if right == py else "normal",
                )
                ax.annotate(
                    gap(py, r),
                    (math.sqrt(py * r), y),
                    xytext=(0, 8),
                    textcoords="offset points",
                    bbox=label_box,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=t["ink2"],
                )
            ax.set_xscale("log")
            ax.set_xlim(*xlim)
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt(v)))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.grid(False, axis="y")
            ax.tick_params(axis="y", length=0, pad=10)
            ax.set_ylim(-0.6, len(order) - 0.4)
            self.panel(ax, theme, panel_title, xlabel)
        axes[0].set_yticks(ys)
        axes[0].set_yticklabels(
            [
                f"{ds}\n{meta[ds]['features']:,} x {meta[ds]['samples']:,}, {meta[ds]['batches']} batches"
                for ds in order
            ],
            fontsize=8.6,
            color=t["ink"],
            linespacing=1.5,
        )
        handles = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=7.5,
                color=t["subject"],
                label=SUBJECT,
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=7.5,
                markerfacecolor=t["bg"],
                markeredgecolor=t["other"],
                markeredgewidth=1.6,
                label=f"{OTHER} 1.10.0",
            ),
        ]
        if baselines:
            handles += [
                Line2D(
                    [],
                    [],
                    color=t["subject"],
                    linewidth=1.0,
                    linestyle=(0, (1, 2)),
                    label=f"{SUBJECT} idle, {fmt(baselines[0])}",
                ),
                Line2D(
                    [],
                    [],
                    color=t["other"],
                    linewidth=1.0,
                    linestyle=(0, (1, 2)),
                    label=f"{OTHER} idle, {fmt(baselines[1])}",
                ),
            ]
        self.finish(fig, stem, theme, title, note, handles, ncol=len(handles))

    def headline(self, theme: str) -> None:
        """Wall time of both tools per dataset, in seconds, one panel per engine family."""
        values = {}
        for e in self.parity:
            s = e["scenario"]
            if not s.get("block") and e.get("r") and e["r"].get("median_s"):
                values[(s["dataset"], engine_key(s))] = (
                    e["python"]["median_s"],
                    e["r"]["median_s"],
                )
        self.paired(
            theme,
            "01-speed",
            values,
            fmt=seconds,
            gap=lambda py, r: times(r / py) + " faster",
            ticks=[0.01, 0.1, 1, 10, 100, 1000],
            xlim=(0.0015, 2500),
            xlabel="wall time, log scale",
            title="The same correction in a fraction of the time",
            note="Wall time of one correction on the same matrix, both on one core: HarmonizePy the median "
            "of three runs on a matrix already in memory, HarmonizR one run that also reads its input "
            "file (at most 5% of its time here). Rows run from the smallest matrix to the largest "
            "(features x samples); how closely the outputs agree is in RESULTS.md.",
        )

    def memory_headline(self, theme: str) -> None:
        """Peak memory of both tools per dataset, each run in its own process."""
        mp = self.memory_peak
        values = {
            (r["dataset"], (r["algorithm"], r["mode"])): (
                r["harmonizepy"]["peak_mb"],
                r["harmonizr"]["peak_mb"],
            )
            for r in mp["results"]
        }
        base = (mp["baseline_mb"]["harmonizepy"], mp["baseline_mb"]["harmonizr"])

        def gap(py: float, r: float) -> str:
            return f"{times(r / py)} less" if r >= py else f"{times(py / r)} more"

        self.paired(
            theme,
            "02-memory",
            values,
            fmt=lambda v: "1 GB" if v == 1000 else f"{v:,.0f} MB",
            gap=gap,
            ticks=[50, 100, 200, 500, 1000],
            xlim=(38, 1700),
            xlabel="peak memory of the process, log scale",
            title="The same correction in less memory",
            note="Peak resident memory of a fresh process that reads both input files, corrects the "
            "matrix once and exits; HarmonizR on one core with BiocParallel serial, so all of its work is "
            "measured. Dotted lines: each tool started and left idle. Loading sva adds about 370 MB to "
            "HarmonizR's ComBat runs before any data is read.",
            baselines=base,
        )

    # ---- 03: scaling ---------------------------------------------------------------

    def scaling_figure(self, theme: str) -> None:
        """Wall time against features and against samples, with the groups each matrix forms."""
        t = THEMES[theme]
        sc = self.scaling
        design = sc["design"]
        fig, axes = self.axes(theme, 2, 12, 5.0, sharey=True)
        sweeps = (
            (
                "features",
                "n_features",
                f"More features ({design['feature_sweep_samples']} samples)",
                "features",
            ),
            (
                "samples",
                "n_samples",
                f"More samples ({design['sample_sweep_features']:,} features)",
                "samples",
            ),
        )
        for ax, (part, xkey, title, unit) in zip(axes, sweeps, strict=True):
            ends: list[tuple[float, float]] = []
            for key, (_label, color, dashed) in ENGINES.items():
                pts = sorted(
                    (r[xkey], r["median_s"]) for r in sc[part] if (r["algorithm"], r["mode"]) == key
                )
                xs, ys = zip(*pts, strict=True)
                ax.plot(
                    xs,
                    ys,
                    color=t[color],
                    linewidth=1.8,
                    linestyle=(0, (4, 2)) if dashed else "-",
                    marker=MARKER[key],
                    markersize=5.5,
                    markerfacecolor=t["bg"] if dashed else t[color],
                    markeredgecolor=t[color],
                    markeredgewidth=1.3,
                    zorder=3,
                )
                ends.append((xs[-1], ys[-1]))
            for (x, y), ly in zip(ends, spread_log([y for _, y in ends], 0.075), strict=True):
                ax.annotate(
                    seconds(y),
                    (x, y),
                    xytext=(x * 1.12, ly),
                    textcoords="data",
                    va="center",
                    fontsize=7.4,
                    color=t["ink"],
                )
            # The affiliation groups each matrix forms, the same for every engine at a given size.
            groups = sorted(
                {(r[xkey], r["n_groups"]) for r in sc[part] if r["algorithm"] == "limma"}
            )
            for x, g in groups:
                ax.annotate(
                    f"{g}",
                    (x, 0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7.2,
                    color=t["ink2"],
                )
            ax.annotate(
                "groups:",
                (groups[0][0], 0),
                xycoords=("data", "axes fraction"),
                xytext=(-14, 6),
                textcoords="offset points",
                ha="right",
                fontsize=7.2,
                color=t["ink2"],
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            log_axis(ax.xaxis, count_fmt)
            ax.set_xlim(groups[0][0] / 1.6, groups[-1][0] * 2.2)
            ax.set_ylim(0.03, 30)
            ax.yaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: seconds(v)))
            ax.yaxis.set_minor_locator(NullLocator())
            self.panel(
                ax,
                theme,
                title,
                f"{unit}, log scale",
                "wall time, log scale" if part == "features" else "",
            )
        self.finish(
            fig,
            "03-scaling",
            theme,
            "Time grows far slower than the matrix",
            f"Synthetic matrices, {design['n_batches']} batches, {design['missing_frac']:.0%} of features "
            "absent from each batch, one core, median of three runs. Fifty times more features costs limma and "
            "parametric ComBat five to six times the time, and 150 times more samples at most seven: the work follows the number of "
            "missingness groups (numbers along the bottom), which grows slowly. The non-parametric modes "
            "rise once groups get large, as each feature is weighed against every other in its group.",
            self.engine_handles(theme, list(ENGINES)),
            ncol=3,
        )

    # ---- 04: memory against size ----------------------------------------------------------------

    def memory(self, theme: str) -> None:
        """Peak memory of one call against the size of its input matrix, one point per matrix."""
        t = THEMES[theme]
        fig, (ax,) = self.axes(theme, 1, 8.0, 5.4)
        lo, hi = 0.25, 600
        for k in (1, 3, 10):
            ax.plot(
                [lo, hi],
                [lo * k, hi * k],
                color=t["muted"],
                linewidth=0.9,
                linestyle=(0, (2, 3)),
                zorder=1,
            )
            # Labels sit below the top of the plot, where the steepest line leaves it.
            xr = min(hi, 1000 / k)
            ax.annotate(
                f"{k}x the input",
                (xr, xr * k),
                xytext=(-4, 4),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize=7.6,
                color=t["ink2"],
                rotation=33,
                rotation_mode="anchor",
            )
        ratios = []
        for part, marker in (("features", "o"), ("samples", "s")):
            by_matrix: dict[float, list[float]] = {}
            for r in self.scaling[part]:
                by_matrix.setdefault(r["input_mb"], []).append(r["peak_mb"])
            for x, peaks in sorted(by_matrix.items()):
                ratios += [p / x for p in peaks]
                ax.plot(
                    [x, x], [min(peaks), max(peaks)], color=t["subject"], linewidth=1.2, zorder=2
                )
                ax.plot(
                    x,
                    max(peaks),
                    marker,
                    markersize=7,
                    color=t["subject"],
                    markerfacecolor=t["subject"] if marker == "o" else t["bg"],
                    markeredgewidth=1.5,
                    zorder=3,
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, 2500)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(
                FixedLocator([0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000])
            )
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} MB"))
            axis.set_minor_locator(NullLocator())
        self.panel(
            ax,
            theme,
            "Peak memory against input size",
            "input matrix as float64, log scale",
            "peak memory of one call, log scale",
        )
        handles = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=7,
                color=t["subject"],
                label="more features, 60 samples",
            ),
            Line2D(
                [],
                [],
                marker="s",
                linestyle="none",
                markersize=7,
                markerfacecolor=t["bg"],
                markeredgecolor=t["subject"],
                markeredgewidth=1.5,
                label="more samples, 5,000 features",
            ),
        ]
        self.finish(
            fig,
            "04-memory-scaling",
            theme,
            f"Peak memory is {min(ratios):.1f} to {max(ratios):.1f} times the input",
            "Every matrix of the scaling sweep, from 0.5 MB to 229 MB of input. Peak memory is Python's "
            "traced peak during harmonize() plus the output array. The mark is the highest of the five "
            "engines and the bar under it reaches the lowest; they differ by at most a quarter.",
            handles,
            ncol=2,
        )

    # ---- 05: blocking ----------------------------------------------------------------

    def blocking(self, theme: str) -> None:
        """Time against observed values kept, each run without and with block=2."""
        t = THEMES[theme]
        recs = self.scaling["blocking"]
        fig, (ax,) = self.axes(theme, 1, 8.0, 5.0)
        sizes = sorted({r["n_features"] for r in recs})
        size_marker = dict(zip(sizes, ("o", "s"), strict=False))
        drawn: dict[tuple[str, float], list[float]] = {}
        for a in (r for r in recs if r["block"] is None):
            b = next(
                r
                for r in recs
                if r["block"] == 2
                and r["n_features"] == a["n_features"]
                and (r["algorithm"], r["mode"]) == (a["algorithm"], a["mode"])
            )
            key = (a["algorithm"], a["mode"])
            color = t[ENGINES[key][1]]
            xa, xb = (
                100 * a["observed_out"] / a["observed_in"],
                100 * b["observed_out"] / b["observed_in"],
            )
            ax.annotate(
                "",
                (xb, b["median_s"]),
                (xa, a["median_s"]),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "linewidth": 1.4,
                    "shrinkA": 6,
                    "shrinkB": 6,
                    "mutation_scale": 11,
                },
            )
            m = size_marker[a["n_features"]]
            # Two engines can take the same time on the same matrix; the later one is
            # drawn larger behind the first, so both outlines stay visible.
            spot = (m, round(xa, 1))
            nested = any(abs(math.log(a["median_s"] / s)) < 0.05 for s in drawn.get(spot, []))
            drawn.setdefault(spot, []).append(a["median_s"])
            ax.plot(
                xa,
                a["median_s"],
                m,
                markersize=10.5 if nested else 7,
                markerfacecolor=t["bg"],
                markeredgecolor=color,
                markeredgewidth=1.5,
                zorder=2.5 if nested else 3,
            )
            ax.plot(xb, b["median_s"], m, markersize=7, color=color, zorder=3)
        ax.set_yscale("log")
        ax.set_xlim(76, 101.5)
        ax.set_ylim(0.025, 6)
        ax.yaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.2, 0.5, 1, 2, 5]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: seconds(v)))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
        g = {
            r["block"]: r["n_groups"]
            for r in recs
            if r["n_features"] == sizes[0] and r["algorithm"] == "limma"
        }
        self.panel(
            ax,
            theme,
            "Each run without blocking, and with block=2",
            "observed input values that reach the output",
            "wall time, log scale",
        )
        handles = [
            Line2D([], [], color=t["limma"], linewidth=1.8, label="limma"),
            Line2D([], [], color=t["parametric"], linewidth=1.8, label="ComBat 1, parametric"),
            Line2D(
                [], [], color=t["nonparametric"], linewidth=1.8, label="ComBat 3, non-parametric"
            ),
            *[
                Line2D(
                    [],
                    [],
                    marker=size_marker[s],
                    linestyle="none",
                    markersize=7,
                    markerfacecolor=t["bg"],
                    markeredgecolor=t["ink2"],
                    markeredgewidth=1.4,
                    label=f"{s:,} features",
                )
                for s in sizes
            ],
        ]
        self.finish(
            fig,
            "05-blocking",
            theme,
            "Blocking buys speed with observed values",
            f"block=2 joins neighbouring batches, so {g[None]} missingness groups become {g[2]} and a run "
            "takes a fraction of the time; a feature missing from one batch of a block loses the whole "
            "block, so fewer of its observed values reach the output. Arrows run from each batch its own "
            f"block (hollow) to block=2 (filled). Synthetic matrices, {self.scaling['design']['n_batches']} "
            "batches, 60 samples.",
            handles,
            ncol=5,
        )

    # ---- 06: what each tool returns per feature --------------------------------------------

    def outcomes_figure(self, theme: str) -> None:
        """Share of input features each tool corrects, returns raw, returns empty, or drops."""
        t = THEMES[theme]
        rows = sorted(self.outcomes["datasets"], key=lambda r: DATASETS.index(r["dataset"]))
        classes = (
            ("corrected", "corrected", mix(t["ink"], t["bg"], 0.78), None),
            ("raw", "returned as measured", mix(t["ink"], t["bg"], 0.45), None),
            ("empty", "returned empty", mix(t["ink"], t["bg"], 0.18), None),
            ("dropped", "left out of the output", t["bg"], "////"),
        )
        fig, (ax,) = self.axes(theme, 1, 11, 0.95 * len(rows) + 1.4)
        ys = []
        labels = []
        for i, r in enumerate(rows):
            for j, (tool, name) in enumerate((("harmonizepy", SUBJECT), ("harmonizr", OTHER))):
                y = (len(rows) - 1 - i) * 2.6 + (1 - j) * 1.0
                ys.append(y)
                labels.append(f"{r['dataset']}  {name}" if j == 0 else name)
                left = 0.0
                for key, _name, color, hatch in classes:
                    share = 100 * r[tool][key] / r["n_features"]
                    if share <= 0:
                        continue
                    ax.barh(
                        y,
                        share,
                        left=left,
                        height=0.78,
                        color=color,
                        edgecolor=t["other"] if hatch else t["bg"],
                        hatch=hatch,
                        linewidth=0.8 if hatch else 1.5,
                        zorder=2,
                    )
                    left += share
                missing = r[tool]["empty"] + r[tool]["dropped"]
                ax.text(
                    101,
                    y,
                    f"{r[tool]['corrected'] + r[tool]['raw']:,} of {r['n_features']:,} returned with values"
                    + (f", {missing:,} without" if missing else ""),
                    va="center",
                    fontsize=7.6,
                    color=t["ink2"],
                )
        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=8.6, color=t["ink"])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, 100)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
        ax.grid(False, axis="y")
        self.panel(
            ax, theme, "Input features by what the output holds for them", "share of input features"
        )
        handles = [
            Patch(facecolor=c, edgecolor=t["other"] if h else c, hatch=h, label=n)
            for _, n, c, h in classes
        ]
        self.finish(
            fig,
            "06-outcomes",
            theme,
            "Every input feature is in the output",
            "ComBat mode 1 on each matrix with an R baseline. Both tools correct the same features; "
            f"{SUBJECT} keeps a row for every feature it cannot correct, with its measured values or "
            f"empty, and its report says why, where {OTHER} leaves such features out.",
            handles,
            ncol=4,
        )

    def all(self) -> list[str]:
        """Draw every figure its inputs allow, in both themes, and return the stems written."""
        drawn = [("01-speed", self.headline)]
        if self.memory_peak:
            drawn.append(("02-memory", self.memory_headline))
        if self.scaling:
            drawn += [
                ("03-scaling", self.scaling_figure),
                ("04-memory-scaling", self.memory),
                ("05-blocking", self.blocking),
            ]
        if self.outcomes:
            drawn.append(("06-outcomes", self.outcomes_figure))
        made = []
        for stem, draw in drawn:
            for theme in THEMES:
                draw(theme)
                made.append(f"{stem}-{theme}")
        return made


def mix(color: str, bg: str, weight: float) -> tuple[float, ...]:
    """`weight` of `color` over `bg`, as an opaque color."""
    return tuple(
        weight * a + (1 - weight) * b for a, b in zip(to_rgb(color), to_rgb(bg), strict=True)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmarks.figures", description=__doc__.splitlines()[0])
    parser.add_argument(
        "--parity",
        default=f"{RESULTS}/benchmark_r_parity.json",
        help="JSON of the R-backed parity pass.",
    )
    parser.add_argument(
        "--memory-peak",
        default=f"{RESULTS}/memory_peak.json",
        help="JSON of benchmarks.memory_peak.",
    )
    parser.add_argument(
        "--scaling", default=f"{RESULTS}/scaling.json", help="JSON of benchmarks.scaling."
    )
    parser.add_argument(
        "--outcomes", default=f"{RESULTS}/outcomes.json", help="JSON of benchmarks.outcomes."
    )
    parser.add_argument("--out", default="benchmarks/figures", help="Output directory.")
    parser.add_argument(
        "--wiki", default="wiki/assets", help="Directory for the wiki's copies; empty to skip them."
    )
    args = parser.parse_args(argv)

    def load(path: str) -> dict[str, Any] | None:
        """Read a result file; a missing optional one skips its figures."""
        if Path(path).is_file():
            return json.loads(Path(path).read_text())
        print(f"skipped: {path} not found")
        return None

    parity = load(args.parity)
    if parity is None:
        parser.error(f"the parity pass is needed for every figure: {args.parity}")
    wiki = Path(args.wiki) if args.wiki else None
    made = Figures(
        parity,
        load(args.scaling),
        load(args.outcomes),
        Path(args.out),
        load(args.memory_peak),
        wiki=wiki,
    ).all()
    print("\n".join(f"wrote {args.out}/{m}.png" for m in made))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
