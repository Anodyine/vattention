#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = {
    "Request Id",
    "request_num_prefill_tokens",
    "kv_fragmentation_percent",
}


@dataclass
class AnalyticalPrediction:
    tokens_per_page: int
    decode_reservation_tokens: int
    context_cap_tokens: int
    exact_mean_abs_error: float
    exact_max_abs_error: float
    average_mean_abs_error: float
    average_max_abs_error: float
    worst_case_mean_abs_error: float
    worst_case_max_abs_error: float
    data: pd.DataFrame


def load_metrics(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "Missing required columns in sequence_metrics.csv: "
            + ", ".join(sorted(missing))
        )
    return df


def clean_for_plot(df: pd.DataFrame, max_context: int | None = None) -> pd.DataFrame:
    work = df.copy()

    if "request_num_ignored" in work.columns:
        ignored = pd.to_numeric(work["request_num_ignored"], errors="coerce").fillna(0)
        work = work[ignored == 0]

    work["request_num_prefill_tokens"] = pd.to_numeric(
        work["request_num_prefill_tokens"], errors="coerce"
    )
    work["kv_fragmentation_percent"] = pd.to_numeric(
        work["kv_fragmentation_percent"], errors="coerce"
    )
    if "kv_blocks_mapped" in work.columns:
        work["kv_blocks_mapped"] = pd.to_numeric(work["kv_blocks_mapped"], errors="coerce")
    if "request_num_decode_tokens" in work.columns:
        work["request_num_decode_tokens"] = pd.to_numeric(
            work["request_num_decode_tokens"], errors="coerce"
        )

    work = work.dropna(
        subset=["request_num_prefill_tokens", "kv_fragmentation_percent"]
    )
    work = work[work["request_num_prefill_tokens"] > 0]
    work = work[
        (work["kv_fragmentation_percent"] >= 0)
        & (work["kv_fragmentation_percent"] <= 100)
    ]
    if max_context is not None:
        work = work[work["request_num_prefill_tokens"] <= max_context]
    return work.sort_values("request_num_prefill_tokens")


def infer_tokens_per_page(df: pd.DataFrame) -> int | None:
    if "kv_blocks_mapped" not in df.columns:
        return None

    work = df.dropna(
        subset=["request_num_prefill_tokens", "kv_blocks_mapped", "kv_fragmentation_percent"]
    ).copy()
    if work.empty:
        return None

    fill_fraction = 1.0 - (work["kv_fragmentation_percent"] / 100.0)
    valid = (work["kv_blocks_mapped"] > 0) & (fill_fraction > 0)
    work = work[valid].copy()
    if work.empty:
        return None

    fill_fraction = 1.0 - (work["kv_fragmentation_percent"] / 100.0)
    candidates = work["request_num_prefill_tokens"] / (work["kv_blocks_mapped"] * fill_fraction)
    rounded = candidates.round().astype(int)
    rounded = rounded[rounded > 0]
    if rounded.empty:
        return None

    return int(rounded.mode().iloc[0])


def infer_decode_reservation_tokens(df: pd.DataFrame, tokens_per_page: int) -> int:
    max_decode = 0
    if "request_num_decode_tokens" in df.columns:
        decode = pd.to_numeric(df["request_num_decode_tokens"], errors="coerce").dropna()
        if not decode.empty:
            max_decode = max(0, int(decode.max()))

    candidates = range(0, max_decode + 3)
    best_offset = 0
    best_matches = -1

    context = df["request_num_prefill_tokens"]
    mapped = pd.to_numeric(df.get("kv_blocks_mapped"), errors="coerce")
    for offset in candidates:
        predicted = ((context + offset - 1) // tokens_per_page) + 1
        matches = int((predicted == mapped).sum())
        if matches > best_matches:
            best_matches = matches
            best_offset = offset

    return best_offset


def build_analytical_prediction(df: pd.DataFrame) -> AnalyticalPrediction | None:
    if "kv_blocks_mapped" not in df.columns:
        return None

    tokens_per_page = infer_tokens_per_page(df)
    if tokens_per_page is None or tokens_per_page <= 0:
        return None

    decode_reservation_tokens = infer_decode_reservation_tokens(df, tokens_per_page)
    context_cap_tokens = int(df["request_num_prefill_tokens"].max())

    analytical = df[["request_num_prefill_tokens"]].reset_index(drop=True).copy()
    analytical["effective_reserved_tokens"] = (
        analytical["request_num_prefill_tokens"] + decode_reservation_tokens
    ).clip(upper=context_cap_tokens)
    analytical["predicted_blocks"] = (
        (analytical["effective_reserved_tokens"] - 1)
        // tokens_per_page
    ) + 1
    analytical["predicted_fragmentation_percent"] = 100.0 * (
        analytical["predicted_blocks"] * tokens_per_page
        - analytical["request_num_prefill_tokens"]
    ) / (analytical["predicted_blocks"] * tokens_per_page)
    analytical["average_fragmentation_percent"] = 100.0 * (
        (tokens_per_page / 2.0)
        / (analytical["request_num_prefill_tokens"] + (tokens_per_page / 2.0))
    )
    analytical["worst_case_fragmentation_percent"] = 100.0 * (
        tokens_per_page
        / (analytical["request_num_prefill_tokens"] + tokens_per_page)
    )

    comparison = analytical.join(
        df[["kv_fragmentation_percent", "kv_blocks_mapped"]].reset_index(drop=True)
    )
    exact_mean_abs_error = (
        comparison["predicted_fragmentation_percent"]
        .sub(comparison["kv_fragmentation_percent"])
        .abs()
        .mean()
    )
    exact_max_abs_error = (
        comparison["predicted_fragmentation_percent"]
        .sub(comparison["kv_fragmentation_percent"])
        .abs()
        .max()
    )
    average_mean_abs_error = (
        comparison["average_fragmentation_percent"]
        .sub(comparison["kv_fragmentation_percent"])
        .abs()
        .mean()
    )
    average_max_abs_error = (
        comparison["average_fragmentation_percent"]
        .sub(comparison["kv_fragmentation_percent"])
        .abs()
        .max()
    )
    worst_case_mean_abs_error = (
        comparison["worst_case_fragmentation_percent"]
        .sub(comparison["kv_fragmentation_percent"])
        .abs()
        .mean()
    )
    worst_case_max_abs_error = (
        comparison["worst_case_fragmentation_percent"]
        .sub(comparison["kv_fragmentation_percent"])
        .abs()
        .max()
    )

    return AnalyticalPrediction(
        tokens_per_page=tokens_per_page,
        decode_reservation_tokens=decode_reservation_tokens,
        context_cap_tokens=context_cap_tokens,
        exact_mean_abs_error=float(exact_mean_abs_error),
        exact_max_abs_error=float(exact_max_abs_error),
        average_mean_abs_error=float(average_mean_abs_error),
        average_max_abs_error=float(average_max_abs_error),
        worst_case_mean_abs_error=float(worst_case_mean_abs_error),
        worst_case_max_abs_error=float(worst_case_max_abs_error),
        data=analytical,
    )


def add_binned_trend(df: pd.DataFrame, ax, bins: int) -> None:
    effective_bins = max(1, min(bins, len(df)))
    binned = df.copy()
    binned["ctx_bin"] = pd.cut(
        binned["request_num_prefill_tokens"],
        bins=effective_bins,
        duplicates="drop",
    )

    trend = (
        binned.groupby("ctx_bin", observed=True)
        .agg(
            ctx_mid=("request_num_prefill_tokens", "median"),
            frag_mean=("kv_fragmentation_percent", "mean"),
            frag_std=("kv_fragmentation_percent", "std"),
            n=("kv_fragmentation_percent", "size"),
        )
        .dropna(subset=["ctx_mid", "frag_mean"])
        .sort_values("ctx_mid")
    )

    if trend.empty:
        return

    ax.plot(
        trend["ctx_mid"],
        trend["frag_mean"],
        linewidth=2.0,
        color="#cf5c36",
        label="Binned mean",
    )

    lower = trend["frag_mean"] - trend["frag_std"].fillna(0)
    upper = trend["frag_mean"] + trend["frag_std"].fillna(0)
    ax.fill_between(
        trend["ctx_mid"],
        lower,
        upper,
        alpha=0.15,
        color="#cf5c36",
        label="\u00b11 std",
    )


def write_summary(
    df: pd.DataFrame,
    out_csv: Path,
    analytical: AnalyticalPrediction | None,
) -> None:
    summary = pd.DataFrame(
        {
            "n_requests": [len(df)],
            "min_context": [df["request_num_prefill_tokens"].min()],
            "max_context": [df["request_num_prefill_tokens"].max()],
            "mean_fragmentation": [df["kv_fragmentation_percent"].mean()],
            "median_fragmentation": [df["kv_fragmentation_percent"].median()],
            "p90_fragmentation": [df["kv_fragmentation_percent"].quantile(0.90)],
            "analytical_tokens_per_page": [
                analytical.tokens_per_page if analytical is not None else pd.NA
            ],
            "analytical_decode_reservation_tokens": [
                analytical.decode_reservation_tokens if analytical is not None else pd.NA
            ],
            "analytical_context_cap_tokens": [
                analytical.context_cap_tokens if analytical is not None else pd.NA
            ],
            "analytical_exact_mean_abs_error": [
                analytical.exact_mean_abs_error if analytical is not None else pd.NA
            ],
            "analytical_exact_max_abs_error": [
                analytical.exact_max_abs_error if analytical is not None else pd.NA
            ],
            "analytical_average_mean_abs_error": [
                analytical.average_mean_abs_error if analytical is not None else pd.NA
            ],
            "analytical_average_max_abs_error": [
                analytical.average_max_abs_error if analytical is not None else pd.NA
            ],
            "analytical_worst_case_mean_abs_error": [
                analytical.worst_case_mean_abs_error if analytical is not None else pd.NA
            ],
            "analytical_worst_case_max_abs_error": [
                analytical.worst_case_max_abs_error if analytical is not None else pd.NA
            ],
        }
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)


def plot_raw_sawtooth(
    df: pd.DataFrame,
    out_plot: Path,
    title: str,
    analytical: AnalyticalPrediction | None,
    show_block_panel: bool,
) -> None:
    has_blocks = "kv_blocks_mapped" in df.columns
    if has_blocks:
        df = df.copy()
        df["kv_blocks_mapped"] = pd.to_numeric(df["kv_blocks_mapped"], errors="coerce")

    if has_blocks and show_block_panel:
        fig, (ax_top, ax_bottom) = plt.subplots(
            2,
            1,
            figsize=(10, 7),
            dpi=140,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
    else:
        fig, ax_top = plt.subplots(figsize=(10, 6), dpi=140)
        ax_bottom = None

    ax_top.plot(
        df["request_num_prefill_tokens"],
        df["kv_fragmentation_percent"],
        linewidth=1.8,
        color="#cf5c36",
        alpha=0.95,
        zorder=2,
        label="Fragmentation trajectory",
    )
    scatter = ax_top.scatter(
        df["request_num_prefill_tokens"],
        df["kv_fragmentation_percent"],
        c=df["kv_blocks_mapped"] if has_blocks else "#1f6feb",
        cmap="viridis" if has_blocks else None,
        s=42,
        edgecolors="white",
        linewidths=0.6,
        zorder=3,
        label="Requests",
    )

    ax_top.set_title(title)
    ax_top.set_ylabel("Fragmentation (%)")
    x_max = float(df["request_num_prefill_tokens"].max())
    ax_top.set_xlim(-0.05 * x_max, 1.02 * x_max)
    ax_top.set_ylim(-5, 100 if not show_block_panel else 105)
    ax_top.axhline(
        5.0,
        linestyle=":",
        linewidth=1.8,
        color="#d1242f",
        label="5% significance threshold",
        zorder=1,
    )
    ax_top.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    if analytical is not None:
        analytical_df = analytical.data
        ax_top.plot(
            analytical_df["request_num_prefill_tokens"],
            analytical_df["worst_case_fragmentation_percent"],
            linestyle=":",
            linewidth=2.2,
            color="#111111",
            label=(
                "Worst-case tail-waste envelope "
                f"(T={analytical.tokens_per_page})"
            ),
            zorder=4,
        )

    ax_top.legend(loc="upper right")

    if has_blocks:
        colorbar = fig.colorbar(scatter, ax=ax_top, pad=0.01)
        colorbar.set_label("KV blocks mapped")

    if has_blocks and show_block_panel:
        step_df = (
            df.dropna(subset=["kv_blocks_mapped"])
            .drop_duplicates(subset=["request_num_prefill_tokens", "kv_blocks_mapped"])
            .sort_values("request_num_prefill_tokens")
        )
        ax_bottom.step(
            step_df["request_num_prefill_tokens"],
            step_df["kv_blocks_mapped"],
            where="post",
            linewidth=2.0,
            color="#1f6feb",
        )
        ax_bottom.scatter(
            step_df["request_num_prefill_tokens"],
            step_df["kv_blocks_mapped"],
            color="#1f6feb",
            s=28,
            zorder=3,
        )
        if analytical is not None:
            analytical_df = analytical.data
            ax_bottom.plot(
                analytical_df["request_num_prefill_tokens"],
                analytical_df["predicted_blocks"],
                linestyle=":",
                linewidth=2.0,
                color="#111111",
            )
        ax_bottom.set_xlabel("Context Length (prefill tokens)")
        ax_bottom.set_ylabel("Blocks")
        ax_bottom.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    else:
        ax_top.set_xlabel("Context Length (prefill tokens)")

    fig.tight_layout()
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot)
    plt.close(fig)


def plot_context_vs_fragmentation(
    df: pd.DataFrame,
    out_plot: Path,
    title: str,
    bins: int,
    analytical: AnalyticalPrediction | None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6), dpi=140)
    ax.scatter(
        df["request_num_prefill_tokens"],
        df["kv_fragmentation_percent"],
        alpha=0.75,
        s=36,
        color="#1f6feb",
        edgecolors="none",
        label="Requests",
    )
    add_binned_trend(df, ax, bins=bins)
    if analytical is not None:
        analytical_df = analytical.data
        ax.plot(
            analytical_df["request_num_prefill_tokens"],
            analytical_df["worst_case_fragmentation_percent"],
            linestyle=":",
            linewidth=2.2,
            color="#111111",
            label=(
                "Worst-case tail-waste envelope "
                f"(T={analytical.tokens_per_page})"
            ),
        )
    ax.set_title(title)
    ax.set_xlabel("Context Length (prefill tokens)")
    ax.set_ylabel("Fragmentation (%)")
    x_max = float(df["request_num_prefill_tokens"].max())
    ax.set_xlim(-0.05 * x_max, 1.02 * x_max)
    ax.set_ylim(-5, 105)
    ax.axhline(
        5.0,
        linestyle=":",
        linewidth=1.8,
        color="#d1242f",
        label="5% significance threshold",
        zorder=1,
    )
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend()
    fig.tight_layout()

    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot context length vs fragmentation")
    parser.add_argument("--input", type=Path, required=True, help="Path to sequence_metrics.csv")
    parser.add_argument("--out-plot", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--out-summary", type=Path, required=True, help="Output summary CSV path")
    parser.add_argument("--title", type=str, default="Context Length vs Fragmentation")
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument(
        "--max-context",
        type=int,
        default=None,
        help="Optional upper bound on request_num_prefill_tokens before plotting.",
    )
    parser.add_argument(
        "--plot-style",
        choices=("raw_sawtooth", "scatter_binned"),
        default="raw_sawtooth",
        help="Plot the raw sawtooth pattern directly, or use the earlier scatter+binned trend view.",
    )
    parser.add_argument(
        "--show-block-panel",
        action="store_true",
        help="For raw_sawtooth plots, include the lower block-count step panel.",
    )
    args = parser.parse_args()

    raw = load_metrics(args.input)
    df = clean_for_plot(raw, max_context=args.max_context)
    if df.empty:
        raise RuntimeError("No valid rows remained after cleaning sequence_metrics.csv")

    analytical = build_analytical_prediction(df)

    if args.plot_style == "raw_sawtooth":
        plot_raw_sawtooth(
            df,
            args.out_plot,
            args.title,
            analytical,
            show_block_panel=args.show_block_panel,
        )
    else:
        plot_context_vs_fragmentation(df, args.out_plot, args.title, args.bins, analytical)
    write_summary(df, args.out_summary, analytical)

    print(f"Plotted {len(df)} requests")
    print(f"Plot: {args.out_plot}")
    print(f"Summary: {args.out_summary}")
    if analytical is not None:
        print(
            "Analytical prediction: "
            f"tokens_per_page={analytical.tokens_per_page}, "
            f"decode_reservation_tokens={analytical.decode_reservation_tokens}, "
            f"context_cap_tokens={analytical.context_cap_tokens}, "
            f"exact_mean_abs_error={analytical.exact_mean_abs_error:.6g}, "
            f"exact_max_abs_error={analytical.exact_max_abs_error:.6g}, "
            f"average_mean_abs_error={analytical.average_mean_abs_error:.6g}, "
            f"average_max_abs_error={analytical.average_max_abs_error:.6g}, "
            f"worst_case_mean_abs_error={analytical.worst_case_mean_abs_error:.6g}, "
            f"worst_case_max_abs_error={analytical.worst_case_max_abs_error:.6g}"
        )


if __name__ == "__main__":
    main()
