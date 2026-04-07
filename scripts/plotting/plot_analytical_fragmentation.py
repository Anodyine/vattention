#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class AnalyticalGeometry:
    architecture: str
    page_size_bytes: int
    dtype_bytes: int | None
    tokens_per_page: int
    page_buffer_token_bytes: int
    max_context: int
    num_layers: int | None = None
    num_kv_heads_local: int | None = None
    head_size: int | None = None
    kv_lora_rank: int | None = None
    qk_rope_head_dim: int | None = None
    source: str = ""


def load_top_level_yaml(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")

    parsed: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip().strip("'").strip('"')
    return parsed


def maybe_int(value: str | None) -> int | None:
    if value is None:
        return None

    normalized = value.strip().lower()
    if normalized in {"", "null", "none"}:
        return None
    return int(value)


def parse_server_log(path: Path) -> dict[str, int | str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing server log: {path}")

    text = path.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "architecture": r"> Architecture:\s+([A-Za-z0-9_]+)",
        "layers": r"> Layers:\s+(\d+), Heads:\s+(\d+), Head Size:\s+(\d+)",
        "tokens_per_page": r"> Tokens Per Page:\s+(\d+)",
        "page_buffer_token_bytes": r"> Page Buffer Token Bytes:\s+(\d+)",
    }

    result: dict[str, int | str] = {}

    architecture_match = re.search(patterns["architecture"], text)
    if architecture_match:
        result["architecture"] = architecture_match.group(1)

    layer_match = re.search(patterns["layers"], text)
    if layer_match:
        result["layers"] = int(layer_match.group(1))
        result["heads"] = int(layer_match.group(2))
        result["head_size"] = int(layer_match.group(3))

    for key in ("tokens_per_page", "page_buffer_token_bytes"):
        match = re.search(patterns[key], text)
        if match:
            result[key] = int(match.group(1))

    return result


def infer_dtype_bytes(dtype_name: str | None) -> int | None:
    if dtype_name is None:
        return None

    normalized = dtype_name.lower()
    if normalized in {"float16", "half", "bfloat16"}:
        return 2
    if normalized in {"float32", "float"}:
        return 4
    if normalized in {"float64", "double"}:
        return 8
    return None


def infer_geometry_from_run(run_dir: Path, server_log: Path | None) -> AnalyticalGeometry:
    cfg = load_top_level_yaml(run_dir / "config.yml")
    hf_config_path = run_dir / "hf_config.json"
    cache_layout_path = run_dir / "cache_layout.json"
    log_info = parse_server_log(server_log) if server_log is not None else {}

    page_size_bytes = int(cfg["block_size"])
    max_context = int(cfg["max_model_len"])
    dtype_bytes = infer_dtype_bytes(cfg.get("dtype"))

    architecture = str(log_info.get("architecture", "unknown"))
    tokens_per_page = maybe_int(str(log_info.get("tokens_per_page", "")))
    page_buffer_token_bytes = maybe_int(str(log_info.get("page_buffer_token_bytes", "")))
    layers = maybe_int(str(log_info.get("layers", "")))
    heads = maybe_int(str(log_info.get("heads", "")))
    head_size = maybe_int(str(log_info.get("head_size", "")))

    if hf_config_path.exists() and cache_layout_path.exists():
        hf_config = json.loads(hf_config_path.read_text(encoding="utf-8"))
        cache_layout = json.loads(cache_layout_path.read_text(encoding="utf-8"))
        total_num_kv_heads = hf_config.get(
            "num_key_value_heads",
            hf_config.get("num_attention_heads"),
        )
        tp_size = int(cache_layout.get("tensor_parallel_size", 1))
        head_dim = hf_config.get("head_dim")
        if head_dim is None and hf_config.get("hidden_size") is not None and hf_config.get("num_attention_heads") is not None:
            head_dim = int(hf_config["hidden_size"]) // int(hf_config["num_attention_heads"])
        return AnalyticalGeometry(
            architecture=str(cache_layout["architecture"]),
            page_size_bytes=int(cache_layout["page_size"]),
            dtype_bytes=infer_dtype_bytes(hf_config.get("torch_dtype")) or dtype_bytes,
            tokens_per_page=int(cache_layout["tokens_per_page"]),
            page_buffer_token_bytes=int(cache_layout["page_buffer_token_bytes"]),
            max_context=int(cache_layout["max_model_len"]),
            num_layers=maybe_int(str(hf_config.get("num_hidden_layers"))),
            num_kv_heads_local=(
                int(total_num_kv_heads) // tp_size
                if str(cache_layout["architecture"]) == "dense_kv" and total_num_kv_heads is not None
                else None
            ),
            head_size=maybe_int(str(head_dim)) if head_dim is not None else None,
            kv_lora_rank=maybe_int(str(hf_config.get("kv_lora_rank"))),
            qk_rope_head_dim=maybe_int(str(hf_config.get("qk_rope_head_dim"))),
            source=f"run:{run_dir} (saved hf_config/cache_layout)",
        )

    if tokens_per_page is None or page_buffer_token_bytes is None:
        raise ValueError(
            "Could not infer cache geometry from saved run artifacts alone. "
            "Pass --server-log or explicit model parameters."
        )

    return AnalyticalGeometry(
        architecture=architecture,
        page_size_bytes=page_size_bytes,
        dtype_bytes=dtype_bytes,
        tokens_per_page=tokens_per_page,
        page_buffer_token_bytes=page_buffer_token_bytes,
        max_context=max_context,
        num_layers=layers,
        num_kv_heads_local=heads,
        head_size=head_size,
        source=f"run:{run_dir}",
    )


def fetch_hf_config(model_id: str, token: str | None) -> dict:
    url = f"https://huggingface.co/{quote(model_id, safe='/')}/resolve/main/config.json"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(url, headers=headers)
    with urlopen(request, timeout=30) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def build_geometry_from_hf_config(
    hf_config: dict,
    *,
    model_id: str,
    page_size: int,
    max_context: int,
    tensor_parallel_size: int,
) -> AnalyticalGeometry:
    dtype_bytes = infer_dtype_bytes(hf_config.get("torch_dtype"))

    if "kv_lora_rank" in hf_config and "qk_rope_head_dim" in hf_config:
        architecture = "mla"
        kv_lora_rank = int(hf_config["kv_lora_rank"])
        qk_rope_head_dim = int(hf_config["qk_rope_head_dim"])
        page_buffer_token_bytes = (kv_lora_rank + qk_rope_head_dim) * dtype_bytes
        num_kv_heads_local = None
        head_size = None
    else:
        architecture = "dense_kv"
        total_num_kv_heads = int(
            hf_config.get(
                "num_key_value_heads",
                hf_config.get("num_attention_heads"),
            )
        )
        hidden_size = int(hf_config["hidden_size"])
        head_size = int(
            hf_config.get(
                "head_dim",
                hidden_size // int(hf_config["num_attention_heads"]),
            )
        )
        num_kv_heads_local = total_num_kv_heads // tensor_parallel_size
        page_buffer_token_bytes = num_kv_heads_local * head_size * dtype_bytes
        kv_lora_rank = None
        qk_rope_head_dim = None

    tokens_per_page = page_size // page_buffer_token_bytes

    return AnalyticalGeometry(
        architecture=architecture,
        page_size_bytes=page_size,
        dtype_bytes=dtype_bytes,
        tokens_per_page=tokens_per_page,
        page_buffer_token_bytes=page_buffer_token_bytes,
        max_context=max_context,
        num_layers=maybe_int(str(hf_config.get("num_hidden_layers"))),
        num_kv_heads_local=num_kv_heads_local,
        head_size=head_size,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        source=f"hf:{model_id}",
    )


def build_geometry_from_args(args: argparse.Namespace) -> AnalyticalGeometry:
    if args.tokens_per_page is not None:
        page_buffer_token_bytes = args.page_buffer_token_bytes
        if page_buffer_token_bytes is None:
            if args.page_size is None:
                raise ValueError("Provide --page-size with --tokens-per-page or set --page-buffer-token-bytes.")
            page_buffer_token_bytes = args.page_size // args.tokens_per_page
        if args.page_size is None:
            raise ValueError("Provide --page-size when using --tokens-per-page.")
        if args.max_context is None:
            raise ValueError("Provide --max-context when using explicit geometry.")
        return AnalyticalGeometry(
            architecture=args.architecture or "unknown",
            page_size_bytes=args.page_size,
            dtype_bytes=args.dtype_bytes,
            tokens_per_page=args.tokens_per_page,
            page_buffer_token_bytes=page_buffer_token_bytes,
            max_context=args.max_context,
            num_layers=args.num_layers,
            num_kv_heads_local=args.num_kv_heads_local,
            head_size=args.head_size,
            kv_lora_rank=args.kv_lora_rank,
            qk_rope_head_dim=args.qk_rope_head_dim,
            source="explicit tokens_per_page",
        )

    if args.page_size is None or args.dtype_bytes is None or args.max_context is None:
        raise ValueError(
            "Explicit analytical geometry requires --page-size, --dtype-bytes, and --max-context."
        )

    if args.architecture == "dense_kv":
        if args.num_kv_heads_local is None or args.head_size is None:
            raise ValueError(
                "Dense-KV analytical geometry requires --num-kv-heads-local and --head-size."
            )
        page_buffer_token_bytes = args.num_kv_heads_local * args.head_size * args.dtype_bytes
    elif args.architecture == "mla":
        if args.kv_lora_rank is None or args.qk_rope_head_dim is None:
            raise ValueError(
                "MLA analytical geometry requires --kv-lora-rank and --qk-rope-head-dim."
            )
        page_buffer_token_bytes = (
            args.kv_lora_rank + args.qk_rope_head_dim
        ) * args.dtype_bytes
    else:
        raise ValueError("Provide either --run-dir or an explicit --architecture.")

    tokens_per_page = args.page_size // page_buffer_token_bytes
    return AnalyticalGeometry(
        architecture=args.architecture,
        page_size_bytes=args.page_size,
        dtype_bytes=args.dtype_bytes,
        tokens_per_page=tokens_per_page,
        page_buffer_token_bytes=page_buffer_token_bytes,
        max_context=args.max_context,
        num_layers=args.num_layers,
        num_kv_heads_local=args.num_kv_heads_local,
        head_size=args.head_size,
        kv_lora_rank=args.kv_lora_rank,
        qk_rope_head_dim=args.qk_rope_head_dim,
        source="explicit model parameters",
    )


def exact_fragmentation_percent(context: np.ndarray, tokens_per_page: int) -> np.ndarray:
    mapped_blocks = np.ceil(context / tokens_per_page)
    return 100.0 * (mapped_blocks * tokens_per_page - context) / (mapped_blocks * tokens_per_page)


def worst_case_fragmentation_percent(context: np.ndarray, tokens_per_page: int) -> np.ndarray:
    return 100.0 * tokens_per_page / (context + tokens_per_page)


def make_formula_strings(geometry: AnalyticalGeometry) -> tuple[str, str]:
    if geometry.architecture == "mla":
        t_formula = (
            "T = floor(P / ((r_kv + d_rope) * b))"
        )
    elif geometry.architecture == "dense_kv":
        t_formula = (
            "T = floor(P / (H_kv,loc * D * b))"
        )
    else:
        t_formula = "T = floor(P / page_buffer_token_bytes)"

    f_formula = "F_exact(C) = 100 * (ceil(C / T) * T - C) / (ceil(C / T) * T)"
    return t_formula, f_formula


def write_summary(geometry: AnalyticalGeometry, out_csv: Path) -> None:
    t_formula, f_formula = make_formula_strings(geometry)
    row = {
        "architecture": geometry.architecture,
        "page_size_bytes": geometry.page_size_bytes,
        "dtype_bytes": geometry.dtype_bytes,
        "page_buffer_token_bytes": geometry.page_buffer_token_bytes,
        "tokens_per_page": geometry.tokens_per_page,
        "max_context": geometry.max_context,
        "num_layers": geometry.num_layers,
        "num_kv_heads_local": geometry.num_kv_heads_local,
        "head_size": geometry.head_size,
        "kv_lora_rank": geometry.kv_lora_rank,
        "qk_rope_head_dim": geometry.qk_rope_head_dim,
        "tokens_per_page_formula": t_formula,
        "exact_fragmentation_formula": f_formula,
        "source": geometry.source,
    }
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out_csv, index=False)


def plot_analytical_curves(
    geometry: AnalyticalGeometry,
    out_plot: Path,
    title: str,
    include_worst_case: bool,
) -> None:
    context = np.arange(1, geometry.max_context + 1)
    exact = exact_fragmentation_percent(context, geometry.tokens_per_page)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
    ax.plot(
        context,
        exact,
        linewidth=1.8,
        color="#1f6feb",
        label="Exact analytical sawtooth",
    )

    if include_worst_case:
        worst = worst_case_fragmentation_percent(context, geometry.tokens_per_page)
        ax.plot(
            context,
            worst,
            linestyle=":",
            linewidth=2.2,
            color="#111111",
            label="Worst-case envelope",
        )

    ax.set_title(title)
    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Fragmentation (%)")
    ax.set_xlim(-0.05 * geometry.max_context, 1.02 * geometry.max_context)
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
    ax.legend(loc="upper right")

    subtitle = (
        f"arch={geometry.architecture}, T={geometry.tokens_per_page}, "
        f"page_buffer_token_bytes={geometry.page_buffer_token_bytes}"
    )
    fig.text(0.5, 0.01, subtitle, ha="center", fontsize=9)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot analytical fragmentation curves from saved run geometry or explicit model parameters."
    )
    parser.add_argument("--run-dir", type=Path, default=None, help="Path to server-output/<run> directory.")
    parser.add_argument("--server-log", type=Path, default=None, help="Optional server log to infer cache geometry.")
    parser.add_argument("--hf-model-id", type=str, default=None, help="Optional Hugging Face repo id, like mistralai/Mistral-Nemo-Base-2407.")
    parser.add_argument("--hf-token-env", type=str, default="HF_TOKEN", help="Environment variable holding a Hugging Face token for gated/private models.")
    parser.add_argument("--tensor-parallel-size", type=int, default=None, help="Tensor parallel degree for local per-worker cache geometry.")
    parser.add_argument("--architecture", choices=("dense_kv", "mla"), default=None)
    parser.add_argument("--page-size", type=int, default=None)
    parser.add_argument("--dtype-bytes", type=int, default=None)
    parser.add_argument("--max-context", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-kv-heads-local", type=int, default=None)
    parser.add_argument("--head-size", type=int, default=None)
    parser.add_argument("--kv-lora-rank", type=int, default=None)
    parser.add_argument("--qk-rope-head-dim", type=int, default=None)
    parser.add_argument("--page-buffer-token-bytes", type=int, default=None)
    parser.add_argument("--tokens-per-page", type=int, default=None)
    parser.add_argument("--out-plot", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument(
        "--title",
        type=str,
        default="Analytical Fragmentation Curve",
    )
    parser.add_argument(
        "--include-worst-case",
        action="store_true",
        help="Also draw the smooth worst-case envelope.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.hf_model_id is not None:
        run_geometry = None
        if args.run_dir is not None:
            run_geometry = infer_geometry_from_run(args.run_dir, args.server_log)

        page_size = (
            args.page_size
            if args.page_size is not None
            else (run_geometry.page_size_bytes if run_geometry is not None else None)
        )
        max_context = (
            args.max_context
            if args.max_context is not None
            else (run_geometry.max_context if run_geometry is not None else None)
        )
        tensor_parallel_size = args.tensor_parallel_size
        if tensor_parallel_size is None and args.run_dir is not None:
            cfg = load_top_level_yaml(args.run_dir / "config.yml")
            tensor_parallel_size = int(cfg["tensor_parallel_size"])

        if page_size is None or max_context is None or tensor_parallel_size is None:
            raise ValueError(
                "HF-based analytical geometry needs page size, max context, and tensor parallel size. "
                "Provide them explicitly or pass --run-dir so they can be inferred."
            )

        token = os.environ.get(args.hf_token_env)
        hf_config = fetch_hf_config(args.hf_model_id, token)
        geometry = build_geometry_from_hf_config(
            hf_config,
            model_id=args.hf_model_id,
            page_size=page_size,
            max_context=max_context,
            tensor_parallel_size=tensor_parallel_size,
        )
    elif args.run_dir is not None:
        geometry = infer_geometry_from_run(args.run_dir, args.server_log)
        if args.max_context is not None:
            geometry = AnalyticalGeometry(
                architecture=geometry.architecture,
                page_size_bytes=geometry.page_size_bytes,
                dtype_bytes=geometry.dtype_bytes,
                page_buffer_token_bytes=geometry.page_buffer_token_bytes,
                tokens_per_page=geometry.tokens_per_page,
                max_context=args.max_context,
                num_layers=geometry.num_layers,
                num_kv_heads_local=geometry.num_kv_heads_local,
                head_size=geometry.head_size,
                kv_lora_rank=geometry.kv_lora_rank,
                qk_rope_head_dim=geometry.qk_rope_head_dim,
                source=geometry.source,
            )
    else:
        geometry = build_geometry_from_args(args)

    plot_analytical_curves(
        geometry=geometry,
        out_plot=args.out_plot,
        title=args.title,
        include_worst_case=args.include_worst_case,
    )
    write_summary(geometry, args.out_summary)

    print(f"Plot: {args.out_plot}")
    print(f"Summary: {args.out_summary}")
    print(
        f"Analytical geometry: arch={geometry.architecture}, "
        f"tokens_per_page={geometry.tokens_per_page}, "
        f"page_buffer_token_bytes={geometry.page_buffer_token_bytes}, "
        f"source={geometry.source}"
    )


if __name__ == "__main__":
    main()
