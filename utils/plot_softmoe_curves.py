#!/usr/bin/env python3
import argparse
import glob
import html
import json
import math
import os
from typing import Dict, List, Sequence, Tuple


PLOT_WIDTH = 1200
PLOT_HEIGHT = 900
MARGIN_LEFT = 80
MARGIN_RIGHT = 20
MARGIN_TOP = 50
MARGIN_BOTTOM = 50
PANEL_GAP = 70
PANEL_HEIGHT = (PLOT_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM - PANEL_GAP) / 2
PANEL_WIDTH = PLOT_WIDTH - MARGIN_LEFT - MARGIN_RIGHT

ACC_COLOR = "#1f77b4"
SERIES_COLORS = {
    "total_loss": "#d62728",
    "ce_loss": "#ff7f0e",
    "prompt_loss": "#9467bd",
    "router_loss": "#2ca02c",
    "proto_loss": "#8c564b",
}
PHASE_COLORS = {
    "warmup": "#777777",
    "train": "#bbbbbb",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True, help="Run directory containing softmoe_losses_task*.jsonl")
    return parser.parse_args()


def load_task_rows(log_dir: str) -> Dict[int, List[dict]]:
    tasks: Dict[int, List[dict]] = {}
    pattern = os.path.join(log_dir, "softmoe_losses_task*.jsonl")
    for path in sorted(glob.glob(pattern)):
        rows: List[dict] = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        if not rows:
            continue
        rows.sort(key=lambda r: (r.get("epoch", 0), r.get("phase_epoch", 0)))
        tasks[rows[0]["task_id"]] = rows
    return tasks


def finite_values(values: Sequence[float]) -> List[float]:
    return [v for v in values if v is not None and math.isfinite(v)]


def padded_range(values: Sequence[float], default: Tuple[float, float]) -> Tuple[float, float]:
    vals = finite_values(values)
    if not vals:
        return default
    lo = min(vals)
    hi = max(vals)
    if lo == hi:
        delta = 1.0 if lo == 0 else max(abs(lo) * 0.1, 1e-6)
        return lo - delta, hi + delta
    span = hi - lo
    pad = span * 0.08
    return lo - pad, hi + pad


def x_scale(value: float, x_min: float, x_max: float, x0: float, width: float) -> float:
    if x_max == x_min:
        return x0 + width / 2.0
    return x0 + (value - x_min) / (x_max - x_min) * width


def y_scale(value: float, y_min: float, y_max: float, y0: float, height: float) -> float:
    if y_max == y_min:
        return y0 + height / 2.0
    return y0 + height - (value - y_min) / (y_max - y_min) * height


def polyline_points(
    xs: Sequence[float],
    ys: Sequence[float],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    x0: float,
    y0: float,
    width: float,
    height: float,
) -> str:
    points = []
    for x, y in zip(xs, ys):
        points.append(
            f"{x_scale(x, x_min, x_max, x0, width):.2f},{y_scale(y, y_min, y_max, y0, height):.2f}"
        )
    return " ".join(points)


def draw_axes(
    parts: List[str],
    title: str,
    x_label: str,
    y_label: str,
    xs: Sequence[float],
    y_min: float,
    y_max: float,
    x0: float,
    y0: float,
    width: float,
    height: float,
) -> None:
    parts.append(
        f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{width:.2f}" height="{height:.2f}" fill="white" stroke="#333" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{x0 + width / 2:.2f}" y="{y0 - 15:.2f}" font-size="18" text-anchor="middle">{html.escape(title)}</text>'
    )

    tick_count = min(6, max(2, len(xs)))
    x_min = min(xs)
    x_max = max(xs)
    for idx in range(tick_count):
        x_value = x_min + (x_max - x_min) * idx / (tick_count - 1) if tick_count > 1 else x_min
        x_pos = x_scale(x_value, x_min, x_max, x0, width)
        parts.append(f'<line x1="{x_pos:.2f}" y1="{y0 + height:.2f}" x2="{x_pos:.2f}" y2="{y0 + height + 6:.2f}" stroke="#333"/>')
        parts.append(
            f'<text x="{x_pos:.2f}" y="{y0 + height + 22:.2f}" font-size="12" text-anchor="middle">{int(round(x_value)) + 1}</text>'
        )

    for idx in range(6):
        y_value = y_min + (y_max - y_min) * idx / 5.0
        y_pos = y_scale(y_value, y_min, y_max, y0, height)
        parts.append(f'<line x1="{x0 - 6:.2f}" y1="{y_pos:.2f}" x2="{x0:.2f}" y2="{y_pos:.2f}" stroke="#333"/>')
        parts.append(
            f'<text x="{x0 - 10:.2f}" y="{y_pos + 4:.2f}" font-size="12" text-anchor="end">{y_value:.2f}</text>'
        )
        if idx not in (0, 5):
            parts.append(
                f'<line x1="{x0:.2f}" y1="{y_pos:.2f}" x2="{x0 + width:.2f}" y2="{y_pos:.2f}" stroke="#eee"/>'
            )

    parts.append(
        f'<text x="{x0 + width / 2:.2f}" y="{y0 + height + 40:.2f}" font-size="14" text-anchor="middle">{html.escape(x_label)}</text>'
    )
    parts.append(
        f'<text x="{x0 - 55:.2f}" y="{y0 + height / 2:.2f}" font-size="14" text-anchor="middle" transform="rotate(-90 {x0 - 55:.2f},{y0 + height / 2:.2f})">{html.escape(y_label)}</text>'
    )


def draw_phase_boundaries(
    parts: List[str],
    rows: Sequence[dict],
    x0: float,
    y0: float,
    width: float,
    height: float,
) -> None:
    xs = [row["epoch"] for row in rows]
    x_min = min(xs)
    x_max = max(xs)
    prev_phase = rows[0]["phase"]
    for prev_row, row in zip(rows, rows[1:]):
        if row["phase"] != prev_phase:
            boundary_epoch = (prev_row["epoch"] + row["epoch"]) / 2.0
            x_pos = x_scale(boundary_epoch, x_min, x_max, x0, width)
            parts.append(
                f'<line x1="{x_pos:.2f}" y1="{y0:.2f}" x2="{x_pos:.2f}" y2="{y0 + height:.2f}" stroke="{PHASE_COLORS.get(row["phase"], "#999")}" stroke-dasharray="6,4" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{x_pos + 6:.2f}" y="{y0 + 16:.2f}" font-size="11" fill="#555">{html.escape(row["phase"])}</text>'
            )
            prev_phase = row["phase"]


def draw_legend(parts: List[str], legend_items: Sequence[Tuple[str, str]], x: float, y: float) -> None:
    for idx, (label, color) in enumerate(legend_items):
        y_pos = y + idx * 18
        parts.append(f'<line x1="{x:.2f}" y1="{y_pos:.2f}" x2="{x + 18:.2f}" y2="{y_pos:.2f}" stroke="{color}" stroke-width="3"/>')
        parts.append(
            f'<text x="{x + 24:.2f}" y="{y_pos + 4:.2f}" font-size="12">{html.escape(label)}</text>'
        )


def build_task_svg(task_id: int, rows: Sequence[dict]) -> str:
    epochs = [row["epoch"] for row in rows]
    x_min = min(epochs)
    x_max = max(epochs)

    acc_values = [row["train_acc"] for row in rows]
    acc_min, acc_max = padded_range(acc_values, (0.0, 1.0))

    loss_series = {
        "total_loss": [row["total_loss"] for row in rows],
        "ce_loss": [row["ce_loss"] for row in rows],
        "prompt_loss": [row["prompt_loss"] for row in rows],
        "router_loss": [row["router_loss"] for row in rows],
        "proto_loss": [row["proto_loss"] for row in rows],
    }
    all_losses: List[float] = []
    for values in loss_series.values():
        all_losses.extend(values)
    loss_min, loss_max = padded_range(all_losses, (-1.0, 1.0))

    top_y0 = MARGIN_TOP
    bottom_y0 = MARGIN_TOP + PANEL_HEIGHT + PANEL_GAP

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{PLOT_WIDTH}" height="{PLOT_HEIGHT}" viewBox="0 0 {PLOT_WIDTH} {PLOT_HEIGHT}">',
        '<rect width="100%" height="100%" fill="#fafafa"/>',
        f'<text x="{PLOT_WIDTH / 2:.2f}" y="28" font-size="22" text-anchor="middle">SoftMoE Task {task_id} Training Curves</text>',
    ]

    draw_axes(
        parts,
        title="Training Accuracy",
        x_label="Epoch",
        y_label="Train Acc",
        xs=epochs,
        y_min=acc_min,
        y_max=acc_max,
        x0=MARGIN_LEFT,
        y0=top_y0,
        width=PANEL_WIDTH,
        height=PANEL_HEIGHT,
    )
    draw_phase_boundaries(parts, rows, MARGIN_LEFT, top_y0, PANEL_WIDTH, PANEL_HEIGHT)
    acc_points = polyline_points(
        epochs,
        acc_values,
        x_min,
        x_max,
        acc_min,
        acc_max,
        MARGIN_LEFT,
        top_y0,
        PANEL_WIDTH,
        PANEL_HEIGHT,
    )
    parts.append(f'<polyline fill="none" stroke="{ACC_COLOR}" stroke-width="2.5" points="{acc_points}"/>')
    draw_legend(parts, [("train_acc", ACC_COLOR)], MARGIN_LEFT + PANEL_WIDTH - 130, top_y0 + 20)

    draw_axes(
        parts,
        title="Loss Curves",
        x_label="Epoch",
        y_label="Loss",
        xs=epochs,
        y_min=loss_min,
        y_max=loss_max,
        x0=MARGIN_LEFT,
        y0=bottom_y0,
        width=PANEL_WIDTH,
        height=PANEL_HEIGHT,
    )
    draw_phase_boundaries(parts, rows, MARGIN_LEFT, bottom_y0, PANEL_WIDTH, PANEL_HEIGHT)
    for name, values in loss_series.items():
        points = polyline_points(
            epochs,
            values,
            x_min,
            x_max,
            loss_min,
            loss_max,
            MARGIN_LEFT,
            bottom_y0,
            PANEL_WIDTH,
            PANEL_HEIGHT,
        )
        parts.append(
            f'<polyline fill="none" stroke="{SERIES_COLORS[name]}" stroke-width="2" points="{points}"/>'
        )
    draw_legend(
        parts,
        [
            ("total_loss", SERIES_COLORS["total_loss"]),
            ("ce_loss", SERIES_COLORS["ce_loss"]),
            ("prompt_loss", SERIES_COLORS["prompt_loss"]),
            ("router_loss", SERIES_COLORS["router_loss"]),
            ("proto_loss", SERIES_COLORS["proto_loss"]),
        ],
        MARGIN_LEFT + PANEL_WIDTH - 170,
        bottom_y0 + 20,
    )

    parts.append("</svg>")
    return "\n".join(parts)


def build_router_proto_svg(task_id: int, rows: Sequence[dict]) -> str:
    epochs = [row["epoch"] for row in rows]
    x_min = min(epochs)
    x_max = max(epochs)

    loss_series = {
        "router_loss": [row["router_loss"] for row in rows],
        "proto_loss": [row["proto_loss"] for row in rows],
    }
    all_losses: List[float] = []
    for values in loss_series.values():
        all_losses.extend(values)
    loss_min, loss_max = padded_range(all_losses, (-1.0, 1.0))

    y0 = MARGIN_TOP
    height = PLOT_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{PLOT_WIDTH}" height="{PLOT_HEIGHT}" viewBox="0 0 {PLOT_WIDTH} {PLOT_HEIGHT}">',
        '<rect width="100%" height="100%" fill="#fafafa"/>',
        f'<text x="{PLOT_WIDTH / 2:.2f}" y="28" font-size="22" text-anchor="middle">SoftMoE Task {task_id} Router/Proto Loss</text>',
    ]

    draw_axes(
        parts,
        title="Router Loss And Proto Loss",
        x_label="Epoch",
        y_label="Loss",
        xs=epochs,
        y_min=loss_min,
        y_max=loss_max,
        x0=MARGIN_LEFT,
        y0=y0,
        width=PANEL_WIDTH,
        height=height,
    )
    draw_phase_boundaries(parts, rows, MARGIN_LEFT, y0, PANEL_WIDTH, height)
    for name in ("router_loss", "proto_loss"):
        points = polyline_points(
            epochs,
            loss_series[name],
            x_min,
            x_max,
            loss_min,
            loss_max,
            MARGIN_LEFT,
            y0,
            PANEL_WIDTH,
            height,
        )
        parts.append(
            f'<polyline fill="none" stroke="{SERIES_COLORS[name]}" stroke-width="2.5" points="{points}"/>'
        )
    draw_legend(
        parts,
        [
            ("router_loss", SERIES_COLORS["router_loss"]),
            ("proto_loss", SERIES_COLORS["proto_loss"]),
        ],
        MARGIN_LEFT + PANEL_WIDTH - 150,
        y0 + 20,
    )

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    tasks = load_task_rows(args.log_dir)
    if not tasks:
        raise SystemExit(f"No softmoe_losses_task*.jsonl found in {args.log_dir}")

    plot_dir = os.path.join(args.log_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    index_links = []
    for task_id, rows in sorted(tasks.items()):
        svg = build_task_svg(task_id, rows)
        out_name = f"task_{task_id:02d}_curves.svg"
        out_path = os.path.join(plot_dir, out_name)
        with open(out_path, "w") as f:
            f.write(svg)
        router_proto_svg = build_router_proto_svg(task_id, rows)
        router_proto_name = f"task_{task_id:02d}_router_proto.svg"
        router_proto_path = os.path.join(plot_dir, router_proto_name)
        with open(router_proto_path, "w") as f:
            f.write(router_proto_svg)
        index_links.append((task_id, out_name, router_proto_name))
        print(f"Wrote {out_path}")
        print(f"Wrote {router_proto_path}")

    index_path = os.path.join(plot_dir, "index.html")
    with open(index_path, "w") as f:
        f.write("<html><body><h1>SoftMoE Curves</h1><ul>\n")
        for task_id, out_name, router_proto_name in index_links:
            f.write(
                f'<li>Task {task_id}: '
                f'<a href="{html.escape(out_name)}">all curves</a> | '
                f'<a href="{html.escape(router_proto_name)}">router/proto</a>'
                "</li>\n"
            )
        f.write("</ul></body></html>\n")
    print(f"Wrote {index_path}")


if __name__ == "__main__":
    main()
