import argparse
import io
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorboard.backend.event_processing import event_accumulator


PRETRAIN_TAGS = {
    "World Model Loss": "Pretrain/wm_loss",
    "Depth Recon Loss": "Pretrain/depth_loss",
    "Semantic Recon Loss": "Pretrain/sem_loss",
    "Reward Loss": "Pretrain/reward_loss",
}

ONLINE_TAGS = {
    "Episode Reward": "Train/Episode_Reward",
    "Imagined Return Mean": "Train/imag_return_mean",
    "Critic Loss": "Train/critic_loss",
}

VISUAL_TAGS = {
    "Depth": "Visuals/Depth_Recon",
    "Semantic": "Visuals/Semantic_Recon",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate polished progress-report figures from TensorBoard event files."
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="./runs/dreamerv3_carla",
        help="TensorBoard log directory (contains events.out.tfevents.* files).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./plots/progress_report",
        help="Output directory for report figures.",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.08,
        help="EMA smoothing factor in (0, 1]. Lower means smoother curves.",
    )
    return parser.parse_args()


def setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfbfd",
            "axes.edgecolor": "#d6d6de",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "font.family": "DejaVu Sans",
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
        }
    )


def load_accumulator(logdir: str) -> event_accumulator.EventAccumulator:
    acc = event_accumulator.EventAccumulator(
        logdir, size_guidance={"scalars": 0, "images": 0}
    )
    acc.Reload()
    return acc


def get_scalar_series(
    acc: event_accumulator.EventAccumulator, tag: str
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    events = acc.Scalars(tag)
    if not events:
        return None

    by_step: Dict[int, Tuple[float, float]] = {}
    for e in events:
        prev = by_step.get(e.step)
        if prev is None or e.wall_time >= prev[0]:
            by_step[e.step] = (e.wall_time, e.value)

    steps = np.array(sorted(by_step.keys()), dtype=np.int64)
    values = np.array([by_step[s][1] for s in steps], dtype=np.float64)
    return steps, values


def ema(values: np.ndarray, alpha: float) -> np.ndarray:
    smoothed = np.empty_like(values, dtype=np.float64)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1.0 - alpha) * smoothed[i - 1]
    return smoothed


def plot_metric(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    color: str,
    alpha: float,
) -> None:
    y_smooth = ema(y, alpha=alpha)
    ax.plot(x, y, color=color, linewidth=1.0, alpha=0.22, label="Raw")
    ax.plot(x, y_smooth, color=color, linewidth=2.4, label="EMA")
    ax.set_title(title, pad=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=False, loc="best")


def plot_pretrain(
    acc: event_accumulator.EventAccumulator, outdir: str, alpha: float
) -> Optional[str]:
    colors = ["#264653", "#2A9D8F", "#E76F51", "#F4A261"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    plotted = 0
    for idx, ((title, tag), color) in enumerate(zip(PRETRAIN_TAGS.items(), colors)):
        series = get_scalar_series(acc, tag)
        if series is None:
            axes[idx].set_visible(False)
            continue
        steps, values = series
        plot_metric(
            ax=axes[idx],
            x=steps,
            y=values,
            title=title,
            xlabel="Training Step",
            ylabel="Loss",
            color=color,
            alpha=alpha,
        )
        plotted += 1

    for ax in axes[plotted:]:
        if ax.has_data():
            continue
        ax.set_visible(False)

    if plotted == 0:
        plt.close(fig)
        return None

    fig.suptitle("Figure 1. Pretraining Loss Curves", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = os.path.join(outdir, "fig1_pretrain_losses.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_online(
    acc: event_accumulator.EventAccumulator, outdir: str, alpha: float
) -> Optional[str]:
    colors = ["#1D3557", "#2A9D8F", "#E63946"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    plotted = 0
    for idx, ((title, tag), color) in enumerate(zip(ONLINE_TAGS.items(), colors)):
        series = get_scalar_series(acc, tag)
        if series is None:
            axes[idx].set_visible(False)
            continue

        steps, values = series
        xlabel = "Episode" if "Episode_Reward" in tag else "Training Step"
        ylabel = "Reward" if "Episode_Reward" in tag else "Value"

        plot_metric(
            ax=axes[idx],
            x=steps,
            y=values,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            color=color,
            alpha=alpha,
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None

    fig.suptitle("Figure 2. Online Training Metrics", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path = os.path.join(outdir, "fig2_online_metrics.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def get_latest_image_halves(
    acc: event_accumulator.EventAccumulator, tag: str
) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    image_events = acc.Images(tag)
    if not image_events:
        return None

    latest = max(image_events, key=lambda x: (x.step, x.wall_time))
    img = Image.open(io.BytesIO(latest.encoded_image_string))
    img_arr = np.array(img)
    if img_arr.ndim == 3:
        img_arr = img_arr[..., 0]

    width = img_arr.shape[1]
    mid = width // 2
    gt = img_arr[:, :mid]
    recon = img_arr[:, mid:]
    return gt, recon, int(latest.step)


def plot_qualitative(acc: event_accumulator.EventAccumulator, outdir: str) -> Optional[str]:
    depth_data = get_latest_image_halves(acc, VISUAL_TAGS["Depth"])
    sem_data = get_latest_image_halves(acc, VISUAL_TAGS["Semantic"])

    if depth_data is None and sem_data is None:
        return None

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))

    panels: List[Tuple[str, Optional[np.ndarray], Optional[int]]] = []
    if depth_data is not None:
        d_gt, d_recon, d_step = depth_data
        panels.extend([("Depth GT", d_gt, d_step), ("Depth Recon", d_recon, d_step)])
    else:
        panels.extend([("Depth GT", None, None), ("Depth Recon", None, None)])

    if sem_data is not None:
        s_gt, s_recon, s_step = sem_data
        panels.extend([("Semantic GT", s_gt, s_step), ("Semantic Recon", s_recon, s_step)])
    else:
        panels.extend([("Semantic GT", None, None), ("Semantic Recon", None, None)])

    for idx, (title, img_data, step) in enumerate(panels):
        ax = axes[idx]
        if img_data is None:
            ax.set_visible(False)
            continue

        ax.imshow(img_data, cmap="gray", interpolation="nearest")
        ax.set_title(title, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(
            0.02,
            0.04,
            f"step={step}",
            color="white",
            fontsize=9,
            transform=ax.transAxes,
            bbox={"facecolor": "black", "alpha": 0.45, "pad": 3, "edgecolor": "none"},
        )

    fig.suptitle("Figure 3. Qualitative Reconstruction (GT vs Recon)", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out_path = os.path.join(outdir, "fig3_qualitative_recon.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path

def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    setup_style()

    acc = load_accumulator(args.logdir)
    tags = acc.Tags()
    print(f"[INFO] Loaded logdir: {args.logdir}")
    print(f"[INFO] Scalar tags: {len(tags.get('scalars', []))}, Image tags: {len(tags.get('images', []))}")

    outputs: List[Tuple[str, Optional[str]]] = []
    outputs.append(("Figure 1 (Pretrain losses)", plot_pretrain(acc, args.outdir, args.ema_alpha)))
    outputs.append(("Figure 2 (Online metrics)", plot_online(acc, args.outdir, args.ema_alpha)))
    outputs.append(("Figure 3 (Qualitative recon)", plot_qualitative(acc, args.outdir)))

    print("\n[INFO] Export results:")
    for name, path in outputs:
        if path is None:
            print(f" - {name}: skipped (required tags not found)")
        else:
            print(f" - {name}: {path}")


if __name__ == "__main__":
    main()
