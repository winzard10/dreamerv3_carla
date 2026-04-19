import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm
from params import PHASE_A_STEPS

RUN_DIR = "./runs/dreamerv3_carla"
OUTPUT_DIR = "./plots/dreamerv3_carla"
SMOOTH_WINDOW = 10


def clean_tag_name(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", tag)


def get_xlabel(tag: str) -> str:
    tag_l = tag.lower()
    if "episode_reward" in tag_l or tag_l.endswith("episode_reward"):
        return "Episode"
    return "Step"


def export_one_run(run_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    ea = event_accumulator.EventAccumulator(
        run_dir,
        size_guidance={"scalars": 5000},
    )
    print("Loading TensorBoard data... (this may take a while)")
    ea.Reload()
    print("Done.")

    tags = ea.Tags().get("scalars", [])
    if not tags:
        print(f"[WARN] No scalar tags found in {run_dir}")
        return

    print(f"Run: {run_dir}")
    print(f"Found {len(tags)} scalar tags")

    for tag in tqdm(tags, desc="Processing tags"):
        events = ea.Scalars(tag)
        if not events:
            continue

        data = pd.DataFrame(
            [(e.step, e.value) for e in events],
            columns=["Step", "Value"]
        )

        data = data.groupby("Step", as_index=False).last().sort_values("Step")

        plt.figure(figsize=(10, 6))
        plt.plot(data["Step"], data["Value"], alpha=0.3, label="Raw")

        window = min(SMOOTH_WINDOW, len(data))
        data["Smoothed"] = data["Value"].rolling(window=window, min_periods=1).mean()
        plt.plot(data["Step"], data["Smoothed"], linewidth=2, label=f"Smoothed({window})")

        plt.axvline(PHASE_A_STEPS, linestyle="--", color="black", linewidth=1.5, label="Phase A → B")

        plt.title(f"DreamerV3 Training: {tag}", fontsize=14)
        plt.xlabel(get_xlabel(tag), fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()

        clean_name = clean_tag_name(tag)
        out_path = os.path.join(output_dir, f"{clean_name}.png")
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close()

        print(f"Saved: {out_path}")


if __name__ == "__main__":
    export_one_run(RUN_DIR, OUTPUT_DIR)