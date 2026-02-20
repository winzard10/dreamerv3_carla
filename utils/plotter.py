import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def find_event_dirs(root):
    event_dirs = []
    for dirpath, _, filenames in os.walk(root):
        if any("tfevents" in f for f in filenames):
            event_dirs.append(dirpath)
    return sorted(set(event_dirs))

def export_one_run(run_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    ea = event_accumulator.EventAccumulator(
        run_dir,
        size_guidance={"scalars": 0},  # load all scalars
    )
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    if not tags:
        print(f"[WARN] No scalar tags found in {run_dir}")
        return

    print(f"\nRun: {run_dir}")
    print(f"Found tags: {tags}")

    for tag in tags:
        events = ea.Scalars(tag)
        if not events:
            continue

        data = pd.DataFrame([(e.step, e.value) for e in events], columns=["Step", "Value"])

        plt.figure(figsize=(10, 6))
        plt.plot(data["Step"], data["Value"], alpha=0.3, label="Raw")

        window = min(10, len(data))
        data["Smoothed"] = data["Value"].rolling(window=window, min_periods=1).mean()
        plt.plot(data["Step"], data["Smoothed"], linewidth=2, label=f"Smoothed({window})")

        clean_name = tag.replace("/", "_")
        plt.title(f"DreamerV3 Training: {tag}", fontsize=14)

        tag_l = tag.lower()
        if tag_l.startswith("pretrain/"):
            plt.xlabel("Training Step", fontsize=12)
        elif "episode_reward" in tag_l or tag_l.endswith("episode_reward"):
            plt.xlabel("Episode", fontsize=12)
        else:
            plt.xlabel("Step", fontsize=12)

        plt.ylabel("Value", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()

        out_path = os.path.join(output_dir, f"{clean_name}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")

def export_tb_to_plots(log_root, output_root="./plots"):
    run_dirs = find_event_dirs(log_root)
    if not run_dirs:
        print(f"[ERROR] No tfevents files found under: {log_root}")
        return

    for run_dir in run_dirs:
        run_name = os.path.relpath(run_dir, log_root).replace(os.sep, "_")
        out_dir = os.path.join(output_root, run_name)
        export_one_run(run_dir, out_dir)

if __name__ == "__main__":
    export_tb_to_plots("./runs/dreamerv3_carla")