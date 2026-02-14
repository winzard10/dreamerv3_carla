import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def export_tb_to_plots(log_dir, output_dir="./plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Initialize Event Accumulator
    # This reads the binary tfevents files in your /runs folder
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    # 2. Extract Tags (e.g., 'Train/Episode_Reward', 'Train/WM_Loss')
    tags = ea.Tags()['scalars']
    print(f"Found tags: {tags}")

    for tag in tags:
        events = ea.Scalars(tag)
        data = pd.DataFrame([(e.step, e.value) for e in events], columns=['Step', 'Value'])

        # 3. Create the Plot
        plt.figure(figsize=(10, 6))
        
        # Plot raw data with transparency
        plt.plot(data['Step'], data['Value'], alpha=0.3, color='blue', label='Raw')
        
        # Plot smoothed data (Rolling Average)
        # This makes your UMich report look much more professional
        data['Smoothed'] = data['Value'].rolling(window=min(10, len(data))).mean()
        plt.plot(data['Step'], data['Smoothed'], color='red', linewidth=2, label='Smoothed')

        # Formatting
        clean_name = tag.replace('/', '_')
        plt.title(f"DreamerV3 Training: {tag}", fontsize=14)
        if 'Pretrain' in tag or 'WM_Loss' in tag:
            # Pre-training is based on individual optimizer updates/steps
            plt.xlabel("Training Step", fontsize=12)
        elif 'Reward' in tag:
            # Performance in RL is traditionally measured across full environment cycles
            plt.xlabel("Episode", fontsize=12)
        elif 'Actor' in tag or 'Critic' in tag:
            # These online updates typically follow global environment steps
            plt.xlabel("Global Step", fontsize=12)
        else:
            plt.xlabel("Step", fontsize=12)

        plt.ylabel("Value", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        # Save to disk
        plt.savefig(f"{output_dir}/{clean_name}.png", dpi=300)
        plt.close()
        print(f"Saved plot for {tag} to {output_dir}")

if __name__ == "__main__":
    # Point this to your specific run folder inside /runs
    run_folder = "./runs/dreamerv3_carla"
    export_tb_to_plots(run_folder)