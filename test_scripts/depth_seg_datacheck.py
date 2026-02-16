import numpy as np
import matplotlib.pyplot as plt

# Load the first sequence file
file_path = "./data/expert_sequences/seq_0.npz"
data = np.load(file_path)

# Extract depth and semantic data
depth_data = data['depth']
semantic_data = data['semantic']

print(f"--- Data Audit for: {file_path} ---")
# Check Depth
print(f"Depth Shape: {depth_data.shape}")
print(f"Depth - Max Value: {depth_data.max()}") # Should be near 255 if using LogarithmicDepth
print(f"Depth - Min Value: {depth_data.min()}")
print(f"Depth - Mean Value: {depth_data.mean():.4f}")

# Check Semantic
print(f"Semantic - Max Value: {semantic_data.max()}") # Should be around 25-28
print(f"Semantic - Mean Value: {semantic_data.mean():.4f}")

# Visualization
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Plot Depth (using 'inferno' or 'magma' colormap to see gradients clearly)
im1 = ax[0].imshow(depth_data[0].squeeze(), cmap='inferno')
ax[0].set_title("Depth Map (Logarithmic)")
plt.colorbar(im1, ax=ax[0])

# Plot Semantic
im2 = ax[1].imshow(semantic_data[0].squeeze(), cmap='tab20')
ax[1].set_title("Semantic Map")
plt.colorbar(im2, ax=ax[1])

plt.tight_layout()
plt.show()