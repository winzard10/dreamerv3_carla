import numpy as np
import torch
import os
from tqdm import tqdm


class SequenceBuffer:
    def __init__(self, capacity, seq_len, device):
        self.capacity = capacity
        self.seq_len = seq_len
        self.device = device
        self.idx = 0
        self.full = False

        # Pre-allocate memory in RAM
        # RGB stored as uint8: [H, W, 3]
        self.rgbs = np.zeros((capacity, 128, 128, 3), dtype=np.uint8)
        self.goals = np.zeros((capacity, 2), dtype=np.float32)
        self.actions = np.zeros((capacity, 2), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)
        self.vectors = np.zeros((capacity, 3), dtype=np.float32)

    def add(self, rgb, vector, goal, action, reward, done):
        self.rgbs[self.idx] = rgb
        self.goals[self.idx] = goal
        self.vectors[self.idx] = vector
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def sample(self, batch_size):
        max_idx = self.capacity if self.full else self.idx

        if max_idx < self.seq_len:
            return None

        obs_rgb, veccs, goals, acts, rews, terms = [], [], [], [], [], []

        attempts = 0
        while len(obs_rgb) < batch_size and attempts < 1000:
            attempts += 1

            # leave room for full sequence
            start = np.random.randint(0, max_idx - self.seq_len + 1)
            end = start + self.seq_len

            # reject sequences that cross episode boundaries
            if np.any(self.dones[start:end - 1]):
                continue

            obs_rgb.append(self.rgbs[start:end])
            veccs.append(self.vectors[start:end])
            goals.append(self.goals[start:end])
            acts.append(self.actions[start:end])
            rews.append(self.rewards[start:end])
            terms.append(self.dones[start:end])

        if len(obs_rgb) < batch_size:
            return None

        rgb = torch.as_tensor(np.array(obs_rgb), device=self.device, dtype=torch.uint8)   # [B, T, H, W, 3]
        vec = torch.as_tensor(np.array(veccs), device=self.device, dtype=torch.float32)
        goal = torch.as_tensor(np.array(goals), device=self.device, dtype=torch.float32)
        act = torch.as_tensor(np.array(acts), device=self.device, dtype=torch.float32)
        rew = torch.as_tensor(np.array(rews), device=self.device, dtype=torch.float32)
        done = torch.as_tensor(np.array(terms), device=self.device, dtype=torch.bool)

        # convert to channels-first for model input: [B, T, 3, H, W]
        rgb = rgb.permute(0, 1, 4, 2, 3)

        return (rgb, vec, goal, act, rew, done)

    def load_from_disk(self, path):
        if not os.path.exists(path):
            print(f"No expert data found at {path}")
            return

        files = [f for f in os.listdir(path) if f.endswith(".npz")]
        files.sort()

        for file in tqdm(files, desc="Loading Expert Data", unit="file"):
            data = np.load(os.path.join(path, file))
            T = len(data["action"])

            for i in range(T):
                vec = data["vector"][i] if "vector" in data else np.zeros(3, dtype=np.float32)
                goal = data["goal"][i] if "goal" in data else np.zeros(2, dtype=np.float32)
                done = bool(data["done"][i]) if "done" in data else (i == T - 1)

                self.add(
                    data["rgb"][i],
                    vec,
                    goal,
                    data["action"][i],
                    data["reward"][i],
                    done,
                )