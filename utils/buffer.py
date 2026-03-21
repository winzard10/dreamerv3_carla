import numpy as np
import torch
import os
from tqdm import tqdm


class SequenceBuffer:
    def __init__(
        self,
        capacity,
        seq_len,
        device,
        recent_ratio=0.5,
        recent_window_fraction=0.2,
        sample_attempt_budget=2000,
    ):
        self.capacity = capacity
        self.seq_len = seq_len
        self.device = device
        self.idx = 0
        self.full = False

        # Plan E: mixed sampling controls
        self.recent_ratio = float(np.clip(recent_ratio, 0.0, 1.0))
        self.recent_window_fraction = float(max(recent_window_fraction, 0.05))
        self.sample_attempt_budget = int(max(sample_attempt_budget, 500))

        # Pre-allocate memory in RAM (uint8 saves ~75% space vs float32)
        self.depths = np.zeros((capacity, 160, 160, 1), dtype=np.uint8)
        self.semantics = np.zeros((capacity, 160, 160, 1), dtype=np.uint8)
        self.goals = np.zeros((capacity, 2), dtype=np.float32)
        self.actions = np.zeros((capacity, 2), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)
        self.vectors = np.zeros((capacity, 3), dtype=np.float32)

    def add(self, depth, semantic, vector, goal, action, reward, done):
        self.depths[self.idx] = depth
        self.semantics[self.idx] = semantic
        self.goals[self.idx] = goal
        self.vectors[self.idx] = vector
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def _draw_start(self, max_idx, prefer_recent):
        """
        Draw a valid start index in [0, max_idx - seq_len].
        prefer_recent=True samples from the latest window first.
        """
        max_start = max_idx - self.seq_len
        if max_start < 0:
            return None

        if prefer_recent:
            recent_window = max(self.seq_len, int(max_idx * self.recent_window_fraction))
            recent_low = max(0, max_start - recent_window + 1)
            recent_high = max_start + 1  # exclusive
            if recent_high > recent_low:
                return int(np.random.randint(recent_low, recent_high))

        # Fallback uniform over all starts
        return int(np.random.randint(0, max_start + 1))

    def _is_valid_window(self, start):
        end = start + self.seq_len
        # Keep original boundary rule: dones allowed only at final step.
        return not np.any(self.dones[start : end - 1])

    def _collect_windows(self, target_count, max_idx, prefer_recent, max_attempts):
        starts = []
        attempts = 0
        while len(starts) < target_count and attempts < max_attempts:
            attempts += 1
            start = self._draw_start(max_idx, prefer_recent=prefer_recent)
            if start is None:
                break
            if not self._is_valid_window(start):
                continue
            starts.append(start)
        return starts

    def sample(self, batch_size):
        # 1. Determine the search range based on whether the buffer is full
        max_idx = self.capacity if self.full else self.idx

        # We need at least seq_len steps to create one sequence
        if max_idx < self.seq_len:
            return None

        # Plan E: 50% recent + 50% uniform by default.
        target_recent = int(round(batch_size * self.recent_ratio))
        target_recent = max(0, min(batch_size, target_recent))
        target_uniform = batch_size - target_recent

        total_budget = max(self.sample_attempt_budget, batch_size * 20)
        recent_budget = max(total_budget // 2, target_recent * 10)
        uniform_budget = max(total_budget // 2, target_uniform * 10)

        starts = []

        # Recent-biased portion
        if target_recent > 0:
            starts.extend(
                self._collect_windows(
                    target_count=target_recent,
                    max_idx=max_idx,
                    prefer_recent=True,
                    max_attempts=recent_budget,
                )
            )

        # Uniform portion (and fills if recent was short)
        remaining = batch_size - len(starts)
        if remaining > 0:
            starts.extend(
                self._collect_windows(
                    target_count=remaining,
                    max_idx=max_idx,
                    prefer_recent=False,
                    max_attempts=uniform_budget,
                )
            )

        # Last fallback try (uniform) if still short
        remaining = batch_size - len(starts)
        if remaining > 0:
            starts.extend(
                self._collect_windows(
                    target_count=remaining,
                    max_idx=max_idx,
                    prefer_recent=False,
                    max_attempts=total_budget,
                )
            )

        # If we couldn't find enough valid sequences, return None
        if len(starts) < batch_size:
            return None

        obs_depth, obs_sem, veccs, goals, acts, rews, terms = [], [], [], [], [], [], []
        for start in starts[:batch_size]:
            end = start + self.seq_len
            obs_depth.append(self.depths[start:end])
            obs_sem.append(self.semantics[start:end])
            veccs.append(self.vectors[start:end])
            goals.append(self.goals[start:end])
            acts.append(self.actions[start:end])
            rews.append(self.rewards[start:end])
            terms.append(self.dones[start:end])

        # 5. Convert to Tensors
        depth = torch.as_tensor(np.array(obs_depth), device=self.device, dtype=torch.uint8)  # [B,T,H,W,1]
        sem = torch.as_tensor(np.array(obs_sem), device=self.device, dtype=torch.uint8)
        vec = torch.as_tensor(np.array(veccs), device=self.device, dtype=torch.float32)
        goal = torch.as_tensor(np.array(goals), device=self.device, dtype=torch.float32)
        act = torch.as_tensor(np.array(acts), device=self.device, dtype=torch.float32)
        rew = torch.as_tensor(np.array(rews), device=self.device, dtype=torch.float32)
        done = torch.as_tensor(np.array(terms), device=self.device, dtype=torch.bool)

        depth = depth.permute(0, 1, 4, 2, 3).float()
        sem = sem.permute(0, 1, 4, 2, 3)

        return (depth, sem, vec, goal, act, rew, done)

    def load_from_disk(self, path):
        if not os.path.exists(path):
            print(f"No expert data found at {path}")
            return

        files = [f for f in os.listdir(path) if f.endswith(".npz")]

        for file in tqdm(files, desc="Loading Expert Data", unit="file"):
            data = np.load(os.path.join(path, file))
            T = len(data["action"])
            for i in range(T):
                vec = data["vector"][i] if "vector" in data else np.zeros(3, dtype=np.float32)
                goal = data["goal"][i] if "goal" in data else np.zeros(2, dtype=np.float32)

                done = bool(data["done"][i]) if "done" in data else (i == T - 1)   # enforce boundary at end of each sequence file
                self.add(
                    data["depth"][i],
                    data["semantic"][i],
                    vec,
                    goal,
                    data["action"][i],
                    data["reward"][i],
                    done,
                )
