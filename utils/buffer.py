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
        if self.idx == 0: self.full = True

    def sample(self, batch_size):
        # 1. Determine the search range based on whether the buffer is full
        max_idx = self.capacity if self.full else self.idx
        
        # We need at least seq_len steps to create one sequence
        if max_idx < self.seq_len:
            return None

        obs_depth, obs_sem, veccs, goals, acts, rews, terms = [], [], [], [], [], [], []
        
        attempts = 0
        while len(obs_depth) < batch_size and attempts < 1000: # Increase search budget
            attempts += 1
            # 2. Pick a random start index that leaves room for a full sequence
            start = np.random.randint(0, max_idx - self.seq_len)
            end = start + self.seq_len
            
            # 3. Validity Check: Does this window cross an episode boundary?
            # We check if any 'done' occurs in the first T-1 steps of the sequence.
            # If a 'done' is at the very last index, the sequence is still valid.
            if np.any(self.dones[start : end - 1]):
                continue
            
            # 4. Collect valid sequence data
            obs_depth.append(self.depths[start:end])
            obs_sem.append(self.semantics[start:end])
            veccs.append(self.vectors[start:end])
            goals.append(self.goals[start:end])
            acts.append(self.actions[start:end])
            rews.append(self.rewards[start:end])
            terms.append(self.dones[start:end])

        # If we couldn't find enough valid sequences, return None
        if len(obs_depth) < batch_size:
            return None

        # 5. Convert to Tensors
        to_torch = lambda x: torch.as_tensor(np.array(x), device=self.device).float()

        return (
            to_torch(obs_depth).permute(0, 1, 4, 2, 3), # [B, T, C, H, W]
            to_torch(obs_sem).permute(0, 1, 4, 2, 3),   # [B, T, C, H, W]
            to_torch(veccs),
            to_torch(goals),
            to_torch(acts),
            to_torch(rews),
            to_torch(terms),
        )


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

                done = (i == T - 1)   # enforce boundary at end of each sequence file
                self.add(
                    data["depth"][i],
                    data["semantic"][i],
                    vec,
                    goal,
                    data["action"][i],
                    data["reward"][i],
                    done
                )