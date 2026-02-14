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
        # Samples continuous sequences of length SEQ_LEN
        high = self.capacity - self.seq_len if self.full else self.idx - self.seq_len
        # Safety check: if we don't have enough data yet
        if high <= 0:
            return None 
            
        start_indices = np.random.randint(0, high, size=batch_size)
        
        # 1. Initialize all lists (added veccs here)
        obs_depth, obs_sem, veccs, goals, acts, rews, terms = [], [], [], [], [], [], []
        
        for start in start_indices:
            end = start + self.seq_len
            obs_depth.append(self.depths[start:end])
            obs_sem.append(self.semantics[start:end])
            veccs.append(self.vectors[start:end]) # <--- 2. Pull vector data
            goals.append(self.goals[start:end])
            acts.append(self.actions[start:end])
            rews.append(self.rewards[start:end])
            terms.append(self.dones[start:end])

        # Move to GPU and convert to float32 for training
        to_torch = lambda x: torch.FloatTensor(np.array(x)).to(self.device)
        
        # 3. Return exactly 6 values to match: 
        # depths, sems, vectors, actions, rewards, _ = buffer.sample(BATCH_SIZE)
        return (
            to_torch(obs_depth).permute(0, 1, 4, 2, 3) / 255.0, 
            to_torch(obs_sem).permute(0, 1, 4, 2, 3) / 255.0, 
            to_torch(veccs),
            to_torch(goals),
            to_torch(acts), 
            to_torch(rews), 
            to_torch(terms)
        )

    def load_from_disk(self, path):
        if not os.path.exists(path):
            print(f"No expert data found at {path}")
            return
        
        files = [f for f in os.listdir(path) if f.endswith(".npz")]
    
        for file in tqdm(files, desc="Loading Expert Data", unit="file"):
            data = np.load(os.path.join(path, file))
            for i in range(len(data['action'])):
                # Handle cases where 'vector' might be missing in old expert data
                vec = data['vector'][i] if 'vector' in data else np.zeros(3)
                goal = data['goal'][i] if 'goal' in data else np.zeros(2)
                self.add(data['depth'][i], 
                            data['semantic'][i], 
                            vec, # Correctly pass 6 arguments
                            goal,
                            data['action'][i], 
                            data['reward'][i], 
                            False)
        print(f"Loaded expert data into buffer. Current size: {self.idx if not self.full else self.capacity}")