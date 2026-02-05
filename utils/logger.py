import numpy as np
import torch

class SequenceBuffer:
    def __init__(self, capacity, seq_len, obs_shape, action_dim, device):
        self.capacity = capacity
        self.seq_len = seq_len
        self.device = device
        
        # Pre-allocate memory for speed
        self.depth_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.sem_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.terminals = np.zeros((capacity,), dtype=np.bool_)
        
        self.idx = 0
        self.full = False

    def add(self, depth, sem, action, reward, done):
        # Circular buffer logic
        self.depth_obs[self.idx] = depth
        self.sem_obs[self.idx] = sem
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.terminals[self.idx] = done
        
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        # DreamerV3 samples overlapping sequences of length L
        n = self.capacity if self.full else self.idx
        indices = np.random.randint(0, n - self.seq_len, size=batch_size)
        
        # Assemble sequences
        depth_seq, sem_seq, act_seq, rew_seq, term_seq = [], [], [], [], []
        
        for i in indices:
            depth_seq.append(self.depth_obs[i:i + self.seq_len])
            sem_seq.append(self.sem_obs[i:i + self.seq_len])
            act_seq.append(self.actions[i:i + self.seq_len])
            rew_seq.append(self.rewards[i:i + self.seq_len])
            term_seq.append(self.terminals[i:i + self.seq_len])

        return (
            torch.as_tensor(np.array(depth_seq), device=self.device).float() / 255.0,
            torch.as_tensor(np.array(sem_seq), device=self.device).float() / 255.0,
            torch.as_tensor(np.array(act_seq), device=self.device),
            torch.as_tensor(np.array(rew_seq), device=self.device),
            torch.as_tensor(np.array(term_seq), device=self.device)
        )