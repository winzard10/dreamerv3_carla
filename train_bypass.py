import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.buffer import SequenceBuffer
from models.encoder import MultiModalEncoder
from models.decoder import MultiModalDecoder

# -----------------------
# Config
# -----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN = 10
BATCH_SIZE = 8
NUM_CLASSES = 28
H, W = 128, 128

# Match your current no-skip model sizes
EMBED_DIM = 1024
DETER_DIM = 512
STOCH_CATEGORICALS = 32
STOCH_CLASSES = 32
STOCH_DIM = STOCH_CATEGORICALS * STOCH_CLASSES

PHASE_A_STEPS = 20000
LR = 8e-5
SEM_SCALE = 10.0

BUFFER_PATH = "./data/expert_sequences"
RUN_DIR = "./runs/bypass_recon"
CKPT_DIR = "checkpoints/bypass_recon"
CKPT_PATH = os.path.join(CKPT_DIR, "bypass_recon_latest.pth")

# Optional warm start from your existing world model
LOAD_ENCODER_DECODER_FROM_WM = False
WM_CKPT_PATH = "checkpoints/world_model/world_model_pretrained.pth"

# -----------------------
# Utils
# -----------------------
def gaussian_nll(x, mean, std=0.1, eps=1e-6):
    var = (std ** 2) + eps
    return 0.5 * ((x - mean) ** 2) / var + torch.log(torch.tensor(std + eps, device=x.device))


def preprocess_batch(depths, sems, vectors, goals):
    """
    Inputs from SequenceBuffer.sample():
      depths:  [B,T,1,H,W]
      sems:    [B,T,1,H,W]
      vectors: [B,T,3]
      goals:   [B,T,2]
    Returns flattened tensors for pure reconstruction training.
    """
    assert depths.ndim == 5 and sems.ndim == 5
    B, T, C, H_, W_ = depths.shape
    assert H_ == H and W_ == W and C == 1

    depth_in = depths.reshape(B * T, 1, H, W).to(dtype=torch.float32) / 255.0
    sem_ids = sems.reshape(B * T, H, W).to(dtype=torch.long)
    sem_ids = torch.clamp(sem_ids, 0, NUM_CLASSES - 1)
    vec_in = vectors.reshape(B * T, -1).to(dtype=torch.float32)
    goal_in = goals.reshape(B * T, -1).to(dtype=torch.float32)
    return depth_in, sem_ids, vec_in, goal_in


class EncoderDecoderBypass(nn.Module):
    """
    Pure reconstruction diagnostic:
      encoder -> projection heads -> decoder
    No RSSM, no reward head, no continue head, no actor/critic.

    Purpose:
      Test whether the encoder embedding itself still contains enough
      structure for reconstruction, without RSSM in the middle.
    """
    def __init__(
        self,
        embed_dim=EMBED_DIM,
        deter_dim=DETER_DIM,
        stoch_dim=STOCH_DIM,
        num_classes=NUM_CLASSES,
        sem_embed_dim=16,
    ):
        super().__init__()

        self.encoder = MultiModalEncoder(
            embed_dim=embed_dim,
            num_classes=num_classes,
            sem_embed_dim=sem_embed_dim,
        )

        self.to_deter = nn.Sequential(
            nn.Linear(embed_dim, deter_dim),
            nn.ELU(),
            nn.Linear(deter_dim, deter_dim),
            nn.ELU(),
        )

        self.to_stoch = nn.Sequential(
            nn.Linear(embed_dim, stoch_dim),
            nn.ELU(),
            nn.Linear(stoch_dim, stoch_dim),
            nn.ELU(),
        )

        self.decoder = MultiModalDecoder(
            deter_dim=deter_dim,
            stoch_dim=stoch_dim,
            num_classes=num_classes,
        )

    def forward(self, depth, sem_ids, vector, goal):
        embed = self.encoder(depth, sem_ids, vector, goal)   # [B, EMBED_DIM]
        deter = self.to_deter(embed)                         # [B, DETER_DIM]
        stoch = self.to_stoch(embed)                         # [B, STOCH_DIM]
        recon_depth, sem_logits = self.decoder(deter, stoch)
        return recon_depth, sem_logits


def maybe_load_encoder_decoder(model: EncoderDecoderBypass):
    if not LOAD_ENCODER_DECODER_FROM_WM:
        print("[Info] Training bypass model from scratch.")
        return

    if not os.path.exists(WM_CKPT_PATH):
        print(f"[Warning] World model checkpoint not found: {WM_CKPT_PATH}")
        print("[Info] Training bypass model from scratch.")
        return

    ckpt = torch.load(WM_CKPT_PATH, map_location=DEVICE, weights_only=False)

    if "encoder" in ckpt:
        missing, unexpected = model.encoder.load_state_dict(ckpt["encoder"], strict=False)
        print(f"[Warm Start] Loaded encoder. Missing={missing}, Unexpected={unexpected}")
    else:
        print("[Warning] No encoder weights found in WM checkpoint.")

    if "decoder" in ckpt:
        missing, unexpected = model.decoder.load_state_dict(ckpt["decoder"], strict=False)
        print(f"[Warm Start] Loaded decoder. Missing={missing}, Unexpected={unexpected}")
    else:
        print("[Warning] No decoder weights found in WM checkpoint.")

    print("[Info] Projection heads (to_deter, to_stoch) remain randomly initialized.")


def main():
    print("Device:", DEVICE)

    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    buffer = SequenceBuffer(capacity=100000, seq_len=SEQ_LEN, device=DEVICE)
    buffer.load_from_disk(BUFFER_PATH)

    writer = SummaryWriter(log_dir=RUN_DIR)

    model = EncoderDecoderBypass().to(DEVICE)
    maybe_load_encoder_decoder(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    global_step = 0
    best_loss = float("inf")

    model.train()
    pbar = tqdm(range(PHASE_A_STEPS), desc="[Bypass Recon] Train")

    for step in pbar:
        batch = buffer.sample(BATCH_SIZE)
        if batch is None:
            continue

        depths, sems, vectors, goals, actions, rewards, dones = batch
        depth_in, sem_ids, vec_in, goal_in = preprocess_batch(depths, sems, vectors, goals)

        optimizer.zero_grad(set_to_none=True)

        recon_depth, sem_logits = model(depth_in, sem_ids.unsqueeze(1), vec_in, goal_in)

        depth_loss = F.mse_loss(recon_depth, depth_in)
        sem_loss = F.cross_entropy(sem_logits, sem_ids)
        depth_nll = gaussian_nll(depth_in, recon_depth, std=0.1).mean()

        loss = depth_nll + SEM_SCALE * sem_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
        optimizer.step()

        global_step += 1

        if global_step % 10 == 0:
            writer.add_scalar("Bypass/loss", loss.item(), global_step)
            writer.add_scalar("Bypass/depth_loss", depth_loss.item(), global_step)
            writer.add_scalar("Bypass/sem_loss", sem_loss.item(), global_step)
            writer.add_scalar("Bypass/depth_nll", depth_nll.item(), global_step)

        if global_step % 100 == 0:
            with torch.no_grad():
                # Depth GT | Recon
                t_depth = depth_in[0:1]
                r_depth = recon_depth[0:1]
                vis_depth = torch.cat([t_depth, r_depth], dim=-1)
                writer.add_image("Visuals/Depth_Recon", vis_depth.squeeze(0), global_step)

                # Semantic GT | Recon
                r_sem_ids = torch.argmax(sem_logits[0:1], dim=1)
                t_sem_ids = sem_ids[0:1]

                t_sem_vis = t_sem_ids.float() / float(NUM_CLASSES - 1)
                r_sem_vis = r_sem_ids.float() / float(NUM_CLASSES - 1)
                t_sem_vis = t_sem_vis.unsqueeze(0)
                r_sem_vis = r_sem_vis.unsqueeze(0)
                vis_sem = torch.cat([t_sem_vis, r_sem_vis], dim=-1)
                writer.add_image("Visuals/Semantic_Recon", vis_sem.squeeze(0), global_step)

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "global_step": global_step,
                    "best_loss": best_loss,
                },
                CKPT_PATH,
            )

        pbar.set_postfix({
            "loss": f"{loss.item():.3f}",
            "depth": f"{depth_loss.item():.4f}",
            "sem": f"{sem_loss.item():.4f}",
        })

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "best_loss": best_loss,
        },
        CKPT_PATH,
    )

    print(f"Done. Best loss: {best_loss:.4f}")
    print(f"Saved checkpoint to: {CKPT_PATH}")


if __name__ == "__main__":
    main()
