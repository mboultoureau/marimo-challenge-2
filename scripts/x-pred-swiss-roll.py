# %% Imports
from dataclasses import dataclass
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_swiss_roll


# %% Config
SEED = 67
N_SAMPLES = 10_000
DATASET_NOISE = 0.5
D_VALUES = [2]

HIDDEN = 256
N_LAYERS = 5
TIME_EMBED_DIM = 256
TIME_MAX_PERIOD = 10_000
TRAIN_STEPS = 5_000
BATCH_SIZE = 256
LR = 1e-3

T_MEAN = -0.8
T_STD = 0.8
T_EPS = 0.05
NOISE_SCALE = 1.0

SAMPLER = "heun"
SAMPLING_STEPS = 50
NUM_GENERATED = 2_000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Using device: {DEVICE}")


# %%
def generate_swiss_roll_2d(n_samples=10_000, noise=0.5, seed=42):
    x_3d, _ = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=seed)
    data = x_3d[:, [0, 2]]
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return data.astype(np.float32)


def make_projection(D, d=2, seed=0):
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((D, d))
    q, _ = np.linalg.qr(matrix)
    return q[:, :d].astype(np.float32)


def project_to_ambient(data_2d, projection):
    return data_2d @ projection.T


def project_back_to_2d(data_D, projection):
    return data_D @ projection


class DenoisingMLP(nn.Module):
    def __init__(
        self,
        dim,
        hidden=256,
        n_layers=5,
        time_embed_dim=TIME_EMBED_DIM,
        time_max_period=TIME_MAX_PERIOD,
    ):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.time_max_period = time_max_period
        self.input_proj = nn.Linear(dim, hidden)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        hidden_layers = []
        for _ in range(n_layers - 2):
            hidden_layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        self.hidden_net = nn.Sequential(*hidden_layers)
        self.output_proj = nn.Linear(hidden, dim)

    @staticmethod
    def timestep_embedding(t, dim, max_period=10_000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / max(half, 1)
        )
        args = t.float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, z_t, t):
        t_freq = self.timestep_embedding(t, self.time_embed_dim, self.time_max_period)
        t_emb = self.time_mlp(t_freq)

        hidden = self.input_proj(z_t) + t_emb
        hidden = torch.relu(hidden)
        hidden = self.hidden_net(hidden)
        return self.output_proj(hidden)


@dataclass
class TrainResult:
    model: nn.Module
    losses: list[float]
    projection: np.ndarray


def sample_t(batch_size, device, mean=T_MEAN, std=T_STD):
    logits = torch.randn(batch_size, 1, device=device) * std + mean
    return torch.sigmoid(logits)


def train_xpred_model(
    D,
    data_2d,
    projection,
    device,
    hidden=HIDDEN,
    n_layers=N_LAYERS,
    time_embed_dim=TIME_EMBED_DIM,
    time_max_period=TIME_MAX_PERIOD,
    n_steps=TRAIN_STEPS,
    batch_size=BATCH_SIZE,
    lr=LR,
    t_mean=T_MEAN,
    t_std=T_STD,
    t_eps=T_EPS,
    noise_scale=NOISE_SCALE,
):
    projection_t = torch.from_numpy(projection).to(device)
    x_all = torch.from_numpy(data_2d).to(device) @ projection_t.T

    model = DenoisingMLP(
        D,
        hidden=hidden,
        n_layers=n_layers,
        time_embed_dim=time_embed_dim,
        time_max_period=time_max_period,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for step in range(n_steps):
        idx = torch.randint(0, x_all.shape[0], (batch_size,), device=device)
        x = x_all[idx]
        eps = torch.randn_like(x) * noise_scale
        t = sample_t(batch_size, device, mean=t_mean, std=t_std)

        z_t = t * x + (1 - t) * eps
        denom = (1 - t).clamp(min=t_eps)
        v_true = (x - z_t) / denom

        x_pred = model(z_t, t)
        v_pred = (x_pred - z_t) / denom

        loss = ((v_pred - v_true) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            losses.append(float(loss.item()))

    model.eval()
    return TrainResult(model=model, losses=losses, projection=projection)


@torch.no_grad()
def generate_samples_xpred(
    model,
    D,
    n_samples=NUM_GENERATED,
    n_steps=SAMPLING_STEPS,
    solver=SAMPLER,
    t_eps=T_EPS,
    noise_scale=NOISE_SCALE,
):
    device = next(model.parameters()).device
    z = torch.randn(n_samples, D, device=device) * noise_scale

    def velocity(pred, z_t, t):
        return (pred - z_t) / (1 - t).clamp(min=t_eps)

    def euler_step(z_t, t, t_next):
        pred = model(z_t, t)
        v = velocity(pred, z_t, t)
        return z_t + (t_next - t) * v

    def heun_step(z_t, t, t_next):
        pred_t = model(z_t, t)
        v_t = velocity(pred_t, z_t, t)

        z_euler = z_t + (t_next - t) * v_t
        pred_next = model(z_euler, t_next)
        v_next = velocity(pred_next, z_euler, t_next)

        return z_t + (t_next - t) * 0.5 * (v_t + v_next)

    if solver not in {"euler", "heun"}:
        raise ValueError(f"Unsupported solver: {solver}")

    if solver == "euler":
        for i in range(n_steps):
            t = torch.full((n_samples, 1), i / n_steps, device=device)
            t_next = torch.full((n_samples, 1), (i + 1) / n_steps, device=device)
            z = euler_step(z, t, t_next)
    else:
        for i in range(n_steps - 1):
            t = torch.full((n_samples, 1), i / n_steps, device=device)
            t_next = torch.full((n_samples, 1), (i + 1) / n_steps, device=device)
            z = heun_step(z, t, t_next)

        # Match the JiT implementation: last step uses Euler.
        t = torch.full((n_samples, 1), (n_steps - 1) / n_steps, device=device)
        t_next = torch.full((n_samples, 1), 1.0, device=device)
        z = euler_step(z, t, t_next)

    return z.cpu().numpy()


@torch.no_grad()
def denoise_snapshot(model, data_D, t_value, noise_scale=NOISE_SCALE):
    device = next(model.parameters()).device
    x = torch.from_numpy(data_D).to(device)
    eps = torch.randn_like(x) * noise_scale
    t = torch.full((x.shape[0], 1), t_value, device=device)
    z_t = t * x + (1 - t) * eps
    x_pred = model(z_t, t)
    return x.cpu().numpy(), z_t.cpu().numpy(), x_pred.cpu().numpy()


# %% Run experiment
torch.manual_seed(SEED)
np.random.seed(SEED)

data_2d = generate_swiss_roll_2d(N_SAMPLES, noise=DATASET_NOISE, seed=SEED)

results = {}
generated_samples = {}

for D in D_VALUES:
    print(f"Training x-pred Swiss roll with D={D}")
    projection = make_projection(D, d=2, seed=D)
    result = train_xpred_model(
        D,
        data_2d,
        projection,
        DEVICE,
        hidden=HIDDEN,
        n_layers=N_LAYERS,
        time_embed_dim=TIME_EMBED_DIM,
        time_max_period=TIME_MAX_PERIOD,
        n_steps=TRAIN_STEPS,
        batch_size=BATCH_SIZE,
        lr=LR,
        t_mean=T_MEAN,
        t_std=T_STD,
        t_eps=T_EPS,
        noise_scale=NOISE_SCALE,
    )
    results[D] = result

    samples_D = generate_samples_xpred(
        result.model,
        D,
        n_samples=NUM_GENERATED,
        n_steps=SAMPLING_STEPS,
        solver=SAMPLER,
        t_eps=T_EPS,
        noise_scale=NOISE_SCALE,
    )
    generated_samples[D] = project_back_to_2d(samples_D, result.projection)

for D in D_VALUES:
    final_loss = results[D].losses[-1]
    print(f"D={D}: final logged loss={final_loss:.6f}")

fig, axes = plt.subplots(len(D_VALUES), 3, figsize=(12, 4 * len(D_VALUES)))
if len(D_VALUES) == 1:
    axes = np.array([axes])

gt_subset = data_2d[:NUM_GENERATED]
lim = max(np.abs(gt_subset).max() * 1.3, 3.0)

for row, D in enumerate(D_VALUES):
    projection = results[D].projection
    data_D = project_to_ambient(gt_subset, projection)
    x_clean, z_noisy, x_pred = denoise_snapshot(
        results[D].model,
        data_D,
        t_value=0.5,
        noise_scale=NOISE_SCALE,
    )

    axes[row, 0].scatter(gt_subset[:, 0], gt_subset[:, 1], s=1, alpha=0.5, c="black")
    axes[row, 0].set_title("Ground truth" if row == 0 else "")
    axes[row, 0].set_ylabel(f"D = {D}", fontsize=14, fontweight="bold")

    samples_2d = np.clip(generated_samples[D], -lim * 2, lim * 2)
    axes[row, 1].scatter(
        samples_2d[:, 0], samples_2d[:, 1], s=1, alpha=0.5, c="tab:blue"
    )
    axes[row, 1].set_title("Generated samples" if row == 0 else "")

    x_pred_2d = project_back_to_2d(x_pred, projection)
    axes[row, 2].scatter(
        x_pred_2d[:, 0], x_pred_2d[:, 1], s=1, alpha=0.5, c="tab:green"
    )
    axes[row, 2].set_title("One-step denoise (t=0.5)" if row == 0 else "")

    for col in range(3):
        axes[row, col].set_xlim(-lim, lim)
        axes[row, col].set_ylim(-lim, lim)
        axes[row, col].set_aspect("equal")
        axes[row, col].tick_params(labelbottom=False, labelleft=False)

fig.suptitle("x-pred Swiss Roll Debugger", fontsize=16, fontweight="bold", y=1.01)
fig.tight_layout()
plt.show()


fig_loss, axes_loss = plt.subplots(1, len(D_VALUES), figsize=(4 * len(D_VALUES), 4))
if len(D_VALUES) == 1:
    axes_loss = [axes_loss]

for i, D in enumerate(D_VALUES):
    axes_loss[i].plot(results[D].losses, color="tab:green")
    axes_loss[i].set_title(f"D = {D}")
    axes_loss[i].set_xlabel("Step (x50)")
    axes_loss[i].set_yscale("log")
    if i == 0:
        axes_loss[i].set_ylabel("v-loss")

fig_loss.suptitle("Training loss curves", fontsize=14, fontweight="bold")
fig_loss.tight_layout()
plt.show()
