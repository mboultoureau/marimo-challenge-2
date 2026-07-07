# %% Imports
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_swiss_roll

is_notebook = False
try:
    is_notebook = hasattr(get_ipython(), "kernel")
except NameError:
    is_notebook = False

if is_notebook:
    from tqdm.notebook import tqdm, trange
else:
    from tqdm.auto import tqdm, trange


# %% Config
SEED = 67
N_SAMPLES = 10000
DATASET_NOISE = 0.2
D_VALUES = [2, 3, 4, 8]

HIDDEN = 256
N_LAYERS = 5
TRAIN_STEPS = 5000
BATCH_SIZE = 256
LR = 1e-3

T_MEAN = -0.8
T_STD = 0.8
NOISE_SCALE = 0.25
STANDARDIZE_DATA = True

SAMPLER = "euler"
SAMPLING_STEPS = 50
NUM_GENERATED = 2_000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Using device: {DEVICE}")


# %%
def generate_swiss_roll_2d(n_samples=10_000, noise=0.5, seed=42, standardize=True):
    x_3d, _ = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=seed)
    data = x_3d[:, [0, 2]]
    if standardize:
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
    def __init__(self, dim, hidden=256, n_layers=5):
        super().__init__()
        layers = [nn.Linear(dim + 1, hidden), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        layers.append(nn.Linear(hidden, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_t, t):
        return self.net(torch.cat([z_t, t], dim=-1))


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
    n_steps=TRAIN_STEPS,
    batch_size=BATCH_SIZE,
    lr=LR,
    t_mean=T_MEAN,
    t_std=T_STD,
    noise_scale=NOISE_SCALE,
    progress_desc=None,
):
    projection_t = torch.from_numpy(projection).to(device)
    x_all = torch.from_numpy(data_2d).to(device) @ projection_t.T

    model = DenoisingMLP(D, hidden=hidden, n_layers=n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    step_iter = trange(
        n_steps,
        desc=progress_desc or f"Train D={D}",
        leave=False,
        dynamic_ncols=True,
    )
    for step in step_iter:
        idx = torch.randint(0, x_all.shape[0], (batch_size,), device=device)
        x = x_all[idx]
        eps = torch.randn_like(x) * noise_scale
        t = sample_t(batch_size, device, mean=t_mean, std=t_std)

        z_t = t * x + (1 - t) * eps
        v_true = x - eps

        x_pred = model(z_t, t)
        v_pred = (x_pred - z_t) / (1 - t)

        loss = ((v_pred - v_true) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            loss_value = float(loss.item())
            losses.append(loss_value)
            step_iter.set_postfix(loss=f"{loss_value:.4f}")

    model.eval()
    return TrainResult(model=model, losses=losses, projection=projection)


@torch.no_grad()
def generate_samples_xpred(
    model,
    D,
    n_samples=NUM_GENERATED,
    n_steps=SAMPLING_STEPS,
    solver=SAMPLER,
    noise_scale=NOISE_SCALE,
):
    device = next(model.parameters()).device
    z = torch.randn(n_samples, D, device=device) * noise_scale

    def velocity(pred, z_t, t):
        return (pred - z_t) / (1 - t)

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


# %% Corruption process preview data
preview_data_2d = generate_swiss_roll_2d(
    N_SAMPLES, noise=DATASET_NOISE, seed=SEED, standardize=STANDARDIZE_DATA
)
preview_D = D_VALUES[0]
preview_projection = make_projection(preview_D, d=2, seed=preview_D)
preview_count = N_SAMPLES
preview_clean_2d = preview_data_2d[:preview_count]
preview_clean_D = project_to_ambient(preview_clean_2d, preview_projection)

rng = np.random.default_rng(SEED)
preview_eps_D = (
    rng.standard_normal(preview_clean_D.shape).astype(np.float32) * NOISE_SCALE
)
preview_t_values = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
preview_corrupted_2d = {}

for t_value in preview_t_values:
    z_t = t_value * preview_clean_D + (1 - t_value) * preview_eps_D
    preview_corrupted_2d[t_value] = project_back_to_2d(z_t, preview_projection)


fig_corrupt, axes_corrupt = plt.subplots(2, 3, figsize=(12, 8))
preview_lim = max(np.abs(preview_clean_2d).max() * 1.3, 3.0)

for ax, t_value in zip(axes_corrupt.flat, preview_t_values[::-1]):
    points_2d = np.clip(
        preview_corrupted_2d[t_value], -preview_lim * 2, preview_lim * 2
    )
    ax.scatter(points_2d[:, 0], points_2d[:, 1], s=2, alpha=0.5, c="tab:purple")
    ax.set_title(f"z_t at t={t_value:.2f}")
    ax.set_xlim(-preview_lim, preview_lim)
    ax.set_ylim(-preview_lim, preview_lim)
    ax.set_aspect("equal")
    # ax.tick_params(labelbottom=False, labelleft=False)

fig_corrupt.suptitle(
    f"Corruption Process in 2D View (D={preview_D}, noise_scale={NOISE_SCALE})",
    fontsize=16,
    fontweight="bold",
    y=1.01,
)
fig_corrupt.tight_layout()
plt.show()


# %% Run experiment
torch.manual_seed(SEED)
np.random.seed(SEED)

data_2d = generate_swiss_roll_2d(
    N_SAMPLES, noise=DATASET_NOISE, seed=SEED, standardize=STANDARDIZE_DATA
)

results = {}
generated_samples = {}
one_step_noisy = {}
one_step_denoised = {}

gt_preview_count = min(N_SAMPLES, max(NUM_GENERATED, 5_000))
gt_subset = data_2d[: min(NUM_GENERATED, N_SAMPLES)]
lim = max(np.abs(gt_subset).max() * 1.3, 3.0)

fig_gt, ax_gt = plt.subplots(figsize=(3, 3))
ax_gt.scatter(
    data_2d[:gt_preview_count, 0],
    data_2d[:gt_preview_count, 1],
    s=2,
    alpha=0.5,
    c="black",
)
ax_gt.set_xlim(-lim, lim)
ax_gt.set_ylim(-lim, lim)
ax_gt.set_aspect("equal")
ax_gt.set_title("Ground Truth Swiss Roll")
# ax_gt.tick_params(labelbottom=False, labelleft=False)
fig_gt.tight_layout()
plt.show()

for D in tqdm(D_VALUES, desc="Sweep dimensions", dynamic_ncols=True):
    projection = make_projection(D, d=2, seed=D)
    result = train_xpred_model(
        D,
        data_2d,
        projection,
        DEVICE,
        hidden=HIDDEN,
        n_layers=N_LAYERS,
        n_steps=TRAIN_STEPS,
        batch_size=BATCH_SIZE,
        lr=LR,
        t_mean=T_MEAN,
        t_std=T_STD,
        noise_scale=NOISE_SCALE,
        progress_desc=f"Train D={D}",
    )
    results[D] = result

    samples_D = generate_samples_xpred(
        result.model,
        D,
        n_samples=NUM_GENERATED,
        n_steps=SAMPLING_STEPS,
        solver=SAMPLER,
        noise_scale=NOISE_SCALE,
    )
    generated_samples[D] = project_back_to_2d(samples_D, result.projection)

    data_D = project_to_ambient(gt_subset, projection)
    _x_clean, z_noisy, x_pred = denoise_snapshot(
        result.model,
        data_D,
        t_value=0.5,
        noise_scale=NOISE_SCALE,
    )
    one_step_noisy[D] = project_back_to_2d(z_noisy, projection)
    one_step_denoised[D] = project_back_to_2d(x_pred, projection)

fig_denoise, axes_denoise = plt.subplots(
    2, len(D_VALUES), figsize=(3 * len(D_VALUES), 6)
)
if len(D_VALUES) == 1:
    axes_denoise = np.array(axes_denoise).reshape(2, 1)

for col, D in enumerate(D_VALUES):
    noisy_2d = np.clip(one_step_noisy[D], -lim * 2, lim * 2)
    denoised_2d = np.clip(one_step_denoised[D], -lim * 2, lim * 2)

    axes_denoise[0, col].scatter(
        noisy_2d[:, 0], noisy_2d[:, 1], s=1, alpha=0.5, c="tab:orange"
    )
    axes_denoise[0, col].set_title(f"Noisy input (D={D})")

    axes_denoise[1, col].scatter(
        denoised_2d[:, 0], denoised_2d[:, 1], s=1, alpha=0.5, c="tab:green"
    )
    axes_denoise[1, col].set_title(f"Denoised output (D={D})")

    for row in range(2):
        axes_denoise[row, col].set_xlim(-lim, lim)
        axes_denoise[row, col].set_ylim(-lim, lim)
        axes_denoise[row, col].set_aspect("equal")
        # axes_denoise[row, col].tick_params(labelbottom=False, labelleft=False)

fig_denoise.suptitle(
    "One-Step Denoise at t=0.5", fontsize=16, fontweight="bold", y=1.01
)
fig_denoise.tight_layout()
plt.show()

fig_samples, axes_samples = plt.subplots(
    1, len(D_VALUES), figsize=(3 * len(D_VALUES), 3)
)
if len(D_VALUES) == 1:
    axes_samples = [axes_samples]

for i, D in enumerate(D_VALUES):
    samples_2d = np.clip(generated_samples[D], -lim * 2, lim * 2)
    axes_samples[i].scatter(
        samples_2d[:, 0], samples_2d[:, 1], s=1, alpha=0.5, c="tab:blue"
    )
    axes_samples[i].set_title(f"Generated samples (D={D})")
    axes_samples[i].set_xlim(-lim, lim)
    axes_samples[i].set_ylim(-lim, lim)
    axes_samples[i].set_aspect("equal")
    # axes_samples[i].tick_params(labelbottom=False, labelleft=False)

fig_samples.suptitle("Generated Samples", fontsize=16, fontweight="bold", y=1.01)
fig_samples.tight_layout()
plt.show()

fig_loss, axes_loss = plt.subplots(1, len(D_VALUES), figsize=(3 * len(D_VALUES), 3))
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

for D in D_VALUES:
    final_loss = results[D].losses[-1]
    print(f"D={D}: final logged loss={final_loss:.6f}")


# %% Noise scale sweep
STANDARDIZE = True

sweep_noise_scales = np.arange(1.0, 0.0, -0.1)
sweep_t_values = np.arange(1.0, -0.25, -0.25)
sweep_data_2d = generate_swiss_roll_2d(
    N_SAMPLES, noise=DATASET_NOISE, seed=SEED, standardize=STANDARDIZE
)
sweep_D = D_VALUES[0]
sweep_projection = make_projection(sweep_D, d=2, seed=sweep_D)
sweep_clean_D = project_to_ambient(sweep_data_2d, sweep_projection)
sweep_lim = max(np.abs(sweep_data_2d).max() * 1.3, 3.0)

fig_sweep, axes_sweep = plt.subplots(
    len(sweep_noise_scales),
    len(sweep_t_values),
    figsize=(3 * len(sweep_t_values), 3 * len(sweep_noise_scales)),
)

rng_sweep = np.random.default_rng(SEED)
sweep_eps_unit = rng_sweep.standard_normal(sweep_clean_D.shape).astype(np.float32)

for row, ns in enumerate(sweep_noise_scales):
    eps_D = sweep_eps_unit * ns
    for col, t_val in enumerate(sweep_t_values):
        z_t = t_val * sweep_clean_D + (1 - t_val) * eps_D
        pts = np.clip(
            project_back_to_2d(z_t, sweep_projection), -sweep_lim * 2, sweep_lim * 2
        )
        axes_sweep[row, col].scatter(
            pts[:, 0], pts[:, 1], s=2, alpha=0.5, c="tab:purple"
        )
        axes_sweep[row, col].set_xlim(-sweep_lim, sweep_lim)
        axes_sweep[row, col].set_ylim(-sweep_lim, sweep_lim)
        axes_sweep[row, col].set_aspect("equal")
        # axes_sweep[row, col].tick_params(labelbottom=False, labelleft=False)
        if row == 0:
            axes_sweep[row, col].set_title(f"t={t_val:.2f}")
        if col == 0:
            axes_sweep[row, col].set_ylabel(f"σ={ns:.2f}")

fig_sweep.suptitle(
    f"Noise Scale Sweep – Corruption Process (D={sweep_D})",
    fontsize=16,
    fontweight="bold",
    y=1.01,
)
fig_sweep.tight_layout()
plt.show()
