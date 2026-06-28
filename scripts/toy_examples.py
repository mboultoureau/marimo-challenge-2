# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.23.10",
#     "torch>=2.0.0",
#     "scikit-learn>=1.4.0",
#     "matplotlib>=3.8.0",
#     "numpy>=1.26.0",
# ]
# ///

import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium", app_title="Diffusion model toy examples")


@app.cell(hide_code=True)
def _():
    import copy

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_swiss_roll, make_moons, make_circles

    return (
        copy,
        make_circles,
        make_moons,
        make_swiss_roll,
        mo,
        nn,
        np,
        plt,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Back to Basics: Let Denoising Generative Models Denoise

    **Reproducing the Toy Experiment (Section 3.3, Figure 2)**
    from [Li & He, 2025](https://arxiv.org/abs/2511.13720)

    ---

    The **manifold hypothesis** says natural data lies on a low-dimensional manifold,
    while noise occupies the full ambient space. This experiment embeds 2D data into
    higher-dimensional spaces via a random orthogonal projection, then trains a small
    MLP to generate samples using three prediction strategies:

    - **$x$-prediction**: directly predict clean data (on-manifold)
    - **$\epsilon$-prediction**: predict the noise (off-manifold)
    - **$v$-prediction**: predict the flow velocity (off-manifold)

    As the ambient dimension D grows, only **$x$-prediction** survives because the
    network only needs to capture the low-dimensional manifold structure, not the
    full D-dimensional space.
    """)
    return


@app.cell(hide_code=True)
def _(D_values, mo):
    dataset_dropdown = mo.ui.dropdown(
        options={
            "Swiss Roll": "swiss_roll",
            "Two Moons": "two_moons",
            "Concentric Circles": "circles",
        },
        value="Swiss Roll",
        label="Dataset",
    )
    n_samples = mo.ui.number(
        start=1000, stop=50000, step=500, value=10000, label="Samples"
    )
    seed = mo.ui.number(start=0, stop=9999, step=1, value=67, label="Seed")
    circles_factor = mo.ui.slider(
        start=0.0, stop=0.99, step=0.01, value=0.5, label="Circles factor"
    )
    noise_d_dropdown = mo.ui.dropdown(
        options={str(value): value for value in D_values},
        value=str(D_values[0]),
        label="Ambient dimension",
    )
    noise_scale = mo.ui.slider(
        start=0.05,
        stop=1.0,
        step=0.05,
        value=0.3,
        label="Noise scale $\sigma$",
        show_value=True,
    )

    mo.vstack(
        [
            mo.md("## Choose dataset"),
            mo.hstack([dataset_dropdown, n_samples, seed]),
        ]
    )
    return (
        circles_factor,
        dataset_dropdown,
        n_samples,
        noise_d_dropdown,
        noise_scale,
        seed,
    )


@app.cell(hide_code=True)
def _(
    circles_factor,
    colors,
    data_preview,
    dataset_dropdown,
    dataset_noise,
    mo,
    noise_d_dropdown,
    noise_scale,
    noised_cache,
    np,
    plt,
    time_slider,
):
    _controls = [dataset_noise]
    _fig_size = (3, 3)
    if dataset_dropdown.value == "circles":
        _controls.append(circles_factor)

    _D = noise_d_dropdown.value
    _t = round(time_slider.value, 4)
    _z_t_2d = noised_cache[(_D, _t)]

    _lim_clean = np.abs(data_preview).max() * 1.2
    _lim_noisy = np.abs(_z_t_2d).max() * 1.2

    _fig_clean, _ax_clean = plt.subplots(figsize=_fig_size)
    _ax_clean.scatter(
        data_preview[:, 0], data_preview[:, 1], s=2, alpha=0.5, c=colors["ground_truth"]
    )
    _ax_clean.set_xlim(-_lim_clean, _lim_clean)
    _ax_clean.set_ylim(-_lim_clean, _lim_clean)
    _ax_clean.set_aspect("equal")
    _ax_clean.set_xticks([])
    _ax_clean.set_yticks([])
    _fig_clean.tight_layout()

    _fig_noisy, _ax_noisy = plt.subplots(figsize=_fig_size)
    _ax_noisy.scatter(_z_t_2d[:, 0], _z_t_2d[:, 1], s=2, alpha=0.5, c=colors["noising"])
    _ax_noisy.set_xlim(-_lim_noisy, _lim_noisy)
    _ax_noisy.set_ylim(-_lim_noisy, _lim_noisy)
    _ax_noisy.set_aspect("equal")
    _ax_noisy.set_xticks([])
    _ax_noisy.set_yticks([])
    _fig_noisy.tight_layout()

    mo.vstack(
        [
            mo.md(f"## {dataset_dropdown.selected_key} parametrization"),
            mo.vstack(_controls),
            mo.as_html(_fig_clean),
            mo.md(r"""
        ## Noising process

        The forward process is a linear interpolation using the flow-matching convention ($t=0$ is noise, $t=1$ is clean data):

        $z_t = t \cdot x + (1 - t) \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)$

        """),
            mo.vstack([noise_d_dropdown, noise_scale]),
            mo.as_html(_fig_noisy),
            mo.md(f"$z_t$ at $t = {_t:.2f}$"),
            time_slider,
        ],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(D_values, mo, pred_types):
    hidden_width = mo.ui.number(
        start=64, stop=1024, step=64, value=256, label="Hidden width", disabled=True
    )
    t_eps = mo.ui.number(
        start=0.01,
        stop=0.20,
        step=0.01,
        value=0.05,
        label="$t_\epsilon$",
        disabled=True,
    )
    solver = mo.ui.dropdown(
        options={"Euler": "euler", "Heun": "heun"},
        value="Euler",
        label="Sampler",
    )
    train_btn = mo.ui.run_button(label="Train all models")
    mo.md(
        f"""
        ## Run the experiment

        Trains {len(D_values) * len(pred_types)} models: {len(D_values)} ambient dimension{"" if len(D_values) < 2 else "s"} x {len(pred_types)} prediction types ($x$-prediction, $\epsilon$-prediction, $v$-prediction).

        {hidden_width}

        {t_eps}

        {solver}

        {train_btn}
        """
    )
    return hidden_width, solver, t_eps, train_btn


@app.cell(hide_code=True)
def _(
    D_values,
    data_preview,
    device,
    hidden_width,
    make_projection,
    mo,
    n_steps,
    noise_scale,
    pred_labels,
    t_eps,
    train_btn,
    train_model,
):
    mo.stop(not train_btn.value)

    data_2d = data_preview

    results = {}
    projections = {}

    with mo.status.progress_bar(
        total=len(D_values) * len(pred_labels),
        title=f"Training {len(D_values) * len(pred_labels)} models",
        subtitle=f"Dimension {D_values[0]}",
    ) as main_bar:
        for _D in D_values:
            _P = make_projection(_D, d=2, seed=_D)
            projections[_D] = _P

            for _pred_type, _pred_label in pred_labels.items():
                with mo.status.progress_bar(
                    total=n_steps,
                    title=_pred_type,
                    remove_on_exit=True,
                ) as step_bar:
                    _model, _losses = train_model(
                        _pred_type,
                        _D,
                        data_2d,
                        _P,
                        device,
                        n_steps=n_steps,
                        hidden=int(hidden_width.value),
                        t_eps=t_eps.value,
                        noise_scale=noise_scale.value,
                        step_bar=step_bar,
                    )
                results[(_D, _pred_type)] = {"model": _model, "losses": _losses}
                main_bar.update(increment=1, subtitle=f"Dimension {_D}")
    return projections, results


@app.cell(hide_code=True)
def _(D_values, colors, mo, plt, pred_labels, results):
    pred_colors = {k: colors[k] for k in ["x_pred", "eps_pred", "v_pred"]}

    fig_loss, axes_loss = plt.subplots(
        len(D_values), 1, figsize=(6, 3 * len(D_values)), squeeze=False
    )

    for _i, _D in enumerate(D_values):
        _ax = axes_loss[_i, 0]
        for _pred_type, _pred_label in pred_labels.items():
            _losses = results[(_D, _pred_type)]["losses"]
            _ax.plot(
                range(len(_losses)),
                _losses,
                label=_pred_label,
                color=pred_colors[_pred_type],
                alpha=0.8,
            )
        _ax.set_title(f"D = {_D}")
        _ax.set_yscale("log")
        _ax.set_ylabel("v-loss")
        _ax.legend(fontsize=8)
        if _i == len(D_values) - 1:
            _ax.set_xlabel("Step (x50)")

    fig_loss.tight_layout()
    mo.vstack(
        [
            mo.md("## Training Loss Curves"),
            mo.as_html(fig_loss),
        ],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(
    D_values,
    colors,
    data_preview,
    dataset_dropdown,
    gen_samples,
    mo,
    n_sample_steps,
    np,
    plt,
    pred_labels,
    time_slider,
    trajectories,
):
    _n = int(gen_samples.value)
    _t = round(time_slider.value, 4)
    _step = round(_t * n_sample_steps)

    fig, axes = plt.subplots(
        len(D_values), 4, figsize=(3 * 4, 3 * len(D_values)), squeeze=False
    )

    _rng = np.random.default_rng(0)
    _idx = _rng.choice(
        len(data_preview), size=min(_n, len(data_preview)), replace=False
    )
    gt_subset = data_preview[_idx]
    lim = max(np.abs(data_preview).max() * 1.3, 3.0)

    for _row, _D in enumerate(D_values):
        axes[_row, 0].scatter(
            gt_subset[:, 0], gt_subset[:, 1], s=1, alpha=0.5, c=colors["ground_truth"]
        )
        axes[_row, 0].set_title("Ground Truth" if _row == 0 else "")
        axes[_row, 0].set_ylabel(f"D = {_D}")
        axes[_row, 0].set_xlim(-lim, lim)
        axes[_row, 0].set_ylim(-lim, lim)
        axes[_row, 0].set_aspect("equal")
        axes[_row, 0].set_xticks([])
        axes[_row, 0].set_yticks([])

        for _col, (_pred_type, _pred_label) in enumerate(pred_labels.items()):
            _ax = axes[_row, _col + 1]
            _samples_2d = trajectories[(_D, _pred_type)][_step]
            _samples_clip = np.clip(_samples_2d, -lim * 2, lim * 2)
            _ax.scatter(
                _samples_clip[:, 0],
                _samples_clip[:, 1],
                s=1,
                alpha=0.5,
                c=colors[_pred_type],
            )
            _ax.set_title(_pred_label if _row == 0 else "")
            _ax.set_xlim(-lim, lim)
            _ax.set_ylim(-lim, lim)
            _ax.set_aspect("equal")
            _ax.set_xticks([])
            _ax.set_yticks([])

    fig.tight_layout()
    mo.vstack(
        [
            mo.md(
                f"## {dataset_dropdown.selected_key} reconstruction from {_n} points ($t={_t:.2f}$)"
            ),
            gen_samples,
            time_slider,
            mo.as_html(fig),
        ],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(
    colors,
    data_preview,
    mo,
    np,
    plt,
    pred_labels,
    time_slider,
    vf_cache,
    vf_grid_lim,
    vf_grid_res,
):
    _t = round(time_slider.value, 4)
    _res = int(vf_grid_res.value)
    _lim = vf_grid_lim

    _xs = np.linspace(-_lim, _lim, _res)
    _ys = np.linspace(-_lim, _lim, _res)
    _xx, _yy = np.meshgrid(_xs, _ys)

    _row_labels = {
        "x": r"$\hat{x}$",
        "eps": r"$\hat{\epsilon}$",
        "v": r"$\hat{v}$",
    }

    def _plot_vector_field(_ax, _vals):
        _u = _vals[:, 0].reshape(_res, _res)
        _vy = _vals[:, 1].reshape(_res, _res)
        _mag = np.sqrt(_u**2 + _vy**2)
        _mag_norm = _mag / _mag.max().clip(1e-8)
        _dir_u = _u / _mag.clip(1e-8)
        _dir_vy = _vy / _mag.clip(1e-8)
        _ax.quiver(
            _xx,
            _yy,
            _dir_u,
            _dir_vy,
            _mag_norm,
            cmap="viridis",
            alpha=0.8,
            zorder=2,
            scale=_res,
            clim=(0, 1),
        )

    def _setup_ax(_ax, _title=None, _ylabel=None):
        _ax.scatter(
            data_preview[:, 0],
            data_preview[:, 1],
            s=1,
            alpha=0.1,
            c=colors["ground_truth"],
            zorder=1,
        )
        if _title:
            _ax.set_title(_title)
        if _ylabel:
            _ax.set_ylabel(_ylabel, fontsize=16)
        _ax.set_xlim(-_lim, _lim)
        _ax.set_ylim(-_lim, _lim)
        _ax.set_aspect("equal")
        _ax.set_xticks([])
        _ax.set_yticks([])

    _fig_grid, _axes_grid = plt.subplots(3, 3, figsize=(18, 18))

    for _col, (_pred_type, _pred_label) in enumerate(pred_labels.items()):
        _all = vf_cache[(_t, _pred_type)]
        for _row, (_q_key, _q_label) in enumerate(_row_labels.items()):
            _ax = _axes_grid[_row, _col]
            _setup_ax(
                _ax,
                _title=_pred_label if _row == 0 else None,
                _ylabel=_q_label if _col == 0 else None,
            )
            if _q_key == "x":
                _ax.scatter(
                    _all["x"][:, 0],
                    _all["x"][:, 1],
                    s=2,
                    alpha=0.3,
                    c=colors[_pred_type],
                    zorder=2,
                )
            else:
                _plot_vector_field(_ax, _all[_q_key])

    _fig_grid.tight_layout()

    mo.vstack(
        [
            mo.md(f"## Prediction Space Grid at $t={_t:.2f}$ ($D=2$)"),
            mo.md(r"""
        Given the interpolation $z_t = t \cdot x + (1 - t) \cdot \epsilon$, each model directly predicts one quantity.
        The other two are recovered algebraically.

        | | From $\hat{x}$ | From $\hat{\epsilon}$ | From $\hat{v}$ |
        |---|---|---|---|
        | $\hat{x}$ | | $\frac{z_t - (1-t) \hat{\epsilon}}{t}$ | $(1-t) \hat{v} + z_t$ |
        | $\hat{\epsilon}$ | $\frac{z_t - t \hat{x}}{1-t}$ | | $z_t - t \hat{v}$ |
        | $\hat{v}$ | $\frac{\hat{x} - z_t}{1-t}$ | $\frac{z_t - \hat{\epsilon}}{t}$ | |

        The velocity field $v_\theta(z_t, t)$ drives the sampling ODE $dz_t/dt = v_\theta(z_t, t)$.

        Rows: derived quantity ($\hat{x}$, $\hat{\epsilon}$, $\hat{v}$). Columns: network prediction type.
        $\hat{x}$ shown as scatter points, $\hat{\epsilon}$ and $\hat{v}$ as unit vector fields colored by magnitude.
        """),
            mo.vstack(
                [
                    vf_grid_res,
                    time_slider,
                    mo.as_html(_fig_grid),
                ],
                align="center",
            ),
        ],
        align="start",
    )
    return


@app.cell(hide_code=True)
def _(data_preview, mo):
    gen_samples = mo.ui.slider(
        start=500,
        stop=len(data_preview),
        step=500,
        value=min(2000, len(data_preview)),
        label="Generated samples",
        show_value=True,
    )
    return (gen_samples,)


@app.cell(hide_code=True)
def _(
    data_preview,
    device,
    mo,
    n_sample_steps,
    np,
    pred_labels,
    results,
    t_eps,
    torch,
    vf_grid_res,
):
    _res = int(vf_grid_res.value)
    _t_eps = t_eps.value
    _t_values = np.linspace(0.0, 1.0, n_sample_steps + 1)

    _lim = max(np.abs(data_preview).max() * 1.5, 3.0)
    _xs = np.linspace(-_lim, _lim, _res)
    _ys = np.linspace(-_lim, _lim, _res)
    _grid = np.stack(np.meshgrid(_xs, _ys), axis=-1).reshape(-1, 2).astype(np.float32)

    vf_cache = {}
    vf_grid_lim = _lim

    with mo.status.progress_bar(
        total=len(_t_values) * len(pred_labels),
        title="Precomputing prediction grid",
        remove_on_exit=True,
    ) as _bar:
        with torch.no_grad():
            _z_t = torch.from_numpy(_grid).to(device)
            for _t_val in _t_values:
                _t_key = round(_t_val, 4)
                _t_tensor = torch.full((_grid.shape[0], 1), _t_val, device=device)
                for _pred_type in pred_labels:
                    _pred = results[(2, _pred_type)]["model"](_z_t, _t_tensor)
                    if _pred_type == "x_pred":
                        _x_hat = _pred
                        _eps_hat = (_z_t - _t_tensor * _x_hat) / (1 - _t_tensor).clamp(
                            min=_t_eps
                        )
                        _v_hat = (_x_hat - _z_t) / (1 - _t_tensor).clamp(min=_t_eps)
                    elif _pred_type == "eps_pred":
                        _eps_hat = _pred
                        _x_hat = (_z_t - (1 - _t_tensor) * _eps_hat) / _t_tensor.clamp(
                            min=_t_eps
                        )
                        _v_hat = (_z_t - _eps_hat) / _t_tensor.clamp(min=_t_eps)
                    else:
                        _v_hat = _pred
                        _x_hat = (1 - _t_tensor) * _v_hat + _z_t
                        _eps_hat = _z_t - _t_tensor * _v_hat
                    vf_cache[(_t_key, _pred_type)] = {
                        "x": _x_hat.cpu().numpy(),
                        "eps": _eps_hat.cpu().numpy(),
                        "v": _v_hat.cpu().numpy(),
                    }
                    _bar.update(increment=1)
    return vf_cache, vf_grid_lim


@app.cell(hide_code=True)
def _(
    D_values,
    gen_samples,
    generate_samples,
    n_sample_steps,
    noise_scale,
    pred_labels,
    projections,
    results,
    solver,
    t_eps,
):
    _n = int(gen_samples.value)
    trajectories = {}
    for _D in D_values:
        _P = projections[_D]
        for _pred_type in pred_labels:
            _traj_D = generate_samples(
                results[(_D, _pred_type)]["model"],
                _pred_type,
                _D,
                n_samples=_n,
                n_steps=n_sample_steps,
                solver=solver.value,
                t_eps=t_eps.value,
                noise_scale=noise_scale.value,
                return_trajectory=True,
            )
            trajectories[(_D, _pred_type)] = [_snap @ _P for _snap in _traj_D]
    return (trajectories,)


@app.cell(hide_code=True)
def _(mo, n_sample_steps):
    time_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=round(1.0 / n_sample_steps, 4),
        value=1.0,
        label="Timestep $t$",
        show_value=True,
    )
    vf_grid_res = mo.ui.slider(
        start=10,
        stop=40,
        step=5,
        value=20,
        label="Grid resolution",
        show_value=True,
    )
    return time_slider, vf_grid_res


@app.cell(hide_code=True)
def _(dataset_dropdown, mo):
    _noise_configs = {
        "swiss_roll": (0.0, 4.0, 0.01, 0.2),
        "two_moons": (0.0, 0.5, 0.005, 0.1),
        "circles": (0.0, 0.2, 0.001, 0.005),
    }
    _start, _stop, _step, _value = _noise_configs.get(
        dataset_dropdown.value, (0.0, 4.0, 0.01, 0.2)
    )
    dataset_noise = mo.ui.slider(
        start=_start, stop=_stop, step=_step, value=_value, label="Dataset noise"
    )
    return (dataset_noise,)


@app.cell(hide_code=True)
def _(
    D_values,
    circles_factor,
    dataset_dropdown,
    dataset_noise,
    generate_2d_data,
    make_projection,
    n_sample_steps,
    n_samples,
    noise_scale,
    np,
    seed,
):
    data_preview = generate_2d_data(
        dataset_dropdown.value,
        n_samples=int(n_samples.value),
        seed=int(seed.value),
        noise=dataset_noise.value,
        circles_factor=circles_factor.value,
    )

    _t_values = np.linspace(0.0, 1.0, n_sample_steps + 1)
    _ns = noise_scale.value

    noised_cache = {}
    for _D in D_values:
        _P = make_projection(_D, d=2, seed=_D)
        _data_D = data_preview @ _P.T
        _rng = np.random.default_rng(0)
        _eps = _rng.standard_normal(_data_D.shape).astype(np.float32) * _ns
        for _t in _t_values:
            _t_key = round(_t, 4)
            _z_t = _t * _data_D + (1 - _t) * _eps
            noised_cache[(_D, _t_key)] = (_z_t @ _P).copy()
    return data_preview, noised_cache


@app.cell(hide_code=True)
def _(copy, make_circles, make_moons, make_swiss_roll, nn, np, torch):
    def generate_2d_data(
        name,
        n_samples=10000,
        seed=42,
        noise=0.5,
        circles_factor=0.5,
    ):
        rng = np.random.default_rng(seed)
        if name == "swiss_roll":
            X_3d, _ = make_swiss_roll(
                n_samples, noise=noise, random_state=seed, hole=False
            )
            data = X_3d[:, [0, 2]]
        elif name == "two_moons":
            data, _ = make_moons(n_samples, noise=noise, random_state=seed)
        elif name == "circles":
            data, _ = make_circles(
                n_samples,
                noise=noise,
                factor=circles_factor,
                random_state=seed,
            )
        else:
            data = rng.standard_normal((n_samples, 2))
        data = (data - data.mean(axis=0)) / data.std(axis=0)
        return data.astype(np.float32)

    def make_projection(D, d=2, seed=0):
        rng = np.random.default_rng(seed)
        M = rng.standard_normal((D, d))
        Q, _ = np.linalg.qr(M)
        return Q[:, :d].astype(np.float32)

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

    def train_model(
        pred_type,
        D,
        data_preview,
        P,
        device,
        n_steps=5000,
        batch_size=256,
        lr=1e-3,
        hidden=256,
        t_eps=0.05,
        noise_scale=1.0,
        step_bar=None,
    ):
        P_t = torch.from_numpy(P).to(device)
        x_all = torch.from_numpy(data_preview).to(device) @ P_t.T

        model = DenoisingMLP(D, hidden=hidden).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        losses = []
        best_loss = float("inf")
        best_state = None

        for step in range(n_steps):
            idx = torch.randint(0, x_all.shape[0], (batch_size,), device=device)
            x = x_all[idx]
            eps = torch.randn_like(x) * noise_scale

            logit_t = torch.randn(batch_size, 1, device=device) * 0.8 + (-0.8)
            t = torch.sigmoid(logit_t)

            z_t = t * x + (1 - t) * eps
            pred = model(z_t, t)

            if pred_type == "x_pred":
                denom = (1 - t).clamp(min=t_eps)
                v_true = (x - z_t) / denom
                v_pred = (pred - z_t) / denom
            elif pred_type == "eps_pred":
                v_true = x - eps
                v_pred = (z_t - pred) / t.clamp(min=t_eps)
            else:
                v_true = x - eps
                v_pred = pred

            loss = ((v_pred - v_true) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                loss_val = loss.item()
                losses.append(loss_val)
                if loss_val < best_loss:
                    best_loss = loss_val
                    best_state = copy.deepcopy(model.state_dict())
                if step_bar is not None:
                    step_bar.update(
                        increment=50 if step > 0 else 1,
                        subtitle=f"loss={loss_val:.4f} (best={best_loss:.4f})",
                    )

        if step_bar is not None:
            step_bar.update(increment=n_steps - (len(losses) - 1) * 50)

        model.load_state_dict(best_state)
        model.eval()
        return model, losses

    @torch.no_grad()
    def generate_samples(
        model,
        pred_type,
        D,
        n_samples=2000,
        n_steps=50,
        solver="heun",
        t_eps=0.05,
        noise_scale=1.0,
        return_trajectory=False,
    ):
        device = next(model.parameters()).device
        z = torch.randn(n_samples, D, device=device) * noise_scale
        trajectory = [z.cpu().numpy()] if return_trajectory else None

        def velocity(pred, z_t, t):
            if pred_type == "x_pred":
                return (pred - z_t) / (1 - t).clamp(min=t_eps)
            if pred_type == "eps_pred":
                return (z_t - pred) / t.clamp(min=t_eps)
            return pred

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
                if return_trajectory:
                    trajectory.append(z.cpu().numpy())
        else:
            for i in range(n_steps - 1):
                t = torch.full((n_samples, 1), i / n_steps, device=device)
                t_next = torch.full((n_samples, 1), (i + 1) / n_steps, device=device)
                z = heun_step(z, t, t_next)
                if return_trajectory:
                    trajectory.append(z.cpu().numpy())

            t = torch.full((n_samples, 1), (n_steps - 1) / n_steps, device=device)
            t_next = torch.full((n_samples, 1), 1.0, device=device)
            z = euler_step(z, t, t_next)
            if return_trajectory:
                trajectory.append(z.cpu().numpy())

        if return_trajectory:
            return trajectory
        return z.cpu().numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D_values = [
        2,
        3,
        # 8,
        # 16,
        # 512,
    ]
    n_steps = 5000
    n_sample_steps = 50
    pred_types = ["x_pred", "eps_pred", "v_pred"]
    pred_labels = {
        "x_pred": "$x$-prediction",
        "eps_pred": r"$\epsilon$-prediction",
        "v_pred": "$v$-prediction",
    }
    colors = {
        "ground_truth": "tab:purple",
        "noising": "tab:orange",
        "x_pred": "tab:blue",
        "eps_pred": "tab:red",
        "v_pred": "tab:green",
    }
    return (
        D_values,
        colors,
        device,
        generate_2d_data,
        generate_samples,
        make_projection,
        n_sample_steps,
        n_steps,
        pred_labels,
        pred_types,
        train_model,
    )


if __name__ == "__main__":
    app.run()
