# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.23.10",
#     "jax[cuda13]>=0.10.2",
#     "equinox>=0.13.8",
#     "optax>=0.2.8",
#     "scikit-learn>=1.4.0",
#     "matplotlib>=3.8.0",
#     "numpy>=1.26.0",
#     "torch>=2.0.0",
#     "torchvision>=0.27.1",
#     "einops>=0.7.0",
#     "numba>=0.59.0",
#     "umap-learn>=0.5.5",
#     "pillow>=10.0.0",
#     "huggingface_hub>=0.23.0",
# ]
# ///

import marimo

__generated_with = "0.23.13"
app = marimo.App(
    width="full",
    app_title="Back to Basics: Let Denoising Generative Models Denoise",
)


@app.cell(hide_code=True)
def _():
    import os
    import time

    # Cap JAX at 50% of GPU memory so the torch JiT model (Part 2) has room to
    # load and sample without OOM. Must be set before `import jax`.
    # https://docs.jax.dev/en/latest/gpu_memory_allocation.html
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".5")

    import marimo as mo
    import numpy as np
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import optax
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from sklearn.datasets import make_swiss_roll, make_moons

    import torch
    from torchvision.transforms.functional import to_pil_image

    import math
    from torch import nn
    import torch.nn.functional as F
    from einops import rearrange, repeat
    import umap
    from PIL import Image
    import base64
    from io import BytesIO

    from huggingface_hub import hf_hub_download

    model_path = None
    print("Downloading JiT-B/16 weights from HuggingFace...")
    _dl_t0 = time.time()
    try:
        model_path = hf_hub_download(
            repo_id="avonne/Just-image-Transformer",
            filename="jit-b-16/checkpoint-last.pth",
        )
        print(
            f"Finished downloading JiT-B/16 weights in "
            f"{time.time() - _dl_t0:.1f}s -> {model_path}"
        )
    except Exception as _dl_err:
        print(f"Failed to download JiT-B/16 weights: {_dl_err}")
    return (
        BytesIO,
        F,
        Image,
        Line2D,
        base64,
        eqx,
        jax,
        jnp,
        make_moons,
        make_swiss_roll,
        math,
        mo,
        model_path,
        nn,
        np,
        optax,
        plt,
        rearrange,
        repeat,
        to_pil_image,
        torch,
        umap,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Back to Basics: Let Denoising Generative Models Denoise

    From Li & He, 2025 ([arXiv:2511.13720](https://arxiv.org/abs/2511.13720))

    > Loading the notebook may take a minute or two as it loads the model weights.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1 — What should the network really predict?

    The **manifold hypothesis** states natural data lies on a low-dimensional manifold while noise occupies the ambient space.
    The current trend in diffusion models relies on networks predicting **noise** ($\epsilon$-prediction) or **flow velocity** ($v$-prediction), which live in the high-dimensional ambient space. <br>
    However Li & He break away from this approach by proposing networks that predict **clean data** ($x$-prediction).
    The upside is that the models only have to capture low-dimensional structures corresponding to the manifold the data belongs to.

    In this part we embed 2D or 3D manifolds into higher-dimensional spaces via a random orthogonal projection, then train a small MLP to generate samples using three prediction targets:

    - **$x$-prediction**: directly predict clean data (on-manifold)
    - **$\epsilon$-prediction**: predict the noise (off-manifold)
    - **$v$-prediction**: predict the flow velocity (off-manifold)

    As the ambient dimension D grows, only **$x$-prediction** survives because the network only needs to capture the low-dimensional manifold structure, not the full D-dimensional space.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    dataset_dropdown = mo.ui.dropdown(
        options={
            "Swiss Roll (2D)": "swiss_roll",
            "Two Moons (2D)": "two_moons",
            "Möbius Strip (3D)": "mobius",
            "Gabriel's Horn (3D)": "horn",
            "Klein Bottle (3D)": "klein",
        },
        value="Swiss Roll (2D)",
        label="Dataset",
    )
    n_samples = mo.ui.number(
        start=1000, stop=25000, step=500, value=5000, label="Samples"
    )
    seed = mo.ui.number(start=0, stop=9999, step=1, value=67, label="Seed")
    noise_scale = mo.ui.slider(
        start=0.05,
        stop=1.0,
        step=0.01,
        value=0.3,
        label=r"Noise scale $\sigma$",
        show_value=True,
        debounce=True,
    )

    mo.vstack(
        [
            mo.md("### Choose a dataset"),
            mo.hstack([dataset_dropdown, n_samples, seed]),
        ],
        align="center",
    )
    return dataset_dropdown, n_samples, noise_scale, seed


@app.cell(hide_code=True)
def _(
    colors,
    data_dim,
    data_preview,
    dataset_dropdown,
    dataset_noise,
    manifold_params,
    mo,
    np,
    plt,
):
    _controls = [dataset_noise]
    _fig_size = (5, 5)
    if manifold_params.value:
        _controls.append(manifold_params)

    _lim_clean = np.abs(data_preview).max() * 1.2

    if data_dim == 3:
        _fig_clean = plt.figure(figsize=_fig_size)
        _ax_clean = _fig_clean.add_subplot(111, projection="3d")
        _ax_clean.scatter(
            data_preview[:, 0],
            data_preview[:, 1],
            data_preview[:, 2],
            s=2,
            alpha=0.5,
            c=colors["ground_truth"],
        )
        _ax_clean.set_xticks([])
        _ax_clean.set_yticks([])
        _ax_clean.set_zticks([])
    else:
        _fig_clean, _ax_clean = plt.subplots(figsize=_fig_size)
        _ax_clean.scatter(
            data_preview[:, 0],
            data_preview[:, 1],
            s=2,
            alpha=0.5,
            c=colors["ground_truth"],
        )
        _ax_clean.set_xlim(-_lim_clean, _lim_clean)
        _ax_clean.set_ylim(-_lim_clean, _lim_clean)
        _ax_clean.set_aspect("equal")
        _ax_clean.set_xticks([])
        _ax_clean.set_yticks([])
    _fig_clean.tight_layout()
    dataset_plot = mo.mpl.interactive(_ax_clean)

    mo.vstack(
        [
            mo.md(f"### {dataset_dropdown.selected_key} parametrization"),
            mo.vstack(_controls),
            dataset_plot,
        ],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(
    colors,
    data_dim,
    mo,
    noise_scale,
    noise_time_slider,
    noised_cache,
    np,
    plt,
):
    _fig_size = (5, 5)
    _D = data_dim
    _t = round(noise_time_slider.value, 4)
    _z_t = noised_cache[(_D, _t)]
    _lim_noisy = np.abs(_z_t).max() * 1.2

    if data_dim == 3:
        _fig_noisy = plt.figure(figsize=_fig_size)
        _ax_noisy = _fig_noisy.add_subplot(111, projection="3d")
        _ax_noisy.scatter(
            _z_t[:, 0],
            _z_t[:, 1],
            _z_t[:, 2],
            s=2,
            alpha=0.5,
            c=colors["noising"],
        )
        _ax_noisy.set_xticks([])
        _ax_noisy.set_yticks([])
        _ax_noisy.set_zticks([])
    else:
        _fig_noisy, _ax_noisy = plt.subplots(figsize=_fig_size)
        _ax_noisy.scatter(
            _z_t[:, 0],
            _z_t[:, 1],
            s=2,
            alpha=0.5,
            c=colors["noising"],
        )
        _ax_noisy.set_xlim(-_lim_noisy, _lim_noisy)
        _ax_noisy.set_ylim(-_lim_noisy, _lim_noisy)
        _ax_noisy.set_aspect("equal")
        _ax_noisy.set_xticks([])
        _ax_noisy.set_yticks([])
    _fig_noisy.tight_layout()
    noising_plot = mo.mpl.interactive(_ax_noisy)

    mo.vstack(
        [
            mo.md(r"""
        ### Noising process

        The forward process is a linear interpolation using the flow-matching convention ($t=0$ is noise, $t=1$ is clean data):

        $z_t = t \cdot x + (1 - t) \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)$

        """),
            noise_scale,
            noising_plot,
            mo.md(f"$z_t$ at $t = {_t:.2f}$"),
            noise_time_slider,
        ],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(D_values, mo):
    # Fixed to match the paper's toy-experiment setup (Fig. 2): a 256-hidden,
    # 5-layer MLP, t_eps=0.05, Euler solver. No longer user-configurable so the
    # JAX training/sampling functions can be JIT-warmed at startup below.
    HIDDEN_WIDTH = 256
    T_EPS = 0.05
    SOLVER = "euler"

    train_btn = mo.ui.run_button(label="Train all models")

    D_values_str = ", ".join(f"{D}" for D in D_values)
    mo.vstack(
        [
            mo.md(
                rf"""
            ### Run the experiment

            For each prediction type ($x$, $\epsilon$ and $v$) we will train one model for each of the {D_values_str} ambient dimensions.

            ---
            """
            ),
            train_btn,
        ],
        align="center",
    )
    return HIDDEN_WIDTH, SOLVER, T_EPS, train_btn


@app.cell(hide_code=True)
def _(
    D_values,
    HIDDEN_WIDTH,
    T_EPS,
    data_dim,
    data_preview,
    jax,
    make_projection,
    master_key,
    mo,
    n_steps,
    noise_scale,
    pred_labels,
    train_btn,
    train_model,
):
    mo.stop(not train_btn.value)

    data_2d = data_preview

    results = {}
    projections = {}
    _train_key = master_key

    with mo.status.progress_bar(
        total=len(D_values) * len(pred_labels),
        title=f"Training {len(D_values) * len(pred_labels)} models",
        subtitle=f"Dimension {D_values[0]}",
    ) as main_bar:
        for _D in D_values:
            _P = make_projection(_D, d=data_dim, seed=_D)
            projections[_D] = _P

            for _pred_type, _pred_label in pred_labels.items():
                _train_key, _model_key = jax.random.split(_train_key)
                with mo.status.progress_bar(
                    total=n_steps,
                    title=_pred_type,
                    remove_on_exit=True,
                    subtitle="Jitting training...",
                ) as step_bar:
                    _model, _losses = train_model(
                        _pred_type,
                        _D,
                        data_2d,
                        _P,
                        _model_key,
                        n_steps=n_steps,
                        hidden=HIDDEN_WIDTH,
                        t_eps=T_EPS,
                        noise_scale=noise_scale.value,
                        step_bar=step_bar,
                    )
                results[(_D, _pred_type)] = {"model": _model, "losses": _losses}
                main_bar.update(increment=1, subtitle=f"Dimension {_D}")
    return projections, results


@app.cell(hide_code=True)
def _(D_values, colors, mo, plt, pred_labels, results):
    pred_colors = {k: colors[k] for k in ["x_pred", "eps_pred", "v_pred"]}

    _ncols = 2
    _nrows = -(-len(D_values) // _ncols)
    fig_loss, axes_loss = plt.subplots(
        _nrows, _ncols, figsize=(15, 3 * _nrows), squeeze=False, sharex=True
    )

    for _i, _D in enumerate(D_values):
        _row, _col = divmod(_i, _ncols)
        _ax = axes_loss[_row, _col]
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
        _ax.set_xlim(0, None)
        _ax.set_yscale("log")
        _ax.set_ylabel("v-loss")
        _ax.legend(loc="upper right")
        if _row == _nrows - 1:
            _ax.set_xlabel("Training step (×50)")

    for _j in range(len(D_values), _nrows * _ncols):
        axes_loss[divmod(_j, _ncols)].axis("off")

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
    data_dim,
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

    _subplot_kw = {"projection": "3d"} if data_dim == 3 else {}
    fig, axes = plt.subplots(
        len(D_values),
        4,
        figsize=(3 * 4, 3 * len(D_values)),
        squeeze=False,
        subplot_kw=_subplot_kw,
    )

    _rng = np.random.default_rng(0)
    _idx = _rng.choice(
        len(data_preview), size=min(_n, len(data_preview)), replace=False
    )
    gt_subset = data_preview[_idx]
    lim = np.abs(data_preview).max() * 1.2

    def _scatter(_ax, _data, _color):
        if data_dim == 3:
            _ax.scatter(_data[:, 0], _data[:, 1], _data[:, 2], s=1, alpha=0.5, c=_color)
            _ax.set_xticks([])
            _ax.set_yticks([])
            _ax.set_zticks([])
        else:
            _ax.scatter(_data[:, 0], _data[:, 1], s=1, alpha=0.5, c=_color)
            _ax.set_xlim(-lim, lim)
            _ax.set_ylim(-lim, lim)
            _ax.set_aspect("equal")
            _ax.set_xticks([])
            _ax.set_yticks([])

    for _row, _D in enumerate(D_values):
        _scatter(axes[_row, 0], gt_subset, colors["ground_truth"])
        axes[_row, 0].set_title("Ground Truth" if _row == 0 else "")
        axes[_row, 0].set_ylabel(f"D = {_D}")

        for _col, (_pred_type, _pred_label) in enumerate(pred_labels.items()):
            _ax = axes[_row, _col + 1]
            _samples = trajectories[(_D, _pred_type)][_step]
            _samples_clip = np.clip(_samples, -lim * 2, lim * 2)
            _scatter(_ax, _samples_clip, colors[_pred_type])
            _ax.set_title(_pred_label if _row == 0 else "")

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
def traj_viz(
    Line2D,
    colors,
    data_dim,
    data_preview,
    dataset_dropdown,
    dataset_noise,
    manifold_params,
    mo,
    np,
    plt,
    pred_labels,
    traj_D,
    traj_batch,
    traj_clean_pts,
    traj_focus,
    traj_focus_idx,
    traj_paths,
    traj_view_t,
):
    _n_pts = len(traj_clean_pts)
    _n_steps = len(list(traj_paths.values())[0]) - 1
    _step = min(int(traj_view_t.value * _n_steps), _n_steps)
    _D = traj_D.value
    _t_start = 0.0  # trajectories always start from pure noise
    _t_actual = _t_start + traj_view_t.value * (1.0 - _t_start)
    _focus_on = traj_focus.value
    _focus_i = min(int(traj_focus_idx.value), _n_pts - 1)
    _is_3d = data_dim == 3
    _ds = dataset_dropdown.value
    _mp = manifold_params.value

    def _noisy_stats(_ds, _mp, _noise, _n=10000):
        """Get raw mean/std from a noisy sample for proper mesh alignment."""
        _rng = np.random.default_rng(0)
        if _ds == "mobius":
            _u = _rng.uniform(0, 2 * np.pi, _n)
            _w = _rng.uniform(
                -_mp.get("strip_width", 0.5), _mp.get("strip_width", 0.5), _n
            )
            _k = _mp.get("n_twists", 1)
            _raw = np.column_stack(
                [
                    (1 + _w * np.cos(_k * _u / 2)) * np.cos(_u),
                    (1 + _w * np.cos(_k * _u / 2)) * np.sin(_u),
                    _w * np.sin(_k * _u / 2),
                ]
            )
        elif _ds == "horn":
            _length = _mp.get("length", 8.0)
            _t, _th = (
                _rng.uniform(1, _length, _n),
                _rng.uniform(0, 2 * np.pi, _n),
            )
            _raw = np.column_stack([_t, (1 / _t) * np.cos(_th), (1 / _t) * np.sin(_th)])
        elif _ds == "klein":
            _u, _v = (
                _rng.uniform(0, 2 * np.pi, _n),
                _rng.uniform(0, 2 * np.pi, _n),
            )
            _neck, _body, _h, _a = (
                _mp.get("neck", 4.0),
                _mp.get("body", 6.0),
                _mp.get("height", 16.0),
                _mp.get("angle", 0.0),
            )
            _rk = _neck * (1 - np.cos(_u) / 2)
            _mask = _u < np.pi
            _x4 = np.where(
                _mask,
                _body * np.cos(_u) * (1 + np.sin(_u)) + _rk * np.cos(_u) * np.cos(_v),
                _body * np.cos(_u) * (1 + np.sin(_u)) + _rk * np.cos(_v + np.pi),
            )
            _y4 = np.where(
                _mask,
                _h * np.sin(_u) + _rk * np.sin(_u) * np.cos(_v),
                _h * np.sin(_u),
            )
            _z4, _w4 = _rk * np.sin(_v), _rk * np.cos(_v) * np.sin(_u)
            _raw = np.column_stack(
                [
                    _x4 * np.cos(_a) - _w4 * np.sin(_a),
                    _y4,
                    _z4 * np.cos(_a) + _w4 * np.sin(_a),
                ]
            )
        else:
            return None, None
        _raw += _rng.standard_normal(_raw.shape) * _noise
        return _raw.mean(0), np.maximum(_raw.std(0), 1e-8)

    _mesh_stats = _noisy_stats(_ds, _mp, dataset_noise.value)

    def _standardize_mesh(_X, _Y, _Z):
        if _mesh_stats[0] is None:
            return _X, _Y, _Z
        _mean, _std = _mesh_stats
        return (
            (_X - _mean[0]) / _std[0],
            (_Y - _mean[1]) / _std[1],
            (_Z - _mean[2]) / _std[2],
        )

    def _make_surface(_ds, _mp, _res=40):
        if _ds == "mobius":
            _sw = _mp.get("strip_width", 0.5)
            _uu, _ww = np.meshgrid(
                np.linspace(0, 2 * np.pi, _res),
                np.linspace(-_sw, _sw, max(_res // 2, 10)),
            )
            _k = _mp.get("n_twists", 1)
            _X = (1 + _ww * np.cos(_k * _uu / 2)) * np.cos(_uu)
            _Y = (1 + _ww * np.cos(_k * _uu / 2)) * np.sin(_uu)
            _Z = _ww * np.sin(_k * _uu / 2)
        elif _ds == "horn":
            _length = _mp.get("length", 8.0)
            _tt, _th = np.meshgrid(
                np.linspace(1, _length, _res), np.linspace(0, 2 * np.pi, _res)
            )
            _X = _tt
            _Y = (1 / _tt) * np.cos(_th)
            _Z = (1 / _tt) * np.sin(_th)
        elif _ds == "klein":
            _uu, _vv = np.meshgrid(
                np.linspace(0, 2 * np.pi, _res),
                np.linspace(0, 2 * np.pi, _res),
            )
            _neck, _body = _mp.get("neck", 4.0), _mp.get("body", 6.0)
            _h, _a = _mp.get("height", 16.0), _mp.get("angle", 0.0)
            _rk = _neck * (1 - np.cos(_uu) / 2)
            _mask = _uu < np.pi
            _x4 = np.where(
                _mask,
                _body * np.cos(_uu) * (1 + np.sin(_uu))
                + _rk * np.cos(_uu) * np.cos(_vv),
                _body * np.cos(_uu) * (1 + np.sin(_uu)) + _rk * np.cos(_vv + np.pi),
            )
            _y4 = np.where(
                _mask,
                _h * np.sin(_uu) + _rk * np.sin(_uu) * np.cos(_vv),
                _h * np.sin(_uu),
            )
            _z4 = _rk * np.sin(_vv)
            _w4 = _rk * np.cos(_vv) * np.sin(_uu)
            _X = _x4 * np.cos(_a) - _w4 * np.sin(_a)
            _Y = _y4
            _Z = _z4 * np.cos(_a) + _w4 * np.sin(_a)
        else:
            return None
        return _standardize_mesh(_X, _Y, _Z)

    def _scatter(_ax, _data, **kwargs):
        if _data.ndim == 1:
            _data = _data.reshape(1, -1)
        if _is_3d:
            _ax.scatter(_data[:, 0], _data[:, 1], _data[:, 2], **kwargs)
        else:
            _ax.scatter(_data[:, 0], _data[:, 1], **kwargs)

    def _plot_line(_ax, _data, **kwargs):
        if _is_3d:
            _ax.plot(_data[:, 0], _data[:, 1], _data[:, 2], **kwargs)
        else:
            _ax.plot(_data[:, 0], _data[:, 1], **kwargs)

    _fig = plt.figure(figsize=(10, 8))
    _ax = _fig.add_subplot(111, projection="3d") if _is_3d else _fig.add_subplot(111)
    _lim = np.abs(data_preview).max() * 1.1

    if _is_3d:
        _mesh = _make_surface(_ds, _mp)
        if _mesh is not None:
            _ax.plot_surface(
                _mesh[0],
                _mesh[1],
                _mesh[2],
                alpha=0.12,
                color="lavender",
                edgecolor="plum",
                linewidth=0.3,
                rstride=2,
                cstride=2,
                zorder=0,
            )

    _scatter(_ax, data_preview, s=1, alpha=0.05, c=colors["ground_truth"], zorder=1)

    _legend_handles = []

    for _pred_type, _pred_label in pred_labels.items():
        _traj = traj_paths[_pred_type]
        _color = colors[_pred_type]
        _legend_handles.append(
            Line2D([0], [0], color=_color, linewidth=2, label=_pred_label)
        )

        if _focus_on:
            _path_past = np.array([_traj[_s][_focus_i] for _s in range(_step + 1)])
            _plot_line(
                _ax,
                _path_past,
                alpha=0.9,
                color=_color,
                linewidth=2.5,
                zorder=5,
            )
            if _step < _n_steps:
                _path_future = np.array(
                    [_traj[_s][_focus_i] for _s in range(_step, _n_steps + 1)]
                )
                _plot_line(
                    _ax,
                    _path_future,
                    alpha=0.5,
                    color=_color,
                    linewidth=1.5,
                    zorder=4,
                )
            _scatter(
                _ax,
                _traj[_step][_focus_i],
                s=60,
                c=_color,
                edgecolors="none",
                zorder=7,
            )
        else:
            for _i in range(_n_pts):
                _path = np.array([_traj[_s][_i] for _s in range(_step + 1)])
                _plot_line(
                    _ax,
                    _path,
                    alpha=0.5,
                    color=_color,
                    linewidth=0.5,
                    zorder=3,
                )
            _scatter(
                _ax,
                _traj[_step],
                s=10,
                c=_color,
                zorder=6,
                edgecolors="white",
                linewidths=0.3,
            )

    if _is_3d:
        _ax.set_xlim(-_lim, _lim)
        _ax.set_ylim(-_lim, _lim)
        _ax.set_zlim(-_lim, _lim)
        _ax.set_xticks([])
        _ax.set_yticks([])
        _ax.set_zticks([])
    else:
        _ax.set_xlim(-_lim, _lim)
        _ax.set_ylim(-_lim, _lim)
        _ax.set_aspect("equal")
        _ax.set_xticks([])
        _ax.set_yticks([])

    _ax.legend(handles=_legend_handles, loc="upper right")
    _fig.tight_layout()

    _row3 = [traj_focus]
    if traj_focus.value:
        _row3.append(traj_focus_idx)
    _title = f"## Denoising trajectories to $t={_t_actual:.2f}$ in $D={_D}$"

    _controls = [
        mo.hstack([traj_D, traj_batch]),
        traj_view_t,
        mo.hstack(_row3),
    ]

    mo.vstack(
        [
            mo.md(_title),
            mo.md(r"""
        Each line traces a point's path through the reverse denoising ODE.
        **$x$-prediction** trajectories stay close to the data manifold, while **$\epsilon$** and **$v$-prediction** trajectories can wander off-manifold.
        """),
            mo.vstack(_controls),
            mo.mpl.interactive(_fig),
        ],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(
    colors,
    data_dim,
    data_preview,
    mo,
    np,
    plt,
    pred_labels,
    time_slider,
    vf_cache,
    vf_grid_lim,
    vf_grid_res,
    vf_streamline,
):
    mo.stop(
        data_dim > 2,
        mo.callout(
            "Vector field visualization is only available for 2D datasets.",
            kind="warn",
        ),
    )
    _t = round(time_slider.value, 4)
    _res = int(vf_grid_res.value)
    _lim = vf_grid_lim

    _xs = np.linspace(-_lim, _lim, _res)
    _ys = np.linspace(-_lim, _lim, _res)
    _xx, _yy = np.meshgrid(_xs, _ys)

    _vf_by_type = {}
    for _pred_type in pred_labels:
        _vals = vf_cache[(_t, _pred_type)]["v"]
        _u = _vals[:, 0].reshape(_res, _res)
        _vy = _vals[:, 1].reshape(_res, _res)
        _mag = np.sqrt(_u**2 + _vy**2)
        _vf_by_type[_pred_type] = (_u, _vy, _mag, _mag.max() or 1e-8)

    _fig, _axes = plt.subplots(1, 3, figsize=(16, 8))

    _mappable = None
    for _ax, (_pred_type, _pred_label) in zip(_axes, pred_labels.items()):
        _u, _vy, _mag, _max_mag = _vf_by_type[_pred_type]
        _mag_norm = _mag / _max_mag

        _ax.scatter(
            data_preview[:, 0],
            data_preview[:, 1],
            s=1,
            alpha=0.1,
            c=colors["ground_truth"],
            zorder=1,
        )
        if vf_streamline.value:
            _strm = _ax.streamplot(
                _xs,
                _ys,
                _u,
                _vy,
                color=_mag_norm,
                cmap="viridis",
                norm=plt.Normalize(0, 1),
                linewidth=1,
                density=1.2,
                arrowsize=1.2,
            )
            _mappable = _strm.lines
        else:
            _dir_u = _u / np.maximum(_mag, 1e-8)
            _dir_vy = _vy / np.maximum(_mag, 1e-8)
            _quiv = _ax.quiver(
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
            _mappable = _quiv

        _ax.set_title(_pred_label, fontsize=14)
        _ax.set_xlim(-_lim, _lim)
        _ax.set_ylim(-_lim, _lim)
        _ax.set_aspect("equal")
        _ax.set_xticks([])
        _ax.set_yticks([])

    _fig.colorbar(_mappable, ax=_axes, fraction=0.02, pad=0.02, label="‖v‖", shrink=0.5)

    _vf_html = mo.as_html(_fig)
    plt.close(_fig)

    mo.vstack(
        [
            mo.md(f"## Velocity fields at $t={_t:.2f}$ ($D=2$)"),
            mo.md(r"""
        Each model predicts either $\hat{x}$, $\hat{\epsilon}$, or $\hat{v}$.
        The panels below shows the **time-varying** velocity field for those prediction types.

        We can recover the velocity field $\hat{v}$ from the predicted $\hat{x}$ or $\hat{\epsilon}$ with the following relationships:
        - $\hat{v}(\hat{x}) = (\hat{x} - z_t) / (1 - t)$
        - $\hat{v}(\hat{\epsilon}) = (z_t - \hat{\epsilon}) / t$

        """),
            mo.vstack(
                [
                    vf_grid_res,
                    time_slider,
                    vf_streamline,
                    _vf_html,
                ],
                align="center",
            ),
        ],
    )
    return


@app.cell(hide_code=True)
def _(dataset_dropdown):
    _3d_datasets = {"mobius", "horn", "klein"}
    data_dim = 3 if dataset_dropdown.value in _3d_datasets else 2
    D_values = [data_dim, 8, 16, 512]
    return D_values, data_dim


@app.cell(hide_code=True)
def _(dataset_dropdown, mo, np):
    _ds = dataset_dropdown.value
    if _ds == "mobius":
        manifold_params = mo.ui.dictionary(
            {
                "n_twists": mo.ui.slider(
                    start=1,
                    stop=5,
                    step=1,
                    value=1,
                    label="Half-twists",
                    show_value=True,
                ),
                "strip_width": mo.ui.slider(
                    start=0.2,
                    stop=1.5,
                    step=0.1,
                    value=0.5,
                    label="Strip width",
                    show_value=True,
                ),
            }
        )
    elif _ds == "horn":
        manifold_params = mo.ui.dictionary(
            {
                "length": mo.ui.slider(
                    start=1.0,
                    stop=20.0,
                    step=0.5,
                    value=8.0,
                    label="Length",
                    show_value=True,
                ),
            }
        )
    elif _ds == "klein":
        manifold_params = mo.ui.dictionary(
            {
                "neck": mo.ui.slider(
                    start=1.0,
                    stop=8.0,
                    step=0.5,
                    value=4.0,
                    label="Neck width",
                    show_value=True,
                ),
                "body": mo.ui.slider(
                    start=3.0,
                    stop=10.0,
                    step=0.5,
                    value=6.0,
                    label="Body size",
                    show_value=True,
                ),
                "height": mo.ui.slider(
                    start=8.0,
                    stop=24.0,
                    step=1.0,
                    value=16.0,
                    label="Height",
                    show_value=True,
                ),
                "angle": mo.ui.slider(
                    start=0.0,
                    stop=round(float(2 * np.pi), 2),
                    step=0.05,
                    value=0.0,
                    label="4D rotation angle",
                    show_value=True,
                ),
            }
        )
    else:
        # two_moons has no additional shape parameters beyond noise/samples/seed.
        manifold_params = mo.ui.dictionary({})
    return (manifold_params,)


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
    T_EPS,
    data_dim,
    data_preview,
    jax,
    jnp,
    mo,
    n_sample_steps,
    np,
    pred_labels,
    results,
    vf_grid_res,
):
    mo.stop(data_dim > 2)
    _res = int(vf_grid_res.value)
    _t_eps = T_EPS
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
        _z_t = jnp.array(_grid)
        for _t_val in _t_values:
            _t_key = round(_t_val, 4)
            _t_tensor = jnp.full((_grid.shape[0], 1), _t_val)
            for _pred_type in pred_labels:
                _pred = jax.vmap(results[(2, _pred_type)]["model"])(_z_t, _t_tensor)
                if _pred_type == "x_pred":
                    _x_hat = _pred
                    _eps_hat = (_z_t - _t_tensor * _x_hat) / jnp.maximum(
                        1 - _t_tensor, _t_eps
                    )
                    _v_hat = (_x_hat - _z_t) / jnp.maximum(1 - _t_tensor, _t_eps)
                elif _pred_type == "eps_pred":
                    _eps_hat = _pred
                    _x_hat = (_z_t - (1 - _t_tensor) * _eps_hat) / jnp.maximum(
                        _t_tensor, _t_eps
                    )
                    _v_hat = (_z_t - _eps_hat) / jnp.maximum(_t_tensor, _t_eps)
                else:
                    _v_hat = _pred
                    _x_hat = (1 - _t_tensor) * _v_hat + _z_t
                    _eps_hat = _z_t - _t_tensor * _v_hat
                vf_cache[(_t_key, _pred_type)] = {
                    "x": np.asarray(_x_hat),
                    "eps": np.asarray(_eps_hat),
                    "v": np.asarray(_v_hat),
                }
                _bar.update(increment=1)
    return vf_cache, vf_grid_lim


@app.cell(hide_code=True)
def _(
    D_values,
    SOLVER,
    T_EPS,
    gen_samples,
    generate_samples,
    jax,
    master_key,
    n_sample_steps,
    noise_scale,
    pred_labels,
    projections,
    results,
):
    _n = int(gen_samples.value)
    trajectories = {}
    _sample_key = master_key
    for _D in D_values:
        _P = projections[_D]
        for _pred_type in pred_labels:
            _sample_key, _model_key = jax.random.split(_sample_key)
            _traj_D = generate_samples(
                results[(_D, _pred_type)]["model"],
                _pred_type,
                _D,
                n_samples=_n,
                n_steps=n_sample_steps,
                solver=SOLVER,
                t_eps=T_EPS,
                noise_scale=noise_scale.value,
                return_trajectory=True,
                key=_model_key,
            )
            trajectories[(_D, _pred_type)] = [_snap @ _P for _snap in _traj_D]
    return (trajectories,)


@app.cell(hide_code=True)
def _(mo, n_sample_steps):
    time_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=round(1.0 / n_sample_steps, 4),
        value=0.0,
        label="Timestep $t$",
        debounce=True,
    )
    noise_time_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=round(1.0 / n_sample_steps, 4),
        value=0.0,
        label="Timestep $t$",
        debounce=True,
    )
    vf_grid_res = mo.ui.slider(
        start=10,
        stop=40,
        step=5,
        value=20,
        label="Grid resolution",
        show_value=True,
        debounce=True,
    )
    vf_streamline = mo.ui.switch(
        label="Streamline plot (instead of quiver)", value=False
    )
    return noise_time_slider, time_slider, vf_grid_res, vf_streamline


@app.cell(hide_code=True)
def _(dataset_dropdown, mo):
    _noise_configs = {
        "swiss_roll": (0.0, 4.0, 0.01, 0.0),
        "two_moons": (0.0, 0.5, 0.005, 0.0),
        "mobius": (0.0, 0.5, 0.01, 0.0),
        "horn": (0.0, 0.5, 0.01, 0.0),
        "klein": (0.0, 5.0, 0.1, 0.0),
    }
    _start, _stop, _step, _value = _noise_configs.get(
        dataset_dropdown.value, (0.0, 4.0, 0.01, 0.0)
    )
    dataset_noise = mo.ui.slider(
        start=_start,
        stop=_stop,
        step=_step,
        value=_value,
        label="Dataset noise",
        debounce=True,
    )
    return (dataset_noise,)


@app.cell(hide_code=True)
def _(
    D_values,
    data_dim,
    dataset_dropdown,
    dataset_noise,
    generate_data,
    make_projection,
    manifold_params,
    n_sample_steps,
    n_samples,
    noise_scale,
    np,
    seed,
):
    data_preview = generate_data(
        dataset_dropdown.value,
        n_samples=int(n_samples.value),
        seed=int(seed.value),
        noise=dataset_noise.value,
        manifold_params=manifold_params.value,
    )

    _t_values = np.linspace(0.0, 1.0, n_sample_steps + 1)
    _ns = noise_scale.value

    noised_cache = {}
    for _D in D_values:
        _P = make_projection(_D, d=data_dim, seed=_D)
        _data_D = data_preview @ _P.T
        _rng = np.random.default_rng(0)
        _eps = _rng.standard_normal(_data_D.shape).astype(np.float32) * _ns
        for _t in _t_values:
            _t_key = round(_t, 4)
            _z_t = _t * _data_D + (1 - _t) * _eps
            noised_cache[(_D, _t_key)] = (_z_t @ _P).copy()
    return data_preview, noised_cache


@app.cell(hide_code=True)
def _(eqx, jax, jnp, make_moons, make_swiss_roll, np, optax):
    def generate_data(
        name,
        n_samples=10000,
        seed=42,
        noise=0.5,
        manifold_params=None,
    ):
        rng = np.random.default_rng(seed)
        _mp = manifold_params or {}
        if name == "swiss_roll":
            hole = bool(_mp.get("hole", False))
            X_3d, _ = make_swiss_roll(
                n_samples, noise=noise, random_state=seed, hole=hole
            )
            data = X_3d[:, [0, 2]]
        elif name == "two_moons":
            data, _ = make_moons(n_samples, noise=noise, random_state=seed)
        elif name == "mobius":
            u = rng.uniform(0, 2 * np.pi, n_samples)
            k = _mp.get("n_twists", 1)
            sw = _mp.get("strip_width", 0.5)
            w = rng.uniform(-sw, sw, n_samples)
            x = (1 + w * np.cos(k * u / 2)) * np.cos(u)
            y = (1 + w * np.cos(k * u / 2)) * np.sin(u)
            z = w * np.sin(k * u / 2)
            data = np.column_stack([x, y, z])
            data += rng.standard_normal(data.shape) * noise
        elif name == "horn":
            length = _mp.get("length", 8.0)
            t = rng.uniform(1, length, n_samples)
            theta = rng.uniform(0, 2 * np.pi, n_samples)
            x = t
            y = (1 / t) * np.cos(theta)
            z = (1 / t) * np.sin(theta)
            data = np.column_stack([x, y, z])
            data += rng.standard_normal(data.shape) * noise
        elif name == "klein":
            u = rng.uniform(0, 2 * np.pi, n_samples)
            v = rng.uniform(0, 2 * np.pi, n_samples)
            neck = _mp.get("neck", 4.0)
            body = _mp.get("body", 6.0)
            h = _mp.get("height", 16.0)
            a = _mp.get("angle", 0.0)
            r_k = neck * (1 - np.cos(u) / 2)
            mask = u < np.pi
            x4 = np.where(
                mask,
                body * np.cos(u) * (1 + np.sin(u)) + r_k * np.cos(u) * np.cos(v),
                body * np.cos(u) * (1 + np.sin(u)) + r_k * np.cos(v + np.pi),
            )
            y4 = np.where(
                mask,
                h * np.sin(u) + r_k * np.sin(u) * np.cos(v),
                h * np.sin(u),
            )
            z4 = r_k * np.sin(v)
            w4 = r_k * np.cos(v) * np.sin(u)
            x = x4 * np.cos(a) - w4 * np.sin(a)
            y = y4
            z = z4 * np.cos(a) + w4 * np.sin(a)
            data = np.column_stack([x, y, z])
            data += rng.standard_normal(data.shape) * noise
        else:
            data = rng.standard_normal((n_samples, 2))
        data = (data - data.mean(axis=0)) / data.std(axis=0)
        return data.astype(np.float32)

    def make_projection(D, d=2, seed=0):
        rng = np.random.default_rng(seed)
        M = rng.standard_normal((D, d))
        Q, _ = np.linalg.qr(M)
        return Q[:, :d].astype(np.float32)

    class DenoisingMLP(eqx.Module):
        layers: list

        def __init__(self, dim, hidden=256, n_layers=5, *, key):
            keys = jax.random.split(key, n_layers)
            self.layers = [eqx.nn.Linear(dim + 1, hidden, key=keys[0])]
            for i in range(1, n_layers - 1):
                self.layers.append(eqx.nn.Linear(hidden, hidden, key=keys[i]))
            self.layers.append(eqx.nn.Linear(hidden, dim, key=keys[n_layers - 1]))

        def __call__(self, z_t, t):
            x = jnp.concatenate([z_t, t], axis=-1)
            for layer in self.layers[:-1]:
                x = jax.nn.relu(layer(x))
            return self.layers[-1](x)

    _T_MEAN = -0.8
    _T_STD = 0.8
    _CHUNK_SIZE = 50
    _optimizer = optax.adam(1e-3)

    def _make_scan_chunk(pred_type, optimizer, batch_size):
        @eqx.filter_jit
        def scan_chunk(carry, x_all, keys, noise_scale, t_eps):
            def scan_body(carry, key_i):
                model, opt_state, best_model_arrays, best_loss = carry
                k_idx, k_eps, k_t = jax.random.split(key_i, 3)

                idx = jax.random.randint(k_idx, (batch_size,), 0, x_all.shape[0])
                x = x_all[idx]
                eps = jax.random.normal(k_eps, x.shape) * noise_scale
                logit_t = jax.random.normal(k_t, (x.shape[0], 1)) * _T_STD + _T_MEAN
                t = jax.nn.sigmoid(logit_t)
                z_t = t * x + (1 - t) * eps

                def loss_fn(model):
                    pred = jax.vmap(model)(z_t, t)
                    if pred_type == "x_pred":
                        denom = jnp.maximum(1 - t, t_eps)
                        v_true = (x - z_t) / denom
                        v_pred = (pred - z_t) / denom
                    elif pred_type == "eps_pred":
                        v_true = x - eps
                        v_pred = (z_t - pred) / jnp.maximum(t, t_eps)
                    else:
                        v_true = x - eps
                        v_pred = pred
                    return jnp.mean((v_pred - v_true) ** 2)

                loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
                updates, opt_state = optimizer.update(grads, opt_state, model)
                model = eqx.apply_updates(model, updates)

                improved = loss < best_loss
                best_loss = jnp.where(improved, loss, best_loss)
                model_arrays = eqx.filter(model, eqx.is_array)
                best_model_arrays = jax.tree.map(
                    lambda new, old: jnp.where(improved, new, old),
                    model_arrays,
                    best_model_arrays,
                )

                return (model, opt_state, best_model_arrays, best_loss), loss

            return jax.lax.scan(scan_body, carry, keys)

        return scan_chunk

    pred_types = ["x_pred", "eps_pred", "v_pred"]
    _scan_fns = {pt: _make_scan_chunk(pt, _optimizer, 256) for pt in pred_types}

    def train_model(
        pred_type,
        D,
        data_preview,
        P,
        key,
        n_steps=5000,
        batch_size=256,
        lr=1e-3,
        hidden=256,
        t_eps=0.05,
        noise_scale=1.0,
        step_bar=None,
    ):
        P_jax = jnp.array(P)
        x_all = jnp.array(data_preview) @ P_jax.T

        key, model_key, scan_key = jax.random.split(key, 3)
        model = DenoisingMLP(D, hidden=hidden, key=model_key)
        opt_state = _optimizer.init(eqx.filter(model, eqx.is_array))

        all_keys = jax.random.split(scan_key, n_steps)
        n_chunks = n_steps // _CHUNK_SIZE

        best_arrays = eqx.filter(model, eqx.is_array)
        carry = (model, opt_state, best_arrays, jnp.array(float("inf")))
        losses = []

        scan_fn = _scan_fns[pred_type]
        _ns = jnp.array(noise_scale)
        _te = jnp.array(t_eps)

        for chunk_i in range(n_chunks):
            chunk_keys = all_keys[chunk_i * _CHUNK_SIZE : (chunk_i + 1) * _CHUNK_SIZE]
            carry, chunk_losses = scan_fn(carry, x_all, chunk_keys, _ns, _te)
            loss_val = float(chunk_losses[-1])
            losses.append(loss_val)
            if step_bar is not None:
                best_loss_val = float(carry[3])
                step_bar.update(
                    increment=_CHUNK_SIZE,
                    subtitle=f"loss={loss_val:.4f} (best={best_loss_val:.4f})",
                )

        model, _, best_arrays, _ = carry
        best_model = eqx.combine(best_arrays, model)
        return best_model, losses

    @eqx.filter_jit
    def _model_batch_eval(model, z_t, t):
        return jax.vmap(model)(z_t, t)

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
        key=None,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)
        z = jax.random.normal(key, (n_samples, D)) * noise_scale
        trajectory = [np.asarray(z)] if return_trajectory else None

        def velocity(pred, z_t, t_val):
            if pred_type == "x_pred":
                return (pred - z_t) / jnp.maximum(1 - t_val, t_eps)
            if pred_type == "eps_pred":
                return (z_t - pred) / jnp.maximum(t_val, t_eps)
            return pred

        def euler_step(z_t, t_val, t_next_val):
            t_arr = jnp.full((z_t.shape[0], 1), t_val)
            pred = _model_batch_eval(model, z_t, t_arr)
            v = velocity(pred, z_t, t_val)
            return z_t + (t_next_val - t_val) * v

        def heun_step(z_t, t_val, t_next_val):
            t_arr = jnp.full((z_t.shape[0], 1), t_val)
            t_next_arr = jnp.full((z_t.shape[0], 1), t_next_val)
            pred_t = _model_batch_eval(model, z_t, t_arr)
            v_t = velocity(pred_t, z_t, t_val)
            z_euler = z_t + (t_next_val - t_val) * v_t
            pred_next = _model_batch_eval(model, z_euler, t_next_arr)
            v_next = velocity(pred_next, z_euler, t_next_val)
            return z_t + (t_next_val - t_val) * 0.5 * (v_t + v_next)

        if solver not in {"euler", "heun"}:
            raise ValueError(f"Unsupported solver: {solver}")

        step_fn = heun_step if solver == "heun" else euler_step

        for i in range(n_steps):
            t_val = i / n_steps
            t_next_val = (i + 1) / n_steps
            if solver == "heun" and i == n_steps - 1:
                z = euler_step(z, t_val, t_next_val)
            else:
                z = step_fn(z, t_val, t_next_val)
            if return_trajectory:
                trajectory.append(np.asarray(z))

        if return_trajectory:
            return trajectory
        return np.asarray(z)

    master_key = jax.random.PRNGKey(67)
    n_steps = 10000
    n_sample_steps = 50
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
        colors,
        generate_data,
        generate_samples,
        make_projection,
        master_key,
        n_sample_steps,
        n_steps,
        pred_labels,
        train_model,
    )


@app.cell(hide_code=True)
def traj_controls(D_values, mo, n_sample_steps):
    traj_D = mo.ui.dropdown(
        options={str(value): value for value in D_values},
        value=str(D_values[0]),
        label="Ambient dimension",
    )
    traj_batch = mo.ui.slider(
        start=10,
        stop=200,
        step=10,
        value=50,
        label="Trajectory points",
        show_value=True,
    )
    traj_view_t = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=round(1.0 / n_sample_steps, 4),
        value=1.0,
        label="View at $t$",
        show_value=True,
    )
    traj_focus = mo.ui.switch(label="Focus on single point", value=False)
    return traj_D, traj_batch, traj_focus, traj_view_t


@app.cell(hide_code=True)
def traj_compute(
    SOLVER,
    T_EPS,
    data_preview,
    jax,
    jnp,
    n_sample_steps,
    noise_scale,
    np,
    pred_labels,
    projections,
    results,
    traj_D,
    traj_batch,
):
    _D = traj_D.value
    _P_np = projections[_D]
    _P = jnp.array(_P_np)
    _t_start = 0.0  # trajectories always start from pure noise
    _n = int(traj_batch.value)
    _ns = noise_scale.value
    _te = T_EPS
    _n_steps = n_sample_steps
    _use_heun = SOLVER == "heun"

    _rng = np.random.default_rng(42)
    _idx = _rng.choice(
        len(data_preview), size=min(_n, len(data_preview)), replace=False
    )
    traj_clean_pts = data_preview[_idx]

    _x_D = jnp.array(traj_clean_pts) @ _P.T
    _eps = jax.random.normal(jax.random.PRNGKey(42), _x_D.shape) * _ns
    _z_start = _t_start * _x_D + (1 - _t_start) * _eps

    def _get_velocity(_pred, _z_t, _t_val, _pred_type, _t_eps):
        if _pred_type == "x_pred":
            return (_pred - _z_t) / jnp.maximum(1 - _t_val, _t_eps)
        elif _pred_type == "eps_pred":
            return (_z_t - _pred) / jnp.maximum(_t_val, _t_eps)
        return _pred

    traj_paths = {}
    for _pred_type in pred_labels:
        _model = results[(_D, _pred_type)]["model"]
        _z = _z_start
        _traj = [np.asarray(_z @ _P)]

        for _i in range(_n_steps):
            _t_val = _t_start + (1.0 - _t_start) * _i / _n_steps
            _t_next = _t_start + (1.0 - _t_start) * (_i + 1) / _n_steps
            _t_arr = jnp.full((_z.shape[0], 1), _t_val)
            _pred = jax.vmap(_model)(_z, _t_arr)
            _v = _get_velocity(_pred, _z, _t_val, _pred_type, _te)

            if _use_heun and _i < _n_steps - 1:
                _z_euler = _z + (_t_next - _t_val) * _v
                _t_next_arr = jnp.full((_z.shape[0], 1), _t_next)
                _pred_next = jax.vmap(_model)(_z_euler, _t_next_arr)
                _v_next = _get_velocity(_pred_next, _z_euler, _t_next, _pred_type, _te)
                _z = _z + (_t_next - _t_val) * 0.5 * (_v + _v_next)
            else:
                _z = _z + (_t_next - _t_val) * _v

            _traj.append(np.asarray(_z @ _P))

        traj_paths[_pred_type] = _traj
    return traj_clean_pts, traj_paths


@app.cell(hide_code=True)
def traj_focus_idx_cell(mo, traj_clean_pts):
    traj_focus_idx = mo.ui.slider(
        start=0,
        stop=len(traj_clean_pts) - 1,
        step=1,
        value=0,
        label="Point index",
        show_value=True,
    )
    return (traj_focus_idx,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 2 — From noise to a real image with Just image Transformers

    The proposed architecture, **Just image Transformers** (JiT) is nothing more than a plain Vision Transformer on patches of raw pixels.<br>
    The paper trains JiT on ImageNet at $256 \times 256$, $512 \times 512$ and $1024 \times 1024$ resolutions.

    In our case we will be using pre-trained [JiT-B/16](https://huggingface.co/avonne/Just-image-Transformer) to generate $256 \times 256$ images on 5 different classes:
    - Golden retriever
    - Volcano
    - Goldfish
    - Giant panda
    - Daisy
    """)
    return


@app.cell(hide_code=True)
def _(JIT_IMG_SIZE, JIT_NOISE_SCALE, mo, noise_seed, to_pil_image, torch):
    _g = torch.Generator().manual_seed(int(noise_seed.value))
    image_tensor = JIT_NOISE_SCALE * torch.randn(
        1, 3, JIT_IMG_SIZE, JIT_IMG_SIZE, generator=_g
    )
    # `image_tensor` is pure Gaussian noise at the real JiT-B/16's input
    # resolution. This exact tensor is reused, unmodified, as the starting
    # point z_0 for the real model's denoising trajectory below.
    noise_preview = ((image_tensor[0].clamp(-3, 3) + 3) / 6).clamp(0, 1)
    img_np = noise_preview.permute(1, 2, 0).numpy()

    mo.vstack(
        [
            mo.md(rf"""
    ### Generating noise

    It all starts with pure Gaussian noise at a ${JIT_IMG_SIZE} \times {JIT_IMG_SIZE}$ resolution. But we will need to process this noisy image beforehand.
    """),
            noise_seed,
            mo.image(src=to_pil_image(noise_preview)),
        ],
        align="center",
    )
    return image_tensor, img_np


@app.cell(hide_code=True)
def _(
    grid_html,
    grid_size,
    img_np,
    mo,
    patch_size,
    plt,
    pos_embed,
    row_slider,
):
    _columns = [
        mo.vstack(
            [mo.md("Noise Patch"), mo.md("Positional embedding")],
            justify="space-around",
        )
    ]
    for _c in range(grid_size):
        _patch = img_np[
            row_slider.value * patch_size : (row_slider.value + 1) * patch_size,
            _c * patch_size : (_c + 1) * patch_size,
            :,
        ]
        _fig_pa, _ax_pa = plt.subplots(figsize=(1, 1))
        _ax_pa.imshow(_patch, aspect="auto")
        _ax_pa.set_xticks([])
        _ax_pa.set_yticks([])
        _fig_pa.tight_layout(pad=0.05)
        _patch_html = mo.as_html(_fig_pa)
        plt.close(_fig_pa)

        _pe_vec = pos_embed[row_slider.value, _c, :].reshape(24, 32)
        _fig_pe, _ax_pe = plt.subplots(figsize=(24 / 32, 1))
        _ax_pe.imshow(_pe_vec, cmap="turbo", aspect="auto")
        _ax_pe.set_xticks([])
        _ax_pe.set_yticks([])
        _fig_pe.tight_layout(pad=0.05)
        _pe_html = mo.as_html(_fig_pe)
        plt.close(_fig_pe)

        _columns.append(
            mo.vstack([_patch_html, _pe_html], align="center", gap="0.15rem")
        )

    mo.vstack(
        [
            mo.md(
                f"The image is separated into a {grid_size}×{grid_size} grid of {patch_size}×{patch_size} patches. Each patch is flattened into a token, and associated with a positional embedding so the Transformer knows where each patch came from."
            ),
            mo.hstack(
                [
                    grid_html.style(
                        {
                            "width": "5in",
                            "height": "5in",
                        }
                    ),
                    row_slider,
                ],
                align="center",
            ),
            mo.md(
                f"Below is one full row of ${patch_size} \\times {patch_size}$ patches, each with its 2D sincos positional embedding."
            ),
            mo.md(
                r"The embeddings are $\mathbb{R}^{768}$ vectors reshaped into $24 \times 32$ images."
            ),
            mo.hstack(_columns, justify="center"),
        ],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _():
    # Shared with the real JiT-B/16 model below: the exact resolution and noise
    # scale it expects as input.
    JIT_IMG_SIZE = 256
    JIT_NOISE_SCALE = 1.0
    return JIT_IMG_SIZE, JIT_NOISE_SCALE


@app.cell(hide_code=True)
def _(JIT_IMG_SIZE, mo, np):
    noise_seed = mo.ui.number(start=0, stop=9999, step=1, value=67, label="Noise seed")

    patch_size = 16
    grid_size = JIT_IMG_SIZE // patch_size  # 16x16 real patch grid

    # 2D sincos positional embedding (same construction as ViT/DiT) for the
    # model's actual patch grid.
    def _sincos_pos_embed_2d(_embed_dim, _size):
        _gh, _gw = np.meshgrid(
            np.arange(_size, dtype=np.float32),
            np.arange(_size, dtype=np.float32),
            indexing="ij",
        )
        _omega = np.arange(_embed_dim // 4, dtype=np.float64)
        _omega = 1.0 / 10000 ** (_omega / (_embed_dim / 4.0))
        _out_h = np.einsum("hw,d->hwd", _gh, _omega)
        _out_w = np.einsum("hw,d->hwd", _gw, _omega)
        return np.concatenate(
            [np.sin(_out_h), np.cos(_out_h), np.sin(_out_w), np.cos(_out_w)], axis=-1
        )

    pos_embed = _sincos_pos_embed_2d(768, grid_size)

    row_slider = mo.ui.slider(
        steps=range(grid_size - 1, -1, -1),
        value=0,
        label="Patch row index",
        full_width=False,
        orientation="vertical",
    )
    return grid_size, noise_seed, patch_size, pos_embed, row_slider


@app.cell(hide_code=True)
def _(JIT_IMG_SIZE, grid_size, img_np, mo, patch_size, plt, row_slider):
    # Full image with a red grid overlay showing how it's cut into patches;
    # the row we zoom into below is highlighted.
    _fig_grid, _ax_grid = plt.subplots(figsize=(4, 4))
    _ax_grid.imshow(img_np)
    _ax_grid.axhspan(
        row_slider.value * patch_size - 0.5,
        (row_slider.value + 1) * patch_size - 0.5,
        color="red",
        alpha=0.5,
        zorder=1,
    )
    for _i in range(grid_size + 1):
        _ax_grid.axhline(_i * patch_size - 0.5, color="red", linewidth=1, zorder=2)
        _ax_grid.axvline(_i * patch_size - 0.5, color="red", linewidth=1, zorder=2)
    _ax_grid.set_xlim(-0.5, JIT_IMG_SIZE - 0.5)
    _ax_grid.set_ylim(JIT_IMG_SIZE - 0.5, -0.5)
    _ax_grid.set_xticks([])
    _ax_grid.set_yticks([])
    _fig_grid.tight_layout(pad=0.1)
    grid_html = mo.as_html(_fig_grid)
    plt.close(_fig_grid)
    return (grid_html,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Timestep & class conditioning

    JiT conditions the denoiser on the timestep $t$ using a sinusoidal frequency embedding.<br>
    The embedding is a 256-dim vector built from $\sin$/$\cos$ at geometrically spaced frequencies of $t$.

    Class labels are conditioned via a learned embedding table during training. Each class embedding is a 768-dim vector.<br>
    The class embeddings below are reshaped into $24 \times 32$ images.
    """)
    return


@app.cell(hide_code=True)
def time_embed_vis(
    CLASSES,
    TimestepEmbedder,
    denoiser,
    mo,
    plt,
    time_embed_slider,
    torch,
):
    _t_tensor = torch.tensor([time_embed_slider.value], dtype=torch.float32)
    _t_emb = (
        TimestepEmbedder.timestep_embedding(_t_tensor, 256)[0].numpy().reshape(1, -1)
    )

    _fig_t, _ax_t = plt.subplots(figsize=(3, 0.2))
    _ax_t.imshow(_t_emb, cmap="turbo", aspect="auto")
    _ax_t.set_xticks([])
    _ax_t.set_yticks([])
    _fig_t.tight_layout(pad=0.1)
    _t_emb_html = mo.as_html(_fig_t)
    plt.close(_fig_t)

    _columns = []
    for _cid, _cname in CLASSES.items():
        _emb_vec = (
            denoiser.net.y_embedder.embedding_table.weight[_cid]
            .detach()
            .cpu()
            .numpy()
            .reshape(24, 32)
        )
        _fig_c, _ax_c = plt.subplots(figsize=(1.5 * 24 / 32, 1.5))
        _ax_c.imshow(_emb_vec, cmap="turbo", aspect="auto")
        _ax_c.set_xticks([])
        _ax_c.set_yticks([])
        _fig_c.tight_layout(pad=0.05)
        _c_html = mo.as_html(_fig_c)
        plt.close(_fig_c)
        _columns.append(mo.vstack([mo.md(f"{_cname}"), _c_html], align="center"))
    _class_embeds = mo.hstack(_columns, justify="center")

    mo.vstack(
        [
            time_embed_slider,
            mo.md(f"Timestep embedding at $t={time_embed_slider.value}$"),
            _t_emb_html,
            mo.md("Class embeddings"),
            _class_embeds,
        ],
        align="center",
    )
    return


@app.cell(hide_code=True)
def time_embed_slider_def(mo):
    time_embed_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.01,
        value=0.5,
        label="Timestep $t$",
        show_value=False,
    )
    return (time_embed_slider,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Letting JiT denoise the noise

    For each class label, we feed the patches and the embeddings to JiT-B/16 and take the resulting (slightly) denoised image $z_{t+1}$, then repeat the process until we get the final image $z_T$.
    At every denoising step, we capture the image $z_t$. This lets us build a trajectory of the image, which is then projected onto a 2D plane with [UMAP](https://umap-learn.readthedocs.io/en/latest/).
    """)
    return


@app.cell(hide_code=True)
def _(JIT_IMG_SIZE, JIT_NOISE_SCALE, mo):
    # Fixed set of ImageNet classes and their trajectory color ramps.
    CLASSES = {
        207: "Golden Retriever",
        980: "Volcano",
        1: "Goldfish",
        388: "Giant Panda",
        985: "Daisy",
    }
    THUMB_EVERY = 5
    THUMB_SIZE = 64

    CLASS_COLORS = {
        207: ("#6c63f0", "#a78bfa", "#f0abfc"),
        980: ("#ef4444", "#f97316", "#fbbf24"),
        1: ("#06b6d4", "#22d3ee", "#a5f3fc"),
        388: ("#22c55e", "#4ade80", "#bbf7d0"),
        985: ("#f472b6", "#fb7185", "#fda4af"),
    }

    class Args:
        model = "JiT-B/16"
        img_size = JIT_IMG_SIZE
        class_num = 1000
        attn_dropout = 0.0
        proj_dropout = 0.0
        label_drop_prob = 0.1
        P_mean = -0.8
        P_std = 0.8
        t_eps = 1e-5
        noise_scale = JIT_NOISE_SCALE
        ema_decay1 = 0.999
        ema_decay2 = 0.9999
        sampling_method = "euler"
        num_sampling_steps = 50  # paper default solver-step count
        cfg = 5.0
        interval_min = 0.1
        interval_max = 1.0

    generate_btn = mo.ui.run_button(label="Generate trajectories")

    _class_list = ", ".join(CLASSES.values())

    mo.vstack(
        [
            mo.md(
                f"Samples denoising trajectories for **{len(CLASSES)} ImageNet classes** ({_class_list})"
            ),
            generate_btn,
        ],
        align="center",
    )
    return Args, CLASSES, CLASS_COLORS, THUMB_EVERY, THUMB_SIZE, generate_btn


@app.cell(hide_code=True)
def _(
    CLASSES,
    Image,
    THUMB_EVERY,
    THUMB_SIZE,
    denoiser,
    device,
    generate_btn,
    image_tensor,
    mo,
    model_path,
    np,
    torch,
    umap,
):
    mo.stop(
        not generate_btn.value,
        mo.md(
            "*Press **Generate trajectories** above to sample denoising "
            "trajectories using the already-loaded JiT model.*"
        ),
    )
    mo.stop(
        not torch.cuda.is_available(),
        mo.callout(
            "Part 2 runs the real JiT model, which requires a CUDA GPU. "
            "No CUDA device was found, so trajectory generation is skipped.",
            kind="warn",
        ),
    )
    mo.stop(
        model_path is None,
        mo.callout(
            "JiT-B/16 weights aren't available — either `huggingface_hub` isn't "
            "installed, or the startup download failed (see the console log "
            "above). Install `huggingface_hub` and reload the notebook.",
            kind="danger",
        ),
    )

    with mo.status.spinner(
        "Generating trajectories (this may take a minute)..."
    ) as _spinner:
        all_frames_flat = {}
        all_thumbs = {}
        all_sigmas = {}
        N_steps = 0

        # Sequentially generate classes to prevent CUDA OOM on smaller VRAM GPUs
        for class_id, class_name in CLASSES.items():
            _spinner.update(f"Generating trajectory for {class_name}...")

            history = []

            def hook(module, args_in):
                # Save just the single batch item to CPU to avoid VRAM hoarding
                history.append(args_in[0][0:1].detach().clone().cpu())

            handle = denoiser.net.register_forward_pre_hook(hook)

            labels = torch.tensor([class_id], device=device)
            with torch.no_grad():
                final = denoiser.generate(labels, z=image_tensor)

            handle.remove()
            history.append(final[0:1].cpu())

            if N_steps == 0:
                N_steps = len(history)

            # Flatten for UMAP
            flat = np.stack([h.flatten().numpy() for h in history])
            all_frames_flat[class_id] = flat

            # Extract thumbnails
            thumbs = []
            for t in history[::THUMB_EVERY]:
                arr = t[0].numpy().transpose(1, 2, 0)
                arr = ((arr + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(arr).resize(
                    (THUMB_SIZE, THUMB_SIZE), Image.BILINEAR
                )
                thumbs.append(np.array(img))
            all_thumbs[class_id] = np.stack(thumbs)

            # Remaining noise fraction at each captured step. Under the flow
            # schedule z_t = t*x + (1-t)*eps, the noise coefficient is (1-t),
            # so this runs a truthful 100% -> 0% across the trajectory.
            t_vals = np.linspace(0, 1, N_steps)
            all_sigmas[class_id] = 1 - t_vals

            # Clear the VRAM aggressively after each class
            torch.cuda.empty_cache()

        _spinner.update("Applying UMAP (2D) dimensionality reduction...")
        all_flat_concat = np.concatenate(list(all_frames_flat.values()), axis=0)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            random_state=42,
        )
        all_coords_2d = reducer.fit_transform(all_flat_concat)

        start_idx = 0
        data = {}
        for class_id in CLASSES:
            n_steps_class = all_frames_flat[class_id].shape[0]
            coords_2d = all_coords_2d[start_idx : start_idx + n_steps_class]
            start_idx += n_steps_class
            data[class_id] = (coords_2d, all_thumbs[class_id], all_sigmas[class_id])

    N = N_steps
    return N, data


@app.cell(hide_code=True)
def _(
    CLASSES,
    CLASS_COLORS,
    THUMB_EVERY,
    class_checkboxes,
    data,
    mo,
    np,
    step_slider,
    thumbs_b64,
):
    _step = step_slider.value
    _n_thumbs = data[list(CLASSES.keys())[0]][1].shape[0]
    _thumb_idx = min(_step // THUMB_EVERY, _n_thumbs - 1)

    cards = []
    for _cid in CLASSES:
        if CLASSES[_cid] not in class_checkboxes.value:
            continue

        _coords, _, _sigmas = data[_cid]
        _c0, _c1, _ = CLASS_COLORS[_cid]
        _name = CLASSES[_cid]
        _sigma = float(_sigmas[_step])
        _noise_pct = int(_sigma * 100)

        _dist_final = np.linalg.norm(_coords[_step] - _coords[-1])
        _dist_max = max(np.linalg.norm(_coords[0] - _coords[-1]), 1e-8)
        _progress = int((1 - _dist_final / _dist_max) * 100)

        _b64 = thumbs_b64[_cid][_thumb_idx]

        cards.append(
            mo.Html(f"""
        <div style="
            background:rgba(240,240,240,0.85);
            border:0.5px solid {_c0};
            border-radius:10px;
            padding:10px 12px;
            display:flex; align-items:center; gap:12px;
            min-width:230px;
        ">
            <img src="data:image/png;base64,{_b64}" width="72" height="72"
                 style="border-radius:6px; border:0.5px solid {_c0};
                        image-rendering:pixelated; flex-shrink:0;"
                 alt="{_name} step {_step}"/>
            <div style="flex:1; min-width:0;">
                <div style="margin-bottom:5px;">
                    <span style="font-size:12px; font-weight:500; color:{_c1};
                                 white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{_name}</span>
                </div>
                <div style="font-size:11px; color:rgba(160,150,210,0.7);
                            margin-bottom:5px; font-family:monospace;">σ = {_sigma:.3f}</div>
                <div style="margin-bottom:4px;">
                    <div style="font-size:10px; color:rgba(140,130,190,0.5); margin-bottom:2px;">noise {_noise_pct}%</div>
                    <div style="background:rgba(30,20,50,0.8); border-radius:3px; height:3px; overflow:hidden;">
                        <div style="width:{_noise_pct}%; height:100%; background:{_c0}; border-radius:3px;"></div>
                    </div>
                </div>
                <div>
                    <div style="font-size:10px; color:rgba(140,130,190,0.5); margin-bottom:2px;">progress (UMAP) {_progress}%</div>
                    <div style="background:rgba(30,20,50,0.8); border-radius:3px; height:3px; overflow:hidden;">
                        <div style="width:{_progress}%; height:100%; background:{_c1}; border-radius:3px;"></div>
                    </div>
                </div>
            </div>
        </div>
        """)
        )

    mo.hstack(cards, gap="0.75rem", wrap=True)
    return


@app.cell(hide_code=True)
def _(
    CLASSES,
    CLASS_COLORS,
    N,
    class_checkboxes,
    data,
    mo,
    np,
    plt,
    step_slider,
):
    step = step_slider.value
    visible_ids = [
        cid for cid, name in CLASSES.items() if name in class_checkboxes.value
    ]

    # Fixed axis limits from the full trajectory range across all classes, so the
    # view never shifts as the denoising step changes.
    _all_data = np.concatenate([data[c][0] for c in CLASSES], axis=0)
    _lim = float(np.abs(_all_data).max()) * 1.1

    _fig, _ax = plt.subplots(figsize=(7.2, 6.2))

    for _class_id in CLASSES:
        if _class_id not in visible_ids:
            continue

        _coords, _thumbs, _sigmas = data[_class_id]
        _c0, _c1, _c2 = CLASS_COLORS[_class_id]
        _name = CLASSES[_class_id]

        _ax.plot(
            _coords[:, 0],
            _coords[:, 1],
            color=_c0,
            linewidth=1,
            alpha=0.25,
            zorder=1,
        )

        _ax.scatter(
            [_coords[step, 0]],
            [_coords[step, 1]],
            s=140,
            color=_c2,
            edgecolors="white",
            linewidths=1.2,
            zorder=4,
        )

        _ax.plot([], [], color=_c1, marker="o", linestyle="-", label=_name, alpha=0.9)

    _ax.set_xlabel("UMAP 1")
    _ax.set_ylabel("UMAP 2")
    _ax.set_title(f"Step {step} / {N - 1}")
    _ax.grid(True, alpha=0.3)
    _ax.set_xlim(-_lim, _lim)
    _ax.set_ylim(-_lim, _lim)
    _ax.set_aspect("equal")
    if visible_ids:
        _ax.legend(loc="best", fontsize=8, framealpha=0.5)
    _fig.tight_layout()

    umap_html = mo.as_html(_fig)
    plt.close(_fig)

    mo.vstack(
        [
            step_slider,
            class_checkboxes,
            umap_html,
        ],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(CLASSES, CLASS_COLORS, N, data, mo, np, plt, speeds_all, step_slider):
    _step_v = step_slider.value

    # Shared x/y scaling so the two plots are directly comparable (both are
    # Euclidean distances in the same 2D UMAP embedding space).
    _all_speeds = np.concatenate([speeds_all[_cid] for _cid in CLASSES])
    _all_dists = np.concatenate(
        [np.linalg.norm(data[_cid][0] - data[_cid][0][-1], axis=1) for _cid in CLASSES]
    )
    _x_max = N - 1
    _y_max = max(_all_speeds.max(), _all_dists.max()) * 1.05

    fig_v, _ax_v = plt.subplots(figsize=(8, 2.5))
    for _cid, _cname in CLASSES.items():
        _, _c1, _ = CLASS_COLORS[_cid]
        _sp = speeds_all[_cid]
        _ax_v.plot(
            range(len(_sp)), _sp, color=_c1, linewidth=1.8, alpha=0.9, label=_cname
        )
    _ax_v.axvline(_step_v, color="black", linestyle=":", linewidth=1.2, alpha=0.6)
    _ax_v.set_title(r"Velocity $‖\Delta x_t‖$ in UMAP space", fontsize=10, loc="left")
    _ax_v.set_xlabel("Step", fontsize=9)
    _ax_v.set_ylabel(r"$‖\Delta x‖$", fontsize=9)
    _ax_v.set_xlim(0, _x_max)
    _ax_v.set_ylim(0, _y_max)
    _ax_v.grid(True, alpha=0.25)
    _ax_v.legend(fontsize=7, loc="upper right")
    fig_v.tight_layout()
    _velocity_html = mo.as_html(fig_v)
    plt.close(fig_v)

    fig_d, _ax_d = plt.subplots(figsize=(8, 2.5))
    for _cid, _cname in CLASSES.items():
        _c0, _c1, _ = CLASS_COLORS[_cid]
        _coords, _, _ = data[_cid]
        _dists = np.linalg.norm(_coords - _coords[-1], axis=1)
        _ax_d.plot(
            range(len(_dists)),
            _dists,
            color=_c1,
            linewidth=1.8,
            alpha=0.9,
            label=_cname,
        )
        _ax_d.fill_between(range(len(_dists)), _dists, color=_c0, alpha=0.12)
    _ax_d.axvline(_step_v, color="black", linestyle=":", linewidth=1.2, alpha=0.6)
    _ax_d.set_title(
        "Distance to final point $‖x_t − x_T‖$ in UMAP space", fontsize=10, loc="left"
    )
    _ax_d.set_xlabel("Step", fontsize=9)
    _ax_d.set_ylabel("$‖x_t − x_T‖$ UMAP", fontsize=9)
    _ax_d.set_xlim(0, _x_max)
    _ax_d.set_ylim(0, _y_max)
    _ax_d.grid(True, alpha=0.25)
    _ax_d.legend(fontsize=7, loc="upper right")
    fig_d.tight_layout()
    _distance_html = mo.as_html(fig_d)
    plt.close(fig_d)

    mo.vstack(
        [
            mo.md("""
    ### Velocity & Convergence in Embedded Space
    """),
            _velocity_html,
            _distance_html,
        ],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(CLASSES, N, THUMB_EVERY, class_checkboxes, mo, thumbs_b64):
    _max_display = 15

    _rows = []
    for _cid, _cname in CLASSES.items():
        if _cname not in class_checkboxes.value:
            continue

        _b64_list = thumbs_b64[_cid]
        _n_thumbs = len(_b64_list)
        _stride = max(1, _n_thumbs // _max_display)
        _disp_idxs = list(range(0, _n_thumbs, _stride))

        # Ensure the very last image is always included
        if _disp_idxs[-1] != _n_thumbs - 1:
            _disp_idxs.append(_n_thumbs - 1)

        _frames = []
        for _i, _ti in enumerate(_disp_idxs):
            _step_for_thumb = min(_ti * THUMB_EVERY, N - 1)

            # Force the first image to 0% and the last to 100%
            if _i == 0:
                _pct = 0
            elif _i == len(_disp_idxs) - 1:
                _pct = 100
            else:
                _pct = int((_step_for_thumb / (N - 1)) * 100)

            _frames.append(
                mo.image(
                    src=f"data:image/png;base64,{_b64_list[_ti]}",
                    width=100,
                    rounded=True,
                    caption=f"{_pct}%",
                )
            )

        _rows.append(
            mo.vstack(
                [
                    mo.md(f"**{_cname}**"),
                    mo.hstack(_frames, wrap=False),
                ],
                align="center",
            )
        )

    mo.vstack(
        [
            mo.md(
                """### Filmstrips
                Sampled sequences of frames from each class's denoising trajectory from pure noise to the final generated image.
                """
            ),
            mo.vstack(_rows, align="center"),
        ],
    )
    return


@app.cell(hide_code=True)
def _(nn, np, pi, rearrange, repeat, torch):
    # --- model_util.py ---
    def broadcat(tensors, dim=-1):
        num_tensors = len(tensors)
        shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
        assert len(shape_lens) == 1, (
            "tensors must all have the same number of dimensions"
        )
        shape_len = list(shape_lens)[0]
        dim = (dim + shape_len) if dim < 0 else dim
        dims = list(zip(*map(lambda t: list(t.shape), tensors)))
        expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
        assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), (
            "invalid dimensions for broadcastable concatentation"
        )
        max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
        expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
        expanded_dims.insert(dim, (dim, dims[dim]))
        expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
        tensors = list(
            map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes))
        )
        return torch.cat(tensors, dim=dim)

    def rotate_half(x):
        x = rearrange(x, "... (d r) -> ... d r", r=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, "... d r -> ... (d r)")

    class VisionRotaryEmbedding(nn.Module):
        def __init__(
            self,
            dim,
            pt_seq_len,
            ft_seq_len=None,
            custom_freqs=None,
            freqs_for="lang",
            theta=10000,
            max_freq=10,
            num_freqs=1,
        ):
            super().__init__()
            if custom_freqs:
                freqs = custom_freqs
            elif freqs_for == "lang":
                freqs = 1.0 / (
                    theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
                )
            elif freqs_for == "pixel":
                freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * torch.pi
            elif freqs_for == "constant":
                freqs = torch.ones(num_freqs).float()
            else:
                raise ValueError(f"unknown modality {freqs_for}")

            if ft_seq_len is None:
                ft_seq_len = pt_seq_len
            t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

            freqs_h = torch.einsum("..., f -> ... f", t, freqs)
            freqs_h = repeat(freqs_h, "... n -> ... (n r)", r=2)

            freqs_w = torch.einsum("..., f -> ... f", t, freqs)
            freqs_w = repeat(freqs_w, "... n -> ... (n r)", r=2)

            freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim=-1)

            self.register_buffer("freqs_cos", freqs.cos())
            self.register_buffer("freqs_sin", freqs.sin())

        def forward(self, t, start_index=0):
            rot_dim = self.freqs_cos.shape[-1]
            end_index = start_index + rot_dim
            t_left, t_mid, t_right = (
                t[..., :start_index],
                t[..., start_index:end_index],
                t[..., end_index:],
            )
            t_mid = (t_mid * self.freqs_cos) + (rotate_half(t_mid) * self.freqs_sin)
            return torch.cat((t_left, t_mid, t_right), dim=-1)

    class VisionRotaryEmbeddingFast(nn.Module):
        def __init__(
            self,
            dim,
            pt_seq_len=16,
            ft_seq_len=None,
            custom_freqs=None,
            freqs_for="lang",
            theta=10000,
            max_freq=10,
            num_freqs=1,
            num_cls_token=0,
        ):
            super().__init__()
            if custom_freqs:
                freqs = custom_freqs
            elif freqs_for == "lang":
                freqs = 1.0 / (
                    theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
                )
            elif freqs_for == "pixel":
                freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
            elif freqs_for == "constant":
                freqs = torch.ones(num_freqs).float()
            else:
                raise ValueError(f"unknown modality {freqs_for}")

            if ft_seq_len is None:
                ft_seq_len = pt_seq_len
            t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

            freqs = torch.einsum("..., f -> ... f", t, freqs)
            freqs = repeat(freqs, "... n -> ... (n r)", r=2)
            freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

            if num_cls_token > 0:
                freqs_flat = freqs.view(-1, freqs.shape[-1])
                cos_img = freqs_flat.cos()
                sin_img = freqs_flat.sin()

                N_img, D = cos_img.shape
                cos_pad = torch.ones(
                    num_cls_token, D, dtype=cos_img.dtype, device=cos_img.device
                )
                sin_pad = torch.zeros(
                    num_cls_token, D, dtype=sin_img.dtype, device=sin_img.device
                )

                self.freqs_cos = torch.cat([cos_pad, cos_img], dim=0).cuda()
                self.freqs_sin = torch.cat([sin_pad, sin_img], dim=0).cuda()
            else:
                self.freqs_cos = freqs.cos().view(-1, freqs.shape[-1]).cuda()
                self.freqs_sin = freqs.sin().view(-1, freqs.shape[-1]).cuda()

        def forward(self, t):
            return t * self.freqs_cos + rotate_half(t) * self.freqs_sin

    class RMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps

        def forward(self, hidden_states):
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(
                variance + self.variance_epsilon
            )
            return (self.weight * hidden_states).to(input_dtype)

    def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)
        grid = np.stack(grid, axis=0)
        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        if cls_token and extra_tokens > 0:
            pos_embed = np.concatenate(
                [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
            )
        return pos_embed

    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        emb = np.concatenate([emb_h, emb_w], axis=1)
        return emb

    def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float64)
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000**omega
        pos = pos.reshape(-1)
        out = np.einsum("m,d->md", pos, omega)
        emb_sin = np.sin(out)
        emb_cos = np.cos(out)
        emb = np.concatenate([emb_sin, emb_cos], axis=1)
        return emb

    return RMSNorm, VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed


@app.cell(hide_code=True)
def _(
    F,
    RMSNorm,
    VisionRotaryEmbeddingFast,
    get_2d_sincos_pos_embed,
    math,
    nn,
    torch,
):
    # --- model_jit.py ---
    def modulate(x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    class BottleneckPatchEmbed(nn.Module):
        def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            pca_dim=768,
            embed_dim=768,
            bias=True,
        ):
            super().__init__()
            img_size = (img_size, img_size)
            patch_size = (patch_size, patch_size)
            num_patches = (img_size[1] // patch_size[1]) * (
                img_size[0] // patch_size[0]
            )
            self.img_size = img_size
            self.patch_size = patch_size
            self.num_patches = num_patches

            self.proj1 = nn.Conv2d(
                in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False
            )
            self.proj2 = nn.Conv2d(
                pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias
            )

        def forward(self, x):
            B, C, H, W = x.shape
            assert H == self.img_size[0] and W == self.img_size[1]
            x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
            return x

    class TimestepEmbedder(nn.Module):
        def __init__(self, hidden_size, frequency_embedding_size=256):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(frequency_embedding_size, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
            )
            self.frequency_embedding_size = frequency_embedding_size

        @staticmethod
        def timestep_embedding(t, dim, max_period=10000):
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period)
                * torch.arange(start=0, end=half, dtype=torch.float32)
                / half
            ).to(device=t.device)
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat(
                    [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
                )
            return embedding

        def forward(self, t):
            t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
            t_emb = self.mlp(t_freq)
            return t_emb

    class LabelEmbedder(nn.Module):
        def __init__(self, num_classes, hidden_size):
            super().__init__()
            self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
            self.num_classes = num_classes

        def forward(self, labels):
            embeddings = self.embedding_table(labels)
            return embeddings

    def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1))
        attn_bias = torch.zeros(query.size(0), 1, L, S, dtype=query.dtype).cuda()

        with torch.cuda.amp.autocast(enabled=False):
            attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value

    class Attention(nn.Module):
        def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=0.0,
            proj_drop=0.0,
        ):
            super().__init__()
            self.num_heads = num_heads
            head_dim = dim // num_heads

            self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
            self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)

        def forward(self, x, rope):
            B, N, C = x.shape
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]

            q = self.q_norm(q)
            k = self.k_norm(k)
            q = rope(q)
            k = rope(k)

            x = scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0
            )
            x = x.transpose(1, 2).reshape(B, N, C)

            x = self.proj(x)
            x = self.proj_drop(x)
            return x

    class SwiGLUFFN(nn.Module):
        def __init__(self, dim: int, hidden_dim: int, drop=0.0, bias=True) -> None:
            super().__init__()
            hidden_dim = int(hidden_dim * 2 / 3)
            self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
            self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
            self.ffn_dropout = nn.Dropout(drop)

        def forward(self, x):
            x12 = self.w12(x)
            x1, x2 = x12.chunk(2, dim=-1)
            hidden = F.silu(x1) * x2
            return self.w3(self.ffn_dropout(hidden))

    class FinalLayer(nn.Module):
        def __init__(self, hidden_size, patch_size, out_channels):
            super().__init__()
            self.norm_final = RMSNorm(hidden_size)
            self.linear = nn.Linear(
                hidden_size, patch_size * patch_size * out_channels, bias=True
            )
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
            )

        @torch.compile
        def forward(self, x, c):
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
            x = modulate(self.norm_final(x), shift, scale)
            x = self.linear(x)
            return x

    class JiTBlock(nn.Module):
        def __init__(
            self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0
        ):
            super().__init__()
            self.norm1 = RMSNorm(hidden_size, eps=1e-6)
            self.attn = Attention(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                qk_norm=True,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )
            self.norm2 = RMSNorm(hidden_size, eps=1e-6)
            mlp_hidden_dim = int(hidden_size * mlp_ratio)
            self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )

        @torch.compile
        def forward(self, x, c, feat_rope=None):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(c).chunk(6, dim=-1)
            )
            x = x + gate_msa.unsqueeze(1) * self.attn(
                modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope
            )
            x = x + gate_mlp.unsqueeze(1) * self.mlp(
                modulate(self.norm2(x), shift_mlp, scale_mlp)
            )
            return x

    class JiT(nn.Module):
        def __init__(
            self,
            input_size=256,
            patch_size=16,
            in_channels=3,
            hidden_size=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4.0,
            attn_drop=0.0,
            proj_drop=0.0,
            num_classes=1000,
            bottleneck_dim=128,
            in_context_len=32,
            in_context_start=8,
        ):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = in_channels
            self.patch_size = patch_size
            self.num_heads = num_heads
            self.hidden_size = hidden_size
            self.input_size = input_size
            self.in_context_len = in_context_len
            self.in_context_start = in_context_start
            self.num_classes = num_classes

            self.t_embedder = TimestepEmbedder(hidden_size)
            self.y_embedder = LabelEmbedder(num_classes, hidden_size)
            self.x_embedder = BottleneckPatchEmbed(
                input_size,
                patch_size,
                in_channels,
                bottleneck_dim,
                hidden_size,
                bias=True,
            )

            num_patches = self.x_embedder.num_patches
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, hidden_size), requires_grad=False
            )

            if self.in_context_len > 0:
                self.in_context_posemb = nn.Parameter(
                    torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True
                )
                torch.nn.init.normal_(self.in_context_posemb, std=0.02)

            half_head_dim = hidden_size // num_heads // 2
            hw_seq_len = input_size // patch_size
            self.feat_rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=0
            )
            self.feat_rope_incontext = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=hw_seq_len,
                num_cls_token=self.in_context_len,
            )

            self.blocks = nn.ModuleList(
                [
                    JiTBlock(
                        hidden_size,
                        num_heads,
                        mlp_ratio=mlp_ratio,
                        attn_drop=attn_drop
                        if (depth // 4 * 3 > i >= depth // 4)
                        else 0.0,
                        proj_drop=proj_drop
                        if (depth // 4 * 3 > i >= depth // 4)
                        else 0.0,
                    )
                    for i in range(depth)
                ]
            )

            self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
            self.initialize_weights()

        def initialize_weights(self):
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

            self.apply(_basic_init)

            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

            w1 = self.x_embedder.proj1.weight.data
            nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
            w2 = self.x_embedder.proj2.weight.data
            nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
            nn.init.constant_(self.x_embedder.proj2.bias, 0)

            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.final_layer.linear.weight, 0)
            nn.init.constant_(self.final_layer.linear.bias, 0)

        def unpatchify(self, x, p):
            c = self.out_channels
            h = w = int(x.shape[1] ** 0.5)
            assert h * w == x.shape[1]
            x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
            x = torch.einsum("nhwpqc->nchpwq", x)
            imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
            return imgs

        def forward(self, x, t, y):
            t_emb = self.t_embedder(t)
            y_emb = self.y_embedder(y)
            c = t_emb + y_emb

            x = self.x_embedder(x)
            x += self.pos_embed

            for i, block in enumerate(self.blocks):
                if self.in_context_len > 0 and i == self.in_context_start:
                    in_context_tokens = y_emb.unsqueeze(1).repeat(
                        1, self.in_context_len, 1
                    )
                    in_context_tokens += self.in_context_posemb
                    x = torch.cat([in_context_tokens, x], dim=1)
                x = block(
                    x,
                    c,
                    self.feat_rope
                    if i < self.in_context_start
                    else self.feat_rope_incontext,
                )

            x = x[:, self.in_context_len :]
            x = self.final_layer(x, c)
            output = self.unpatchify(x, self.patch_size)
            return output

    def JiT_B_16(**kwargs):
        return JiT(
            depth=12,
            hidden_size=768,
            num_heads=12,
            bottleneck_dim=128,
            in_context_len=32,
            in_context_start=4,
            patch_size=16,
            **kwargs,
        )

    def JiT_B_32(**kwargs):
        return JiT(
            depth=12,
            hidden_size=768,
            num_heads=12,
            bottleneck_dim=128,
            in_context_len=32,
            in_context_start=4,
            patch_size=32,
            **kwargs,
        )

    def JiT_L_16(**kwargs):
        return JiT(
            depth=24,
            hidden_size=1024,
            num_heads=16,
            bottleneck_dim=128,
            in_context_len=32,
            in_context_start=8,
            patch_size=16,
            **kwargs,
        )

    def JiT_L_32(**kwargs):
        return JiT(
            depth=24,
            hidden_size=1024,
            num_heads=16,
            bottleneck_dim=128,
            in_context_len=32,
            in_context_start=8,
            patch_size=32,
            **kwargs,
        )

    def JiT_H_16(**kwargs):
        return JiT(
            depth=32,
            hidden_size=1280,
            num_heads=16,
            bottleneck_dim=256,
            in_context_len=32,
            in_context_start=10,
            patch_size=16,
            **kwargs,
        )

    def JiT_H_32(**kwargs):
        return JiT(
            depth=32,
            hidden_size=1280,
            num_heads=16,
            bottleneck_dim=256,
            in_context_len=32,
            in_context_start=10,
            patch_size=32,
            **kwargs,
        )

    JiT_models = {
        "JiT-B/16": JiT_B_16,
        "JiT-B/32": JiT_B_32,
        "JiT-L/16": JiT_L_16,
        "JiT-L/32": JiT_L_32,
        "JiT-H/16": JiT_H_16,
        "JiT-H/32": JiT_H_32,
    }
    return JiT_models, TimestepEmbedder


@app.cell(hide_code=True)
def _(JiT_models, nn, torch):
    # --- denoiser.py ---
    class Denoiser(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.net = JiT_models[args.model](
                input_size=args.img_size,
                in_channels=3,
                num_classes=args.class_num,
                attn_drop=args.attn_dropout,
                proj_drop=args.proj_dropout,
            )
            self.img_size = args.img_size
            self.num_classes = args.class_num

            self.label_drop_prob = args.label_drop_prob
            self.P_mean = args.P_mean
            self.P_std = args.P_std
            self.t_eps = args.t_eps
            self.noise_scale = args.noise_scale

            self.ema_decay1 = args.ema_decay1
            self.ema_decay2 = args.ema_decay2
            self.ema_params1 = None
            self.ema_params2 = None

            self.method = args.sampling_method
            self.steps = args.num_sampling_steps
            self.cfg_scale = args.cfg
            self.cfg_interval = (args.interval_min, args.interval_max)

        def drop_labels(self, labels):
            drop = (
                torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
            )
            out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
            return out

        def sample_t(self, n: int, device=None):
            z = torch.randn(n, device=device) * self.P_std + self.P_mean
            return torch.sigmoid(z)

        def forward(self, x, labels):
            labels_dropped = self.drop_labels(labels) if self.training else labels

            t = self.sample_t(x.size(0), device=x.device).view(
                -1, *([1] * (x.ndim - 1))
            )
            e = torch.randn_like(x) * self.noise_scale

            z = t * x + (1 - t) * e
            v = (x - z) / (1 - t).clamp_min(self.t_eps)

            x_pred = self.net(z, t.flatten(), labels_dropped)
            v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

            loss = (v - v_pred) ** 2
            loss = loss.mean(dim=(1, 2, 3)).mean()

            return loss

        @torch.no_grad()
        def generate(self, labels, z=None):
            device = labels.device
            bsz = labels.size(0)
            if z is None:
                z = self.noise_scale * torch.randn(
                    bsz, 3, self.img_size, self.img_size, device=device
                )
            else:
                z = z.to(device)
                if z.size(0) == 1 and bsz > 1:
                    z = z.expand(bsz, -1, -1, -1).contiguous()
            timesteps = (
                torch.linspace(0.0, 1.0, self.steps + 1, device=device)
                .view(-1, *([1] * z.ndim))
                .expand(-1, bsz, -1, -1, -1)
            )

            if self.method == "euler":
                stepper = self._euler_step
            elif self.method == "heun":
                stepper = self._heun_step
            else:
                raise NotImplementedError

            for i in range(self.steps - 1):
                t = timesteps[i]
                t_next = timesteps[i + 1]
                z = stepper(z, t, t_next, labels)
            z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
            return z

        @torch.no_grad()
        def _forward_sample(self, z, t, labels):
            x_cond = self.net(z, t.flatten(), labels)
            v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

            x_uncond = self.net(
                z, t.flatten(), torch.full_like(labels, self.num_classes)
            )
            v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

            low, high = self.cfg_interval
            interval_mask = (t < high) & ((low == 0) | (t > low))
            cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

            return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

        @torch.no_grad()
        def _euler_step(self, z, t, t_next, labels):
            v_pred = self._forward_sample(z, t, labels)
            z_next = z + (t_next - t) * v_pred
            return z_next

        @torch.no_grad()
        def _heun_step(self, z, t, t_next, labels):
            v_pred_t = self._forward_sample(z, t, labels)
            z_next_euler = z + (t_next - t) * v_pred_t
            v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)

            v_pred = 0.5 * (v_pred_t + v_pred_t_next)
            z_next = z + (t_next - t) * v_pred
            return z_next

    return (Denoiser,)


@app.cell(hide_code=True)
def load_denoiser(Args, Denoiser, mo, model_path, torch):
    mo.stop(
        not torch.cuda.is_available(),
        mo.callout(
            "Part 2 runs the real JiT model, which requires a CUDA GPU. "
            "No CUDA device was found, so the model won't be loaded.",
            kind="warn",
        ),
    )
    mo.stop(
        model_path is None,
        mo.callout(
            "JiT-B/16 weights aren't available — either `huggingface_hub` isn't "
            "installed, or the startup download failed (see the console log "
            "above). Install `huggingface_hub` and reload the notebook.",
            kind="danger",
        ),
    )

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    args = Args()
    denoiser = Denoiser(args).to(device)

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model", checkpoint)
    cleaned = {k[4:] if k.startswith("net.") else k: v for k, v in state_dict.items()}
    denoiser.net.load_state_dict(cleaned, strict=True)
    denoiser.eval()
    return denoiser, device


@app.cell(hide_code=True)
def _(BytesIO, CLASSES, Image, base64, data, np):
    def compute_speeds(coords):
        return np.linalg.norm(np.diff(coords, axis=0), axis=1)

    speeds_all = {cid: compute_speeds(data[cid][0]) for cid in CLASSES}

    def arr_to_b64(arr):
        buf = BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    thumbs_b64 = {}
    for _cid in CLASSES:
        _, _thumbs, _ = data[_cid]
        thumbs_b64[_cid] = [arr_to_b64(_thumbs[i]) for i in range(_thumbs.shape[0])]
    return speeds_all, thumbs_b64


@app.cell(hide_code=True)
def _(N, mo):
    get_step, set_step = mo.state(0)

    step_slider = mo.ui.slider(
        0,
        N - 1,
        value=get_step(),
        on_change=set_step,
        label="Denoising step",
        full_width=False,
    )
    return (step_slider,)


@app.cell(hide_code=True)
def _(CLASSES, mo):
    class_checkboxes = mo.ui.multiselect(
        options=list(CLASSES.values()),
        value=list(CLASSES.values()),
        label="Visible classes",
    )
    return (class_checkboxes,)


if __name__ == "__main__":
    app.run()
