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
# ]
# ///

import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium", app_title="Diffusion model toy examples")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import optax
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_swiss_roll, make_moons, make_circles

    return (
        eqx,
        jax,
        jnp,
        make_circles,
        make_moons,
        make_swiss_roll,
        mo,
        np,
        optax,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Back to Basics: Let Denoising Generative Models Denoise

    From Li & He, 2025 ([arXiv:2511.13720](https://arxiv.org/abs/2511.13720))

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
def _(mo):
    dataset_dropdown = mo.ui.dropdown(
        options={
            "Swiss Roll": "swiss_roll",
            "Two Moons": "two_moons",
            "Concentric Circles": "circles",
            "Torus": "torus",
            "Möbius Strip": "mobius",
            "Trefoil Knot": "trefoil",
            "Helix": "helix",
            "Double Helix": "double_helix",
            "Catenoid": "catenoid",
            "Saddle": "saddle",
            "Gabriel's Horn": "horn",
            "Clifford Torus": "clifford",
            "Klein Bottle": "klein",
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
    noise_scale = mo.ui.slider(
        start=0.05,
        stop=1.0,
        step=0.01,
        value=0.3,
        label="Noise scale $\sigma$",
        show_value=True,
    )

    mo.vstack(
        [
            mo.md("## Choose dataset"),
            mo.hstack([dataset_dropdown, n_samples, seed]),
        ],
        align="center",
    )
    return circles_factor, dataset_dropdown, n_samples, noise_scale, seed


@app.cell(hide_code=True)
def _(D_values_all, dataset_dropdown, mo):
    _3d_datasets = {
        "torus",
        "mobius",
        "trefoil",
        "helix",
        "double_helix",
        "catenoid",
        "saddle",
        "horn",
        "clifford",
        "klein",
    }
    data_dim = 3 if dataset_dropdown.value in _3d_datasets else 2
    D_values = [D for D in D_values_all if D >= data_dim]
    noise_d_dropdown = mo.ui.dropdown(
        options={str(value): value for value in D_values},
        value=str(D_values[0]),
        label="Ambient dimension",
    )
    return D_values, data_dim, noise_d_dropdown


@app.cell(hide_code=True)
def _(dataset_dropdown, mo, np):
    _ds = dataset_dropdown.value
    if _ds == "torus":
        manifold_params = mo.ui.dictionary(
            {
                "R": mo.ui.slider(
                    start=0.5,
                    stop=3.0,
                    step=0.1,
                    value=1.5,
                    label="Major radius R",
                    show_value=True,
                ),
                "r": mo.ui.slider(
                    start=0.1,
                    stop=1.5,
                    step=0.1,
                    value=0.5,
                    label="Minor radius r",
                    show_value=True,
                ),
            }
        )
    elif _ds == "mobius":
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
    elif _ds == "helix":
        manifold_params = mo.ui.dictionary(
            {
                "turns": mo.ui.slider(
                    start=1, stop=10, step=1, value=4, label="Turns", show_value=True
                ),
                "radius": mo.ui.slider(
                    start=0.3,
                    stop=2.0,
                    step=0.1,
                    value=1.0,
                    label="Radius",
                    show_value=True,
                ),
                "pitch": mo.ui.slider(
                    start=0.2,
                    stop=2.0,
                    step=0.1,
                    value=0.5,
                    label="Pitch",
                    show_value=True,
                ),
            }
        )
    elif _ds == "double_helix":
        manifold_params = mo.ui.dictionary(
            {
                "turns": mo.ui.slider(
                    start=1, stop=10, step=1, value=4, label="Turns", show_value=True
                ),
                "radius": mo.ui.slider(
                    start=0.3,
                    stop=2.0,
                    step=0.1,
                    value=1.0,
                    label="Radius",
                    show_value=True,
                ),
                "pitch": mo.ui.slider(
                    start=0.2,
                    stop=2.0,
                    step=0.1,
                    value=0.5,
                    label="Pitch",
                    show_value=True,
                ),
            }
        )
    elif _ds == "catenoid":
        manifold_params = mo.ui.dictionary(
            {
                "height": mo.ui.slider(
                    start=0.5,
                    stop=3.0,
                    step=0.1,
                    value=1.5,
                    label="Height",
                    show_value=True,
                ),
            }
        )
    elif _ds == "saddle":
        manifold_params = mo.ui.dictionary(
            {
                "extent": mo.ui.slider(
                    start=1.0,
                    stop=5.0,
                    step=0.5,
                    value=2.0,
                    label="Extent",
                    show_value=True,
                ),
                "curvature": mo.ui.slider(
                    start=0.1,
                    stop=2.0,
                    step=0.1,
                    value=0.5,
                    label="Curvature",
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
    elif _ds == "clifford":
        manifold_params = mo.ui.dictionary(
            {
                "r1": mo.ui.slider(
                    start=0.3,
                    stop=2.0,
                    step=0.1,
                    value=1.0,
                    label="Radius r₁",
                    show_value=True,
                ),
                "r2": mo.ui.slider(
                    start=0.3,
                    stop=2.0,
                    step=0.1,
                    value=1.0,
                    label="Radius r₂",
                    show_value=True,
                ),
                "angle": mo.ui.slider(
                    start=0.0,
                    stop=round(float(np.pi), 2),
                    step=0.05,
                    value=round(float(np.pi / 4), 2),
                    label="4D rotation angle",
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
        manifold_params = mo.ui.dictionary({})
    return (manifold_params,)


@app.cell(hide_code=True)
def _(
    circles_factor,
    colors,
    data_dim,
    data_preview,
    dataset_dropdown,
    dataset_noise,
    manifold_params,
    mo,
    noise_d_dropdown,
    noise_scale,
    noised_cache,
    np,
    plt,
    time_slider,
):
    _controls = [dataset_noise]
    _fig_size = (4, 4)
    if dataset_dropdown.value == "circles":
        _controls.append(circles_factor)
    if manifold_params.value:
        _controls.append(manifold_params)

    _D = noise_d_dropdown.value
    _t = round(time_slider.value, 4)
    _z_t = noised_cache[(_D, _t)]

    _lim_clean = np.abs(data_preview).max() * 1.2
    _lim_noisy = np.abs(_z_t).max() * 1.2

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
    _fig_clean.tight_layout()
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
def _(D_values, mo):
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

    D_values_str = ", ".join(f"{D}" for D in D_values)
    # Trains {len(D_values) * len(pred_types)} models: {len(D_values)} ambient dimension{"" if len(D_values) < 2 else "s"} x {len(pred_types)} prediction types.
    mo.vstack(
        [
            mo.md(
                f"""
            ## Run the experiment

            For each prediction type ($x$, $\epsilon$ and $v$) we will train one model for each of the {D_values_str} ambient dimensions.

            ---
            """
            ),
            mo.hstack(
                [
                    hidden_width,
                    t_eps,
                    solver,
                ]
            ),
            train_btn,
        ],
        align="center",
    )
    return hidden_width, solver, t_eps, train_btn


@app.cell(hide_code=True)
def _(
    D_values,
    data_dim,
    data_preview,
    hidden_width,
    jax,
    make_projection,
    master_key,
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
        len(D_values), 1, figsize=(12, 3 * len(D_values)), squeeze=False, sharex=True
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
        _ax.set_xlim(0, None)
        _ax.set_yscale("log")
        _ax.set_ylabel("v-loss")
        _ax.legend(loc="upper right")
        if _i == len(D_values) - 1:
            _ax.set_xlabel("Step (x50)")
    # _handles, _labels = axes_loss[0, 0].get_legend_handles_labels()
    # fig_loss.legend(_handles, _labels, loc="right")
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
        figsize=(4 * 4, 4 * len(D_values)),
        squeeze=False,
        subplot_kw=_subplot_kw,
    )

    _rng = np.random.default_rng(0)
    _idx = _rng.choice(
        len(data_preview), size=min(_n, len(data_preview)), replace=False
    )
    gt_subset = data_preview[_idx]
    lim = max(np.abs(data_preview).max() * 1.3, 3.0)

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
    colors,
    data_dim,
    data_preview,
    dataset_dropdown,
    dataset_noise,
    jax,
    jnp,
    manifold_params,
    mo,
    np,
    plt,
    pred_labels,
    projections,
    results,
    t_eps,
    traj_D,
    traj_batch,
    traj_clean_pts,
    traj_focus,
    traj_focus_idx,
    traj_noise_t,
    traj_paths,
    traj_show_vf,
    traj_view_t,
):
    _n_pts = len(traj_clean_pts)
    _n_steps = len(list(traj_paths.values())[0]) - 1
    _step = min(int(traj_view_t.value * _n_steps), _n_steps)
    _D = traj_D.value
    _t_start = traj_noise_t.value
    _t_actual = _t_start + traj_view_t.value * (1.0 - _t_start)
    _focus_on = traj_focus.value
    _focus_i = min(int(traj_focus_idx.value), _n_pts - 1)
    _is_3d = data_dim == 3
    _ds = dataset_dropdown.value
    _mp = manifold_params.value
    _show_vf = traj_show_vf.value and not _is_3d

    def _noisy_stats(_ds, _mp, _noise, _n=10000):
        """Get raw mean/std from a noisy sample for proper mesh alignment."""
        _rng = np.random.default_rng(0)
        _3d_surfaces = {
            "torus",
            "mobius",
            "catenoid",
            "saddle",
            "horn",
            "clifford",
            "klein",
        }
        _3d_curves = {"trefoil", "helix", "double_helix"}
        if _ds == "torus":
            _u, _v = (
                _rng.uniform(0, 2 * np.pi, _n),
                _rng.uniform(0, 2 * np.pi, _n),
            )
            _R, _r = _mp.get("R", 1.5), _mp.get("r", 0.5)
            _raw = np.column_stack(
                [
                    (_R + _r * np.cos(_v)) * np.cos(_u),
                    (_R + _r * np.cos(_v)) * np.sin(_u),
                    _r * np.sin(_v),
                ]
            )
        elif _ds == "mobius":
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
        elif _ds == "catenoid":
            _ch = _mp.get("height", 1.5)
            _u, _th = (
                _rng.uniform(-_ch, _ch, _n),
                _rng.uniform(0, 2 * np.pi, _n),
            )
            _raw = np.column_stack(
                [np.cosh(_u) * np.cos(_th), np.cosh(_u) * np.sin(_th), _u]
            )
        elif _ds == "saddle":
            _ext = _mp.get("extent", 2.0)
            _k = _mp.get("curvature", 0.5)
            _u, _v = (
                _rng.uniform(-_ext, _ext, _n),
                _rng.uniform(-_ext, _ext, _n),
            )
            _raw = np.column_stack([_u, _v, _k * (_u**2 - _v**2)])
        elif _ds == "horn":
            _length = _mp.get("length", 8.0)
            _t, _th = (
                _rng.uniform(1, _length, _n),
                _rng.uniform(0, 2 * np.pi, _n),
            )
            _raw = np.column_stack([_t, (1 / _t) * np.cos(_th), (1 / _t) * np.sin(_th)])
        elif _ds == "clifford":
            _r1, _r2, _a = (
                _mp.get("r1", 1.0),
                _mp.get("r2", 1.0),
                _mp.get("angle", np.pi / 4),
            )
            _u, _v = (
                _rng.uniform(0, 2 * np.pi, _n),
                _rng.uniform(0, 2 * np.pi, _n),
            )
            _x4, _y4, _z4, _w4 = (
                _r1 * np.cos(_u),
                _r1 * np.sin(_u),
                _r2 * np.cos(_v),
                _r2 * np.sin(_v),
            )
            _xr = _x4 * np.cos(_a) - _w4 * np.sin(_a)
            _wr = _x4 * np.sin(_a) + _w4 * np.cos(_a)
            _d = 2.5 - _wr
            _raw = np.column_stack([_xr / _d, _y4 / _d, _z4 / _d])
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
        elif _ds == "trefoil":
            _t = _rng.uniform(0, 2 * np.pi, _n)
            _raw = np.column_stack(
                [
                    np.sin(_t) + 2 * np.sin(2 * _t),
                    np.cos(_t) - 2 * np.cos(2 * _t),
                    -np.sin(3 * _t),
                ]
            )
        elif _ds == "helix":
            _R, _p, _turns = (
                _mp.get("radius", 1.0),
                _mp.get("pitch", 0.5),
                _mp.get("turns", 4),
            )
            _t = _rng.uniform(0, 2 * np.pi * _turns, _n)
            _raw = np.column_stack(
                [_R * np.cos(_t), _R * np.sin(_t), _p * _t / (2 * np.pi)]
            )
        elif _ds == "double_helix":
            _R, _p, _turns = (
                _mp.get("radius", 1.0),
                _mp.get("pitch", 0.5),
                _mp.get("turns", 4),
            )
            _half = _n // 2
            _t1, _t2 = (
                _rng.uniform(0, 2 * np.pi * _turns, _half),
                _rng.uniform(0, 2 * np.pi * _turns, _n - _half),
            )
            _raw = np.column_stack(
                [
                    np.concatenate([_R * np.cos(_t1), _R * np.cos(_t2 + np.pi)]),
                    np.concatenate([_R * np.sin(_t1), _R * np.sin(_t2 + np.pi)]),
                    np.concatenate([_p * _t1 / (2 * np.pi), _p * _t2 / (2 * np.pi)]),
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

    def _standardize_curve(_pts):
        if _mesh_stats[0] is None:
            return _pts
        _mean, _std = _mesh_stats
        return (_pts - _mean) / _std

    def _make_surface(_ds, _mp, _res=40):
        if _ds == "torus":
            _uu, _vv = np.meshgrid(
                np.linspace(0, 2 * np.pi, _res),
                np.linspace(0, 2 * np.pi, _res),
            )
            _R, _r = _mp.get("R", 1.5), _mp.get("r", 0.5)
            _X = (_R + _r * np.cos(_vv)) * np.cos(_uu)
            _Y = (_R + _r * np.cos(_vv)) * np.sin(_uu)
            _Z = _r * np.sin(_vv)
        elif _ds == "mobius":
            _sw = _mp.get("strip_width", 0.5)
            _uu, _ww = np.meshgrid(
                np.linspace(0, 2 * np.pi, _res),
                np.linspace(-_sw, _sw, max(_res // 2, 10)),
            )
            _k = _mp.get("n_twists", 1)
            _X = (1 + _ww * np.cos(_k * _uu / 2)) * np.cos(_uu)
            _Y = (1 + _ww * np.cos(_k * _uu / 2)) * np.sin(_uu)
            _Z = _ww * np.sin(_k * _uu / 2)
        elif _ds == "catenoid":
            _ch = _mp.get("height", 1.5)
            _uu, _th = np.meshgrid(
                np.linspace(-_ch, _ch, _res), np.linspace(0, 2 * np.pi, _res)
            )
            _X = np.cosh(_uu) * np.cos(_th)
            _Y = np.cosh(_uu) * np.sin(_th)
            _Z = _uu
        elif _ds == "saddle":
            _ext = _mp.get("extent", 2.0)
            _k = _mp.get("curvature", 0.5)
            _uu, _vv = np.meshgrid(
                np.linspace(-_ext, _ext, _res), np.linspace(-_ext, _ext, _res)
            )
            _X, _Y = _uu, _vv
            _Z = _k * (_uu**2 - _vv**2)
        elif _ds == "horn":
            _length = _mp.get("length", 8.0)
            _tt, _th = np.meshgrid(
                np.linspace(1, _length, _res), np.linspace(0, 2 * np.pi, _res)
            )
            _X = _tt
            _Y = (1 / _tt) * np.cos(_th)
            _Z = (1 / _tt) * np.sin(_th)
        elif _ds == "clifford":
            _r1, _r2 = _mp.get("r1", 1.0), _mp.get("r2", 1.0)
            _a = _mp.get("angle", np.pi / 4)
            _uu, _vv = np.meshgrid(
                np.linspace(0, 2 * np.pi, _res),
                np.linspace(0, 2 * np.pi, _res),
            )
            _x4, _y4 = _r1 * np.cos(_uu), _r1 * np.sin(_uu)
            _z4, _w4 = _r2 * np.cos(_vv), _r2 * np.sin(_vv)
            _xr = _x4 * np.cos(_a) - _w4 * np.sin(_a)
            _wr = _x4 * np.sin(_a) + _w4 * np.cos(_a)
            _d = 2.5 - _wr
            _X, _Y, _Z = _xr / _d, _y4 / _d, _z4 / _d
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

    def _make_curve(_ds, _mp):
        if _ds == "trefoil":
            _t = np.linspace(0, 2 * np.pi, 500)
            _pts = np.column_stack(
                [
                    np.sin(_t) + 2 * np.sin(2 * _t),
                    np.cos(_t) - 2 * np.cos(2 * _t),
                    -np.sin(3 * _t),
                ]
            )
        elif _ds == "helix":
            _R, _p, _turns = (
                _mp.get("radius", 1.0),
                _mp.get("pitch", 0.5),
                _mp.get("turns", 4),
            )
            _t = np.linspace(0, 2 * np.pi * _turns, 500)
            _pts = np.column_stack(
                [_R * np.cos(_t), _R * np.sin(_t), _p * _t / (2 * np.pi)]
            )
        elif _ds == "double_helix":
            _R, _p, _turns = (
                _mp.get("radius", 1.0),
                _mp.get("pitch", 0.5),
                _mp.get("turns", 4),
            )
            _t = np.linspace(0, 2 * np.pi * _turns, 300)
            _h1 = np.column_stack(
                [_R * np.cos(_t), _R * np.sin(_t), _p * _t / (2 * np.pi)]
            )
            _h2 = np.column_stack(
                [
                    _R * np.cos(_t + np.pi),
                    _R * np.sin(_t + np.pi),
                    _p * _t / (2 * np.pi),
                ]
            )
            _pts = np.vstack([_h1, _h2])
        else:
            return None
        return _standardize_curve(_pts)

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
        else:
            _curve = _make_curve(_ds, _mp)
            if _curve is not None:
                _plot_line(
                    _ax,
                    _curve,
                    color="lavender",
                    linewidth=3,
                    alpha=0.6,
                    zorder=0,
                )

    _scatter(_ax, data_preview, s=1, alpha=0.05, c=colors["ground_truth"], zorder=1)

    # Velocity field (2D only)
    if _show_vf:
        _vf_res = 15
        _P_jnp = jnp.array(projections[_D])
        _te = t_eps.value
        _xs = np.linspace(-_lim, _lim, _vf_res)
        _ys = np.linspace(-_lim, _lim, _vf_res)
        _xx, _yy = np.meshgrid(_xs, _ys)
        _grid_2d = np.stack([_xx.ravel(), _yy.ravel()], axis=-1).astype(np.float32)
        _z_grid_D = jnp.array(_grid_2d) @ _P_jnp.T
        _t_arr_vf = jnp.full((_grid_2d.shape[0], 1), _t_actual)

        _vf_max = 1e-8
        _vf_data = {}
        for _pred_type in pred_labels:
            _model = results[(_D, _pred_type)]["model"]
            _pred = jax.vmap(_model)(_z_grid_D, _t_arr_vf)
            if _pred_type == "x_pred":
                _v_D = (_pred - _z_grid_D) / jnp.maximum(1 - _t_actual, _te)
            elif _pred_type == "eps_pred":
                _v_D = (_z_grid_D - _pred) / jnp.maximum(_t_actual, _te)
            else:
                _v_D = _pred
            _v_2d = np.asarray(_v_D @ _P_jnp)
            _vf_data[_pred_type] = _v_2d
            _vf_max = max(_vf_max, np.sqrt((_v_2d**2).sum(axis=1)).max())

        from matplotlib.colors import LinearSegmentedColormap, Normalize
        from matplotlib.cm import ScalarMappable

        _vf_cmaps = {
            "x_pred": LinearSegmentedColormap.from_list(
                "x_vf", ["darkred", "red", "yellow"]
            ),
            "eps_pred": LinearSegmentedColormap.from_list(
                "eps_vf", ["darkblue", "blue", "cyan"]
            ),
            "v_pred": LinearSegmentedColormap.from_list(
                "v_vf", ["darkgreen", "green", "lime"]
            ),
        }

        for _pred_type in pred_labels:
            _v_2d = _vf_data[_pred_type]
            _u_arr = _v_2d[:, 0].reshape(_vf_res, _vf_res)
            _v_arr = _v_2d[:, 1].reshape(_vf_res, _vf_res)
            _mag = np.sqrt(_u_arr**2 + _v_arr**2)
            _dir_u = _u_arr / np.maximum(_mag, 1e-8)
            _dir_v = _v_arr / np.maximum(_mag, 1e-8)
            _mag_norm = _mag / _vf_max
            _ax.quiver(
                _xx,
                _yy,
                _dir_u,
                _dir_v,
                _mag_norm,
                cmap=_vf_cmaps[_pred_type],
                clim=(0, 1),
                scale=_vf_res * 1.2,
                width=0.004,
                zorder=2,
            )

        for _pred_type, _pred_label in pred_labels.items():
            _sm = ScalarMappable(cmap=_vf_cmaps[_pred_type], norm=Normalize(0, 1))
            _cbar = _fig.colorbar(
                _sm, ax=_ax, fraction=0.015, pad=0.01, shrink=0.3, aspect=10
            )
            _cbar.set_label(_pred_label, fontsize=9)
            _cbar.ax.tick_params(labelsize=7)

    from matplotlib.lines import Line2D

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
    _title = f"## Denoising trajectories from $t_0={_t_start:.2f}$ to $t={_t_actual:.2f}$ in $D={_D}$"

    _controls = [
        mo.hstack([traj_D, traj_batch]),
        mo.hstack([traj_noise_t, traj_view_t]),
        mo.hstack(_row3),
    ]
    if not _is_3d:
        _controls.append(traj_show_vf)

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
        if _title:
            _string = _title.split("$")[1]
            _title = f"$\mathbf{{{_string}}}$-prediction"
        if _ylabel:
            _string = _ylabel.split("$")[1]
            _ylabel = f"$\mathbf{{{_string}}}$"

        _ax.scatter(
            data_preview[:, 0],
            data_preview[:, 1],
            s=1,
            alpha=0.1,
            c=colors["ground_truth"],
            zorder=1,
        )
        if _title:
            _ax.set_title(_title, fontsize=20)
        if _ylabel:
            _ax.set_ylabel(_ylabel, fontsize=20, rotation=0, labelpad=15)
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

        Rows: derived quantity ($\hat{x}$, $\hat{\epsilon}$, $\hat{v}$). Columns: network prediction type.<br>
        $\hat{x}$ shown as scatter points, $\hat{\epsilon}$ and $\hat{v}$ as unit vector fields colored by magnitude.

        ---
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
    data_dim,
    data_preview,
    jax,
    jnp,
    mo,
    n_sample_steps,
    np,
    pred_labels,
    results,
    t_eps,
    vf_grid_res,
):
    mo.stop(data_dim > 2)
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
    gen_samples,
    generate_samples,
    jax,
    master_key,
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
                solver=solver.value,
                t_eps=t_eps.value,
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
        "torus": (0.0, 0.5, 0.01, 0.1),
        "mobius": (0.0, 0.5, 0.01, 0.1),
        "trefoil": (0.0, 1.0, 0.01, 0.2),
        "helix": (0.0, 0.5, 0.01, 0.1),
        "double_helix": (0.0, 0.5, 0.01, 0.1),
        "catenoid": (0.0, 0.5, 0.01, 0.1),
        "saddle": (0.0, 1.0, 0.01, 0.2),
        "horn": (0.0, 0.5, 0.01, 0.05),
        "clifford": (0.0, 0.3, 0.01, 0.05),
        "klein": (0.0, 5.0, 0.1, 1.0),
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
        circles_factor=circles_factor.value,
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
def _(eqx, jax, jnp, make_circles, make_moons, make_swiss_roll, np, optax):
    def generate_data(
        name,
        n_samples=10000,
        seed=42,
        noise=0.5,
        circles_factor=0.5,
        manifold_params=None,
    ):
        rng = np.random.default_rng(seed)
        _mp = manifold_params or {}
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
        elif name == "torus":
            u = rng.uniform(0, 2 * np.pi, n_samples)
            v = rng.uniform(0, 2 * np.pi, n_samples)
            R = _mp.get("R", 1.5)
            r_t = _mp.get("r", 0.5)
            x = (R + r_t * np.cos(v)) * np.cos(u)
            y = (R + r_t * np.cos(v)) * np.sin(u)
            z = r_t * np.sin(v)
            data = np.column_stack([x, y, z])
            data += rng.standard_normal(data.shape) * noise
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
        elif name == "trefoil":
            t = rng.uniform(0, 2 * np.pi, n_samples)
            x = np.sin(t) + 2 * np.sin(2 * t)
            y = np.cos(t) - 2 * np.cos(2 * t)
            z = -np.sin(3 * t)
            data = np.column_stack([x, y, z])
            data += rng.standard_normal(data.shape) * noise
        elif name == "helix":
            turns = _mp.get("turns", 4)
            R = _mp.get("radius", 1.0)
            p = _mp.get("pitch", 0.5)
            t = rng.uniform(0, 2 * np.pi * turns, n_samples)
            x = R * np.cos(t)
            y = R * np.sin(t)
            z = p * t / (2 * np.pi)
            data = np.column_stack([x, y, z])
            data += rng.standard_normal(data.shape) * noise
        elif name == "double_helix":
            turns = _mp.get("turns", 4)
            R = _mp.get("radius", 1.0)
            p = _mp.get("pitch", 0.5)
            half = n_samples // 2
            t1 = rng.uniform(0, 2 * np.pi * turns, half)
            t2 = rng.uniform(0, 2 * np.pi * turns, n_samples - half)
            x = np.concatenate([R * np.cos(t1), R * np.cos(t2 + np.pi)])
            y = np.concatenate([R * np.sin(t1), R * np.sin(t2 + np.pi)])
            z = np.concatenate([p * t1 / (2 * np.pi), p * t2 / (2 * np.pi)])
            data = np.column_stack([x, y, z])
            data += rng.standard_normal(data.shape) * noise
        elif name == "catenoid":
            ch = _mp.get("height", 1.5)
            u = rng.uniform(-ch, ch, n_samples)
            theta = rng.uniform(0, 2 * np.pi, n_samples)
            x = np.cosh(u) * np.cos(theta)
            y = np.cosh(u) * np.sin(theta)
            z = u
            data = np.column_stack([x, y, z])
            data += rng.standard_normal(data.shape) * noise
        elif name == "saddle":
            ext = _mp.get("extent", 2.0)
            k = _mp.get("curvature", 0.5)
            u = rng.uniform(-ext, ext, n_samples)
            v = rng.uniform(-ext, ext, n_samples)
            x = u
            y = v
            z = k * (u**2 - v**2)
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
        elif name == "clifford":
            r1 = _mp.get("r1", 1.0)
            r2 = _mp.get("r2", 1.0)
            a = _mp.get("angle", np.pi / 4)
            u = rng.uniform(0, 2 * np.pi, n_samples)
            v = rng.uniform(0, 2 * np.pi, n_samples)
            x4 = r1 * np.cos(u)
            y4 = r1 * np.sin(u)
            z4 = r2 * np.cos(v)
            w4 = r2 * np.sin(v)
            x_rot = x4 * np.cos(a) - w4 * np.sin(a)
            w_rot = x4 * np.sin(a) + w4 * np.cos(a)
            denom = 2.5 - w_rot
            x = x_rot / denom
            y = y4 / denom
            z = z4 / denom
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
    D_values_all = [
        2,
        3,
        8,
        16,
        32,
        128,
        512,
    ]
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
        D_values_all,
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
    traj_noise_t = mo.ui.slider(
        start=0.0,
        stop=0.95,
        step=0.05,
        value=0.3,
        label="Denoising start $t_0$",
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
    traj_show_vf = mo.ui.switch(label="Show velocity field", value=False)
    return (
        traj_D,
        traj_batch,
        traj_focus,
        traj_noise_t,
        traj_show_vf,
        traj_view_t,
    )


@app.cell(hide_code=True)
def traj_compute(
    data_preview,
    jax,
    jnp,
    n_sample_steps,
    noise_scale,
    np,
    pred_labels,
    projections,
    results,
    solver,
    t_eps,
    traj_D,
    traj_batch,
    traj_noise_t,
):
    _D = traj_D.value
    _P_np = projections[_D]
    _P = jnp.array(_P_np)
    _t_start = traj_noise_t.value
    _n = int(traj_batch.value)
    _ns = noise_scale.value
    _te = t_eps.value
    _n_steps = n_sample_steps
    _use_heun = solver.value == "heun"

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


if __name__ == "__main__":
    app.run()
