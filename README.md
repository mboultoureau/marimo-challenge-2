# Back to Basics: Let Denoising Generative Models Denoise

A Marimo notebook implementation exploring the principles discussed in the paper *"Back to Basics: Let Denoising Generative Models Denoise"* for the Marimo Notebook Competition #2.

## Overview

* **The Challenge:** [Marimo Notebook Competition #2](https://marimo.io/pages/events/notebook-competition-2) — A developer challenge focused on building data science notebooks using Marimo's reactive Python architecture.
* **The Paper:** [*Back to Basics: Let Denoising Generative Models Denoise* (alphaXiv:2511.13720)](https://www.alphaxiv.org/abs/2511.13720) — An analysis of diffusion mechanics that evaluates simplifying step scheduling down to its fundamental denoising properties.

## Setup

The Python virtual environment is managed with [uv](https://docs.astral.sh/uv/).

Refer to [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to install it.

Create the virtual environment with all the necessary dependencies :
```bash
uv sync
```

Run the marimo notebook:
```bash
uv run marimo run denoising.py
```

## Contributing

A pre-commit hook configuration is provided.
