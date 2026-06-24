# Back to Basics: Let Denoising Generative Models Denoise

A Marimo notebook implementation exploring the principles discussed in the paper *"Back to Basics: Let Denoising Generative Models Denoise"* for the Marimo Notebook Competition #2.

## Overview

* **The Challenge:** [Marimo Notebook Competition #2](https://marimo.io/pages/events/notebook-competition-2) — A developer challenge focused on building data science notebooks using Marimo's reactive Python architecture.
* **The Paper:** [*Back to Basics: Let Denoising Generative Models Denoise* (alphaXiv:2511.13720)](https://www.alphaxiv.org/abs/2511.13720) — An analysis of diffusion mechanics that evaluates simplifying step scheduling down to its fundamental denoising properties.

## Setup

```bash
# Create the virtual environment
python3 -m venv .venv

# Activate on Linux/macOS
source .venv/bin/activate

# Activate on Windows
.venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Edit the notebook
marimo edit denoising.py
```
