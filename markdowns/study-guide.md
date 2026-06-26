# Study Guide

## Suggested Reading Order

### Theory-first

1. [Lilian Weng — "What are Diffusion Models?"](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) — single-page diffusion overview
2. [DDPM paper (Ho et al.)](https://arxiv.org/abs/2006.11239) — the foundation
3. [Flow Matching paper (Lipman et al.)](https://arxiv.org/abs/2210.02747) — the framework JiT uses
4. [DiT paper (Peebles & Xie)](https://arxiv.org/abs/2212.09748) — the direct predecessor
5. [The JiT paper](https://arxiv.org/abs/2511.13720) + [walkthrough video](https://www.youtube.com/watch?v=u5yKZzTTEHo)
6. Manifold hypothesis resources (Wikipedia + Carlsson)

### Intuition-first, implementation-focused

1. [But how do AI images actually work? — 3Blue1Brown / Welch Labs](https://www.youtube.com/watch?v=iv-5mZ_9CPY)
2. [ViT paper](https://arxiv.org/abs/2010.11929) or [Kilcher walkthrough](https://www.youtube.com/watch?v=TrdevFK_am4)
3. [DiT GitHub repo](https://github.com/facebookresearch/DiT) — study the code structure
4. [The JiT paper](https://arxiv.org/abs/2511.13720) + [walkthrough video](https://www.youtube.com/watch?v=u5yKZzTTEHo)
5. [marimo docs](https://docs.marimo.io/) — UI elements for the notebook
6. [Lilian Weng blog post](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) — as a reference

---

## Key Equations

These are the core equations you need to internalize and demonstrate in the notebook.

### 1. Linear interpolation (the noising process)

$$z_t = t \cdot x + (1 - t) \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

At time $t \in [0, 1]$, the noisy sample $z_t$ is a linear mix between the clean image $x$ (at $t=1$) and pure noise $\epsilon$ (at $t=0$). This is the flow matching interpolation — simpler than DDPM's noise schedule.

### 2. Flow velocity

$$v = x - \epsilon$$

The velocity field of the flow. This is the quantity the ODE solver needs to transport noise $\epsilon$ toward clean data $x$ along the linear path.

### 3. The nine prediction-loss combinations

The paper identifies three prediction targets $\{x, \epsilon, v\}$ and three loss spaces $\{x, \epsilon, v\}$. Since $z_t = t \cdot x + (1-t) \cdot \epsilon$ and $v = x - \epsilon$, any two of $\{x, \epsilon, v\}$ determine the third:

$$x = z_t + (1 - t) \cdot v$$

$$\epsilon = z_t - t \cdot v$$

$$v = \frac{z_t - \epsilon}{t} = \frac{x - z_t}{1 - t}$$

This means a network predicting any one quantity can be converted to any loss space. The paper's key finding: the **prediction space** matters, not just the loss space.

### 4. x-prediction with v-loss (what JiT uses)

The network directly predicts the clean image:

$$\hat{x}_\theta = f_\theta(z_t, t)$$

Then the predicted velocity is recovered:

$$\hat{v}_\theta = \frac{\hat{x}_\theta - z_t}{1 - t}$$

And the loss is computed in v-space:

$$\mathcal{L} = \mathbb{E}_{t, x, \epsilon} \left[ \left\| \hat{v}_\theta(z_t, t) - v \right\|^2 \right]$$

This combination is crucial: predicting $x$ keeps the network in a low-dimensional manifold (the manifold of natural images), while the v-loss provides proper gradient weighting across timesteps.

### 5. Sampling ODE

$$\frac{dz_t}{dt} = v_\theta(z_t, t)$$

Starting from $z_0 \sim \mathcal{N}(0, I)$ and integrating forward to $t = 1$ gives the generated image. Solved numerically with Euler or Heun's method:

**Euler step:**

$$z_{t + \Delta t} = z_t + \Delta t \cdot v_\theta(z_t, t)$$

**Heun step (2nd-order):**

$$\tilde{z}_{t+\Delta t} = z_t + \Delta t \cdot v_\theta(z_t, t)$$

$$z_{t+\Delta t} = z_t + \frac{\Delta t}{2} \left[ v_\theta(z_t, t) + v_\theta(\tilde{z}_{t+\Delta t}, t + \Delta t) \right]$$

### 6. Timestep sampling distribution (logit-normal)

$$t \sim \sigma(\mathcal{N}(0, 1))$$

where $\sigma$ is the sigmoid function. This concentrates samples around $t = 0.5$ where the learning signal is strongest, rather than sampling uniformly.

### 7. Classifier-Free Guidance (CFG)

$$\hat{v}_\text{guided} = (1 + w) \cdot v_\theta(z_t, t, c) - w \cdot v_\theta(z_t, t, \varnothing)$$

where $c$ is the class label, $\varnothing$ is the null (unconditional) class, and $w$ is the guidance scale. The paper uses $w = 0.35$ (i.e., `cfg = 1.35`).

### 8. The manifold argument (why x-prediction works)

If clean data $x$ lies on a $d$-dimensional manifold embedded in $D$-dimensional space ($d \ll D$):

- **x-prediction**: the network only needs to output points on the $d$-dimensional manifold — feasible even for under-capacity networks
- **$\epsilon$-prediction**: the network must predict $D$-dimensional noise — requires capacity proportional to $D$
- **v-prediction**: same problem as $\epsilon$ since $v = x - \epsilon$ and $\epsilon$ dominates in high dimensions

This is why x-prediction succeeds where $\epsilon$/$v$-prediction fail catastrophically as $D$ increases.
