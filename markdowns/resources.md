# Resources for the project

## 1. Diffusion Models

Building enough knowledge and intuition to understand diffusion models and the novelties of the selected paper.

### Papers

| URL | Title | Description |
| --- | --- | --- |
| https://arxiv.org/abs/2006.11239 | Denoising Diffusion Probabilistic Models (DDPM) | **Essential.** The baseline the whole field builds on. Introduces epsilon-prediction, the noise schedule, and the training/sampling loop. |
| https://arxiv.org/abs/1503.03585 | Deep Unsupervised Learning using Nonequilibrium Thermodynamics | The original diffusion idea — read the intro/abstract for historical context |
| https://arxiv.org/abs/1907.05600 | Generative Modeling by Estimating Gradients of the Data Distribution | Score-based perspective — connects noise prediction to score functions |
| https://arxiv.org/abs/2208.11970 | Understanding Diffusion Models: A Unified Perspective | Tutorial paper unifying DDPM, score-based, and SDE viewpoints. Especially Section 4 (noise conditioned score networks) |

### YouTube videos

| URL | Title | Description |
| --- | --- | --- |
| https://www.youtube.com/watch?v=iv-5mZ_9CPY | But how do AI images and videos actually work? — Welch Labs / 3Blue1Brown | Guest video on 3Blue1Brown channel. Goes well beyond the typical "model learns to remove noise" summary — covers diffusion models, CLIP, and the math of turning text into images with strong visual intuitions |
| https://www.youtube.com/watch?v=fbLgFrlTnGU | What are Diffusion Models? | Short tutorial covering forward/reverse diffusion processes, the optimization objective, continuous-time formulation, and probability flow ODE |
| https://www.youtube.com/watch?v=EhndHhIvWWw | Diffusion Models: DDPM — Generative AI Animated | Animated walkthrough of how the DDPM paper formulates the diffusion process, training algorithm, and sampling algorithm. Good visualizations of the noise schedule and denoising steps |
| https://www.youtube.com/watch?v=R0uMcXsfo2o | The physics behind diffusion models — Julia Turc | Explains how diffusion models are grounded in physics — the same math that describes ink dispersing in water or heat traveling through a room. Builds intuition for the SDE/ODE connection used in flow matching |
| https://www.youtube.com/watch?v=utAxCAsWM-c | Variational Autoencoders (VAEs): Motivation, Math, ELBO & Generation of Data | Covers VAE motivation, mathematical foundations, ELBO derivation, and data generation. Relevant because Latent Diffusion Models depend on a VAE encoder/decoder — understanding VAEs helps explain why JiT removes them |

### Websites

| URL | Title | Description |
| --- | --- | --- |
| https://lilianweng.github.io/posts/2021-07-11-diffusion-models/ | What are Diffusion Models? — Lilian Weng | Best single-page overview: math + intuition + diagrams |

---

## 2. Flow Matching

JiT uses flow matching (linear interpolation $z_t = t \cdot x + (1-t) \cdot \epsilon$) rather than the DDPM noise schedule. This is critical to understand.

### Papers

| URL | Title | Description |
| --- | --- | --- |
| https://arxiv.org/abs/2210.02747 | Flow Matching for Generative Modeling | **Essential.** Defines the flow matching framework, linear interpolation paths, and velocity prediction. JiT builds directly on this |
| https://arxiv.org/abs/2209.03003 | Flow Straight and Fast: Rectified Flow | Rectified flow — another angle on the same linear interpolation idea |
| https://arxiv.org/abs/2209.15571 | Building Normalizing Flows with Stochastic Interpolants | Theoretical underpinning of stochastic interpolants |

### YouTube videos

| URL | Title | Description |
| --- | --- | --- |
| https://www.youtube.com/watch?v=7NNxK3CqaDk | Flow Matching for Generative Modeling — Paper Explained | Explains the Lipman et al. paper directly. Covers how flow matching serves as the basis for models like Stable Diffusion 3 |
| https://www.youtube.com/watch?v=DDq_pIfHqLs | How I Understand Flow Matching | Beginner-friendly explanation of flow matching as combining advantages of Continuous Normalizing Flows and Diffusion Models |
| https://www.youtube.com/watch?v=7cMzfkWFWhI | Flow Matching — Explanation + PyTorch Implementation | Covers theory + hands-on code. Good for the SWE teammate to see a working implementation |
| https://www.youtube.com/watch?v=5ZSwYogAxYg | Yaron Lipman — Flow Matching: Simplifying and Generalizing Diffusion Models | Talk by the original author of flow matching. Authoritative |
| https://www.youtube.com/watch?v=GCoP2w-Cqtg | MIT 6.S184 Lecture 01 — Generative AI with SDEs (2025) | MIT course lecture covering flow matching and diffusion from first principles |
| https://www.youtube.com/watch?v=4GQ1H80P_u8 | Flow Matching for Uniform Sampling of Diffusion Models | |

### Websites

| URL | Title | Description |
| --- | --- | --- |
| https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html | An Introduction to Flow Matching — Cambridge MLG | Clean mathematical walkthrough with diagrams |
| https://peterroelants.github.io/posts/flow_matching_intro/ | Flow Matching: A Visual Introduction — Peter Roelants | Visual, interactive blog post with code — good for building intuition |

---

## 3. Vision Transformers (the architecture)

JiT is a plain ViT operating on raw pixel patches. Understanding ViT is needed to implement the model.

### Papers

| URL | Title | Description |
| --- | --- | --- |
| https://arxiv.org/abs/2010.11929 | An Image is Worth 16x16 Words (ViT) | **Essential.** The base architecture — patch embedding, positional encoding, transformer blocks. JiT is literally a ViT |
| https://arxiv.org/abs/2212.09748 | Scalable Diffusion Models with Transformers (DiT) | **Essential.** The direct predecessor to JiT — introduces adaLN-Zero conditioning for class/timestep in diffusion transformers. JiT simplifies DiT |

### YouTube videos

| URL | Title | Description |
| --- | --- | --- |
| https://www.youtube.com/watch?v=TrdevFK_am4 | Yannic Kilcher — ViT paper explained | Clear 30-minute walkthrough of the ViT architecture |

---

## 4. Manifold Hypothesis (the core theoretical insight)

The paper's key claim is that natural images lie on low-dimensional manifolds, making x-prediction fundamentally easier than epsilon/v-prediction. This is *the* core contribution.

| URL | Title | Description |
| --- | --- | --- |
| https://en.wikipedia.org/wiki/Manifold_hypothesis | Wikipedia — Manifold hypothesis | Quick primer on the assumption |
| https://www.ams.org/journals/bull/2009-46-02/S0273-0979-09-01249-X/ | Carlsson, "Topology and Data", 2009 | Cited by the paper — mathematical foundation for data lying on low-dimensional manifolds |
| http://www.acad.bg/ebook/ml/MITPress-SemiSupervisedLearning.pdf | Chapelle et al., "Semi-Supervised Learning" (Chapter 1) | The manifold assumption in ML context — cited as ref [4] in the paper |
| https://www.youtube.com/watch?v=aircAruvnKk | 3Blue1Brown — "But what is a neural network?" | Helps build intuition about how networks learn low-dimensional structure |

---

## 5. The Paper Itself + Direct Walkthroughs

| URL | Title | Description |
| --- | --- | --- |
| https://arxiv.org/abs/2511.13720 | Back to Basics: Let Denoising Generative Models Denoise | The selected paper (accepted at CVPR 2026) |
| https://www.youtube.com/watch?v=u5yKZzTTEHo | Paper Walkthrough | Direct walkthrough explaining x-prediction vs epsilon/v-prediction, the manifold argument, and JiT architecture |
| https://www.youtube.com/watch?v=I-rzvLxVNdw | Back to Basics (Nov 2025) — alternative walkthrough | A second walkthrough — useful for a different perspective or if the first video leaves gaps |
| https://www.youtube.com/watch?v=iDniTU2XNXw | Why Denoising Models Should Actually Denoise — JiT Explained | Another take, focusing on the "why" behind x-prediction |

---

## 6. Latent Diffusion & Predecessors (what JiT removes)

The paper argues against latent diffusion — understanding LDM helps explain *why* JiT's approach is simpler.

| URL | Title | Description |
| --- | --- | --- |
| https://arxiv.org/abs/2112.10752 | High-Resolution Image Synthesis with Latent Diffusion Models | The dominant paradigm (Stable Diffusion). JiT's whole point is that you don't need the VAE encoder/decoder |
| https://arxiv.org/abs/2410.19324 | Simpler Diffusion (SiD2) | Recent pixel-space diffusion work — cited by the paper as related but using different prediction targets |

---

## 7. Training & Evaluation Concepts

| Concept | URL | Description |
| --- | --- | --- |
| FID (Fréchet Inception Distance) | https://arxiv.org/abs/1706.08500 | The evaluation metric used throughout — explain it in the notebook |
| Classifier-Free Guidance (CFG) | https://arxiv.org/abs/2207.12598 | Used in the paper's sampling — `cfg=1.35` is referenced in results |
| adaLN-Zero | https://arxiv.org/abs/2212.09748 (DiT, Section 3.2) | The conditioning mechanism JiT uses for timestep/class |

---

## 8. Implementation References

| URL | Title | Description |
| --- | --- | --- |
| https://github.com/LTH14/JiT | Official JiT repo (LTH14/JiT) | **The authors' own PyTorch implementation.** Start here for the actual model code |
| https://github.com/facebookresearch/DiT | DiT official repo | Reference implementation of the predecessor architecture — JiT simplifies this |
| https://github.com/lucidrains/denoising-diffusion-pytorch | lucidrains/denoising-diffusion-pytorch | Clean PyTorch DDPM implementation for reference |
| https://docs.marimo.io/ | marimo documentation | For building the interactive notebook UI elements |
