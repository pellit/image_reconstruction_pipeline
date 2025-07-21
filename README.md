# Image Reconstruction Pipeline: FFT × Fractal × Phase × Dudeney

<img src="https://img.shields.io/badge/License-MIT-green" alt="license"/>
<img src="https://img.shields.io/badge/Python-3.9%2B-blue"/>

**Restore hidden details from undersampled images/videos combining classic signal processing, fractal extrapolation and a novel _Dudeney Patch_ entropy step.**

## ✨ Features
- 2‑D & 3‑D (video) Fourier pipeline
- Fractal high‑frequency boosting with tunable fractal dimension
- Iterative phase‑retrieval (Gerchberg–Saxton)
- 🚀 **Dudeney Patch:** first open‑source implementation of isometric entropy‑preserving patching
- Full metrics: PSNR, SSIM, MS‑SSIM, Shannon entropy

## 🖥️ Quick start
```bash
git clone https://github.com/your‑user/image_reconstruction_pipeline.git
cd image_reconstruction_pipeline
pip install -r requirements.txt
jupyter notebook notebooks/demo.ipynb
```

## 📂 Structure
```
src/core library
notebooks/demo & experiments
examples/ place your images / videos here
```

## 📐 Mathematical Foundations

Below is the compact yet rigorous model that motivates each stage of the pipeline.

### 1. Signal Model
An image is treated as a square–integrable function:
```math
I(x,y) : \Omega \subset \mathbb{R}^{2}\;\to\;[0,1],\qquad
\lVert I\rVert_2^2=\iint_\Omega I^2\,dx\,dy<\infty .
```
For a video clip we extend to:
```math
I(x,y,t):\Omega\times[0,T]\to[0,1].
```

### 2. Fourier Decomposition (Isomorphism)
The centred Fourier operator:

```math
\tilde I(k_x,k_y)=\!\int_\Omega I(x,y)\,e^{-2\pi i(k_xx+k_yy)}\,dx\,dy
```
is unitary, so energy is preserved:

```math
\lVert I\rVert_2=\lVert\tilde I\rVert_2
```
Cropping with a low-pass mask $\chi_{|k|<k_c}$ keeps energy but discards information above $k_c$.

### 3. Fractal High-Frequency Extrapolation
Natural images follow:
```math
|\tilde I(k)|^2\propto|k|^{-\beta},\qquad\beta=2D_f-2,
```
where $D_f$ is the fractal (Hausdorff) dimension.
Missing bands are filled with the power-law prior:

```math
\tilde I_{\text{boost}}(k)=
\begin{cases}
\tilde I(k), & |k|<k_c \\\\
\tilde I(k_c)\bigl(\tfrac{|k|}{k_c}\bigr)^{-\beta/2}, & |k|\ge k_c
\end{cases}
```

### 4. Phase Retrieval
We solve:
```math
\min_\varphi\;\bigl\lVert I-\mathcal{F}^{-1}\!\bigl(|\tilde I|e^{i\varphi}\bigr)\bigr\rVert_2^2
\quad\text{s.t.}\quad I\ge0,
```
with the Gerchberg–Saxton iteration (alternating magnitude and positivity constraints).

### 5. Dudeney Patch – Entropy-Preserving Isometry
Let $P=\{P_j\}$ be a dissection of $\Omega$ and $T_j$ an isometry that maps $P_j$ to $Q_{\sigma(j)}$. The operator:
```math
(\mathcal{D}_\sigma I)(x)=I\!\bigl(T_{\sigma^{-1}(j)}^{-1}x\bigr),\;x\in Q_j
```
is unitary; total energy and the grey-level histogram are conserved.
We search for the permutation $\sigma^{\ast}$ that maximises the conditional entropy:
```math
\sigma^{\ast}=\arg\max_\sigma H\!\bigl(I(P_j)\mid\text{neigh}\bigr).
```

### 6. Quality & Information Bounds

* **Shannon sampling:** $f_s\!\ge\!2B\Rightarrow$ perfect inversion is possible.
* **Spectral SNR:**
```math
\mathrm{SNR}_k=10\log_{10}
\frac{\sum|\tilde I|^{2}}{\sum|\tilde I-\tilde I_{\text{rec}}|^{2}}.
```


* **Entropy:**
```math
H(I)=-\sum p\log_2p.
```
The Dudeney step seeks $H(I_{\text{rec}})\!\approx\!H(I)$ while respecting the Bekenstein bound $I\le 2\pi ER/\hbar c$.

### 7. Error Maps

Directional error magnitude uses Sobel gradients:
```math
\mathrm{err}_g=\sqrt{\bigl(g_x^{\text{orig}}-g_x^{\text{rec}}\bigr)^2+\bigl(g_y^{\text{orig}}-g_y^{\text{rec}}\bigr)^2 }
```

Hot-spots guide further optimisation of the patch operator.

> **TL;DR**  Every stage is unitary or entropy-conserving except the
fractal extrapolation, which follows the empirical \(1/f\) law of natural
images—so no artificial energy is injected and each boosted frequency is
statistically plausible under a fractal prior.



## 📜 License
MIT © 2025 — Feel free to use, modify, and star ⭐

