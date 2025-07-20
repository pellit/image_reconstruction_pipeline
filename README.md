
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
src/            core library
notebooks/      demo & experiments
examples/       place your images / videos here
```

## 📜 License
MIT © 2025 — Feel free to use, modify, and star ⭐
