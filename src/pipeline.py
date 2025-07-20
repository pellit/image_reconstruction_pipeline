
"""Core functions for the Image Reconstruction Pipeline"""
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

__all__ = [
    "fft2d", "ifft2d",
    "low_pass", "fractal_boost",
    "phase_retrieval",
    "reconstruct_image"
]

# ---------------- FFT helpers ----------------
def fft2d(img):
    """Centered 2‑D FFT"""
    return fftshift(fft2(img))

def ifft2d(F):
    """Inverse centered 2‑D FFT (real part)"""
    return np.real(ifft2(ifftshift(F)))

# -------------- Signal operators -------------
def low_pass(F, k_c=0.15):
    h, w = F.shape
    u, v = np.meshgrid(np.linspace(-.5, .5, w), np.linspace(-.5, .5, h))
    mask = np.sqrt(u**2 + v**2) < k_c
    return F * mask

def fractal_boost(F_low, k_c=0.15, D_f=1.8):
    h, w = F_low.shape
    u, v = np.meshgrid(np.linspace(-.5, .5, w), np.linspace(-.5, .5, h))
    k = np.sqrt(u**2 + v**2)
    alpha = D_f - 2
    boost = (k > k_c) * (k / k_c) ** (-alpha)
    return F_low + boost * F_low.max()

def phase_retrieval(mag, n_iter=30):
    phase = np.zeros_like(mag)
    for _ in range(n_iter):
        F_tmp = mag * np.exp(1j * phase)
        img_tmp = ifft2d(F_tmp)
        img_tmp = np.clip(img_tmp, 0, 1)
        phase = np.angle(fft2d(img_tmp))
    return mag * np.exp(1j * phase)

# -------------- High‑level API ---------------
def reconstruct_image(img, k_c=0.15, D_f=1.8, n_iter=30):
    """Full reconstruction given a grayscale image in [0,1]"""
    F      = fft2d(img)
    F_low  = low_pass(F, k_c=k_c)
    F_bo   = fractal_boost(F_low, k_c=k_c, D_f=D_f)
    F_rec  = phase_retrieval(np.abs(F_bo), n_iter=n_iter)
    rec    = np.clip(ifft2d(F_rec), 0, 1)
    psnr   = peak_signal_noise_ratio(img, rec, data_range=1)
    ssim   = structural_similarity(img, rec, data_range=1)
    return rec, psnr, ssim
