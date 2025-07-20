
"""Very first draft of Dudeney Patch (entropy‑preserving isometric shuffle)"""
import numpy as np

def patch_image(img, block=4, seed=0):
    """Divide img into block×block tiles and shuffle them pseudo‑randomly.
    This is a placeholder for a future entropy‑guided optimisation."""
    np.random.seed(seed)
    h, w = img.shape
    bh, bw = h//block, w//block
    patches = [img[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
               for i in range(block) for j in range(block)]
    np.random.shuffle(patches)
    # re‑assemble
    out = np.zeros_like(img)
    idx = 0
    for i in range(block):
        for j in range(block):
            out[i*bh:(i+1)*bh, j*bw:(j+1)*bw] = patches[idx]
            idx += 1
    return out
