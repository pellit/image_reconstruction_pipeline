
import cv2, numpy as np, os
from src.pipeline import reconstruct_image
def test_psnr():
    img_path = os.path.join(os.path.dirname(__file__), 'sample.png')
    if not os.path.exists(img_path):
        import urllib.request
        urllib.request.urlretrieve('https://upload.wikimedia.org/wikipedia/commons/thumb/2/24/Wolf-Canis_lupus.jpg/256px-Wolf-Canis_lupus.jpg', img_path)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
    rec, psnr, _ = reconstruct_image(img)
    assert psnr > 30  # basic quality gate
