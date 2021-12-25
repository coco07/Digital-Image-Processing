# Program for structural similarity Calculation

from skimage.metrics import structural_similarity as ssimg

def ssim(origImg, distImg):
    (score, diff) = ssimg(origImg, distImg, full=True)
    return score