# Program for Peak Signal to Noise Ratio Calculation

import math

def PeakSignaltoNoiseRatio(origImg, distImg):
    origImg = origImg.astype(float)
    distImg = distImg.astype(float)
    M, N = origImg.shape
    error = origImg - distImg
    MSE = (error * error).sum() / (M * N)
    if (MSE > 0):
        PSNR = 10 * math.log(255 * 255 / MSE) / math.log(10)
    else:
        PSNR = 99
    return PSNR
