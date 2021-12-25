import numpy as np
import cv2

from IQMS.AverageDifference import AverageDifference
from IQMS.MaximumDifference import MaximumDifference
from IQMS.MeanSquareError import MeanSquareError
from IQMS.NormalizedAbsoluteError import NormalizedAbsoluteError
from IQMS.NormalizedCrossCorrelation import NormalizedCrossCorrelation
from IQMS.PeakSignaltoNoiseRatio import PeakSignaltoNoiseRatio
from IQMS.StructuralContent import StructuralContent
from IQMS.StructuralSimilarity import ssim


def calculate_IQMs(origImg, distImg):
    # If the input image is rgb, convert it to gray image
    noOfDim = origImg.ndim
    if (noOfDim == 3):
        origImg = cv2.cvtColor(origImg, cv2.COLOR_BGR2GRAY)

    # If the distorted image is rgb, convert it to gray image
    noOfDim = distImg.ndim
    if (noOfDim == 3):
        distImg = cv2.cvtColor(distImg, cv2.COLOR_BGR2GRAY)

    # Size Validation
    origSiz = len(origImg)
    distSiz = len(distImg)

    if (origSiz != distSiz):
        print('Error: Original Image & Distorted Image should be of same dimensions')
        return

    temp_vec = np.zeros(8)

    # Mean Square Error
    MSE = MeanSquareError(origImg, distImg)
    temp_vec[0] = MSE

    # Peak Signal to Noise Ratio
    PSNR = PeakSignaltoNoiseRatio(origImg, distImg)
    temp_vec[1] = PSNR

    # Normalized Cross-Correlation
    NK = NormalizedCrossCorrelation(origImg, distImg)
    temp_vec[2] = NK

    # Average Difference
    AD = AverageDifference(origImg, distImg)
    temp_vec[3] = AD

    # Structural Content
    SC = StructuralContent(origImg, distImg)
    temp_vec[4] = SC

    # Maximum Difference
    MD = MaximumDifference(origImg, distImg)
    temp_vec[5] = MD

    # Normalized Absolute Error
    NAE = NormalizedAbsoluteError(origImg, distImg)
    temp_vec[6] = NAE

    # Structural Similarity
    SSIM = ssim(origImg, distImg)
    temp_vec[7] = SSIM

    return temp_vec
