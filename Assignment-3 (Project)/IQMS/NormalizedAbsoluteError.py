# Program for Normalized Absolute Error Calculation

def NormalizedAbsoluteError(origImg, distImg):
    origImg = origImg.astype(float)
    distImg = distImg.astype(float)
    error = origImg - distImg
    NAE = abs(error).sum() / origImg.sum()
    return NAE
