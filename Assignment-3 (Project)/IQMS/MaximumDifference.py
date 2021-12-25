# Program for Maximum Difference Calculation

def MaximumDifference(origImg, distImg):
    origImg = origImg.astype(float)
    distImg = distImg.astype(float)
    error = origImg - distImg
    MD = error.max()
    return MD
