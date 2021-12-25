# Program for Average Difference Calculation

def AverageDifference(origImg, distImg):
    origImg = origImg.astype(float)
    distImg = distImg.astype(float)
    M, N = origImg.shape
    error = origImg - distImg
    AD = error.sum() / (M * N)
    return AD
