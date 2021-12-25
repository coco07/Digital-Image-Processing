# Program for Mean Square Error Calculation

def MeanSquareError(origImg, distImg):
    origImg = origImg.astype(float)
    distImg = distImg.astype(float)
    M, N = origImg.shape
    error = origImg - distImg
    MSE = (error * error).sum() / (M * N)
    return MSE
