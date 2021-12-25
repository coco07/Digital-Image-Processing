# Program for Normalized Cross Correlation Calculation

def NormalizedCrossCorrelation(origImg, distImg):
    origImg = origImg.astype(float)
    distImg = distImg.astype(float)
    NK = (origImg * distImg).sum() / (origImg * origImg).sum()
    return NK
