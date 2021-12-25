# Program for Structural Content Calculation

def StructuralContent(origImg, distImg):
    origImg = origImg.astype(float)
    distImg = distImg.astype(float)
    SC = (origImg * origImg).sum() / (distImg * distImg).sum()
    return SC
