import numpy as np 
from scipy.stats import skew
from skimage import color

__author__ = 'GDS-COMPUTING'

def chromaticM(img):
    image = color.rgb2hsv(img)
    histH, binH = np.histogram(image[:,:,0].flatten(), normed=False)
    histS, binS = np.histogram(image[:,:,1].flatten(), normed=False)
    histV, binV = np.histogram(image[:,:,2].flatten(), normed=False)
    meanH = np.mean(image[:,:,0])
    stdH = np.std(image[:,:,0])
    skewH = skew(image[:,:,0].flatten(), axis=0)
    meanS = np.mean(image[:,:,1])
    stdS = np.std(image[:,:,1])
    skewS = skew(image[:,:,1].flatten(), axis=0)
    meanV = np.mean(image[:,:,2])
    stdV = np.std(image[:,:,2])
    skewV = skew(image[:,:,2].flatten(), axis=0)
    percentageMinH = (np.min(histH)/np.sum(histH))*100
    percentageMinS = (np.min(histS)/np.sum(histS))*100
    percentageMinV = (np.min(histV)/np.sum(histV))*100
    percentageMaxH = (np.max(histH)/np.sum(histH))*100
    percentageMaxS = (np.max(histS)/np.sum(histS))*100
    percentageMaxV = (np.max(histV)/np.sum(histV))*100
    
    return meanH, meanS, meanV, stdH, stdS, stdV, skewH, skewS, skewV, percentageMinH, percentageMinS, percentageMinV, percentageMaxH, percentageMaxS, percentageMaxV

def predict_spoof(img, model):
    feature = np.array([chromaticM(img)])
    return model.predict(feature)

# copyright (c) 2017 - Computing Laboratory, Telkom University #
