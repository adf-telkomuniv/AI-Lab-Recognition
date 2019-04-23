##
# @license
# Copyright 2018 AI Lab - Telkom University. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# =============================================================================

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


# @author ANDITYA ARIFIANTO
# copyright (c) 2018 - Artificial Intelligence Laboratory and Computing Laboratory, Telkom University #