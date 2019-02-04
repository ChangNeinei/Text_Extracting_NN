import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation
from skimage.morphology import square

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions
    #noisy = skimage.util.random_noise(image)
    im = skimage.restoration.denoise_tv_chambolle(image)
    grey_image = skimage.color.rgb2gray(im)
    thres = skimage.filters.threshold_isodata(grey_image)
    binary = np.ones(grey_image.shape) * (grey_image >= thres)
    binary = skimage.morphology.erosion(binary, selem = square(10))
    bw = skimage.morphology.closing(binary, square(3))

    bw = np.ones(grey_image.shape) - bw
    labels = skimage.measure.label(bw)
    label_num = labels.max()

    #bboxes = np.zeros((label_num, 4))
    bboxes = []
    for i in range(label_num):
    	y, x = np.where(labels == i + 1)
    	if (y.max() - y.min()) > 20 and (x.max() - x.min()) > 20:
    		#bboxes[i, :] = np.array([y.min() - 3, x.min() - 3, y.max() + 3, x.max() + 3])
    		bboxes.append([y.min() - 5, x.min() - 5, y.max() + 5, x.max() + 5])

    bw = np.ones(grey_image.shape) - bw

    return bboxes, bw

