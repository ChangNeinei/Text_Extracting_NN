import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation
from skimage.transform import resize
import copy 
import math
from skimage.morphology import erosion
from skimage.morphology import square

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
   
    bboxes, bw = findLetters(im1)
    
    plt.imshow(bw, cmap='gray')

    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()
    
    # sort rows (the same rows are nearby)
    bboxes.sort(key=lambda x:x[0])

    grey_image = skimage.color.rgb2gray(im1)

    # find the rows using..RANSAC, counting, clustering, etc.
    y_1, x_1, y_2, x_2 = bboxes[0]
    origin_row = (y_1 + y_2) // 2
    err = abs(origin_row - y_1) 
    sort_temp = []
    sorted_bboxes = []
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        row = (minr + maxr) // 2
        if abs(origin_row - row) < err:
            sort_temp.append(bbox)
        else:
            sort_temp.sort(key=lambda x:x[1])
            sorted_bboxes.append(sort_temp)
            sort_temp = []
            sort_temp.append(bbox)
            origin_row = row
    sort_temp.sort(key=lambda x:x[1])
    sorted_bboxes.append(sort_temp)
    data_x_temp = []
    data_x = []
    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    for same_line in sorted_bboxes:
        for i, sort_loc in enumerate(same_line):
            minr, minc, maxr, maxc = sort_loc
            patch_image = grey_image[minr : maxr + 1, minc : maxc + 1]
            patch_image = erosion(patch_image, selem = square(8))
            r_diff = maxr - minr
            c_diff = maxc - minc
            pad_r = r_diff // 8
            pad_c = c_diff // 8
            patch_image = np.pad(patch_image, ((pad_r,pad_r),(pad_c,pad_c)), mode='maximum')
            patch_image = resize(patch_image, (32, 32))
            patch_image_T = patch_image.T
            data_x_temp.append(patch_image_T.flatten())
        data_x.append(np.array(data_x_temp))
        data_x_temp = []
    
    # load the weights
    # run the crops through your neural network and print them out
     
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))

    for group in data_x:
        h_1 = forward(group, params, 'layer1')
        predict_y = forward(h_1, params, 'output',softmax)
        max_output = np.amax(predict_y, axis = 1, keepdims = True)
        pred_class = np.ones(predict_y.shape) * (predict_y//max_output == 1)
        pred = np.nonzero(pred_class)[1]
        print(letters[pred])



