import numpy as np
import os

import skimage
import skimage.measure as ms
import skimage.color
import skimage.filters
import skimage.morphology
import skimage.segmentation

from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float
import time
import matplotlib.patches as patches

import matplotlib.pyplot as plt


def process_img(image, file):
    file = file.split('.')[0]
    path = '../npy/' + file + '.npy'

    if os.path.exists(path):
        print(path + " exists")
        image = np.load(path)
        return image

    im1 = image[:, :, 0]

    start = time.time()
    im1 = denoise_bilateral(im1, sigma_color=0.1, sigma_spatial=10, multichannel=False)
    dur = time.time() - start
    print('dur:', dur)
    print("file saved at:", path)

    np.save(path, im1)
    return image


def show_box(boxes, im):
    plt.imshow(im)
    currentAxis = plt.gca()

    for box in boxes:
        h = box[2] - box[0]
        w = box[3] - box[1]
        rect = patches.Rectangle((box[1], box[0]), w, h, linewidth=1, edgecolor='r', facecolor='none')
        currentAxis.add_patch(rect)

    plt.show()


def merge(bboxes):
    num = len(bboxes)
    overlap_matrix = np.zeros((num, num))  # element i,j means how much of i'th area overlap with j

    for i in range(num):
        for j in range(i+1, num):
            box1 = bboxes[i]
            box2 = bboxes[j]

            area1 = (box1[2] - box1[0]) * (box1[2] - box1[0])
            area2 = (box2[2] - box2[0]) * (box2[2] - box2[0])

            overlap = 0
            if max(box1[0], box2[0]) < min(box1[2], box2[2]):
                w = min(box1[2], box2[2]) - max(box1[0], box2[0])
                if max(box1[1], box2[1]) < min(box1[3], box2[3]):
                    h = min(box1[3], box2[3]) - max(box1[1], box2[1])
                    overlap = w * h

            if area1 / area2 > 1.2 or area2 / area1 > 1.2:
                overlap_matrix[i][j] = overlap / area1
                overlap_matrix[j][i] = overlap / area2

    merge_idx = np.where(overlap_matrix > 0.3)

    # combine boxes which overlaps
    num_merge = len(merge_idx[0])

    if num_merge > 0:
        i = merge_idx[0][0]
        j = merge_idx[1][0]
        box1 = bboxes[i]
        box2 = bboxes[j]

        x1 = min(box1[0], box2[0])
        y1 = min(box1[1], box2[1])

        x2 = max(box1[2], box2[2])
        y2 = max(box1[3], box2[3])

        bboxes.remove(box1)
        bboxes.remove(box2)
        bboxes.append((x1, y1, x2, y2))

    return num_merge, bboxes


# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image, file):

    # denoise and greyscale
    bw = process_img(image, file)

    # threshold
    im = bw < 0.5

    # label connectivity
    im = ms.label(im, connectivity=2)
    num_labels = np.max(im) + 1

    # get boxes
    bboxes = []
    for idx in range(1, num_labels):
        imi = (im == idx)
        pts = np.where(imi)
        box = (np.min(pts[0]-15), np.min(pts[1]-15), np.max(pts[0]+15), np.max(pts[1]+15))
        h = box[2] - box[0]
        w = box[3] - box[1]

        # not small, have enough writing
        if w > 50 or h > 50:
            if len(pts[0]) > 100:
                bboxes.append(box)

    merge_num, bboxes = merge(bboxes)
    while merge_num > 0:
        merge_num, bboxes = merge(bboxes)

    return bboxes, bw
