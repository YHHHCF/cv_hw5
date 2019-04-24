import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import scipy.io
import matplotlib.pyplot as plt
from q5_train import *

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

from nn import *
from q4 import *
import pickle
import string

# load the model and letters
model = load_ckpt('./../results/model.t7')
letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images', img)))
    bboxes, bw = findLetters(im1, img)

    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()

    # find the rows using..RANSAC, counting, clustering, etc.
    keys = np.arange(len(bboxes))

    boxes_dict = {}

    avg_h = 0

    for i in range(len(bboxes)):
        boxes_dict[i] = bboxes[i]
        avg_h += (bboxes[i][2] - bboxes[i][0])

    avg_h /= len(bboxes)
    clusters = {}

    for i in range(len(boxes_dict)):
        box = boxes_dict[i]
        keys = clusters.keys()

        # if there is a corresponding cluster, get into it
        lo = box[0] - 0.5 * avg_h
        hi = box[2] + 0.5 * avg_h
        center = 0.5 * (lo + hi)

        has_cluster = False
        for key in keys:
            if lo < key < hi:
                clusters[key].append(box)
                has_cluster = True
                break

        # if there is no corresponding cluster, create one
        if not has_cluster:
            clusters[center] = []
            clusters[center].append(box)

    keys = clusters.keys()
    keys = sorted(keys)

    for key in keys:
        values = sorted(clusters[key], key=lambda x: x[1])
        clusters[key] = values

    words = []

    # crop the bounding boxes
    for key in clusters.keys():
        values = clusters[key]
        figs = np.zeros((len(values), 3, 32, 32))

        for i in range(len(values)):
            box = values[i]
            area = bw[box[0]:box[2], box[1]:box[3]]

            if area.shape[0] > area.shape[1] + 5:
                pad_len = int((area.shape[0] - area.shape[1]) / 2)
                area = np.pad(area, ((0, 0), (pad_len, pad_len)), 'edge')
            elif area.shape[1] > area.shape[0] + 5:
                pad_len = int((area.shape[1] - area.shape[0]) / 2)
                area = np.pad(area, ((pad_len, pad_len), (0, 0)), 'edge')

            area = skimage.filters.gaussian(area, sigma=0.5)
            area = skimage.transform.resize(area, (32, 32))
            area = area.T
            area = np.stack((area, area, area), 0)
            figs[i] = area

        # forward to the network
        figs = torch.tensor(figs, dtype=torch.float32)
        probs = model(figs)
        pred = torch.argmax(probs, 1)
        row = []
        for idx in pred:
            row.append(letters[idx])
        words.append(''.join(row))

    print(words)

