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

# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# load the pretrained network and create letters
q3_weights_optim = pickle.load(open('q3_weights.pickle','rb'), encoding="ASCII")
letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
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
        figs = np.zeros((len(values), 1024))

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
            area = area.reshape(1024)
            figs[i] = area

        # forward to the network
        out = forward(figs, q3_weights_optim, 'hidden')
        probs = forward(out, q3_weights_optim, 'output', softmax)
        pred = np.argmax(probs, axis=1)
        row = []
        for idx in pred:
            row.append(letters[idx])
        words.append(''.join(row))

    print(words)


