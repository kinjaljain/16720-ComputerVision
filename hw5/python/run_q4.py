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
import skimage.transform

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

    plt.imshow(1 - bw, cmap='gray')
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()
    # find the rows using..RANSAC, counting, clustering, etc.


    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    line_threshold = (bboxes[0][2] - bboxes[0][0]) / 1.1
    word_separation_threshold = (bboxes[0][3] - bboxes[0][1]) * 1.9
    all_lines = []
    all_word_idxs = []
    for idx in range(len(bboxes)):
        line = []
        if any(idx in rows for rows in all_word_idxs):
            continue
        else:
            word_c = bboxes[idx][2]
            word = [np.abs(word_c - val[2]) for val in bboxes]
            word = np.asarray(word, dtype=float)
            word = word > line_threshold
            word = word * 1.0
            idxs = np.where(word == 0)[0]
            all_word_idxs.append(idxs)
            for i in idxs:
                line.append(bboxes[int(i)])
            line.sort(key=lambda x: x[1])
            all_lines.append(line)

    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle', 'rb'))

    output = ""
    prev_word = None
    for line in all_lines:
        for bbox in line:
            minr, minc, maxr, maxc = bbox
            if prev_word and (maxc - prev_word) > word_separation_threshold:
                output = output + " "
            prev_word = maxc

            char = bw[minr:maxr, minc:maxc]
            img = np.pad(char, ((55, 55), (55, 55)), 'constant', constant_values=0.0)
            img = skimage.img_as_float(img).astype(np.float32)
            img = skimage.transform.resize(img, (32, 32))
            img = skimage.morphology.dilation(img, skimage.morphology.square(3))
            img = (1 - img).T
            img = np.reshape(img, (-1, 32 * 32))
            h1 = forward(img, params, 'layer1')
            probs = forward(h1, params, 'output', softmax)
            pred = np.argmax(probs)
            output += letters[pred]
        output += '\n'

    print(output)
