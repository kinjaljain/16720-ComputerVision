import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage.io
import skimage.morphology
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

    plt.imshow(bw, cmap='gray')
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()


    # # find the rows using..RANSAC, counting, clustering, etc.
    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    data = []
    line_sep = (bboxes[0][2] - bboxes[0][0]) / 1.1
    word_sep = (bboxes[0][3] - bboxes[0][1]) * 2
    lines = []
    words = []

    for i in range(len(bboxes)):
        line = []
        if any(i in rows for rows in words):
            continue
        else:
            row_ref = bboxes[i][2]
            row_diff = [np.abs(row_ref - bbox[2]) for bbox in bboxes]
            row_diff = np.array([float(i) for i in row_diff])
            row_diff = (row_diff > line_sep) * 1.0
            word = np.where(row_diff == 0)[0]
            words.append(word)
            for k in word:
                line.append(bboxes[int(k)])
            line.sort(key=lambda x: x[1])
            lines.append(line)


    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle', 'rb'))

    line_string = ''
    prev_word_c = None
    for line in lines:
        for bbox in line:
            minr, minc, maxr, maxc = bbox
            H = maxc - minc
            W = maxr - minr
            if prev_word_c and maxc - prev_word_c > word_sep:
                line_string += ''
            prev_word_c = maxc
            char = bw[minr:maxr, minc:maxc]
            char = np.pad(char, ((60, 60), (60, 60)), 'constant', constant_values=0.0)
            char = skimage.img_as_float(char).astype(np.float32)
            char = skimage.transform.resize(char, (32, 32))
            char = skimage.morphology.dilation(char, skimage.morphology.square(3))
            char = char.T
            char = np.reshape(char, (-1, 32*32))

            h1 = forward(char, params, 'layer1')
            probs = forward(h1, params, 'output', softmax)
            pred_idx = np.argmax(probs)
            line_string += letters[pred_idx]
        line_string += '\n'

    print("Data: {}".format(line_string))
