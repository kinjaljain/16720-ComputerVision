import numpy as np
import cv2
import skimage.color
import scipy.ndimage
from matchPics import matchPics
from opts import get_opts
from matplotlib import pyplot as plt

# Q2.1.6
# Read the image and convert to grayscale, if necessary
img = cv2.imread('../data/cv_cover.jpg')
opts = get_opts()
x = []

for i in range(36):
    # Rotate Image
    rotated_img = scipy.ndimage.rotate(img, i * 10)

    # Compute features, descriptors and Match features
    matches, locs1, locs2 = matchPics(img, rotated_img, opts)

    # Update histogram
    x.append(len(matches))

# Display histogram
angles = [i for i in range(0, 360, 10)]
angles = np.asarray(angles)

plt.bar(angles, x, 5, color='red')
plt.xlabel('Rotation Angle')
plt.ylabel('Number of Matches')
plt.title('Number of Matches vs Rotation Angle')
plt.savefig("../results/histogram.png")
plt.show()
# plt.hist(x, bins=angles)
# plt.show()

print(x)
