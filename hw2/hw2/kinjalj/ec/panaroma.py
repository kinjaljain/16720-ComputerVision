#Import necessary functions
import numpy as np
import cv2
from matchPics import matchPics
from planarH import computeH_ransac, compositeH
from opts import get_opts


#Write script for Q4.2x
def get_panaroma(img1, img2, H):
    # Get image corners from images read using cv2
    w1, h1 = img1.shape[0], img1.shape[1]
    w2, h2 = img2.shape[0], img2.shape[1]

    img1_corners = np.expand_dims(np.float32([[0, 0], [0, w1], [h1, 0], [h1, w1]]), axis=1)
    img2_corners = np.expand_dims(np.float32([[0, 0], [0, w2], [h2, 0], [h2, w2]]), axis=1)

    # perspectiveTransform works for set of points unlike warpPerspective which works on entire image
    img2_corners = cv2.perspectiveTransform(img2_corners, H)
    composite_corners = np.r_[img1_corners, img2_corners]
    [x_min, y_min] = np.int32(composite_corners.min(axis=0).flatten() - 0.5)
    [x_max, y_max] = np.int32(composite_corners.max(axis=0).flatten() + 0.5)

    rotation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

    # Warp second image to the complete scene
    composite_img = cv2.warpPerspective(img2, rotation_matrix.dot(H), (x_max - x_min, y_max - y_min))
    composite_img[-y_min:w1 - y_min, -x_min:h1 - x_min] = img1
    return composite_img

# Read the images using cv2
img1 = cv2.imread('../ec/img_left.JPG')
img2 = cv2.imread('../ec/img_right.JPG')

# get opts for matching points and ransac
opts = get_opts()

matches, locs1, locs2 = matchPics(np.transpose(img1, (1, 0, 2)), np.transpose(img2, (1, 0, 2)), opts)
bestH2to1, inliers = computeH_ransac(matches, locs1, locs2, opts)
img = get_panaroma(img1, img2, bestH2to1)
cv2.imwrite('panaroma.png', img)
