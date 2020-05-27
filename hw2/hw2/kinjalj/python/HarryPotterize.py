import numpy as np
import cv2
from opts import get_opts

#Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac, compositeH

#Write script for Q2.2.4
opts = get_opts()
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')

matches, locs1, locs2 = matchPics(np.transpose(cv_cover, (1, 0, 2)), np.transpose(cv_desk, (1, 0, 2)), opts)
bestH2to1, inliers = computeH_ransac(matches, locs1, locs2, opts)
composite_img = compositeH(bestH2to1, hp_cover, cv_desk, cv_cover)
