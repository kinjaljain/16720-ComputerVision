import numpy as np
import cv2
from opts import get_opts

#Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac, compositeH
from loadVid import loadVid

#Write script for Q3.1
opts = get_opts()
cv_cover = cv2.imread('../data/cv_cover.jpg')  # 440, 350, 3
ar_source = loadVid('../data/ar_source.mov')
np.save("ar_source.npy", ar_source)
book = loadVid('../data/book.mov')
np.save("book.npy", book)
# ar_source = np.load('ar_source.npy', allow_pickle=True)  # 511, 360, 640, 3
# book = np.load('book.npy', allow_pickle=True)  # 641, 480, 640, 3

#Processing the video one frame at a time
cap = cv2.VideoWriter('ar.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15.0, (book.shape[0], book.shape[1]))
for frame_num in range(ar_source.shape[0]):
    print(frame_num)
    frame_source = ar_source[frame_num]
    frame_book = book[frame_num]
    matches, locs1, locs2 = matchPics(np.transpose(cv_cover, (1, 0, 2)), np.transpose(frame_book, (1, 0, 2)), opts)
    bestH2to1, inliers = computeH_ransac(matches, locs1, locs2, opts)
    frame_source = frame_source[48:-48, 145:495]  # crop black part from top and bottom of ar_cource video
    composite_img = compositeH(bestH2to1, frame_source, frame_book, cv_cover)
    # cv2.imwrite('ar_final/frame_{}.png'.format(frame_num), composite_img)
    cap.write(composite_img)
cap.release()
