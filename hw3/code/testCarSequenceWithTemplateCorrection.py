import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation
parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

seq = np.load("../data/carseq.npy")
original_rect = [59, 116, 145, 151]
rect = [59, 116, 145, 151]
rect_corrected = [59, 116, 145, 151]

rectangles = np.asarray(rect)
corrected_rectangles = np.asarray(rect_corrected)
both = np.vstack((rectangles, corrected_rectangles))
p0 = np.zeros(2)
p0_corrected = np.zeros(2)

frame_height, frame_width, num_frames = seq.shape
for i in range(0, num_frames-1):
    # print('Frame: {}'.format(i))
    p = LucasKanade(seq[:, :, i], seq[:, :, i+1], rect, threshold, int(num_iters), p0)
    p_corrected = LucasKanade(seq[:, :, 0], seq[:, :, i], original_rect, threshold, int(num_iters), p0_corrected)
    rect[0] += p[1]
    rect[2] += p[1]
    rect[1] += p[0]
    rect[3] += p[0]
    rect = np.asarray(rect)
    # rectangles += [rect]

    rect_corrected[0] = original_rect[0] + p_corrected[1]
    rect_corrected[2] = original_rect[2] + p_corrected[1]
    rect_corrected[1] = original_rect[1] + p_corrected[0]
    rect_corrected[3] = original_rect[3] + p_corrected[0]
    rect_corrected = np.asarray(rect_corrected)
    # corrected_rectangles += [rect_corrected]
    both = np.vstack((both, rect, rect_corrected))

    p0 = p
    p0_corrected = p_corrected
    if i in [1, 100, 200, 300, 400]:
        patch_width = rect[2] - rect[0]
        patch_height = rect[3] - rect[1]
        plt.imshow(seq[:, :, i], cmap="gray")
        frame_rect = patches.Rectangle((rect[0], rect[1]), patch_width, patch_height, linewidth=1,
                                       edgecolor='b', facecolor='none')
        # Get the current reference
        g = plt.gca()
        g.add_patch(frame_rect)

        patch_width = rect_corrected[2] - rect_corrected[0]
        patch_height = rect_corrected[3] - rect_corrected[1]
        plt.imshow(seq[:, :, i], cmap="gray")
        frame_rect = patches.Rectangle((rect_corrected[0], rect_corrected[1]), patch_width, patch_height,
                                       linewidth=1, edgecolor='r', facecolor='none')
        # Get the current reference
        g = plt.gca()
        g.add_patch(frame_rect)
        plt.title('Frame: {}'.format(i))
        plt.savefig("q1_4_car_corrected_" + str(i) + ".png")
        plt.show()

np.save("carseqrects-wcrt.npy", both)
print('Done')
