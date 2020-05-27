import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-1, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=5, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold
    
seq = np.load("../data/girlseq.npy")
rect = [280, 152, 330, 318]

rectangles = [np.asarray(rect)]
p0 = np.zeros(2)

frame_height, frame_width, num_frames = seq.shape
for i in range(0, num_frames-1):
    # print('Frame: {}'.format(i))
    p = LucasKanade(seq[:, :, i], seq[:, :, i+1], rect, threshold, int(num_iters))
    rect[0] += p[1]
    rect[2] += p[1]
    rect[1] += p[0]
    rect[3] += p[0]
    rect = np.asarray(rect)
    rectangles += [rect]
    p0 = p
    if i in [1, 20, 40, 60, 80]:
        patch_width = rect[2] - rect[0]
        patch_height = rect[3] - rect[1]
        plt.imshow(seq[:, :, i], cmap="gray")
        frame_rect = patches.Rectangle((rect[0], rect[1]), patch_width, patch_height,
                                       linewidth=1, edgecolor='r', facecolor='none')
        # Get the current reference
        g = plt.gca()
        g.add_patch(frame_rect)
        plt.title("Frame: {}".format(i))
        plt.savefig("q1_3_girl_" + str(i) + ".png")
        plt.show()

np.save("girlseqrects.npy", np.vstack(rectangles))
print('Done')
