import argparse
import numpy as np
import matplotlib.pyplot as plt
from SubtractDominantMotion import SubtractDominantMotion

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=2e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.08, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/antseq.npy')
frame_height, frame_width, num_frames = seq.shape

for i in range(0, num_frames-1):
    print('Frame: {}'.format(i))
    mask = SubtractDominantMotion(seq[:, :, i], seq[:, :, i+1], threshold, int(num_iters), tolerance)
    result = np.nonzero(mask)

    if i in [30, 60, 90, 120]:
        plt.imshow(seq[:, :, i], cmap="gray")
        frame = plt.plot(result[1], result[0], ".", MarkerEdgeColor="blue", linewidth="0.1")
        plt.title("Frame: {}".format(i))
        plt.savefig("q2_3_ant_" + str(i) + ".png")
        plt.show()
