import helper
import matplotlib.pyplot as plt
import numpy as np

from submission import eightpoint, essentialMatrix, triangulate

'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
pts = np.load('../data/some_corresp.npz')
pts1, pts2 = pts["pts1"], pts["pts2"]
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')
M = np.max(im1.shape)
F = eightpoint(pts1, pts2, M)
intrinsics = np.load('../data/intrinsics.npz')
K1, K2 = intrinsics['K1'], intrinsics['K2']
E = essentialMatrix(F, K1, K2)
M2s = helper.camera2(E)
M1 = np.eye(4)[:3]
C1 = np.dot(K1, M1)

err_min = np.inf
P_best = None
M2_best = None
print(M2s.shape[-1])

for i in range(M2s.shape[-1]):
    M2 = M2s[:, :, i]
    C2 = np.dot(K2, M2)
    P, err = triangulate(C1, pts1, C2, pts2)

    if len(P[P[:, 2] > 0]) != P.shape[0]:
        continue

    if err_min > err:
        err_min = err
        M2_best = M2
        P_best = P

    # if np.min(w[:, -1]) > 0:
    #     break

C2 = np.dot(K2, M2_best)
np.savez('q3_3.npz', M2=M2_best, C2=C2, P=P_best)
