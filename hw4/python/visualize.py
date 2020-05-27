import helper
from submission import *

'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''


def main():
    pts = np.load('../data/some_corresp.npz')
    pts1 = pts["pts1"]
    pts2 = pts["pts2"]
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')
    M = np.max(im1.shape)
    F = eightpoint(pts1, pts2, M)

    intrinsic = np.load('../data/intrinsics.npz')
    K1, K2 = intrinsic['K1'], intrinsic['K2']
    E = essentialMatrix(F, K1, K2)

    coords = np.load('../data/templeCoords.npz')
    x1, y1 = coords['x1'], coords['y1']
    x2, y2 = [], []
    for i in range(x1.shape[0]):
        coor = epipolarCorrespondence(im1, im2, F, x1[i][0], y1[i][0])
        x2.append(coor[0])
        y2.append(coor[1])
    x2 = np.array(x2).reshape(-1, 1)
    y2 = np.array(y2).reshape(-1, 1)

    M2_ = helper.camera2(E)
    M1 = np.eye(4)[0:3]
    C1 = np.dot(K1, M1)

    temple_pts1 = np.hstack((x1, y1))
    temple_pts2 = np.hstack((x2, y2))

    plt.scatter(temple_pts1[:, 0], temple_pts1[:, 1])
    plt.show()

    plt.scatter(temple_pts2[:, 0], temple_pts2[:, 1])
    plt.show()

    err_min = np.inf
    w_ = None
    M2_best = None
    for i in range(M2_.shape[-1]):
        M2 = M2_[:, :, i]
        C2 = np.dot(K2, M2)
        w, err = triangulate(C1, np.hstack((x1, y1)), C2, np.hstack((x2, y2)))
        if np.min(w[:, -1]) > 0:
            M2_best = M2
            w_ = w
            # break
        # if err_min > err:
        #     err_min = err
        #     x_best = M2
        #     w_ = w

    C2 = np.dot(K2, M2_best)
    np.savez('q4_2.npz', F=F, M1=M1, M2=M2_best, C1=C1, C2=C2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(w_[:, 0], w_[:, 1], w_[:, 2])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


if __name__ == "__main__":
    main()
