"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import matplotlib.pyplot as plt
import numpy as np
import helper
import scipy

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    T = np.array([[1 / M, 0, 0],
                  [0, 1 / M, 0],
                  [0, 0, 1]])
    pts1 = pts1 / M
    pts2 = pts2 / M
    A = []
    for i in range(pts1.shape[0]):
        A.append(np.array([pts1[i][0] * pts2[i][0],
                           pts1[i][0] * pts2[i][1],
                           pts1[i][0],
                           pts1[i][1] * pts2[i][0],
                           pts1[i][1] * pts2[i][1],
                           pts1[i][1], pts2[i][0],
                           pts2[i][1], 1]))
    A = np.vstack(A)
    _, _, v = np.linalg.svd(A)
    F = v[-1, :].reshape(3, 3)
    F = helper._singularize(helper.refineF(F, pts1, pts2))
    unnormalized_F = np.dot(np.dot(T.T, F), T)
    return unnormalized_F


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    E = np.dot(K2.T, np.dot(F, K1))
    return E

'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    n, temp = pts1.shape
    P = np.zeros((n,3))
    Phomo = np.zeros((n,4))
    for i in range(n):
        x1 = pts1[i,0]
        y1 = pts1[i,1]
        x2 = pts2[i,0]
        y2 = pts2[i,1]
        A1 = x1*C1[2,:] - C1[0,:]
        A2 = y1*C1[2,:] - C1[1,:]
        A3 = x2*C2[2,:] - C2[0,:]
        A4 = y2*C2[2,:] - C2[1,:]
        A = np.vstack((A1,A2,A3,A4))
        u, s, vh = np.linalg.svd(A)
        p = vh[-1, :]
        p = p/p[3]
        P[i, :] = p[0:3]
        Phomo[i, :] = p
        # print(p)
    p1_proj = np.matmul(C1,Phomo.T)
    lam1 = p1_proj[-1,:]
    p1_proj = p1_proj/lam1
    p2_proj = np.matmul(C2,Phomo.T)
    lam2 = p2_proj[-1,:]
    p2_proj = p2_proj/lam2
    err1 = np.sum((p1_proj[[0,1],:].T-pts1)**2)
    err2 = np.sum((p2_proj[[0,1],:].T-pts2)**2)
    err = err1 + err2
    print(n)
    print(err)

    return P,err

'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    x1, y1 = np.rint(x1), np.rint(y1)
    pts = np.array([x1, y1, 1])
    epipolar_line = np.dot(F, pts)
    epipolar_line = epipolar_line / np.linalg.norm(epipolar_line)  # normalize

    # create window
    window_size = 12
    half_window_size = window_size // 2
    window = np.arange(-half_window_size, half_window_size + 1)
    window_y, window_x = np.meshgrid(window, window)
    h, w = im2.shape[0:2]

    # get points along the line
    y = np.arange(h)
    x = np.rint(-(epipolar_line[1] * y + epipolar_line[2]) / epipolar_line[0])

    valid = (x >= half_window_size) & (x < w - half_window_size) & \
            (y >= half_window_size) & (y < h - half_window_size)
    y, x = np.rint(y[valid]), np.rint(x[valid])

    # create gaussian weighting of the window
    sd = 5
    gaussian_weight = np.dot((np.exp(-0.5 * ((window_y ** 2 + window_x ** 2) / (sd ** 2)))), 1)
    gaussian_weight = gaussian_weight / (sd * np.sqrt(2 * np.pi))
    gaussian_weight = np.sum(gaussian_weight)

    # check where dif is minimal
    min_err = np.inf
    y_best, x_best = None, None
    im1_patch = im1[int(y1 - half_window_size): int(y1 + half_window_size + 1),
                int(x1 - half_window_size): int(x1 + half_window_size + 1), :]  # window
    for i in range(x.shape[0]):
        im2_patch = im2[int(y[i] - half_window_size): int(y[i] + half_window_size + 1),
                    int(x[i] - half_window_size): int(x[i] + half_window_size + 1), :]
        err = np.linalg.norm((im1_patch - im2_patch) * gaussian_weight)
        if err < min_err:
            min_err = err
            y_best = y[i]
            x_best = x[i]

    return x_best, y_best

'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M, nIters, tol):
    # Replace pass by your implementation
    max_inliers = -1
    F, inliers = None, None
    pts1_hom = np.vstack((np.transpose(pts1), np.ones((1, pts1.shape[0]))))
    pts2_hom = np.vstack((np.transpose(pts2), np.ones((1, pts1.shape[0]))))

    for i in range(nIters):
        print("iteration: ", i + 1)
        indices = np.random.choice(pts1.shape[0], 8, replace=False)
        # indices = np.random.randint(0, pts1.shape[0], (8,))
        print(indices)
        F_cur = eightpoint(pts1[indices, :], pts2[indices, :], M)
        # get the epipolar lines
        epipolar_lines = np.dot(F_cur, pts1_hom)
        epipolar_lines = epipolar_lines / np.sqrt(np.sum(epipolar_lines[:2, :] ** 2, axis=0))

        # get the deviation of pts2 from the epipolar lines
        dist = abs(np.sum(pts2_hom * epipolar_lines, axis=0))

        # determine the inliners
        tmp_inliers = np.transpose(dist < tol)

        if tmp_inliers[tmp_inliers].shape[0] > max_inliers:
            max_inliers = tmp_inliers[tmp_inliers].shape[0]
            F = F_cur
            inliers = tmp_inliers

        print(max_inliers)
    return F, inliers

        # for f in Fs:
        #     pts = np.vstack((pts1.T, np.ones([1, pts1.shape[0]])))
        #     # pts1_homo = np.vstack((np.transpose(pts1), np.ones((1, N))))
        #     # print(f.shape)
        #     # print(pts.shape)
        #     epipolar_lines = np.dot(f, pts)
        #     # print(epipolar_lines.shape)
        #     # epipolar_lines = epipolar_lines / np.linalg.norm(epipolar_lines[:2])
        #     epipolar_lines = epipolar_lines / np.sqrt(np.sum(epipolar_lines[:2, :] ** 2, axis=0))
        #
        #     # pts2_homo = np.vstack((np.transpose(pts2), np.ones((1, N))))
        #     pts2 = np.vstack((pts2.T, np.ones([1, pts1.shape[0]])))
        #     # dist = abs(np.sum(pts2 * epipolar_lines, axis=0))
        #     dist = abs(np.sum(pts2 * epipolar_lines, axis=0))
        #
        #     # determine the inliners
        #     inliers_ = dist < tol
        #     # num_inliers = np.sum(inliers_)
        #     # if num_inliers > max_inliers:
        #     #     max_inliers = num_inliers
        #     #     F = f
        #     #     inliers = inliers_
        #     tot_inliers = inliers_[inliers_.T].shape[0]
        #     if tot_inliers > max_inliers:
        #         print('in if')
        #         bestF = F
        #         max_inliers = tot_inliers
        #         inliers = inliers_[inliers_.T]

    # return F, inliers


'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    theta = np.linalg.norm(r)
    v = np.vstack(r / theta)
    sin = np.sin(theta)
    cos = np.cos(theta)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    Ksquare = np.dot(K, K.T)
    I = np.eye(3)
    R = I + sin * K + (1 - cos) * Ksquare
    return R
    # theta = np.linalg.norm(r)
    # I = np.eye(3)
    # zero_tolerance = 1e-2
    #
    # if theta < zero_tolerance:
    #     return I
    #
    # v = np.vstack(r / theta)
    # sin = np.sin(theta)
    # cos = np.cos(theta)
    # K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    # Ksquare = np.dot(K, K.T)
    #
    # R = I * cos + sin * K + (1 - cos) * Ksquare
    # return R

'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    A = (R - R.T) / 2
    p = np.vstack([A[2, 1], A[0, 2], A[1, 0]])
    s = np.linalg.norm(p)
    c = (R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2

    if s == 0 and c == 1:
        return np.zeros((3, 1))

    elif s == 0 and c == -1:
        I = np.eye(3)
        r_plus_i = R + I
        i = np.where(r_plus_i.any(axis=0))[0]
        v = (R + I)[:, i[0]]
        u = v / np.linalg.norm(v)
        r = u * np.pi
        if np.linalg.norm(r) == np.pi and ((r[0] == 0 and r[1] == 0 and r[2] < 0)
                                           or (r[0] == 0 and r[1] < 0) or
                                           (r[0] < 0)):
            r = -r
        return r

    else:
        u = p / s
        theta = np.arctan2(s, c)
        return u * theta


'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation

    N = p1.shape[0]
    P = x[:-6].reshape(-1, 3)
    r = x[-6:-3]
    t = np.vstack(x[-3:])  # t:3*1
    R = rodrigues(r)
    M2 = np.hstack([R, t])

    C1 = np.dot(K1, M1)
    C2 = np.dot(K2, M2)

    p1_hat = np.zeros(p1.shape)
    p2_hat = np.zeros(p1.shape)
    err = np.zeros(N)
    for i in range(N):
        point = np.append(P[i, :], 1)
        p1_hat_i = np.dot(C1, point)
        p1_hat_i = p1_hat_i / p1_hat_i[-1]
        p1_hat_i = p1_hat_i[0:-1]
        p2_hat_i = np.dot(C2, point)
        p2_hat_i = p2_hat_i / p2_hat_i[-1]
        p2_hat_i = p2_hat_i[0:-1]
        p1_hat[i, :] = p1_hat_i
        p2_hat[i, :] = p2_hat_i
    residuals = np.concatenate([(p1 - p1_hat).reshape([-1]), (p2 - p2_hat).reshape([-1])]).reshape(-1,1)  # residuals: 4N*1
    return residuals.flatten()


'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    R2, t2 = invRodrigues(M2_init[:, :3]).flatten(), M2_init[:, 3].flatten()
    x_init = np.hstack((P_init.flatten(), R2, t2))
    residualError = lambda x: rodriguesResidual(K1, M1, p1, K2, p2, x)
    print(residualError(x_init))
    x_best, _ = scipy.optimize.leastsq(residualError, x_init)
    P, R2, t2 = x_best[:-6].reshape((-1, 3)), rodrigues(x_best[-6:-3].reshape((3, 1))), x_best[-3:].reshape((3, 1))
    M2 = np.hstack((R2, t2))

    return M2, P

'''
Q6.1 Multi-View Reconstruction of keypoints.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx3 matrix with the 2D image coordinates and confidence per row
            C2, the 3x4 camera matrix
            pts2, the Nx3 matrix with the 2D image coordinates and confidence per row
            C3, the 3x4 camera matrix
            pts3, the Nx3 matrix with the 2D image coordinates and confidence per row
    Output: P, the Nx3 matrix with the corresponding 3D points for each keypoint per row
            err, the reprojection error.
'''
def MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3, Thres):
    # Replace pass by your implementation
    P12, err12 = triangulate(C1, pts1[:, :2], C2, pts2[:, :2])
    # P23, err23 = triangulate(C2, pts2[:,:2], C3, pts3[:,:2])
    # P13, err13 = triangulate(C1, pts1[:,:2], C2, pts3[:,:2])

    return P12, err12


if __name__ == "__main__":

    # Q2.1
    pts = np.load('../data/some_corresp.npz')
    pts1, pts2 = pts["pts1"], pts["pts2"]
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')
    M = np.max(im1.shape)
    F = eightpoint(pts1, pts2, M)
    print(F)
    # helper.displayEpipolarF(im1, im2, F)
    np.savez('q2_1.npz', F=F, M=M)

    # Q3.1
    intrinsics = np.load('../data/intrinsics.npz')
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    E = essentialMatrix(F, K1, K2)
    print(E)

    # Q3.2
    C1 = np.eye(4)[:3]
    C2 = helper.camera2(E)[1]
    print(C1, C2)
    w, e = triangulate(C1, pts1, C2, pts2)
    print(w, e)

    # Q4.1
    # helper.epipolarMatchGUI(im1, im2, F)
    np.savez('q4_1.npz', F=F, pts1=pts1, pts2=pts2)

    # Q5.1
    pts = np.load('../data/some_corresp_noisy.npz')
    pts1, pts2 = pts['pts1'], pts['pts2']
    nIters = 50
    tol = 1
    print(pts1.shape)
    print(pts2.shape)
    print("***")
    FRansac, inliers = ransacF(pts1, pts2, M, nIters, tol)
    # F = eightpoint(pts1, pts2, M)
    # pts1Valid = pts1[np.where(inliers is True)]
    # pts2Valid = pts2[np.where(inliers is True)]
    E = essentialMatrix(FRansac, K1, K2)
    M2_ = helper.camera2(E)
    M1 = np.eye(4)[:3]
    C1 = np.dot(K1, M1)
    pts1 = pts1[inliers]
    pts2 = pts2[inliers]

    err_min = np.inf
    w_ = None
    M2_best = None
    print(M2_.shape[-1])

    for i in range(M2_.shape[-1]):
        M2 = M2_[:, :, i]
        C2 = np.dot(K2, M2)
        w, err = triangulate(C1, pts1, C2, pts2)
        if err < err_min and np.min(w[:, -1]) >= 0:
            err_min = err
            M2_best = M2
            w_ = w

        # if np.min(w[:, -1]) > 0:
        #     M2_best = M2
        #     w_ = w

    C2 = np.dot(K2, M2_best)
    M2_init = M2_best
    P_init = w_

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(P_init[:, 0], P_init[:, 1], P_init[:, 2])
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # plt.show()

    P_init, err = triangulate(C1, pts1, C2, pts2)
    M2Final, wFinal = bundleAdjustment(K1, M1, pts1, K2, M2_init, pts2, P_init)

    np.savez('q4_2.npz', F=F, M1=M1, M2=M2Final, C1=C1, C2=C2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(wFinal[:, 0], wFinal[:, 1], wFinal[:, 2])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


    # Q6.1
    time_0 = np.load('../data/q6/time0.npz')
    pts1 = time_0['pts1']  # Nx3 matrix
    pts2 = time_0['pts2']  # Nx3 matrix
    pts3 = time_0['pts3']  # Nx3 matrix
    M1_0 = time_0['M1']
    M2_0 = time_0['M2']
    M3_0 = time_0['M3']
    K1_0 = time_0['K1']
    K2_0 = time_0['K2']
    K3_0 = time_0['K3']
    C1_0 = np.dot(K1_0, M1_0)
    C2_0 = np.dot(K1_0, M2_0)
    C3_0 = np.dot(K1_0, M3_0)
    Thres = 575
    P_mv, err_mv = MultiviewReconstruction(C1_0, pts1, C2_0, pts2, C3_0, pts3, Thres)
    M2_opt, P2_opt = bundleAdjustment(K2_0, M2_0, pts2[:, :2], K3_0, M3_0, pts3[:, :2], P_mv)
    helper.plot_3d_keypoint(P2_opt)
