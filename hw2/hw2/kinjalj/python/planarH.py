import numpy as np
import cv2
import skimage.transform


def computeH(x1, x2):
    # Q2.2.1
    # Compute the homography between two sets of points
    num_points = x1.shape[0]
    m = np.zeros((2 * num_points, 3 * 3))
    for i in range(num_points):
        x_1, y_1, = x1[i, :]
        x_2, y_2, = x2[i, :]
        m[2 * i, :] = [-x_2, - y_2, -1, 0, 0, 0, x_1 * x_2, x_1 * y_2, x_1]
        m[2 * i + 1, :] = [0, 0, 0, -x_2, -y_2, -1, x_2 * y_1, y_1 * y_2, y_1]

    _, _, eigen_vectors = np.linalg.svd(m)
    eigen_vector = eigen_vectors[-1, :]  # svd returns in descending order
    H2to1 = eigen_vector.reshape(3, 3)
    return H2to1


def computeH_norm(x1, x2):
    # Q2.2.2
    x1_ = x1.astype(float)
    x2_ = x2.astype(float)
    num_points = x1.shape[0]

    # Compute the centroid of the points
    centroid1_ = np.mean(x1, axis=0)
    centroid2_ = np.mean(x2, axis=0)

    # Shift the origin of the points to the centroid
    x1_ = x1_ - centroid1_
    x2_ = x2_ - centroid2_

    # Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    # basically both x and y shouldn't be more than 1 apart, so divide by absolute maximum
    x1_max_0 = np.max(abs(x1_[:, 0]))
    x1_max_1 = np.max(abs(x1_[:, 1]))
    x2_max_0 = np.max(abs(x2_[:, 0]))
    x2_max_1 = np.max(abs(x2_[:, 1]))
    x1_[:, 0] = x1_[:, 0] / x1_max_0
    x1_[:, 1] = x1_[:, 1] / x1_max_1
    x2_[:, 0] = x2_[:, 0] / x2_max_0
    x2_[:, 1] = x2_[:, 1] / x2_max_1

    # Similarity transform 1
    # T1 = np.array([[1 / x1_max_0, 0, -centroid1_[0] / x1_max_0], [0, 1 / x1_max_1, -centroid1_[1] / x1_max_1], [0, 0, 1]])
    T1 = np.dot(np.c_[x1_, np.ones(num_points)].T, np.linalg.pinv(np.c_[x1, np.ones(num_points)].T))

    # Similarity transform 2
    # T2 = np.array([[1 / x2_max_0, 0, -centroid2_[0] / x2_max_0], [0, 1 / x2_max_1, -centroid2_[1] / x2_max_1], [0, 0, 1]])
    T2 = np.dot(np.c_[x2_, np.ones(num_points)].T, np.linalg.pinv(np.c_[x2, np.ones(num_points)].T))

    # Compute homography
    H2to1 = computeH(x1_, x2_)

    # Denormalization denormalized matrix = inv(T1) * H * T2
    H2to1 = np.linalg.pinv(T1).dot(H2to1).dot(T2)

    return H2to1


def computeH_ransac(matches, locs1, locs2, opts):
    # Q2.2.3
    # Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol  # the tolerance value for considering a point to be an inlier
    locs1 = locs1[matches[:, 0]]
    locs2 = locs2[matches[:, 1]]
    num_points = locs1.shape[0]
    max_inliers = 0
    bestH2to1 = np.empty([3, 3], dtype=float)
    inliers = np.empty([num_points, 1], dtype=int)

    for i in range(max_iters):
        points = np.sort(np.random.randint(low=0, high=num_points, size=4))
        p1 = locs1[points]
        p2 = locs2[points]
        try:
            H2to1 = computeH_norm(p1, p2)
            x1_hom = np.vstack((locs1.T, np.ones((1, num_points))))
            x2_hom = np.vstack((locs2.T, np.ones((1, num_points))))
            new_p1 = H2to1.dot(x2_hom)

            new_p1 = new_p1 / new_p1[-1, :]
            error = new_p1 - x1_hom

            dist = np.linalg.norm(error, axis=0)
            consensus = np.where(dist <= inlier_tol)  # Locations where matches are part of the consensus
            inliers[consensus] = 1  # Setting valid locations to 1
            not_consensus = np.where(dist > inlier_tol)  # Locations where matches are not a part of the consensus
            inliers[not_consensus] = 0
            num_inliers = consensus[0].shape[0]
            if num_inliers > max_inliers:
                max_inliers = num_inliers
                bestH2to1 = H2to1
        except:
            pass

    return bestH2to1, inliers


def compositeH(H2to1, template, img, cv_cover):
    # Create a composite image after warping the template image on top
    # of the image using the homography
    # Note that the homography we compute is from the image to the template;
    # x_template = H2to1*x_photo
    # For warping the template to the image, we need to invert it.

    # Create mask of same size as template
    # Warp template by appropriate homography
    template = cv2.resize(template, (cv_cover.shape[1], cv_cover.shape[0]))  # scale to size of cv_cover
    template[np.where(template == 0.0)] = 1

    # Warp mask by appropriate homography
    template = cv2.warpPerspective(template, np.linalg.inv(H2to1), (img.shape[1], img.shape[0]))
    non_black = np.nonzero(template)
    img[non_black] = 0

    # Use mask to combine the warped template and the image
    compositeimg = template + img
    cv2.imwrite('composite_img_tol_1.5_iter_2000.png', compositeimg)
    return compositeimg
