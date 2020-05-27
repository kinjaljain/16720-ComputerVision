import numpy as np
from scipy.ndimage import affine_transform
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    """

    # put your implementation here
    p = np.zeros(6)
    M = np.array([[1.0 + p[0], p[1], p[2]], [p[3], 1.0 + p[4], p[5]]])

    template_height, template_width = It.shape[0], It.shape[1]
    image_height, image_width = It1.shape[0], It1.shape[1]
    x_range = np.arange(0, template_width, 1)
    y_range = np.arange(0, template_height, 1)
    interpolated_template_spline = RectBivariateSpline(y_range, x_range, It)
    interpolated_current_image_spline = RectBivariateSpline(np.arange(0, image_height, 1),
                                                            np.arange(0, image_width, 1), It1)
    xv, yv = np.meshgrid(x_range, y_range)
    template = np.vstack((xv.reshape(1, -1), yv.reshape(1, -1), np.ones((1, template_height * template_width))))

    for i in range(num_iters):
        # M = np.append(M, np.asarray([[0, 0, 1]]), axis=0)
        # warped_image = affine_transform(template, M)
        warped_image = np.dot(M, template)

        selected_x = np.bitwise_and((warped_image[0, :] >= 0), (warped_image[0, :] < template_width))
        selected_y = np.bitwise_and((warped_image[1, :] >= 0), (warped_image[1, :] < template_height))
        selected_mesh = np.bitwise_and(selected_x, selected_y)
        base, warped_image = template[:2, selected_mesh],  warped_image[:2, selected_mesh]

        interpolated_template = interpolated_template_spline.ev(base[1, :], base[0, :]).flatten()
        interpolated_current_image = interpolated_current_image_spline.ev(warped_image[1, :], warped_image[0, :]).flatten()

        # get dif between template and current image
        difference = interpolated_template - interpolated_current_image

        grad_x = interpolated_current_image_spline.ev(warped_image[1, :], warped_image[0, :], dy=1).flatten()
        grad_y = interpolated_current_image_spline.ev(warped_image[1, :], warped_image[0, :], dx=1).flatten()

        jacobian_gd = np.vstack((base[0, :] * grad_x,
                                 base[1, :] * grad_x,
                                 grad_x,
                                 base[0, :] * grad_y,
                                 base[1, :] * grad_y,
                                 grad_y))

        gd = np.dot(jacobian_gd, difference)
        hessian = np.dot(jacobian_gd, jacobian_gd.T)
        dp = np.dot(np.linalg.pinv(hessian), gd)
        p += dp
        M = np.array([[1+p[0], p[1], p[2]], [p[3], 1+p[4], p[5]]])
        if np.linalg.norm(dp) <= threshold:
            break

    return M
