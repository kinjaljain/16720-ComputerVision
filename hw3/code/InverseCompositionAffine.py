import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """

    # put your implementation here
    p = np.zeros(6)
    M = np.array([[1.0 + p[0], p[1], p[2]], [p[3], 1.0 + p[4], p[5]]])

    template_height, template_width = It.shape[0], It.shape[1]
    image_height, image_width = It1.shape[0], It1.shape[1]
    x_range = np.arange(0, template_width, 1)
    y_range = np.arange(0, template_height, 1)

    # get rectangular splines
    interpolated_template_spline = RectBivariateSpline(y_range, x_range, It)
    interpolated_current_image_spline = RectBivariateSpline(np.arange(0, image_height, 1),
                                                            np.arange(0, image_width, 1), It1)

    # get template grid
    xv, yv = np.meshgrid(x_range, y_range)
    xv, yv = xv.reshape(1, -1), yv.reshape(1, -1)
    template = np.vstack((xv, yv, np.ones((1, template_height * template_width))))
    grad_x = interpolated_current_image_spline.ev(yv, xv, dy=1).flatten()
    grad_y = interpolated_current_image_spline.ev(yv, xv, dx=1).flatten()
    jacobian_gd = np.vstack((xv * grad_x,
                             yv * grad_x,
                             grad_x,
                             xv * grad_y,
                             yv * grad_y,
                             grad_y))
    h = np.linalg.pinv(jacobian_gd)

    for i in range(num_iters):
        warped_image = np.dot(M, template)
        interpolated_template = interpolated_template_spline.ev(yv, xv).flatten()
        interpolated_current_image = interpolated_current_image_spline.ev(warped_image[1, :],
                                                                          warped_image[0, :]).flatten()
        difference = interpolated_template - interpolated_current_image
        dp = np.dot(h.T, difference)
        p += dp
        M = np.array([[1.0 + p[0], p[1], p[2]], [p[3], 1.0 + p[4], p[5]]])
        if np.linalg.norm(dp) <= threshold:
            break
    return M
