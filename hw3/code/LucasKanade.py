import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
    # Put your implementation here
    p = p0
    x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
    template_height, template_width = It.shape[0], It.shape[1]
    image_height, image_width = It1.shape[0], It1.shape[1]

    # get rectangular splines
    interpolated_template_spline = RectBivariateSpline(np.arange(0, template_height, 1),
                                                       np.arange(0, template_width, 1), It)
    interpolated_current_image_spline = RectBivariateSpline(np.arange(0, image_height, 1),
                                                            np.arange(0, image_width, 1), It1)

    # get template grid
    x, y = np.meshgrid(np.arange(x1, x2, 1),  np.arange(y1, y2, 1))

    for i in range(num_iters):
        # p' <- p + dp
        p_x, p_y = x + p[1], y + p[0]

        # interpolate the template and current image using above splines
        interpolated_template = interpolated_template_spline.ev(y, x).reshape(-1, 1)
        interpolated_current_image = interpolated_current_image_spline.ev(p_y, p_x).reshape(-1, 1)

        # get dif between template and current image
        difference = interpolated_template - interpolated_current_image

        # take gradient in x and y on current image
        grad_y, grad_x = interpolated_current_image_spline.ev(p_y, p_x, dy=1), \
                         interpolated_current_image_spline.ev(p_y, p_x, dx=1)
        grad_y, grad_x = grad_y.reshape(1, -1), grad_x.reshape(1, -1)
        grad_combined = np.vstack((grad_x, grad_y))

        # get jacobian and multiply with gradient
        jacobian = np.eye(2)
        E = np.dot(grad_combined.T, jacobian)

        # find hessian, dp and recompute p
        hessian = np.dot(E.T, E)
        dp = np.dot(np.linalg.pinv(hessian), np.matmul(E.T, difference))
        dp = [x.item() for x in dp]
        p += dp

        # check if change is below threshold
        if np.linalg.norm(dp) <= threshold:
            break

    return p
