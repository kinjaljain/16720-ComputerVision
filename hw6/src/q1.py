# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

# Imports
import numpy as np
import skimage.io
import skimage.color
from matplotlib import pyplot as plt
from utils import integrateFrankot, enforceIntegrability


def renderNDotLSphere(center, rad, light, pxSize, res):

    """
    Question 1 (b)

    Render a sphere with a given center and radius. The camera is
    orthographic and looks towards the sphere in the negative z
    direction. The camera's sensor axes are centerd on and aligned
    with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,)

    rad : float
        The radius of the bowl

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """
    sphere_x, sphere_y, sphere_z = center[0], center[1], center[2]
    x, y = res[0], res[1]
    x_ = np.arange(0, x, 1)
    y_ = np.arange(0, y, 1)
    n_x = pxSize * x
    n_y = pxSize * y
    center_x, center_y = n_x/2, n_y/2
    a = np.arange(-center_x, center_x, pxSize)
    b = np.arange(-center_y, center_y, pxSize)
    r_sqr = rad * rad

    image = np.zeros((x, y))
    for x, im_x in zip(a, x_):
        for y, im_y in zip(b, y_):
            z_sqr = r_sqr - (np.power(x-sphere_x, 2) + np.power(y-sphere_y, 2))
            if z_sqr < 0:
                continue
            z = np.sqrt(z_sqr) + sphere_z
            normal = np.asarray([2*(x-sphere_x), 2*(y-sphere_y), 2*(z-sphere_z)])
            normal = normal / np.linalg.norm(normal)
            i = np.dot(normal, light)
            if i < 0:
                continue
            image[im_x, im_y] = i
    return image


def loadData(path = "../data/"):

    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Paramters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """
    sources = np.load(path + 'sources.npy')
    imgs = []
    for i in range(1, 8):
        imgs.append(skimage.io.imread(path + 'input_' + str(i) + '.tif'))
    s = imgs[0].shape[:-1]
    imgs = [skimage.color.rgb2xyz(img)[:, :, 1].flatten() for img in imgs]
    imgs = np.asarray(imgs).reshape(7, -1)

    I = imgs
    L = sources.T
    s = s

    # u, sv, vh = np.linalg.svd(I, full_matrices=False)
    # print(sv)
    #
    # L_inv = np.linalg.pinv(L)
    # B = np.dot(I.T, L_inv).T
    # print(B)

    return I, L, s


def estimatePseudonormalsCalibrated(I, L):

    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """
    # both the approaches give same results
    # a = np.linalg.lstsq(L.T, I)
    # B = a[0]


    a = np.linalg.pinv(L.T)
    B = np.dot(a, I)
    return B


def estimateAlbedosNormals(B):

    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''

    P = B.shape[1]
    albedos = np.zeros(P)
    normals = np.zeros((3, P))
    for i in range(B.shape[1]):
        n = B[:, i]
        albedos[i] = np.linalg.norm(n)
        normals[:, i] = n/albedos[i]

    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):

    """
    Question 1 (f)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `coolwarm` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """

    albedoIm = albedos.reshape(s)
    normals = normals.T
    normals = (normals + 1) / 2
    normalIm = normals.reshape((s[0], s[1], 3))
    plt.imshow(albedoIm, cmap='gray')
    plt.show()
    plt.imshow(normalIm, cmap='rainbow')
    plt.show()
    return albedoIm, normalIm


def estimateShape(normals, s):

    """
    Question 1 (i)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """
    n1, n2, n3 = normals[0, :], normals[1, :], -normals[2, :]
    zx = (-n1/n3).reshape(s)
    zy = (-n2/n3).reshape(s)
    surface = integrateFrankot(zx, zy)
    return surface


def plotSurface(surface):

    """
    Question 1 (i)

    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    Returns
    -------
        None

    """
    x, y = surface.T.shape
    X = np.arange(0, x, 1)
    Y = np.arange(0, y, 1)
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, surface, cmap='coolwarm')
    ax.set_title('Surface plot')
    plt.show()

if __name__ == '__main__':

    # Put your main code here

    # q1.b
    # root_3 = np.sqrt(3)
    # # 0.75cm = 7500 micrometer
    # image1 = renderNDotLSphere(center=np.zeros(3), rad=7500, light=np.asarray([1/root_3, 1/root_3, 1/root_3]),
    #                           pxSize=7, res=np.asarray([3840, 2160]))
    # plt.imshow(image1.T, cmap='gray')
    # # plt.title("1,1,1")
    # plt.show()
    # image2 = renderNDotLSphere(center=np.zeros(3), rad=7500, light=np.asarray([1/root_3, -1/root_3, 1/root_3]),
    #                           pxSize=7, res=np.asarray([3840, 2160]))
    # plt.imshow(image2.T, cmap='gray')
    # # plt.title("1,-1,1")
    # plt.show()
    # image3 = renderNDotLSphere(center=np.zeros(3), rad=7500, light=np.asarray([-1/root_3, -1/root_3, 1/root_3]),
    #                           pxSize=7, res=np.asarray([3840, 2160]))
    # plt.imshow(image3.T, cmap='gray')
    # # plt.title("-1,-1,1")
    # plt.show()

    # q1.c, q1.d
    I, L, s = loadData()

    # q1.e
    B = estimatePseudonormalsCalibrated(I, L)
    print(B)
    albedos, normals = estimateAlbedosNormals(B)
    print(albedos, normals)

    # q1.f
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)

    # q1.i
    surface = estimateShape(normals, s)
    plotSurface(surface)
