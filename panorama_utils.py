from scipy.signal import convolve2d
import numpy as np
import scipy.ndimage.filters as filters
import imageio
import skimage.color


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    build gaussian pyramid
    :param im: original gray scale image (2d array)
    :param max_levels: maximal levels of pyramid (int)
    :param filter_size: size of filter kernel (odd int)
    :return: tuple -  pyr, filter_vec. pyr is the pyramid (list) filter_vec
    is the filter kernel (2d array)
    """
    pyr = []
    filter_vec = build_filter(filter_size)
    for level in range(max_levels):
        pyr.append(im)
        im = reduce(im, filter_vec)
        if im.shape[0] < 16 or im.shape[1] < 16:
            break
    return pyr, filter_vec


def build_filter(filter_size):
    """
    build kernel filter
    :param filter_size: size of the kernel (odd int)
    :return: vector of size (1, filter_size) represent the kernel
    """
    filter_vec = np.array([1, 1])
    if filter_size > 1:
        for i in range(filter_size - 2):
            filter_vec = np.convolve(filter_vec, np.array([1, 1]),)
        return (np.array(filter_vec) / sum(filter_vec)).reshape(1, filter_size)
    filter_vec = np.array(filter_vec) / sum(filter_vec)
    return np.array([filter_vec])


def reduce(im, filter_vec):
    """
    reduce image
    :param im: original image (2d array)
    :param filter_vec: filter kernel
    :return: 2d array of size (row/2, col/2)
    """
    row_blur = filters.convolve(im, filter_vec, mode='constant', cval=0.0)
    reduce_im = np.apply_along_axis(lambda x: x[::2], 1, row_blur)
    col_blur = filters.convolve(reduce_im, filter_vec.T, mode='constant',
                            cval=0.0)
    return np.apply_along_axis(lambda x: x[::2], 0, col_blur)


def read_image(filename, representation):
    """
    read image with specified representation
    :param filename:  the filename to read
    :param representation: 1-grayscale, 2-rgb
    :return: the image with the specified representation
    """
    im = imageio.imread(filename)
    if is_rgb(im):
        if representation == 1:
            im = skimage.color.rgb2gray(im)
            return np.asarray(im, dtype=np.float64)
    im = np.asarray(im)
    if np.any((im < 1)&(im > 0)):
        return im
    return np.asarray(im, dtype=np.float64)/255


def is_rgb(im):
    """check if an image is rgb or not"""
    return len(im.shape) == 3


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im,
                     filter_size_mask):
    """
    blend two gray scale images using pyramid
    :param im1: gray scale image (2d array)
    :param im2: gray scale image (2d array)
    :param mask: mask for the blending (2d bool array)
    :param max_levels: maximal levels for pyramids
    :param filter_size_im: filter kernel for images
    :param filter_size_mask: filter kernel for mask
    :return: blend image (2d array)
    """
    lap1, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    lap2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)[0]
    g_mask = np.array(build_gaussian_pyramid(mask.astype('float64'), max_levels
                                             , filter_size_mask)[0])
    neg_mask = 1 - g_mask
    l_out = np.multiply(g_mask, lap1) + np.multiply(neg_mask, lap2)
    return np.clip(laplacian_to_image(l_out, filter_vec, np.ones(
        l_out.shape)), 0, 1)


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    build lalacian pyramid
    :param im: original gray scale image (2d array)
    :param max_levels: maximal levels of pyramid (int)
    :param filter_size: size of filter kernel (odd int)
    :return: tuple -  pyr, filter_vec. pyr is the pyramid (list) filter_vec
    is the filter kernel (2d array)
    """
    if (im.shape[0]) % 2:
        im = im[:-1, :]
    if (im.shape[1]) % 2:
        im = im[:, :-1]
    filter_vec = build_filter(filter_size)
    pyr = []
    for level in range(max_levels - 1):
        red = reduce(im, filter_vec)
        exp = expand(red, filter_vec)
        laplac = im - exp
        pyr.append(laplac)
        im = red
        if im.shape[0] <= 16 or im.shape[1] <= 16:
            break
    pyr.append(im)
    return pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    restore image using laplacian pyramid
    :param lpyr: laplacian pyramid (list)
    :param filter_vec: filter kernel (2d array)
    :param coeff:
    :return:
    """
    pyr = lpyr.copy()
    pyr = np.multiply(pyr, coeff)
    orig = []
    if len(pyr) == 1:
        return pyr[0]
    for p in range(len(lpyr) - 1, 0, -1):
        orig = expand(pyr[p], filter_vec)
        orig += pyr[p - 1]
        pyr[p - 1] = orig
    return orig


def expand(im, filter_vec):
    """
    expand image
    :param im: original image (2d array)
    :param filter_vec: filter kernel
    :return: 2d array of size (row*2, col*2)
    """
    col_num = list(range(1, im.shape[1] + 1))
    expand_rows = np.insert(im, col_num, 0, axis=1)
    expand_rows = filters.convolve(expand_rows, 2*filter_vec, mode='constant',
                                   cval=0.0)
    row_num = list(range(1, im.shape[0] + 1))
    expand_cols = np.insert(expand_rows, row_num, 0, axis=0)
    return filters.convolve(expand_cols, 2*filter_vec.T, mode='constant',
                            cval=0.0)
