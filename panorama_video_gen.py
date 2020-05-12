

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import map_coordinates
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass
import shutil
from imageio import imwrite

import panorama_utils

K = 0.04


def harris_corner_detector(im):
    """
    Detects harris corners.
    Make sure the returned coordinates are x major!!!
    :param im: A 2D array representing an image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y]
    coordinates of the ith corner points.
    """
    im_x, im_y = conv_der(im)
    r = non_maximum_suppression(create_r_mat(im_x, im_y))
    piks = np.where(r == 1)
    points = np.array(np.array([piks[1], piks[0]]).T)
    return points


def create_r_mat(im_x, im_y):
    """
    create response matrix (det(M) - K(Tr(M))
    :param im_x: derivative in x direction. 2-D array
    :param im_y: derivative in y direction. 2-D array
    :return: response matrix. 2-D array
    """
    im_xx = panorama_utils.blur_spatial(im_x * im_x, 3)
    im_xy = panorama_utils.blur_spatial(im_x * im_y, 3)
    im_yx = panorama_utils.blur_spatial(im_y * im_x, 3)
    im_yy = panorama_utils.blur_spatial(im_y * im_y, 3)
    det = np.multiply(im_xx, im_yy) - np.multiply(im_yx, im_xy)
    trace = im_xx + im_yy
    r_mat = det - K*(trace**2)
    return r_mat


def conv_der(im):
    """
    derivative using convolution
    :param im: gray scale image. 2-D array
    :return: derivative of image. 2-D array
    """
    x_kernel = np.array([[1, 0, -1]])
    y_kernel = x_kernel.reshape(3, 1)
    dx = convolve2d(im, x_kernel, mode='same', boundary='symm')
    dy = convolve2d(im, y_kernel, mode='same',  boundary='symm')
    return dx, dy


def sample_descriptor(im, pos, desc_rad):
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image.
    :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
    """
    k = 1 + desc_rad * 2
    res = np.zeros((pos.shape[0], k, k))
    for i in range(pos.shape[0]):
        patch = make_patch(im, pos[i, :], desc_rad)
        res[i, :, :] = patch
    return res


def make_patch(im, p, desc_rad):
    """
    create a patch for one point
    :param im: gray scale image
    :param p: point (1,2) array
    :param desc_rad: radius of patch
    :return: a normalized and interpolated patch of size (1 + desc_rad*2,
    1 + desc_rad*2)
    """
    k = 1 + desc_rad*2
    x_vec = np.arange((p[0]) - desc_rad, (p[0]) + desc_rad + 1)
    y_vec = np.arange((p[1]) - desc_rad, (p[1]) + desc_rad + 1)
    mesh = np.meshgrid(x_vec, y_vec)
    patch = map_coordinates(im, [mesh[1].flatten(), mesh[0].flatten()],
                            order=1, prefilter=False).reshape(k,k)
    patch = patch - np.mean(patch)
    std = np.linalg.norm(patch)
    if std != 0:
        return patch/std
    return patch


def find_features(pyr):
    """
    Detects and extracts feature points from a pyramid.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return: A list containing:
                1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                   These coordinates are provided at the pyramid level pyr[0].
                2) A feature descriptor array with shape (N,K,K)
    """
    points = spread_out_corners(pyr[0], 7, 7, 20)
    desc = sample_descriptor(pyr[2], points/4, 3)
    return [points, desc]


def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
                1) An array with shape (M,) and dtype int of matching indices in desc1.
                2) An array with shape (M,) and dtype int of matching indices in desc2.
    """
    m_d1, m_d2,  s = create_matches(desc1, desc2, min_score)
    intersect = m_d1 & m_d2 & s
    res = np.where(intersect==1)
    return [res[0], res[1]]


def create_matches(desc1, desc2, min_score):
    """
    create 3 boolean matrices according to 3 conditions for matching
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: 3 bolean matrices of size (desc1.shape[0], desc2.shape[0]),
    each one represent a different condition - desc1 best 2 matches,
    desc2 best 2 matches and matches that are > min_score
    """
    desc1 = desc1.reshape(desc1.shape[0], desc1.shape[1]*desc1.shape[2])
    desc2 = desc2.reshape(desc2.shape[0], desc2.shape[1]*desc2.shape[2])
    matches = np.dot(desc1, desc2.T)
    m_d1 = np.zeros(matches.shape)
    m_d2 = np.zeros(matches.shape)
    m_score = np.zeros(matches.shape)
    d1_matches = np.argpartition(matches, -2, axis=1)[:, -2:]
    d1_row = np.array([np.arange(desc1.shape[0]), np.arange(desc1.shape[0])])
    m_d1[d1_row.T, d1_matches] = 1
    d2_matches = np.argpartition(matches.T, -2, axis=1)[:, -2:]
    d2_cols = np.array([np.arange(desc2.shape[0]), np.arange(desc2.shape[0])])
    m_d2[d2_matches, d2_cols.T] = 1
    m_score[np.where(matches > min_score)] = 1
    return m_d1.astype('bool'), m_d2.astype('bool'), m_score.astype('bool')


def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
    """
    homg_pos = np.ones((pos1.shape[0], 3))
    homg_pos[:, :-1] = pos1
    new_pos = (H12 @ homg_pos.T).T
    new_pos[:, 0] = new_pos[:, 0]/new_pos[:, 2]
    new_pos[:, 1] = new_pos[:, 1]/new_pos[:, 2]
    return new_pos[:, :-1]


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
    Computes homography between two sets of points using RANSAC.
    :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
                1) A 3x3 normalized homography matrix.
                2) An Array with shape (S,) where S is the number of inliers,
                    containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """
    inliers = np.array([])
    k = 0
    while k < num_iter:
        ind_1 = np.random.randint(0, points1.shape[0])
        ind_2 = np.random.randint(0, points1.shape[0])
        p1 = np.array([points1[ind_1, :], points1[ind_2, :]])
        p2 = np.array([points2[ind_1, :], points2[ind_2, :]])
        if translation_only:
            h12 = estimate_rigid_transform(np.array([p1[1, :]]), np.array([
                p2[1, :]]), translation_only)
        else:
            h12 = estimate_rigid_transform(p1, p2, translation_only)
        trans_p1 = apply_homography(points1, h12)
        err = np.sqrt((trans_p1[:, 0] - points2[:, 0])**2 + (trans_p1[:,
                                                             1] - points2[:, 1])**2)
        good_ind = np.where(err < inlier_tol)
        if len(good_ind[0]) > len(inliers):
            inliers = good_ind[0]
        k += 1
    h12 = estimate_rigid_transform(points1[inliers, :], points2[inliers, :],
                                   translation_only)
    return [h12, inliers]


def display_matches(im1, im2, points1, points2, inliers):
    """
    Dispalay matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """
    stack_im = np.hstack((im1, im2))
    points2[:, 0] = points2[:, 0] + im1.shape[1]
    plt.imshow(stack_im, cmap='gray')
    plt.scatter(points1[:, 0], points1[:, 1], c='r', s=1)
    plt.scatter(points2[:, 0], points2[:, 1], c='r', s=1)
    for i in range(len(points1)):
        x = points1[i][0], points2[i][0]
        y = points1[i][1], points2[i][1]
        if i not in inliers:
            plt.plot(x, y, mfc='r', c='b', lw=.4, ms=5, marker='o')
        else:
            plt.plot(x, y, mfc='r', c='y', lw=.6, ms=5, marker='o')
    plt.show()


def accumulate_homographies(H_succesive, m):
    """
    Convert a list of succesive homographies to a
    list of homographies to a common reference frame.
    :param H_successive: A list of M-1 3x3 homography
      matrices where H_successive[i] is a homography which transforms points
      from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to
      accumulate the given homographies.
    :return: A list of M 3x3 homography matrices,
      where H2m[i] transforms points from coordinate system i to coordinate system m
    """
    H2m = []
    if m > 0:
        accum_mat = H_succesive[m - 1]
        accum_mat = accum_mat/accum_mat[2,2]
        H2m.insert(0, accum_mat)
        for i in range(m-1, 0, -1):
            accum_mat = accum_mat @ H_succesive[i - 1]
            accum_mat = accum_mat/accum_mat[2,2]
            H2m.insert(0, accum_mat)
    H2m.append(np.eye(3))
    accum_mat = np.linalg.inv(H_succesive[m])
    H2m.append(accum_mat)
    for i in range(m + 1, len(H_succesive)):
        accum_mat = accum_mat @ np.linalg.inv(H_succesive[i])
        accum_mat = accum_mat/accum_mat[2,2]
        H2m.append(accum_mat)
    return H2m


def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
     and the second row is the [x,y] of the bottom right corner
    """
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]])
    new_coord = apply_homography(corners, homography)
    min_x = np.floor(np.min(new_coord[:, 0]))
    max_x = np.ceil(np.max(new_coord[:, 0]))
    min_y = np.floor(np.min(new_coord[:, 1]))
    max_y = np.ceil(np.max(new_coord[:, 1]))
    return np.array([[min_x, min_y], [max_x, max_y]]).astype('int')


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """
    corners = compute_bounding_box(homography, image.shape[1], image.shape[0])
    x_vec = np.arange(corners[0, 0], corners[1, 0])
    y_vec = np.arange(corners[0, 1], corners[1, 1])
    x_coord, y_coord = np.meshgrid(x_vec, y_vec)
    x_coord, y_coord = x_coord.astype('int64'), y_coord.astype('int64')
    inv_h = np.linalg.inv(homography)
    back_coord = apply_homography(np.array([x_coord.flatten(),
                                            y_coord.flatten()]).T,  inv_h)
    res = map_coordinates(image, [back_coord[:, 1], back_coord[:, 0]],
                          order=1,  prefilter=False).reshape \
        (y_vec.shape[0], x_vec.shape[0])
    return res


def warp_image(image, homography):
    """
    Warps an RGB image with a given homography.
    :param image: an RGB image.
    :param homography: homograhpy.
    :return: A warped image.
    """
    return np.dstack([warp_channel(image[...,channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
    Filters rigid transformations encoded as homographies by the amount of translation from left to right.
    :param homographies: homograhpies to filter.
    :param minimum_right_translation: amount of translation below which the transformation is discarded.
    :return: filtered homographies..
    """
    translation_over_thresh = [0]
    last = homographies[0][0,-1]
    for i in range(1, len(homographies)):
        if homographies[i][0,-1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0,-1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
    Computes rigid transforming points1 towards points2, using least squares method.
    points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
    :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
    :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
    :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
    :return: A 3x3 array with the computed homography.
    """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2,:2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
    Finds local maximas of an image.
    :param image: A 2D array representing an image.
    :return: A boolean array with the same shape as the input image, where True indicates local maximum.
    """
    # Find local maximas.
    neighborhood = generate_binary_structure(2,2)
    local_max = maximum_filter(image, footprint=neighborhood)==image
    local_max[image<(image.max()*0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num)+1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:,0], centers[:,1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
    Splits the image im to m by n rectangles and uses harris_corner_detector on each.
    :param im: A 2D array representing an image.
    :param m: Vertical number of rectangles.
    :param n: Horizontal number of rectangles.
    :param radius: Minimal distance of corner points from the boundary of the image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    corners = [np.empty((0,2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n+1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m+1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j+1], x_bound[i]:x_bound[i+1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis,:]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:,0]>radius) & (corners[:,0]<im.shape[1]-radius) &
             (corners[:,1]>radius) & (corners[:,1]<im.shape[0]-radius))
    ret = corners[legit,:]
    return ret


class PanoramicVideoGenerator:
    """
    Generates panorama from a set of images.
    """

    def __init__(self, data_dir, file_prefix, num_images):
        """
        The naming convention for a sequence of images is file_prefixN.jpg,
        where N is a running number 001, 002, 003...
        :param data_dir: path to input images.
        :param file_prefix: see above.
        :param num_images: number of images to produce the panoramas with.
        """
        self.file_prefix = file_prefix
        self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        self.images = []
        print('found %d images' % len(self.files))

    def align_images(self, translation_only=False):
        """
        compute homographies between all images to a common coordinate system
        :param translation_only: see estimte_rigid_transform
        """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for file in self.files:
            image = panorama_utils.read_image(file, 1)

            # plt.imshow(image, cmap='gray')
            # plt.show()
            self.images.append(image)
            self.h, self.w = image.shape
            pyramid, _ = panorama_utils.build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))


        # Compute homographies between successive pairs of images.
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            im1 = panorama_utils.read_image(self.files[i], 1)
            im2 = panorama_utils.read_image(self.files[i], 1)
            points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, .8)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, 100, 10,
                                             translation_only)


            # Uncomment for debugging: display inliers and outliers among matching points.
            # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
        self.homographies = self.homographies[self.frames_for_panoramas]


    def generate_panoramic_images(self, number_of_panoramas):
        """
        combine slices from input images to panoramas.
        :param number_of_panoramas: how many different slices to take from each input image
        """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
        warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
            # homography warps the slice center to the coordinate system of the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
            # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) * panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
        self.x_strips = x_strip_boundary
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = panorama_utils.read_image(self.files[frame_index], 2)
            warped_image = warp_image(image, self.homographies[i])

            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            y_bottom = y_offset + warped_image.shape[0]

            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i:i + 2]
                image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
                x_end = boundaries[0] + image_strip.shape[1]
                self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip
                # plt.imshow(self.panoramas[panorama_index, :, :])
                # plt.show()


        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view between the left most image and the right most image
        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        print(crop_left, crop_right)
        assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
        self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

    def save_panoramas_to_video(self):
        """
        save panoramas to video. this function create 2 videos - 1 for the
        original panoramas and 1 for blended panoramas
        :return:
        """
        assert self.panoramas is not None
        self.bounding_boxes = self.bounding_boxes.astype('int')
        crop_left = int(self.bounding_boxes[0][1, 0])
        rows = self.nearestPower(self.panoramas.shape[1])
        cols = self.nearestPower(self.panoramas.shape[2])
        out_folder_blend = 'tmp_folder_for_panoramic_frames_blend/%s' % \
                     self.file_prefix
        out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
        try:
            shutil.rmtree(out_folder)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder)
        try:
            shutil.rmtree(out_folder_blend)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder_blend)
        try:
            shutil.rmtree(out_folder)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder)
        #take 2 consecutive panoramas
        for i in range(len(self.panoramas)):
            p = self.panoramas[i, :, :, :]
            p1 = self.panoramas[i, :rows, :cols, :]
            if i < len(self.panoramas) - 1:
                p2 = self.panoramas[i + 1, :rows, :cols, :]
            else:
                p2 = p1
        #find the boundaries between image strips and make barcod mask
            x_strips = self.x_strips[i, :]
            mask = np.ones(p1.shape[:-1]).astype('bool')
            diff = self.cunstruct_diff_strip(crop_left, crop_left + cols,
                                             x_strips)
            max_level = 3
            max_filter = ((np.min(diff)/(2**max_level)//2)*2 + 1).astype('int')
            if max_filter < 3:
                max_filter = 3
            mask[:, self.construct_barcod(diff)] = 0
            blend = np.zeros(p.shape)
        # blend images in all channels
            for j in range(3):
                blend[:rows, :cols, j] = panorama_utils.pyramid_blending(p1[:, :, j],
                                                    p2[:, :, j],
                                                                         mask,
                                                                         max_level,
                                                                         max_filter, max_filter)
            blend[rows:, :, :] = p[rows:, :, :]
            blend[:, cols:, :] = p[:, cols:, :]
            # save individual panorama images to 'tmp_folder_for_panoramic_frames' and
            # 'tmp_folder_for_panoramic_frames_blend'
            imwrite('%s/panorama%02d.png' % (out_folder_blend, i + 1),  blend)
            imwrite('%s/panorama%02d.png' % (out_folder, i + 1),  p)
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 12 -i %s/panorama%%02d.png %s.mp4' %
                  (out_folder_blend, self.file_prefix + '_blended'))
        os.system('ffmpeg -framerate 12 -i %s/panorama%%02d.png %s.mp4' %
                  (out_folder, self.file_prefix))

    def cunstruct_diff_strip(self, crop_left, crop_right, x_strips):
        """
        create array of differences of strips for specific image
        :param crop_left: column where the panorama starts
        :param crop_right: column where panorama ends
        :param x_strips: centers of strips 1-D array
        :return: 1-D array of difference between successive strips
        """
        x_strips = x_strips[(x_strips > crop_left) & (
                x_strips < crop_right)]
        x_offset = x_strips[0] - crop_left
        diff= np.hstack((x_offset, np.diff(x_strips)))
        return diff

    def construct_barcod(self, diff_arr):
        """
        create barcod mask
        :param diff_arr: 1-D array of difference between successive strips
        :return: 1-D array of indexes where the mask is false
        """
        mask = np.array([])
        i = 0
        start = 0
        while i < (len(diff_arr) - 2):
            mask = np.append(mask, np.arange(start, start + diff_arr[i] + 1))
            if i < (len(diff_arr) - 1):
                start = (mask[-1] + diff_arr[i + 1])
            i += 2
        return mask.astype('int')

    def nearestPower(self, b):
        """
        find the nearset power of 2 to a number
        :param b: upper bound for power of 2
        :return: highest power of 2 that bounded from up by b
        """
        high=100000  #any big number
        sum=1
        c = 1
        while True:
            sum = sum*2
            if 0 < (b-sum) < high:
                high = b-sum
                c = sum
            if(b - sum) < 0:
                break
        return c


    def show_panorama(self, panorama_index, figsize=(20, 20)):
        """
        display panorama
        :param panorama_index: number of panorama
        :param figsize: tuple of (rows, cols for figure
        :return:
        """
        assert self.panoramas is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()