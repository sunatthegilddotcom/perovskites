# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 15:24:08 2021
@author: Preetham

This file contains utility functions for loading and editing PL images,
especially TIF stacks, which are the popular formats used by most image
acquisition softwares like ImageJ, micromanager etc.

"""
# imports
import numpy as np
import os

from skimage.util import img_as_ubyte
from skimage.filters import rank
from skimage.morphology import disk
from scipy.signal import correlate2d, find_peaks
from skimage.transform import resize
###############################################################################
# THIS PIECE OF CODE MUST BE REMOVED WHILE MAKING A PACKAGE.
# THE NEED OF TIFFFILE CAN BE INCLUDED IN THE ENV. DEPENDENCIES
try:
    import tifffile as tf
except ImportError:
    print('Installing tifffile package ...')
    os.system("pip install tifffile")
    print('Done!')
    import tifffile as tf


###############################################################################
# IMAGE LOADING FUNCTIONS
###############################################################################
def read_image(img_path):
    """
    Returns PL stack array or DF image read from img_path,
    and returns in skimage friendly format of shape : (height, width, nframes)

    Parameters
    -----------------
    img_path : str
        The path to the image

    Returns
    -----------------
    img_stack : numpy.ndarray
        The image stack as numpy array

    """
    img_stack = tf.imread(img_path)
    if len(img_stack.shape) > 2:
        img_stack = np.swapaxes(img_stack, 0, 2)
        img_stack = np.swapaxes(img_stack, 0, 1)
    return img_stack


def mean_over_depth(img_arr):
    """
    Returns a 2D- array of mean in depth (time) of image stack. The shape of
    the returned 2D-array would be (height, width).

    Parameters
    ------------------
    img_arr : numpy.ndarray
        The 3D or 2D image stack array

    Returns
    ------------------
    mean_img_arr : numpy.ndarray
        The 2D mean array of the image stack

    """
    if len(img_arr.shape) < 3:
        return img_arr
    return np.mean(img_arr, axis=2)


###############################################################################
# IMAGE EDITING FUNCTIONS
###############################################################################
def normalize(im_arr, pix_range=(0, 1), axis=None, minmax=None):
    """Normalizes pixel values between pix_range.

    Parameters
    -----------
    im_arr : numpy array
        Array to be scaled or normalized
    pix_range : tuple (optional)
        range for the normalized array values
    axis : int or None
        axis to be used for normalization, if None, then all elements
        normalized uniformly
    minmax : tuple or other iterable or None
        min and max values to be used for normalizing
        (must be in order : min, then max)

    Returns
    -----------
    Normalized array
    """
    if minmax is None:
        minmax = [np.min(im_arr, axis=axis), np.max(im_arr, axis=axis)]

    norm_val = (im_arr-minmax[0])/(minmax[1] - minmax[0])
    norm_val *= max(pix_range) - min(pix_range)
    norm_val += min(pix_range)

    return norm_val


def channel_corners(img, blur_kernel=5, threshold=0.2, get_square=True):
    """This function is required to remove the electrical contacts (if any
    present), and obtain the perovskite channel in the PL image. It removes the
    contacts and returns the corners of the channel - 4x2 array with rows as
    points and columns as the img indices
    [left_top; right_top; left_bottom; right_bottom]

    Parameters
    ------------------------
    img : numpy.ndarray
        A 2-D array of the PL image

    blur_kernel : int (default=5)
        This the kernel size used for blurring before applying the gradient
        extraction algorithm to get the contact edges. A kernel too small might
        let the noise interfere with the gradient finding process, while a
        large kernel might smoothen even sharp contact edges. 5 is a good
        value.

    threshold : float (default=0.2)
        An empirical value that decides how good the extracted channel is.
        Higher the threshold, a channel is more likely to be extracted, but
        sometimes it would extract a channel from images with no channel too.
        Lower the threshold, the image must show strict channel boundary to
        extract channel and blurry channel boundaries are not well recognized.

    get_square : bool (default=True)
        If True, extracts a square-shaped cropped portion of the extract
        channel, with midpoint in the xy-center of the channel

    Returns
    -------------------------
    numpy.ndarray
        Returns the corners of the channel - 4x2 array with rows as points and
        columns as the img indices
        [left_top; right_top; left_bottom; right_bottom]

    """
    # The blurring kernel
    # img_arr = opening(closing(img, disk(blur_kernel)), disk(blur_kernel))
    img_arr = rank.median(img_as_ubyte(normalize(img)),
                          selem=disk(blur_kernel))

    # The derivative kernel
    kernel1 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    nderiv = 1
    for i in np.arange(nderiv):
        img_arr = abs(correlate2d(img_arr, kernel1, mode='valid'))

    # Sharpening channel edges and normalizing - calculative the 2nd derivative
    img_arr = normalize(np.exp(normalize(img_arr)))

    # Edges of the image
    left_edge = img_arr[:, 0]
    right_edge = img_arr[:, -1]
    pad_offset = int(kernel1.shape[0]/2)*nderiv

    # Finding the corners on the left edge
    peaks, props = find_peaks(left_edge, distance=img_arr.shape[0]/10,
                              prominence=(None, 1))
    peaks = peaks[np.argsort(props['prominences'])[-2:]]
    lt = [min(peaks), 0]
    lb = [max(peaks), 0]

    # ---- DECIDING WHETHER THE POINTS CORRESPOND TO A CHANNEL OR NOT ----
    # 1. if the channel width obtained is too small <OR>
    # 2. Using the ratio of median pixel intensities inside is to that outside
    #    channel
    #   (PL outside channel is dark and has low pixel median)

    ratio = np.median(normalize(img)[:lt[0], :]) / \
        np.median(normalize(img)[lt[0]:lb[0], :])

    if (lb[0] - lt[0]) < img_arr.shape[0]/3.5 or ratio > threshold:
        lt = [0, 0]
        lb = [img.shape[0], 0]
        rt = [0, img.shape[0]]
        rb = [img.shape[0], img.shape[0]]

        return np.array([lt, rt, lb, rb])

    # Finding the corners on the right edge
    peaks, props = find_peaks(right_edge, distance=img_arr.shape[0]/10,
                              prominence=(None, 1))
    peaks = peaks[np.argsort(props['prominences'])[-2:]]
    rt = [min(peaks), img_arr.shape[1]-1]
    rb = [max(peaks), img_arr.shape[1]-1]

    pad_offset = pad_offset*np.array([[1, 1], [1, 2], [2, 1], [2, 2]])
    points = np.array([lt, rt, lb, rb])+pad_offset

    # To get a square region in the channel
    if get_square:
        height = np.abs(points[2, 0] - points[0, 0])
        width = np.abs(points[1, 1] - points[0, 1])
        min_len = min(height, width)
        mid_point_col = np.mean([points[0, 1], points[1, 1]])
        mid_point_row = np.mean([points[0, 0], points[2, 0]])

        mid_point = [mid_point_row, mid_point_col]
        points = (mid_point + 0.5*min_len*np.array([[-1, -1],
                                                    [-1,  1],
                                                    [1, -1],
                                                    [1,  1]])).astype(int)

    return points


def crop_image(img_arr, new_corner_indices):
    """
    Crops the image when the corners are given.
    If the corners are not rectangular, the biggest rectangle contained
    with atleast 2 vertices taken from the user-specified corner indices,
    is cropped.
    The rows of the 4x2 new_corner_indices array must be in this order:
        left-top, right-top, left-bottom, right-bottom
    ________________            _________________
    |               |           |               |
    |*              |           |*              |
    |           *   |           |-----------*   |
    |               |  ----->   |:          :   |
    |               |           |:          :   |
    |             * |           |------------ * |
    |*              |           |*              |
    |_______________|           |_______________|

    Parameters
    --------------------
    img_arr : numpy.ndarray
        A 2D image array

    new_corner_indices : seq.
        Any sequence (list/tuple/array) with four integers indicating the
        indices for (left-top, right-top, left-bottom, right-bottom)

    Returns
    --------------------
    numpy.ndarray
        The cropped image as shown in figure above.


    """
    new_corner_indices = np.array(new_corner_indices)

    top_row = max(new_corner_indices[0, 0], new_corner_indices[1, 0])
    bottom_row = min(new_corner_indices[2, 0], new_corner_indices[3, 0])
    left_edge = max(new_corner_indices[0, 1], new_corner_indices[2, 1])
    right_edge = min(new_corner_indices[1, 1], new_corner_indices[3, 1])

    return img_arr[top_row:bottom_row+1, left_edge:right_edge+1]


def get_channel(img, get_square=True):
    """Returns the perovskite channel of the 2D image
    (effective for secondary PL frame) and also the corners of the rectagular
    channel.

    Parameters
    -----------
    img : numpy.ndarray
        2D matrix of image
    get_square : bool
        Whether to return a square shaped image

    Returns
    -----------
    tuple
        The normalized and cropped image array, and the corners of the channel
        in the sequence (left-top, right-top, left-bottom, right-bottom)

    """

    points = channel_corners(img, blur_kernel=5, get_square=get_square)
    return normalize(crop_image(img, points)), points


def resize_based_on_FOV(img, fov, target_img_um=50,
                        final_img_pix=32,
                        extract_channel=True):
    """
    Crops the square-shaped img based on the microscope's field of view (fov)
    in um into the target physical size of target_img_um (in um.) and resizes
    it to the target_img_pix size (in pixels.). So the image obtained will have
    pixel size of final_img_pix and a physical size of target_img_um in um.

    Parameters
    ----------
    img : numpy.ndarray
        The 2D square-shaped image array.
    fov : float
        The physical dimension of the field of view of the microscope's
        camera in um.
    target_img_um : int, optional
        The target image physical size in um. The default is 50.
    final_img_pix : TYPE, optional
        The final size into which the image will be resized.
        The default is 32.
    extract_channel : bool, optional
        Whether to extract the channel from the image

    Raises
    --------
    ValueError
        When the target_img_um is greater than FOV or if its equivalent value
        in pixels is larger than the image size.

    Returns
    -------
    img_arr : numpy.ndarray
        The 2D image aray.
    new_corner_indices : numpy.ndarray
        List of the (row, col) indices used in cropping in the order-
        left-top, right-top, left-bottom, right-bottom

    """
    img_arr = img.copy()
    pix_per_um = img_arr.shape[0]/fov
    if target_img_um > fov:
        raise ValueError("The target image's physical dimension in um must be\
                         smaller than field of view in um, i.e.,"+str(fov))

    # Find the equivalent target size in pix before finaly resizing
    target_im_pix_raw = pix_per_um * target_img_um
    if img_arr.shape[0] < target_im_pix_raw:
        raise ValueError("The target_im_um is larger than the physical size\
                         of image in um.")
    # extract the channel
    if extract_channel:
        img_arr, _ = get_channel(img_arr, get_square=True)
        if img_arr.shape[0] < target_im_pix_raw:
            raise ValueError("The target_im_um is larger than the physical\
                             size of image in um, after extracting channel.")

    # go to the center of the channel, and crop uniformly
    # until the srt default physical size is obtained.
    c_row, c_col = (np.array(img_arr.shape)/2).astype(int)
    dh, dw = (0.5*target_im_pix_raw*np.ones(2)).astype(int)
    new_corner_indices = np.array([[c_row-dh, c_col-dw],
                                   [c_row-dh, c_col+dw],
                                   [c_row+dh, c_col-dw],
                                   [c_row+dh, c_col+dw]])

    img_arr = crop_image(img_arr, new_corner_indices)

    # Finally resize image to target size
    img_arr = resize(img_arr, (final_img_pix, final_img_pix),
                     anti_aliasing=True)

    return img_arr, new_corner_indices


def img_as_feed(img_path, fov, img_arr=None, time_frame=0,
                target_img_um=50, final_img_pix=32,
                extract_channel=True,
                return_corner_inds=False):
    """
    Reads the image stack from the path or from img_arr, picks just the frame
    corresponding to the time point specified, and returns image array ready
    to be fed to a CNN.

    Parameters
    ----------
    img_path : str
        The path to the image stack.
    fov : float
        The field of view of the microscope used.
    img_arr : numpy.ndarray, optional
        The image array to be fed. It is optional and if passed, the image_path
        is ignored irrespective of its value. If the img_path is unknown,
        set it to None. The default is None.
    time_frame : int, optional
        The time cooresponding to the time frame of the image stack,
        to be used (>=0 and <50). The default is 0.
    target_img_um : int, optional
        The target physical size of the image. The default is 50.
    final_img_pix : int, optional
        The final image size in pixels. The default is 32.
    extract_channel : bool, optional
        Whether to extract the channel in the image. The default is True.
    return_corner_inds : bool, optional
        Whether to return the (row, col) indices used in cropping in the order-
        left-top, right-top, left-bottom, right-bottom
        The default is False.

    Returns
    -------
    feed_img = numpy.ndarray
        A 2D image array with the size of final_img_pix with pixel values in
        range [0, 255].
    new_corner_indices : numpy.ndarray
        The array of (row, col) indices used in cropping in the order-
        left-top; right-top; left-bottom; right-bottom.
    """

    if img_path is not None:
        img = read_image(img_path)[:, :, time_frame]
    elif img_arr is not None:
        img = img_arr.copy()
    else:
        raise Exception('One among img_path and img_arr must not be None.')

    img, new_corner_indices =\
        resize_based_on_FOV(img, fov,
                            target_img_um=target_img_um,
                            final_img_pix=final_img_pix,
                            extract_channel=extract_channel)

    # ALso the image channel dimension must be explicitly mentioned for
    # feeding into the tf.keras NN models
    img = img.reshape((*img.shape, 1))
    if return_corner_inds:
        return img_as_ubyte(img), new_corner_indices
    else:
        return img_as_ubyte(img)
