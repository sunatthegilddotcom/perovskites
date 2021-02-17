# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 15:24:08 2021
@author: Preetham

This file contains utility functions for loading and editing PL images,
especially TIF stacks, which are the popular formats used by most image
acquisition softwares like ImageJ, micromanager etc.

"""
# imports
from skimage.util import img_as_ubyte
from skimage.filters import rank
from skimage.morphology import disk
from scipy.signal import correlate2d, find_peaks
import numpy as np
import os

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
    if len(img_stack.shape)>2:
        img_stack = np.swapaxes(img_stack,0,2)
        img_stack = np.swapaxes(img_stack,0,1)
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
    if len(img_arr.shape)<3:
        return img_arr
    return np.mean(img_arr, axis=2)


###############################################################################
# IMAGE EDITING FUNCTIONS
###############################################################################
def normalize(im_arr, pix_range=(0,1), axis=None, minmax=None):
    """Normalizes pixel values between pix_range.
    
    Parameters
    -----------
    im_arr : numpy array
        Array to be scaled or normalized
    pix_range : tuple (optional)
        range for the normalized array values
    axis : int or None
        axis to be used for normalization, if None, then all elements normalized uniformly
    minmax : tuple or other iterable or None
        min and max values to be used for normalizing (must be in order : min, then max)
        
    Returns
    -----------
    Normalized array
    """
    if minmax is None:
        return min(pix_range) + ((im_arr-np.min(im_arr, axis=axis))/(np.max(im_arr, axis=axis)-np.min(im_arr, axis=axis)))*(max(pix_range) - min(pix_range))
    else:
        return min(pix_range) + ((im_arr-minmax[0])/(minmax[1] - minmax[0]))*(max(pix_range) - min(pix_range))


def channel_corners(img, blur_kernel=5, threshold=0.2, get_square=True):
    """    
    This function is required to remove the electrical contacts (if any present),
    and obtain the perovskite channel in the PL image. It removes the contacts
    and returns the corners of the channel - 4x2 array with rows as points and
    columns as the img indices
    [left_top; right_top; left_bottom; right_bottom]
    
    Parameters
    ------------------------
    img : numpy.ndarray
        A 2-D array of the PL image
    
    blur_kernel : int (default=5)
        This the kernel size used for blurring before applying the gradient
        extraction algorithm to get the contact edges. A kernel too small might
        let the noise interfere with the gradient finding process, while a large
        kernel might smoothen even sharp contact edges. 5 is a good value.
        
    threshold : float (default=0.2)
        An empirical value that decides how good the extracted channel is.
        Higher the threshold, a channel is more likely to be extracted, but sometimes
        it would extract a channel from images with no channel too. Lower the threshold,
        the image must show strict channel boundary to extract channel and blurry
        channel boundaries are not well recognized.
        
    get_square : bool (default=True)
        If True, extracts a square-shaped cropped portion of the extract channel,
        with midpoint in the xy-center of the channel 
        
    Returns
    -------------------------
    numpy.ndarray
        Returns the corners of the channel - 4x2 array with rows as points and
        columns as the img indices
        [left_top; right_top; left_bottom; right_bottom]
    
    """
    #The blurring kernel
    #img_arr = opening(closing(img, disk(blur_kernel)), disk(blur_kernel))
    img_arr = rank.median(img_as_ubyte(normalize(img)), selem=disk(blur_kernel))
    
    #The derivative kernel
    kernel1 = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
    
    nderiv = 1
    for i in np.arange(nderiv):
        img_arr = abs(correlate2d(img_arr, kernel1, mode='valid'))
    
    #Sharpening channel edges and normalizing - calculative the 2nd derivative
    img_arr = normalize(np.exp(normalize(img_arr)))
    
    #Edges of the image
    left_edge = img_arr[:,0]
    right_edge = img_arr[:,-1]
    pad_offset = int(kernel1.shape[0]/2)*nderiv
    
    #Finding the corners on the left edge
    peaks, props = find_peaks(left_edge, distance=img_arr.shape[0]/10, prominence=(None,1))
    peaks = peaks[np.argsort(props['prominences'])[-2:]]
    lt = [min(peaks), 0]
    lb = [max(peaks), 0]
    
    # ---- DECIDING WHETHER THE POINTS CORRESPOND TO A CHANNEL OR NOT ----
    #1. if the channel width obtained is too small <OR>
    #2. Using the ratio of median pixel intensities inside is to that outside channel
    #   (PL outside channel is dark and has low pixel median)
    
    ratio = np.median(normalize(img)[:lt[0], :])/np.median(normalize(img)[lt[0]:lb[0], :])
    
    if (lb[0] - lt[0]) < img_arr.shape[0]/3.5 or ratio>threshold:
        lt = [0, 0]
        lb = [img.shape[0], 0]
        rt = [0, img.shape[0]]
        rb = [img.shape[0], img.shape[0]]
        
        return np.array([lt, rt, lb, rb])
    
    #Finding the corners on the right edge
    peaks, props = find_peaks(right_edge, distance=img_arr.shape[0]/10, prominence=(None,1))
    peaks = peaks[np.argsort(props['prominences'])[-2:]]
    rt = [min(peaks), img_arr.shape[1]-1]
    rb = [max(peaks), img_arr.shape[1]-1]
    
    pad_offset = pad_offset*np.array([[1, 1], [1, 2], [2, 1], [2,2]])
    points = np.array([lt, rt, lb, rb])+pad_offset
    
    # To get a square region in the channel
    if get_square:
        height = np.abs(points[2,0] - points[0,0])
        width = np.abs(points[1,1] - points[0,1])
        min_len = min(height, width)
        mid_point_col = np.mean([points[0, 1], points[1, 1]])
        mid_point_row = np.mean([points[0, 0], points[2, 0]])
        
        mid_point = [mid_point_row, mid_point_col]
        points = (mid_point + 0.5*min_len*np.array([[-1, -1],
                                                   [-1,  1],
                                                   [ 1, -1],
                                                   [ 1,  1]])).astype(int)
    
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
        Any sequence (list/tuple/array) with four integers indicating the indices
        for (left-top, right-top, left-bottom, right-bottom)
        
    Returns
    --------------------
    numpy.ndarray
        The cropped image as shown in figure above.
        
    
    """
    top_row = max(new_corner_indices[0,0], new_corner_indices[1,0])
    bottom_row = min(new_corner_indices[2,0], new_corner_indices[3,0])
    left_edge = max(new_corner_indices[0,1], new_corner_indices[2,1])
    right_edge = min(new_corner_indices[1,1], new_corner_indices[3,1])
    
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







