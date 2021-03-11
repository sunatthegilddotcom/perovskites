# -*- coding: utf-8 -*-
"""
This module has functions to be used for loading the X and y data from the
data pickle data. Note that this pickle file and other relevant data csv files
are available only to the team members.
"""
import sys
import os
import json
import pickle
import numpy as np
import pandas as pd
import traceback

from tqdm import tqdm
from sklearn.model_selection import train_test_split

# %%
# Append the current folder to sys path
curr_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(curr_dir)
settings_path = os.path.join(parent_dir, 'settings.json')

with open(settings_path, 'r') as file:
    MODEL_INFO = json.load(file)

sys.path.append(curr_dir)
import image_processing as impr
from miscellaneous import booleanize

# Convert the boolean strings (if any) in the settings dictionary to boolean
# type values
MODEL_INFO = booleanize(MODEL_INFO)

# Load the filenames of all the data files
DATA_FOLDER_PATH = MODEL_INFO['data_info']['shared_drive_path']
DATA_PICKLE_PATH = MODEL_INFO['data_info']['pickle_filename']
META_DF_PATH = MODEL_INFO['data_info']['meta_df_filename']
DATA_DF_PATH = MODEL_INFO['data_info']['data_df_filename']

# Check in the local folder if the module is not run in COlab
if not os.path.exists(DATA_FOLDER_PATH):
    DATA_FOLDER_PATH = MODEL_INFO['data_info']['local_data_folder']
DATA_PICKLE_PATH = os.path.join(DATA_FOLDER_PATH, DATA_PICKLE_PATH)
META_DF_PATH = os.path.join(DATA_FOLDER_PATH, META_DF_PATH)
DATA_DF_PATH = os.path.join(DATA_FOLDER_PATH, DATA_DF_PATH)

# Inform the user if they do not have access to the data.
is_pickle_present = os.path.exists(DATA_PICKLE_PATH)
is_df_present = os.path.exists(META_DF_PATH) and os.path.exists(DATA_DF_PATH)
if not (is_pickle_present or is_df_present):
    print("YOU DO NOT HAVE ACCESS TO THE DATASETS. PLEASE CONTACT THE REPO\
          \nCONTRIBUTORS AT THE EMAILS PROVIDED IN THE README FOR ACCESS.")

##########################################################
########## THE MAIN DATA LOADER CLASS ###################
##########################################################

class PLDataLoader:
    """
    This class loads the data from the "Perovskites_DIRECT" shared data folder.
    Note that only the team members have access to this folder.
    """

    def __init__(self):
        """
        Initializes the class and loads the data in class variables

        Returns
        -------
        None.

        """
        self.X = None
        self.y = None
        self.meta_df = None
        self.data_df = None
        try:
            # Create the data pickle file, if one doesn't exist
            if not os.path.exists(DATA_PICKLE_PATH):
                self.meta_df = pd.read_csv(META_DF_PATH, index_col=0,
                                           header=[0, 1])
                self.data_df = pd.read_csv(DATA_DF_PATH, index_col=0)
                _create_PL_data_pickle(self.meta_df, self.data_df,
                                       pickle_path=DATA_PICKLE_PATH)
            data_tuple = _get_data_from_pickle(DATA_PICKLE_PATH)
            self.X, self.y, self.meta_df, self.data_df = data_tuple
        except FileNotFoundError():
            print("YOU DO NOT HAVE ACCESS TO THE DATASET FILES.")

    def sample(self, frac=0.1, random_state=42, return_dfs=False):
        """
        Samples subset of the whole dataset and returns X and y

        Parameters
        ----------
        frac : float, optional
            The sample size as fraction of total dataset. The default is 0.1.
        random_state : int, optional
            The random state. The default is 42.
        return_dfs : bool, optional
            Whether to return the meta_df and data_df containing the meta data
            and the Ld time series features separately. The default is False.

        Raises
        ------
        ValueError
            If the frac is not between 0 and 1

        Returns
        -------
        X : numpy.ndarray
            Array of images of shape (N, image_height, image_width, channels)
        y : numpy.ndarray
            Array of corresponding y values
        meta_df : pd.DataFrame
            A pandas DataFrame containing the paths to images in the shared
            drive folder and other metadata. (returned only when return_dfs is
            True).
        data_df : pd.DataFrame
            A pandas DataFrame containing other features (returned only when
            return_dfs is True).

        """
        if frac > 1 or frac < 0:
            raise ValueError("frac's value must be between 0 and 1.")

        size = int(frac*len(self.y))
        np.random.seed(random_state)
        sample_inds = np.random.choice(np.arange(len(self.y)),
                                       replace=False, size=size)

        X = self.X[sample_inds, :, :, :]
        y = self.y[sample_inds]
        meta_df = self.meta_df.iloc[sample_inds]
        data_df = self.data_df.iloc[sample_inds]

        if return_dfs:
            return X, y, meta_df, data_df
        else:
            return X, y

    def train_test_split(self, test_size=0.2, random_state=42, shuffle=False,
                         return_dfs=False):
        """
        Generates the train and test datasets.

        Parameters
        ----------
        test_size : float, optional
            The test size as a fraction of total dataset. The default is 0.2.
        random_state : int, optional
            The random state. The default is 42.
        shuffle : bool, optional
            Whether to shuffle the dataset or not. The default is False.
        return_dfs : bool, optional
            Whether to return the meta_Df and the data_df. The default is
            False.

        Returns
        -------
        X_train, X_test : numpy.ndarray, numpy.ndarray
            Arrays of images of shape (N, image_height, image_width, channels)
        y_train, y_test : numpy.ndarray, numpy.ndarray
            Arrays of corresponding y values
        meta_df_train, meta_df_test : pd.DataFrame, pd.DataFrame
            Pandas DataFrames containing the paths to images in the shared
            drive folder and other metadata. (returned only when return_dfs is
            True).
        data_df_train, data_df_test : pd.DataFrame, pd.DataFrame
            Pandas DataFrames containing other features (returned only when
            return_dfs is True)

        """
        train_inds, test_inds = train_test_split(np.arange(len(self.y)),
                                                 test_size=test_size,
                                                 random_state=random_state,
                                                 shuffle=shuffle)
        X_train, y_train = self.X[train_inds, :, :, :], self.y[train_inds]
        X_test, y_test = self.X[test_inds, :, :, :], self.y[test_inds]

        if return_dfs:
            meta_df_train = self.meta_df.iloc[train_inds]
            data_df_train = self.data_df.iloc[train_inds]
            meta_df_test = self.meta_df.iloc[test_inds]
            data_df_test = self.data_df.iloc[test_inds]
            return (X_train, X_test, y_train, y_test, meta_df_train,
                    meta_df_test, data_df_train, data_df_test)
        else:
            return X_train, X_test, y_train, y_test


####################################################################
# UTILITY FUNCTIONS FOR PLDataLoader()
####################################################################

def _load_image_from_path(path, fov,
                          target_img_um=MODEL_INFO['target_image_size_um'],
                          final_img_pix=MODEL_INFO['target_image_size_pix'],
                          time_frame=0,
                          extract_channel=MODEL_INFO['extract_channel']):
    """
    Loads the image into an image array from the specified path

    Parameters
    ----------
    path : str
        The path to the image.
    fov : flost
        The field of view of the microscope.
    target_img_um : int, optional
        The target physical size of image. The default is
        MODEL_INFO['target_image_size_um'].
    final_img_pix : int, optional
        The target size of image in pixels. The default is
        MODEL_INFO['target_image_size_pix'].
    time_frame : int, optional
        Must be less than 50. The time frame of the image stack to be loaded.
        The default is 0.
    extract_channel : bool, optional
        Whether to extract the channel.
        The default is MODEL_INFO['extract_channel'].

    Returns
    -------
    numpy.ndarray
        The image array.

    """
    return impr.img_as_feed(path, fov=fov, time_frame=time_frame,
                            target_img_um=target_img_um,
                            final_img_pix=final_img_pix,
                            extract_channel=extract_channel)


def _get_image_data_from_paths(meta_df, data_df,
                               target_img_um=MODEL_INFO['target_image_size_um'],
                               final_img_pix=MODEL_INFO['target_image_size_pix'],
                               y_col=MODEL_INFO['y_col'],
                               fov_col=MODEL_INFO['FOV_col'],
                               time_frame=0,
                               extract_channel=MODEL_INFO['extract_channel'],
                               skip_bad_images=True):
    """
    Returns the image array, the y values and their corresponding indices as
    from meta_df.

    Parameters
    ----------
    meta_df : pd.DataFrame
        The metadata dataframe.
    data_df : pd.DataFrame
        DESCRIPTION.
    target_img_um : int, optional
        The target physical size of image. The default is
        MODEL_INFO['target_image_size_um'].
    final_img_pix : int, optional
        The target size of image in pixels. The default is
        MODEL_INFO['target_image_size_pix'].
    y_col : str, optional
        The name of the y_col from meta_index.columns(). The default is
        MODEL_INFO['y_col'].
    fov_col : str, optional
        The name of the fov_col from meta_index.columns(). The default is
        MODEL_INFO['FOV_col'].
    time_frame : int, optional
        The time frame of the primary video to be used. The default is 0.
    extract_channel : bool, optional
        Whether the channel needs to be extracted or not. The default is
        MODEL_INFO['extract_channel'].
    skip_bad_images : bool, optional
        Whether to skip the bad runs. Otherwise, it raises error. The default
        is True.

    Raises
    ------
    Exception
        When image is not read properly.

    Returns
    -------
    X  : numpy.ndarray
        An array of image data set.
    y : numpy.ndarray
        An array of y values
   used_indices : list
        An aray of indices from meta_df used in bulding X & y.

    """
    image_list = []
    y_list = []
    used_indices = []
    for idx in tqdm(meta_df.index):
        # Load the image
        try:
            fov = meta_df.loc[idx, (fov_col, fov_col)].values[0]
        except:
            fov = meta_df.loc[idx, (fov_col, fov_col)]

        try:
            img = _load_image_from_path(idx, fov=fov,
                                        target_img_um=target_img_um,
                                        final_img_pix=final_img_pix,
                                        time_frame=time_frame,
                                        extract_channel=extract_channel)
            image_list.append(img)
        except:
            # If an error rises, stop or continue based on skip_bad_images
            # arguement.
            if skip_bad_images:
                continue
            else:
                traceback.print_exc()
                raise Exception("Image loading failed!")

        # Now get the y_value based on y_col string
        try:
            y_val = data_df.loc[idx, y_col].values[0]
        except:
            y_val = data_df.loc[idx, y_col]

        y_list.append(y_val)
        used_indices.append(idx)

    return np.array(image_list), np.array(y_list), used_indices


def _create_data_pickle(data_tuple, pickle_path):
    """
    Creates pickle file

    Parameters
    ----------
    data_tuple : any
        The tuple or any other datatype to be stored.
    pickle_path : str
        The path to the pickle file.

    Returns
    -------
    None.

    """
    with open(pickle_path, 'wb') as file:
        pickle.dump(data_tuple, file)


def _get_data_from_pickle(pickle_path):
    """
    Reads data from a pickle file.

    Parameters
    ----------
    pickle_path : str
        The path to the pickle file.

    Returns
    -------
    data_tuple : any
        The stored variable.

    """
    with open(pickle_path, 'rb') as file:
        data_tuple = pickle.load(file)
    return data_tuple


def _create_PL_data_pickle(meta_df, data_df, pickle_path=DATA_PICKLE_PATH):
    """
    Creates a PL data pickle file.

    Parameters
    ----------
    meta_df : pd.DataFrame
        A pandas DataFrame containing the paths to images in the shared
        drive folder and other metadata.
    data_df : pd.DataFrame
        A pandas DataFrame containing other features
    pickle_path : str, optional
        The path to the pickle file. The default is DATA_PICKLE_PATH.

    Returns
    -------
    None.

    """
    X, y, inds = _get_image_data_from_paths(meta_df, data_df,
                                            skip_bad_images=True)
    data_tuple = (X, y, meta_df.loc[inds], data_df.loc[inds])
    _create_data_pickle(data_tuple, pickle_path)
