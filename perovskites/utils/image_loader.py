# -*- coding: utf-8 -*-
from miscellaneous import booleanize
import image_processing as impr
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
    MODEL_INFO = json.loads(file.read())

sys.path.append(curr_dir)

# Convert the boolean strings (if any) in the settings dictionary to boolean
# type values
MODEL_INFO = booleanize(MODEL_INFO)
SHARED_DRIVE_PATH = MODEL_INFO['data_info']['shared_drive_path']
DATA_PICKLE_PATH = MODEL_INFO['data_info']['pickle_filename']
META_DF_PATH = MODEL_INFO['data_info']['features_meta_filtered.csv']
DATA_DF_PATH = MODEL_INFO['data_info']['features_intact_filtered.csv']

DATA_PICKLE_PATH = SHARED_DRIVE_PATH + '/' + DATA_PICKLE_PATH
META_DF_PATH = SHARED_DRIVE_PATH + '/' + META_DF_PATH
DATA_DF_PATH = SHARED_DRIVE_PATH + '/' + DATA_DF_PATH
if not (os.path.exists(DATA_PICKLE_PATH) or os.path.exists(SHARED_DRIVE_PATH)):
    print("YOU DO NOT HAVE ACCESS TO THE DATASETS. PLEASE CONTACT THE REPO\
          CONTRIBUTORS AT THE EMAILS PROVIDED IN THE README FOR ACCESS.")


class PLDataLoader:
    """
    This class loads the data from the "Perovskites_DIRECT" shared data folder.
    Note that only the team members have access to this folder.
    """

    def __init__(self):
        self.X = None
        self.y = None
        self.meta_df = None
        self.data_df = None
        try:
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
            image_list.append(_load_image_from_path(idx, fov=fov,
                                                    target_img_um=target_img_um,
                                                    final_img_pix=final_img_pix,
                                                    time_frame=time_frame,
                                                    extract_channel=extract_channel))
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
    with open(pickle_path, 'wb') as file:
        pickle.dump(data_tuple, file)


def _get_data_from_pickle(pickle_path):
    with open(pickle_path, 'rb') as file:
        data_tuple = pickle.load(file)
    return data_tuple


def _create_PL_data_pickle(meta_df, data_df, pickle_path=DATA_PICKLE_PATH):
    X, y, inds = _get_image_data_from_paths(meta_df, data_df,
                                            skip_bad_images=True)
    data_tuple = (X, y, meta_df.loc[inds], data_df.loc[inds])
    _create_data_pickle(data_tuple, pickle_path)
