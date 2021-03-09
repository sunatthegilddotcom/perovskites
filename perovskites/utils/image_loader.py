# -*- coding: utf-8 -*-
import sys
import os
import json
import numpy as np
import pandas as pd

from tqdm import tqdm
from tensorflow.python.keras.utils.data_utils import Sequence

# Append the current folder to sys path
curr_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(curr_dir)
settings_path = os.path.join(parent_dir, 'settings.json')
with open(settings_path, 'r') as file:
    MODEL_INFO = json.loads(file.read())
    
sys.path.append(curr_dir)
import image_processing as impr
from miscellaneous import booleanize

# Convert the boolean strings (if any) in the settings dictionary to boolean
# type values
MODEL_INFO = booleanize(MODEL_INFO)

class PLDataSequence(Sequence):
    def __init__(self, batch_size,
                 data_df, meta_df,
                 y_col=MODEL_INFO['y_col'], fov_col=MODEL_INFO['FOV_col'],
                 target_img_um=MODEL_INFO['target_image_size_um'],
                 final_img_pix=MODEL_INFO['target_image_size_pix'],
                 ):
        self.x, self.data_df = meta_df.index.values, data_df
        self.meta_df = meta_df
        self.batch_size = batch_size
        self.y_col = y_col
        self.fov_col = fov_col
        self.target_img_um = target_img_um
        self.final_img_pix = final_img_pix

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.data_df.loc[batch_x, self.y_col].values
        
        return np.stack([
            impr.img_as_feed(x_path, fov=self.meta_df.loc[x_path, (self.fov_col, self.fov_col)], time=0,
                             target_img_um=self.target_img_um,
                             final_img_pix=self.final_img_pix,) for x_path in batch_x
        ], axis=0), batch_y


def load_image_from_path(path, fov,
                    target_img_um=MODEL_INFO['target_image_size_um'],
                    final_img_pix=MODEL_INFO['target_image_size_pix'],
                    time_frame=0,
                    extract_channel=MODEL_INFO['extract_channel']):
    
    return impr.img_as_feed(path, fov=fov, time_frame=time_frame,
                             target_img_um=target_img_um,
                             final_img_pix=final_img_pix,
                             extract_channel=extract_channel)


def get_image_data(meta_df, data_df,
                   target_img_um=MODEL_INFO['target_image_size_um'],
                   final_img_pix=MODEL_INFO['target_image_size_pix'],
                   y_col=MODEL_INFO['y_col'],
                   fov_col=MODEL_INFO['FOV_col'],
                   time_frame=0,
                   extract_channel=MODEL_INFO['extract_channel']):
    
    image_list = []
    y_list = []
    for idx in tqdm(meta_df.index):
        
        # Load the image
        try:
            fov = meta_df.loc[idx, (fov_col, fov_col)].values[0]
        except:
            fov = meta_df.loc[idx, (fov_col, fov_col)]
            
        image_list.append(load_image_from_path(idx, fov=fov,
                            target_img_um=target_img_um,
                            final_img_pix=final_img_pix,
                            time_frame=time_frame,
                            extract_channel=extract_channel))
        
        # Now get the y_value based on y_col string
        try:
            y_val = data_df.loc[idx, y_col].values[0]
        except:
            y_val = data_df.loc[idx, y_col]
            
        y_list.append(y_val)
    
    return np.array(image_list), np.array(y_list)



