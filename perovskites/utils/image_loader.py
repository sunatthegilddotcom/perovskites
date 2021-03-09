# -*- coding: utf-8 -*-
import sys
import os
import json
import numpy as np
import pandas as pd
from tensorflow.python.keras.utils.data_utils import Sequence


# Append the current folder to sys path
curr_dir = os.path.dirname(os.path.abspath(__name__))
parent_dir = os.path.dirname(curr_dir)
settings_path = os.path.join(parent_dir, 'settings.json')
with open(settings_path, 'r') as file:
    MODEL_INFO = json.load(file)

sys.path.append(curr_dir)
import image_processing as impr

class PLDataSequence(Sequence):
    def __init__(self, x_paths_list, batch_size,
                 data_df, meta_df,
                 y_col=MODEL_INFO['y_col'], fov_col=MODEL_INFO['FOV_col'],
                 target_img_um=MODEL_INFO['target_image_size_um'],
                 final_img_px=MODEL_INFO['target_image_size_px'],
                 ):
        self.x, self.data_df = x_paths_list, data_df
        self.meta_df = meta_df
        self.batch_size = batch_size
        self.y_col = y_col
        self.fov_col = fov_col
        self.target_img_um = target_img_um
        self.final_img_px = final_img_px

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.data_df.loc[batch_x, self.y_col].values
        
        return np.stack([
            impr.img_as_feed(x_path, fov=self.meta_df.loc[x_path, self.fov_col], time=0,
                             target_img_um=self.target_img_um,
                             final_img_px=self.final_img_px,) for x_path in batch_x
        ], axis=0), batch_y


