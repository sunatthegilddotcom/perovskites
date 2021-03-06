# -*- coding: utf-8 -*-
"""
This module contains functions for training or using a trained Convolutional
Neural Network (CNN) to obtain y prediction (a single output like Ld80) from
a PL image.
"""
import os
import sys
import json
import re
import numpy as np

import pandas as pd
from tensorflow.keras import layers, Sequential
from tqdm.keras import TqdmCallback
from keras.models import model_from_json

###############################################################################
# LOAD SETTINGS FROM THE 'settings.json' file and other files
###############################################################################
# Append the current folder to sys path
parent_dir = os.path.dirname(__file__)
settings_path = os.path.join(parent_dir, 'settings.json')
utils_path = os.path.join(parent_dir, 'utils')
with open(settings_path, 'r') as file:
    MODEL_INFO = json.load(file)
sys.path.append(utils_path)

# Import the booleanize function
from miscellaneous import booleanize
import regression_visualization as reg_vis

# Convert the boolean strings (if any) in the settings
# dictionary to boolean-type values
MODEL_INFO = booleanize(MODEL_INFO)

# Get the paths from the settings file.
SHARED_DRIVE_PATH = MODEL_INFO['shared_drive_path']
models_folder = MODEL_INFO['cnn_model_info']['models_folder_path']
models_folder = os.path.join(*re.split('[\\\\/]', models_folder))

local_models_folder = os.path.join(parent_dir, models_folder)
drive_models_folder = os.path.join(SHARED_DRIVE_PATH, models_folder)

# Get the names of individual files to be saved
history_csv_name = MODEL_INFO['cnn_model_info']['history_csv_name']
model_json_name = MODEL_INFO['cnn_model_info']['model_json_name']
model_h5_name = MODEL_INFO['cnn_model_info']['model_h5_name']
fit_json_name = MODEL_INFO['cnn_model_info']['fit_json_name']

# The size of the image to be fed to the CNN.
# It's recommended to make changes to this in the settings to keep modules
# compatible to each other.
final_img_size = MODEL_INFO['target_image_size_pix']

# default font for plots
default_font = {'color': 'k',
                'fontsize': 14,
                'fontfamily': 'Arial'
                }
###############################################################################
# The CNN predictor class
###############################################################################


class CNNPredictor:
    """
    This class the following methods for training, testing, modifying the
    CNN model we have built for perovskites PL images to predict the log(Ld80)
    values. It initializes with the model we have built when called.
    1. summary() : Gives summary of model
    2. get_layers_dict() : Returns the model as dictionary
    3. load_layers_from_dict() : Loads the model from a dictionary
    4. save_model() : Saves the model
    5. load_model() : Loads a model
    6. fit() : Fits the training data
    7. evaluate_error() : Evaluates error at X
    8. predict() : Predicts y at given X
    9. full_analysis() : Does fitting, saving and also visualizations
    """

    def __init__(self, name,
                 loss_metric="mean_absolute_percentage_error",
                 optimizer="Ftrl",
                 y_label=MODEL_INFO['y_col']):
        """
        Initializes the CNN model designed to best perform on the perovskites
        PL image data to return log(Ld80) values as predictions.

        Parameters
        -----------
        name : str
            A unique name of the model

        Returns
        -------
        None.

        """
        self.name = name  # The unique name of the model
        self.loss_metric = loss_metric
        self.optimizer = optimizer
        self.epochs = 0
        self.batch_size = 0
        self.feed_shape = []
        self.model = Sequential()
        self.y_label = y_label

        # The first convolutional layer with a small kernel
        self.model.add(layers.Conv2D(filters=32, kernel_size=3,
                                     padding='valid',
                                     use_bias=True,
                                     input_shape=(final_img_size,
                                                  final_img_size, 1),
                                     kernel_initializer='normal',
                                     activation='relu'))

        # The averagePooling layer
        self.model.add(layers.AveragePooling2D(pool_size=2))

        # The next convolutional layers with larger kernels
        self.model.add(layers.Conv2D(filters=32, kernel_size=6,
                                     padding='valid',
                                     use_bias=True,
                                     kernel_initializer='normal',
                                     activation='relu'))
        self.model.add(layers.Conv2D(filters=16, kernel_size=6,
                                     padding='valid', use_bias=True,
                                     kernel_initializer='normal',
                                     activation='relu'))

        # Flatten the 2D layers to 1D
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(units=64,
                                    kernel_initializer='normal',
                                    activation='relu',
                                    use_bias=True,))
        self.model.add(layers.Dense(units=32,
                                    kernel_initializer='normal',
                                    activation='relu',
                                    use_bias=True,))
        self.model.add(layers.Dense(units=4,
                                    kernel_initializer='normal',
                                    activation='relu',
                                    use_bias=True,))

        # The final output
        self.model.add(layers.Dense(units=1, activation='linear',
                                    use_bias=True,))

        # Compile the model
        self.model.compile(loss=self.loss_metric,
                           optimizer=self.optimizer)

        # Create an empty dataframe for model's training history
        self.history_df = pd.DataFrame()

        self.callbacks = [
            TqdmCallback(verbose=0),
        ]

    def summary(self,):
        """
        Returns the summary of the CNN model

        Returns
        -------
        str
            A string summary of the model.

        """
        return self.model.summary()

    def get_layers_as_dict(self,):
        """
        Returns the model as a dictionaryof keras layers which can be
        edited if needed and stored again to the class variable.

        Returns
        -------
        dict
            The model's dictionary of keras layers
        """
        return json.loads(self.model.to_json())

    def load_layers_from_dict(self, model_dict):
        """
        Loads the model from a dictionary input.

        Parameters
        ----------
        model_dict : dict
            The dictionary representing the model with keras layers.

        Returns
        -------
        None.

        """
        json_string = json.dumps(model_dict)
        self.model = model_from_json(json_string,)

    def save_model(self, save_to_drive=False,
                   assign_new_name=None,
                   overwrite_existing_model=False):
        """
        Saves a folder with all the relevant files containing
        information about the current model.

        Parameters
        ----------
        save_to_drive : bool, optional
            Whether to save in the shared drive path or the local models
            folder. The default is False.
        assign_new_name : str, optional
            The new name of the model. The default is None.
        overwrite_existing_model : bool, optional
            Whether to overwite if another model exists with same name.

        Raises
        ------
        FileNotFoundError
            When you do not have access to the shared drive and the arguement,
            save_to_drive = True
        ValueError
            When a model already exists with the same name.

        Returns
        -------
        None.

        """
        # Change the name if needed
        if assign_new_name:
            self.name = assign_new_name

        # Raise error if you do not have access to the shared drive
        if save_to_drive and not os.path.exists(drive_models_folder):
            raise FileNotFoundError("YOU DO NOT HAVE ACCESS TO THE\n\
                                    SHARED DRIVE.")

        # The path to the models folder
        if save_to_drive:
            model_folder = os.path.join(drive_models_folder, self.name)
        else:
            model_folder = os.path.join(local_models_folder, self.name)

        # Raise error if another model exists with the same name
        if os.path.exists(model_folder) and not overwrite_existing_model:
            raise ValueError("A saved model already exists with this name.\n\
                             Try again with the arguement\n\
                             assign_new_name = <new_unique_model_name> or \n\
                             use overwrite_existing_model = True.")

        os.makedirs(model_folder, exist_ok=True)

        # The training history as csv file
        curr_model_hist = os.path.join(model_folder, history_csv_name)
        self.history_df.to_csv(curr_model_hist)

        # The model's parameters as json file
        curr_model_json = os.path.join(model_folder, model_json_name)
        model_dict = self.get_layers_as_dict()
        with open(curr_model_json, "w") as file:
            json.dump(model_dict, file, indent=4)

        # The model's weights as h5 file
        curr_model_h5 = os.path.join(model_folder, model_h5_name)
        self.model.save_weights(curr_model_h5)

        # Save the model fit's parameters
        fit_dict = dict(epochs=self.epochs, batch_size=self.batch_size,
                        feed_shape=list(self.feed_shape),
                        loss_metric=self.loss_metric,
                        optimizer=self.optimizer,
                        y_label=self.y_label)
        curr_fit_json = os.path.join(model_folder, fit_json_name)
        with open(curr_fit_json, "w") as file:
            json.dump(fit_dict, file, indent=4)

    def load_model(self, model_folder, from_drive=False):
        """
        Loads the model and its weights (if any) to the current class's
        model.

        Parameters
        ----------
        model_folder : str
            Either the name of the model or a complete path to the folder
            containing the model's csv file, h5 file and the json file.
        from_drive : bool, optional
            Whether to look for the model in the shared drive or locally.
            The default is False.

        Raises
        ------
        FileNotFoundError
            If from_drive=True and you do not have access to the shared drive,
            if the saved_models folder doesn't exist or if the model_folder
            doesn't exist.

        Returns
        -------
        None.

        """
        # The path to the saved models folder
        if from_drive:
            saved_models_folder = drive_models_folder
        else:
            saved_models_folder = local_models_folder

        if not os.path.exists(saved_models_folder):
            if from_drive:
                raise FileNotFoundError("YOU DO NOT HAVE ACCESS TO THE\n\
                                        SHARED DRIVE.")
            else:
                raise FileNotFoundError("THERE IS NO SAVED_MODELS FOLDER IN\n\
                                        THE LOCAL REPOSITORY.")

        # If the model_folder is just the name of the model you want to load
        if not ('/' in model_folder or '\\' in model_folder):
            model_folder = os.path.join(saved_models_folder, model_folder)

        if not os.path.exists(model_folder):
            raise FileNotFoundError("NO MODEL WITH THAT NAME EXISTS!")

        # Load the files from the model folder provided
        # The training history as csv file
        load_model_hist = os.path.join(model_folder, history_csv_name)
        self.history_df = pd.read_csv(load_model_hist)

        # The model's parameters as json file
        load_model_json = os.path.join(model_folder, model_json_name)
        with open(load_model_json, "r") as file:
            model_dict = json.load(file)
            self.load_layers_from_dict(model_dict)

        # The model's weights as h5 file
        load_model_h5 = os.path.join(model_folder, model_h5_name)
        self.model.load_weights(load_model_h5, by_name=True)

        # The previous fit's properties
        load_fit_json = os.path.join(model_folder, fit_json_name)
        with open(load_fit_json, "r") as file:
            fit_dict = json.load(file)
        self.epochs = fit_dict['epochs']
        self.batch_size = fit_dict['batch_size']
        self.feed_shape = list(fit_dict['feed_shape'])
        self.loss_metric = fit_dict['loss_metric']
        self.optimizer = fit_dict['optimizer']
        self.y_label = fit_dict['y_label']

        # Compile the model
        self.model.compile(loss=self.loss_metric,
                           optimizer=self.optimizer)

    def fit(self, X, y, epochs=1, batch_size=None,
            validation_split=0.2,):
        """
        Trains the model over X and y.

        Parameters
        ----------
        X : numpy.ndarray or tf.tensor
            The training set's X.
        y : numpy.ndarray or tf.tensor
            The training set's y.
        epochs : int, optional
            The number of iterations. The default is 1.
        batch_size : int, optional
            The batch size for training. The default is None.
        validation_split : frac, optional
            The fraction for train-validation split. The default is 0.2.

        Returns
        -------
        None.

        """
        if batch_size is None:
            batch_size = 0.1
        if batch_size < 1:
            batch_size = int(batch_size*len(y))

        self.epochs += epochs
        self.batch_size = batch_size
        self.feed_shape = list(X.shape)

        # TThis just allows to print code with variable colunmn length
        # based on the size of training set.
        _print_fit_table(len(y), self.epochs, self.batch_size, self.feed_shape)
        print("\n")

        history = self.model.fit(x=X, y=y, epochs=epochs,
                                 batch_size=batch_size,
                                 verbose=1,
                                 validation_split=validation_split,
                                 callbacks=self.callbacks)
        self.history_df = pd.DataFrame(data=history.history)

    def evaluate_error(self, X, y):
        """
        Evaluates loss or error at given X and y.

        Parameters
        ----------
        X : numpy.ndarray or tf.tensor
            The test set's X.
        y : numpy.ndarray or tf.tensor
            The test set's y.

        Returns
        -------
        numpy.ndarray
            The array of losses corresponding to test set rows.

        """
        return self.model.evaluate(x=X, y=y,
                                   verbose=0,
                                   callbacks=self.callbacks)

    def predict(self, X):
        """
        Predicts the y value based on X using the trained weights

        Parameters
        ----------
        X : numpy.ndarray
            The X array for predictions.

        Returns
        -------
        numpy.ndarray
            The array of predicted y values.

        """
        return self.model.predict(X, verbose=0,)

    def fit_plot(self, ax,):
        """
        Plots the error vs epoch plot for the current model.

        Parameters
        ----------
        ax : A Matplotlib Axes object()

        Returns
        -------
        None.

        """
        ax.plot(np.arange(self.epochs), self.history_df['loss'], 'o-',
                label='Training loss', linewidth=1.2)
        ax.plot(np.arange(self.epochs), self.history_df['val_loss'], 'o-',
                label='Validation loss', linewidth=1.2)
        ax.set_xlabel("epoch", **default_font)
        ax.set_ylabel(self.loss_metric, **default_font)
        ax.legend(fontsize=int(default_font['fontsize']*0.8))

    def parity_plot(self, ax, y_train, y_test, y_train_pred, y_test_pred,
                    data_df_train, data_df_test,):
        """
        The parity plot with the environental conditions being used as the
        merker styling.

        Parameters
        ----------
        ax : Matplotlib.Axes() object
            The Matplotlib.Axes() object for plotting
        y_train : numpy.ndarray
            The training y array
        y_test : numpy.ndarray
            The testing y array
        y_train_pred : numpy.ndarray
            The prediction of training y array
        y_test_pred : numpy.ndarray
            The prediction of testing y array
        data_df_train : pandas.DataFrame()
            The data_df dataframe containing the environmental features of
            training dataset.
        data_df_test : pandas.DataFrame()
            The data_df dataframe containing the environmental features of
            testing dataset.

        Returns
        -------
        None.

        """
        reg_vis.styled_parity_plot(ax, y_train, y_test,
                                   y_train_pred, y_test_pred,
                                   data_df_train, data_df_test,
                                   y_label=self.y_label,)

    def parity_legend(self, ax, save_path=None, dpi=100):
        """
        Makes a legend for the parity plot to descrive the marker styles.

        Parameters
        ----------
        ax : Matplotlib.Axes() object
            The Matplotlib.Axes() object for plotting
        save_path : str, optional
            path to save the new legend plot. The default is None.
        dpi : int, optional
            The quality of image in dpi. The default is 100.

        Returns
        -------
        None.

        """
        reg_vis.parity_legend(ax=ax, save_path=save_path, dpi=dpi)


##############################################################################
# The functions below this are just utility functions to be used within this
# module.
##############################################################################

def _print_fit_table(train_size, epochs, batch_size, feed_img_shape):
	"""A utility function just to print a neat table showing
	the arguements"""
    num_size_str_len = max(len(str(epochs)),
                           len(str(train_size)),
                           len(str(batch_size)),
                           len(str(feed_img_shape)))

    fir_row_space = num_size_str_len - len(str(train_size))
    fir_row = "|    Size of train set     | "
    fir_row += ''.join([' ' for i in range(fir_row_space)])
    fir_row += str(train_size) + ' |'

    sec_row_space = num_size_str_len - len(str(epochs))
    sec_row = "|      No. of epochs       | "
    sec_row += ''.join([' ' for i in range(sec_row_space)])
    sec_row += str(epochs) + ' |'

    thir_row_space = num_size_str_len - len(str(batch_size))
    thir_row = "|        Batch size        | "
    thir_row += ''.join([' ' for i in range(thir_row_space)])
    thir_row += str(batch_size) + ' |'

    fou_row_space = num_size_str_len - len(str(feed_img_shape))
    fou_row = "|     Feed image shape     | "
    fou_row += ''.join([' ' for i in range(fou_row_space)])
    fou_row += str(feed_img_shape) + ' |'

    main_row_dash = len(fir_row)-14
    main_row = ''.join(['-' for i in range(int(main_row_dash/2))])
    main_row += ' CNN Training '
    main_row += ''.join(['-' for i in range(int(main_row_dash/2))])

    bot_row = list(main_row.partition(' CNN Training '))
    bot_row[1] = ''.join(['-' for i in ' CNN Training '])
    bot_row = ''.join(bot_row)

    print(main_row)
    print(fir_row)
    print(sec_row)
    print(thir_row)
    print(fou_row)
    print(bot_row)
