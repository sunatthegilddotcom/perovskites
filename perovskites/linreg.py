# -*- coding: utf-8 -*-
"""
This module contains functions for performing LASSO linear regression on the
current perovskites dataset.
"""
import os
import sys
import json
import pickle
import re

import numpy as np
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.linear_model import Ridge, TheilSenRegressor


###############################################################################
# LOAD SETTINGS FROM THE 'settings.json' file.
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
import image_loader as loader

# Convert the boolean strings (if any) in the settings
# dictionary to boolean-type values
MODEL_INFO = booleanize(MODEL_INFO)

# Get the paths from the settings file.
SHARED_DRIVE_PATH = MODEL_INFO['shared_drive_path']
models_folder = MODEL_INFO['linreg_model_info']['models_folder_path']
models_folder = os.path.join(*re.split('[\\\\/]', models_folder))

local_models_folder = os.path.join(parent_dir, models_folder)
drive_models_folder = os.path.join(SHARED_DRIVE_PATH, models_folder)

# Get the names of individual files to be saved
fit_pickle_name = MODEL_INFO['linreg_model_info']['fit_pickle_name']

##############################################################################
# The main Linear Regression wrapper function
##############################################################################


def linear_regressor(X_train, y_train,
                     X_test, y_test,
                     feat_labels, y_label,
                     model_type='Lasso',
                     alpha_tuning_params={},
                     model_folder=None,
                     save_to_drive=False,
                     overwrite_existing_model=False
                     ):
    """
    Fits a Linear Regression model to the specified training data. The test
    data is passed just so that it can be saved together with the training
    data into a pickle file at the end if model_folder is not None. The type
    of Linear regressor is decided using the model_type. For regularized models
    like 'Lasso' and 'Ridge', an additional alpha tuning cross-validation step
    is performed to get the optimum regularization alpha value.

    The following are the model_type values currently allowed.
    1. 'SimpleLinear'
    2. 'Lasso'
    3. 'Ridge'
    4. 'TheilSen'

    Parameters
    ----------
    X_train : numpy.ndarray
        The training data's X.
    y_train : numpy.ndarray
        The training data's y.
    X_test : numpy.ndarray
        The testset's X.
    y_test : numpy.ndarray
        The testset's y.
    feat_labels : seq
        The labels to feature columns.
    y_label : str
        The label to the y variable.
    model_type : str, optional
        The model type. The default is 'Lasso'.
    alpha_tuning_params : dict, optional
        The extra arguements for alpha_tuning function. The default is {}.
        The allowed keys are -
        alpha_list : numpy.ndarray, optional
            The initial list of alpha values.
            The default is np.logspace(-3, 1, 5).
        k_fold : int or str, optional
            The integer number of k-folds for cross validation.
            The default is 5.
            Use k_fold=1 for finding the best regularization hyperparameter on
            the enitre dataset without cross validation.
            It can also be a string and the only string allowed is -
            'leave_one_out' to do leave-one-out cross validation on the
            dataset.
        max_iter : int, optional
            The maximum number of iterations for increasing the resolution of
            the grid search within the alpha_list's min-max range.
            The default is 2.
        tol : float, optional
            The tolerance to stop iterations. The default is 1e-2.
        model_type : str, optional
            The model's name - either 'Lasso' or 'Ridge'.
            The default is 'Lasso'.
        scoring : str, optional
            The scoring method to use. The default is
            'mean_absolute_percentage_error'.
    model_folder : str, optional
        Either the name of the model or a complete path to the folder where a
        pickle file containing info about the fit model will be saved.
        The default is None.
    save_to_drive : bool, optional
        Whether to save the model in shared drive. Works only when you have
        access to the shared drive. The default is False.
    overwrite_existing_model : bool, optional
        Whether to overwrite an existing model while saving.

    Raises
    ------
    FileNotFoundError
        If model_folder is not None, save_to_drive is True, and you do not
        have access to the shared drive.

    ValueError
        When the length of feat_labels is not equal to the number of features
        or columns in X.

    Returns
    -------
    fit_results : dict
        A dictionary consisting of the current function's arguements and the
        fit model.

    """
    if len(feat_labels) != X_train.shape[1]:
        raise ValueError("The feats_labels and X's columns must be of same\n\
                         length.")
    # Create the linear regression instance
    reg = linear_model_selector(model_type)

    # Get the best regularization parameter by cross-validation
    if model_type == 'Lasso' or model_type == 'Ridge':
        tuning_tuple = alpha_tuning(X_train, y_train,
                                    **alpha_tuning_params)
        best_alpha, alpha_tuning_scores, alpha_tuning_scoring = tuning_tuple
        reg.set_params(alpha=best_alpha)
        print("------------------------------------------")
        print("The best regularization alpha: ", best_alpha)
        print("------------------------------------------\n")

    # Scale the data
    scaler = StandardScaler()
    X_train_fit = scaler.fit_transform(X_train)
    X_test_fit = scaler.transform(X_test, copy=True)

    # Fit the model
    fit_results = {}

    print('Fitting ', model_type, ' model on traning dataset...\n')
    reg.fit(X_train_fit, y_train)
    fit_results['model_type'] = model_type
    fit_results['X_train'] = X_train_fit
    fit_results['y_train'] = y_train
    fit_results['X_test'] = X_test_fit
    fit_results['y_test'] = y_test
    fit_results['feat_labels'] = feat_labels,
    fit_results['y_label'] = y_label
    fit_results['alpha_tuning_params'] = alpha_tuning_params
    fit_results['alpha_tuning_scores'] = alpha_tuning_scores
    fit_results['alpha_tuning_scoring'] = alpha_tuning_scoring
    fit_results['model'] = reg
    fit_results['scaler'] = scaler

    if model_folder is not None:
        # Raise error if you do not have access to the shared drive
        if save_to_drive and not os.path.exists(drive_models_folder):
            raise FileNotFoundError("YOU DO NOT HAVE ACCESS TO THE\n\
                                    SHARED DRIVE.")

        # The path to the models folder
        if save_to_drive:
            saved_models_folder = drive_models_folder
        else:
            saved_models_folder = local_models_folder

        # If the model_folder is just the name of the model, not path
        if not ('/' or '\\' in model_folder):
            model_folder = os.path.join(saved_models_folder, model_folder)

        # Prompt new name input if another model exists with the same name
        while os.path.exists(model_folder) and not overwrite_existing_model:
            print_str = "A saved model already exists at\n" + model_folder
            print_str += "\nPlease enter a new model name or path:\n>>"
            model_folder = input(print_str)
            if not ('/' or '\\' in model_folder):
                model_folder = os.path.join(saved_models_folder, model_folder)

        os.makedirs(model_folder, exist_ok=True)
        fit_pickle_path = os.path.join(model_folder, fit_pickle_name)
        with open(fit_pickle_path, 'wb') as file:
            pickle.dump(fit_results, file)

    return fit_results

##############################################################################
# Utility functions for linear_regressor()
##############################################################################


def linear_model_selector(model_type):
    """
    Returns a new regressor instance based on the model_type provided.
    The available model types are -
    1. 'SimpleLinear'
    2. 'Lasso'
    3. 'Ridge'
    4. 'TheilSen'

    Parameters
    ----------
    model_type : str
        The name of the linear model. It must be one of the names listed above.

    Returns
    -------
    reg_fit : sklearn.linear_model object
        The sklearn's Linear model object.

    """
    if model_type == 'Lasso':
        reg_fit = Lasso(max_iter=5000)
    elif model_type == 'Ridge':
        reg_fit = Ridge(max_iter=5000)
    elif model_type == 'SimpleLinear':
        reg_fit = LinearRegression()
    elif model_type == 'TheilSen':
        reg_fit = TheilSenRegressor()

    return reg_fit


def score(y_true, y_pred, scoring='mean_absolute_percentage_error',
          always_return=True):
    """
    Returns the error score using the sklearn's list of error metrics. Note
    that the scoring string must be one of the available errors in sklearn.

    The available scoring functions can be found here-
    https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics

    Parameters
    -----------
    y_true : np.array
        True Y values
    y_pred : np.array
        Predicted y values
    scoring : str, optional
        The scoring to return. The default is 'mean_absolute_percentage_error'.
    always_return : bool, optional
        Set this to True if you want the function to return the mean
        absolute percentage error if the specified error is not available
        in the scikit version of yours.

    Returns
    -----------
    float
        The required score value
    """
    try:
        error = eval('metrics.'+scoring+'(y_true, y_pred)')
    except AttributeError:
        # Because some errors are not available
        # in older versions
        if always_return:
            print("This scoring metric is not available. Returning the\n\
                  mean absolute percentage error instead...\n")
            denominator = np.maximum(1e-10, y_true)
            error = np.mean(np.divide(np.abs(y_true-y_pred), denominator))
        else:
            raise Exception("This scoring metric is not available.")

    return error


def alpha_tuning(X, y,
                 alpha_list=np.logspace(-3, 1, 5),
                 k_fold=5,
                 max_iter=2,
                 tol=1e-2,
                 model_type='Lasso',
                 scoring='mean_absolute_percentage_error'):
    """
    Uses cross-validation to find the best regulariation alpha value for the
    given  model_type over the given dataset. Use k_fold=1 for finding the
    best regularization hyperparameter on the enitre dataset without cross
    validation.This function works only for 'Lasso' and 'Ridge' model types.

    Parameters
    ----------
    X : numpy.ndarray
        The dataset's X.
    y : numpy.ndarray
        The dataset's y.
    alpha_list : numpy.ndarray, optional
        The initial list of alpha values. The default is np.logspace(-3, 1, 5).
    k_fold : int or str, optional
        The integer number of k-folds for cross validation. The default is 5.
        Use k_fold=1 for finding the best regularization hyperparameter on
        the enitre dataset without cross validation.
        It can also be a string and the only string allowed is -
        'leave_one_out' to do leave-one-out cross validation on the dataset.
    max_iter : int, optional
        The maximum number of iterations for increasing the resolution of
        the grid search within the alpha_list's min-max range.
        The default is 2.
    tol : float, optional
        The tolerance to stop iterations. The default is 1e-2.
    model_type : str, optional
        The model's name - either 'Lasso' or 'Ridge'. The default is 'Lasso'.
    scoring : str, optional
        The scoring method to use. The default is
        'mean_absolute_percentage_error'.

    Raises
    ------
    ValueError
        When model_type is not 'Lasso' or 'Ridge'.

    Returns
    -------
    best_alpha : float
        The best regularization hyperparameter value for the given dataset
        and model type.
    grid_search_results : dict
        The dictionary with two keys - 'alpha' and 'score' containing
        arrays of alpha values and correponding scores.
    scoring : str
        The scoring metric used for tuning

    """
    if model_type != 'Lasso' and model_type != 'Ridge':
        raise ValueError("The tuning of the regularization alpha parameter\n\
                         is available only to the 'Lasso' and 'Ridge' model\n\
                             types.")
    reg_fit = linear_model_selector(model_type)
    k_fold_name = str(k_fold)
    if k_fold == 'leave_one_out':
        k_fold = len(y)

    grand_alpha_list = []
    grand_score_list = []
    # Check if a valid scoring string is passed
    if scoring not in ['r2', 'explained_variance', 'max_error']:
        scoring = 'neg_'+scoring
    if scoring not in sorted(metrics.SCORERS.keys()):
        print(scoring, " is not available in sklearn.metrics. Using\n\
              'neg_mean_absolute_error' instead.\n")
        scoring = 'neg_mean_absolute_error'
    if k_fold == 1:
        scoring = scoring.partition('neg_')[-1]

    def scan_alpha_vals(reg_fit, alpha_list):
        # Get the best alpha over the entire X
        best_score = np.inf
        if k_fold == 1:
            for alpha in alpha_list:
                reg_fit = reg_fit.set_params(alpha=alpha)
                reg_fit.fit(X, y)
                y_pred = reg_fit.predict(X)
                error_score = score(y, y_pred, scoring=scoring)

                # The best score is the minimum score.
                if error_score < best_score:
                    best_score = error_score
                    best_alpha = alpha

                grand_alpha_list.append(alpha)
                grand_score_list.append(error_score)

        # other use the k_fold number of splits for tuning with
        # validatoin error
        else:
            grid = GridSearchCV(estimator=reg_fit,
                                param_grid={'alpha': alpha_list},
                                scoring=scoring,
                                cv=k_fold,
                                verbose=1)
            grid.fit(X, y)
            best_alpha = grid.best_params_['alpha']
            best_score = grid.best_score_
            grand_alpha_list.extend(list(alpha_list))
            mean_scores_list = grid.cv_results_['mean_test_score']
            grand_score_list.extend(list(mean_scores_list))

        return best_alpha

    print("------------------------------------------")
    print("Regularization hyperparameter tuning by cv")
    print("------------------------------------------")
    print('alpha range        :  ({0:.2e}, {0:.2e})'.format(alpha_list.min(),
                                                            alpha_list.max()))
    print('No. of folds       : ', k_fold_name)
    print('Maximum iterations : ', max_iter)
    print('Tolerance          :  {0:.2e}'.format(tol))
    print('Model type         : ', model_type)
    print('Scoring metric     : ', scoring, '\n')

    # Now, iterate over better resolutions of alpha list until the best alpha
    # is obtained.
    best_alpha = np.inf
    for i in range(max_iter):
        best_alpha_new = scan_alpha_vals(reg_fit, alpha_list)
        delta_alpha = min(np.abs(best_alpha_new - [alpha_list.min(),
                                                   alpha_list.max()]))
        alpha_list = np.linspace(best_alpha_new-delta_alpha,
                                 best_alpha_new+delta_alpha, 5)

        if np.abs(best_alpha - best_alpha_new) < tol:
            break
        else:
            best_alpha = best_alpha_new

    grand_alpha_list, unique_inds = np.unique(grand_alpha_list,
                                              return_index=True)
    grand_score_list = np.abs(np.array(grand_score_list)[unique_inds])
    return_tuple = (best_alpha,
                    dict(alpha=grand_alpha_list, score=grand_score_list),
                    scoring)
    return return_tuple
