# -*- coding: utf-8 -*-
"""
This module contains functions for miscellaneous or general purposes.
"""
import os
import re

def booleanize(input_dict):
    
    """
    This function is to convert all boolean strings in a dictionary to
    boolean types for easy use in python and returns a copy of it.
    
    Parameters
    ----------
    input_dict : dict
        The input dictionary
        
    Returns
    ----------
    dict
        The output dictionary
    """
    input_dict_copy = input_dict.copy()
    for key, val in input_dict_copy.items():
        if val == 'true' or val == 'True' or val == 'False' or val == 'false':
            input_dict_copy[key] = val == 'true' or val == 'True'
    
    return input_dict_copy


def convert_to_long_path(path):
    """Converts regular filepath string to a raw string for long path
    compatibility. NOTE: This function is important especially when a 
    Windows operating system is in use, because Windows allows only
    a maximum path length of 260 chars. It can also be used to make
    a file path compatible to the current os.
    
    Parameters
    -----------
    path : str
        The file or directory path
    
    Returns
    -----------
    str
        The file or directory path made compatible for Windows, irrespective
        of its length
    """
    if os.name == 'nt':
        path_list = path.partition(":")
        path1 = path_list[0]
        path2 = path_list[2]
        
        # path1
        while path1.startswith("\\") or path1.startswith("?") or path1.startswith("/"):
            path1 = path1.lstrip("\\")
            path1 = path1.lstrip("?")
            path1 = path1.lstrip("/")
        path1 = "\\\\?\\" + path1
        
        #path2
        str_list = re.split('[\\\\/]', path2)
        while '' in str_list:
            str_list.remove('')
        path2 = '\\'.join(str_list)
        
        return path1 + ':\\' + path2
    
    else:
        return os.path.join(*re.split('[\\\\/]', path))
