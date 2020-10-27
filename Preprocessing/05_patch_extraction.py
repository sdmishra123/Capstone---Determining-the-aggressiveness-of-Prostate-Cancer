#!/usr/bin/env python
# coding: utf-8

# ##### Author : Sapna Mishra
# ##### Project : Determining the Aggressiveness of Cancer using mpMRI Scans
# ##### Last Modified: 8th Oct 2020
# ##### Task:- Region of Interest Extraction

# In[1]:


import pandas as pd
import numpy as np
import SimpleITK as sitk
import pickle
from pathlib import Path
from scipy import ndimage
from skimage import exposure
from sklearn.preprocessing import MinMaxScaler

import nibabel as nib
from nibabel.testing import data_path
from nilearn import plotting
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, IntSlider, ToggleButtons


# In[2]:


problem_cases = []

def generate_patches(row, patch_sizes):
 
    path_to_resampled_file = row.resampled_nifti
    reported_pos = row.pos

    if 'tse_tra' in row.DCMSerDescr:
        patch_size = patch_sizes.get('tse_tra')
    elif 'adc' in row.DCMSerDescr:
        patch_size = patch_sizes.get('adc')
    elif 'bval' in row.DCMSerDescr:
        patch_size = patch_sizes.get('bval')
    elif 'ktrans' in row.DCMSerDescr:
        patch_size = patch_sizes.get('ktrans')
    else:
        print("Incorrect Sequence Name")

    def load_image(path_to_resampled_file):
        
        sitk_image = sitk.ReadImage(str(path_to_resampled_file))
        image_array = sitk.GetArrayViewFromImage(sitk_image)
        
        return sitk_image, image_array
    
    def calculate_location_of_finding(sitk_image, reported_pos):
        location_of_finding = sitk_image.TransformPhysicalPointToIndex(reported_pos)
        return location_of_finding

    def equalize_image(image_array):
        equalized_image_array = exposure.equalize_hist(image_array)
        return equalized_image_array
    
    def extract_patch(image_array, location_of_finding, patch_size):
        x = location_of_finding[0]
        y = location_of_finding[1]

        x_start = x - (patch_size // 2)
        x_end = x + (patch_size // 2)
        y_start = y - (patch_size // 2)
        y_end = y + (patch_size // 2)

        try:
            extracted_patch = image_array[location_of_finding[2], y_start:y_end, x_start:x_end]
        except IndexError:
            extracted_patch = image_array[-1, y_start:y_end, x_start:x_end]
            problem_cases.append(row.ProxID)
            problem_cases.append(row.DCMSerDescr)
            print('Problem with image:', row.ProxID, path_to_resampled_file)
            pass 

        return extracted_patch
    
    def generate_rotations(image_array):
        patch_45 = ndimage.rotate(image_array, 45, reshape=False)
        patch_90 = ndimage.rotate(image_array, 90, reshape=False)
        patch_180 = ndimage.rotate(image_array, 180, reshape=False)
        patch_270 = ndimage.rotate(image_array, 270, reshape=False)
        return (patch_45, patch_90, patch_180, patch_270)
    
    def rescale_intensities_of_patch(patch): 
        scaler = MinMaxScaler(feature_range = (0,1)).fit(patch)
        rescaled_patch = scaler.transform(patch)
        return rescaled_patch

    sitk_image, image_array = load_image(path_to_resampled_file)
    location_of_finding = calculate_location_of_finding(sitk_image, reported_pos)
    
    raw_image_array = image_array.copy()
    equalized_image_array = equalize_image(image_array)
      
    patch = extract_patch(raw_image_array, location_of_finding, patch_size)
    eq_patch = extract_patch(equalized_image_array, location_of_finding, patch_size)
    eq_45 = generate_rotations(eq_patch)[0]
    eq_90 = generate_rotations(eq_patch)[1]
    eq_180 = generate_rotations(eq_patch)[2]
    eq_270 = generate_rotations(eq_patch)[3]

    patch_01 = rescale_intensities_of_patch(patch)
    eq_patch_01 = rescale_intensities_of_patch(eq_patch)
    eq_45_01 = rescale_intensities_of_patch(eq_45)
    eq_90_01 = rescale_intensities_of_patch(eq_90)
    eq_180_01 = rescale_intensities_of_patch(eq_180)
    eq_270_01 = rescale_intensities_of_patch(eq_270)
    
    patch_values = pd.Series({'patch':patch_01, 'eq_patch':eq_patch_01, 'eq_45':eq_45_01, 'eq_90':eq_90_01, 'eq_180':eq_180_01, 'eq_270':eq_270_01})
    return patch_values


# In[3]:


def add_patch_columns_to_df(dataframe, patch_sizes):
    new_data = dataframe.apply(generate_patches, patch_sizes = patch_sizes, axis = 1)
    merged_frame = pd.concat([dataframe, new_data], axis=1)
    return merged_frame


# In[4]:


def remove_problem_cases(dataframe, problem_cases):
    problem_cases = set(problem_cases)
    to_delete = []
    for row_id, row in dataframe.iterrows():
        if (row.ProxID in problem_cases) and (row.DCMSerDescr in problem_cases):
            to_delete.append(row.ProxID)
    clean_dataframe = dataframe[~dataframe['ProxID'].isin(set(to_delete))]
    return clean_dataframe


# In[5]:


def persist_data(is_training_data, dataframe):
    if is_training_data:
        dataframe.to_csv('C:/Sapna/Graham/Capstone/data/train/generated/dataframes/training.csv')
        dataframe.to_pickle('C:/Sapna/Graham/Capstone/data/train/generated/dataframes/training.pkl')
    else:
        dataframe.to_csv('C:/Sapna/Graham/Capstone/data/test/generated/dataframes/testing.csv')
        dataframe.to_pickle('C:/Sapna/Graham/Capstone/data/test/generated/dataframes/testing.pkl')


# In[7]:


def main():
    is_training_data = False
    dataset_type = input('Which dataset are you working with? (1-Train; 2-Test):')
    if dataset_type == str(1):
        is_training_data = True

    patch_sizes = {
        'tse_tra': 32,
        'tse_sag': 32,
        'adc': 8,
        'bval':8,
        'ktrans':8
    }
    
    if is_training_data:
        dataset = pd.read_pickle('C:/Sapna/Graham/Capstone/data/train/generated/dataframes/training_meta_data.pkl')
        dataset= dataset[dataset["sequence_type"]!="tse_sag"]
        complete_dataset = add_patch_columns_to_df(dataset, patch_sizes)
        clean_dataset = remove_problem_cases(complete_dataset, problem_cases)
        persist_data(is_training_data, clean_dataset)
    else:
        dataset = pd.read_pickle('C:/Sapna/Graham/Capstone/data/test/generated/dataframes/test_meta_data.pkl')
        dataset= dataset[dataset["sequence_type"]!="tse_sag"]
        complete_dataset = add_patch_columns_to_df(dataset, patch_sizes)
        clean_dataset = remove_problem_cases(complete_dataset, problem_cases)
        persist_data(is_training_data, clean_dataset)

main()


# In[ ]:




