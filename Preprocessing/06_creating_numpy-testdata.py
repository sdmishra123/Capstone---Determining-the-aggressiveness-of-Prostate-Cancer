#!/usr/bin/env python
# coding: utf-8

# ##### Author : Sapna Mishra
# ##### Project : Determining the Aggressiveness of Cancer using mpMRI Scans
# ##### Last Modified: 8th Oct 2020
# ##### Task:- Region of Interest Extraction

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# In[10]:


def generate_image_sequence(is_training_data, data):
    
    t2_tra_sequence = data[data['sequence_type'] == 'tse_tra']
#     t2_sag_sequence = data[data['sequence_type'] == 'tse_sag']
    adc_sequence    = data[data['sequence_type'] == 'adc']
    bval_sequence   = data[data['sequence_type'] == 'bval']
    ktrans_sequence = data[data['sequence_type'] == 'ktrans']


    def balance_classes(sequence):
         
        #oversampled by adding 4 rotations (45, 90, 180, 270) to the dataset - UPSAMPLING
        #Train set:
        #Gleason score 1: 37%
        #Gleason score 2: 32%
        #Gleason score 3: 18%
        #Gleason score 4: 7%
        #Gleason score 5: 6%

        patch_sequence = []

        for row_id, row in sequence.iterrows():
            patch_sequence.append(row.eq_patch)

        return (np.array(patch_sequence))
    
    def zero_mean_unit_variance(image_array):

        # https://stackoverflow.com/questions/41652330/centering-of-array-of-images-in-python
        # https://stackoverflow.com/questions/36394340/centering-a-numpy-array-of-images
       
        image_array_float = np.array(image_array, dtype=np.float, copy = True)
        mean = np.mean(image_array_float, axis=(0))
        std = np.std(image_array_float, axis=(0))
        standardized_images = (image_array_float - mean) / std
        
        return standardized_images

    
    t2_tra_images      = balance_classes(t2_tra_sequence)
#     t2_sag_sequence = balance_classes(t2_sag_sequence)
    adc_images         = balance_classes(adc_sequence)
    bval_images        = balance_classes(bval_sequence)
    ktrans_images      = balance_classes(ktrans_sequence)
       
    t2_tra_norm   = zero_mean_unit_variance(t2_tra_images)
#     t2_sag_norm   = zero_mean_unit_variance(t2_tra_images)
    adc_norm      = zero_mean_unit_variance(adc_images)
    bval_norm     = zero_mean_unit_variance(bval_images)
    ktrans_norm   = zero_mean_unit_variance(ktrans_images)

    return {
            'tse_tra':(t2_tra_norm),
#             'tse_sag':(t2_tra_norm)
            'adc':(adc_norm),
            'bval':(bval_norm),
            'ktrans':(ktrans_norm)
    } 


# In[11]:


def persist_numpy_to_disk(is_training_data, data):
    
    tse_tra_images = data.get('tse_tra')
#     tse_sag_images = data.get('tse_sag')[0]
    adc_images = data.get('adc')
    bval_images = data.get('bval')
    ktrans_images = data.get('ktrans')

    if is_training_data:
        root_path = 'C:/Sapna/Graham/Capstone/data/test/generated/numpy'
        
        np.save(Path(root_path + '/tse_tra/X_train.npy'),tse_tra_images)       
#         np.save(Path(root_path + '/tse_sag/X_train.npy'), tse_sag_images)
        np.save(Path(root_path + '/adc/X_train.npy'), adc_images)
        np.save(Path(root_path + '/bval/X_train.npy'), bval_images)
        np.save(Path(root_path + '/ktrans/X_train.npy'), ktrans_images)

    else:
        root_path = 'C:/Sapna/Graham/Capstone/data/test/generated/numpy'
        
        np.save(Path(root_path + '/tse_tra/X_test.npy'), tse_tra_images)        
#         np.save(Path(root_path + '/tse_tra/X_test.npy'), tse_sag_images)
        np.save(Path(root_path + '/adc/X_test.npy'), adc_images)
        np.save(Path(root_path + '/bval/X_test.npy'), bval_images)
        np.save(Path(root_path + '/ktrans/X_test.npy'), ktrans_images)


# In[12]:


def main():
    is_training_data = False
    dataset_type = input('Which dataset are you working with? (1-Train; 2-Test):')
    if dataset_type == str(1):
        is_training_data = True

    if is_training_data:
        data = pd.read_pickle('C:/Sapna/Graham/Capstone/data/train/generated/dataframes/training.pkl')
    else:
        data = pd.read_pickle('C:/Sapna/Graham/Capstone/data/test/generated/dataframes/testing.pkl')
    
    numpy_data = generate_image_sequence(is_training_data, data)
    persist_numpy_to_disk(is_training_data, numpy_data)

main()


# In[ ]:




