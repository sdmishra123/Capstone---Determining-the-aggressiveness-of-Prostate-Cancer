#!/usr/bin/env python
# coding: utf-8

# ##### Author : Sapna Mishra
# ##### Project : Determining the Aggressiveness of Cancer using mpMRI Scans
# ##### Last Modified: 7th Oct 2020
# ##### Task: Resampling the NRRD files

# In[1]:


import SimpleITK as sitk
import numpy as np
from pathlib import Path


# In[2]:


path_to_nifti = Path()
path_to_nifti_resampled = Path()

def resample_voxel_spacing(is_training_data, desired_voxel_spacing):
    
    # Setting the paths

    path_to_nifti = Path()
    path_to_nifti_resampled = Path()
    counter = 1

    if is_training_data:
        path_to_nifti = Path('C:/Sapna/Graham/Capstone/data/train/generated/nrrd')
        path_to_nifti_resampled =Path('C:/Sapna/Graham/Capstone/data/train/generated/nrrd_resampled' )  
    else:
        path_to_nifti = Path('C:/Sapna/Graham/Capstone/data/test/generated/nrrd')
        path_to_nifti_resampled =Path('C:/Sapna/Graham/Capstone/data/test/generated/nrrd_resampled' )
    
    # Methods

    def resample_image(desired_voxel_spacing, source_file_path):
        image = sitk.ReadImage(str(source_file_path))
        original_image_spacing = image.GetSpacing()

        if original_image_spacing != desired_voxel_spacing:
            ### HOW TO RESAMPLE SITK_IMAGE TO A NEW SPACING ###
            ### SOURCE: https://github.com/SimpleITK/SimpleITK/issues/561 ###
            
            # Converting to np array for calculations of new_size
            original_size_array = np.array(image.GetSize(), dtype = np.int)
            original_spac_array = np.array(image.GetSpacing())
            desired_spac_array = np.array(desired_voxel_spacing)
            
            new_size = original_size_array * (original_spac_array / desired_spac_array)
            new_size = np.ceil(new_size).astype(np.int)
            new_size = [int(s) for s in new_size]
            new_size = tuple(new_size)
            
            # Create the resample filter
            resample = sitk.ResampleImageFilter()
            resample.SetInterpolator(sitk.sitkLinear) 
            resample.SetSize(new_size)
            resample.SetOutputOrigin(image.GetOrigin()) 
            resample.SetOutputSpacing(desired_voxel_spacing)
            resample.SetOutputDirection(image.GetDirection())
            
            try:
                resampled_image = resample.Execute(image)
                # Print the changes
                print('\n')
                print('Resampling:', "/".join(source_file_path.parts[-5:]))
                print('original spacing:', image.GetSpacing())
                print('desired spacing:', desired_voxel_spacing)
                print('resampled spacing:', resampled_image.GetSpacing())
                print('original size:', image.GetSize())
                print('resampled size:', resampled_image.GetSize())
                print('\n')
            except:
                print('Problem with resampling image.')
                
        else:
            resampled_image = image
        
        return resampled_image
    
    def write_resampled_image(image, path, counter):
        writer = sitk.ImageFileWriter()
        writer.SetFileName(str(path))
        writer.Execute(image)
        print('Saving image to:', "/".join(path.parts[-5:]))
        counter = counter + 1
        return counter        

    patient_folders = [x for x in path_to_nifti.iterdir() if x.is_dir()]
    for subdirectory in patient_folders:
        if 't2_tra' in str(subdirectory):
            for file_path in subdirectory.rglob('*.*'):
                path_t2_resampled = path_to_nifti_resampled.joinpath("/".join(file_path.parts[-2:]))
                t2_resampled = resample_image(desired_voxel_spacing.get('t2_tra'), file_path)
                counter = write_resampled_image(t2_resampled, path_t2_resampled,counter)
        if 't2_sag' in str(subdirectory):
            for file_path in subdirectory.rglob('*.*'):
                path_t2_resampled = path_to_nifti_resampled.joinpath("/".join(file_path.parts[-2:]))
                t2_resampled = resample_image(desired_voxel_spacing.get('t2_sag'), file_path)
                counter = write_resampled_image(t2_resampled, path_t2_resampled,counter)
        if 'adc' in str(subdirectory):
            for file_path in subdirectory.rglob('*.*'):
                path_adc_resampled = path_to_nifti_resampled.joinpath("/".join(file_path.parts[-2:]))
                adc_resampled  = resample_image(desired_voxel_spacing.get('adc'), file_path)
                counter = write_resampled_image(adc_resampled, path_adc_resampled,counter) 
        if 'bval' in str(subdirectory):
            for file_path in subdirectory.rglob('*.*'):
                path_bval_resampled = path_to_nifti_resampled.joinpath("/".join(file_path.parts[-2:]))
                bval_resampled = resample_image(desired_voxel_spacing.get('bval'), file_path)
                counter = write_resampled_image(bval_resampled, path_bval_resampled,counter)    
        if 'ktrans' in str(subdirectory):
            for file_path in subdirectory.rglob('*.*'):
                path_ktrans_resampled = path_to_nifti_resampled.joinpath("/".join(file_path.parts[-2:]))
                ktrans_resampled = resample_image(desired_voxel_spacing.get('ktrans'), file_path)
                counter = write_resampled_image(ktrans_resampled, path_ktrans_resampled,counter)               

    print('\n ++++ Files reviewed for resampling:', counter)


# In[3]:


def main():
    is_training_data = False
    
    # Obtain input about datset from user
    dataset_type = input('Which dataset is being resampled? (1-Train; 2-Test):')
    if dataset_type == str(1):
        is_training_data = True
    
    # Set desired spacing based on EDA
    desired_voxel = {'t2_tra':(0.5,0.5,3.0),
                     't2_sag':(0.5,0.5,3.0),
                     'adc':(2.0,2.0,3.0),
                     'bval':(2.0,2.0,3.0),
                     'ktrans':(1.5,1.5,4.0)} 
    
    resample_voxel_spacing(is_training_data, desired_voxel)

main()


# In[ ]:




