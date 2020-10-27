#!/usr/bin/env python
# coding: utf-8

# ##### Author : Sapna Mishra
# ##### Project : Determining the Aggressiveness of Cancer using mpMRI Scans
# ##### Last Modified: 7th Oct 2020
# ##### Task:- Creating directories

# In[22]:


from pathlib import Path


# In[23]:


def generate_patient_ids(dataset_type):
    """
    This function generates the patient_ids for the directories to be created below. 
    Ids are extracted from the raw dataset file structure.
    """
    
    patient_ids = []
    path_to_date = Path()
    
    if dataset_type == str(1):
        path_to_data = Path('C:/Sapna/Graham/Capstone/data/train/PROSTATEx - Copy')
    else:
        path_to_data = Path('C:/Sapna/Graham/Capstone/data/test/PROSTATEx - Copy')
    
    # Get list of patient_ids in folder
    patient_folders = [x for x in path_to_data.iterdir() if x.is_dir()]
    for patient_folder in patient_folders:
        patient_ids.append(str(patient_folder.stem))
    return patient_ids 


# In[24]:


def generate_nrrd_ds(patient_ids, dataset_type):
    """
    This function generates the directory structure for the nifti files
    generated from the dicom files.

    Directory structure for generated data:
    ProstateX/generated/train/nrrd
    ProstateX/generated/test/nrrd
    """
    for patient_id in patient_ids:
        if dataset_type == str(1):
            new_path = Path(str('C:/Sapna/Graham/Capstone/data/train/generated/nrrd/' + patient_id))
            new_path.mkdir(parents = True, exist_ok = True)
            new_path.joinpath('tse_tra').mkdir(parents = True, exist_ok = True)
            new_path.joinpath('tse_sag').mkdir(parents = True, exist_ok = True)
            new_path.joinpath('bval').mkdir(parents = True, exist_ok = True)
            new_path.joinpath('adc').mkdir(parents = True, exist_ok = True)
            new_path.joinpath('ktrans').mkdir(parents = True, exist_ok = True)

        else:
            new_path = Path(str('C:/Sapna/Graham/Capstone/data/test/generated/nrrd/' + patient_id))
            new_path.mkdir(parents = True, exist_ok = True)
            new_path.joinpath('tse_tra').mkdir(parents = True, exist_ok = True)
            new_path.joinpath('tse_sag').mkdir(parents = True, exist_ok = True)
            new_path.joinpath('bval').mkdir(parents = True, exist_ok = True)
            new_path.joinpath('adc').mkdir(parents = True, exist_ok = True)
            new_path.joinpath('ktrans').mkdir(parents = True, exist_ok = True)


# In[25]:


def generate_nrrd_resampled_ds(patient_ids, dataset_type):
    """
    This function generates the directory structure for the nifti files
    generated from the dicom files.

    Directory structure for generated data:
    ProstateX/generated/train/nrrd_resampled
    ProstateX/generated/test/nrrd_resampled
    """
    for patient_id in patient_ids:
        if dataset_type == str(1):
            new_path = Path(str('C:/Sapna/Graham/Capstone/data/train/generated/nrrd_resampled/' + patient_id))
            new_path.mkdir(parents = True, exist_ok = True)
            new_path.joinpath('tse_tra').mkdir(parents = True, exist_ok = True)
            new_path.joinpath('tse_sag').mkdir(parents = True, exist_ok = True)
            new_path.joinpath('bval').mkdir(parents = True, exist_ok = True)
            new_path.joinpath('adc').mkdir(parents = True, exist_ok = True)
            new_path.joinpath('ktrans').mkdir(parents = True, exist_ok = True)

        else:
            new_path = Path(str('C:/Sapna/Graham/Capstone/data/test/generated/nrrd_resampled/' + patient_id))
            new_path.mkdir(parents = True, exist_ok = True)
            new_path.joinpath('tse_tra').mkdir(parents = True, exist_ok = True)
            new_path.joinpath('tse_sag').mkdir(parents = True, exist_ok = True)
            new_path.joinpath('bval').mkdir(parents = True, exist_ok = True)
            new_path.joinpath('adc').mkdir(parents = True, exist_ok = True)
            new_path.joinpath('ktrans').mkdir(parents = True, exist_ok = True)


# In[26]:


def generate_numpy_ds(dataset_type):
    """
    This function generates the directory structure for the final numpy
    arrays for the training and test sets. 
    
    Director structure for processed data:
    ProstateX/generated/train/numpy
    ProstateX/generated/test/numpy
    """
    if dataset_type == str(1):
        new_path = Path('C:/Sapna/Graham/Capstone/data/train/generated/numpy/')
        new_path.mkdir(parents = True, exist_ok = True)
        new_path.joinpath('tse_tra').mkdir(parents = True, exist_ok = True)
        new_path.joinpath('tse_sag').mkdir(parents = True, exist_ok = True)
        new_path.joinpath('bval').mkdir(parents = True, exist_ok = True)
        new_path.joinpath('adc').mkdir(parents = True, exist_ok = True)
        new_path.joinpath('ktrans').mkdir(parents = True, exist_ok = True)
    else:
        new_path = Path('C:/Sapna/Graham/Capstone/data/test/generated/numpy/')
        new_path.mkdir(parents = True, exist_ok = True)
        new_path.joinpath('tse_tra').mkdir(parents = True, exist_ok = True)
        new_path.joinpath('tse_sag').mkdir(parents = True, exist_ok = True)
        new_path.joinpath('bval').mkdir(parents = True, exist_ok = True)
        new_path.joinpath('adc').mkdir(parents = True, exist_ok = True)
        new_path.joinpath('ktrans').mkdir(parents = True, exist_ok = True)


# In[27]:


def generate_dataframe_ds(dataset_type):
    if dataset_type == str(1):
        new_path = Path('C:/Sapna/Graham/Capstone/data/train/generated/dataframes/')
        new_path.mkdir(parents = True, exist_ok = True)

    else:
        new_path = Path('C:/Sapna/Graham/Capstone/data/test/generated/dataframes/')
        new_path.mkdir(parents = True, exist_ok = True)


# In[28]:


def generate_logs_ds(dataset_type):
    if dataset_type == str(1):
        new_path = Path('C:/Sapna/Graham/Capstone/data/train/generated/logs/')
        new_path.mkdir(parents = True, exist_ok = True)

    else:
        new_path = Path('C:/Sapna/Graham/Capstone/data/test/generated/logs/')
        new_path.mkdir(parents = True, exist_ok = True)


# In[30]:


def main():
    dataset_type = input('Generate directory structure for which type of data (1-Train; 2-Test):')
    patient_ids = generate_patient_ids(dataset_type)
    generate_nrrd_ds(patient_ids, dataset_type)
    generate_nrrd_resampled_ds(patient_ids, dataset_type)
    generate_numpy_ds(dataset_type)
    generate_dataframe_ds(dataset_type)
    generate_logs_ds(dataset_type)
    print('Done creating directory structure...')
main()

