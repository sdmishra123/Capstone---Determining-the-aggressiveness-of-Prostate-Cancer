#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd
import pickle
from pathlib import Path
import os
import re


# In[82]:


def generate_cases_meta_df(is_training_data,sequence_type):
    """
    This function generates a data frame containing the necessary information (ProxID, DCMSerDesc,
    and path to resampled NIFTI file) for cases so that they can be joined to tabular information 
    provided by the research team. Data that will be merged with dataset are found in ProstateX-Images
    and ProstateX-Images-KTrans files (Train and Test Respectively) 
    """

    if is_training_data:
        path_lesion_information = 'C:/Sapna/Graham/Capstone/data/train/lesion_info'
        path_resampled_nifti = 'C:/Sapna/Graham/Capstone/data/train/generated/nrrd_resampled'
    else:
        path_lesion_information = 'C:/Sapna/Graham/Capstone/data/test/lesion_info'
        path_resampled_nifti = 'C:/Sapna/Graham/Capstone/data/test/generated/nrrd_resampled'
    
    patient_data = {}
    for f1, filename1 in enumerate(os.listdir(path_resampled_nifti)):
        if filename1 == sequence_type:
            for f2, filename2 in enumerate(os.listdir(str(Path(path_resampled_nifti))+'/'+ filename1)):     
                
                split = filename2.split('.')
                constructed_DCMSerDescr = split[0]

                path_to_resampled = str(path_resampled_nifti) + str("/") + sequence_type + str("/") + str(filename2) 

                if 'tse_sag' in constructed_DCMSerDescr:
                    sequence_type = 'tse_sag'
                    name1 = constructed_DCMSerDescr[15:]
                    constructed_DCMSerDescr = name1[:10]
                    constructed_DCMSerDescr = constructed_DCMSerDescr[0:10]

                elif 'tse_tra' in constructed_DCMSerDescr:          
                    sequence_type = 'tse_tra'
                    name1 = constructed_DCMSerDescr[15:]
                    constructed_DCMSerDescr = name1[:10]  
                    constructed_DCMSerDescr = constructed_DCMSerDescr[0:10]

                elif 'ADC' in constructed_DCMSerDescr:    
                    sequence_type = 'adc'
                    name1 = constructed_DCMSerDescr[15:]  
                    name2 = name1.rsplit("_",1)
                    constructed_DCMSerDescr = name2[0]

                elif 'BVAL' in constructed_DCMSerDescr:  
                    sequence_type = 'bval'
                    name1 = constructed_DCMSerDescr[15:]  
                    name2 = name1.rsplit("_",1)
                    constructed_DCMSerDescr = name2[0]

                elif 'Ktrans' in constructed_DCMSerDescr:    
                    sequence_type = 'ktrans'
                    constructed_DCMSerDescr = constructed_DCMSerDescr 

                else:
                    print("Sequence type is incorrect")

                patient_id = filename2[0:14]
                key = patient_id
                value = [constructed_DCMSerDescr, path_to_resampled, sequence_type]
                patient_data[key] = value
        
    cases_meta_data_df = pd.DataFrame.from_dict(patient_data, orient = 'index')
    cases_meta_data_df = cases_meta_data_df.reset_index()
    cases_meta_data_df.columns = ['ProxID', 'DCMSerDescr', 'resampled_nifti', 'sequence_type']
    return cases_meta_data_df


# In[83]:


def join_data(is_training_data, sequence_df_array):
    """
    This function combines information provided by the research team in ProstateX-Images
    and ProstateX-Images-KTrans (Train/Test) files with paths to the resampled NIFTI files. 
    The function accepts a boolean is_training_data to determine if it is training or test
    data that needs to be processed. A list containing data frames of the joined data
    is the second parameter. The function concatenates the data frames in this list and
    returns a final data frame of all the data.
    """

    if is_training_data:
        prostateX_images = pd.read_csv('C:/Sapna/Graham/Capstone/data/train/lesion_info/ProstateX-2-Images-Train.csv')
        prostateX_images_ktrans = pd.read_csv('C:/Sapna/Graham/Capstone/data/train/lesion_info/ProstateX-2-Images-KTrans-Train-V1.csv')
        prostateX_findings = pd.read_csv('C:/Sapna/Graham/Capstone/data/train/lesion_info/ProstateX-2-Findings-Train.csv')
    else:
        prostateX_images = pd.read_csv('C:/Sapna/Graham/Capstone/data/test/lesion_info/ProstateX-2-Images-Test.csv')
        prostateX_images_ktrans = pd.read_csv('C:/Sapna/Graham/Capstone/data/test/lesion_info/ProstateX-Images-KTrans-Test.csv')
        prostateX_findings = pd.read_csv('C:/Sapna/Graham/Capstone/data/test/lesion_info/ProstateX-2-Findings-Test.csv')
  
    df_collection = []
    
    # Merging info for the DICOM series
    for dataframe in sequence_df_array[0:10]:
        # Convert DCMSerDescr values to lowercase in both frames (sanitize)
        dataframe.loc[:,'DCMSerDescr'] = dataframe.loc[:,'DCMSerDescr'].apply(lambda x: x.lower())
        prostateX_images.loc[:,'DCMSerDescr'] = prostateX_images.loc[:,'DCMSerDescr'].apply(lambda x: x.lower())
        
        # Keep only important columns from researcher provided data
        prostateX_images = prostateX_images[['ProxID', 'DCMSerDescr', 'fid', 'pos','WorldMatrix', 'ijk']]

        # Merge NIFTI paths with researcher provided data
        first_merge = pd.merge(dataframe, prostateX_images, how = 'inner', on = ['ProxID', 'DCMSerDescr'])
        
        # Merge findings (cancer/not cancer)
        final_merge = pd.merge(first_merge, prostateX_findings, how = 'inner', on = ['ProxID', 'fid', 'pos'])
        df_collection.append(final_merge)
    
   
    # Merging info for the KTRANS series
    first_merge = pd.merge(dataframe, prostateX_images_ktrans, how = 'inner', on = ['ProxID'])
    
    # Merge findings (Gleason scores)
    final_merge = pd.merge(first_merge, prostateX_findings, how = 'inner', on = ['ProxID', 'fid', 'pos'])
    df_collection.append(final_merge)

    final_dataframe = pd.concat(df_collection, ignore_index=True)

    return final_dataframe


# In[84]:


def repair_values(is_training_data, dataframe):
    """
    This function accepts a data frame and reformats entries in select columns
    to make them more acceptable for use in patch analysis (i.e. converting strings of 
    coordinate values to tuples of float).
    """

    def convert_to_tuple(dataframe, column):
        """
        This function converts row values (represented as string of floats
        delimited by spaces) to a tuple of floats. It accepts the original data
        frame and a string for the specified column that needs to be converted.
        """  
        pd_series_containing_lists_of_strings = dataframe[column].str.split() 
        list_for_new_series = []
        for list_of_strings in pd_series_containing_lists_of_strings:
            container_list = []
            for item in list_of_strings:
                if column == 'pos':
                    container_list.append(float(item))
                else:
                    container_list.append(int(item))
            list_for_new_series.append(tuple(container_list))
        
        return pd.Series(list_for_new_series)    

    # Call function to convert select columns
    dataframe = dataframe.assign(pos_tuple = convert_to_tuple(dataframe, 'pos'))
    dataframe = dataframe.assign(ijk_tuple = convert_to_tuple(dataframe, 'ijk'))
    
    # Drop old columns, rename new ones, and reorder...
    dataframe = dataframe.drop(columns = ['pos','ijk', 'WorldMatrix'])
    dataframe = dataframe.rename(columns = {'pos_tuple':'pos', 'ijk_tuple':'ijk'})

    if is_training_data:
        repaired_df = dataframe.loc[:,['ProxID', 'DCMSerDescr', 'resampled_nifti', 'sequence_type', 'fid', 'pos', 'ijk', 'zone', 'ggg']]
    else:
        repaired_df = dataframe.loc[:,['ProxID', 'DCMSerDescr', 'resampled_nifti', 'sequence_type', 'fid', 'pos', 'ijk', 'zone']]
    
    return repaired_df


# In[85]:


def save_data_to_directory(is_training_data, dataframe):
    if is_training_data:
        dataframe.to_csv('C:/Sapna/Graham/Capstone/data/train/generated/dataframes/training_meta_data.csv')
        dataframe.to_pickle('C:/Sapna/Graham/Capstone/data/train/generated/dataframes/training_meta_data.pkl')
    else:
        dataframe.to_csv('C:/Sapna/Graham/Capstone/data/test/generated/dataframes/test_meta_data.csv')
        dataframe.to_pickle('C:/Sapna/Graham/Capstone/data/test/generated/dataframes/test_meta_data.pkl')


# In[86]:


def main():
    is_training_data = False
    dataset_type = input('Which dataset are you working with? (1-Train; 2-Test):')
    if dataset_type == str(1):
        is_training_data = True
    
    t2_tse_tra_meta = generate_cases_meta_df(is_training_data, 'tse_tra')
    t2_tse_sag_meta = generate_cases_meta_df(is_training_data, 'tse_sag')
    adc_meta = generate_cases_meta_df(is_training_data, 'adc')
    bval_meta = generate_cases_meta_df(is_training_data, 'bval')
    ktrans_meta = generate_cases_meta_df(is_training_data, 'ktrans')

    sequence_df_array = [t2_tse_tra_meta, t2_tse_sag_meta , adc_meta, bval_meta, ktrans_meta]
    
    complete_df = join_data(is_training_data, sequence_df_array)
    final_df = repair_values(is_training_data, complete_df)
    
    final_dataframe_deduplicated = final_df.drop_duplicates(subset=['ProxID','sequence_type', 'pos'], keep = 'first')
    save_data_to_directory(is_training_data, final_dataframe_deduplicated)
main()

