# Capstone Project - Graham School of Continuing Liberal and Professional Studies (Uchicago) Partnered with The University of Chicago Medicine School


## Determining Aggressiveness of Prostate Cancer Based On Multiparametric MRI (mpMRI)


### Background

40% of one in nine men suffering from Prostate Cancer (PCa) are indolent – cancer which may exist for a long period without causing any symptoms or death. 
However, a large percentage of these patients undergo unnecessary treatment leading to significant complications. Therefore, this project addresses the crucial need for 
developing non-invasive tools for assessing PCa aggressiveness to determine the optimal treatment allowing doctors to reduce unnecessary biopsies and 
aggressive treatments for indolent cancer cases. The model uses Deep learning methods of Image Recognition and Computer Vision technique of 
deep Convolutional Neural Networks using the multiple Magnetic Resonance Imaging (MRI) of each patient. With most of the cancer lesions present in the 
peripheral parts of the prostate for patients as seen in the data, the modelling approach is computationally more intensive and robust to various types of cancer lesions and 
Gleason scores which further improves the diagnostic accuracy of determining the clinical aggressiveness of prostate cancer. 


### Problem Statement

Prostate cancer that's detected early, when it's still confined to the prostate gland, has a better chance of successful treatment. 
An elevated prostate-specific antigen (PSA) test and/or abnormality of the prostate identified during a digital rectal exam (DRE) typically are the 
first indication of the possibility of prostate cancer. Neither of these tests, however, provides a definitive diagnosis of the disease. 
Approximately 40-50% of the Prostate Cancer patients are given more severe and radical treatment due to the inability to determine the aggressiveness of Cancer in the Prostate.

Recently mpMRI is increasingly being used for PCa diagnosis. Clinical trials such as PROMIS (Prostate MR Imaging Study) [1] and PRECISION (Prostate Evaluation for Clinically Important Disease: Sampling Using Image Guidance) [2] further support the concept of targeted biopsy with mpMRI, and the results have led to increased acceptance of mpMRI for PCa screening. However, 15-30% of clinically significant cancers are missed even by expert radiologists using conventional Magnetic Resonance Imaging (MRI) due to the subjective nature of prostate MRI interpretation. Below are the collateral issues faced by the two main agencies in this case: The Patient and the Radiologist

(1)	Each miss treated patient has to undergo continuous biopsies and a complete  
            prostate removal procedure which are enormously painful and cause a huge 
            economic burden on patients’ budget.
(2)	The Radiologists on the other hand face with a lower efficiency in performance 
            with such a high miss treatment rate, caused due to subjectivity of translation of 
            MRI reports.

Therefore, the performance of MR imaging still needs to improve before it can be used for the screening of large populations. 
Both these problems can be mitigated with the help of  CAD tools with a combination of newer data analytics techniques such as AI and specifically deep learning methods  
which would  help to  improve  the  diagnostic  accuracy  of  detecting  prostate  cancer and its aggressiveness based on the science and math behind the MRI data generation. 
Deep neural networks are now the state-of-the-art machine learning models across a variety of areas, from image analysis to natural language processing, and widely deployed 
in academia and industry. These developments have a huge potential for medical imaging technology, medical data analysis, medical diagnostics and healthcare in general, 
slowly being realized.

### Data Source: 

PROSTATEx-2 Challenge Data - https://wiki.cancerimagingarchive.net/display/Public/SPIE-AAPM-NCI+PROSTATEx+Challenges#23691656d4622c5ad5884bdb876d6d441994da38

### Variables and Scope:

The scope of this project is to determine the Gleason score (measure of aggressiveness of PCa) from MRI scan images of patients by using computer vision models.
The input to the model will be MRI images and the response variable is the Gleason score.
Gleason scores are provided for the respective region of interest on a pathology-defined scale:
a.	Grade Group 1 (Gleason score < 6): Only individual with discrete well-formed glands
b.	Grade Group 2 Grade Group 2 (Gleason score 3+4 = 7): Predominantly well-formed glands with lesser component of poorly-formed/fused/cribriform glands
c.	Grade Group 3 (Gleason score 4+3 = 7): Predominantly poorly formed/fused/cribriform glands with lesser component of well-formed glands
d.	Grade Group 4 (Gleason score 4+4=8; 3+3=8; 5+3=8): 1) Only poorly-formed/fused/cribriform glands or (2) predominantly well-formed glands and lesser component 
lacking glands or (3) predominantly lacking glands and lesser component of well-formed glands
e.	Grade Group 5 (Gleason scores 9-10): Lacks gland formation (or with necrosis) with or without poorly formed/fused/cribriform glands

The database for this challenge contains a total of 112 MRI cases, each from a single examination from a distinct patient, with each case consisting of five sets of 
MRI scan data: two sets of T2-weighted images (transaxial and sagittal; DICOM format), Ktrans images (computed from dynamic contrast-enhanced (DCE) images; mhd format),  
apparent diffusion coefficient (ADC) images and Bvalue (b-value) images (both computed from diffusion-weighted (DWI) imaging; DICOM format). 
These cases contain a total of 182 findings (lesions): the training set contains 112 findings, and the test set contains 70 findings. L
ocation and a reference thumbnail image is provided for each lesion, and each lesion will have known pathology-defined Gleason Grade Group. 

### Requirements:
Slicer Software - To convert the DICOM images and MHD format images to NRRD format
Python
SimpleITK
nibabel
ipywidgets


### Steps:
1. Download the data from the above mentioned website - PROSTATEx-2 Challenge Data
2. Download the Preprocessing code folder change the path in each of the codes and run it on Python except for the first code (00_dicom_to_nrrd.py) which you need torun on the Slicer Python Interface to convert the DICOM to NRRD images.


### Team Members:
This was a group effort involving Radhika Singh Ghelot & Divya Ravindran
