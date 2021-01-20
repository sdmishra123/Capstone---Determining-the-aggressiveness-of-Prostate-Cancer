#!/usr/bin/env python
# coding: utf-8

# # Import Required Libraries

# In[1]:


import tensorflow.keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, AveragePooling2D, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Concatenate, SpatialDropout2D
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.initializers import glorot_uniform
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import collections
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd


# In[2]:


# t2_tra_samples = np.load('C:/Sapna/Graham/Capstone/data/train/generated/numpy/tse_tra/X_train.npy')
# t2_tra_labels  = np.load('C:/Sapna/Graham/Capstone/data/train/generated/numpy/tse_tra/Y_train.npy')

t2_tra_samples = np.load('/home/sdmishra/capstone/data/X_train.npy')
t2_tra_labels  = np.load('/home/sdmishra/capstone/data/Y_train.npy')


# In[3]:


target = []
for i in t2_tra_labels:
    if i == 1:
        target.append("0")
    elif i == 2:
        target.append("1")
    elif i == 3:
        target.append("2")
    elif i == 4:
        target.append("3")
    else:
        target.append("4")

print("The frequency count of the GS:")
collections.Counter(target)


# In[4]:


# labels = to_categorical(target,num_classes=5)
labels = np.array(pd.get_dummies(target))


# In[5]:


t2_tra_samples_v1 = np.expand_dims(t2_tra_samples, axis=3)
t2_tra_samples_v1.shape


# # Split the train and test data

# In[6]:


X_train_orig,X_test_orig,Y_train_orig,Y_test_orig = train_test_split(t2_tra_samples_v1,labels,test_size=0.20, 
                                                                     stratify=labels,  
                                                                     random_state=42)


# In[7]:


categorical_train_labels = pd.DataFrame(Y_train_orig).idxmax(axis=1)
collections.Counter(categorical_train_labels)


# In[8]:


categorical_test_labels = pd.DataFrame(Y_test_orig).idxmax(axis=1)
collections.Counter(categorical_test_labels)


# In[9]:


x_train,x_val,y_train,y_val = train_test_split(X_train_orig,Y_train_orig,test_size=0.20, 
                                                                     stratify=Y_train_orig,  
                                                                     random_state=42)


# In[11]:


categorical_train_labels = pd.DataFrame(y_train).idxmax(axis=1)
collections.Counter(categorical_train_labels)


# In[13]:


categorical_val_labels = pd.DataFrame(y_val).idxmax(axis=1)
collections.Counter(categorical_val_labels)



def identity_block(X, f, filters, stage, block):
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
 
    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    
    return X



def convolutional_block(X, f, filters, stage, block, s = 2):
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)


    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)


    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1',
                        kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    
    return X



img_height, img_width, channel = X_train_orig.shape[1],X_train_orig.shape[2],X_train_orig.shape[3]


# In[18]:


def ResNet50(input_shape = (img_height,img_width,channel), classes = 5):
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL
    X = AveragePooling2D(pool_size=(2,2), padding='same')(X)

    # Output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model


# In[19]:


model = ResNet50(input_shape = (img_height,img_width,channel), classes = 5)


# In[20]:


model.summary()



# determine Loss function and Optimizer
model.compile(loss='categorical_crossentropy',
              optimizer="adagrad",
              metrics=['accuracy'])


# In[22]:


checkpoint_vl = ModelCheckpoint("weights.best_vl.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
checkpoint_va = ModelCheckpoint("weights.best_va.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint_va, checkpoint_vl]


# In[23]:


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)


# # Stratified Data Augmentation 

# In[24]:


categorical_train_labels = pd.DataFrame(y_train).idxmax(axis=1)
print(collections.Counter(categorical_train_labels))

categorical_val_labels = pd.DataFrame(y_val).idxmax(axis=1)
print(collections.Counter(categorical_val_labels))


# In[25]:


index_labels_0 = categorical_train_labels[categorical_train_labels==0].index.tolist()
index_labels_1 = categorical_train_labels[categorical_train_labels==1].index.tolist()
index_labels_2 = categorical_train_labels[categorical_train_labels==2].index.tolist()
index_labels_3 = categorical_train_labels[categorical_train_labels==3].index.tolist()
index_labels_4 = categorical_train_labels[categorical_train_labels==4].index.tolist()

index_labels_v0 = categorical_val_labels[categorical_val_labels==0].index.tolist()
index_labels_v1 = categorical_val_labels[categorical_val_labels==1].index.tolist()
index_labels_v2 = categorical_val_labels[categorical_val_labels==2].index.tolist()
index_labels_v3 = categorical_val_labels[categorical_val_labels==3].index.tolist()
index_labels_v4 = categorical_val_labels[categorical_val_labels==4].index.tolist()


# In[26]:


x_train_0  = np.take(x_train, index_labels_0,axis=0)
y_train_0  = np.take(y_train, index_labels_0,axis=0)
x_train_v0 = np.take(x_val, index_labels_v0,axis=0)
y_train_v0 = np.take(y_val, index_labels_v0,axis=0)

categorical_train_labels = pd.DataFrame(y_train_0).idxmax(axis=1)
print(collections.Counter(categorical_train_labels))

categorical_val_labels = pd.DataFrame(y_train_v0).idxmax(axis=1)
print(collections.Counter(categorical_val_labels))


# In[27]:


x_train_1  = np.take(x_train, index_labels_1,axis=0)
y_train_1  = np.take(y_train, index_labels_1,axis=0)
x_train_v1 = np.take(x_val, index_labels_v1,axis=0)
y_train_v1 = np.take(y_val, index_labels_v1,axis=0)

categorical_train_labels = pd.DataFrame(y_train_1).idxmax(axis=1)
print(collections.Counter(categorical_train_labels))

categorical_val_labels = pd.DataFrame(y_train_v1).idxmax(axis=1)
print(collections.Counter(categorical_val_labels))


# In[28]:


x_train_2  = np.take(x_train, index_labels_2,axis=0)
y_train_2  = np.take(y_train, index_labels_2,axis=0)

x_train_v2 = np.take(x_val, index_labels_v2,axis=0)
y_train_v2 = np.take(y_val, index_labels_v2,axis=0)

categorical_train_labels = pd.DataFrame(y_train_2).idxmax(axis=1)
print(collections.Counter(categorical_train_labels))

categorical_val_labels = pd.DataFrame(y_train_v2).idxmax(axis=1)
print(collections.Counter(categorical_val_labels))


# In[29]:


x_train_3  = np.take(x_train, index_labels_3,axis=0)
y_train_3  = np.take(y_train, index_labels_3,axis=0)
x_train_v3 = np.take(x_val, index_labels_v3,axis=0)
y_train_v3 = np.take(y_val, index_labels_v3,axis=0)

categorical_train_labels = pd.DataFrame(y_train_3).idxmax(axis=1)
print(collections.Counter(categorical_train_labels))

categorical_val_labels = pd.DataFrame(y_train_v3).idxmax(axis=1)
print(collections.Counter(categorical_val_labels))


# In[30]:


x_train_4  = np.take(x_train, index_labels_4,axis=0)
y_train_4  = np.take(y_train, index_labels_4,axis=0)
x_train_v4 = np.take(x_val, index_labels_v4,axis=0)
y_train_v4 = np.take(y_val, index_labels_v4,axis=0)

categorical_train_labels = pd.DataFrame(y_train_4).idxmax(axis=1)
print(collections.Counter(categorical_train_labels))

categorical_val_labels = pd.DataFrame(y_train_v4).idxmax(axis=1)
print(collections.Counter(categorical_val_labels))


# ### Data Augmentation of Class : 0

# In[31]:


data_x_aug0 = x_train_0
data_y_aug0 = y_train_0

for i in range(0,10):
    datagen = ImageDataGenerator(
#         zoom_range=[0.5,1.0],
        rotation_range = 40,
        horizontal_flip = True,
#         vertical_flip = True,
        width_shift_range=2,
#         height_shift_range=2,
#         shear_range=15.0,
    )

    datagen.fit(data_x_aug0)

    for X_batch, y_batch in datagen.flow(x_train_0, y_train_0, batch_size=100):
        break

    data_x_aug0 = np.append(data_x_aug0,X_batch,axis=0)
    data_y_aug0 = np.append(data_y_aug0,y_batch,axis=0)


data_final_x0 = data_x_aug0
data_final_y0 = data_y_aug0

###########################################################################################################################

data_x_augv0 = x_train_v0
data_y_augv0 = y_train_v0

for i in range(0,10):
    datagen = ImageDataGenerator(
#          zoom_range=[0.5,1.0],
        rotation_range = 40,
        horizontal_flip = True,
#         vertical_flip = True,
        width_shift_range=2,
#         height_shift_range=2,
#         shear_range=15.0,
    )

    datagen.fit(data_x_augv0)

    for X_batch, y_batch in datagen.flow(x_train_v0, y_train_v0, batch_size=100):
        break

    data_x_augv0 = np.append(data_x_augv0,X_batch,axis=0)
    data_y_augv0 = np.append(data_y_augv0,y_batch,axis=0)


data_final_xv0 = data_x_augv0
data_final_yv0 = data_y_augv0


print("Train X1 size:",data_final_x0.shape)
print("Train Y1 size:",data_final_y0.shape)

print("Val X1 size:",data_final_xv0.shape)
print("Val Y1 size:",data_final_yv0.shape)


# ### Data Augmentation of Class : 1

# In[32]:


data_x_aug1 = x_train_1
data_y_aug1 = y_train_1

for i in range(0,10):
    datagen = ImageDataGenerator(
#         zoom_range=[0.5,1.0],
        rotation_range = 50,
        horizontal_flip = True,
        vertical_flip = True,
        width_shift_range=2,
#         height_shift_range=2,
#         shear_range=15.0
    )

    datagen.fit(data_x_aug1)

    for X_batch, y_batch in datagen.flow(x_train_1, y_train_1, batch_size=100):
        break

    data_x_aug1 = np.append(data_x_aug1,X_batch,axis=0)
    data_y_aug1 = np.append(data_y_aug1,y_batch,axis=0)


data_final_x1 = data_x_aug1
data_final_y1 = data_y_aug1

#########################################
data_x_augv1 = x_train_v1
data_y_augv1 = y_train_v1

for i in range(0,10):
    datagen = ImageDataGenerator(
        zoom_range=[0.5,1.0],
        rotation_range = 50,
        horizontal_flip = True,
        vertical_flip = True,
        width_shift_range=2,
#         height_shift_range=2,
#         shear_range=15.0
    )

    datagen.fit(data_x_augv1)

    for X_batch, y_batch in datagen.flow(x_train_v1, y_train_v1, batch_size=100):
        break

    data_x_augv1 = np.append(data_x_augv1,X_batch,axis=0)
    data_y_augv1 = np.append(data_y_augv1,y_batch,axis=0)


data_final_xv1 = data_x_augv1
data_final_yv1 = data_y_augv1

print("Train X1 size:",data_final_x1.shape)
print("Train Y1 size:",data_final_y1.shape)

print("Val X1 size:",data_final_xv1.shape)
print("Val Y1 size:",data_final_yv1.shape)


# ### Data Augmentation of Class : 2

# In[33]:


data_x_aug2 = x_train_2
data_y_aug2 = y_train_2

for i in range(0,10):
    datagen = ImageDataGenerator(
#         zoom_range=[0.5,1.0],
        rotation_range = 70,
        horizontal_flip = True,
        vertical_flip = True,
        width_shift_range=2,
#         height_shift_range=2,
#         shear_range=15.0,
    )

    datagen.fit(data_x_aug2)

    for X_batch, y_batch in datagen.flow(x_train_2, y_train_2, batch_size=100):
        break

    data_x_aug2 = np.append(data_x_aug2,X_batch,axis=0)
    data_y_aug2 = np.append(data_y_aug2,y_batch,axis=0)


data_final_x2 = data_x_aug2
data_final_y2 = data_y_aug2


data_x_augv2 = x_train_v2
data_y_augv2 = y_train_v2

for i in range(0,10):
    datagen = ImageDataGenerator(
        zoom_range=[0.5,1.0],
        rotation_range = 70,
        horizontal_flip = True,
        vertical_flip = True,
        width_shift_range=2,
#         height_shift_range=2,
#         shear_range=15.0,
    )

    datagen.fit(data_x_augv2)

    for X_batch, y_batch in datagen.flow(x_train_v2, y_train_v2, batch_size=100):
        break

    data_x_augv2 = np.append(data_x_augv2,X_batch,axis=0)
    data_y_augv2 = np.append(data_y_augv2,y_batch,axis=0)


data_final_xv2 = data_x_augv2
data_final_yv2 = data_y_augv2


print("Train X1 size:",data_final_x2.shape)
print("Train Y1 size:",data_final_y2.shape)


print("Val X1 size:",data_final_xv2.shape)
print("Val Y1 size:",data_final_yv2.shape)


# ### Data Augmentation of Class : 3

# In[34]:


data_x_aug3 = x_train_3
data_y_aug3 = y_train_3

for i in range(0,10):
    datagen = ImageDataGenerator(
#         zoom_range=[0.5,1.0],
        rotation_range = 80,
        horizontal_flip = True,
        vertical_flip = True,
        width_shift_range=2,
#         height_shift_range=2,
#         shear_range=15.0
    )

    datagen.fit(data_x_aug3)

    for X_batch, y_batch in datagen.flow(x_train_3, y_train_3, batch_size=500):
        break

    data_x_aug3 = np.append(data_x_aug3,X_batch,axis=0)
    data_y_aug3 = np.append(data_y_aug3,y_batch,axis=0)


data_final_x3 = data_x_aug3
data_final_y3 = data_y_aug3

data_x_augv3 = x_train_v3
data_y_augv3 = y_train_v3

for i in range(0,10):
    datagen = ImageDataGenerator(
        zoom_range=[0.5,1.0],
        rotation_range = 80,
        horizontal_flip = True,
        vertical_flip = True,
        width_shift_range=2,
#         height_shift_range=2,
#         shear_range=15.0,
    )

    datagen.fit(data_x_augv3)

    for X_batch, y_batch in datagen.flow(x_train_v3, y_train_v3, batch_size=100):
        break

    data_x_augv3 = np.append(data_x_augv3,X_batch,axis=0)
    data_y_augv3 = np.append(data_y_augv3,y_batch,axis=0)


data_final_xv3 = data_x_augv3
data_final_yv3 = data_y_augv3

print("Train X1 size:",data_final_x3.shape)
print("Train Y1 size:",data_final_y3.shape)

print("Val X1 size:",data_final_xv3.shape)
print("Val Y1 size:",data_final_yv3.shape)


# ### Data Augmentation of Class : 4

# In[35]:


data_x_aug4 = x_train_4
data_y_aug4 = y_train_4

for i in range(0,10):
    datagen = ImageDataGenerator(
#         zoom_range=[0.5,1.0],
        rotation_range = 75,
        horizontal_flip = True,
        vertical_flip = True,
#         width_shift_range=2,
#         height_shift_range=2,
#         shear_range=15.0
    )

    datagen.fit(data_x_aug4)

    for X_batch, y_batch in datagen.flow(x_train_4, y_train_4, batch_size=500):
        break

    data_x_aug4 = np.append(data_x_aug4,X_batch,axis=0)
    data_y_aug4 = np.append(data_y_aug4,y_batch,axis=0)


data_final_x4 = data_x_aug4
data_final_y4 = data_y_aug4

data_x_augv4 = x_train_v4
data_y_augv4 = y_train_v4

for i in range(0,10):
    datagen = ImageDataGenerator(
        zoom_range=[0.5,1.0],
        rotation_range = 75,
        horizontal_flip = True,
        vertical_flip = True,
#         width_shift_range=2,
#         height_shift_range=2,
#         shear_range=15.0
    )

    datagen.fit(data_x_augv4)

    for X_batch, y_batch in datagen.flow(x_train_v4, y_train_v4, batch_size=100):
        break

    data_x_augv4 = np.append(data_x_augv4,X_batch,axis=0)
    data_y_augv4 = np.append(data_y_augv4,y_batch,axis=0)


data_final_xv4 = data_x_augv4
data_final_yv4 = data_y_augv4

print("Train X1 size:",data_final_x4.shape)
print("Train Y1 size:",data_final_y4.shape)

print("Val X1 size:",data_final_xv4.shape)
print("Val Y1 size:",data_final_yv4.shape)


# In[36]:


x_train_final = np.append(data_final_x0,data_final_x1,axis=0)
y_train_final = np.append(data_final_y0,data_final_y1,axis=0)
print(x_train_final.shape)
print(y_train_final.shape)
categorical_train_labels = pd.DataFrame(y_train_final).idxmax(axis=1)
print(collections.Counter(categorical_train_labels))

print("*********************************************************************************************")
x_train_final = np.append(x_train_final,data_final_x2,axis=0)
y_train_final = np.append(y_train_final,data_final_y2,axis=0)
print(x_train_final.shape)
print(y_train_final.shape)
categorical_train_labels = pd.DataFrame(y_train_final).idxmax(axis=1)
print(collections.Counter(categorical_train_labels))

print("*********************************************************************************************")
x_train_final = np.append(x_train_final,data_final_x3,axis=0)
y_train_final = np.append(y_train_final,data_final_y3,axis=0)
print(x_train_final.shape)
print(y_train_final.shape)
categorical_train_labels = pd.DataFrame(y_train_final).idxmax(axis=1)
print(collections.Counter(categorical_train_labels))

print("*********************************************************************************************")
x_train_final = np.append(x_train_final,data_final_x4,axis=0)
y_train_final = np.append(y_train_final,data_final_y4,axis=0)
print(x_train_final.shape)
print(y_train_final.shape)
categorical_train_labels = pd.DataFrame(y_train_final).idxmax(axis=1)
print(collections.Counter(categorical_train_labels))

print("*********************************************************************************************")
print("*********************************************************************************************")
print("*********************************************************************************************")

x_train_vfinal = np.append(data_final_xv0,data_final_xv1,axis=0)
y_train_vfinal = np.append(data_final_yv0,data_final_yv1,axis=0)
print(x_train_vfinal.shape)
print(y_train_vfinal.shape)
categorical_val_labels = pd.DataFrame(y_train_vfinal).idxmax(axis=1)
print(collections.Counter(categorical_val_labels))

print("*********************************************************************************************")
x_train_vfinal = np.append(x_train_vfinal,data_final_xv2,axis=0)
y_train_vfinal = np.append(y_train_vfinal,data_final_yv2,axis=0)
print(x_train_vfinal.shape)
print(y_train_vfinal.shape)
categorical_val_labels = pd.DataFrame(y_train_vfinal).idxmax(axis=1)
print(collections.Counter(categorical_val_labels))

print("*********************************************************************************************")
x_train_vfinal = np.append(x_train_vfinal,data_final_xv3,axis=0)
y_train_vfinal = np.append(y_train_vfinal,data_final_yv3,axis=0)
print(x_train_vfinal.shape)
print(y_train_vfinal.shape)
categorical_val_labels = pd.DataFrame(y_train_vfinal).idxmax(axis=1)
print(collections.Counter(categorical_val_labels))

print("*********************************************************************************************")
x_train_vfinal = np.append(x_train_vfinal,data_final_xv4,axis=0)
y_train_vfinal = np.append(y_train_vfinal,data_final_yv4,axis=0)
print(x_train_vfinal.shape)
print(y_train_vfinal.shape)
categorical_val_labels = pd.DataFrame(y_train_vfinal).idxmax(axis=1)
print(collections.Counter(categorical_val_labels))


# In[37]:


categorical_train_labels = pd.DataFrame(y_train_final).idxmax(axis=1)
collections.Counter(categorical_train_labels)


# # Normalize image vectors

# In[41]:


X_train = x_train_final/255.
X_val   = x_train_vfinal/255.
X_test  = X_test_orig/255.

Y_train = y_train_final
Y_val   = y_train_vfinal
Y_test  = Y_test_orig


# In[42]:


print(X_train.shape)
print(X_val.shape)
print(X_test.shape)


# # Model Execution

# In[ ]:


history = model.fit(
                    X_train, Y_train, 
                    epochs = 150, 
                    batch_size = 64,
                    validation_data = (X_val,Y_val),
                    callbacks=[es],
                    shuffle=True
                   )


# In[ ]:


preds = model.evaluate(X_test_orig, Y_test_orig)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


# In[ ]:


model.save_weights('resnet.h5')

