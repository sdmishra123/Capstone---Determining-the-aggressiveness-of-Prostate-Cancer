#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow.keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, AveragePooling2D, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Concatenate, SpatialDropout2D
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import initializers
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.utils import plot_model
import collections
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import backend as k
import numpy as np
import pandas as pd


# In[2]:


# t2_tra_samples = np.load('/home/sdmishra/capstone/data/x_train_t2.npy')
# t2_tra_labels  = np.load('/home/sdmishra/capstone/data/y_train_all.npy')
# adc_samples = np.load('/home/sdmishra/capstone/data/x_train_adc.npy')
# bval_samples = np.load('/home/sdmishra/capstone/data/x_train_bval.npy')


# In[3]:


t2_tra_samples = np.load('C:/Sapna/Graham/Capstone/data/train/generated/numpy/tse_tra/X_train_t2_3D.npy',allow_pickle=True)
t2_tra_labels  = np.load('C:/Sapna/Graham/Capstone/data/train/generated/numpy/tse_tra/Y_train_t2_3D.npy',allow_pickle=True)
adc_samples    = np.load('C:/Sapna/Graham/Capstone/data/train/generated/numpy/adc/X_train_adc_3D.npy',allow_pickle=True)
bval_samples   = np.load('C:/Sapna/Graham/Capstone/data/train/generated/numpy/bval/X_train_bval_3D.npy',allow_pickle=True)


# In[4]:


# t2_samples = np.expand_dims(t2_tra_samples, axis=3)
# print(t2_samples.shape)
# adc_samples = np.expand_dims(adc_samples, axis=3)
# print(adc_samples.shape)
# bval_samples = np.expand_dims(bval_samples, axis=3)
# print(bval_samples.shape)

print("#####################################################")
print(t2_tra_samples.shape)
print(adc_samples.shape)
print(bval_samples.shape)

t2_samples    = t2_tra_samples.reshape((112,60,60,3))
adc_samples   = adc_samples.reshape((112,16,16,3))
bval_samples  = bval_samples.reshape((112,16,16,3))

print("#####################################################")
print(t2_samples.shape)
print(adc_samples.shape)
print(bval_samples.shape)


# In[5]:


target_tra = []
for i in t2_tra_labels:
    if i == 1:
        target_tra.append("0")
    elif i == 2 or i == 3:
        target_tra.append("1")
    # elif i == 3:
    #     target_tra.append("2")
    # elif i == 4:
    #     target_tra.append("3")
    else:
        target_tra.append("2")


# In[6]:


t2_labels = np.array(pd.get_dummies(target_tra))


# In[7]:


t2_categorical = pd.DataFrame(t2_labels).idxmax(axis=1)
print(collections.Counter(t2_categorical))


# In[8]:


# data_X = np.stack(t2_samples, axis=0)


# In[9]:


x_train_t2,x_test_t2,y_train_t2,y_test_t2 = train_test_split(t2_samples,t2_labels,
                                                                     test_size=0.20, 
                                                                     stratify=t2_labels,  
                                                                     random_state=42)

x_train_t2,x_val_t2,y_train_t2,y_val_t2 = train_test_split(x_train_t2,y_train_t2,
                                                                     test_size=0.20, 
                                                                     stratify=y_train_t2,  
                                                                     random_state=42)

categorical_train_labels = pd.DataFrame(y_train_t2).idxmax(axis=1)
print(collections.Counter(categorical_train_labels))
categorical_val_labels = pd.DataFrame(y_val_t2).idxmax(axis=1)
print(collections.Counter(categorical_val_labels))
categorical_test_labels = pd.DataFrame(y_test_t2).idxmax(axis=1)
print(collections.Counter(categorical_test_labels))
print(x_train_t2.shape)
print(x_test_t2.shape)


# In[10]:


x_train_adc,x_test_adc,y_train_adc,y_test_adc = train_test_split(adc_samples,t2_labels,
                                                                     test_size=0.20, 
                                                                     stratify=t2_labels,  
                                                                     random_state=42)

x_train_adc,x_val_adc,y_train_adc,y_val_adc = train_test_split(x_train_adc,y_train_adc,
                                                                     test_size=0.20, 
                                                                     stratify=y_train_adc,  
                                                                     random_state=42)

categorical_train_labels = pd.DataFrame(y_train_adc).idxmax(axis=1)
print(collections.Counter(categorical_train_labels))
categorical_val_labels = pd.DataFrame(y_val_adc).idxmax(axis=1)
print(collections.Counter(categorical_val_labels))
categorical_test_labels = pd.DataFrame(y_test_adc).idxmax(axis=1)
print(collections.Counter(categorical_test_labels))
print(x_train_adc.shape)
print(x_test_adc.shape)


# In[11]:


x_train_bval,x_test_bval,y_train_bval,y_test_bval = train_test_split(bval_samples,t2_labels,
                                                                     test_size=0.20, 
                                                                     stratify=t2_labels,  
                                                                     random_state=42)

x_train_bval,x_val_bval,y_train_bval,y_val_bval = train_test_split(x_train_bval,y_train_bval,
                                                                     test_size=0.20, 
                                                                     stratify=y_train_bval,  
                                                                     random_state=42)

categorical_train_labels = pd.DataFrame(y_train_bval).idxmax(axis=1)
print(collections.Counter(categorical_train_labels))
categorical_val_labels = pd.DataFrame(y_val_bval).idxmax(axis=1)
print(collections.Counter(categorical_val_labels))
categorical_test_labels = pd.DataFrame(y_test_bval).idxmax(axis=1)
print(collections.Counter(categorical_test_labels))
print(x_train_bval.shape)
print(x_test_bval.shape)


# In[12]:


t2_img_height,   t2_img_width,   t2_channel    =  x_train_t2.shape[1],x_train_t2.shape[2],x_train_t2.shape[3]
adc_img_height,  adc_img_width,  adc_channel   =  x_train_adc.shape[1],x_train_adc.shape[2],x_train_adc.shape[3]
bval_img_height, bval_img_width, bval_channel  =  x_train_bval.shape[1],x_train_bval.shape[2],x_train_bval.shape[3]


# In[13]:


batch_size = 64
num_classes = 3
epochs = 150
l = 12
num_filter = 24
compression = 0.5
dropout_rate = 0.2
wt_decay = 0.001


# In[14]:


def add_denseblock(input, num_filter = 12, dropout_rate = 0.2):
    global compression
    temp = input
    for _ in range(l):
        BatchNorm = BatchNormalization()(temp)
        relu = Activation('relu')(BatchNorm)
        #Conv2D_3_3 = Conv2D(int(num_filter*compression), (3,3), use_bias=False ,padding='same', kernel_initializer=(random_normal(stddev=np.sqrt(2.0/(9*int(num_filter*compression))))))(relu)
        Conv2D_3_3 = Conv2D(int(num_filter*compression), (3,3), use_bias=False ,padding='same',kernel_regularizer=l2(wt_decay) , kernel_initializer=(initializers.RandomNormal(stddev=np.sqrt(2.0/(9*int(num_filter))))))(relu)
        # Conv2D_3_3 = Conv2D(int(num_filter*compression), (3,3), use_bias=False ,padding='same',kernel_regularizer=l2(wt_decay) , kernel_initializer=glorot_uniform(seed=42) )(relu)
        if dropout_rate>0:
            Conv2D_3_3 = SpatialDropout2D(dropout_rate)(Conv2D_3_3)
        concat = Concatenate(axis=-1)([temp,Conv2D_3_3])
        
        temp = concat
        
    return temp

def add_transition(input, num_filter = 12, dropout_rate = 0.2):
    global compression
    BatchNorm = BatchNormalization()(input)
    relu = Activation('relu')(BatchNorm)
    Conv2D_BottleNeck = Conv2D(int(int(input.shape[-1])*compression), (1,1), use_bias=False ,padding='same',kernel_regularizer=l2(wt_decay) , kernel_initializer=(initializers.RandomNormal(stddev=np.sqrt(2.0/(9*int(num_filter*compression))))))(relu)
    # Conv2D_BottleNeck = Conv2D(int(int(input.shape[-1])*compression), (1,1), use_bias=False ,padding='same',kernel_regularizer=l2(wt_decay) , kernel_initializer=glorot_uniform(seed=42))(relu)
    # Conv2D_BottleNeck = Conv2D(int(num_filter*compression), (1,1), use_bias=False ,padding='same',kernel_initializer=(random_normal(stddev=np.sqrt(2.0/(9*int(num_filter*compression))))))(relu)
    if dropout_rate>0:
        Conv2D_BottleNeck = SpatialDropout2D(dropout_rate)(Conv2D_BottleNeck)
    avg = AveragePooling2D(pool_size=(2,2))(Conv2D_BottleNeck)
    
    return avg

def output_layer(input):
    global compression
    BatchNorm = BatchNormalization()(input)
    relu = Activation('relu')(BatchNorm)
    AvgPooling = AveragePooling2D(pool_size=(2,2))(relu)
    flat = Flatten()(AvgPooling)
    # output = Dense(num_classes, activation='softmax', kernel_regularizer=l2(wt_decay))(flat)
    output = Dense(9, activation='relu', kernel_initializer='he_normal')(flat)
    output = flat
    
    return output


input = Input(shape=(t2_img_height, t2_img_width, t2_channel))
# First_Conv2D = Conv2D(int(num_filter), (3,3), use_bias=False , padding='same', kernel_regularizer=l2(wt_decay), kernel_initializer=glorot_uniform(seed=42))(input)
First_Conv2D = Conv2D(int(num_filter), (3,3), use_bias=False , padding='same', kernel_regularizer=l2(wt_decay), kernel_initializer=(initializers.RandomNormal(stddev=np.sqrt(2.0/(9*int(num_filter*compression))))))(input)

First_Block = add_denseblock(First_Conv2D, num_filter, dropout_rate)
First_Transition = add_transition(First_Block, num_filter, dropout_rate)

#First_Transition = merge([First_Transition,First_Conv2D], mode='concat', concat_axis=-1)

Second_Block = add_denseblock(First_Transition, num_filter, dropout_rate)
Second_Transition = add_transition(Second_Block, num_filter, dropout_rate)

#Second_Transition = Concatenate(axis=-1)([Second_Transition,First_Transition,First_Conv2D])

Third_Block = add_denseblock(Second_Transition, num_filter, dropout_rate)
Third_Transition = add_transition(Third_Block, num_filter, dropout_rate)

#Third_Transition = Concatenate(axis=-1)([Third_Transition,Second_Transition,First_Transition,First_Conv2D])

Last_Block = add_denseblock(Third_Transition,  num_filter, dropout_rate)
output = output_layer(Last_Block)


# In[25]:


t2_model = Model(inputs=[input], outputs=[output])
# t2_model.summary()
# print(t2_model.output_shape)


# In[15]:


def add_denseblock(input, num_filter = 12, dropout_rate = 0.2):
    global compression
    temp = input
    for _ in range(l):
        BatchNorm = BatchNormalization()(temp)
        relu = Activation('relu')(BatchNorm)
        #Conv2D_3_3 = Conv2D(int(num_filter*compression), (3,3), use_bias=False ,padding='same', kernel_initializer=(random_normal(stddev=np.sqrt(2.0/(9*int(num_filter*compression))))))(relu)
        Conv2D_3_3 = Conv2D(int(num_filter*compression), (3,3), use_bias=False ,padding='same',kernel_regularizer=l2(wt_decay) , kernel_initializer=(initializers.RandomNormal(stddev=np.sqrt(2.0/(9*int(num_filter))))))(relu)
        # Conv2D_3_3 = Conv2D(int(num_filter*compression), (3,3), use_bias=False ,padding='same',kernel_regularizer=l2(wt_decay) , kernel_initializer=glorot_uniform(seed=42) )(relu)
        if dropout_rate>0:
            Conv2D_3_3 = SpatialDropout2D(dropout_rate)(Conv2D_3_3)
        concat = Concatenate(axis=-1)([temp,Conv2D_3_3])
        
        temp = concat
        
    return temp

def add_transition(input, num_filter = 12, dropout_rate = 0.2):
    global compression
    BatchNorm = BatchNormalization()(input)
    relu = Activation('relu')(BatchNorm)
    Conv2D_BottleNeck = Conv2D(int(int(input.shape[-1])*compression), (1,1), use_bias=False ,padding='same',kernel_regularizer=l2(wt_decay) , kernel_initializer=(initializers.RandomNormal(stddev=np.sqrt(2.0/(9*int(num_filter*compression))))))(relu)
    # Conv2D_BottleNeck = Conv2D(int(int(input.shape[-1])*compression), (1,1), use_bias=False ,padding='same',kernel_regularizer=l2(wt_decay) , kernel_initializer=glorot_uniform(seed=42))(relu)
    # Conv2D_BottleNeck = Conv2D(int(num_filter*compression), (1,1), use_bias=False ,padding='same',kernel_initializer=(random_normal(stddev=np.sqrt(2.0/(9*int(num_filter*compression))))))(relu)
    if dropout_rate>0:
        Conv2D_BottleNeck = SpatialDropout2D(dropout_rate)(Conv2D_BottleNeck)
    avg = AveragePooling2D(pool_size=(2,2))(Conv2D_BottleNeck)
    
    return avg

def output_layer(input):
    global compression
    BatchNorm = BatchNormalization()(input)
    relu = Activation('relu')(BatchNorm)
    AvgPooling = AveragePooling2D(pool_size=(2,2))(relu)
    flat = Flatten()(AvgPooling)
    # output = Dense(num_classes, activation='softmax', kernel_regularizer=l2(wt_decay))(flat)
#     output = Dense(9, activation='relu', kernel_initializer='he_normal')(flat)
    output = flat
    
    return output


input = Input(shape=(adc_img_height, adc_img_width, adc_channel))
# First_Conv2D = Conv2D(int(num_filter), (3,3), use_bias=False , padding='same', kernel_regularizer=l2(wt_decay), kernel_initializer=glorot_uniform(seed=42))(input)
First_Conv2D = Conv2D(int(num_filter), (3,3), use_bias=False , padding='same', kernel_regularizer=l2(wt_decay), kernel_initializer=(initializers.RandomNormal(stddev=np.sqrt(2.0/(9*int(num_filter*compression))))))(input)

First_Block = add_denseblock(First_Conv2D, num_filter, dropout_rate)
First_Transition = add_transition(First_Block, num_filter, dropout_rate)

#First_Transition = merge([First_Transition,First_Conv2D], mode='concat', concat_axis=-1)

Second_Block = add_denseblock(First_Transition, num_filter, dropout_rate)
Second_Transition = add_transition(Second_Block, num_filter, dropout_rate)

#Second_Transition = Concatenate(axis=-1)([Second_Transition,First_Transition,First_Conv2D])

Third_Block = add_denseblock(Second_Transition, num_filter, dropout_rate)
Third_Transition = add_transition(Third_Block, num_filter, dropout_rate)

#Third_Transition = Concatenate(axis=-1)([Third_Transition,Second_Transition,First_Transition,First_Conv2D])

Last_Block = add_denseblock(Third_Transition,  num_filter, dropout_rate)
output = output_layer(Last_Block)


# In[25]:


adc_model = Model(inputs=[input], outputs=[output])
# adc_model.summary()
# print(adc_model.output_shape)


# In[16]:


def add_denseblock(input, num_filter = 12, dropout_rate = 0.2):
    global compression
    temp = input
    for _ in range(l):
        BatchNorm = BatchNormalization()(temp)
        relu = Activation('relu')(BatchNorm)
        #Conv2D_3_3 = Conv2D(int(num_filter*compression), (3,3), use_bias=False ,padding='same', kernel_initializer=(random_normal(stddev=np.sqrt(2.0/(9*int(num_filter*compression))))))(relu)
        Conv2D_3_3 = Conv2D(int(num_filter*compression), (3,3), use_bias=False ,padding='same',kernel_regularizer=l2(wt_decay) , kernel_initializer=(initializers.RandomNormal(stddev=np.sqrt(2.0/(9*int(num_filter))))))(relu)
        # Conv2D_3_3 = Conv2D(int(num_filter*compression), (3,3), use_bias=False ,padding='same',kernel_regularizer=l2(wt_decay) , kernel_initializer=glorot_uniform(seed=42) )(relu)
        if dropout_rate>0:
            Conv2D_3_3 = SpatialDropout2D(dropout_rate)(Conv2D_3_3)
        concat = Concatenate(axis=-1)([temp,Conv2D_3_3])
        
        temp = concat
        
    return temp

def add_transition(input, num_filter = 12, dropout_rate = 0.2):
    global compression
    BatchNorm = BatchNormalization()(input)
    relu = Activation('relu')(BatchNorm)
    Conv2D_BottleNeck = Conv2D(int(int(input.shape[-1])*compression), (1,1), use_bias=False ,padding='same',kernel_regularizer=l2(wt_decay) , kernel_initializer=(initializers.RandomNormal(stddev=np.sqrt(2.0/(9*int(num_filter*compression))))))(relu)
    # Conv2D_BottleNeck = Conv2D(int(int(input.shape[-1])*compression), (1,1), use_bias=False ,padding='same',kernel_regularizer=l2(wt_decay) , kernel_initializer=glorot_uniform(seed=42))(relu)
    # Conv2D_BottleNeck = Conv2D(int(num_filter*compression), (1,1), use_bias=False ,padding='same',kernel_initializer=(random_normal(stddev=np.sqrt(2.0/(9*int(num_filter*compression))))))(relu)
    if dropout_rate>0:
        Conv2D_BottleNeck = SpatialDropout2D(dropout_rate)(Conv2D_BottleNeck)
    avg = AveragePooling2D(pool_size=(2,2))(Conv2D_BottleNeck)
    
    return avg

def output_layer(input):
    global compression
    BatchNorm = BatchNormalization()(input)
    relu = Activation('relu')(BatchNorm)
    AvgPooling = AveragePooling2D(pool_size=(2,2))(relu)
    flat = Flatten()(AvgPooling)
    # output = Dense(num_classes, activation='softmax', kernel_regularizer=l2(wt_decay))(flat)
#     output = Dense(9, activation='relu', kernel_initializer='he_normal')(flat)
    output = flat
    
    return output


input = Input(shape=(bval_img_height, bval_img_width, bval_channel))
# First_Conv2D = Conv2D(int(num_filter), (3,3), use_bias=False , padding='same', kernel_regularizer=l2(wt_decay), kernel_initializer=glorot_uniform(seed=42))(input)
First_Conv2D = Conv2D(int(num_filter), (3,3), use_bias=False , padding='same', kernel_regularizer=l2(wt_decay), kernel_initializer=(initializers.RandomNormal(stddev=np.sqrt(2.0/(9*int(num_filter*compression))))))(input)

First_Block = add_denseblock(First_Conv2D, num_filter, dropout_rate)
First_Transition = add_transition(First_Block, num_filter, dropout_rate)

#First_Transition = merge([First_Transition,First_Conv2D], mode='concat', concat_axis=-1)

Second_Block = add_denseblock(First_Transition, num_filter, dropout_rate)
Second_Transition = add_transition(Second_Block, num_filter, dropout_rate)

#Second_Transition = Concatenate(axis=-1)([Second_Transition,First_Transition,First_Conv2D])

Third_Block = add_denseblock(Second_Transition, num_filter, dropout_rate)
Third_Transition = add_transition(Third_Block, num_filter, dropout_rate)

#Third_Transition = Concatenate(axis=-1)([Third_Transition,Second_Transition,First_Transition,First_Conv2D])

Last_Block = add_denseblock(Third_Transition,  num_filter, dropout_rate)
output = output_layer(Last_Block)


# In[25]:


bval_model = Model(inputs=[input], outputs=[output])
# bval_model.summary()
# print(bval_model)


# In[44]:


mergedOutput = Concatenate()([t2_model.output, adc_model.output, bval_model.output])
out = Dense(128, activation='relu')(mergedOutput)
out = Dense(256, activation='relu')(out)
out = Dense(512, activation='relu')(out)
out = Dense(1024, activation='relu')(out)
out = Dropout(0.8)(out)

out = Dense(32, activation='softmax')(out)
out = Dense(num_classes, activation='softmax')(out)

full_model = Model([t2_model.input, adc_model.input, bval_model.input],out) 
# full_model.summary()
# plot_model(full_model,  show_shapes=True)


# In[45]:


full_model.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])


# In[46]:


checkpoint_vl = ModelCheckpoint("weights.best_vl.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
checkpoint_va = ModelCheckpoint("weights.best_va.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint_va, checkpoint_vl]
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)


# Data Augmentation

# In[74]:


x_train_t2_v1 = x_train_t2/255.
x_val_t2_v1   = x_val_t2/255.
x_test_t2_v1  = x_test_t2/255.

y_train_t2_v1 = y_train_t2
y_val_t2_v1   = y_val_t2
y_test_t2_v1  = y_test_t2

x_train_adc_v1 = x_train_adc/255.
x_val_adc_v1   = x_val_adc/255.
x_test_adc_v1  = x_test_adc/255.

y_train_adc_v1 = y_train_adc
y_val_adc_v1   = y_val_adc
y_test_adc_v1  = y_test_adc

x_train_bval_v1 = x_train_bval/255.
x_val_bval_v1   = x_val_bval/255.
x_test_bval_v1  = x_test_bval/255.

y_train_bval_v1 = y_train_bval
y_val_bval_v1   = y_val_bval
y_test_bval_v1  = y_test_bval


# In[76]:


print(x_train_t2_v1.shape)
print(x_train_adc_v1.shape)
print(x_train_bval_v1.shape)


# In[71]:


def generator_three_img(X1, X2, X3, y, batch_size):
    
    gen = ImageDataGenerator(rotation_range=40,
                                   vertical_flip=True,
                                   horizontal_flip=True,
                                   fill_mode='nearest',
                                   samplewise_center=True)
    
    genX1 = gen.flow(X1, y,  batch_size=batch_size, seed=1)
    genX2 = gen.flow(X2, y,  batch_size=batch_size, seed=1)
    genX3 = gen.flow(X3, y,  batch_size=batch_size, seed=1)
    
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        X3i = genX3.next()
        yield [X1i[0], X2i[0], X3i[0]], X1i[1]


# In[72]:


history = full_model.fit([x_train_t2_v1,x_train_adc_v1,x_train_bval_v1], 
                            y = y_train_t2_v1, 
                            batch_size = 64, 
                            epochs = 100, 
                            verbose = 1, 
                            validation_data =([x_val_t2_v1,x_val_adc_v1,x_val_bval_v1], y_val_t2_v1),     
                            shuffle = True, 
                            callbacks = [es])
 


# In[43]:


# hist = full_model.fit_generator(generator_three_img(x_train_t2, x_train_adc, x_train_bval,
#                 y_train_t2, batch_size),
#                 verbose=1,
#                 epochs=100,
#                 callbacks = [es],
#                 steps_per_epoch=len(x_train_t2) // batch_size,              
#                 validation_data=([x_val_t2_v1,x_val_adc_v1,x_val_bval_v1], y_val_t2_v1),
#                 validation_steps=x_val_t2_v1.shape[0] // 16
#                 )


# In[ ]:


preds = full_model.evaluate(x=[x_test_t2_v1,x_test_adc_v1,x_test_bval_v1], y=y_test_t2_v1)
# preds = full_model.evaluate(x=[x_test_t2,x_test_adc],
#                       y=y_test_t2_v1)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

# full_model.save_weights('densenet.h5')


# In[57]:


from keras.models import load_model
load_model = full_model
# load_model.summary()
# plot_model(load_model,  show_shapes=True)


# In[58]:


load_model.load_weights('densenet_aug_3D.h5')
load_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[27]:


history = load_model


# In[ ]:


score = load_model.evaluate([x_test_t2_v1,x_test_adc_v1,x_test_bval_v1], y_test_t2, verbose=1)


# In[ ]:


def some_function(x):
    a = np.zeros(x.shape)
    a[:,np.argmax(x, axis=1)] = 1
    return a

b = some_function(predicted)
# b


# In[ ]:


predicted_test  = load_model.predict([x_test_t2_v1,x_test_adc_v1,x_test_bval_v1])
predicted_train = load_model.predict([x_train_t2,x_train_adc,x_train_bval])
predicted_val   = load_model.predict([x_val_t2_v1,x_val_adc_v1,x_val_bval_v1])


# In[59]:


for layer in load_model.layers: print(layer.get_config(), layer.get_weights())


# In[60]:


print(x_train_t2.shape)
print(x_train_adc.shape)
print(x_train_bval.shape)
#213


# In[66]:


first_layer_weights = load_model.layers[-2].get_weights()[0]
first_layer_biases  = load_model.layers[-2].get_weights()[1]
print(first_layer_weights.shape)
print(first_layer_biases.shape)
# first_layer_biases
# first_layer_weights

# new_array = np.append(first_layer_weights,first_layer_biases)
# new_array.shape


# In[ ]:


from sklearn.decomposition import PCA
pca = PCA()
principalComponents = pca.fit_transform(predicted_train)


# In[ ]:


principalDf = pd.DataFrame(data = principalComponents)
# principalDf
# y_train_t2_v1


# In[ ]:


categorical_train_labels = pd.DataFrame(y_train_t2).idxmax(axis=1)
print(collections.Counter(categorical_train_labels))
# categorical_train_labels[0:]


# In[ ]:


finalDf = pd.concat([principalDf, categorical_train_labels[0:]], axis = 1)
finalDf.columns = ["principal component 1","principal component 2", "principal component 3", "target"]
# finalDf


# In[ ]:


pca.explained_variance_ratio_


# In[ ]:





# In[ ]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)


# In[ ]:




