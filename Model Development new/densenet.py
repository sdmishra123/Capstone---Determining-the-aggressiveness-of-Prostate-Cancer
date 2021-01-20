import tensorflow.keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, AveragePooling2D, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Concatenate, SpatialDropout2D
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import initializers
from tensorflow.keras.initializers import he_normal
#from tensorflow.keras.utils.np_utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#from tensorflow.keras.utils import plot_model
import collections
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import backend as k
import numpy as np
import pandas as pd



# In[2]:


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


# In[7]:


# labels = to_categorical(target,num_classes=5)
labels = np.array(pd.get_dummies(target))


# In[8]:


t2_tra_samples_v1 = np.expand_dims(t2_tra_samples, axis=3)
t2_tra_samples_v1.shape


# In[9]:


categorical_test_labels = pd.DataFrame(labels).idxmax(axis=1)
collections.Counter(categorical_test_labels)


# # Split the train and test data

# In[10]:


X_train_orig,X_test_orig,Y_train_orig,Y_test_orig = train_test_split(t2_tra_samples_v1,labels,test_size=0.20, 
                                                                     stratify=labels,  
                                                                     random_state=42)


# In[11]:


categorical_train_labels = pd.DataFrame(Y_train_orig).idxmax(axis=1)
collections.Counter(categorical_train_labels)


# In[12]:


categorical_test_labels = pd.DataFrame(Y_test_orig).idxmax(axis=1)
collections.Counter(categorical_test_labels)


# In[13]:


x_train,x_val,y_train,y_val = train_test_split(X_train_orig,Y_train_orig,test_size=0.20, 
                                                                     stratify=Y_train_orig,  
                                                                     random_state=42)


# In[16]:


categorical_train_labels = pd.DataFrame(y_train).idxmax(axis=1)
collections.Counter(categorical_train_labels)


# In[17]:


categorical_val_labels = pd.DataFrame(y_val).idxmax(axis=1)
collections.Counter(categorical_val_labels)


img_height, img_width, channel = X_train_orig.shape[1],X_train_orig.shape[2],X_train_orig.shape[3]


# # Densenet Model 
# ## 1.DenseBlocks and Layers

# In[20]:


# Hyperparameters
batch_size = 64
num_classes = 5
epochs = 150
l = 12
num_filter = 24
compression = 0.5
dropout_rate = 0.2
wt_decay = 0.001



# Dense Block
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


# In[22]:


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


# In[23]:


def output_layer(input):
    global compression
    BatchNorm = BatchNormalization()(input)
    relu = Activation('relu')(BatchNorm)
    AvgPooling = AveragePooling2D(pool_size=(2,2))(relu)
    flat = Flatten()(AvgPooling)
    # output = Dense(num_classes, activation='softmax', kernel_regularizer=l2(wt_decay))(flat)
    output = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(flat)
    
    return output


# In[24]:


input = Input(shape=(img_height, img_width, channel,))
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


model = Model(inputs=[input], outputs=[output])
model.summary()


# In[26]:


def step_down_20(epochs, crate):
    if ((epochs != 0) and (epochs%20 == 0)):
        crate = crate * 0.8
    return crate


# In[27]:


class LossHistory(tensorflow.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []
 
    def on_epoch_end(self, batch, logs={}):
        optimizer = self.model.optimizer
        lr_temp = step_down_20(optimizer.iterations,optimizer.lr)
        self.losses.append(logs.get('loss'))
        self.lr.append(lr_temp)


# # Compile the Model

# In[28]:


# determine Loss function and Optimizer
model.compile(loss='categorical_crossentropy',
              optimizer="adagrad",
              metrics=['accuracy'])


# In[29]:


checkpoint_vl = ModelCheckpoint("weights.best_vl.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
checkpoint_va = ModelCheckpoint("weights.best_va.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint_va, checkpoint_vl]


# In[30]:


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)


# # Stratified Data Augmentation 

# In[50]:


categorical_train_labels = pd.DataFrame(y_train).idxmax(axis=1)
print(collections.Counter(categorical_train_labels))

categorical_val_labels = pd.DataFrame(y_val).idxmax(axis=1)
print(collections.Counter(categorical_val_labels))


# In[72]:


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


# In[84]:


x_train_0  = np.take(x_train, index_labels_0,axis=0)
y_train_0  = np.take(y_train, index_labels_0,axis=0)
x_train_v0 = np.take(x_val, index_labels_v0,axis=0)
y_train_v0 = np.take(y_val, index_labels_v0,axis=0)



# In[85]:


x_train_1  = np.take(x_train, index_labels_1,axis=0)
y_train_1  = np.take(y_train, index_labels_1,axis=0)
x_train_v1 = np.take(x_val, index_labels_v1,axis=0)
y_train_v1 = np.take(y_val, index_labels_v1,axis=0)


# In[87]:


x_train_2  = np.take(x_train, index_labels_2,axis=0)
y_train_2  = np.take(y_train, index_labels_2,axis=0)
x_train_v2 = np.take(x_val, index_labels_v2,axis=0)
y_train_v2 = np.take(y_val, index_labels_v2,axis=0)


# In[89]:


x_train_3  = np.take(x_train, index_labels_3,axis=0)
y_train_3  = np.take(y_train, index_labels_3,axis=0)
x_train_v3 = np.take(x_val, index_labels_v3,axis=0)
y_train_v3 = np.take(y_val, index_labels_v3,axis=0)

# In[90]:


x_train_4  = np.take(x_train, index_labels_4,axis=0)
y_train_4  = np.take(y_train, index_labels_4,axis=0)
x_train_v4 = np.take(x_val, index_labels_v4,axis=0)
y_train_v4 = np.take(y_val, index_labels_v4,axis=0)

# ### Data Augmentation of Class : 0

# In[91]:


data_x_aug0 = x_train_0
data_y_aug0 = y_train_0

for i in range(0,60):
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

for i in range(0,20):
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

# In[92]:


data_x_aug1 = x_train_1
data_y_aug1 = y_train_1

for i in range(0,60):
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

for i in range(0,20):
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

# In[93]:


data_x_aug2 = x_train_2
data_y_aug2 = y_train_2

for i in range(0,100):
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

for i in range(0,50):
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

# In[95]:


data_x_aug3 = x_train_3
data_y_aug3 = y_train_3

for i in range(0,250):
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

for i in range(0,50):
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

# In[96]:


data_x_aug4 = x_train_4
data_y_aug4 = y_train_4

for i in range(0,300):
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

for i in range(0,50):
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


# In[102]:


x_train_final = np.append(data_final_x0,data_final_x1,axis=0)
y_train_final = np.append(data_final_y0,data_final_y1,axis=0)

x_train_final = np.append(x_train_final,data_final_x2,axis=0)
y_train_final = np.append(y_train_final,data_final_y2,axis=0)

x_train_final = np.append(x_train_final,data_final_x3,axis=0)
y_train_final = np.append(y_train_final,data_final_y3,axis=0)

x_train_final = np.append(x_train_final,data_final_x4,axis=0)
y_train_final = np.append(y_train_final,data_final_y4,axis=0)

x_train_vfinal = np.append(data_final_xv0,data_final_xv1,axis=0)
y_train_vfinal = np.append(data_final_yv0,data_final_yv1,axis=0)

x_train_vfinal = np.append(x_train_vfinal,data_final_xv2,axis=0)
y_train_vfinal = np.append(y_train_vfinal,data_final_yv2,axis=0)

x_train_vfinal = np.append(x_train_vfinal,data_final_xv3,axis=0)
y_train_vfinal = np.append(y_train_vfinal,data_final_yv3,axis=0)

x_train_vfinal = np.append(x_train_vfinal,data_final_xv4,axis=0)
y_train_vfinal = np.append(y_train_vfinal,data_final_yv4,axis=0)



# In[103]:


categorical_train_labels = pd.DataFrame(y_train_final).idxmax(axis=1)
collections.Counter(categorical_train_labels)


# # Normalize image vectors

# In[104]:


X_train = x_train_final/255.
X_val   = x_train_vfinal/255.
X_test  = X_test_orig/255.

Y_train = y_train_final
Y_val   = y_train_vfinal
Y_test  = Y_test_orig


# # Model Execution

# In[ ]:


history = model.fit(
                    X_train, Y_train, 
                    epochs = 150, 
                    batch_size = 64,
                    validation_data = (X_val,Y_val)
                    # callbacks=[es],
                    # shuffle=True
                   )


# In[ ]:


preds = model.evaluate(X_test_orig, Y_test_orig)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

model.save_weights('densenet.h5')

