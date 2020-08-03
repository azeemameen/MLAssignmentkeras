#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import (Input, Dense, Concatenate)
from keras.utils import np_utils
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.layers import Dropout
import numpy as np

# Load data
from keras.datasets import mnist

# input image dimensions
img_rows, img_cols = 28, 28                          
input_shape = (img_rows * img_cols, )

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], np.prod(x_train.shape[1:]))
x_test = x_test.reshape(x_test.shape[0], np.prod(x_test.shape[1:]))

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


# In[2]:


#Noice factor

noise_factor = 0.25
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)


# In[3]:


#Full dataset is obtained to include in the autoencorder model

x_feat_train = np.concatenate((x_train, x_test), axis=0)
x_feat_train_noisy = np.concatenate((x_train_noisy, x_test_noisy), axis=0)


# In[4]:


#autoencorder is built using Dense neural network

def DEEP_DAE(features_shape, act='relu'):

    # Input
    x = Input(name='inputs', shape=features_shape, dtype='float32')
    o = x
    
    # Encoder / Decoder
    o = Dense(1024, activation=act, name='dense1')(o)
    o = Dense(1024, activation=act, name='dense2')(o)
    o = Dense(1024, activation=act, name='dense3')(o)
    dec = Dense(784, activation='sigmoid', name='dense_dec')(o)
    
    # Print network summary
    Model(inputs=x, outputs=dec).summary()
    
    return Model(inputs=x, outputs=dec)


# In[6]:


#compliling the autoencoder

batch_size = 128
epochs = 5
             
autoenc = DEEP_DAE(input_shape)
autoenc.compile(optimizer='adadelta', loss='binary_crossentropy')

autoenc.fit(x_feat_train_noisy, x_feat_train, epochs=epochs, 
            batch_size=batch_size, shuffle=True)


# In[7]:


#Extracting the new features by predicting 

def FEATURES(model):
    input_ = model.get_layer('inputs').input
    feat1 = model.get_layer('dense1').output
    feat2 = model.get_layer('dense2').output
    feat3 = model.get_layer('dense3').output
    feat = Concatenate(name='concat')([feat1, feat2, feat3])
    model = Model(inputs=[input_],
                      outputs=[feat])
    return model

_model = FEATURES(autoenc)
features_train = _model.predict(x_train)
features_test = _model.predict(x_test)
print(features_train.shape, ' train samples shape')
print(features_test.shape, ' train samples shape')


# In[21]:


#Model 1 - Image classifier built using output from autoencoder and Y labels of orginal dataset using dense neural network


def DNN(features_shape, num_classes, act='relu'):

    # Input
    x = Input(name='inputs', shape=features_shape, dtype='float32')
    o = x
    
    # Encoder / Decoder
    o = Dense(1024, activation=act, name='dense1')(o)
    o = Dense(1024, activation=act, name='dense2')(o)
    o = Dense(1024, activation=act, name='dense3')(o)
    y_pred = Dense(num_classes, activation='sigmoid', name='pred')(o)
    
    # Print network summary
    Model(inputs=x, outputs=y_pred).summary()
    
    return Model(inputs=x, outputs=y_pred)


# In[22]:



input_shape2 = (features_train.shape[1], )
num_classes = 10

y_train_ohe = np_utils.to_categorical(y_train, num_classes)
y_test_ohe = np_utils.to_categorical(y_test, num_classes)
 
batch_size = 128
epochs = 20
model_fname = 'dnn'

callbacks = [ModelCheckpoint(monitor='val_acc', filepath=model_fname + '.hdf5',
                             save_best_only=True, save_weights_only=True,
                             mode='min')]
            
deep = DNN(input_shape2, num_classes)
deep.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['acc'])

history = deep.fit(features_train, y_train_ohe, epochs=epochs, 
                   batch_size=batch_size, shuffle=True,
                   validation_data=(features_test, y_test_ohe), 
                   callbacks=callbacks)


# In[23]:


#Model 2 - Image classifier built using output from autoencoder and Y labels of orginal dataset using convolutional neural network


model = Sequential()
input_shape2 = (features_train.shape[1], )
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=input_shape2))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))



# In[ ]:


#Training and validating the data and getting accuracy 

model.fit(features_train, y_train_ohe, epochs=20, batch_size=32, validation_data=(features_test, y_test_ohe), verbose=1)
loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)

print('Loss for the CNN model = %.3f' % loss)
print('Accuracy for the CNN model = %.3f' % (accuracy * 100.0))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




