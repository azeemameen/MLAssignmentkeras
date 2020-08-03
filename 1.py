#!/usr/bin/env python
# coding: utf-8

# In[16]:


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


# In[2]:


# loading the data from MNIST library 
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


# In[4]:


#reshape the train and test data
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)


# In[5]:


#Normalizing the training and test data to Pixel values 0-1
X_train = X_train.astype('float32')
X_train = X_train/255

# convert from integers to floats
X_test = X_test.astype('float32')
X_test = X_test/255


# In[6]:


#Coverting Y value into categorical values
Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)


# In[10]:


#Creating the Nerual netwok

model = Sequential()
input_shape = (28, 28, 1)
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))


# In[11]:


#Compiling the model
opt = SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


# In[12]:


#Training and validating the data

model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_data=(X_test, Y_test), verbose=1)


# In[13]:


loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)

print('Loss for the CNN model = %.3f' % loss)
print('Accuracy for the CNN model = %.3f' % (accuracy * 100.0))


# In[14]:


#adding noice factor to the dataset 


# In[17]:


# define the noise factor of 0.25 
noise_factor = 0.25
X_train = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_test = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)
X_train = np.clip(X_train, 0., 1.)
X_test = np.clip(X_test, 0., 1.)


# In[18]:


#Training and validating the data and getting accuracy 

model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_data=(X_test, Y_test), verbose=1)
loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)

print('Loss for the CNN model = %.3f' % loss)
print('Accuracy for the CNN model = %.3f' % (accuracy * 100.0))

