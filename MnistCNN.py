
# coding: utf-8

# In[ ]:


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import plot_model
from keras import optimizers
from keras.utils import to_categorical

import numpy as np
import matplotlib.pyplot as plt
import cv2


# In[ ]:


(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[ ]:


numClass = np.unique(y_train).size
y_train = to_categorical(y_train,numClass,dtype='uint8')
y_test = to_categorical(y_test,numClass,dtype='uint8')

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


# In[ ]:


model = Sequential()
 
model.add(Conv2D(32, 3, 3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

print(model.summary())


# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


history = model.fit(X_train, y_train, batch_size=32, nb_epoch=10, verbose=1,validation_split=0.1)


# In[ ]:



# Plot training & validation accuracy values

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# In[ ]:


score = model.evaluate(x_test, x_test, verbose=0)

