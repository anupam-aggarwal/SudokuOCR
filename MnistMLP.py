
# coding: utf-8

# In[48]:


# organize imports
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.datasets import mnist
from keras.utils import np_utils

# fix a random seed for reproducibility
np.random.seed(9)

# user inputs
nb_epoch            = 25
num_classes         = 10
batch_size          = 64
train_size          = 60000
test_size           = 10000
v_length            = 784
model_save_path     = "output/mlp"

# split the mnist data into train and test
(trainData, trainLabels), (testData, testLabels) = mnist.load_data()

# reshape and scale the data
trainData   = trainData.reshape(train_size, v_length)
testData    = testData.reshape(test_size, v_length)
trainData   = trainData.astype("float32")
testData    = testData.astype("float32")
trainData  /= 255
testData   /= 255

# convert class vectors to binary class matrices --> one-hot encoding
mTrainLabels  = np_utils.to_categorical(trainLabels, num_classes)
mTestLabels   = np_utils.to_categorical(testLabels, num_classes)

# create the MLP model
model = Sequential()
model.add(Dense(512, input_shape=(v_length,)))
model.add(Activation("relu"))
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(Dense(num_classes))
model.add(Activation("softmax"))

# compile the model
model.compile(loss="categorical_crossentropy",
              optimizer='adam',
              metrics=["accuracy"])

# fit the model
history = model.fit(trainData, 
                    mTrainLabels,
                    validation_data=(testData, mTestLabels),
                    batch_size=batch_size,
                    nb_epoch=nb_epoch,
                    verbose=2)

# evaluate the model
scores = model.evaluate(testData, mTestLabels, verbose=0)

# print the results
print ("[INFO] test score - {}".format(scores[0]))
print ("[INFO] test accuracy - {}".format(scores[1]))

# save tf.js specific files in model_save_path
# tfjs.converters.save_keras_model(model, model_save_path)


# In[49]:


# Plot training & validation accuracy values
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
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


# In[50]:


#to save model to disk
# serialize model to JSON
model_json = model.to_json()
with open("./output/mlp/Final_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("./output/mlp/Final_model.h5")
print("Saved model to disk")


# In[51]:


import cv2


# In[62]:


count = 0


# In[95]:


x= np.load("nums.npy")
ls = x.reshape(81,784)

temp = []
for img in ls:
    temp.append(int(np.average(img)))
temp = np.asarray(temp)

x = x[temp>5]


# In[96]:


cls = model.predict_classes(ls[temp > 5])

for i in range(x.shape[0]):
    cv2.imwrite('./dataset/'+str(cls[i]) + '/'+str(count)+'.jpg',x[i])
    count = count+1

