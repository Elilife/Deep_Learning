#Importing
import matplotlib.pyplot as plt
import time
import numpy as np
np.random.seed(123) #for reproducibility
#!pip install -q keras
import keras
from keras import backend as K
K.set_image_dim_ordering('th')
#core layers from CNN
from keras.layers import Dense, Dropout, Activation, Flatten
#convolutional layers that will to train on image data
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical, np_utils
from keras.optimizers import SGD
#from keras.utils.vis_utils import plot_model


#getting the data from tensorflow.examples.tutorials.mnist import input_data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#the keras library also includes it, wow!

from keras.datasets import mnist

# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print('loaded data done')

print (X_train.shape)
#visualizing one data image
#plt.imshow(X_train[0])
#plt.show()
# preselection of data for fast computation and comparison
X_train=X_train[0:1000,:,:]
print(y_train.shape)
y_train=y_train[0:1000]
#PRE-PROCESSING
#preparing the data set to be understood as image in the CNN
#transform our dataset from having shape (n, width, height) to (n, depth, width, height).
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
#X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
print (X_train.shape)
#Convert data type into float and normalize values between 0 and 1

X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
X_train /= 255
#X_test /= 255

#checking the labeling data
print (y_train.shape)

# Convert 1-dimensional class arrays to 10-dimensional class matrices
Y_train = to_categorical(y_train, 10)
#Y_test = to_categorical(y_test, 10)

print (Y_train.shape)
print (Y_train[1])
# Create the model: MODEL ARCHITECTURE
model = Sequential()

# Add the input layer
#The input shape parameter should be the shape of 1 sample.
#In this case, it's the same (1, 28, 28) that corresponds to
#the (depth, width, height) of each digit image.
#But what do the first 3 parameters represent? They correspond
#to the number of convolution filters to use, the number of
#rows in each convolution kernel, and the number of columns in
#each convolution kernel, respectively.
model.add(Conv2D(32, 3, 3, activation='relu',input_shape=(1, 28, 28)))
# Add more hidden layer
model.add(Conv2D(32, 3, 3, activation='relu'))
#MaxPooling2D is a way to reduce the number of parameters in
#our model by sliding a 2x2 pooling filter across the previous
#layer and taking the max of the 4 values in the 2x2 filter.
model.add(MaxPooling2D(pool_size=(2,2)))
#this takes care of overfitting
model.add(Dropout(0.25))

#adding Fully connected Dense layers

#For Dense layers, the first parameter is the output size
#of the layer. Keras automatically handles the connections
#between layers.


#Also note that the weights from the Convolution layers must be
#flattened (made 1-dimensional) before passing them to the fully
#connected Dense layer.
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Add the output layer
#Note that the final layer has an output size of 10,
#corresponding to the 10 classes of digits.
model.add(Dense(10,activation='softmax'))
# Compile the model
print('starting to compile the model')
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#plotting the model build
print(model.summary())

#fitting the models1

start_time = time.time()

model_history=model.fit(X_train, Y_train,batch_size=32,validation_split=0.33,epochs=50, verbose=0)
print("--- %s seconds ---" % (time.time() - start_time))
print(model_history.history.keys())
# summarize history for accuracy
plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print(model_history.history['val_acc'][-1])
print(model_history.history['val_loss'][-1])
