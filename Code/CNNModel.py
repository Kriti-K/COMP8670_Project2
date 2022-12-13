


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#read dataset
dataset = pd.read_csv('I:\PycharmProjects\Datasets\icml_face_data.csv')
print(dataset.shape)
print(dataset.head())


#separate data into X and Y
emotions = dataset['emotion']
pixels = dataset['pixels']
print(emotions.head())
print(pixels.head())


#reshape pixels into images and normalize values
pixelList = pixels.tolist()
p = []
for i in pixelList:
    arr = []
    for num in i.split(' '):
        arr.append(num)
    arr = np.asarray(arr).reshape(48, 48)
    p.append(arr.astype('float32'))



p = np.asarray(p)
p = np.expand_dims(X, -1)
print(p)


#show first 5 images of data
for i in range(5):
    plt.figure(i)
    plt.imshow(p[i], cmap='gray')
plt.show()


#split data into train and test subsets
xtrain, xtest, ytrain, ytest = train_test_split(p, emotions, test_size=0.1, shuffle=True)
print("xtrain shape: ", xtrain.shape)
print("ytrain shape: ", ytrain.shape)
print("xtest shape: ", xtest.shape)
print("ytest shape: ", ytest.shape)


#hyper-parameters
learningRate = 0.001
batch = 64
epochs = 10
imgSize = (48,48,1)
classes = 7


#transform the label data to a binary matrix so that it can work with the loss function
ytrain = keras.utils.to_categorical(ytrain, classes)
ytest = keras.utils.to_categorical(ytest, classes)

#define the CNN and print the summary
model = keras.Sequential([
    keras.Input(shape=(48,48,1)),

    # Here we normalize the input image and rescale the rgb values from the range [0,255] to [0,1]
    keras.layers.Rescaling(scale=1./255, input_shape=(imgSize)), 

    #first convolutional block
    layers.Conv2D(32, kernel_size=(3, 3), strides=(1,1), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.2),

    #second convolutional block
    layers.Conv2D(64, kernel_size=(3, 3), strides=(1,1), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.2),

    #third convolutional block
    layers.Conv2D(128, kernel_size=(3, 3), strides=(1,1), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.2),
    
    #fourth convolutional block
    layers.Conv2D(256, kernel_size=(3, 3), strides=(1,1), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.2),
    

    #Fully connected layer
    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.2),
    

    #Output Layer
    layers.Dense(classes, activation='softmax')
    
])

model.summary()

#define the loss function and optimizer with the learning rate
optim = keras.optimizers.Adam(learning_rate=learningRate)
lossFn = tf.keras.losses.CategoricalCrossentropy() 

#Compile and Train the model
model.compile(loss=lossFn, optimizer=optim, metrics=["accuracy"])
stats = model.fit(xtrain, ytrain, batch_size=batch, epochs=epochs, validation_split=0.1)


#Evaluate the model on the test data
results = model.evaluate(xtest, ytest)
print("ACCURACY:", results[1])


#Graph the losses and accuracies
plt.plot(range(epochs), stats.history["loss"])
plt.plot(range(epochs), stats.history["val_loss"])
plt.legend(["Training Loss", "Validation Loss"])
plt.title("Training and Validation Loss")
plt.show()
plt.plot(range(epochs), stats.history["accuracy"])
plt.plot(range(epochs), stats.history["val_accuracy"])
plt.legend(["Training accuracy", "Validation accuracy"])
plt.title("Training and Validation Accuracy")
plt.show()



