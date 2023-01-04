import tensorflow as tf
from tensorflow import keras 
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist #includes 60k images for training
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() #load dataset

print(train_images.shape)
print(train_images[0,23,23]) #each pixel ranges from 0 to 255 with 255 being white and 0 being black
print(train_labels[:25]) #each label ranges from 0 to 9 with each number representing a classification

#map each # from 0 to their corresponding object
class_names = ['tshirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'boot']

#Preprocessing 
#Makes the greyscale from 0 to 1 instead of 0 to 255
train_images = train_images/255.0 
test_images = test_images/255.0

#Building Neural Network Architecture
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)), #Input layer, takes the 28x28 pixel matrix and flattens it to an array of 784 
    keras.layers.Dense(128, activation = 'relu'), #Hidden layer, 128 neurons each with relu function
    keras.layers.Dense(10, activation = 'softmax') #Output layer, 10 output neurons each with softmax activation function
])
#Note: Use softmax when your output is discrete and sigmoid when your output is continuous 

#Defining Optimizor, Loss Function and Metrics
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',  
              metrics = ['accuracy'] #output u want to see from the network
)   

#training model
model.fit(train_images, train_labels, epochs = 5) 

#testing data
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 1) 
print("Test Accuracy = ", test_acc) 

#predicting 
print(test_images.shape)
predictions = model.predict(test_images)
print(class_names[np.argmax(predictions[6])]) #prints out what the model thinks the image is

plt.figure() #shows what the image was 
plt.imshow(test_images[6], cmap='gray')
plt.show()




