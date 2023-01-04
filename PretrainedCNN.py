import os 
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
import tensorflow_datasets as tfds
keras = tf.keras

tfds.disable_progress_bar()

#split the data into 80% training, 10% test and 10% validation 
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split = ['train[:80%]', 'train[80%:90%]', 'train[90%:100%]'],
    with_info = True, 
    as_supervised = True)

#Image sizes are different, this function reformats them to the same shape
IMG_SIZE = 160

def format_example(image, label): 
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1 
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

#Creating a base model from a pretrained model from google
base_model = keras.applications.MobileNetV2(input_shape = IMG_SHAPE,include_top = False, weights = 'imagenet')
# include top: do we include the classifier that comes with the model or not
# weights = imagenets is just a presaved weights value

base_model.summary()

#Find the shape of the last layer of pretrained model; returns shape 32,5,5,1280
for image, _ in train_batches.take(1):
    pass
    feature_batch = base_model(image)
    print(feature_batch.shape)

#freeze the base so that it wont be tweaked when training our model
base_model.trainable = False

#takes the 1280 5x5 feature maps returned from the pretrained model, takes the avg of each 5x5 feature map
#and returns a 1280 size array with the averages
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
#Output layer
prediction_layer = keras.layers.Dense(1)

model = tf.keras.Sequential([
    base_model, 
    global_average_layer, 
    prediction_layer
])

model.summary()    

base_learning_rate = 0.0001
model.compile(
    optimizer = tf.keras.optimizers.RMSprop(learning_rate = base_learning_rate),
    loss = keras.losses.BinaryCrossentropy(from_logits = True),
    metrics = ["accuracy"])

initial_epochs = 3
validation_steps = 20

#loss0, accuracy0 = model.evaluate(validation_batches, steps = validation_steps)

#history = model.fit(train_batches, epochs= initial_epochs, validation_data = validation_batches)
#model.save("dogs_vs_cats.h5")
new_model = keras.models.load_model("dogs_vs_cats.h5")

#convert test to numpy array to start predicting with it

numpy_img = []
numpy_label = []
for images, labels in test.take(-1):
    numpy_img.append(images.numpy())
    numpy_label.append(labels.numpy())

#numpy_img = tf.convert_to_tensor(numpy_img, dtype = tf.float32)
pred = new_model.predict(test_batches)
for i in range(64):
    if pred[i] < 1:
        print("cat")
    else:
        print("dog")

    plt.figure()
    plt.imshow(numpy_img[i])
    plt.show()
