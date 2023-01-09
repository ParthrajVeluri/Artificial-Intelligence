from keras.datasets import imdb 
from keras.utils import pad_sequences 
from keras import preprocessing
import tensorflow as tf 
import os 
import numpy as np 

VOCAB_SIZE = 88584 #Number of unique words in the dataset
MAXLEN = 250 #Max length of the review
BATCH_SIZE = 64 

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = VOCAB_SIZE)

#All Reviews are different length, but we need constant length to feed into the RNN
#Pad each review so that a review > 250 words will have everything clipped after 250
#And a review < 250 words will have necessary amounts of '0' added to it to make the review
# of length 250
train_data = pad_sequences(train_data, MAXLEN)
test_data = pad_sequences(test_data, MAXLEN)

#Embedding layer creates a tensor which maps similar words together
#and unsimilar words far away each output from the Embedding Layer
#will have 32 dimensions. LSTM is the RNN layer which 
# calculates on those vectors and learns from them what each sentence means
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32), 
    tf.keras.layers.LSTM(32), 
    tf.keras.layers.Dense(1, activation = "sigmoid")
])

model.summary()

#model.compile(loss="binary_crossentropy", optimizer= "rmsprop", metrics = ["acc"])

#history = model.fit(train_data, train_labels, epochs=10, validation_split = 0.2)

#model.save("RNN", include_optimizer = True, save_format = "tf")

loaded_model = tf.keras.models.load_model("RNN", compile = False )
loaded_model.compile(loss="binary_crossentropy", optimizer = "rmsprop", metrics = ["acc"])
loaded_model.summary()

print("Model Evaluation: ")
#results = loaded_model.evaluate(test_data, test_labels)
#print(results)

#Predictions 
word_index = imdb.get_word_index()

#Encode function
def encode_text(text):
  tokens = preprocessing.text.text_to_word_sequence(text) #Breaks the sentence into ann array
  #Replace unrecognized words with 0 in the sentence
  tokens = [word_index[word] if word in word_index else 0 for word in tokens] 
  return pad_sequences([tokens], MAXLEN)[0]

text = "that movie was just amazing, so amazing"
encoded = encode_text(text)
print(encoded)

#Prediction Function
def predict(text):
    encoded_text = encode_text(text)
    pred = np.zeros((1,250))
    pred[0] = encoded_text
    result = loaded_model.predict(pred)
    print(result[0])

#Predicts with 88% accuracy
positive_review = "That movie was so amazing would definitely watch again!"
predict(positive_review)

negative_review = "Top Gun Maverick was one of the worst movies I have ever watched."
predict(negative_review)