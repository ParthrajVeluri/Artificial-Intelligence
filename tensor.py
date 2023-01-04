from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from urllib.parse import urlencode
import urllib

import tensorflow._api.v2.compat.v2.feature_column as fc

import tensorflow as tf

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
y_train = dftrain.pop('survived') #remove survived column from the dataset
y_eval = dfeval.pop('survived') #remove survived column from the dataset

dftrain.age.hist(bins=20, ax=plt.subplot(1,3,1)) #subplot(#of rows, #of cols, index)

dftrain['class'].value_counts().plot(kind='barh', ax = plt.subplot(1,3,2))

categorical_cols = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone'] #non numerical data (Must be mapped to numerical data)
numeric_cols = ['age', 'fare'] #numerical data, good to go

feature_cols = [] #all the features 

for feature_name in categorical_cols: 
    vocabulary = dftrain[feature_name].unique() #gets list of all unique values from given feature column 
    feature_cols.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary)) #tensorflow handles the conversion to numerical

for feature_name in numeric_cols: 
    feature_cols.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32)) #inserts numerical values as float32

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

# We create a linear estimtor by passing the feature columns we created earlier
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_cols)

linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on tetsing data

clear_output()  # clears consoke output
print(f"accuracy = {result['accuracy']}")  # the result variable is simply a dict of stats about our model
pred_dicts = list(linear_est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

probs.plot(kind='hist', bins=20, title='predicted probabilities', ax = plt.subplot(1,3,3))

plt.show()
