from keras.utils import pad_sequences
import keras 
import tensorflow as tf
import os
import numpy as np

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path_to_file, 'rb').read().decode(encoding='utf8')

vocab = sorted(set(text))
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

def text_to_int(text):
    return np.array([char2idx[c] for c in text])

def int_to_text(ints):
    try: 
        ints = ints.numpy()
    except:
        pass
    return ''.join(idx2char[ints])


print("Text:", text[:13])
print("Encoded:", text_to_int(text[:13]))

seq_length = 100  # length of sequence for a training example
examples_per_epoch = len(text)//(seq_length+1)

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_to_int(text))

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_text(text): 
    input_text = text[:-1]
    output_text = text[1:]
    return input_text, output_text

dataset = sequences.map(split_input_text)

BATCH_SIZE = 64
VOCAB_SIZE = len(vocab)  # vocab is number of unique characters
EMBEDDING_DIM = 256
RNN_UNITS = 1024

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000


data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

def build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM,
                              batch_input_shape=[BATCH_SIZE, None]),
    tf.keras.layers.LSTM(RNN_UNITS,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(VOCAB_SIZE)
  ])
  return model

model = build_model(VOCAB_SIZE,EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
model.summary()

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

for input_example_batch, target_example_batch in data.take(1):
    example_batch_predictions = model(input_example_batch)  # ask our model for a prediction on our first batch of training data (64 entries)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")  # print out the output shape
    # lets examine one prediction
    pred = example_batch_predictions[0]
    print(len(pred))
    print(pred)
    # notice this is a 2d array of length 100, where each interior array is the prediction for the next character at each time step
    # If we want to determine the predicted character we need to sample the output distribution (pick a value based on probabillity)
    sampled_indices = tf.random.categorical(pred, num_samples=1)

    # now we can reshape that array and convert all the integers to numbers to see the actual characters
    sampled_indices = np.reshape(sampled_indices, (1, -1))[0]
    predicted_chars = int_to_text(sampled_indices)

    predicted_chars  # and this is what the model predicted for training sequence 1

model.compile(optimizer='adam', loss=loss)
# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

#history = model.fit(data, epochs=50, callbacks=[checkpoint_callback])
model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

checkpoint_num = 50
model.load_weights("./training_checkpoints/ckpt_" + str(checkpoint_num))
model.build(tf.TensorShape([1, None]))

def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 10000

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
    
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))

inp = input("Type a starting string: ")
print(generate_text(model, inp))