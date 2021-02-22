import numpy as np
import os
from util import load_data_and_labels_from_csv_file, pad_sentences, text_to_sequence, load_vocab_json
import keras
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import tensorflow as tf
from keras import backend as K


def f1_score(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = tf.cast(c1, tf.float64) / tf.cast(c2, tf.float64)
    # How many relevant items are selected?
    recall = tf.cast(c1, tf.float64) / tf.cast(c3, tf.float64)
    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score.numpy()

#load model
model = keras.models.load_model("model/cp.ckpt/")
#print(model.summary())

vocab_file = "model/vocab.json"
vocabulary, seq_len = load_vocab_json(vocab_file)

#x_text = ["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&     C's apply 08452810075over18's", "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today."]

data_file = "data/SMSSpamCollection"
x_text, y = load_data_and_labels_from_csv_file(data_file)
#x_text = x_text[:5]
#y = y[:5]

#x = [process_text(sen).split() for sen in x_text]
x = pad_sentences(x_text, max_sequence_length=seq_len, is_max_sequence_length_modifiable=False)
x = text_to_sequence(x, vocabulary)

# Shuffle data
#np.random.seed(1000) #same shuffling each time
shuffle_indices = np.random.permutation(np.arange(len(y)))
x = x[shuffle_indices]
y = y[shuffle_indices]

checkpoint_path = "model/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

checkpoint = ModelCheckpoint(filepath=checkpoint_path,  monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto') # Create callback to save the weights
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=checkpoint_dir, histogram_freq=1)
adam = Adam(lr=2e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())
print("Traning Model...")
epochs = 5
batch_size = 32
validation_split = 0.1
verbose = 1

model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_split=validation_split, callbacks=[checkpoint, tensorboard_callback])
