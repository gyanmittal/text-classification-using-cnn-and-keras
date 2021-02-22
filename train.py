import numpy as np
import os
from util import load_data_and_labels_from_csv_file, build_vocab, pad_sentences, text_to_sequence, save_vocab_json
import keras
from keras.layers import Embedding, Reshape, Conv2D, MaxPool2D, Concatenate, Flatten, Dropout, Dense
from keras.callbacks import ModelCheckpoint 
from keras.optimizers import Adam
from keras.models import Input, Model

data_file = "data/SMSSpamCollection"
# Load data
print("Loading data...")
x_text, y = load_data_and_labels_from_csv_file(data_file)


# Build vocabulary
print("Build the vocabulary")
vocabulary = build_vocab(x_text, max_vocab_size=10000)
#print(vocabulary)

# Pad sentence
print("Padding sentences...")
x_text = pad_sentences(x_text, max_sequence_length=100)

seq_len = len(x_text[0]) 
print("The sequence length is: ", seq_len)

# Represent sentence with word index, using word index to represent a sentence
x = text_to_sequence(x_text, vocabulary)

# Shuffle data
#np.random.seed(1000) #same shuffling each time
shuffle_indices = np.random.permutation(np.arange(len(y)))
x = x[shuffle_indices]
y = y[shuffle_indices]

"""
## Build CNN model
"""
vocab_size_or_no_features = len(vocabulary) 

embed_dim = 300
filter_sizes = [3,4,5]
num_filters = 512
drop_out = 0.5

# this returns a tensor
print("Creating Model...")
inputs = Input(shape=(seq_len,), dtype='int32')
embedding = Embedding(input_dim=vocab_size_or_no_features, output_dim=embed_dim, input_length=seq_len)(inputs)
reshape = Reshape((seq_len,embed_dim,1))(embedding)

conv0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embed_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
maxpool0 = MaxPool2D(pool_size=(seq_len - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv0)
conv1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embed_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
maxpool1 = MaxPool2D(pool_size=(seq_len - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv1)
conv2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embed_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
maxpool2 = MaxPool2D(pool_size=(seq_len - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv2)

concatenate_maxpool = Concatenate(axis=1)([maxpool0, maxpool1, maxpool2])
flatten = Flatten()(concatenate_maxpool)
dropout = Dropout(drop_out)(flatten)
output = Dense(units=1, activation='sigmoid')(dropout)

model = Model(inputs=inputs, outputs=output) # Create model

checkpoint_path = "model/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

checkpoint = ModelCheckpoint(filepath=checkpoint_path,  monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto') # Create callback to save the weights
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=checkpoint_dir, histogram_freq=0)
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

epochs = 20
batch_size = 32
verbose = 1
validation_split = 0.1
print("Traning Model...")

model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_split=validation_split, callbacks=[checkpoint, tensorboard_callback])

# Save Vocabulary
vocab_file = checkpoint_dir + "/vocab.json"
save_vocab_json(vocab_file, vocabulary, seq_len)
