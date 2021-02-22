import keras
from util import load_vocab_json, load_vocab_json, load_data_and_labels_from_csv_file, pad_sentences, text_to_sequence
from keras import backend as K
import tensorflow as tf

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
    return precision.numpy(), recall.numpy(), f1_score.numpy()

#load model
model = keras.models.load_model("model/cp.ckpt/")
#print(model.summary())

vocab_file = "model/vocab.json"
vocabulary, seq_len = load_vocab_json(vocab_file)

data_file = "data/SMSSpamCollection"
x_text, y = load_data_and_labels_from_csv_file(data_file)
x = pad_sentences(x_text, max_sequence_length=seq_len, is_max_sequence_length_modifiable=False)
x = text_to_sequence(x, vocabulary)
'''
x = x[:10]
y = y[:10]
x_text = x_text[:10]
'''

print("Generate predictions")
predictions = model.predict(x)
#print(predictions)
print("Ham caught as Spam\n\n")
count = 0
for text in x_text:
    if (predictions[count] >= 0.5 and y[count] == 0):
        print((count+1), "\t\t", text, "\t\t", predictions[count])
    count += 1 
print("Spam classified as Ham\n\n")
count = 0

for text in x_text:
    if (predictions[count] < 0.5 and y[count] == 1):
        print((count+1), "\t\t", text, "\t\t", predictions[count])
    count += 1

precision, recall, f1_score = f1_score(y, predictions)

print("precision:\t", precision * 100)
print("recall:\t", recall * 100)
print("f1_score:\t",f1_score * 100)
