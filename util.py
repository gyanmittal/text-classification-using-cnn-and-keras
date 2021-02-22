import numpy as np
import re
import itertools
from collections import Counter
import json
import math

def process_text(text):
    text = text.strip().lower()
    text = text.replace("<br />", " ")
    text = text.replace("<br/>", " ")
    text = ' ' + text + ' '
    text = re.sub(r"[^A-Za-z0-9\'\.*$@#]", " ", text) #replace all the characters with space except mentioned here
    text = re.sub(r"['.]", "", text) # Remove . and '
    text = re.sub(r" +", " ", text) #replace muliple space with single space
    text = re.sub(r' [0-9 ]+ \s*', ' <n/> ', text)  # Replace with special notation in case only digits
    text = re.sub(r' [^a-z]+ \s*', ' <sd/> ', text)    # Replace with special notation in case only special character, with or without digit
    text = re.sub(r'(.)\1+', r'\1\1', text) #strip in case of consecutive more than 2 characters to two characters
    text = re.sub(r'([^a-z])\1+', r'\1', text) #strip in case of special characters occur more than once
    return text

def load_data_and_labels_from_csv_file(csv_file):
    examples = list(open(csv_file, 'r', encoding='utf-8').readlines())
    examples = [s.strip() for s in examples]
    x_text = [sen.split("\t") for sen in examples]

    labels = []
    data = []
    for text in x_text:
        current_line_data = process_text(text[1])
        labels.append([0]) if text[0] == 'ham' else labels.append([1])
        data.append(current_line_data.split())
    return data, np.array(labels)

def pad_sentences(sentences, padding_word='<PAD/>', max_sequence_length=200, is_max_sequence_length_modifiable=True):
   
    padded_sentences = []
    max_length = max([len(sentence) for sentence in sentences])
    if (max_length < max_sequence_length and is_max_sequence_length_modifiable==True): max_sequence_length = max_length 
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = max_sequence_length - len(sentence)
        if(num_padding > 0):
            new_sentence = sentence + [padding_word] * num_padding
        else:
            new_sentence = sentence[:max_sequence_length]
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences, max_vocab_size=20000, min_word_freq=2, padding_word='<PAD/>', unknown_word='<UNK/>'):
    word_counts = Counter(itertools.chain(*sentences)) # Count words
    vocabulay_inv = [[unknown_word, math.inf], [padding_word,math.inf]] + [[x[0], x[1]] for x in word_counts.most_common()] # Sort the word as frequency order
    if(len(vocabulay_inv) > (max_vocab_size+2)):
        vocabulay_inv = vocabulay_inv[:(max_vocab_size+2)]
    vocabulary = {word[0]: [i,word[1]] for i, word in enumerate(vocabulay_inv) if word[1] >= min_word_freq} # Build vocabulary, word: index
    return vocabulary

def text_to_sequence(sentences, vocabulary, unknown_word='<UNK/>'):
    x = np.array([[vocabulary[word][0] if word in vocabulary else vocabulary[unknown_word][0] for word in sen] for sen in sentences])
    return x

def save_vocab_json(file_path, word2id, seq_len):
    json.dump(dict(src_word2id=word2id, seq_len={"seq_len":seq_len}), open(file_path, 'w'), indent=2)
    #json.dump(dict(seq_len={"seq_len":seq_len}), open(file_path, 'w'), indent=2)

def load_vocab_json(file_path):
    entry = json.load(open(file_path, 'r'))
    src_word2id = entry['src_word2id']
    seq_len = entry['seq_len']
    return src_word2id, seq_len['seq_len']
