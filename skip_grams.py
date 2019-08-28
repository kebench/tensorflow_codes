import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

BATCH_SIZE = 64
EMBEDDING_DIMENSION = 5
NEGATIVE_SAMPLES = 8
LOG_DIR = "logs/word2vec_intro"

def get_skipgram_batches(batch_size):
    instance_indices = list(range(len(skip_gram_pairs)))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    x = [skip_gram_pairs[i][0] for i in batch]
    y = [skip_gram_pairs[i][1] for i in batch]
    return x, y

digit_to_word_map = {
    0: "PAD", #will be used for zero-padding preprocessing of data
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Six",
    7: "Seven",
    8: "Eight",
    9: "Nine"
}
sentences = []

# Create two kinds of sentences---odd and even
for i in range(10000):
    rand_odd_inits = np.random.choice(range(1, 10, 2), 3)
    sentences.append(" ".join([digit_to_word_map[r] for r in rand_odd_inits]))
    rand_even_inits = np.random.choice(range(2, 10, 2), 3)
    sentences.append(" ".join([digit_to_word_map[r] for r in rand_even_inits]))


# Map word to IDs
word_to_index_map = {}
index = 0

for sentence in sentences:
    for word in sentence.lower().split():
        if word not in word_to_index_map:
            word_to_index_map[word] = index
            index += 1

index_to_word_map = {index: word for word, index in word_to_index_map.items()}
vocabulary_size = len(index_to_word_map)

# Generate the Skip-Gras pairs
skip_gram_pairs = []

for sentence in sentences:
    tokenized_sentence = sentence.lower().split()
    
    # Create an array of [target_word, context_word]
    for i in range(1, len(tokenized_sentence) - 1):
        word_context_pair = [[word_to_index_map[tokenized_sentence[i - 1]],
        word_to_index_map[tokenized_sentence[i + 1]]],
        word_to_index_map[tokenized_sentence[i]]]

        skip_gram_pairs.append([word_context_pair[1], word_context_pair[0][0]])
        skip_gram_pairs.append([word_context_pair[1], word_context_pair[0][1]])

print(skip_gram_pairs[0:10])