import numpy as np
import tensorflow as tf

# SOME CONSTANTS
BATCH_SIZE = 128
EMBEDDING_DIMENSION = 64
NUM_CLASSES = 2
HIDDEN_LAYER_SIZE = 32
TIME_STEPS = 6
ELEMENT_SIZE = 1

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
even_sentences = []
odd_sentences = []
seqlens = []

for i in range(10000):
    rand_seq_lens = np.random.choice(range(3, 7))
    seqlens.append(rand_seq_lens)
    rand_odd_ints = np.random.choice(range(1, 10, 2), rand_seq_lens)
    rand_even_ints = np.random.choice(range(2, 10, 2), rand_seq_lens)

    #Add Padding to make the inputs equal of size
    if(rand_seq_lens < 6):
        rand_odd_ints = np.append(rand_odd_ints, [0] * (6 - rand_seq_lens))
        rand_even_ints = np.append(rand_even_ints, [0] * (6 - rand_seq_lens))

    even_sentences.append(" ".join([digit_to_word_map[r] for r in rand_even_ints]))
    odd_sentences.append(" ".join([digit_to_word_map[r] for r in rand_odd_ints]))

data = even_sentences + odd_sentences
seqlens *= 2

word_to_index_map = {}
index = 0

for sent in data:
    for word in sent.lower().split():
        if word not in word_to_index_map:
            word_to_index_map[word] = index
            index += 1

index_to_word_map = {index: word for word, index in word_to_index_map.items()}
vocabulary_size = len(index_to_word_map)

#create the labels and split the data to train and sets