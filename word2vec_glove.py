import os
import zipfile
import numpy as np
import tensorflow as tf

def get_glove(word_to_index_map, vocabulary_size):
    embeddings_weights = {}
    count_all_words = 0
    with zipfile.ZipFile(PATH_TO_GLOVE) as z:
        with z.open("glove.840B.300d.txt") as f:
            for line in f:
                vals = line.split()
                word = str(vals[0].decode("utf-8"))
                if word in word_to_index_map:
                    print(word)
                    count_all_words += 1
                    coefs = np.asarray(vals[1:], dtype="float32")
                    coefs /= np.linalg.norm(coefs)
                    embeddings_weights[word] = coefs
                if count_all_words == vocabulary_size - 1:
                    break
    return embeddings_weights

def get_sentence_batch(data_x, data_y, data_seqlens):
    instance_indices = list(range(data_x))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:BATCH_SIZE]
    x = [[word_to_index_map[word] for word in data_x[i].split()] for i in batch]
    y = [data_y[i] for i in batch]
    seqlens = [data_seqlens[i] for i in batch]

    return x, y, seqlens

PATH_TO_GLOVE = os.path.join(os.getcwd(), "glove/glove.840B.300d.zip")
PRE_TRAINED = True
GLOVE_SIZE = 300
BATCH_SIZE = 128
EMBEDDING_DIMESION = 64
NUM_CLASSES = 2
HIDDEN_LAYER_SIZE = 32
TIME_STEPS = 6

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
    rand_seq_len = np.random.choice(range(3, 7))
    seqlens.append(rand_seq_len)
    rand_odd_ints = np.random.choice(range(1, 10, 2), rand_seq_len)
    rand_even_ints = np.random.choice(range(2, 10, 2), rand_seq_len)

    if(rand_seq_len < 6):
        rand_odd_ints = np.append(rand_odd_ints, [0] * (6 - rand_seq_len))
        rand_even_ints = np.append(rand_even_ints, [0] * (6 - rand_seq_len))

    even_sentences.append(" ".join([digit_to_word_map[r] for r in rand_even_ints]))
    odd_sentences.append(" ".join([digit_to_word_map[r] for r in rand_odd_ints]))

data = even_sentences + odd_sentences
seqlens *= 2

labels = [1] * 10000 + [0] * 10000
for i in range(len(labels)):
    label = labels[i]
    one_hot_encoding = [0] * 2
    one_hot_encoding[label] = 1
    labels[i] = one_hot_encoding

word_to_index_map = {}
index = 0
for sent in data:
    for word in sent.split():
        if word not in word_to_index_map:
            word_to_index_map[word] = index
            index += 1

index_to_word_map = {index: word for word, index in word_to_index_map.items()}
vocabulary_size = len(index_to_word_map)
word_to_embeddings_dict = get_glove(word_to_index_map, vocabulary_size)

embedding_martix = np.zeros((vocabulary_size, GLOVE_SIZE))

for word, index in word_to_index_map.items():
    if not word == "PAD":
        word_embedding = word_to_embeddings_dict[word]
        embedding_martix[index, :] = word_embedding

data_indices = list(range(len(data)))
np.random.shuffle(data_indices)
data = np.array(data)[data_indices]
labels = np.array(labels)[data_indices]
seqlens = np.array(labels)[data_indices]
train_x = data[:10000]
train_y = data[:10000]
train_seqlens = data[:10000]
test_x = data[10000:]
test_y = data[10000:]
test_seqlens = data[10000:]
