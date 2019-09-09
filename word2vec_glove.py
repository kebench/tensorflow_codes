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
    instance_indices = list(range(len(data_x)))
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
seqlens = np.array(seqlens)[data_indices]
train_x = data[:10000]
train_y = labels[:10000]
train_seqlens = seqlens[:10000]
test_x = data[10000:]
test_y = labels[10000:]
test_seqlens = seqlens[10000:]

_inputs = tf.placeholder(tf.int32, shape=[BATCH_SIZE, TIME_STEPS])
embedding_placeholder = tf.placeholder(tf.float32, [vocabulary_size, GLOVE_SIZE])

_labels = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_CLASSES])
_seqlens = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

if PRE_TRAINED:
    # We want to update the value of the word vectos so Trainable is set to True
    embeddings = tf.Variable(tf.constant(0.0, shape=[vocabulary_size, GLOVE_SIZE]), trainable=True)
    # If using pretrained embeddings, assign them to the embeddings variable
    embedding_init = embeddings.assign(embedding_placeholder)
    embed = tf.nn.embedding_lookup(embeddings, _inputs)
else:
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, EMBEDDING_DIMESION], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, _inputs)

# Bidirectional RNN for better understanding of context words
with tf.name_scope("biGRU"):
    with tf.name_scope("forward"):
        gru_fw_cell = tf.contrib.rnn.GRUCell(HIDDEN_LAYER_SIZE)
        gru_fw_cell = tf.contrib.rnn.DropoutWrapper(gru_fw_cell)

    with tf.name_scope("backward"):
        gru_bw_cell = tf.contrib.rnn.GRUCell(HIDDEN_LAYER_SIZE)
        gru_bw_cell = tf.contrib.rnn.DropoutWrapper(gru_bw_cell)

    output, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=gru_fw_cell, cell_bw=gru_bw_cell, inputs=embed, sequence_length=_seqlens, dtype=tf.float32, scope="BiGRU")
    # Concat Bi Directional State Vectors
    states = tf.concat(values=states, axis=1)

weights = {
    "linear_layer": tf.Variable(tf.truncated_normal([2 * HIDDEN_LAYER_SIZE, NUM_CLASSES], mean=0, stddev=0.01))
}

biases = {
    "linear_layer": tf.Variable(tf.truncated_normal([NUM_CLASSES], mean=0, stddev=0.01))
}

final_output = tf.matmul(states, weights["linear_layer"]) + biases["linear_layer"]
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=final_output, labels=_labels))

train_step = tf.train.RMSPropOptimizer(1e-3, 0.9).minimize(cross_entropy)
correct_prediction = tf.equal(tf.arg_max(_labels, 1), tf.arg_max(final_output, 1))
accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32))) * 100

# TRAAAAAAIN
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(embedding_init, feed_dict={
        embedding_placeholder: embedding_martix
    })

    for step in range(1000):
        x_batch, y_batch, seqlens_batch = get_sentence_batch(train_x, train_y, train_seqlens)
        feed_dict={
            _inputs: x_batch,
            _labels: y_batch,
            _seqlens: seqlens_batch
        }

        sess.run(train_step, feed_dict=feed_dict)

        if step % 100 == 0:
            acc = sess.run(accuracy, feed_dict=feed_dict)
            print("Accuracy at %d: {%.5f}" % (step, acc))

    for test_batch in range(5):
        x_test, y_test, seqlens_test = get_sentence_batch(test_x, test_y, test_seqlens)
        batch_pred, batch_acc = sess.run([tf.arg_max(final_output, 1), accuracy], feed_dict={
            _inputs: x_test,
            _labels: y_test,
            _seqlens: seqlens_test
        })

        print("Test batch accuracy at %d: {%.5f}" % (test_batch, batch_acc))

