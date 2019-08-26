# RNN using Triple layer of Long Short Term Memory (LSTM)
import numpy as np
import tensorflow as tf

# FUNCTIONS
def get_sentence_batch(batch_size, data_x, data_y, data_seqlens):
    instance_indices = list(range(len(data_x)))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    x = [[word_to_index_map[word] for word in data_x[i].lower().split()] for i in batch]
    y = [data_y[i] for i in batch]
    seqlens = [data_seqlens[i] for i in batch]
    return x, y, seqlens

# Create a new cell for every layer
def make_cell(lstm_size):
    return tf.contrib.rnn.BasicLSTMCell(lstm_size, state_is_tuple=True, forget_bias=1.0)

# SOME CONSTANTS
BATCH_SIZE = 128
EMBEDDING_DIMENSION = 64
NUM_CLASSES = 2
HIDDEN_LAYER_SIZE = 32
TIME_STEPS = 6
ELEMENT_SIZE = 1
NUM_LSTM_LAYER = 3

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

    #create the sentences
    even_sentences.append(" ".join([digit_to_word_map[r] for r in rand_even_ints]))
    odd_sentences.append(" ".join([digit_to_word_map[r] for r in rand_odd_ints]))

# add the odd and even sentences to form the data
data = even_sentences + odd_sentences
seqlens *= 2

# map each word to an integer
word_to_index_map = {}
index = 0

for sent in data:
    for word in sent.lower().split():
        if word not in word_to_index_map:
            word_to_index_map[word] = index
            index += 1

# create an inverse map
index_to_word_map = {index: word for word, index in word_to_index_map.items()}
vocabulary_size = len(index_to_word_map)

# print(word_to_index_map)
# print(index_to_word_map)

#create the labels and split the data to train and sets
labels = [1] * 10000 + [0] * 10000
for i in range(len(labels)):
    label = labels[i]
    one_hot_encoding = [0] * 2
    one_hot_encoding[label] = 1
    labels[i] = one_hot_encoding

# randomize data, labels and seqlens
data_indices = list(range(len(data)))
np.random.shuffle(data_indices)

data = np.array(data)[data_indices]
labels = np.array(labels)[data_indices]
seqlens = np.array(seqlens)[data_indices]

# for training
train_x = data[:10000]
train_y = labels[:10000]
train_seqlens = seqlens[:10000]

# For testing
test_x = data[10000:]
test_y = labels[10000:]
test_seqlens = seqlens[10000:]

_inputs = tf.placeholder(tf.int32, shape=[BATCH_SIZE, TIME_STEPS])
_labels = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_CLASSES])
# Seqlen for dynamic computation
_seqlens = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

# Word Embedding for scaling.
# Helpful when training large datasets
with tf.name_scope("embeddings"):
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, EMBEDDING_DIMENSION], -1.0, 1.0), name="embedding")
    embed = tf.nn.embedding_lookup(embeddings, _inputs)

# Create the Long Short Term Memory (LSTM) cell
with tf.variable_scope("LSTM"):
    # lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_LAYER_SIZE, forget_bias=1.0)
    # DON'T USE THE SAME CELL FOR THE FIRST AND DEEPER LAYERS! Using the same cell will cause dimension inequality
    cell = tf.contrib.rnn.MultiRNNCell(cells=[make_cell(HIDDEN_LAYER_SIZE) for _ in range(NUM_LSTM_LAYER)], state_is_tuple = True)
    output, states = tf.nn.dynamic_rnn(cell, embed, sequence_length=_seqlens, dtype=tf.float32)

weights = {
    "linear_layer": tf.Variable(tf.truncated_normal([HIDDEN_LAYER_SIZE, NUM_CLASSES], mean=0, stddev=0.01))
}

biases = {
    "linear_layer": tf.Variable(tf.truncated_normal([NUM_CLASSES], mean=0, stddev=0.01))
}

# Extract the last output and use it in a linear layer
final_output = tf.matmul(states[NUM_LSTM_LAYER - 1][1], weights["linear_layer"]) + biases["linear_layer"]


# Train the model and minimize the error. Won't execute unless run in the session. lol
softmax = tf.nn.softmax_cross_entropy_with_logits(logits=final_output, labels=_labels)
cross_entropy = tf.reduce_mean(softmax)
train_step = tf.train.RMSPropOptimizer(1e-3, 0.9).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(_labels, 1), tf.argmax(final_output, 1))
accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32))) * 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1000):
        x_batch, y_batch, seqlen_batch = get_sentence_batch(BATCH_SIZE, train_x, train_y, train_seqlens)
        feed_dict = {_inputs: x_batch, _labels: y_batch, _seqlens: seqlen_batch}
        sess.run(train_step, feed_dict=feed_dict)
    
        if step % 100 == 0:
            acc = sess.run(accuracy, feed_dict=feed_dict)
            print("Accuracy at %d: %.5f" % (step, acc))

    for test_batch in range(5):
        x_test, y_test, seqlen_test = get_sentence_batch(BATCH_SIZE, test_x, test_y, test_seqlens)
        feed_dict_test = {_inputs: x_test, _labels: y_test, _seqlens: seqlen_test}
        batch_pred, batch_acc = sess.run([tf.argmax(final_output, 1), accuracy], feed_dict=feed_dict_test)
        print("Test Batch Accuracy %d: %.5f" % (test_batch, batch_acc))

    output_example = sess.run([output], feed_dict=feed_dict_test)
    states_example = sess.run([states[1]], feed_dict=feed_dict_test)