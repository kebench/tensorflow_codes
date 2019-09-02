import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

BATCH_SIZE = 64
EMBEDDING_DIMENSION = 5
NEGATIVE_SAMPLES = 8
LOG_DIR = "logs/word2vec_intro"

working_directory = os.path.join(os.getcwd(), LOG_DIR)

def get_skipgram_batches(batch_size):
    instance_indices = list(range(len(skip_gram_pairs)))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    x = [skip_gram_pairs[i][0] for i in batch]
    y = [[skip_gram_pairs[i][1]] for i in batch]
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

# reverse the mapping of word to index
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

train_inputs = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
train_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1])

with tf.name_scope("embeddings"):
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, EMBEDDING_DIMENSION], -1, 1, name="embeddings"))
    # Look Up Table
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

# Noise Constrastive Estimation Loss Function - close approximation to the ordinary softmax function
nce_weights = tf.Variable(tf.random_uniform([vocabulary_size, EMBEDDING_DIMENSION], 1.0 / math.sqrt(EMBEDDING_DIMENSION)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, inputs=embed, labels=train_labels, num_sampled=NEGATIVE_SAMPLES, num_classes=vocabulary_size))
tf.summary.scalar("NCE_loss", loss)

#adjustment of the optimization of learning rate
global_step = tf.Variable(0, trainable=False)
# Applies Exponential Decay to the learning rate. Makes learning efficient and minimizes the loss
learning_rate = tf.train.exponential_decay(learning_rate=0.1, global_step=global_step, decay_steps=1000, decay_rate=0.95, staircase=True)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

merged = tf.summary.merge_all()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(LOG_DIR, graph=tf.get_default_graph())

    saver = tf.train.Saver()

    # Create and open the metadata file to connect embedding vectors to associated labels or images.
    with open(os.path.join(LOG_DIR, "metadata.tsv"), "w") as metadata:
        metadata.write("Name\tClass\n")
        for key, val in index_to_word_map.items():
            metadata.write("%s\t%d\n" % (val, key))

    # point the tensorboard to the embedding variable and link it with the metadata file
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embeddings.name

    embedding.metadata_path = "metadata.tsv" #os.path.join(LOG_DIR, "metadata.tsv")
    projector.visualize_embeddings(train_writer, config)

    tf.global_variables_initializer().run()

    for step in range(1000):
        x_batch, y_batch = get_skipgram_batches(BATCH_SIZE)
        summary, _ = sess.run([merged, train_step], feed_dict={train_inputs: x_batch, train_labels: y_batch})
        train_writer.add_summary(summary, step)

        if step % 100 == 0:
            saver.save(sess, os.path.join(LOG_DIR, "w2v_model.ckpt"), step)
            loss_value = sess.run(loss, feed_dict={train_inputs: x_batch, train_labels: y_batch})
            print("Loss at %d: %.5f" % (step, loss_value))

    # Normalize embeddings before using
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    normalized_embeddings_matrix = sess.run(normalized_embeddings)

# Select a word and check how close the rest of the words are to the chosen word.
ref_word = normalized_embeddings_matrix[word_to_index_map["one"]]

cosine_dists = np.dot(normalized_embeddings_matrix, ref_word)
ff = np.argsort(cosine_dists)[::-1][1:10]
print("\n\nWORDS CLOSES TO ONE\n")
for f in ff:
    print(index_to_word_map[f])
    print(cosine_dists[f])
