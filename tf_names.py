import tensorflow as tf

with tf.Graph().as_default():
    c1 = tf.constant(4, dtype=tf.float64, name='c')
    c2 = tf.constant(5, dtype=tf.int32, name='c') #will have underscore and a number concatenated

print(c1.name)
print(c2.name)

#name scopes
#group together nodes by name
with tf.Graph().as_default():
    c1 = tf.constant(4, dtype=tf.float64, name='c')
    
    #group the objects. This will add a prefix before the name
    #useful when dividing graphs and assigning semantic meaning to a group of nodes
    with tf.name_scope("prefix_name"):
        c2 = tf.constant(4, dtype=tf.int32, name='c')
        c3 = tf.constant(4, dtype=tf.float64, name='c')

print(c1.name)
print(c2.name)
print(c3.name)