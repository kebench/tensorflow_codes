import tensorflow as tf

# matrix
A = tf.constant([[1,2,3],
                [4,5,6]])

print(A.get_shape())

# vector
x = tf.constant([[1],[0],[1]])
print(x.get_shape())

# transform the 1D vector to a 2D single-column matrix
x = tf.expand_dims(x, 1)
# x = tf.constant([[1],[0],[1]]) <== same result
# print(x.get_shape())
# matrix mul
b = tf.matmul(A,x)

sess = tf.InteractiveSession()
print("\nMatmul Result: {}".format(b.eval()))
sess.close()