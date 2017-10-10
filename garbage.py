import tensorflow as tf
import numpy as np

x = tf.constant(np.random.randint(10 , size=(2,3,4,2)))
shape = tf.shape(x)
y = tf.argmax(x, axis=3)

# print tf.ones(y.get_shape())
# u = tf.scatter_nd(tf.cast(y, tf.int32), tf.ones(y.get_shape()), [24, 2])
u = tf.one_hot(y, 2, 1.0, -1.0)

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    x_, y_, u_ = session.run([x, y, u])
    print x_
    print x_.shape
    print y_
    print y_.shape
    print u_
    print u_.shape