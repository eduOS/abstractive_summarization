import tensorflow as tf
import numpy



a = numpy.array([[0.8, 0.1, 0.05, 0.05, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005]])
log_prob = tf.log(tf.nn.softmax(a))
b = tf.multinomial(log_prob, 5)
c = tf.multinomial(log_prob, 5)
d = tf.multinomial(log_prob, 5)
e = tf.multinomial(log_prob, 5)
f = tf.multinomial(log_prob, 5)
g = tf.multinomial(log_prob, 5)
h = tf.multinomial(log_prob, 5)


with tf.Session() as sess:
    print(sess.run([b,c,d,e,f,g,h]))
