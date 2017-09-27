import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest


### Building Blocks ###

# fully_connected layer
def fully_connected(inputs,
                    num_outputs,
                    decay=0.999,
                    activation_fn=None,
                    is_training=True,
                    reuse=None,
                    scope=None):
  """Adds a fully connected layer.

  """
  if not isinstance(num_outputs, int):
    raise ValueError('num_outputs should be integer, got %s.', num_outputs)
  with tf.variable_scope(scope,
                         'fully_connected',
                         [inputs],
                         reuse=reuse) as sc:
    dtype = inputs.dtype.base_dtype
    num_input_units = inputs.get_shape()[-1].value

    static_shape = inputs.get_shape().as_list()
    static_shape[-1] = num_outputs

    out_shape = tf.unstack(tf.shape(inputs))
    out_shape[-1] = num_outputs

    weights_shape = [num_input_units, num_outputs]
    weights = tf.contrib.framework.model_variable('weights',
                                                  shape=weights_shape,
                                                  dtype=dtype,
                                                  initializer=tf.contrib.layers.xavier_initializer(),
                                                  collections=tf.GraphKeys.WEIGHTS,
                                                  trainable=True)
    biases = tf.contrib.framework.model_variable('biases',
                                                 shape=[num_outputs,],
                                                 dtype=dtype,
                                                 initializer=tf.zeros_initializer(),
                                                 collections=tf.GraphKeys.BIASES,
                                                 trainable=True)
    if len(static_shape) > 2:
      # Reshape inputs
      inputs = tf.reshape(inputs, [-1, num_input_units])
    outputs = tf.matmul(inputs, weights)
    moving_mean = tf.contrib.framework.model_variable('moving_mean',
                                                      shape=[num_outputs,],
                                                      dtype=dtype,
                                                      initializer=tf.zeros_initializer(),
                                                      trainable=False)
    if is_training:
      # Calculate the moments based on the individual batch.
      mean, _ = tf.nn.moments(outputs, [0], shift=moving_mean)
      # Update the moving_mean moments.
      update_moving_mean = tf.assign_sub(moving_mean, (moving_mean - mean) * (1.0 - decay))
      #tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
      outputs = outputs + biases
    else:
      outputs = outputs + biases
    if activation_fn:
      outputs = activation_fn(outputs)
    if len(static_shape) > 2:
      # Reshape back outputs
      outputs = tf.reshape(outputs, tf.stack(out_shape))
      outputs.set_shape(static_shape)
    return outputs

# convolutional layer
def convolution2d(inputs,
                  num_outputs,
                  kernel_size,
                  pool_size=None,
                  decay=0.999,
                  activation_fn=None,
                  is_training=True,
                  reuse=None,
                  scope=None):
  """Adds a 2D convolution followed by a maxpool layer.

  """
  with tf.variable_scope(scope,
                            'Conv',
                            [inputs],
                            reuse=reuse) as sc:
    dtype = inputs.dtype.base_dtype
    num_filters_in = inputs.get_shape()[-1].value
    weights_shape = list(kernel_size) + [num_filters_in, num_outputs]
    weights = tf.contrib.framework.model_variable(name='weights',
                                                  shape=weights_shape,
                                                  dtype=dtype,
                                                  initializer=tf.contrib.layers.xavier_initializer(),
                                                  collections=tf.GraphKeys.WEIGHTS,
                                                  trainable=True)
    biases = tf.contrib.framework.model_variable(name='biases',
                                                 shape=[num_outputs,],
                                                 dtype=dtype,
                                                 initializer=tf.zeros_initializer(),
                                                 collections=tf.GraphKeys.BIASES,
                                                 trainable=True)
    outputs = tf.nn.conv2d(inputs, weights, [1,1,1,1], padding='SAME')
    moving_mean = tf.contrib.framework.model_variable('moving_mean',
                                                      shape=[num_outputs,],
                                                      dtype=dtype,
                                                      initializer=tf.zeros_initializer(),
                                                      trainable=False)
    if is_training:
      # Calculate the moments based on the individual batch.
      mean, _ = tf.nn.moments(outputs, [0, 1, 2], shift=moving_mean)
      # Update the moving_mean moments.
      update_moving_mean = tf.assign_sub(moving_mean, (moving_mean - mean) * (1.0 - decay))
      #tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
      outputs = outputs + biases
    else:
      outputs = outputs + biases
    if pool_size:
      pool_shape = [1] + list(pool_size) + [1]
      outputs = tf.nn.max_pool(outputs, pool_shape, pool_shape, padding='SAME')
    if activation_fn:
      outputs = activation_fn(outputs)
    return outputs


### Regularization ###

def params_decay(decay):
  """ Add ops to decay weights and biases

  """
  params = tf.get_collection_ref(tf.GraphKeys.WEIGHTS) + tf.get_collection_ref(tf.GraphKeys.BIASES)
  while len(params) > 0:
    p = params.pop()
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,
        p.assign(decay*p + (1-decay)*tf.truncated_normal(p.get_shape(), stddev=0.01)))


### Nets ###

# ResDNN
def ResDNN(inputs,
           num_layers,
           decay=0.99999,
           activation_fn=tf.nn.relu,
           is_training=True,
           reuse=None,
           scope=None):
  """ a deep neural net with fully connected layers

  """
  with tf.variable_scope(scope,
                         "ResDNN",
                         [inputs],
                         reuse=reuse) as sc:
    size = inputs.get_shape()[-1].value
    outputs = inputs
    # residual layers
    for i in xrange(num_layers):
      outputs -= fully_connected(activation_fn(outputs), size, decay=decay, activation_fn=activation_fn,
          is_training=is_training)
    return outputs


# ResCNN
def ResCNN(inputs,
           num_layers,
           kernel_size,
           pool_size,
           pool_layers=1,
           decay=0.99999,
           activation_fn=tf.nn.relu,
           is_training=True,
           reuse=None,
           scope=None):
  """ a convolutaional neural net with conv2d and max_pool layers

  """
  with tf.variable_scope(scope,
                         "ResCNN",
                         [inputs],
                         reuse=reuse) as sc:
    size = inputs.get_shape()[-1].value
    if not pool_size:
      pool_layers = 0
    outputs = inputs
    # residual layers
    for j in xrange(pool_layers+1):
      if j > 0:
        pool_shape = [1] + list(pool_size) + [1]
        inputs = tf.nn.max_pool(outputs, pool_shape, pool_shape, padding='SAME')
        outputs = inputs
      with tf.variable_scope("layer{0}".format(j)) as sc:
        for i in xrange(num_layers):
          outputs -= convolution2d(activation_fn(outputs), size, kernel_size, decay=decay,
              activation_fn=activation_fn, is_training=is_training)
    return outputs
