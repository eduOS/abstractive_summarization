import tensorflow as tf


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
  with tf.variable_scope(scope, 'Conv', [inputs], reuse=reuse):
    dtype = inputs.dtype.base_dtype
    num_filters_in = inputs.get_shape()[-1].value
    weights_shape = [1] + list(kernel_size) + [num_filters_in, num_outputs]
    # 1, 3, emb_dim, emb_dim
    weights = tf.get_variable(name='weights',
                              shape=weights_shape,
                              dtype=dtype,
                              initializer=tf.contrib.layers.xavier_initializer(),
                              collections=tf.GraphKeys.WEIGHTS,
                              trainable=True)
    biases = tf.get_variable(name='biases',
                             shape=[num_outputs, ],
                             dtype=dtype,
                             initializer=tf.zeros_initializer(),
                             collections=tf.GraphKeys.BIASES,
                             trainable=True)
    outputs = tf.nn.conv2d(inputs, weights, [1, 1, 1, 1], padding='SAME')
    if is_training:
      outputs = outputs + biases
    else:
      outputs = outputs + biases
    if pool_size:
      pool_shape = [1, 1] + list(pool_size) + [1]
      outputs = tf.nn.max_pool(outputs, pool_shape, pool_shape, padding='SAME')
    if activation_fn:
      outputs = activation_fn(outputs)
    return outputs


def params_decay(decay):
  """ Add ops to decay weights and biases

  """
  params = tf.get_collection_ref(tf.GraphKeys.WEIGHTS) + tf.get_collection_ref(tf.GraphKeys.BIASES)
  while len(params) > 0:
    p = params.pop()
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,
                         p.assign(decay*p + (1-decay)*tf.truncated_normal(p.get_shape(), stddev=0.01)))


# ResCNN
def CResCNN(inputs, conditions, conv_layers, kernel_size, pool_size, pool_layers=1,
            decay=0.99999, activation_fn=tf.nn.relu, is_training=True, reuse=None, scope=None):
  """ a convolutaional neural net with conv2d and max_pool layers

  """
  with tf.variable_scope(scope, "CResCNN", [inputs], reuse=reuse):
    layer_size = inputs.get_shape()[-1].value
    if not pool_size:
      pool_layers = 0
    i_outputs = inputs
    # residual layers
    for j in range(pool_layers):
      if j > 0:
        pool_shape = [1, 1] + list(pool_size) + [1]
        # 1, 1, 2, 1
        inputs = tf.nn.max_pool(i_outputs, pool_shape, pool_shape, padding='SAME')
        i_outputs = inputs
      with tf.variable_scope("input_layer{0}".format(j)):
        for i in range(conv_layers):
          i_outputs -= convolution2d(activation_fn(i_outputs), layer_size, kernel_size, decay=decay,
                                     activation_fn=activation_fn, is_training=is_training)
    # maybe dropout is useful
    # squeeze the highth dimension
    i_outputs = tf.squeeze(i_outputs, [1])
    # make the embedding sequence to be only one embedding
    inputs_emb = tf.reduce_max(i_outputs, axis=1)

    c_outputs = conditions
    for j in range(pool_layers*2):
      if j > 0:
        pool_shape = [1, 1] + list(pool_size*2) + [1]
        # 1, 1, 2, 1
        conditions = tf.nn.max_pool(c_outputs, pool_shape, pool_shape, padding='SAME')
        c_outputs = conditions
      with tf.variable_scope("condition_layer{0}".format(j)):
        for i in range(conv_layers*2):
          c_outputs -= convolution2d(activation_fn(c_outputs), layer_size, kernel_size, decay=decay,
                                     activation_fn=activation_fn, is_training=is_training)

    c_outputs = tf.squeeze(c_outputs, [1])
    conditions_emb = tf.reduce_max(c_outputs, axis=1)

    # concatenate the two embeding and make the embedding to be twice long
    return tf.concat(1, [inputs_emb, conditions_emb])
