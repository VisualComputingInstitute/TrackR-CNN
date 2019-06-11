import tensorflow as tf
from network.Util import smart_shape
RNNCell = tf.nn.rnn_cell.RNNCell
LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple


def _conv2d(x, W, strides=None):
  if strides is None:
    strides = [1, 1]
  return tf.nn.conv2d(x, W, strides=[1] + strides + [1], padding="SAME")


def dynamic_conv_rnn(cell, inputs, sequence_length=None, initial_state=None,
                     dtype=None, parallel_iterations=None, swap_memory=False,
                     time_major=False, scope=None):
  # inputs should have shape (time, batch, height, width, feature)
  input_shape = smart_shape(inputs)
  num_units = cell.num_units()
  h, final_state = tf.nn.dynamic_rnn(cell, inputs, sequence_length, initial_state, dtype, parallel_iterations,
                                     swap_memory, time_major, scope)
  h = tf.reshape(h, tf.stack([input_shape[0], input_shape[1], input_shape[2], input_shape[3], num_units]))
  return h, final_state


# similar to https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/rnn/python/ops/core_rnn_cell_impl.py
# for maximal flexibility we allow to pass the weights externally
class ConvLSTMCell(RNNCell):
  def __init__(self, num_units, height, width, filter_size, forget_bias=1.0, activation=tf.tanh, W=None, b=None):
    self._num_units = num_units
    self._height = height
    self._width = width
    self._size = num_units * height * width
    self._forget_bias = forget_bias
    self._activation = activation
    self._filter_size = list(filter_size)
    if W is not None:
      W_shape = W.get_shape().as_list()
      assert len(W_shape) == 4
      assert W_shape[:2] == self._filter_size
      assert W_shape[-1] == 4 * self._num_units
      self._W = W
    else:
      self._W = None
    if b is not None:
      b_shape = b.get_shape().as_list()
      assert len(b_shape) == 1
      assert b_shape[0] == 4 * self._num_units
      self._b = b
    else:
      self._b = None

  def __call__(self, inputs, state, scope=None):
    #inputs: `2-D` tensor with shape `[batch_size x input_size]`.
    #state:  tuple with shapes `[batch_size x s] for s in self.state_size
    with tf.variable_scope(scope or type(self).__name__):  # "ConvLSTMCell"
      # Parameters of gates are concatenated into one multiply for efficiency.
      c, h = state
      concat = self._conv(inputs, h)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = tf.split(axis=3, num_or_size_splits=4, value=concat)
      batch = inputs.get_shape().as_list()[0]
      if batch is None:
        batch = tf.shape(inputs)[0]
      i, j, f, o = [tf.reshape(x, [batch, -1]) for x in [i, j, f, o]]

      new_c = (c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) *
               self._activation(j))
      new_h = self._activation(new_c) * tf.sigmoid(o)

      new_state = LSTMStateTuple(new_c, new_h)
      return new_h, new_state

  def _conv(self, inputs, h):
    batch = inputs.get_shape().as_list()[0]
    if batch is None:
      batch = tf.shape(inputs)[0]
    n_input_features = inputs.get_shape().as_list()[-1]

    #inputs = tf.reshape(inputs, [batch, self._height, self._width, n_input_features])
    h = tf.reshape(h, [batch, self._height, self._width, self._num_units])
    inp = tf.concat([inputs, h], axis=3)

    if self._W is not None:
      W = self._W
      assert W.get_shape().as_list()[2] == n_input_features + self._num_units
    else:
      W = tf.get_variable("W", shape=(self._filter_size + [n_input_features + self._num_units, 4 * self._num_units]))
    if self._b is not None:
      b = self._b
    else:
      zero_initializer = tf.constant_initializer(0.0, dtype=inputs.dtype)
      b = tf.get_variable("b", shape=(4 * self._num_units), initializer=zero_initializer)
    y = _conv2d(inp, W) + b
    return y

  def num_units(self):
    return self._num_units

  @property
  def state_size(self):
    return LSTMStateTuple(self._size, self._size)

  @property
  def output_size(self):
    return self._size
