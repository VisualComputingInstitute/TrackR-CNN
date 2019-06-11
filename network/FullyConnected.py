import tensorflow as tf

from network.Layer import Layer
from network.Util import get_activation, prepare_input, apply_dropout


# See also savitar1
class FullyConnected(Layer):
  def __init__(self, name, inputs, n_features, tower_setup, activation="relu", dropout=0.0, batch_norm=False,
               batch_norm_decay=Layer.BATCH_NORM_DECAY_DEFAULT, l2=Layer.L2_DEFAULT, W_initializer=None):
    super(FullyConnected, self).__init__()
    inp, n_features_inp = prepare_input(inputs)

    with tf.variable_scope(name):
      inp = apply_dropout(inp, dropout)
      if batch_norm:
        inp = tf.expand_dims(inp, axis=0)
        inp = tf.expand_dims(inp, axis=0)
        inp = self.create_and_apply_batch_norm(inp, n_features_inp, batch_norm_decay, tower_setup)
        inp = tf.squeeze(inp, axis=[0, 1])
      W = self.create_weight_variable("W", [n_features_inp, n_features], l2, tower_setup, initializer=W_initializer)
      b = self.create_bias_variable("b", [n_features], tower_setup)
      z = tf.matmul(inp, W) + b
      h = get_activation(activation)(z)
    self.outputs = [h]
    self.n_features = n_features
