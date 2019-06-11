import tensorflow as tf

from core import Extractions
from network.Layer import Layer
from network.Util import prepare_input, smart_shape
from network.layer_implementations.ConvLSTMCell import ConvLSTMCell, dynamic_conv_rnn, LSTMStateTuple


class ConvLSTM(Layer):
  def __init__(self, name, inputs, tower_setup, n_features, filter_size=(5, 5), l2=Layer.L2_DEFAULT):
    super(ConvLSTM, self).__init__()
    filter_size = list(filter_size)
    inp, n_features_inp = prepare_input(inputs)
    shape = smart_shape(inp)
    batch_size = 1
    height = shape[1]
    width = shape[2]
    # input is assumed to have shape (time, height, width, feature)
    # we need to introduce a batch 1 dimension -> (time, batch=1, height, width, feature)
    inp = inp[:, tf.newaxis]

    with tf.variable_scope(name):
      # TODO: maybe we need a different weight init here because we don't use relu
      W = self.create_weight_variable("W_lstm", filter_size + [n_features_inp + n_features, 4 * n_features],
                                      l2, tower_setup)
      b = self.create_bias_variable("b_lstm", [4 * n_features], tower_setup)
      lstm = ConvLSTMCell(n_features, height, width, filter_size, W=W, b=b)
      initial_state = lstm.zero_state(batch_size, tower_setup.dtype)
      if not tower_setup.is_training:
        c0 = tf.placeholder_with_default(initial_state[0], initial_state[0].get_shape(), "c0_placeholder")
        h0 = tf.placeholder_with_default(initial_state[1], initial_state[1].get_shape(), "h0_placeholder")
        self.placeholders.append(c0)
        self.placeholders.append(h0)
        initial_state = LSTMStateTuple(c0, h0)
      y, (c, h) = dynamic_conv_rnn(lstm, inp, initial_state=initial_state, time_major=True)
      y = tf.reshape(y, shape[:-1] + [n_features])
    self.outputs.append(y)
    self.extractions[Extractions.RECURRENT_STATE] = (c, h)


class BidirectionalConvLSTM(Layer):
  def __init__(self, name, inputs, tower_setup, n_features, filter_size=(5, 5), l2=Layer.L2_DEFAULT):
    super(BidirectionalConvLSTM, self).__init__()
    filter_size = list(filter_size)
    inp, n_features_inp = prepare_input(inputs)
    shape = smart_shape(inp)
    batch_size = 1
    height = shape[1]
    width = shape[2]
    # input is assumed to have shape (time, height, width, feature)
    # we need to introduce a batch 1 dimension -> (time, batch=1, height, width, feature)
    inp = inp[:, tf.newaxis]

    with tf.variable_scope(name):
      inp_fwd = inp
      inp_bwd = tf.reverse(inp, axis=[0])
      W_fwd = self.create_weight_variable("W_lstm_fwd", filter_size + [n_features_inp + n_features, 4 * n_features],
                                          l2, tower_setup)
      W_bwd = self.create_weight_variable("W_lstm_bwd", filter_size + [n_features_inp + n_features, 4 * n_features],
                                          l2, tower_setup)
      b_fwd = self.create_bias_variable("b_lstm_fwd", [4 * n_features], tower_setup)
      b_bwd = self.create_bias_variable("b_lstm_bwd", [4 * n_features], tower_setup)
      lstm_fwd = ConvLSTMCell(n_features, height, width, filter_size, W=W_fwd, b=b_fwd)
      lstm_bwd = ConvLSTMCell(n_features, height, width, filter_size, W=W_bwd, b=b_bwd)
      initial_state_fwd = lstm_fwd.zero_state(batch_size, tower_setup.dtype)
      initial_state_bwd = lstm_bwd.zero_state(batch_size, tower_setup.dtype)
      y_fwd, _ = dynamic_conv_rnn(lstm_fwd, inp_fwd, initial_state=initial_state_fwd, time_major=True)
      y_bwd, _ = dynamic_conv_rnn(lstm_bwd, inp_bwd, initial_state=initial_state_bwd, time_major=True)
      y_bwd = tf.reverse(y_bwd, axis=[0])
      y_fwd = tf.reshape(y_fwd, shape[:-1] + [n_features])
      y_bwd = tf.reshape(y_bwd, shape[:-1] + [n_features])
      y = tf.concat([y_fwd, y_bwd], axis=-1)
    self.outputs.append(y)
