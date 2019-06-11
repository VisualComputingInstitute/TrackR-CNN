import tensorflow as tf
import numpy as np

from network.Layer import Layer
from network.Util import conv2d, conv2d_dilated, get_activation, prepare_input, apply_dropout, max_pool,\
  max_pool3d, conv2d_transpose, conv3d


class Conv(Layer):
  output_layer = False

  def __init__(self, name, inputs, n_features, tower_setup, filter_size=(3, 3), old_order=False,
               strides=(1, 1), dilation=None, pool_size=(1, 1), pool_strides=None, activation="relu", dropout=0.0,
               batch_norm=False, bias=False, batch_norm_decay=Layer.BATCH_NORM_DECAY_DEFAULT, l2=Layer.L2_DEFAULT,
               padding="SAME", W_initializer=None, b_initializer=None, freeze_batchnorm_override=None):
    super(Conv, self).__init__()
    # mind the order of dropout, conv, activation and batchnorm!
    # batchnorm -> activation -> dropout -> conv -> pool
    # if old_order: dropout -> conv -> batchnorm -> activation -> pool (used for example in tensorpack)
    curr, n_features_inp = prepare_input(inputs)
    filter_size = list(filter_size)
    strides = list(strides)
    pool_size = list(pool_size)
    if pool_strides is None:
      pool_strides = pool_size

    with tf.variable_scope(name):
      if isinstance(W_initializer, list):
        W_initializer = tf.constant_initializer(W_initializer)
      W = self.create_weight_variable("W", filter_size + [n_features_inp, n_features], l2, tower_setup,
                                      initializer=W_initializer)
      b = None
      if bias:
        if b_initializer is not None:
          b_initializer = tf.constant_initializer(b_initializer)
          b = self.create_bias_variable("b", [n_features], tower_setup, initializer=b_initializer)
        else:
          b = self.create_bias_variable("b", [n_features], tower_setup)

      if old_order:
        curr = apply_dropout(curr, dropout)
        if dilation is None:
          curr = conv2d(curr, W, strides, padding=padding)
        else:
          curr = conv2d_dilated(curr, W, dilation, padding=padding)
        if bias:
          curr += b
        if batch_norm:
          curr = self.create_and_apply_batch_norm(curr, n_features, batch_norm_decay, tower_setup, freeze_batchnorm_override=freeze_batchnorm_override)
        curr = get_activation(activation)(curr)
      else:
        if batch_norm:
          curr = self.create_and_apply_batch_norm(curr, n_features_inp, batch_norm_decay, tower_setup, freeze_batchnorm_override=freeze_batchnorm_override)
        curr = get_activation(activation)(curr)
        curr = apply_dropout(curr, dropout)
        if dilation is None:
          curr = conv2d(curr, W, strides, padding=padding)
        else:
          curr = conv2d_dilated(curr, W, dilation, padding=padding)
        if bias:
          curr += b

      if pool_size != [1, 1]:
        curr = max_pool(curr, pool_size, pool_strides)
    self.outputs = [curr]


class ConvTranspose(Layer):
  output_layer = False

  def __init__(self, name, inputs, n_features, tower_setup, filter_size=(3, 3),
               strides=(1, 1), activation="relu",  # dropout=0.0,
               batch_norm=False, bias=False, batch_norm_decay=Layer.BATCH_NORM_DECAY_DEFAULT, l2=Layer.L2_DEFAULT,
               padding="SAME", W_initializer=None, freeze_batchnorm_override=None):
    super(ConvTranspose, self).__init__()
    # TODO this uses the tensorpack order of operations
    curr, n_features_inp = prepare_input(inputs)
    filter_size = list(filter_size)
    strides = list(strides)

    with tf.variable_scope(name):
      W = self.create_weight_variable("W", filter_size + [n_features, n_features_inp], l2, tower_setup,
                                      initializer=W_initializer)
      b = None
      if bias:
        b = self.create_bias_variable("b", [n_features], tower_setup)

      # curr = apply_dropout(curr, dropout)
      curr = conv2d_transpose(curr, W, strides, padding=padding)
      if bias:
        curr = tf.nn.bias_add(curr, b)
      if batch_norm:
        curr = self.create_and_apply_batch_norm(curr, n_features, batch_norm_decay, tower_setup,
                                                freeze_batchnorm_override=freeze_batchnorm_override)
      curr = get_activation(activation)(curr)
    self.outputs = [curr]


class ConvForOutput(Layer):
  output_layer = False

  def __init__(self, name, inputs, dataset, n_features, tower_setup, filter_size=(1, 1),
               input_activation=None, dilation=None, l2=Layer.L2_DEFAULT, dropout=0.0):
    super().__init__()
    if n_features == -1:
      n_features = dataset.num_classes()
    filter_size = list(filter_size)
    inp, n_features_inp = prepare_input(inputs)
    if input_activation is not None:
      inp = get_activation(input_activation)(inp)
    inp = apply_dropout(inp, dropout)

    with tf.variable_scope(name):
      W = self.create_weight_variable("W", filter_size + [n_features_inp, n_features], l2, tower_setup)
      b = self.create_bias_variable("b", [n_features], tower_setup)
      if dilation is None:
        output = conv2d(inp, W) + b
      else:
        output = conv2d_dilated(inp, W, dilation) + b
      self.outputs = [output]


class ResidualUnit(Layer):
  output_layer = False

  def __init__(self, name, inputs, tower_setup, n_convs=2, n_features=None, dilations=None, strides=None,
               filter_size=None, activation="relu", dropout=0.0, batch_norm_decay=Layer.BATCH_NORM_DECAY_DEFAULT,
               l2=Layer.L2_DEFAULT):
    super().__init__()
    curr, n_features_inp = prepare_input(inputs)
    res = curr
    assert n_convs >= 1, n_convs

    if dilations is not None:
      assert strides is None
    elif strides is None:
      strides = [[1, 1]] * n_convs
    if filter_size is None:
      filter_size = [[3, 3]] * n_convs
    if n_features is None:
      n_features = n_features_inp
    if not isinstance(n_features, list):
      n_features = [n_features] * n_convs

    with tf.variable_scope(name):
      curr = self.create_and_apply_batch_norm(curr, n_features_inp, batch_norm_decay, tower_setup, "bn0")
      curr = get_activation(activation)(curr)
      if tower_setup.is_training:
        curr = apply_dropout(curr, dropout)

      if strides is None:
        strides_res = [1, 1]
      else:
        strides_res = np.prod(strides, axis=0).tolist()
      if (n_features[-1] != n_features_inp) or (strides_res != [1, 1]):
        W0 = self.create_weight_variable("W0", [1, 1] + [n_features_inp, n_features[-1]], l2, tower_setup)
        if dilations is None:
          res = conv2d(curr, W0, strides_res)
        else:
          res = conv2d(curr, W0)

      W1 = self.create_weight_variable("W1", filter_size[0] + [n_features_inp, n_features[0]], l2, tower_setup)
      if dilations is None:
        curr = conv2d(curr, W1, strides[0])
      else:
        curr = conv2d_dilated(curr, W1, dilations[0])
      for idx in range(1, n_convs):
        curr = self.create_and_apply_batch_norm(curr, n_features[idx - 1], batch_norm_decay,
                                                tower_setup, "bn" + str(idx + 1))
        curr = get_activation(activation)(curr)
        Wi = self.create_weight_variable("W" + str(idx + 1), filter_size[idx] + [n_features[idx - 1], n_features[idx]],
                                         l2, tower_setup)
        if dilations is None:
          curr = conv2d(curr, Wi, strides[idx])
        else:
          curr = conv2d_dilated(curr, Wi, dilations[idx])

    curr += res
    self.outputs = [curr]


class Upsampling(Layer):
  def __init__(self, name, inputs, tower_setup, n_features, concat, activation="relu", filter_size=(3, 3),
               l2=Layer.L2_DEFAULT):
    super(Upsampling, self).__init__()
    filter_size = list(filter_size)
    assert isinstance(concat, list)
    assert len(concat) > 0
    curr, n_features_inp = prepare_input(inputs)
    concat_inp, n_features_concat = prepare_input(concat)

    curr = tf.image.resize_nearest_neighbor(curr, tf.shape(concat_inp)[1:3])
    curr = tf.concat([curr, concat_inp], axis=3)
    n_features_curr = n_features_inp + n_features_concat

    with tf.variable_scope(name):
      W = self.create_weight_variable("W", filter_size + [n_features_curr, n_features], l2, tower_setup)
      b = self.create_bias_variable("b", [n_features], tower_setup)
      curr = conv2d(curr, W) + b
      curr = get_activation(activation)(curr)

    self.outputs = [curr]


class Conv3DOverBatch(Layer):
  output_layer = False

  def __init__(self, name, inputs, n_features, tower_setup, filter_size=(3, 3, 3), old_order=False,
               strides=(1, 1, 1), pool_size=(1, 1, 1), pool_strides=None, activation="relu", dropout=0.0,
               batch_norm=False, bias=False, batch_norm_decay=Layer.BATCH_NORM_DECAY_DEFAULT, l2=Layer.L2_DEFAULT,
               padding="SAME", init_type="random"):
    super(Conv3DOverBatch, self).__init__()
    # mind the order of dropout, conv, activation and batchnorm!
    # batchnorm -> activation -> dropout -> conv -> pool
    # if old_order: dropout -> conv -> batchnorm -> activation -> pool (used for example in tensorpack)
    curr, n_features_inp = prepare_input(inputs)
    filter_size = list(filter_size)
    strides = list(strides)
    pool_size = list(pool_size)
    if pool_strides is None:
      pool_strides = pool_size

    # Introduce a fake "batch dimension", previous dimension becomes time
    curr = tf.expand_dims(curr, axis=0)

    with tf.variable_scope(name):
      if init_type == "identity":
        assert n_features == n_features_inp
        identity_filter = np.zeros(filter_size, dtype=np.float32)
        identity_filter[int(filter_size[0]/2.0), int(filter_size[1]/2.0), int(filter_size[2]/2.0)] = 1.0
        init_weights = np.zeros(list(filter_size) + [n_features_inp, n_features], dtype=np.float32)
        for i in range(n_features):
          init_weights[:, :, :, i, i] = identity_filter
        W = self.create_weight_variable("W", filter_size + [n_features_inp, n_features], l2, tower_setup, initializer=init_weights)
      elif init_type == "avgtime":
        assert n_features == n_features_inp
        avgtime_filter = np.zeros(filter_size, dtype=np.float32)
        avgtime_filter[:, int(filter_size[1]/2.0), int(filter_size[2]/2.0)] = 1.0/filter_size[0]
        init_weights = np.zeros(list(filter_size) + [n_features_inp, n_features], dtype=np.float32)
        for i in range(n_features):
          init_weights[:, :, :, i, i] = avgtime_filter  # Maybe we'ld like sth else than per-channel averaging?
        W = self.create_weight_variable("W", filter_size + [n_features_inp, n_features], l2, tower_setup, initializer=init_weights)
      else:  # random
        W = self.create_weight_variable("W", filter_size + [n_features_inp, n_features], l2, tower_setup)
      b = None
      if bias:
        b = self.create_bias_variable("b", [n_features], tower_setup)

      if old_order:
        curr = apply_dropout(curr, dropout)
        curr = conv3d(curr, W, strides, padding=padding)
        if bias:
          curr += b
        if batch_norm:
          curr = self.create_and_apply_batch_norm(curr, n_features, batch_norm_decay, tower_setup)
        curr = get_activation(activation)(curr)
      else:
        if batch_norm:
          curr = self.create_and_apply_batch_norm(curr, n_features_inp, batch_norm_decay, tower_setup)
        curr = get_activation(activation)(curr)
        curr = apply_dropout(curr, dropout)
        curr = conv3d(curr, W, strides, padding=padding)
        if bias:
          curr += b

      if pool_size != [1, 1, 1]:
        curr = max_pool3d(curr, pool_size, pool_strides)

    # Remove the fake dimension again
    curr = tf.squeeze(curr, axis=0)

    self.outputs = [curr]


class SepConv3DOverBatch(Layer):
  output_layer = False

  def __init__(self, name, inputs, n_features, tower_setup, filter_size=(3, 3, 3), old_order=False,
               strides=(1, 1, 1), pool_size=(1, 1, 1), pool_strides=None, activation="relu", dropout=0.0,
               batch_norm=False, bias=False, batch_norm_decay=Layer.BATCH_NORM_DECAY_DEFAULT, l2=Layer.L2_DEFAULT,
               padding="SAME", init_type="random"):
    super(SepConv3DOverBatch, self).__init__()
    # mind the order of dropout, conv, activation and batchnorm!
    # batchnorm -> activation -> dropout -> conv -> pool
    # if old_order: dropout -> conv -> batchnorm -> activation -> pool (used for example in tensorpack)
    curr, n_features_inp = prepare_input(inputs)
    filter_size = list(filter_size)
    strides = list(strides)
    pool_size = list(pool_size)
    if pool_strides is None:
      pool_strides = pool_size

    assert not batch_norm
    assert dropout == 0.0
    assert old_order

    # Introduce a fake "batch dimension", previous dimension becomes time
    curr = tf.expand_dims(curr, axis=0)

    with tf.variable_scope(name):
      # Depthwise conv3d
      curr_split = tf.split(curr, n_features_inp, axis=-1)

      if init_type == "identity":
        filter_initializer = np.zeros(filter_size + [1, 1], dtype=np.float32)
        filter_initializer[int(filter_size[0]/2.0), int(filter_size[1]/2.0), int(filter_size[2]/2.0), 0, 0] = 1.0
      elif init_type == "avgtime":
        filter_initializer = np.zeros(filter_size + [1, 1], dtype=np.float32)
        filter_initializer[:, int(filter_size[1]/2.0), int(filter_size[2]/2.0), 0, 0] = 1.0/filter_size[0]
      else:  # random
        filter_initializer = None

      for channel in range(n_features_inp):
        with tf.variable_scope("channel_" + str(channel)):
          curr = curr_split[channel]
          W = self.create_weight_variable("W", filter_size + [1, 1], l2, tower_setup, initializer=filter_initializer)
          b = None
          if bias:
            b = self.create_bias_variable("b", [n_features], tower_setup)

          curr = conv3d(curr, W, strides, padding=padding)
          if bias:
            curr += b
          curr_split[channel] = curr

      # 1x1 conv3d over all channels
      curr = tf.concat(curr_split, axis=-1)
      if init_type == "identity" or init_type == "avgtime":
        assert n_features == n_features_inp
        init_weights = np.zeros([1, 1, 1, n_features_inp, n_features], dtype=np.float32)
        for i in range(n_features):
          init_weights[:, :, :, i, i] = 1.0
        W = self.create_weight_variable("W", [1, 1, 1, n_features_inp, n_features], l2, tower_setup, initializer=init_weights)
      else:  # random
        W = self.create_weight_variable("W", [1, 1, 1, n_features_inp, n_features], l2, tower_setup)
      b = None
      if bias:
        b = self.create_bias_variable("b", [n_features], tower_setup)

      curr = conv3d(curr, W, strides, padding=padding)
      if bias:
        curr += b
      curr = get_activation(activation)(curr)

      if pool_size != [1, 1, 1]:
        curr = max_pool3d(curr, pool_size, pool_strides)

    # Remove the fake dimension again
    curr = tf.squeeze(curr, axis=0)

    self.outputs = [curr]


class AveragePooling(Layer):
  def __init__(self, name, inputs):
    super().__init__()
    assert len(inputs) == 1
    inp = inputs[0]
    out = tf.reduce_mean(inp, axis=[1,2])
    self.outputs = [out]


class Collapse(Layer):
  def __init__(self, name, inputs):
    super().__init__()
    assert len(inputs) == 1
    inp = inputs[0]
    if inp.shape[0].value is None:
      out_dim = np.prod(inp.shape[1:].as_list())
      out = tf.reshape(inp, [-1, out_dim])
    else:
      out = tf.reshape(inp, [tf.shape(inp)[0], -1])
    self.outputs = [out]


class MaxPool(Layer):
  def __init__(self, name, inputs, pool_size=(2, 2), pool_strides=None):
    super().__init__()
    assert len(inputs) == 1
    pool_size = list(pool_size)
    inp = inputs[0]
    out = max_pool(inp, pool_size, pool_strides)
    self.outputs = [out]
