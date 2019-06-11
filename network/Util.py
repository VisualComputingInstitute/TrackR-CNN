import tensorflow as tf
from tensorflow.python.layers.utils import deconv_output_length
from core.Util import smart_shape


def conv2d(x, W, strides=(1, 1), padding="SAME"):
  strides = list(strides)
  return tf.nn.conv2d(x, W, strides=[1] + strides + [1], padding=padding)


def conv3d(x, W, strides=(1, 1), padding="SAME"):
  strides = list(strides)
  return tf.nn.conv3d(x, W, strides=[1] + strides + [1], padding=padding)


def conv2d_transpose(x, W, strides=(1, 1), padding="SAME"):
  strides = list(strides)

  # Compute output size, cf. tf.layers.Conv2DTranspose
  W_shape = tf.shape(W)
  inputs_shape = tf.shape(x)

  # Infer the dynamic output shape:
  out_height = deconv_output_length(inputs_shape[1], W_shape[0], padding, strides[0])
  out_width = deconv_output_length(inputs_shape[2], W_shape[1], padding, strides[1])
  output_shape = (inputs_shape[0], out_height, out_width, W_shape[2])

  return tf.nn.conv2d_transpose(x, W, tf.stack(output_shape), strides=[1] + strides + [1], padding=padding)


def conv2d_dilated(x, W, dilation, padding="SAME"):
  res = tf.nn.atrous_conv2d(x, W, dilation, padding=padding)
  shape = x.get_shape().as_list()
  shape[-1] = W.get_shape().as_list()[-1]
  res.set_shape(shape)
  return res


def create_batch_norm_vars(n_out, tower_setup, scope_name="bn"):
  with tf.device(tower_setup.variable_device), tf.variable_scope(scope_name):
    initializer_zero = tf.constant_initializer(0.0, dtype=tf.float32)
    beta = tf.get_variable("beta", [n_out], tf.float32, initializer_zero)
    initializer_gamma = tf.constant_initializer(1.0, dtype=tf.float32)
    gamma = tf.get_variable("gamma", [n_out], tf.float32, initializer_gamma)
    mean_ema = tf.get_variable("mean_ema", [n_out], tf.float32, initializer_zero, trainable=False)
    var_ema = tf.get_variable("var_ema", [n_out], tf.float32, initializer_zero, trainable=False)
    return beta, gamma, mean_ema, var_ema


_activations = {"relu": tf.nn.relu, "linear": lambda x: x, "elu": tf.nn.elu}


def get_activation(act_str):
  assert act_str.lower() in _activations, "Unknown activation function " + act_str
  return _activations[act_str.lower()]


def prepare_input(inputs):
  if len(inputs) == 1:
    inp = inputs[0]
    dim = int(inp.get_shape()[-1])
  else:
    dims = [int(inp.get_shape()[3]) for inp in inputs]
    dim = sum(dims)
    inp = tf.concat(inputs, axis=3)
  return inp, dim


def apply_dropout(inp, dropout):
  if dropout == 0.0:
    return inp
  else:
    keep_prob = 1.0 - dropout
    return tf.nn.dropout(inp, keep_prob)


def max_pool(x, shape, strides=None, padding="SAME"):
  if strides is None:
    strides = shape
  return tf.nn.max_pool(x, ksize=[1] + shape + [1],
                        strides=[1] + strides + [1], padding=padding)


def max_pool3d(x, shape, strides=None, padding="SAME"):
  if strides is None:
    strides = shape
  return tf.nn.max_pool3d(x, ksize=[1] + shape + [1],
                          strides=[1] + strides + [1], padding=padding)


def bootstrapped_ce_loss(raw_ce, fraction, n_valid_pixels_per_im):
  ks = tf.maximum(tf.cast(tf.round(tf.cast(n_valid_pixels_per_im, tf.float32) * fraction), tf.int32), 1)

  def bootstrapped_ce_for_one_img(args):
    one_ce, k = args
    hardest = tf.nn.top_k(tf.reshape(one_ce, [-1]), k, sorted=False)[0]
    return tf.reduce_mean(hardest)
  loss_per_im = tf.map_fn(bootstrapped_ce_for_one_img, [raw_ce, ks], dtype=tf.float32)
  return tf.reduce_mean(loss_per_im)


def class_balanced_ce_loss(raw_ce, targets, n_classes):
  # for now we do the balancing for each image individually.
  # we could also balance over the whole mini-batch though

  def class_balanced_ce_for_one_img(args):
    ce, target = args
    cls_losses = []
    for cls in range(n_classes):
      cls_mask = tf.equal(target, cls)
      n_cls = tf.reduce_sum(tf.cast(cls_mask, tf.int32))
      cls_loss = tf.reduce_sum(tf.boolean_mask(ce, cls_mask)) / tf.cast(tf.maximum(n_cls, 1), tf.float32)
      cls_losses.append(cls_loss)
    return tf.add_n(cls_losses) / n_classes

  loss_per_im = tf.map_fn(class_balanced_ce_for_one_img, [raw_ce, targets], dtype=tf.float32)
  return tf.reduce_mean(loss_per_im)
