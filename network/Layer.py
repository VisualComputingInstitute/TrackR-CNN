import tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages

from network.Util import create_batch_norm_vars


class Layer:
  BATCH_NORM_DECAY_DEFAULT = 0.95
  BATCH_NORM_EPSILON = 1e-5
  L2_DEFAULT = 1e-4

  def __init__(self):
    self.summaries = []
    self.regularizers = []
    self.losses = []
    self.update_ops = []
    self.outputs = []
    self.placeholders = []
    self.measures = {}
    self.extractions = {}
    self.n_params = 0

  def add_scalar_summary(self, op, name):
    summary = tf.summary.scalar(name, op)
    self.summaries.append(summary)

  def add_image_summary(self, im, name):
    summary = tf.summary.image(name, im)
    self.summaries.append(summary)

  def create_and_apply_batch_norm(self, inp, n_features, decay, tower_setup, scope_name="bn", freeze_batchnorm_override=None):
    beta, gamma, moving_mean, moving_var = create_batch_norm_vars(n_features, tower_setup, scope_name)
    self.n_params += 2 * n_features
    if tower_setup.is_main_train_tower:
      assert tower_setup.is_training
    if tower_setup.is_training and\
      (not tower_setup.freeze_batchnorm or (freeze_batchnorm_override is not None and freeze_batchnorm_override)):
        xn, batch_mean, batch_var = tf.nn.fused_batch_norm(inp, gamma, beta, epsilon=Layer.BATCH_NORM_EPSILON,
                                                           is_training=True)
        if tower_setup.is_main_train_tower:
          update_op1 = moving_averages.assign_moving_average(
            moving_mean, batch_mean, decay, zero_debias=False, name='mean_ema_op')
          update_op2 = moving_averages.assign_moving_average(
            moving_var, batch_var, decay, zero_debias=False, name='var_ema_op')
          self.update_ops.append(update_op1)
          self.update_ops.append(update_op2)
        return xn
    else:
      # According to the tensorpack code, this updates gamma and beta but not the moving averages
      # Fused version should have better performance.
      xn, _, _ = tf.nn.fused_batch_norm(inp, gamma, beta, moving_mean, moving_var, Layer.BATCH_NORM_EPSILON, is_training=False)
      return xn

  def create_weight_variable(self, name, shape, l2, tower_setup, trainable=True, initializer=None):
    with tf.device(tower_setup.variable_device):
      if initializer is None:
        # He initialization
        initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
      self.n_params += np.prod(shape)
      if type(initializer) == np.ndarray:
        W = tf.get_variable(name, dtype=tower_setup.dtype, initializer=initializer, trainable=trainable)
      else:
        W = tf.get_variable(name, shape, tower_setup.dtype, initializer, trainable=trainable)
      if l2 > 0.0:
        self.regularizers.append(l2 * tf.nn.l2_loss(W))
      if tower_setup.use_weight_summaries:
        summ = tf.summary.histogram(name, W)
        self.summaries.append(summ)
        self.add_scalar_summary(tf.reduce_max(tf.abs(W)), name + "/W_abs_max")
      return W

  def create_bias_variable(self, name, shape, tower_setup, trainable=True, initializer=None):
    with tf.device(tower_setup.variable_device):
      if initializer is None:
        initializer = tf.constant_initializer(0.0, dtype=tower_setup.dtype)
      self.n_params += np.prod(shape)
      b = tf.get_variable(name, shape, tower_setup.dtype, initializer, trainable=trainable)
      if tower_setup.use_weight_summaries:
        summ = tf.summary.histogram(name, b)
        self.summaries.append(summ)
        self.add_scalar_summary(tf.reduce_max(tf.abs(b)), name + "/b_abs_max")
      return b
