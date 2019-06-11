import tensorflow as tf

from network.Layer import Layer
from network.Util import prepare_input


class LinearCombination(Layer):
  output_layer = False

  def __init__(self, name, inputs, tower_setup, initial_weights, hack_gradient_magnitude=1.0):
    super().__init__()
    assert len(initial_weights) == len(inputs)
    with tf.variable_scope(name):
      initializer = tf.constant_initializer(initial_weights)
      weights = self.create_bias_variable("linear_combination_weights", len(inputs), tower_setup,
                                          initializer=initializer)
      if hack_gradient_magnitude > 1.0:
        # https://stackoverflow.com/a/43948872
        @tf.custom_gradient
        def amplify_gradient_layer(x):
          def grad(dy):
            return hack_gradient_magnitude * dy
          return tf.identity(x), grad
        weights = amplify_gradient_layer(weights)
      y = inputs[0] * weights[0]
      for n in range(1, len(inputs)):
        y += inputs[n] * weights[n]
      self.outputs.append(y)
      for n in range(len(inputs)):
        self.add_scalar_summary(weights[n], "linear_combination_weights_" + str(n))


class NNUpsizing(Layer):
  output_layer = False

  def __init__(self, name, inputs, tower_setup, factor=2):
    super().__init__()
    with tf.variable_scope(name):
      inp, dim = prepare_input(inputs)
      height = tf.shape(inp)[1]
      width = tf.shape(inp)[2]
      resized_input = tf.image.resize_images(inputs[0], (height*factor, width*factor),
                                             tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      self.outputs.append(resized_input)


class StackingLayer(Layer):
  output_layer = False

  def __init__(self, name, inputs, tower_setup):
    super().__init__()
    assert len(inputs) == 2
    self.outputs.append(tf.concat(inputs, axis=-1))
