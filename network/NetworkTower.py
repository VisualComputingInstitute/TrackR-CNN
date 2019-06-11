import tensorflow as tf
import inspect

from core import Measures
from core.Util import import_submodules
from core.Log import log
from network.Layer import Layer


def all_subclasses(cls):
  return set(cls.__subclasses__()).union(
    [s for c in cls.__subclasses__() for s in all_subclasses(c)])


def get_layer_class(layer_class):
  if not hasattr(get_layer_class, "_imported"):
    get_layer_class._imported = True
    import_submodules("network")
  constructors = [l for l in all_subclasses(Layer) if l.__name__ == layer_class]
  assert len(constructors) == 1, constructors
  class_ = constructors[0]
  assert class_ is not None, ("Unknown layer class", layer_class)
  return class_


class TowerSetup:
  def __init__(self, gpu_idx, reuse_variables, dataset, variable_device, is_main_train_tower, is_training,
               freeze_batchnorm, network_name, use_weight_summaries=False):
    self.gpu_idx = gpu_idx
    self.reuse_variables = reuse_variables
    self.dataset = dataset
    self.variable_device = variable_device
    self.dtype = tf.float32
    self.use_weight_summaries = use_weight_summaries
    self.is_main_train_tower = is_main_train_tower
    self.is_training = is_training
    self.freeze_batchnorm = freeze_batchnorm
    self.network_name = network_name


class NetworkTower:
  def __init__(self, config, tower_setup, input_tensors_dict, dataset):
    network_def = config.dict("network")
    self.setup = tower_setup
    self.layers = {}
    self.summaries = []
    self.losses = []
    self.regularizers = []
    self.update_ops = []
    self.placeholders = []
    self.measures = {}
    self.extractions = {}

    if tower_setup.is_main_train_tower:
      print("inputs:", file=log.v4)
      for k, v in input_tensors_dict.items():
        print(k, v.get_shape().as_list(), file=log.v4)
    print("network:", file=log.v4)

    gpu_str = "/gpu:" + str(tower_setup.gpu_idx)
    tower_name = "tower_gpu_" + str(tower_setup.gpu_idx)
    with tf.variable_scope(tf.get_variable_scope(), reuse=tower_setup.reuse_variables), \
            tf.device(gpu_str), tf.name_scope(tower_name):
      for name, layer_def in network_def.items():
        layer = self._create_layer(name, layer_def, tower_setup, input_tensors_dict, dataset)
        self.layers[name] = layer
        if tower_setup.is_main_train_tower:
          print(name + ": ", file=log.v4, end="")
          for out in layer.outputs:
            print(out.get_shape().as_list(), file=log.v4, end="")
          print(", " + str(layer.n_params) + " params", file=log.v4, end="")
          print(file=log.v4)
    for layer in self.layers.values():
      self.summaries.extend(layer.summaries)
      self.losses.extend(layer.losses)
      self.regularizers.extend(layer.regularizers)
      self.update_ops.extend(layer.update_ops)
      self.placeholders.extend(layer.placeholders)
      for k, v in layer.measures.items():
        if k in self.measures:
          if k == Measures.LOSS:
            self.measures[Measures.LOSS] += v
          elif k == Measures.N_EXAMPLES:
            pass
          else:
            # most measures can only be used once in the network for now
            assert k not in self.measures
        else:
          self.measures[k] = v

      # if an extraction key is used more than once, pack it in a list
      for k, v in layer.extractions.items():
        if k in self.extractions:
          if not isinstance(self.extractions[k], list):
            self.extractions[k] = [self.extractions[k]]
          self.extractions[k].append(v)
        else:
          self.extractions[k] = v

    if len(self.losses) == 0:
      loss = tf.constant(0, dtype=tf.float32)
    else:
      loss = tf.add_n(self.losses)
    if len(self.regularizers) == 0:
      reg = tf.constant(0, dtype=tf.float32)
    else:
      reg = tf.add_n(self.regularizers)
    self.total_loss_with_regularizer = loss + reg
    self.n_params = sum([l.n_params for l in self.layers.values()], 0)

  def _create_layer(self, name, layer_def, tower_setup, input_tensors_dict, dataset):
    layer_class = layer_def["class"]
    class_ = get_layer_class(layer_class)
    spec = inspect.getargspec(class_.__init__)
    args = spec[0]

    # mess with the layer def
    layer_def = layer_def.copy()
    if "tower_setup" in args:
      layer_def["tower_setup"] = tower_setup
    if "dataset" in args:
      layer_def["dataset"] = dataset
    if "network_input_dict" in args:
      layer_def["network_input_dict"] = input_tensors_dict
    if "from" in layer_def:
      inputs = sum([self.layers[x].outputs for x in layer_def["from"]], [])
      del layer_def["from"]
    else:
      inputs = [input_tensors_dict["inputs"]]
    if "concat" in layer_def:
      concat = sum([self.layers[x].outputs for x in layer_def["concat"]], [])
      layer_def["concat"] = concat
    layer_def["inputs"] = inputs
    layer_def["name"] = name
    del layer_def["class"]

    # check if all args are specified
    defaults = spec[3]
    if defaults is None:
      defaults = []
    n_non_default_args = len(args) - len(defaults)
    non_default_args = args[1:n_non_default_args]  # without self
    for arg in non_default_args:
      assert arg in layer_def, (name, arg)
    layer = class_(**layer_def)
    return layer
