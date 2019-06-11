import glob
import numpy as np
import tensorflow as tf
import pickle
#from tensorflow.contrib.framework import list_variables, load_variable

from core.Log import log


class Saver:
  def __init__(self, config, session):
    self.session = session
    self.build_networks = config.bool("build_networks", True)
    max_saves_to_keep = config.int("max_saves_to_keep", 0)
    keep_checkpoint_every_n_hours = config.float("keep_checkpoint_every_n_hours", 10000.0)
    self.load_epoch_no = config.int("load_epoch_no", 0)
    self.tf_saver = tf.train.Saver(max_to_keep=max_saves_to_keep,
                                   keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours, pad_step_number=True)
    self.model = config.string("model")
    self.model_base_dir = config.dir("model_dir", "models")
    self.model_dir = self.model_base_dir + self.model + "/"
    self.load = config.string("load", "")
    self.load_init_savers = None
    try:
      self.load_init = config.string("load_init", "")
      if self.load_init == "":
        self.load_init = []
      else:
        self.load_init = [self.load_init]
    except TypeError:
      self.load_init = config.string_list("load_init", [])

  def save_model(self, epoch):
    tf.gfile.MakeDirs(self.model_dir)
    self.tf_saver.save(self.session, self.model_dir + self.model, epoch)

  def try_load_weights(self):
    start_epoch = 0
    fn = None
    if self.load != "":
      fn = self.load.replace(".index", "")
    else:
      files = sorted(glob.glob(self.model_dir + self.model + "-*.index"))
      if len(files) > 0:
        if self.load_epoch_no > 0:
          epoch_string = str(self.load_epoch_no).zfill(8)
          files = [f for f in files if epoch_string in f]
        if len(files) > 0:
          fn = files[-1].replace(".index", "")

    if fn is not None:
      if self.build_networks:
        print("loading model from", fn, file=log.v1)
        self.tf_saver.restore(self.session, fn)
      if self.model == fn.split("/")[-2]:
        start_epoch = int(fn.split("-")[-1])
        print("starting from epoch", start_epoch + 1, file=log.v1)
    elif self.build_networks:
      if self.load_init_savers is None:
        self.load_init_savers = [self._create_load_init_saver(x) for x in self.load_init]
      assert len(self.load_init) == len(self.load_init_savers)
      for fn, load_init_saver in zip(self.load_init, self.load_init_savers):
        if fn.endswith(".pickle"):
          print("trying to initialize model from wider-or-deeper mxnet model", fn, file=log.v1)
          load_wider_or_deeper_mxnet_model(fn, self.session)
        elif fn.startswith("DeepLabRGB:"):
          fn = fn.replace("DeepLabRGB:", "")
          print("initializing DeepLab RGB weights from", fn)
          self._initialize_deep_lab_rgb_weights(fn)
        else:
          print("initializing model from", fn, file=log.v1)
          assert load_init_saver is not None
          load_init_saver.restore(self.session, fn)
    elif not self.build_networks and self.load_epoch_no > 0:
      start_epoch = self.load_epoch_no
    return start_epoch

  def _create_load_init_saver(self, filename):
    if self.load != "":
      return None
    if len(glob.glob(self.model_dir + self.model + "-*.index")) > 0:
      return None
    if filename == "" or filename.endswith(".pickle") or filename.startswith("DeepLabRGB:"):
      return None
    from tensorflow.contrib.framework import list_variables
    vars_and_shapes_file = [x for x in list_variables(filename) if x[0] != "global_step"]
    vars_file = [x[0] for x in vars_and_shapes_file]
    vars_to_shapes_file = {x[0]: x[1] for x in vars_and_shapes_file}
    vars_model = tf.global_variables()
    assert all([x.name.endswith(":0") for x in vars_model])
    vars_intersection = [x for x in vars_model if x.name[:-2] in vars_file]
    vars_missing_in_graph = [x for x in vars_model if x.name[:-2] not in vars_file and "Adam" not in x.name and
                             "beta1_power" not in x.name and "beta2_power" not in x.name]
    if len(vars_missing_in_graph) > 0:
      print("the following variables will not be initialized since they are not present in the initialization model",
            [v.name for v in vars_missing_in_graph], file=log.v1)

    var_names_model = [x.name for x in vars_model]
    vars_missing_in_file = [x for x in vars_file if x + ":0" not in var_names_model
                            and "RMSProp" not in x and "Adam" not in x and "Momentum" not in x]
    if len(vars_missing_in_file) > 0:
      print("the following variables will not be loaded from the file since they are not present in the graph",
            vars_missing_in_file, file=log.v1)

    vars_shape_mismatch = [x for x in vars_intersection if x.shape.as_list() != vars_to_shapes_file[x.name[:-2]]]
    if len(vars_shape_mismatch) > 0:
      print("the following variables will not be loaded from the file since the shapes in the graph and in the file "
            "don't match:", [(x.name, x.shape) for x in vars_shape_mismatch if "Adam" not in x.name], file=log.v1)
      vars_intersection = [x for x in vars_intersection if x not in vars_shape_mismatch]
    return tf.train.Saver(var_list=vars_intersection)

  def _initialize_deep_lab_rgb_weights(self, fn):
    vars_ = tf.global_variables()
    var_w = [x for x in vars_ if x.name == "xception_65/entry_flow/conv1_1/weights:0"]
    assert len(var_w) == 1, len(var_w)
    var_w = var_w[0]
    from tensorflow.contrib.framework import load_variable
    w = load_variable(fn, "xception_65/entry_flow/conv1_1/weights")
    val_new_w = self.session.run(var_w)
    val_new_w[:, :, :3, :] = w
    placeholder_w = tf.placeholder(tf.float32)
    assign_op_w = tf.assign(var_w, placeholder_w)
    self.session.run(assign_op_w, feed_dict={placeholder_w: val_new_w})


def load_wider_or_deeper_mxnet_model(model_path, session):
  params = pickle.load(open(model_path, "rb"), encoding="bytes")
  vars_ = tf.global_variables()

  model_name = model_path.split("/")[-1]
  if model_name.startswith("ilsvrc"):
    layer_mapping = {"res0": "res2a", "res1": "res2b1", "res2": "res2b2", "res3": "res3a", "res4": "res3b1",
                     "res5": "res3b2", "res6": "res4a", "res7": "res4b1", "res8": "res4b2", "res9": "res4b3",
                     "res10": "res4b4", "res11": "res4b5", "res12": "res5a", "res13": "res5b1", "res14": "res5b2",
                     "res15": "res6a", "res16": "res7a", "output": "linear1000", "conv0": "conv1a",
                     "collapse": "bn7"}
  elif model_name.startswith("ade"):
    layer_mapping = {"res0": "res2a", "res1": "res2b1", "res2": "res2b2", "res3": "res3a", "res4": "res3b1",
                     "res5": "res3b2", "res6": "res4a", "res7": "res4b1", "res8": "res4b2", "res9": "res4b3",
                     "res10": "res4b4", "res11": "res4b5", "res12": "res5a", "res13": "res5b1", "res14": "res5b2",
                     "res15": "res6a", "res16": "res7a", "output": "linear150", "conv0": "conv1a",
                     "conv1": ["bn7", "conv6a"]}
  elif model_name.startswith("voc"):
    layer_mapping = {"res0": "res2a", "res1": "res2b1", "res2": "res2b2", "res3": "res3a", "res4": "res3b1",
                     "res5": "res3b2", "res6": "res4a", "res7": "res4b1", "res8": "res4b2", "res9": "res4b3",
                     "res10": "res4b4", "res11": "res4b5", "res12": "res5a", "res13": "res5b1", "res14": "res5b2",
                     "res15": "res6a", "res16": "res7a", "output": "linear21", "conv0": "conv1a",
                     "conv1": ["bn7", "conv6a"]}
  else:
    assert False, model_name

  # from str (without :0) to var
  var_dict = {v.name[:-2]: v for v in vars_ if "Adam" not in v.name and "_power" not in v.name
              and "global_step" not in v.name}

  # from our var name to mxnet var name
  mxnet_dict = create_mxnet_dict(layer_mapping, var_dict)

  for k, v in mxnet_dict.items():
    assert v in params, (k, v)

  # use a placeholders to avoid memory issues when putting the weights as constants in the graph
  feed_dict = {}
  ops = []
  for idx, (k, v) in enumerate(mxnet_dict.items()):
    # print(idx, "/", len(mxnet_dict), "loading", k, file=log.v5)
    val = params[v]
    if val.ndim == 1:
      pass
    elif val.ndim == 2:
      val = np.swapaxes(val, 0, 1)
    elif val.ndim == 4:
      val = np.moveaxis(val, [0, 1, 2, 3], [3, 2, 0, 1])
    else:
      assert False, val.ndim
    var = var_dict[k]
    if var.get_shape() == val.shape:
      placeholder = tf.placeholder(tf.float32)
      op = tf.assign(var, placeholder)
      feed_dict[placeholder] = val
      ops.append(op)
    elif k.startswith("conv0"):
      print("warning, sizes for", k, "do not match, initializing matching part assuming the first 3 dimensions are RGB",
            file=log.v1)
      val_new = session.run(var)
      val_new[..., :3, :] = val
      placeholder = tf.placeholder(tf.float32)
      op = tf.assign(var, placeholder)
      feed_dict[placeholder] = val_new
      ops.append(op)
    else:
      print("skipping", k, "since the shapes do not match:", var.get_shape(), "and", val.shape, file=log.v1)
  session.run(ops, feed_dict=feed_dict)
  print("done loading mxnet model", file=log.v1)


def create_mxnet_dict(layer_mapping, var_dict):
  mxnet_dict = {}
  for vn in var_dict:
    sp = vn.split("/")
    if sp[0] not in layer_mapping:
      print("warning,", vn, "not found in mxnet model", file=log.v1)
      continue
    layer = layer_mapping[sp[0]]
    if "bn" in sp[1]:
      if isinstance(layer, list):
        layer = layer[0]
      layer = layer.replace("res", "bn")
      if sp[2] == "beta":
        postfix = "_beta"
      elif sp[2] == "gamma":
        postfix = "_gamma"
      elif sp[2] == "mean_ema":
        postfix = "_moving_mean"
      elif sp[2] == "var_ema":
        postfix = "_moving_var"
      else:
        assert False, sp
    else:
      if isinstance(layer, list):
        layer = layer[1]
      postfix = "_weight"

    if "ema" in vn:
      layer = "aux:" + layer
    else:
      layer = "arg:" + layer

    if sp[1] == "W0":
      branch = "_branch1"
    elif sp[1] == "W1":
      branch = "_branch2a"
    elif sp[1] == "W2":
      branch = "_branch2b1"
    elif sp[1] == "W3":
      branch = "_branch2b2"
    elif sp[1] == "W":
      branch = ""
    elif sp[1] == "bn0":
      branch = "_branch2a"
    elif sp[1] == "bn2":
      branch = "_branch2b1"
    elif sp[1] == "bn3":
      branch = "_branch2b2"
    # for collapse
    elif sp[1] == "bn":
      branch = ""
    elif sp[1] == "b":
      branch = ""
      postfix = "_bias"
    else:
      assert False, sp

    mxnet_dict[vn] = (layer + branch + postfix).encode("utf-8")
  return mxnet_dict
