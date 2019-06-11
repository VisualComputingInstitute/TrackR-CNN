import tensorflow as tf
from tensorflow.python.client import timeline

from core.Log import log
from core.Util import average_gradients, clip_gradients
from core import Measures, Extractions
from core.Measures import accumulate_measures
from core.Extractions import accumulate_extractions


class Trainer:
  def __init__(self, config, train_network, test_network, global_step, session):
    self.opt_str = config.string("optimizer", "adam").lower()
    self.train_network = train_network
    self.test_network = test_network
    self.session = session
    self.global_step = global_step
    self.validation_step_number = 0
    self.gradient_clipping = config.float("gradient_clipping", -1.0)
    self.learning_rates = config.int_key_dict("learning_rates")
    self.learning_rate_keys_are_steps = config.bool("learning_rate_keys_are_steps", False)
    self.curr_learning_rate = self.learning_rates[1]
    self.lr_var = tf.placeholder(tf.float32, shape=[], name="learning_rate")
    self.loss_scale_var = tf.placeholder_with_default(1.0, shape=[], name="loss_scale")
    self.use_gradient_checkpointing = config.bool("use_gradient_checkpointing", False)
    if self.use_gradient_checkpointing:
      import memory_saving_gradients
      from tensorflow.python.ops import gradients
      #new_grad_fun = memory_saving_gradients.gradients_speed
      new_grad_fun = memory_saving_gradients.gradients_collection
      tf.__dict__["gradients"] = new_grad_fun
      gradients.__dict__["gradients"] = new_grad_fun
    self.opt, self.reset_opt_op = self.create_optimizer(config)
    self.collect_run_metadata = config.bool("collect_run_metadata", False)

    grad_norm = None
    if train_network is not None:
      self._step_op, grad_norm = self.create_step_op_and_grad_norm()
      self._update_ops = self.train_network.update_ops
    else:
      self._step_op = None
      self._update_ops = None
    # if step_num % summary_interval == 0, extract and write summaries; always write summaries if summary_interval==1
    self.summary_interval = config.int("summary_interval", 1)
    self.summary_writer, self.summary_op_train, self.summary_op_test = self.init_summaries(config, grad_norm)

  def create_optimizer(self, config):
    momentum = config.float("momentum", 0.9)
    if self.opt_str == "sgd_nesterov":
      return tf.train.MomentumOptimizer(self.lr_var, momentum, use_nesterov=True), None
    elif self.opt_str == "sgd_momentum":
      return tf.train.MomentumOptimizer(self.lr_var, momentum), None
    elif self.opt_str == "sgd":
      return tf.train.GradientDescentOptimizer(self.lr_var), None
    elif self.opt_str == "adam":
      opt = tf.train.AdamOptimizer(self.lr_var)
      all_vars = tf.global_variables()
      opt_vars = [v for v in all_vars if "Adam" in v.name]
      reset_opt_op = tf.variables_initializer(opt_vars, "reset_optimizer")
      return opt, reset_opt_op
    elif self.opt_str == "none":
      return None, None
    else:
      assert False, ("unknown optimizer", self.opt_str)

  def reset_optimizer(self):
    assert self.opt_str == "adam", "reset not implemented for other optimizers yet"
    assert self.reset_opt_op is not None
    self.session.run(self.reset_opt_op)

  def init_summaries(self, config, grad_norm=None):
    summdir = config.dir("summary_dir", "summaries")
    model = config.string("model")
    summdir += model + "/"
    tf.gfile.MakeDirs(summdir)
    summary_writer = None
    summary_op = None
    summary_op_test = None
    if config.bool("write_summaries", True):
      summary_writer = tf.summary.FileWriter(summdir, self.session.graph)
      if self.train_network is not None:
        train_summs = self.train_network.summaries
        if grad_norm is not None:
          grad_norm_summary = tf.summary.scalar("grad_norm", grad_norm)
          train_summs.append(grad_norm_summary)
        # better do not merge ALL summaries, since otherwise we get summaries from different networks
        # and might execute (parts of) the test network while training
        # self.summary_op = tf.merge_all_summaries()
        if len(train_summs) > 0:
          summary_op = tf.summary.merge(train_summs)
      if self.test_network is not None and len(self.test_network.summaries) > 0:
        summary_op_test = tf.summary.merge(self.test_network.summaries)
    return summary_writer, summary_op, summary_op_test

  def adjust_learning_rate(self, epoch, n_examples_processed_total, learning_rate=None):
    if learning_rate is None:
      if self.learning_rate_keys_are_steps:
        key = max([k for k in self.learning_rates.keys() if k <= n_examples_processed_total + 1])
      else:
        key = max([k for k in self.learning_rates.keys() if k <= epoch + 1])
      new_lr = self.learning_rates[key]
    else:
      new_lr = learning_rate
    if self.curr_learning_rate != new_lr:
      print("changing learning rate to", new_lr, file=log.v1)
      self.curr_learning_rate = new_lr

  def create_step_op_and_grad_norm(self):
    if self.opt is None:
      return tf.no_op("dummy_step_op"), None

    losses_with_regularizers = self.train_network.tower_total_losses_with_regularizers
    setups = self.train_network.tower_setups
    tower_grads = []
    for l, s in zip(losses_with_regularizers, setups):
      gpu_str = "/gpu:" + str(s.gpu_idx)
      with tf.device(gpu_str), tf.name_scope("tower_gpu_" + str(s.gpu_idx) + "_opt"):
        var_list = (
          tf.trainable_variables() +
          tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
        # TODO: check if gate_gradients=False is safe
        # TODO: check if colocate_gradients_with_ops is useful
        grads_raw = self.opt.compute_gradients(l, var_list=var_list, gate_gradients=False,
                                               colocate_gradients_with_ops=True)
        # filter out gradients w.r.t. disconnected variables
        grads_filtered = [g for g in grads_raw if g[0] is not None]
        tower_grads.append(grads_filtered)

    with tf.device(setups[0].variable_device):
      if len(tower_grads) == 1:
        grads = tower_grads[0]
      else:
        # average the gradients over the towers
        grads = average_gradients(tower_grads)

      # grad clipping
      if self.gradient_clipping != -1:
        grads, norm = clip_gradients(grads, self.gradient_clipping)
      else:
        norm = None

      if len(grads) == 0:
        return tf.no_op("dummy_step_op"), None

      step_op = self.opt.apply_gradients(grads, global_step=self.global_step)
    return step_op, norm

  def validation_step(self, epoch=None, n_examples_processed_total=None, feed_dict=None, extraction_keys=()):
    ops = {Measures.MEASURES: self.test_network.tower_measures}
    res = self._step(self.test_network, feed_dict, ops, self.summary_op_test, extraction_keys,
                     self.validation_step_number)
    self.validation_step_number += 1
    return res

  def train_step(self, epoch, n_examples_processed_total=None, feed_dict=None, loss_scale=1.0, learning_rate=None,
                 extraction_keys=()):
    self.adjust_learning_rate(epoch, n_examples_processed_total, learning_rate)

    if feed_dict is None:
      feed_dict = {}
    else:
      feed_dict = feed_dict.copy()
    feed_dict[self.lr_var] = self.curr_learning_rate
    feed_dict[self.loss_scale_var] = loss_scale

    ops = {"_update_ops": self._update_ops, "_step": self._step_op, "global_step": self.global_step,
           Measures.MEASURES: self.train_network.tower_measures}
    res = self._step(self.train_network, feed_dict, ops, self.summary_op_train, extraction_keys, step_number=None)
    return res

  def _step(self, network, feed_dict, ops, summary_op, extraction_keys, step_number):
    if feed_dict is None:
      feed_dict = {}

    if summary_op is not None:
      ops["summaries"] = summary_op
    if self.collect_run_metadata:
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
    else:
      run_options = None
      run_metadata = None

    if len(extraction_keys) > 0:
      ops[Extractions.EXTRACTIONS] = [{k: [v] for k, v in extractions.items() if k in extraction_keys}
                                      for extractions in network.tower_extractions]

    res = self.session.run(ops, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
    if "summaries" in res:
      summary_str = res["summaries"]
      del res["summaries"]
    else:
      summary_str = None
    if step_number is None:
      step_number = res["global_step"]

    if step_number % self.summary_interval == 0 and self.summary_writer is not None:
      if self.collect_run_metadata and step_number > 50:
        # this is experimental, TODO: make this cleaner
        # the 50 is to allow for some warmup
        self.summary_writer.add_run_metadata(run_metadata, tag="timing", global_step=step_number)
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open('timing.json', 'w') as f:
          f.write(chrome_trace)
      if summary_str is not None:
        self.summary_writer.add_summary(summary_str, global_step=step_number)
    res[Measures.MEASURES] = accumulate_measures({}, *res[Measures.MEASURES])
    if len(extraction_keys) > 0:
      res[Extractions.EXTRACTIONS] = accumulate_extractions({}, *res[Extractions.EXTRACTIONS])
    return res
