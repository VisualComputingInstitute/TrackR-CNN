import tensorflow as tf

from core.Log import log
from network.NetworkTower import NetworkTower, TowerSetup


class Network:
  def __init__(self, config, dataset, is_trainnet, freeze_batchnorm, name, reuse_variables=None):
    self.name = name
    self.batch_size = -1
    if not is_trainnet:
      self.batch_size = config.int("batch_size_eval", -1)
    if self.batch_size == -1:
      self.batch_size = config.int("batch_size")
    self.own_dataset_per_gpu = config.bool("own_dataset_per_gpu", False)
    self.input_tensors_dict = dataset.create_input_tensors_dict(self.batch_size)
    self._towers = self._build_towers(config, dataset, is_trainnet, freeze_batchnorm, reuse_variables)
    if is_trainnet:
      print("number of parameters:", "{:,}".format(self._towers[0].n_params), file=log.v1)
    self.tower_total_losses_with_regularizers = [t.total_loss_with_regularizer for t in self._towers]
    self.tower_setups = [t.setup for t in self._towers]
    self.tower_measures = [t.measures for t in self._towers]
    self.tower_extractions = [t.extractions for t in self._towers]

    # for now put the extractions from the dataset into the first tower.
    for k in self.input_tensors_dict:
      assert k not in self.tower_extractions[0]
    self.tower_extractions[0].update(self.input_tensors_dict)

    self.update_ops = sum([t.update_ops for t in self._towers], [])
    self.summaries = sum([t.summaries for t in self._towers], [])
    self.placeholders = sum([t.placeholders for t in self._towers], [])
    self.summaries.extend(dataset.summaries)

  def _build_towers(self, config, dataset, is_trainnet, freeze_batchnorm, reuse_variables):
    if is_trainnet or config.bool("multi_gpu_eval", False):
      n_gpus = config.int("gpus", 1)
    else:
      # only use 1 gpu for testing
      n_gpus = 1

    towers = []
    with tf.name_scope(self.name):
      for gpu_idx in range(n_gpus):
        if n_gpus == 1:
          input_tensors_dict_sliced = self.input_tensors_dict
          variable_device = "/gpu:0"
        else:
          if self.own_dataset_per_gpu:
            if gpu_idx == 0:
              input_tensors_dict_sliced = self.input_tensors_dict
            else:
              input_tensors_dict_sliced = dataset.create_input_tensors_dict(self.batch_size)
          else:
            assert self.batch_size % n_gpus == 0, "batch_size must be divisible by the number of gpus"
            slice_size = self.batch_size // n_gpus
            slice_start = slice_size * gpu_idx
            slice_end = slice_size * (gpu_idx + 1)
            input_tensors_dict_sliced = {k: v[slice_start:slice_end] for k, v in self.input_tensors_dict.items()}
          variable_device = "/cpu:0"
        reuse_variables_in_tower = reuse_variables
        if gpu_idx > 0 or reuse_variables_in_tower is None:
          reuse_variables_in_tower = not is_trainnet or gpu_idx > 0
        tower_setup = TowerSetup(gpu_idx=gpu_idx, reuse_variables=reuse_variables_in_tower, dataset=dataset,
                                 variable_device=variable_device, is_training=is_trainnet,
                                 is_main_train_tower=is_trainnet and gpu_idx == 0, freeze_batchnorm=freeze_batchnorm,
                                 network_name=self.name)
        tower = NetworkTower(config, tower_setup, input_tensors_dict_sliced, dataset)
        towers.append(tower)
    return towers
