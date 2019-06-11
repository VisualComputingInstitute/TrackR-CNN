from abc import abstractmethod
import tensorflow as tf

from datasets.Dataset import AbstractDataset, DataKeys


class FeedDataset(AbstractDataset):
  def __init__(self, config, subset, data_keys_to_use, num_classes=2):
    super().__init__(config, subset, num_classes)
    self._data_keys_to_use = data_keys_to_use

    self._batch_size = -1
    if subset == "val":
      self._batch_size = config.int("batch_size_val", -1)
      if self._batch_size == -1:
        self._batch_size = config.int("batch_size_eval", -1)
    if self._batch_size == -1:
      self._batch_size = config.int("batch_size")

    self._placeholders = self._create_placeholders()

  def get_batch_size(self):
    return self._batch_size

  def _create_placeholders(self):
    dtypes_and_shapes = {
      DataKeys.IMAGES: (tf.float32, (None, None, 3)),
      DataKeys.SEGMENTATION_LABELS: (tf.uint8, (None, None, 1)),
      DataKeys.BBOX_GUIDANCE: (tf.uint8, (None, None, 1)),
      DataKeys.IMAGE_FILENAMES: (tf.string, ()),
      DataKeys.OBJ_TAGS: (tf.string, ()),
      DataKeys.LASER_GUIDANCE: (tf.float32, (None, None, 1)),
      DataKeys.BBOXES_y0x0y1x1: (tf.float32, (4,)),
      DataKeys.IDS: (tf.int32, ()),
      DataKeys.N_VALID_IDS: (tf.int32, (2,))
    }
    placeholders = {}
    for key in self._data_keys_to_use:
      dtype, shape = dtypes_and_shapes[key]
      key_placeholders = [tf.placeholder(dtype, shape, name=key + "_placeholder_{}".format(idx))
                          for idx in range(self._batch_size)]
      placeholders[key] = key_placeholders
    return placeholders

  def n_examples_per_epoch(self):
    raise NotImplementedError()

  def create_input_tensors_dict(self, batch_size):
    examples = [self._create_input_tensors_dict_single_example(idx_in_minibatch=idx) for idx in range(batch_size)]
    keys = examples[0].keys()
    data = {}
    for k in keys:
      if batch_size > 1:
        # raw images don't work with batch_size > 1 since they have different sizes
        if k == DataKeys.RAW_IMAGES:
          continue
        elif k == DataKeys.N_VALID_IDS:
          data[k] = examples[0][k]
        else:
          data[k] = tf.stack([example[k] for example in examples], axis=0)
      else:
        data[k] = tf.expand_dims(examples[0][k], axis=0)
    self.create_summaries(data)
    return data

  def _create_input_tensors_dict_single_example(self, idx_in_minibatch):
    raw_example = {}
    for data_key in self._data_keys_to_use:
      raw_example[data_key] = self._placeholders[data_key][idx_in_minibatch]
    if DataKeys.IMAGES in raw_example:
      raw_example[DataKeys.RAW_IMAGES] = raw_example[DataKeys.IMAGES]
    example = self.process_raw_example(raw_example)
    return example

  def get_placeholders(self, key):
    return self._placeholders[key]

  @abstractmethod
  def get_feed_dict_for_next_step(self):
    pass
