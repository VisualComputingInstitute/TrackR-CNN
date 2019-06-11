from abc import ABC, abstractmethod
from random import shuffle

import tensorflow as tf

from core.Log import log
from datasets import DataKeys
from datasets.Augmentors import parse_augmentors
from datasets.Resize import resize, ResizeMode, jointly_resize
from datasets.util.BoundingBox import encode_bbox_as_mask, get_bbox_from_segmentation_mask
from datasets.util.Normalization import normalize, unnormalize


class AbstractDataset(ABC):
  def __init__(self, config, subset, num_classes):
    self.summaries = []
    self.config = config
    self.subset = subset
    self.n_classes = num_classes
    self.use_bbox_guidance = config.bool("use_bbox_guidance", False)
    self.use_mask_guidance = config.bool("use_mask_guidance", False)
    self.use_laser_guidance = config.bool("use_laser_guidance", False)
    self.use_clicks_guidance = config.bool("use_clicks_guidance", False)
    self.epoch_length_train = config.int("epoch_length_train", -1)
    self.epoch_length_val = config.int("epoch_length_val", -1)
    self.shuffle_buffer_size = config.int("shuffle_buffer_size", 5000)
    self.use_summaries = self.config.bool("use_summaries", False)
    self.normalize_imgs = self.config.bool("normalize_imgs", True)

  @abstractmethod
  def n_examples_per_epoch(self):
    if self.subset == "train" and self.epoch_length_train != -1:
      return self.epoch_length_train
    elif self.subset == "val" and self.epoch_length_val != -1:
      return self.epoch_length_val
    else:
      return None

  @abstractmethod
  def create_input_tensors_dict(self, batch_size):
    pass

  def num_classes(self):
    return self.n_classes

  def load_example(self, input_filenames):
    raw_example = self.load_raw_example(*input_filenames)
    processed = self.process_raw_example(raw_example)
    return processed

  def process_raw_example(self, example):
    example = self.postproc_example_initial(example)
    example = self.augment_example_before_resize(example)
    example = self.postproc_example_before_resize(example)
    example = self.resize_example(example)
    example = self.augment_example_after_resize(example)
    example = self.postproc_example_before_assembly(example)
    example = self.assemble_example(example)
    return example

  def load_raw_example(self, img_filename, label_filename=None, *args):
    img_tensors = self.load_image(img_filename)
    if not isinstance(img_tensors, dict):
      img_tensors = {DataKeys.IMAGES: img_tensors}
    label_tensors = self.load_annotation(img_tensors[DataKeys.IMAGES], img_filename, label_filename)
    if not isinstance(label_tensors, dict):
      label_tensors = {DataKeys.SEGMENTATION_LABELS: label_tensors}

    # merge the two dicts
    # the keys need to be disjoint!
    for k in img_tensors.keys():
      assert k not in label_tensors.keys()
    example = img_tensors
    example.update(label_tensors)
    return example

  def load_image(self, img_filename):
    img_data = tf.read_file(img_filename)
    img = tf.image.decode_image(img_data, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img.set_shape((None, None, 3))
    return img

  def load_annotation(self, img, img_filename, annotation_filename):
    ann_data = tf.read_file(annotation_filename)
    ann = tf.image.decode_image(ann_data, channels=1)
    ann.set_shape(img.get_shape().as_list()[:-1] + [1])
    ann = self.postproc_annotation(annotation_filename, ann)
    return ann

  def postproc_annotation(self, ann_filename, ann):
    return ann

  def resize_example(self, tensors):
    resize_mode_str = self.config.string("resize_mode_" + self.subset, "")
    if resize_mode_str == "":
      print("Using resize_mode_train for", self.subset, "since resize_mode_" + self.subset,
            "not specified in the config", file=log.v1)
      resize_mode_str = self.config.string("resize_mode_train")
    size = self.config.int_list("input_size_" + self.subset, [])
    if len(size) == 0:
      size = self.config.int_list("input_size_train", [])
    resize_mode = ResizeMode(resize_mode_str)
    tensors = resize(tensors, resize_mode, size)
    return tensors

  def jointly_resize_examples(self, tensors_batch):
    resize_mode_str = self.config.string("resize_mode_" + self.subset, "")
    if resize_mode_str == "":
      print("Using resize_mode_train for", self.subset, "since resize_mode_" + self.subset,
            "not specified in the config", file=log.v1)
      resize_mode_str = self.config.string("resize_mode_train")
    size = self.config.int_list("input_size_" + self.subset, [])
    if len(size) == 0:
      size = self.config.int_list("input_size_train", [])
    resize_mode = ResizeMode(resize_mode_str)
    tensors_batch = jointly_resize(tensors_batch, resize_mode, size)
    return tensors_batch

  def augment_example_before_resize(self, tensors):
    augmentors_str = self.config.string_list("augmentors_" + self.subset, [])
    augmentors = parse_augmentors(augmentors_str, self.config)
    for aug in augmentors:
      tensors = aug.apply_before_resize(tensors)
    return tensors

  def augment_example_after_resize(self, tensors):
    augmentors_str = self.config.string_list("augmentors_" + self.subset, [])
    augmentors = parse_augmentors(augmentors_str, self.config)
    for aug in augmentors:
      tensors = aug.apply_after_resize(tensors)
    return tensors

  def jointly_augment_examples_before_resize(self, tensors_batch):
    augmentors_str = self.config.string_list("augmentors_" + self.subset, [])
    augmentors = parse_augmentors(augmentors_str, self.config)
    for aug in augmentors:
      tensors_batch = aug.batch_apply_before_resize(tensors_batch)
    return tensors_batch

  def jointly_augment_examples_after_resize(self, tensors_batch):
    augmentors_str = self.config.string_list("augmentors_" + self.subset, [])
    augmentors = parse_augmentors(augmentors_str, self.config)
    for aug in augmentors:
      tensors_batch = aug.batch_apply_after_resize(tensors_batch)
    return tensors_batch

  def postproc_example_initial(self, tensors):
    if DataKeys.IMAGES in tensors and DataKeys.RAW_IMAGES not in tensors:
      tensors[DataKeys.RAW_IMAGES] = tensors[DataKeys.IMAGES]
    if DataKeys.IMAGES in tensors and DataKeys.RAW_IMAGE_SIZES not in tensors:
      tensors[DataKeys.RAW_IMAGE_SIZES] = tf.shape(tensors[DataKeys.IMAGES])[0:2]
    if DataKeys.SEGMENTATION_LABELS in tensors and DataKeys.BBOXES_y0x0y1x1 not in tensors:
      print("deriving bboxes from segmentation masks", file=log.v5)
      segmentation_labels = tensors[DataKeys.SEGMENTATION_LABELS]
      bbox = get_bbox_from_segmentation_mask(segmentation_labels)
      tensors[DataKeys.BBOXES_y0x0y1x1] = bbox
    return tensors

  def postproc_example_before_assembly(self, tensors):
    tensors_postproc = tensors.copy()
    if self.normalize_imgs:
      tensors_postproc[DataKeys.IMAGES] = normalize(tensors[DataKeys.IMAGES])
    return tensors_postproc

  def postproc_example_before_resize(self, tensors):
    tensors_postproc = tensors.copy()
    if (self.use_bbox_guidance) \
            and DataKeys.BBOXES_y0x0y1x1 in tensors and DataKeys.BBOX_GUIDANCE not in tensors:
      bbox = tensors[DataKeys.BBOXES_y0x0y1x1]
      img = tensors[DataKeys.IMAGES]
      bbox_guidance = encode_bbox_as_mask(bbox, tf.shape(img))
      tensors_postproc[DataKeys.BBOX_GUIDANCE] = bbox_guidance
    return tensors_postproc

  def assemble_example(self, tensors):
    tensors_assembled = tensors.copy()
    inputs_to_concat = [tensors[DataKeys.IMAGES]]

    if self.use_bbox_guidance and DataKeys.BBOX_GUIDANCE in tensors:
      print("using bbox guidance", file=log.v1)
      bbox_guidance = tf.cast(tensors[DataKeys.BBOX_GUIDANCE], tf.float32)
      inputs_to_concat.append(bbox_guidance)
    if self.use_laser_guidance and DataKeys.LASER_GUIDANCE in tensors:
      print("using laser guidance", file=log.v1)
      laser_guidance = tf.cast(tensors[DataKeys.LASER_GUIDANCE], tf.float32)
      inputs_to_concat.append(laser_guidance)
    if self.use_clicks_guidance:
      print("using guidance from clicks", file=log.v1)
      neg_dist_transform = tensors[DataKeys.NEG_CLICKS]
      pos_dist_transform = tensors[DataKeys.POS_CLICKS]
      inputs_to_concat.append(neg_dist_transform)
      inputs_to_concat.append(pos_dist_transform)
    if self.use_mask_guidance:
      print("using mask guidance", file=log.v1)
      mask = tf.cast(tensors[DataKeys.SEGMENTATION_LABELS], tf.float32)
      inputs_to_concat.append(mask)
    if len(inputs_to_concat) > 1:
      inputs = tf.concat(inputs_to_concat, axis=-1)
    else:
      inputs = inputs_to_concat[0]

    tensors_assembled[DataKeys.INPUTS] = inputs
    return tensors_assembled

  def create_summaries(self, data):
    if DataKeys.IMAGES in data:
      if self.normalize_imgs:
        unnormed = unnormalize(data[DataKeys.IMAGES])
      else:
        unnormed = data[DataKeys.IMAGES]
      # allow visualization if only 2 color channels are there
      if unnormed.shape[-1] == 2:
        pad = tf.zeros_like(unnormed[:, :, :, :1])
        unnormed = tf.concat([unnormed, pad], axis=3)
      self.summaries.append(tf.summary.image(self.subset + "data/images", unnormed))
    if DataKeys.SEGMENTATION_LABELS in data:
      self.summaries.append(tf.summary.image(self.subset + "data/ground truth segmentation labels",
                                             tf.cast(data[DataKeys.SEGMENTATION_LABELS], tf.float32)))
    if DataKeys.SEGMENTATION_INSTANCE_LABELS in data:
      self.summaries.append(tf.summary.image(self.subset + "data/ground truth segmentation instance labels",
                                             tf.cast(data[DataKeys.SEGMENTATION_INSTANCE_LABELS], tf.float32)))
    if DataKeys.BBOX_GUIDANCE in data:
      self.summaries.append(tf.summary.image(self.subset + "data/bbox guidance",
                                             tf.cast(data[DataKeys.BBOX_GUIDANCE], tf.float32)))
    if DataKeys.SIGNED_DISTANCE_TRANSFORM_GUIDANCE in data:
      self.summaries.append(tf.summary.image(self.subset + "data/signed_distance_transform_guidance",
                                             data[DataKeys.SIGNED_DISTANCE_TRANSFORM_GUIDANCE]))
    if DataKeys.UNSIGNED_DISTANCE_TRANSFORM_GUIDANCE in data:
      self.summaries.append(tf.summary.image(self.subset + "data/unsigned_distance_transform_guidance",
                                             data[DataKeys.UNSIGNED_DISTANCE_TRANSFORM_GUIDANCE]))
    if DataKeys.LASER_GUIDANCE in data:
      self.summaries.append(tf.summary.image(self.subset + "data/laser guidance",
                                             tf.cast(data[DataKeys.LASER_GUIDANCE], tf.float32)))


class FileListDataset(AbstractDataset):
  def __init__(self, config, dataset_name, subset, default_path, num_classes):
    super().__init__(config, subset, num_classes)
    self.inputfile_lists = None
    self.fraction = config.float("data_fraction", 1.0)
    self.data_dir = config.string(dataset_name + "_data_dir", default_path)
    self._num_parallel_calls = config.int("num_parallel_calls", 32)
    self._prefetch_buffer_size = config.int("prefetch_buffer_size", 20)

  def _load_inputfile_lists(self):
    if self.inputfile_lists is not None:
      return
    self.inputfile_lists = self.read_inputfile_lists()
    assert len(self.inputfile_lists) > 0
    for l in self.inputfile_lists:
      assert len(l) > 0
    # make sure all lists have the same length
    assert all([len(l) == len(self.inputfile_lists[0]) for l in self.inputfile_lists])
    if self.fraction < 1.0:
      n = int(self.fraction * len(self.inputfile_lists[0]))
      self.inputfile_lists = tuple([l[:n] for l in self.inputfile_lists])

  def n_examples_per_epoch(self):
    self._load_inputfile_lists()
    n_examples = super().n_examples_per_epoch()
    if n_examples is None:
      return len(self.inputfile_lists[0])
    else:
      return n_examples

  def create_input_tensors_dict(self, batch_size):
    self._load_inputfile_lists()
    if self.subset == "train":
      # shuffle lists together, for this zip, shuffle, and unzip
      zipped = list(zip(*self.inputfile_lists))
      shuffle(zipped)
      inputfile_lists_shuffled = tuple([x[idx] for x in zipped] for idx in range(len(self.inputfile_lists)))
    else:
      inputfile_lists_shuffled = self.inputfile_lists
    tfdata = tf.data.Dataset.from_tensor_slices(inputfile_lists_shuffled)
    if self.subset == "train":
      tfdata = tfdata.shuffle(buffer_size=self.shuffle_buffer_size)

    def _load_example(*input_filenames):
      example = self.load_example(input_filenames)
      # this has different sizes and therefore cannot be batched
      if batch_size > 1:
        if DataKeys.SEGMENTATION_LABELS_ORIGINAL_SIZE in example:
          del example[DataKeys.SEGMENTATION_LABELS_ORIGINAL_SIZE]
        if DataKeys.RAW_IMAGES in example:
          del example[DataKeys.RAW_IMAGES]
      return example

    def _filter_example(tensors):
      if DataKeys.SKIP_EXAMPLE in tensors:
        return tf.logical_not(tensors[DataKeys.SKIP_EXAMPLE])
      else:
        return tf.constant(True)

    tfdata = tfdata.map(_load_example, num_parallel_calls=self._num_parallel_calls)
    tfdata = tfdata.filter(_filter_example)
    tfdata = tfdata.repeat()
    tfdata = self._batch(tfdata, batch_size)
    tfdata = tfdata.prefetch(buffer_size=self._prefetch_buffer_size)
    # TODO: maybe we can improve the performance like this
    #tf.contrib.data.prefetch_to_device("/gpu:0", self._prefetch_buffer_size)
    res = tfdata.make_one_shot_iterator().get_next()

    if self.use_summaries:
      self.create_summaries(res)
    return res

  def _batch(self, tfdata, batch_size):
    if batch_size > 1:
      tfdata = tfdata.batch(batch_size, drop_remainder=True)
    elif batch_size == 1:
      # like this we are able to retain the batch size in the shape information
      tfdata = tfdata.map(lambda x: {k: tf.expand_dims(v, axis=0) for k, v in x.items()})
    else:
      assert False, ("invalid batch size", batch_size)
    return tfdata

  # Override to add extraction keys that will be used by trainer.
  def get_extraction_keys(self):
    return []

  @abstractmethod
  def read_inputfile_lists(self):
    raise NotImplementedError
